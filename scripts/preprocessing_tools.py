import os
import subprocess
import geopandas
import xarray as xr
import numpy as np
import rioxarray
from concurrent.futures import ThreadPoolExecutor, as_completed
import ee
import datetime as dt
import dask
import dask.bag as db
import re
import rasterio
from rasterio.enums import Resampling
from typing import List, Tuple
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt
import h5py
from sklearn.model_selection import train_test_split
import pandas as pd

#################################################
#######       00 - ArcticDEM mosaic       #######
#################################################

def clip_reproject_arcticdem(region,region_id,demDir,projectDir):

    xmin, ymin, xmax, ymax = region.total_bounds
    
    outDir = f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/"

    os.makedirs(outDir, exist_ok=True)

    region_mosaic = f"{outDir}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif"

    cmdClip = f"gdal_translate -projwin {xmin} {ymax} {xmax} {ymin} {demDir} {region_mosaic}"

    return subprocess.call(cmdClip, shell=True) 

#################################################
######          01 - Download CS-2         ######
#################################################

# Define custom seasonal grouping (DJF, MAM, JJA, SON)
def assign_season_month(ts):
    month = ts.dt.month
    return xr.where(month.isin([12, 1, 2]), "winter",
           xr.where(month.isin([3, 4, 5]), "spring",
           xr.where(month.isin([6, 7, 8]), "summer", "fall")))

# For DJF: assign December to following year
def season_year(ts):
    year = ts.dt.year
    return xr.where(ts.dt.month == 12, year + 1, year)

def save_seasonal_cryosat(region_id,dataDir,CSfile,projectDir):
    
    outDir = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/"
    # Create output directory if it doesn"t exist
    os.makedirs(outDir, exist_ok=True)

    # Load dataset
    CSdata = xr.open_dataset(os.path.join(dataDir,CSfile), decode_coords="all")
    CSdata["elev"] = CSdata["elev_diff"] + CSdata["elev_diff_ref"]
    CSdata["elev"] = CSdata["elev"].T  # flip orientation if needed

    # Make sure CRS is attached
    CSdata["elev"] = CSdata["elev"]

    # Mask out cells where std > 25
    CSdata["elev"] = CSdata["elev"].where(CSdata["elev_diff_error"] <= 25)

    ### Add extra padding around the data to ensure no ice is right at the margin of the domain
    # Original coords
    x = CSdata.coords["x"].values
    y = CSdata.coords["y"].values

    # Infer spacing (assumes regular grid)
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    pad_width = 2

    # Extend coordinates
    new_x = np.concatenate([
        x[0] - dx * np.arange(pad_width, 0, -1),
        x,
        x[-1] + dx * np.arange(1, pad_width + 1)
    ])
    new_y = np.concatenate([
        y[0] - dy * np.arange(pad_width, 0, -1),
        y,
        y[-1] + dy * np.arange(1, pad_width + 1)
    ])

    # Reindex dataset (fills with NaN automatically)
    CSdata_padded = CSdata.reindex(x=new_x, y=new_y)


    ### Sort data based on season they belong to

    # Add season labels
    CSdata_padded = CSdata_padded.assign_coords(season=assign_season_month(CSdata_padded["time"]))


    CSdata_padded = CSdata_padded.assign_coords(season_year=season_year(CSdata_padded["time"]))

    # Group by both season_year and season at once
    seasonal = CSdata_padded["elev"].groupby(["season_year", "season"]).median("time")

    # Save seasonal means to GeoTIFFs
    for (year, season), da in seasonal.groupby(["season_year", "season"]):
        # Drop the extra "stacked" dimension so only y,x remain
        da = da.squeeze(drop=True)

        # Ensure dims are in correct order
        if "y" in da.dims and "x" in da.dims:
            da = da.transpose("y", "x")

        # Write CRS
        da = da.rio.write_crs("EPSG:3413")

        # Build filename
        outPath = os.path.join(outDir, f"CS_seasonal_{int(year)}_{str(season)}.tif")

        da.rio.to_raster(outPath)
    return print(f"Saved region {region_id}")


#################################################
######       02 - Download Sentinel 1      ######
#################################################

def mask_edges(image):
    edge = image.lt(-30.0)
    maskedImage = image.mask().And(edge.Not())
    return image.updateMask(maskedImage)

def season_ranges():
    ranges = []
    ranges.append({'season': 'fall', 'year': 2014, 'start': dt.datetime(2014, 9, 1), 'end': dt.datetime(2014, 12, 1)})
    for y in range(2015, 2025):
        ranges.append({'season': 'winter', 'year': y, 'start': dt.datetime(y-1, 12, 1), 'end': dt.datetime(y, 3, 1)})
        ranges.append({'season': 'spring', 'year': y, 'start': dt.datetime(y, 3, 1), 'end': dt.datetime(y, 6, 1)})
        ranges.append({'season': 'summer', 'year': y, 'start': dt.datetime(y, 6, 1), 'end': dt.datetime(y, 9, 1)})
        ranges.append({'season': 'fall', 'year': y, 'start': dt.datetime(y, 9, 1), 'end': dt.datetime(y, 12, 1)})
    ranges.append({'season': 'spring', 'year': 2025, 'start': dt.datetime(2025, 3, 1), 'end': dt.datetime(2025, 6, 1)})
    return ranges

def export_season_mean(season, year, start_dt, end_dt,
                       S1_collection, AOI, FOLDER, SCALE, CRS, MAX_PIXELS):
    start = ee.Date(start_dt.isoformat()[:19] + 'Z')
    end   = ee.Date(end_dt.isoformat()[:19] + 'Z')

    coll = S1_collection.filterDate(start, end)

    size = coll.size().getInfo()
    if size == 0:
        print(f"Skipping {season} {year}: collection empty")
        return

    img = coll.mean().clip(AOI)

    desc = f"S1_{season}_{year}"
    file_prefix = desc

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=FOLDER,
        fileNamePrefix=file_prefix,
        region=AOI,
        scale=SCALE,
        crs=CRS,
        maxPixels=MAX_PIXELS
    )
    task.start()
    print(f"Export started -> {desc}")

def download_sentinel1(region_id,projectDir):
    region = geopandas.read_file(f"{projectDir}/data/initial/regions/region-{region_id}.shp").to_crs('EPSG:4326')

    xmin, ymin, xmax, ymax = region.total_bounds

    buffer = 0.5

    AOI = ee.Geometry.Polygon(
            [[[xmin, ymin-buffer],
            [xmax, ymin-buffer],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin]]])
    
    COLLECTION = 'COPERNICUS/S1_GRD' # This is the name of the Sentinel-1 image collection
    FOLDER = f'Sentinel1_region-{region_id}'  # Will be created automatically by Drive if it doesn't exist
    SCALE = 40               # meters
    CRS = 'EPSG:3413'
    MAX_PIXELS = 1e13

    S1_collection = ee.ImageCollection(COLLECTION).filterBounds(AOI).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH')).filter(ee.Filter.eq('instrumentMode', 'EW')).select('HH')
        
    S1_collection = S1_collection.map(mask_edges)

    # ---- Run exports ----
    for entry in season_ranges():
        export_season_mean(
            entry['season'], entry['year'], entry['start'], entry['end'],
            S1_collection, AOI, FOLDER, SCALE, CRS, MAX_PIXELS
        )

    return print(f"Sentinel-1 downloads initialized for region {region_id}")

#################################################
######       04 - Seasonal ArcticDEM       ######
#################################################

def process_file(dem_file, bit_dir, out_dir):
    """Load DEM + bitmask, apply mask, and save result."""
    filename = os.path.basename(dem_file)
    bit_file = os.path.join(bit_dir, filename[:-12] + '_bitmask_50m.tif')

    if not os.path.exists(bit_file):
        print(f"⚠️ Skipping {filename}: bitmask not found.")
        return None

    # Use chunks so that operations are lazy/dask-backed
    dem = rioxarray.open_rasterio(dem_file, masked=True, chunks={'x': 2048, 'y': 2048})
    bit = rioxarray.open_rasterio(bit_file, masked=True, chunks={'x': 2048, 'y': 2048})

    # Apply mask lazily
    dem_masked = dem.where(bit == 0)

    # Write to raster
    out_file = os.path.join(out_dir, filename)
    dem_masked.rio.to_raster(out_file, tiled=True, windowed=True)

    return out_file

def bit_masking(region_id,projectDir,cores):

    dem_dir = f'{projectDir}/data/initial/ArcticDEM/region-{region_id}/ArcticDEM_50m_strips/'
    bit_dir = f'{projectDir}/data/initial/ArcticDEM/region-{region_id}/ArcticDEM_50m_strips_bitmask/'
    out_dir = f'{projectDir}/data/initial/ArcticDEM/region-{region_id}/ArcticDEM_50m_strips_masked/'
    os.makedirs(out_dir, exist_ok=True)

    dem_files = [os.path.join(dem_dir, _) for _ in os.listdir(dem_dir) if _.endswith('.tif')]

    bag = db.from_sequence(dem_files, npartitions=cores)
    results = bag.map(lambda f: process_file(f, bit_dir, out_dir)).compute()
    return print(f"Finished bit masking for region {region_id}")


### GET DATE ###
# This function extracts the date from the strip file name
def parse_date_from_name(path):
    m = re.search(r'_(\d{8})_', os.path.basename(path))
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d")

### ASSIGN SEASON AND YEAR ###
# This function assigns a season and year given a date
def month_to_season_and_label_year(dt):
    m, y = dt.month, dt.year
    if m in (12, 1, 2):
        season = "winter"
        label_year = y if m in (1, 2) else y + 1
    elif m in (3, 4, 5):
        season, label_year = "spring", y
    elif m in (6, 7, 8):
        season, label_year = "summer", y
    else:
        season, label_year = "fall", y
    return season, label_year

### PLANE-FITTING ###
# This function returns a, b and c for the linear plane fit function through a 2d residual
def fit_plane_numpy(residuals_2d, x_coords, y_coords, valid_mask):
    vm = valid_mask & np.isfinite(residuals_2d)
    n = int(vm.sum())
    X = np.column_stack([x_coords[vm].ravel(), y_coords[vm].ravel(), np.ones(n)])
    z = residuals_2d[vm].ravel()
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    a, b, c = coeffs
    return float(a), float(b), float(c)

### CO-REGISTRATION ###
def coreg_two_pass(strip_da, reference_da, arctic_da):
    # Re-project the refence DEM onto the strip and match the resolution
    ref_on_strip = reference_da.rio.reproject_match(strip_da)
    # Extract x- and y-coordinates and put into meshgrid
    y_coords, x_coords = np.meshgrid(strip_da.y.values, strip_da.x.values, indexing='ij')

    # --- PASS 1 ---
    # Calculate the residuals between the strip and reference DEM
    resid1 = (strip_da - ref_on_strip).values
    # Mask out pixels with nan values in either strip or reference DEM
    mask_overlap = np.isfinite(strip_da.values) & np.isfinite(ref_on_strip.values)
    # Get a, b, and c from plane-fit
    a1, b1, c1 = fit_plane_numpy(resid1, x_coords, y_coords, mask_overlap)
    # Calculate the plane 
    plane1 = a1 * x_coords + b1 * y_coords + c1
    # Apply the plane-coorection to the original strip
    strip_pass1 = strip_da - xr.DataArray(plane1, coords=strip_da.coords, dims=strip_da.dims)

    # --- PASS 2 ---
    # Reproject the ArcticDEM mosaic onto the strip
    arctic_on_strip = arctic_da.rio.reproject_match(strip_da)
    # Calculate the residuals between the corrected strip and the ArcticDEM values
    diff_vs_arctic = (strip_pass1 - arctic_on_strip).values
    # Create a mask which masks out pixels that differ more than 15 meters from ArcticDEM
    mask15 = np.isfinite(diff_vs_arctic) & (np.abs(diff_vs_arctic) <= 15.0)
    # Calculate the residuals between the corrected strip and the reference DEM
    resid2 = (strip_pass1 - ref_on_strip).values
    # Get a, b, and c for the plane that fits through the masked residuals
    a2, b2, c2 = fit_plane_numpy(resid2, x_coords, y_coords, mask_overlap & mask15)
    # Calculate the plane
    plane2 = a2 * x_coords + b2 * y_coords + c2
    # Subtract the plane from the corrected strip to get the extra corrected strip
    #strip_final = strip_da - xr.DataArray(plane2, coords=strip_da.coords, dims=strip_da.dims)
    strip_pass2 = strip_pass1 - xr.DataArray(plane2, coords=strip_da.coords, dims=strip_da.dims)

    # Filter out points that differ more than 25 m from the ArcticDEM mosaic
    diff_vs_arctic2 = (strip_pass2 - arctic_on_strip)
    strip_final = strip_pass2.where(diff_vs_arctic2 <= 25.0)

    return strip_final

### CREATE REFERENCE DEM ###
# This function open the CryoSat-2 DEM which corresponds to the given year and season
# It secondly fills missing values with the ArcticDEM mosaic
def open_reference_for_label(label_year, season, cryosat_root, arctic_dem_mosaic):
    if label_year < 2010:
        return None, None
    cryo_path = os.path.join(cryosat_root, f'CS_seasonal_{label_year}_{season}.tif')
    if not os.path.exists(cryo_path):
        return None, None
    cryo = rioxarray.open_rasterio(cryo_path).squeeze()
    adem_on_cryo = arctic_dem_mosaic.rio.reproject_match(cryo)
    ref = cryo.fillna(adem_on_cryo).squeeze()
    return ref, arctic_dem_mosaic


def process_strip(p, cryosat_root, arctic_dem_mosaic, project_crs, out_dir):
    dt = parse_date_from_name(p)
    if dt is None:
        return None
    season, label_year = month_to_season_and_label_year(dt)
    if label_year < 2010:
        return None
    reference, arctic = open_reference_for_label(label_year, season, cryosat_root, arctic_dem_mosaic)
    if reference is None:
        print(f"Skipping {p}: no reference for {season} {label_year}")
        return None
    try:
        strip = rioxarray.open_rasterio(p, masked=True).rio.reproject(project_crs).squeeze()
    except Exception as e:
        print(f"Failed to open/reproject strip {p}: {e}")
        return None
    try:
        strip_corr = coreg_two_pass(strip, reference, arctic)
    except Exception as e:
        print(f"Coreg failed for {p}: {e}")
        return None
    out_name = os.path.splitext(os.path.basename(p))[0] + "_coreg2.tif"
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        strip_corr.rio.write_crs(project_crs, inplace=True)
        strip_corr.rio.to_raster(out_path, compress='lzw',tiled=True)
        print(f"Wrote corrected strip: {out_path}")
    except Exception as e:
        print(f"Failed to write corrected strip {p}: {e}")
    return out_path

def co_registration(region_id,project_crs,projectDir,cores):
    strip_dir = f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/ArcticDEM_50m_strips_masked/"
    out_dir = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/co-registered-strips/"
    os.makedirs(out_dir, exist_ok=True)

    arcticdem_path = f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif"
    cryosat_root = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal"

    arctic_dem_mosaic = rioxarray.open_rasterio(arcticdem_path, masked=True).squeeze()

    # The criteria which a pixel has to fulfill
    # diff_criteria = 25

    strip_files = [os.path.join(strip_dir, f) for f in os.listdir(strip_dir) if f.endswith(".tif")]
    print(f"Found {len(strip_files)} strips")

    bag = db.from_sequence(strip_files, npartitions=cores)
    bag = bag.map(lambda p: process_strip(p,cryosat_root,arctic_dem_mosaic,project_crs,out_dir))
    results = bag.compute(scheduler="threads", num_workers=cores)

    results = [r for r in results if r is not None]
    return print(f"Total corrected strips written: {len(results)}")

def is_in_year_season(path, target_year, target_season):
    """Check if strip belongs to a given year/season (Dec counts toward next winter)."""
    date_str = os.path.basename(path).split('_')[3]
    year = int(date_str[:4])
    month = int(date_str[4:6])

    season_months = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall":   [9, 10, 11]
    }

    # December belongs to next year's winter
    if month == 12:
        year += 1
        month = 1  # treat December as January for season grouping

    return (year == target_year) and (month in season_months[target_season])

def seasonal_arcticdem(region_id,projectDir):

    dask.config.set({
        "temporary_directory": "/tmp/dask",
        "array.slicing.split_large_chunks": True,
    })

    strip_dir = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/co-registered-strips/"
    out_dir = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/seasonal-strips/"
    os.makedirs(out_dir, exist_ok=True)

    strip_files = [os.path.join(strip_dir, f) for f in os.listdir(strip_dir) if f.endswith(".tif")]

    season_months = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall":   [9, 10, 11]
    }

    years = sorted({
        int(os.path.basename(f).split('_')[3][:4]) +
        (1 if int(os.path.basename(f).split('_')[3][4:6]) == 12 else 0)
        for f in strip_files
    })
    years = [y for y in years if y >= 2014]

    file = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/CS_seasonal_2018_summer.tif"
    cs = rioxarray.open_rasterio(file)

    minx, miny, maxx, maxy = cs.rio.bounds()
    target_res = 100  # in meters

    # Compute new width and height for strips
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # --- Process each year/season ---
    for year in years:
        for season in season_months:
            files_in_group = [f for f in strip_files if is_in_year_season(f, year, season)]
            if not files_in_group:
                continue

            print(f"Processing {year} {season}: {len(files_in_group)} files")

            sum_da = None
            count_da = None

            for f in files_in_group:
                da = rioxarray.open_rasterio(
                    f, masked=True, chunks={"x": 2048, "y": 2048}
                )

                # Downsample directly to 100 m grid (saves memory)
                da = da.rio.reproject(
                    da.rio.crs,
                    shape=(height, width),
                    transform=transform,
                    resampling=Resampling.average,
                )

                # Replace NaNs with 0 for summing
                valid = (~da.isnull()).astype("float32")
                da = da.fillna(0)

                if sum_da is None:
                    sum_da = da
                    count_da = valid
                else:
                    sum_da = sum_da + da
                    count_da = count_da + valid


            # Compute mean lazily, then write to disk
            mean_da = (sum_da / count_da).where(count_da > 0)

            out_file = os.path.join(out_dir, f"arcticdem_seasonal_{year}_{season}.tif")
            print(f"Saving {out_file} ...")
            mean_da.rio.to_raster(out_file, compress="LZW")
            print(f"Saved {out_file}\n")
    return print(f"Finished seasonal means of ArcticDEM for region {region_id}")


#################################################
######        05 - Regrid Sentinel-1       ######
#################################################

def regrid_sentinel1(region_id,projectDir):

    print(f"Starting on region {region_id}")
    s1_dir = f"{projectDir}/data/initial/sentinel-1/region-{region_id}/"
    out_dir = f"{projectDir}/data/interim/region-{region_id}/sentinel-1/"
    os.makedirs(out_dir, exist_ok=True)

    s1_files = [os.path.join(s1_dir, f) for f in os.listdir(s1_dir) if f.endswith(".tif")]

    csfile = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/CS_seasonal_2018_summer.tif"
    cs = rioxarray.open_rasterio(csfile)

    minx, miny, maxx, maxy = cs.rio.bounds()
    target_res = 100  # in meters

    # Compute new width and height for strips
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    for file in s1_files:
        da = rioxarray.open_rasterio(file, masked=True)  # masked=True handles nodata
        # Reproject/resample to target resolution and bounds
        da = da.rio.reproject(da.rio.crs, shape=(height, width),transform=transform,resampling=rasterio.enums.Resampling.average)

        out_file = os.path.join(out_dir, os.path.basename(file))
        da.rio.to_raster(out_file, compress="LZW")
        print(f"Saved {out_file}")
    return print(f"Finished region {region_id}")


#################################################
######         06 - Create icemask         ######
#################################################

def rasterize(gdf,raster):
    # create shapes (geometry, value) for rasterization
    shapes = ((geom, 1) for geom in gdf.geometry)

    # rasterize to match reference raster’s shape, transform, CRS
    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=raster.squeeze().shape,   # (height, width)
        transform=raster.rio.transform(),
        fill=0,
        dtype="uint8"
    )

    # wrap result as xarray DataArray with same coords/projection
    mask_da = raster.squeeze().copy(data=mask)
    mask_da = mask_da.expand_dims(dim="band")  # put band dimension back
    return mask_da

def get_icemask(region_id, projectDir, project_crs):
    rgi = geopandas.read_file(f"{projectDir}/data/initial/RGI/RGI2000-v7.0-C-03_arctic_canada_north/RGI2000-v7.0-C-03_arctic_canada_north.shp").to_crs(project_crs)

    region = geopandas.read_file(f"{projectDir}/data/initial/regions/region-{region_id}.shp")

    rgi_region = geopandas.clip(rgi, region)

    csfile = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/CS_seasonal_2018_summer.tif"
    cs = rioxarray.open_rasterio(csfile)

    ademfile = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/seasonal-strips/arcticdem_seasonal_2022_spring.tif"
    adem = rioxarray.open_rasterio(ademfile)

    out_dir = f"{projectDir}/data/interim/region-{region_id}/masks/"
    os.makedirs(out_dir, exist_ok = True)

    mask_500m = rasterize(rgi_region,cs)
    mask_100m = rasterize(rgi_region,adem)

    mask_500m.rio.to_raster(os.path.join(out_dir,'mask_500m.tif'))
    mask_100m.rio.to_raster(os.path.join(out_dir,'mask_100m.tif'))
    return print(f"Masks saved for region {region_id}")

#################################################
######          07 - Create tiles          ######
#################################################

TILE_SIZE = 64
OVERLAP = 8
STRIDE = TILE_SIZE - OVERLAP

def read_mask(mask_path: str):
    with rasterio.open(mask_path) as src:
        arr = src.read(1).astype(np.uint8)
        profile = src.profile
    mask = (arr != 0).astype(np.uint8)
    return mask, profile

def tile_touches_border(top: int, left: int, nrows: int, ncols: int) -> bool:
    return (top == 0) or (left == 0) or (top + TILE_SIZE == nrows) or (left + TILE_SIZE == ncols)

def tile_has_zero_border(tile: np.ndarray) -> bool:
    top_zero = tile[0, :].sum() == 0
    bottom_zero = tile[-1, :].sum() == 0
    left_zero = tile[:, 0].sum() == 0
    right_zero = tile[:, -1].sum() == 0
    return top_zero or bottom_zero or left_zero or right_zero

def generate_candidate_tiles(mask: np.ndarray) -> List[Tuple[int, int, np.ndarray, int, bool]]:
    """
    Scan the whole mask with stride STRIDE and return candidate tiles.
    Each candidate is a tuple: (top, left, flat_indices_of_ones, ones_count, touches_border_flag)
    Only tiles with at least one '1' are returned (all-zero tiles excluded).
    For tiles that touch the image border, we enforce tile_has_zero_border(tile) else exclude.

    Note: flat indices are indices into mask_flat = mask.ravel(order='C').
    """
    nrows, ncols = mask.shape
    mask_flat = mask.ravel()
    candidates = []

    for top in range(0, nrows, STRIDE):
        for left in range(0, ncols, STRIDE):
            # ensure tile fits entirely within image
            top_clamped = max(0, min(top, nrows - TILE_SIZE))
            left_clamped = max(0, min(left, ncols - TILE_SIZE))
            tile = mask[top_clamped:top_clamped + TILE_SIZE, left_clamped:left_clamped + TILE_SIZE]
            if tile.shape != (TILE_SIZE, TILE_SIZE):
                continue
            touches = tile_touches_border(top_clamped, left_clamped, nrows, ncols)
            if touches and (not tile_has_zero_border(tile)):
                # border tile must have a zero border row/col
                continue
            ones_positions = np.flatnonzero(tile.ravel())
            if ones_positions.size == 0:
                # exclude all-zero tiles
                continue
            # Convert tile-local flat indices to global flat indices in the whole image
            # Compute global start index
            global_row = top_clamped
            global_col = left_clamped
            # For each local flat index p, row = p // TILE_SIZE, col = p % TILE_SIZE
            rows = ones_positions // TILE_SIZE + global_row
            cols = ones_positions % TILE_SIZE + global_col
            global_inds = (rows * ncols + cols)
            candidates.append((top_clamped, left_clamped, global_inds, int(ones_positions.size), bool(touches)))

    return candidates

def minimal_tiles_greedy(mask_path: str) -> geopandas.GeoDataFrame:
    """
    Produce a minimal set of tiles covering all ones using a greedy algorithm.

    Parameters
    ----------
    mask_path : path to binary mask GeoTIFF (0/1)

    Returns
    -------
    GeoDataFrame of selected tiles (tile_id, top, left, ones_count, all_ones, geometry)
    """
    mask, profile = read_mask(mask_path)
    nrows, ncols = mask.shape
    transform = profile["transform"]
    crs = profile.get("crs")

    total_ones = int(mask.sum())
    if total_ones == 0:
        # Empty mask: return empty GeoDataFrame
        gdf_empty = geopandas.GeoDataFrame(columns=["tile_id", "top", "left", "ones_count", "all_ones", "geometry"], geometry="geometry", crs=crs)
        print("No ones in mask — no tiles created.")
        return gdf_empty

    candidates = generate_candidate_tiles(mask)
    if len(candidates) == 0:
        # fallback: there are ones but no valid candidates (possible if ones only on border and border rule filtered them)
        # In this rare case, relax border rule for tiles containing ones and touching border.
        nrows, ncols = mask.shape
        candidates = []
        for top in range(0, nrows, STRIDE):
            for left in range(0, ncols, STRIDE):
                top_clamped = max(0, min(top, nrows - TILE_SIZE))
                left_clamped = max(0, min(left, ncols - TILE_SIZE))
                tile = mask[top_clamped:top_clamped + TILE_SIZE, left_clamped:left_clamped + TILE_SIZE]
                if tile.shape != (TILE_SIZE, TILE_SIZE):
                    continue
                ones_positions = np.flatnonzero(tile.ravel())
                if ones_positions.size == 0:
                    continue
                rows = ones_positions // TILE_SIZE + top_clamped
                cols = ones_positions % TILE_SIZE + left_clamped
                global_inds = (rows * ncols + cols)
                candidates.append((top_clamped, left_clamped, global_inds, int(ones_positions.size), True))

    # Prepare uncovered mask as boolean flat array
    mask_flat = mask.ravel()
    uncovered = mask_flat.copy().astype(bool)

    # Convert candidates to arrays for quick iteration
    cand_tops = [c[0] for c in candidates]
    cand_lefts = [c[1] for c in candidates]
    cand_inds = [c[2] for c in candidates]  # arrays of global indices
    cand_ones_counts = np.array([c[3] for c in candidates], dtype=int)

    selected_indices = []  # indices into candidates
    covered_count = 0

    # Greedy loop
    while True:
        # compute gains for each remaining candidate
        best_gain = 0
        best_idx = -1
        for i, inds in enumerate(cand_inds):
            if i in selected_indices:
                continue
            # number of uncovered ones this candidate would cover
            gain = int(uncovered[inds].sum())
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        # select best_idx
        selected_indices.append(best_idx)
        inds = cand_inds[best_idx]
        # mark covered
        newly_covered = uncovered[inds]
        covered_count += int(newly_covered.sum())
        uncovered[inds] = False
        # stop when all covered
        if covered_count >= total_ones:
            break

    # Build GeoDataFrame from selected candidates
    geoms = []
    tops = []
    lefts = []
    ones_counts = []
    all_ones_flags = []

    for sel_id, ci in enumerate(selected_indices):
        top = cand_tops[ci]
        left = cand_lefts[ci]
        # compute ones_count inside tile (from original mask)
        tile = mask[top:top + TILE_SIZE, left:left + TILE_SIZE]
        ones_count = int(tile.sum())
        all_ones = (ones_count == TILE_SIZE * TILE_SIZE)
        x_min, y_max = transform * (left, top)
        x_max, y_min = transform * (left + TILE_SIZE, top + TILE_SIZE)
        geom = box(x_min, y_min, x_max, y_max)

        geoms.append(geom)
        tops.append(int(top))
        lefts.append(int(left))
        ones_counts.append(ones_count)
        all_ones_flags.append(bool(all_ones))

    gdf = geopandas.GeoDataFrame({
        "tile_id": list(range(len(geoms))),
        "top": tops,
        "left": lefts,
        "ones_count": ones_counts,
        "all_ones": all_ones_flags
    }, geometry=geoms, crs=crs)

    print(f"Selected {len(gdf)} tiles covering {covered_count}/{total_ones} ones.")
    return gdf

def create_tiles(region_id, projectDir):

    mask_path = f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif"

    gdf = minimal_tiles_greedy(mask_path)

    outdir = f"{projectDir}/data/interim/region-{region_id}/masks/"
    gdf.to_file(outdir+"tiles.gpkg", driver="GPKG")

    return print(f"Tiles saved for region {region_id}")

#################################################
######         08 - Create netcdfs         ######
#################################################

def fill_na_nc(da,max_dist=5):
    arr = da.values
    mask = np.isnan(arr)

    # Distance from each NaN to nearest valid pixel
    dist, inds = distance_transform_edt(mask, return_indices=True)

    # Fill only where distance <= max_dist
    filled = arr.copy()
    close_enough = dist <= max_dist
    filled[mask & close_enough] = arr[tuple(inds[:, mask & close_enough])]

    # Put back into DataArray
    da_filled = da.copy(data=filled)
    return da_filled

def mask_da_nc(da,mask):
    arr = da.values
    mask_arr = mask.values
    arr[mask_arr == 0] = 0
    masked_da = da.copy(data=arr)
    return masked_da

def season_to_date_nc(season, year):
    return {
        "winter": f"{year}-01-15",
        "spring": f"{year}-04-15",
        "summer": f"{year}-07-15",
        "fall":   f"{year}-10-15"
    }[season]

def create_netcdfs(region_id,projectDir):
    mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)
    years = list(range(2014, 2025))
    seasons = ['winter', 'spring', 'summer', 'fall']

    rasters = []
    times   = []

    for year in years:
        for season in seasons:

            s1file = f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_{season}_{year}.tif"

            if not os.path.exists(s1file):
                continue
            s1 = rioxarray.open_rasterio(s1file,masked=True)


            s1 = fill_na_nc(s1)
            s1 = mask_da_nc(s1,mask100)
            if "band" in s1.dims:
                s1 = s1.squeeze("band")

            timestamp = season_to_date_nc(season, year)

            rasters.append(s1)
            times.append(np.datetime64(timestamp))


    stacked = xr.concat(rasters, dim="time")
    stacked = stacked.assign_coords(time=("time", times))
    stacked.name = "s1-backscatter"

    # --- Save ---
    outfile = f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered.nc"
    stacked.to_netcdf(outfile)

    print("Saved:", outfile)

    rasters = []
    times   = []


    for year in years:
        for season in seasons:


            cs = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/CS_seasonal_{year}_{season}.tif",masked=True)
            cs= fill_na_nc(cs).rio.reproject_match(s1)
            cs = mask_da_nc(cs,mask100)


            if "band" in cs.dims:
                cs = cs.squeeze("band")

            timestamp = season_to_date_nc(season, year)

            rasters.append(cs)
            times.append(np.datetime64(timestamp))


    stacked = xr.concat(rasters, dim="time")
    stacked = stacked.assign_coords(time=("time", times))
    stacked.name = "cs-elevation"

    # --- Save ---
    outfile = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered.nc"
    stacked.to_netcdf(outfile)

    print("Saved:", outfile)

    rasters = []
    times   = []

    for year in years:
        for season in seasons:

            ademfile = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/seasonal-strips/arcticdem_seasonal_{year}_{season}.tif"

            if not os.path.exists(ademfile):
                continue
            adem = rioxarray.open_rasterio(ademfile,masked=True)

            adem = fill_na_nc(adem)
            adem = mask_da_nc(adem,mask100)

            if "band" in adem.dims:
                adem = adem.squeeze("band")

            timestamp = season_to_date_nc(season, year)

            rasters.append(adem)
            times.append(np.datetime64(timestamp))


    stacked = xr.concat(rasters, dim="time")
    stacked = stacked.assign_coords(time=("time", times))
    stacked.name = "adem-elevation"

    # --- Save ---
    outfile = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/ArcticDEM_gathered.nc"
    stacked.to_netcdf(outfile)

    print("Saved:", outfile)

    return print(f"All netcdfs saved for region {region_id}")


#################################################
###### 09 - Normalize, tile and split data ######
#################################################

def mask_da(da,mask):
    arr = da.values
    mask_arr = mask.values
    arr[mask_arr == 0] = 0
    masked_da = da.copy(data=arr)
    return masked_da

def flag_tile(raster):
    if np.isnan(np.min(raster)):
        return 'incomplete'
    else:
        return 'complete'
    
def zscore(df):
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    z = (df - mean) / std
    return z, std, mean

def normalize_tiling(region_id,projectDir,project_crs,normalization="raw"):

    print(f"Running region {region_id}")

    tiles = geopandas.read_file(f"{projectDir}/data/interim/region-{region_id}/masks/tiles.gpkg")

    mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)

    s1 = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered.nc")['s1-backscatter'].rio.write_crs(project_crs)

    cs = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered.nc")['cs-elevation'].rio.write_crs(project_crs)

    adem = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/ArcticDEM_gathered.nc")['adem-elevation'].rio.write_crs(project_crs)


    if normalization == 'zscore':
        s1_n, s1_std, s1_mean = zscore(s1)
        adem_n, adem_std, adem_mean = zscore(adem)
        cs_n, cs_std, cs_mean = zscore(cs)
    elif normalization == 'raw':
        s1_n = s1.copy()
        adem_n = adem.copy()
        cs_n = cs.copy()
    else:
        print('Normalization not valid')

    h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_normalization_{normalization}.h5"


    with h5py.File(h5_path, "w") as f:
        for time in s1.time.values:
            s1_t = s1_n.sel(time=time)
            cs_t = cs_n.sel(time=time)
            if time in adem.time.values:
                adem_t = adem_n.sel(time=time)
            else:
                nans = np.zeros_like(s1_t.values)*np.nan
                adem_t = s1_t.copy(data=nans)


            year = time.astype(str)[:4]
            month = time.astype(str)[5:7]

            timegroup = f"{year}_{month}"
            grp_time = f.create_group(timegroup)

            # Make sure that there are 0's outside the mask, and only nan values of missing ice pixels
            s1_t = mask_da(s1_t,mask100.squeeze())
            cs_t = mask_da(cs_t,mask100.squeeze())
            adem_t = mask_da(adem_t,mask100.squeeze())


            for t in range(len(tiles)):

                bounds = tiles.iloc[t].geometry.bounds
                cs_tile = cs_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
                s1_tile = s1_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
                adem_tile = adem_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
                mask_tile = mask100.squeeze().rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values

                cs_tile = cs_tile.astype(np.float32)
                s1_tile = s1_tile.astype(np.float32)
                adem_tile = adem_tile.astype(np.float32)
                mask_tile = mask_tile.astype(np.float32)


                #adem_flag = flag_tile(adem_tile)
                #s1_flag = flag_tile(s1_tile)
                #cs_flag = flag_tile(cs_tile)

                tile_id = f"tile_{tiles.iloc[t].tile_id}"

                grp_tile = grp_time.create_group(tile_id)

                # Create datasets within tile subgroup + flip CS data
                grp_tile.create_dataset("cs_tile", data=cs_tile, compression="gzip", compression_opts=4)
                grp_tile.create_dataset("s1_tile", data=s1_tile, compression="gzip", compression_opts=4)
                grp_tile.create_dataset("adem_tile", data=adem_tile, compression="gzip", compression_opts=4)
                grp_tile.create_dataset("mask_tile", data=mask_tile, compression="gzip", compression_opts=4)

                # Add flags as attributes
                grp_tile.attrs["cs_flag"] = flag_tile(cs_tile)
                grp_tile.attrs["s1_flag"] = flag_tile(s1_tile)
                grp_tile.attrs["adem_flag"] = flag_tile(adem_tile)

            print(f"✅ Added data for {timegroup}")
    return print(f"HDF5 file saved for region {region_id}")


def load_complete_tiles(region_ids,normalization,projectDir):
    """
    Load cs_tile, s1_tile, and adem_tile arrays from an HDF5 file,
    but only for tiles where all flags == 'complete'.
    Returns three numpy arrays: cs_all, s1_all, adem_all.
    """

    cs_list, s1_list, adem_list, s1_full, cs_full, mask_list, mask_full = [], [], [], [], [], [], []
    time, tile, region = [], [], []
    timef, tilef, regionf = [], [], []

    for region_id in region_ids:

        h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_normalization_{normalization}.h5"


        with h5py.File(h5_path, "r") as f:
            for timegroup in f.keys():
                grp_time = f[timegroup]

                for tile_id in grp_time.keys():
                    grp_tile = grp_time[tile_id]

                    # Check all three flags
                    cs_flag = grp_tile.attrs.get("cs_flag", "incomplete")
                    s1_flag = grp_tile.attrs.get("s1_flag", "incomplete")
                    adem_flag = grp_tile.attrs.get("adem_flag", "incomplete")

                    if (cs_flag == "complete") and (s1_flag == "complete"):
                        s1_full.append(grp_tile["s1_tile"][:])
                        cs_full.append(grp_tile["cs_tile"][:])
                        mask_full.append(grp_tile["mask_tile"][:])
                        timef.append(timegroup)
                        tilef.append(tile_id)
                        regionf.append(region_id)


                    if (cs_flag == "complete") and (s1_flag == "complete") and (adem_flag == "complete"):
                        cs_list.append(grp_tile["cs_tile"][:])
                        s1_list.append(grp_tile["s1_tile"][:])
                        adem_list.append(grp_tile["adem_tile"][:])
                        mask_list.append(grp_tile["mask_tile"][:])
                        time.append(timegroup)
                        tile.append(tile_id)
                        region.append(region_id)


    df = pd.DataFrame(data={'cs':cs_list,'s1':s1_list,'adem':adem_list,'mask_list':mask_list,'tile':tile,'region':region,'time':time})
    df_full = pd.DataFrame(data={'cs':cs_full,'s1':s1_full,'mask_full':mask_full,'tile':tilef,'region':regionf,'time':timef})

    return df, df_full

def split_save_data(df,df_full,projectDir,normalization,random_number=7):
    y = df[['adem']]
    X = df.drop(columns=['adem'])
    
    X_train, X_pre, y_train, y_pre = train_test_split(X, y, test_size=0.2, random_state=random_number)
    X_test, X_val, y_test, y_val = train_test_split(X_pre, y_pre, test_size=0.5, random_state=random_number)
    X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(X_val, y_val, test_size=0.2, random_state=random_number)

    outDir = f"{projectDir}/data/interim/normalization-{normalization}/"
    os.makedirs(outDir, exist_ok=True)

    X_test.to_pickle(f"{outDir}/X_test.pkl")
    y_test.to_pickle(f"{outDir}/y_test.pkl")
    X_train.to_pickle(f"{outDir}/X_train.pkl")
    y_train.to_pickle(f"{outDir}/y_train.pkl")
    X_val.to_pickle(f"{outDir}/X_val.pkl")
    y_val.to_pickle(f"{outDir}/y_val.pkl")
    X_train_hpo.to_pickle(f"{outDir}/X_train_hpo.pkl")
    y_train_hpo.to_pickle(f"{outDir}/y_train_hpo.pkl")
    X_val_hpo.to_pickle(f"{outDir}/X_val_hpo.pkl")
    y_val_hpo.to_pickle(f"{outDir}/y_val_hpo.pkl")

    df_full.to_pickle(f"{outDir}/X_all.pkl")
    return print("Data has been split and saved")