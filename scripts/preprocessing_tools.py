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
from shapely.geometry import box, Point
from scipy.ndimage import distance_transform_edt, binary_dilation
import h5py
from sklearn.model_selection import train_test_split
import pandas as pd
import earthaccess
from rasterio.transform import Affine, xy
from rasterio.features import shapes
import gc
import pickle

# #################################################
# #######       00 - Generate regions       #######
# #################################################

# def generate_regions(region_ids,cs_files,project_crs,projectDir):
#     for i in range(len(region_ids)):
#         cs_file = cs_files[i]
#         ds = xr.open_dataset(cs_file, decode_coords="all")

#         CSelev = ds["elev_diff"].transpose("time", "y", "x")

#         # Tell rioxarray where spatial dims are
#         CSelev = CSelev.rio.set_spatial_dims(x_dim="x", y_dim="y")

#         # Tell it which variable contains CRS info
#         CSelev = CSelev.rio.write_coordinate_system()
#         CSelev = CSelev.rio.reproject(project_crs)
#         CSelev = CSelev.mean(dim="time")

#         # Get data and transform
#         data = CSelev.squeeze().values

#         # Create mask: True where data exists
#         mask = ~np.isnan(data)

#         # Polygonize
#         results = (
#             {'properties': {'value': v}, 'geometry': s}
#             for s, v in shapes(data, mask=mask, transform=CSelev.rio.transform())
#         )

#         # Convert to GeoDataFrame
#         gdf = geopandas.GeoDataFrame.from_features(list(results), crs=CSelev.rio.crs)

#         # Keep only valid-data polygons (value not nan)
#         gdf = gdf.dropna(subset=["value"])

#         # Merge into one polygon
#         gdf_union = gdf.union_all()

#         outer_poly = gdf_union.convex_hull

#         # add 5 km buffer
#         outer_poly_buffered = outer_poly.buffer(5000)

#         # Wrap into GeoDataFrame
#         gdf_out = geopandas.GeoDataFrame(
#             geometry=[outer_poly_buffered],
#             crs=CSelev.rio.crs
#         )

#         region_id = region_ids[i]
#         # Save to shapefile
#         out_dir = f"{projectDir}/data/initial/regions/"
#         os.makedirs(out_dir, exist_ok = True)
#         outfile = f"{out_dir}/region-{region_id}.shp"
#         gdf_out.to_file(outfile)
#         print(f"Saved region-{region_id}.shp")


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
######         01 - Create icemask         ######
#################################################

def rasterize(gdf,raster):
    # create shapes (geometry, value) for rasterization
    shapes = ((geom, 1) for geom in gdf.geometry)

    # rasterize to match reference raster’s shape, transform, CRS
    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=raster.squeeze().shape,   # (height, width)
        transform=raster.rio.transform(),
        fill=0
    )

    # wrap result as xarray DataArray with same coords/projection
    mask_da = raster.squeeze().copy(data=mask)
    mask_da = mask_da.expand_dims(dim="band")  # put band dimension back
    return mask_da


def get_icemask(region_id, projectDir, project_crs):
    if region_id in ["01","02","03","04","05","06","07"]:
        rgi = geopandas.read_file(f"{projectDir}/data/initial/RGI/RGI2000-v7.0-C-03_arctic_canada_north/RGI2000-v7.0-C-03_arctic_canada_north.shp").to_crs(project_crs)
    elif region_id in ["08","09","10","11","12","13","14","15"]:
        rgi = geopandas.read_file(f"{projectDir}/data/initial/RGI/RGI2000-v7.0-C-04_arctic_canada_south/RGI2000-v7.0-C-04_arctic_canada_south.shp").to_crs(project_crs)
    else:
        print("RGI not found, mask will be zero")

    region = geopandas.read_file(f"{projectDir}/data/initial/regions/region-{region_id}.shp")

    rgi_region = geopandas.clip(rgi, region)

    mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)

    out_dir = f"{projectDir}/data/interim/region-{region_id}/masks/"
    os.makedirs(out_dir, exist_ok = True)

    mask_100m = rasterize(rgi_region,mosaic)
    mask_100m = mask_100m.rio.write_nodata(None)
    mask_100m.attrs.pop("_FillValue", None)
    mask_100m.encoding.pop("_FillValue", None)
    mask_100m = mask_100m.fillna(0)
    mask_100m = mask_100m.astype(np.uint8)


    mask_100m.rio.to_raster(os.path.join(out_dir,'mask_100m.tif'))
    return print(f"Masks saved for region {region_id}")

#################################################
######          02 - Download CS-2         ######
#################################################

def assign_season_month(ts):
    month = ts.dt.month
    return xr.where(month.isin([12, 1, 2]), "winter",
           xr.where(month.isin([3, 4, 5]), "spring",
           xr.where(month.isin([6, 7, 8]), "summer", "fall")))

def season_year(ts):
    year = ts.dt.year
    return xr.where(ts.dt.month == 12, year + 1, year)

def season_to_date_nc(season, year):
    return {
        "winter": f"{year}-01-15",
        "spring": f"{year}-04-15",
        "summer": f"{year}-07-15",
        "fall":   f"{year}-10-15"
    }[season]


def fill_na_bilinear_radius(da, max_dist=5):

    # --- ensure increasing coords ---
    flipped_y = False
    if not np.all(np.diff(da["y"].values) > 0):
        da = da.sortby("y")
        flipped_y = True

    flipped_x = False
    if not np.all(np.diff(da["x"].values) > 0):
        da = da.sortby("x")
        flipped_x = True

    nan_mask = da.isnull()

    dist = xr.DataArray(
        distance_transform_edt(nan_mask.values),
        coords=da.coords,
        dims=da.dims,
    )

    # bilinear interpolation
    da_interp = da.interpolate_na("x", method="linear")
    da_interp = da_interp.interpolate_na("y", method="linear")

    out = da.where(~nan_mask | (dist > max_dist), da_interp)

    # --- restore original orientation ---
    if flipped_y:
        out = out.sortby("y", ascending=False)
    if flipped_x:
        out = out.sortby("x", ascending=False)

    return out

def fill_na_nearest_radius(da, max_dist=5):

    # --- ensure increasing coords ---
    flipped_y = False
    if not np.all(np.diff(da["y"].values) > 0):
        da = da.sortby("y")
        flipped_y = True

    flipped_x = False
    if not np.all(np.diff(da["x"].values) > 0):
        da = da.sortby("x")
        flipped_x = True

    data = da.values
    nan_mask = np.isnan(data)

    # distance + nearest valid indices
    dist, inds = distance_transform_edt(
        nan_mask,
        return_distances=True,
        return_indices=True
    )

    # nearest valid values
    nearest = data[tuple(inds)]

    out = data.copy()

    fill_mask = nan_mask & (dist <= max_dist)
    out[fill_mask] = nearest[fill_mask]

    out = xr.DataArray(out, coords=da.coords, dims=da.dims)

    # --- restore original orientation ---
    if flipped_y:
        out = out.sortby("y", ascending=False)
    if flipped_x:
        out = out.sortby("x", ascending=False)

    return out


def save_seasonal_cryosat(region_id,dataDir,CSfiles,projectDir,project_crs):
    
    outDir = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/"
    # Create output directory if it doesn"t exist
    os.makedirs(outDir, exist_ok=True)

    mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)
    mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)



    if len(CSfiles) == 0:
        raise ValueError(f"No CryoSat files found for region {region_id}")
    dataDir = f"{projectDir}/data/initial/cryoswath/"
    datasets = []
    datasets2 = []
    for f in CSfiles:
        ds = xr.open_dataset(os.path.join(dataDir,f), decode_coords="all")
        ds = ds.transpose("time", "y", "x")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds = ds.rio.write_coordinate_system()
        ds = ds.rio.reproject(project_crs)
        ds = ds.rio.write_crs(project_crs)
        ds["elev"] = ds["elev_diff"] + ds["elev_diff_ref"]
        ds["elev"] = ds["elev"].where(ds["elev_diff_error"] <= 25)
        ds = ds.assign_coords(season=assign_season_month(ds["time"]))
        ds = ds.assign_coords(season_year=season_year(ds["time"]))
        dselev = ds["elev"]
        dselev = dselev.rio.write_crs(project_crs)
        seasonal = dselev.groupby(["season_year", "season"]).median("time")

        # Convert grouped array to list of DataArrays with time coord
        seasonal_list = []

        for year in seasonal["season_year"].values:
            for season in seasonal["season"].values:
                
                da = seasonal.sel(season_year=year, season=season)

                # skip empty seasons
                if da.notnull().sum() == 0:
                    continue

                timestamp = np.datetime64(season_to_date_nc(str(season), int(year)))

                da = da.expand_dims(time=[timestamp])
                seasonal_list.append(da)

        # concatenate into one time series
        seasonal_list = [da.drop_vars(["season_year","season"], errors="ignore") for da in seasonal_list]
        seasonal_time = xr.concat(seasonal_list, dim="time", coords="minimal")

        # sort by time
        seasonal_time = seasonal_time.sortby("time")

        seasonal_filled = seasonal_time.copy()

        mosaic_500 = (mosaic.coarsen(x=5, y=5, boundary="trim").mean())
        mosaic_coarse = mosaic_500.rio.reproject_match(mosaic)
        mosaic_coarse = fill_na_nearest_radius(mosaic_coarse,50)

        pad = 10

        dx = float(seasonal_filled.x[1] - seasonal_filled.x[0])
        dy = float(seasonal_filled.y[1] - seasonal_filled.y[0])

        seasonal_expanded = seasonal_filled.pad(x=(pad, pad), y=(pad, pad), constant_values=np.nan)

        seasonal_expanded = seasonal_expanded.assign_coords(
            x=np.arange(seasonal_expanded.sizes["x"]) * dx + seasonal_filled.x.values[0] - pad*dx,
            y=np.arange(seasonal_expanded.sizes["y"]) * dy + seasonal_filled.y.values[0] - pad*dy
        )

        seasonal_expanded = seasonal_expanded.rio.write_crs(project_crs)

        seasonal_expanded = fill_na_nearest_radius(seasonal_expanded,20).rio.reproject_match(mosaic)

        seasonal_expanded = seasonal_expanded.where(mask100 != 0, mosaic_coarse)

        seasonal_time = seasonal_time.rio.reproject_match(mosaic)
        datasets.append(seasonal_time)
        datasets2.append(seasonal_expanded)

    CS1 = xr.concat(datasets, dim="source")
    CS1 = CS1.median(dim="source")
    CS2 = xr.concat(datasets2, dim="source")
    CS2 = CS2.median(dim="source")
    CS2_updated = CS1.combine_first(CS2)

    out_nc = os.path.join(outDir, f"cs_raw_gathered_region-{region_id}.nc")
    out_nc2 = os.path.join(outDir, f"cs_gathered_region-{region_id}.nc")

    CS1.name = "cs-elevation"
    CS1.to_netcdf(out_nc)
    CS2_updated.name = "cs-elevation"
    CS2_updated.to_netcdf(out_nc2)

    return print(f"Saved region {region_id}")


#################################################
#### 03 - Download and preprocess Sentinel 1 ####
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


def regrid_sentinel1(region_id,projectDir):

    print(f"Starting on region {region_id}")

    mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)

    out_dir = f"{projectDir}/data/interim/region-{region_id}/sentinel-1/"
    os.makedirs(out_dir, exist_ok=True)

    years = list(range(2014, 2025))
    seasons = ['winter', 'spring', 'summer', 'fall']

    rasters = []
    times   = []

    for year in years:
        for season in seasons:
            s1file = f"{projectDir}/data/initial/sentinel-1/region-{region_id}/S1_{season}_{year}.tif"

            if not os.path.exists(s1file):
                continue
            s1 = rioxarray.open_rasterio(s1file,masked=True)

            s1 = s1.rio.reproject_match(mosaic,resampling=rasterio.enums.Resampling.average)
            if "band" in s1.dims:
                s1 = s1.squeeze("band")
            
            s1 = fill_na_bilinear_radius(s1)

            timestamp = season_to_date_nc(season, year)

            rasters.append(s1)
            times.append(np.datetime64(timestamp))
    stacked = xr.concat(rasters, dim="time", coords="minimal")
    stacked = stacked.assign_coords(time=("time", times))
    stacked.name = "s1-backscatter"

    # --- Save ---
    outfile = f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered_region-{region_id}.nc"
    stacked.to_netcdf(outfile)

    return print(f"Finished region {region_id}")

#################################################
######     04/05 - Seasonal ArcticDEM      ######
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

    # Filter out points that differ more than 50 m from the ArcticDEM mosaic
    diff_vs_arctic2 = (strip_pass2 - arctic_on_strip)
    #strip_final = strip_pass2.where(diff_vs_arctic2 <= 50.0)
    # This new line ensures that areas where diff_vs_arctic2 are nan are still kept.
    strip_final = strip_pass2.where((diff_vs_arctic2 <= 25.0) | diff_vs_arctic2.isnull())


    return strip_final

### CREATE REFERENCE DEM ###
def open_reference_for_label(label_year, season, cryosat_ds, arctic_dem_mosaic):
    """
    Select seasonal CryoSat slice from NetCDF and fill gaps with ArcticDEM.
    CryoSat and ArcticDEM are already on same grid.
    """

    if label_year < 2010:
        return None, None

    # map season → month
    season_month = {
        "winter": 1,
        "spring": 4,
        "summer": 7,
        "fall": 10
    }

    month = season_month[season]
    time_str = f"{label_year:04d}-{month:02d}-15"

    try:
        cryo = cryosat_ds.sel(time=time_str, method="nearest").squeeze()
    except Exception:
        return None, None

    if cryo is None:
        return None, None

    # already same grid → no reprojection needed
    ref = cryo.fillna(arctic_dem_mosaic)

    return ref, arctic_dem_mosaic



def process_strip(p, cryosat_root, arctic_dem_mosaic, project_crs, out_dir):
    dt = parse_date_from_name(p)
    if dt is None:
        return None
    season, label_year = month_to_season_and_label_year(dt)
    if label_year < 2010:
        return None
    reference, arctic = open_reference_for_label(label_year,season,cryosat_root,arctic_dem_mosaic)

    #reference, arctic = open_reference_for_label(label_year, season, cryosat_root, arctic_dem_mosaic)
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
    #cryosat_root = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal"

    arctic_dem_mosaic = rioxarray.open_rasterio(arcticdem_path, masked=True).squeeze()

    cryosat_path = f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_raw_gathered_region-{region_id}.nc"

    # load CryoSat NetCDF once
    cryosat_ds = xr.open_dataset(cryosat_path)["cs-elevation"]

    # The criteria which a pixel has to fulfill
    # diff_criteria = 25

    strip_files = [os.path.join(strip_dir, f) for f in os.listdir(strip_dir) if f.endswith(".tif")]
    print(f"Found {len(strip_files)} strips")

    bag = db.from_sequence(strip_files, npartitions=cores)
    #bag = bag.map(lambda p: process_strip(p,cryosat_root,arctic_dem_mosaic,project_crs,out_dir))
    bag = bag.map(lambda p: process_strip(p,cryosat_ds,arctic_dem_mosaic,project_crs,out_dir))
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


def seasonal_arcticdem(region_id, label_tagger, projectDir):

    dask.config.set({
        "temporary_directory": "/tmp/dask",
        "array.slicing.split_large_chunks": True,
    })

    strip_dir = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/co-registered-strips/"
    strip_files = [os.path.join(strip_dir, f) for f in os.listdir(strip_dir) if f.endswith(".tif")]

    years = list(range(2014, 2025))
    seasons = ['winter','spring','summer','fall']

    mosaic = rioxarray.open_rasterio(
        f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",
        masked=True
    ).squeeze("band")

    # --- load labels ---
    labels_file = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/labels_{label_tagger}_region-{region_id}.csv"
    labels_df = pd.read_csv(labels_file)

    label_dict = dict(zip(labels_df["file"], labels_df["label"]))

    label_weights = {1:1.0, 2:0.5, 3:0.25}

    rasters = []
    rasters_fill = []
    times = []

    for year in years:
        for season in seasons:

            files_in_group = [f for f in strip_files if is_in_year_season(f, year, season)]
            if not files_in_group:
                continue

            print(f"Processing {year} {season}: {len(files_in_group)} files")

            # das = []
            # weights = []

            weighted_sum = None
            weight_total = None

            for f in files_in_group:

                fname = os.path.basename(f)
                label = label_dict.get(fname, None)

                if label is None or label == 4:
                    continue

                weight = label_weights[label]

                da = rioxarray.open_rasterio(
                    f,
                    masked=True,
                    chunks={"x":2048,"y":2048}
                )

                da = da.rio.reproject_match(
                    mosaic,
                    resampling=rasterio.enums.Resampling.average
                ).squeeze("band")

                da = da.astype("float32")

                # --- filter label 3 strips ---
                if label == 3:
                    diff = da - mosaic
                    da = da.where(np.abs(diff) <= 10)

                valid = (~da.isnull()).astype("float32")

                da = da.fillna(0)

                weighted = da * weight
                weights = valid * weight

                if weighted_sum is None:
                    weighted_sum = weighted
                    weight_total = weights
                else:
                    weighted_sum = weighted_sum + weighted
                    weight_total = weight_total + weights

            # if len(das) == 0:
            #     continue

            # # --- stack strips lazily ---
            # stack = xr.concat(das, dim="strip")

            # weights = xr.DataArray(
            #     weights,
            #     dims=["strip"]
            # )

            # --- weighted mean ---
            mean_da = weighted_sum / weight_total
            mean_da = mean_da.where(weight_total > 0)

            timestamp = season_to_date_nc(season, year)

            mean_da_fill = fill_na_bilinear_radius(mean_da.copy())

            rasters.append(mean_da)
            rasters_fill.append(mean_da_fill)
            times.append(np.datetime64(timestamp))

    stacked = xr.concat(rasters, dim="time", coords="minimal")
    stacked = stacked.assign_coords(time=("time", times))
    stacked.name = "adem-elevation"

    outfile = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/ArcticDEM_raw_gathered_region-{region_id}.nc"
    stacked.to_netcdf(outfile)

    stacked_fill = xr.concat(rasters_fill, dim="time", coords="minimal")
    stacked_fill = stacked_fill.assign_coords(time=("time", times))
    stacked_fill.name = "adem-elevation"

    outfile_fill = f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/ArcticDEM_gathered_region-{region_id}.nc"
    stacked_fill.to_netcdf(outfile_fill)

    print("Saved:", outfile)
    print(f"All netcdfs saved for region {region_id}")


#################################################
######          06 - Create tiles          ######
#################################################


def create_tiles(
    region_id,
    projectDir,
    NAME,
    TILE_SIZE: int,
    OVERLAP: int,
    BUFFER_PIXELS: int
):

    mask_path = (
        f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif"
    )

    # Step between tile origins
    stride = TILE_SIZE - OVERLAP
    if stride <= 0:
        raise ValueError("OVERLAP must be smaller than TILE_SIZE")

    # -------------------------
    # READ MASK
    # -------------------------
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    mask = (mask == 1)

    # -------------------------
    # CREATE BUFFERED VALID AREA
    # Ensure at least BUFFER_PIXELS zeros around edge
    # -------------------------
    dist = distance_transform_edt(~mask)
    valid_area = dist < BUFFER_PIXELS

    # Bounding box of valid mask
    rows, cols = np.where(valid_area)
    if len(rows) == 0:
        raise ValueError("No valid pixels after applying buffer")

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # Ensure bounds allow a tile
    row_max = max(row_max, row_min + TILE_SIZE)
    col_max = max(col_max, col_min + TILE_SIZE)

    # -------------------------
    # GENERATE TILE STARTS
    # (This fixes the coverage issue)
    # -------------------------
    row_starts = list(range(row_min, row_max - TILE_SIZE + 1, stride))
    col_starts = list(range(col_min, col_max - TILE_SIZE + 1, stride))

    # Force coverage at far edges
    last_row_start = row_max - TILE_SIZE
    last_col_start = col_max - TILE_SIZE

    if last_row_start not in row_starts:
        row_starts.append(last_row_start)

    if last_col_start not in col_starts:
        col_starts.append(last_col_start)

    row_starts = sorted(set(row_starts))
    col_starts = sorted(set(col_starts))

    # -------------------------
    # GENERATE TILES
    # -------------------------
    tiles = []
    tile_id = 0

    for r0 in row_starts:
        for c0 in col_starts:
            r1 = r0 + TILE_SIZE
            c1 = c0 + TILE_SIZE

            # Skip if outside raster bounds
            if r1 > height or c1 > width:
                continue

            # Only keep tiles that contain valid pixels
            if not np.any(valid_area[r0:r1, c0:c1]):
                continue

            # Convert to geographic bounds
            x_min, y_max = xy(transform, r0, c0, offset="ul")
            x_max, y_min = xy(transform, r1, c1, offset="ul")

            geom = box(x_min, y_min, x_max, y_max)

            mask_count = np.sum(mask[r0:r1, c0:c1])

            tiles.append({
                "tile_id": tile_id,
                "top": r0,
                "left": c0,
                "geometry": geom,
                "mask_count": mask_count
            })

            tile_id += 1

    # -------------------------
    # EXPORT TO GEOPACKAGE
    # -------------------------
    gdf = geopandas.GeoDataFrame(tiles, crs=crs)
    print(f"All tiles: {len(gdf)}")
    gdf = gdf[gdf["mask_count"] > 0]
    outdir = f"{projectDir}/data/interim/region-{region_id}/masks/"
    gdf.to_file(f"{outdir}tiles_{NAME}.gpkg", driver="GPKG")

    print(f"Tiles saved for region {region_id}")
    print(f"Total tiles: {len(gdf)}")



#################################################
###### 09 - Normalize, tile and split data ######
#################################################

# def mask_da(da,mask):
#     arr = da.values
#     mask_arr = mask.values
#     arr[mask_arr == 0] = 0
#     masked_da = da.copy(data=arr)
#     return masked_da

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


def normalize_tiling_masking(region_id,projectDir,project_crs,experiment):

    print(f"Running region {region_id}")

    tiles = geopandas.read_file(f"{projectDir}/data/interim/region-{region_id}/masks/tiles_{experiment}.gpkg")

    mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)
    mask = mask100.values.astype(bool)
    dist = distance_transform_edt(~mask)
    expanded = dist <= 20
    mask_expanded = xr.DataArray(expanded.astype(np.uint8),coords=mask100.coords,dims=mask100.dims)
    
    s1 = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered_region-{region_id}.nc")['s1-backscatter'].rio.write_crs(project_crs)

    cs = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered_region-{region_id}.nc")['cs-elevation'].rio.write_crs(project_crs)

    adem = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/ArcticDEM_gathered_region-{region_id}.nc")['adem-elevation'].rio.write_crs(project_crs)


    s1_n, _, _ = zscore(s1)
    s1_n = s1_n.where(mask_expanded, 0)
    mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)
    mosaic = fill_na_bilinear_radius(mosaic,50)
    cs = cs - mosaic
    adem = adem - mosaic
    cs_mean = cs.mean(skipna=True)
    cs_std = cs.std(skipna=True)
    adem_n = (adem - cs_mean) / cs_std
    adem_n = adem_n.where(mask_expanded, 0)
    cs_n = (cs - cs_mean) / cs_std
    cs_n = cs_n.where(mask_expanded, 0)



    h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_{experiment}.h5"


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


            for t in range(len(tiles)):

                bounds = tiles.iloc[t].geometry.bounds
                cs_tile = cs_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
                s1_tile = s1_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
                adem_tile = adem_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values

                cs_tile = cs_tile.astype(np.float32)
                s1_tile = s1_tile.astype(np.float32)
                adem_tile = adem_tile.astype(np.float32)

                tile_id = f"tile_{tiles.iloc[t].tile_id}"

                grp_tile = grp_time.create_group(tile_id)

                # Create datasets within tile subgroup + flip CS data
                grp_tile.create_dataset("cs_tile", data=cs_tile, compression="gzip", compression_opts=4)
                grp_tile.create_dataset("s1_tile", data=s1_tile, compression="gzip", compression_opts=4)
                grp_tile.create_dataset("adem_tile", data=adem_tile, compression="gzip", compression_opts=4)

                # Add flags as attributes
                grp_tile.attrs["cs_flag"] = flag_tile(cs_tile)
                grp_tile.attrs["s1_flag"] = flag_tile(s1_tile)
                grp_tile.attrs["adem_flag"] = flag_tile(adem_tile)

            print(f"✅ Added data for {timegroup}")
    return print(f"HDF5 file saved for region {region_id}")


def load_complete_tiles(region_ids,experiment,projectDir):
    """
    Load cs_tile, s1_tile, and adem_tile arrays from an HDF5 file,
    but only for tiles where all flags == 'complete'.
    Returns three numpy arrays: cs_all, s1_all, adem_all.
    """

    cs_list, s1_list, adem_list, s1_full, cs_full, mask_list, mask_full = [], [], [], [], [], [], []
    cs_list, s1_list, adem_list, s1_full, cs_full = [], [], [], [], []
    time, tile, region = [], [], []
    timef, tilef, regionf = [], [], []

    for region_id in region_ids:

        h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_{experiment}.h5"


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
                        #mask_full.append(grp_tile["mask_tile"][:])
                        timef.append(timegroup)
                        tilef.append(tile_id)
                        regionf.append(region_id)


                    if (cs_flag == "complete") and (s1_flag == "complete") and (adem_flag == "complete"):
                        cs_list.append(grp_tile["cs_tile"][:])
                        s1_list.append(grp_tile["s1_tile"][:])
                        adem_list.append(grp_tile["adem_tile"][:])
                        #mask_list.append(grp_tile["mask_tile"][:])
                        time.append(timegroup)
                        tile.append(tile_id)
                        region.append(region_id)


    # df = pd.DataFrame(data={'cs':cs_list,'s1':s1_list,'adem':adem_list,'mask_list':mask_list,'tile':tile,'region':region,'time':time})
    # df_full = pd.DataFrame(data={'cs':cs_full,'s1':s1_full,'mask_full':mask_full,'tile':tilef,'region':regionf,'time':timef})

    df = pd.DataFrame(data={'cs':cs_list,'s1':s1_list,'adem':adem_list,'tile':tile,'region':region,'time':time})
    df_full = pd.DataFrame(data={'cs':cs_full,'s1':s1_full,'tile':tilef,'region':regionf,'time':timef})

    return df, df_full

def split_save_data_train_val(df,df_full,projectDir,experiment,random_number=7):
    y = df[['adem']]
    X = df.drop(columns=['adem'])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_number)
    X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(X_val, y_val, test_size=0.2, random_state=random_number)

    outDir = f"{projectDir}/data/interim/{experiment}/"
    os.makedirs(outDir, exist_ok=True)

    #X_test.to_pickle(f"{outDir}/X_test.pkl")
    #y_test.to_pickle(f"{outDir}/y_test.pkl")
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

def normalize_tiling_masking_prediction(region_id,projectDir,project_crs,experiment):

    print(f"Running region {region_id}")

    tiles = geopandas.read_file(f"{projectDir}/data/interim/region-{region_id}/masks/tiles_{experiment}.gpkg")

    mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)
    mask = mask100.values.astype(bool)
    dist = distance_transform_edt(~mask)
    expanded = dist <= 20
    mask_expanded = xr.DataArray(expanded.astype(np.uint8),coords=mask100.coords,dims=mask100.dims)
    
    s1 = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered_region-{region_id}.nc")['s1-backscatter'].rio.write_crs(project_crs)

    cs = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered_region-{region_id}.nc")['cs-elevation'].rio.write_crs(project_crs)
    

    s1_n, _, _ = zscore(s1)
    s1_n = s1_n.where(mask_expanded, 0)
    mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)
    mosaic = fill_na_bilinear_radius(mosaic,50)
    cs = cs - mosaic
    cs_mean = cs.mean(skipna=True)
    cs_std = cs.std(skipna=True)
    cs_n = (cs - cs_mean) / cs_std
    cs_n = cs_n.where(mask_expanded, 0)



    h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_pred_{experiment}.h5"


    with h5py.File(h5_path, "w") as f:
        for time in s1.time.values:
            s1_t = s1_n.sel(time=time)
            cs_t = cs_n.sel(time=time)

            year = time.astype(str)[:4]
            month = time.astype(str)[5:7]

            timegroup = f"{year}_{month}"
            grp_time = f.create_group(timegroup)


            for t in range(len(tiles)):

                bounds = tiles.iloc[t].geometry.bounds
                cs_tile = cs_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
                s1_tile = s1_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values

                cs_tile = cs_tile.astype(np.float32)
                s1_tile = s1_tile.astype(np.float32)

                tile_id = f"tile_{tiles.iloc[t].tile_id}"

                grp_tile = grp_time.create_group(tile_id)

                # Create datasets within tile subgroup + flip CS data
                grp_tile.create_dataset("cs_tile", data=cs_tile, compression="gzip", compression_opts=4)
                grp_tile.create_dataset("s1_tile", data=s1_tile, compression="gzip", compression_opts=4)

                # Add flags as attributes
                grp_tile.attrs["cs_flag"] = flag_tile(cs_tile)
                grp_tile.attrs["s1_flag"] = flag_tile(s1_tile)

            print(f"✅ Added data for {timegroup}")
    return print(f"HDF5 file saved for region {region_id}")

def load_complete_tiles_prediction(region_ids,experiment,projectDir):

    cs_list, s1_list = [], []
    time, tile, region = [], [], []

    for region_id in region_ids:

        h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_pred_{experiment}.h5"


        with h5py.File(h5_path, "r") as f:
            for timegroup in f.keys():
                grp_time = f[timegroup]

                for tile_id in grp_time.keys():
                    grp_tile = grp_time[tile_id]

                    # Check all three flags
                    cs_flag = grp_tile.attrs.get("cs_flag", "incomplete")
                    s1_flag = grp_tile.attrs.get("s1_flag", "incomplete")


                    if (cs_flag == "complete") and (s1_flag == "complete"):
                        cs_list.append(grp_tile["cs_tile"][:])
                        s1_list.append(grp_tile["s1_tile"][:])
                        time.append(timegroup)
                        tile.append(tile_id)
                        region.append(region_id)



    df = pd.DataFrame(data={'cs':cs_list,'s1':s1_list,'tile':tile,'region':region,'time':time})

    outDir = f"{projectDir}/data/interim/{experiment}/"
    os.makedirs(outDir, exist_ok=True)

    df.to_pickle(f"{outDir}/X_all_pred.pkl")

    return print(f"Data has been saved: {outDir}/X_all_pred.pkl")

############ END OF PREPROCESSING
####### THIS IS WHERE I AM AT, THE FUNCTIONS BETWEEN HERE AND IS2 ARE LIKELY NOT IN USE

# def tiling_masking_prediction(region_id,projectDir,project_crs,experiment):

#     print(f"Running region {region_id}")

#     tiles = geopandas.read_file(f"{projectDir}/data/interim/region-{region_id}/masks/tiles_{experiment}.gpkg")

#     mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)
#     mask = mask100.values.astype(bool)
#     dist = distance_transform_edt(~mask)
#     expanded = dist <= 20
#     mask_expanded = xr.DataArray(expanded.astype(np.uint8),coords=mask100.coords,dims=mask100.dims)
    
#     s1 = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered_region-{region_id}.nc")['s1-backscatter'].rio.write_crs(project_crs)

#     cs = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered_region-{region_id}.nc")['cs-elevation'].rio.write_crs(project_crs)
    
#     s1 = s1.where(mask_expanded, 0)
#     mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)
#     mosaic = fill_na_bilinear_radius(mosaic,50)
#     cs = cs - mosaic
#     cs = cs.where(mask_expanded, 0)



#     h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_pred_{experiment}.h5"


#     with h5py.File(h5_path, "w") as f:
#         for time in s1.time.values:
#             s1_t = s1.sel(time=time)
#             cs_t = cs.sel(time=time)

#             year = time.astype(str)[:4]
#             month = time.astype(str)[5:7]

#             timegroup = f"{year}_{month}"
#             grp_time = f.create_group(timegroup)


#             for t in range(len(tiles)):

#                 bounds = tiles.iloc[t].geometry.bounds
#                 cs_tile = cs_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
#                 s1_tile = s1_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values

#                 cs_tile = cs_tile.astype(np.float32)
#                 s1_tile = s1_tile.astype(np.float32)

#                 tile_id = f"tile_{tiles.iloc[t].tile_id}"

#                 grp_tile = grp_time.create_group(tile_id)

#                 # Create datasets within tile subgroup + flip CS data
#                 grp_tile.create_dataset("cs_tile", data=cs_tile, compression="gzip", compression_opts=4)
#                 grp_tile.create_dataset("s1_tile", data=s1_tile, compression="gzip", compression_opts=4)

#                 # Add flags as attributes
#                 grp_tile.attrs["cs_flag"] = flag_tile(cs_tile)
#                 grp_tile.attrs["s1_flag"] = flag_tile(s1_tile)

#             print(f"✅ Added data for {timegroup}")
#     return print(f"HDF5 file saved for region {region_id}")


# def tiling_masking(region_id,projectDir,project_crs,experiment):

#     print(f"Running region {region_id}")

#     tiles = geopandas.read_file(f"{projectDir}/data/interim/region-{region_id}/masks/tiles_{experiment}.gpkg")

#     mask100 = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True)
#     mask = mask100.values.astype(bool)
#     dist = distance_transform_edt(~mask)
#     expanded = dist <= 20
#     mask_expanded = xr.DataArray(expanded.astype(np.uint8),coords=mask100.coords,dims=mask100.dims)
    
#     s1 = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/sentinel-1/S1_gathered_region-{region_id}.nc")['s1-backscatter'].rio.write_crs(project_crs)

#     cs = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered_region-{region_id}.nc")['cs-elevation'].rio.write_crs(project_crs)

#     adem = xr.open_dataset(f"{projectDir}/data/interim/region-{region_id}/ArcticDEM/ArcticDEM_gathered_region-{region_id}.nc")['adem-elevation'].rio.write_crs(project_crs)


#     s1 = s1.where(mask_expanded, 0)
#     mosaic = rioxarray.open_rasterio(f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",masked=True)
#     mosaic = fill_na_bilinear_radius(mosaic,50)
#     cs = cs - mosaic
#     adem = adem - mosaic
#     cs = cs.where(mask_expanded, 0)
#     adem = adem.where(mask_expanded, 0)

#     h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_{experiment}.h5"


#     with h5py.File(h5_path, "w") as f:
#         for time in s1.time.values:
#             s1_t = s1.sel(time=time)
#             cs_t = cs.sel(time=time)
#             if time in adem.time.values:
#                 adem_t = adem.sel(time=time)
#             else:
#                 nans = np.zeros_like(s1_t.values)*np.nan
#                 adem_t = s1_t.copy(data=nans)


#             year = time.astype(str)[:4]
#             month = time.astype(str)[5:7]

#             timegroup = f"{year}_{month}"
#             grp_time = f.create_group(timegroup)



#             for t in range(len(tiles)):

#                 bounds = tiles.iloc[t].geometry.bounds
#                 cs_tile = cs_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
#                 s1_tile = s1_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values
#                 adem_tile = adem_t.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3]).squeeze().values

#                 cs_tile = cs_tile.astype(np.float32)
#                 s1_tile = s1_tile.astype(np.float32)
#                 adem_tile = adem_tile.astype(np.float32)

#                 tile_id = f"tile_{tiles.iloc[t].tile_id}"

#                 grp_tile = grp_time.create_group(tile_id)

#                 # Create datasets within tile subgroup + flip CS data
#                 grp_tile.create_dataset("cs_tile", data=cs_tile, compression="gzip", compression_opts=4)
#                 grp_tile.create_dataset("s1_tile", data=s1_tile, compression="gzip", compression_opts=4)
#                 grp_tile.create_dataset("adem_tile", data=adem_tile, compression="gzip", compression_opts=4)

#                 # Add flags as attributes
#                 grp_tile.attrs["cs_flag"] = flag_tile(cs_tile)
#                 grp_tile.attrs["s1_flag"] = flag_tile(s1_tile)
#                 grp_tile.attrs["adem_flag"] = flag_tile(adem_tile)

#             print(f"✅ Added data for {timegroup}")
#     return print(f"HDF5 file saved for region {region_id}")





# def load_normalize_complete_tiles_prediction(region_ids,experiment,projectDir):

#     cs_list, s1_list = [], []
#     time, tile, region = [], [], []

#     for region_id in region_ids:

#         h5_path = f"{projectDir}/data/interim/region-{region_id}/seasonal_tiles_pred_{experiment}.h5"


#         with h5py.File(h5_path, "r") as f:
#             for timegroup in f.keys():
#                 grp_time = f[timegroup]

#                 for tile_id in grp_time.keys():
#                     grp_tile = grp_time[tile_id]

#                     # Check all three flags
#                     cs_flag = grp_tile.attrs.get("cs_flag", "incomplete")
#                     s1_flag = grp_tile.attrs.get("s1_flag", "incomplete")


#                     if (cs_flag == "complete") and (s1_flag == "complete"):
#                         cs_list.append(grp_tile["cs_tile"][:])
#                         s1_list.append(grp_tile["s1_tile"][:])
#                         time.append(timegroup)
#                         tile.append(tile_id)
#                         region.append(region_id)



#     df = pd.DataFrame(data={'cs':cs_list,'s1':s1_list,'tile':tile,'region':region,'time':time})

#     with open(f"{projectDir}/data/interim/{experiment}/normalization_{experiment}.pkl", "rb") as f:
#         norm_params = pickle.load(f)

#     cs_mean = norm_params["cs"]["mean"]
#     cs_std = norm_params["cs"]["std"]

#     s1_mean = norm_params["s1"]["mean"]
#     s1_std = norm_params["s1"]["std"]

#     df["cs"] = df["cs"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))
#     df["s1"] = df["s1"].apply(lambda arr: np.where(arr != 0, (arr - s1_mean) / s1_std, 0))

#     outDir = f"{projectDir}/data/interim/{experiment}/"
#     os.makedirs(outDir, exist_ok=True) 

#     df.to_pickle(f"{outDir}/X_all_pred.pkl")

#     return print(f"Data has been saved: {outDir}/X_all_pred.pkl")


# def split_save_data(df,df_full,projectDir,experiment,random_number=7):
#     y = df[['adem']]
#     X = df.drop(columns=['adem'])
    
#     X_train, X_pre, y_train, y_pre = train_test_split(X, y, test_size=0.2, random_state=random_number)
#     X_test, X_val, y_test, y_val = train_test_split(X_pre, y_pre, test_size=0.5, random_state=random_number)
#     X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(X_val, y_val, test_size=0.2, random_state=random_number)

#     outDir = f"{projectDir}/data/interim/{experiment}/"
#     os.makedirs(outDir, exist_ok=True)

#     X_test.to_pickle(f"{outDir}/X_test.pkl")
#     y_test.to_pickle(f"{outDir}/y_test.pkl")
#     X_train.to_pickle(f"{outDir}/X_train.pkl")
#     y_train.to_pickle(f"{outDir}/y_train.pkl")
#     X_val.to_pickle(f"{outDir}/X_val.pkl")
#     y_val.to_pickle(f"{outDir}/y_val.pkl")
#     X_train_hpo.to_pickle(f"{outDir}/X_train_hpo.pkl")
#     y_train_hpo.to_pickle(f"{outDir}/y_train_hpo.pkl")
#     X_val_hpo.to_pickle(f"{outDir}/X_val_hpo.pkl")
#     y_val_hpo.to_pickle(f"{outDir}/y_val_hpo.pkl")

#     df_full.to_pickle(f"{outDir}/X_all.pkl")
#     return print("Data has been split and saved")


# def split_save_data_year(df, df_full, year_test, year_val,
#                          projectDir, experiment, random_number=7):

#     # Separate target and features
#     y = df[['adem']]
#     X = df.drop(columns=['adem'])

#     # Extract year from YYYY_MM strings
#     years = X["time"].str[:4].astype(int)

#     # Test set
#     test_mask = years == year_test
#     X_test = X.loc[test_mask].copy()
#     y_test = y.loc[test_mask].copy()

#     # Validation set
#     val_mask = years == year_val
#     X_val = X.loc[val_mask].copy()
#     y_val = y.loc[val_mask].copy()

#     # Training set = everything else
#     train_mask = ~(test_mask | val_mask)
#     X_train = X.loc[train_mask].copy()
#     y_train = y.loc[train_mask].copy()

#     del X
#     del y

#     # HPO split from validation year only
#     X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(
#         X_val,
#         y_val,
#         test_size=0.2,
#         random_state=random_number
#     )

#     outDir = f"{projectDir}/data/interim/{experiment}/"
#     os.makedirs(outDir, exist_ok=True)

#     X_test.to_pickle(f"{outDir}/X_test.pkl")
#     y_test.to_pickle(f"{outDir}/y_test.pkl")

#     X_train.to_pickle(f"{outDir}/X_train.pkl")
#     y_train.to_pickle(f"{outDir}/y_train.pkl")

#     X_val.to_pickle(f"{outDir}/X_val.pkl")
#     y_val.to_pickle(f"{outDir}/y_val.pkl")

#     X_train_hpo.to_pickle(f"{outDir}/X_train_hpo.pkl")
#     y_train_hpo.to_pickle(f"{outDir}/y_train_hpo.pkl")

#     X_val_hpo.to_pickle(f"{outDir}/X_val_hpo.pkl")
#     y_val_hpo.to_pickle(f"{outDir}/y_val_hpo.pkl")

#     df_full.to_pickle(f"{outDir}/X_all.pkl")

#     print(
#         f"Data split using year_test={year_test} as test set "
#         f"and year_val={year_val} as validation set"
#     )

# def normalize_split_save_data_year(df, df_full, year_test, year_val,
#                          projectDir, experiment, random_number=7):

#     # Separate target and features
#     y = df[['adem']]
#     X = df.drop(columns=['adem'])

#     # Extract year from YYYY_MM strings
#     years = X["time"].str[:4].astype(int)

#     # Test set
#     test_mask = years == year_test
#     X_test = X.loc[test_mask].copy()
#     y_test = y.loc[test_mask].copy()

#     # Validation set
#     val_mask = years == year_val
#     X_val = X.loc[val_mask].copy()
#     y_val = y.loc[val_mask].copy()

#     # Training set = everything else
#     train_mask = ~(test_mask | val_mask)
#     X_train = X.loc[train_mask].copy()
#     y_train = y.loc[train_mask].copy()

#     del X
#     del y

#     cs_values = np.concatenate([arr.ravel() for arr in X_train["cs"]])
#     cs_values = cs_values[cs_values != 0]
#     cs_mean = cs_values.mean()
#     cs_std = cs_values.std()
#     del cs_values

#     s1_values = np.concatenate([arr.ravel() for arr in X_train["s1"]])
#     s1_values = s1_values[s1_values != 0]
#     s1_mean = s1_values.mean()
#     s1_std = s1_values.std()
#     del s1_values

#     X_train["cs"] = X_train["cs"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))
#     X_test["cs"] = X_test["cs"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))
#     X_val["cs"] = X_val["cs"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))

#     y_train["adem"] = y_train["adem"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))
#     y_test["adem"] = y_test["adem"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))
#     y_val["adem"] = y_val["adem"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))

#     X_train["s1"] = X_train["s1"].apply(lambda arr: np.where(arr != 0, (arr - s1_mean) / s1_std, 0))
#     X_test["s1"] = X_test["s1"].apply(lambda arr: np.where(arr != 0, (arr - s1_mean) / s1_std, 0))
#     X_val["s1"] = X_val["s1"].apply(lambda arr: np.where(arr != 0, (arr - s1_mean) / s1_std, 0))

#     df_full["s1"] = df_full["s1"].apply(lambda arr: np.where(arr != 0, (arr - s1_mean) / s1_std, 0))
#     df_full["cs"] = df_full["cs"].apply(lambda arr: np.where(arr != 0, (arr - cs_mean) / cs_std, 0))

#     # HPO split from validation year only
#     X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(
#         X_val,
#         y_val,
#         test_size=0.2,
#         random_state=random_number
#     )

#     outDir = f"{projectDir}/data/interim/{experiment}/"
#     os.makedirs(outDir, exist_ok=True)

#     X_test.to_pickle(f"{outDir}/X_test.pkl")
#     y_test.to_pickle(f"{outDir}/y_test.pkl")

#     X_train.to_pickle(f"{outDir}/X_train.pkl")
#     y_train.to_pickle(f"{outDir}/y_train.pkl")

#     X_val.to_pickle(f"{outDir}/X_val.pkl")
#     y_val.to_pickle(f"{outDir}/y_val.pkl")

#     X_train_hpo.to_pickle(f"{outDir}/X_train_hpo.pkl")
#     y_train_hpo.to_pickle(f"{outDir}/y_train_hpo.pkl")

#     X_val_hpo.to_pickle(f"{outDir}/X_val_hpo.pkl")
#     y_val_hpo.to_pickle(f"{outDir}/y_val_hpo.pkl")

#     df_full.to_pickle(f"{outDir}/X_all.pkl")

#     norm_params = {
#     "cs": {"mean": cs_mean,"std": cs_std,},
#     "s1": {"mean": s1_mean,"std": s1_std,},
#     }

#     with open(os.path.join(outDir, f"normalization_{experiment}.pkl"), "wb") as f:
#         pickle.dump(norm_params, f)

#     print(
#         f"Data split using year_test={year_test} as test set "
#         f"and year_val={year_val} as validation set"
#     )

# def split_save_data_region(df,df_full,region_id,projectDir,experiment,random_number=7):
#     # Separate target and features
#     y = df[['adem']]
#     X = df.drop(columns=['adem'])

#     # Test set = selected region
#     test_mask = X["region"] == region_id

#     X_test = X.loc[test_mask].copy()
#     y_test = y.loc[test_mask].copy()

#     # Remaining regions
#     X_remaining = X.loc[~test_mask].copy()
#     y_remaining = y.loc[~test_mask].copy()

#     del X
#     del y 

#     # 80/20 train/val split on remaining data
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_remaining,
#         y_remaining,
#         test_size=0.2,
#         random_state=random_number
#     )

#     # HPO split from validation set (same as before)
#     X_train_hpo, X_val_hpo, y_train_hpo, y_val_hpo = train_test_split(
#         X_val,
#         y_val,
#         test_size=0.2,
#         random_state=random_number
#     )

#     outDir = f"{projectDir}/data/interim/{experiment}/"
#     os.makedirs(outDir, exist_ok=True)

#     X_test.to_pickle(f"{outDir}/X_test.pkl")
#     y_test.to_pickle(f"{outDir}/y_test.pkl")

#     X_train.to_pickle(f"{outDir}/X_train.pkl")
#     y_train.to_pickle(f"{outDir}/y_train.pkl")

#     X_val.to_pickle(f"{outDir}/X_val.pkl")
#     y_val.to_pickle(f"{outDir}/y_val.pkl")

#     X_train_hpo.to_pickle(f"{outDir}/X_train_hpo.pkl")
#     y_train_hpo.to_pickle(f"{outDir}/y_train_hpo.pkl")

#     X_val_hpo.to_pickle(f"{outDir}/X_val_hpo.pkl")
#     y_val_hpo.to_pickle(f"{outDir}/y_val_hpo.pkl")

#     df_full.to_pickle(f"{outDir}/X_all.pkl")

#     print(f"Data split using region_id={region_id} as test set")

#################################################
######          Download ICESat-2          ######
#################################################

def season_timestamp(year, season):
    return {
        "winter": f"{year}-01-15",
        "spring": f"{year}-04-15",
        "summer": f"{year}-07-15",
        "fall":   f"{year}-10-15"
    }[season]

# ---------------------------------------------------------
# 4. Bin ICESat-2 track points into mask grid cells
# ---------------------------------------------------------
def bin_points_to_grid(lon, lat, h, mask_crs, x_coords, y_coords, res_x, res_y):

    # Raw geodataframe
    gdf = geopandas.GeoDataFrame({
        "h": h,
        "lat": lat,
        "lon": lon
    }, geometry=[Point(xy) for xy in zip(lon, lat)], crs="EPSG:4326")

    # Reproject to target CRS of mask
    gdf = gdf.to_crs(mask_crs)

    # Convert coordinates to pixel indices
    xi = np.floor((gdf.geometry.x - x_coords[0]) / res_x).astype(int)
    yi = np.floor((gdf.geometry.y - y_coords[0]) / res_y).astype(int)

    # Keep points that fall inside the raster
    valid = (
        (xi >= 0) & (xi < len(x_coords)) &
        (yi >= 0) & (yi < len(y_coords))
    )

    df = pd.DataFrame({
        "xi": xi[valid],
        "yi": yi[valid],
        "h": gdf["h"].values[valid]
    })

    # # Mean per pixel
    # df_mean = df.groupby(["yi", "xi"]).mean()

    # Median per pixel
    df_median = df.groupby(["yi", "xi"])["h"].median()

    # Create an empty grid
    grid = np.full((len(y_coords), len(x_coords)), np.nan)

    # for (yy, xx), row in df_mean.iterrows():
    #     grid[yy, xx] = row.h

    for (yy, xx), h_med in df_median.items():
        grid[yy, xx] = h_med

    return grid


import time

def download_icesat2(region_id,projectDir):
    # ---------------------------------------------------------
    # 2. Load region and mask
    # ---------------------------------------------------------

    region = geopandas.read_file(f'{projectDir}/data/initial/regions/region-{region_id}.shp').to_crs('EPSG:4326')

    xmin, ymin, xmax, ymax = region.total_bounds

    buffer = 0.5

    AOI = [
        (xmin, ymin-buffer),
        (xmax, ymin-buffer),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin-buffer)
    ]

    # Load mask grid (this defines CRS & resolution)
    mask = rioxarray.open_rasterio(f"{projectDir}/data/interim/region-{region_id}/masks/mask_100m.tif",masked=True).squeeze()
    mask_crs = mask.rio.crs
    x_coords = mask.x.values
    y_coords = mask.y.values
    res_x, res_y = mask.rio.resolution()


    # ---------------------------------------------------------
    # 3. Seasons and time mapping
    # ---------------------------------------------------------

    years = list(range(2018, 2025))
    seasons = ['winter', 'spring', 'summer', 'fall']

    # ---------------------------------------------------------
    # 5. Earthaccess configuration
    # ---------------------------------------------------------
    directions = ['gt1l','gt1r','gt2l','gt2r','gt3l','gt3r']
    out_dir = f"{projectDir}/data/initial/icesat-2/tmp/"
    os.makedirs(out_dir, exist_ok=True)

    max_retries = 3

    # ---------------------------------------------------------
    # 6. Process all year/season combinations
    # ---------------------------------------------------------
    slices = []

    for year in years:
        for season in seasons:

            print(f"\n=== {season.upper()} {year} ===")

            # Determine start/end search dates
            if season == 'winter':
                start_date = f"{year-1}-12-01"
                end_date = f"{year}-02-28"
            elif season == 'spring':
                start_date = f"{year}-03-01"
                end_date = f"{year}-05-31"
            elif season == 'summer':
                start_date = f"{year}-06-01"
                end_date = f"{year}-08-31"
            elif season == 'fall':
                start_date = f"{year}-09-01"
                end_date = f"{year}-11-30"

            # Search ICESat-2 ATL06 files
            results = earthaccess.search_data(
                short_name='ATL06',
                version = '007',
                polygon=AOI,
                temporal=(start_date, end_date)
            )

            if not results:
                print("⚠️ No ATL06 data found.")
                continue

            # Download with retries
            for attempt in range(max_retries):
                try:
                    files_download = earthaccess.download(results, out_dir, threads=8)
                    print("✅ Downloaded ICESat-2")
                    break
                except Exception as e:
                    print(f"⚠️ Download attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(30)
                    else:
                        print("❌ Skipping this season.")
                        files_download = []
                        break

            # Collect ATL06 points
            h, lat, lon = [], [], []
            files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".h5")]

            for fpath in files:
                try:
                    with h5py.File(fpath, "r") as file:
                        for direction in directions:
                            if direction not in file:
                                continue

                            g = file[direction]['land_ice_segments']
                            # The below two lines is the former code for quality flag
                            # q = g['atl06_quality_summary'][:]      # quality flag
                            # good = q == 0                          # keep only good data

                            q = g['atl06_quality_summary'][:]      # quality flag
                            sigma = g['h_li_sigma'][:]             # elevation uncertainty

                            # Keep only high-quality elevations with sigma < 0.5 m
                            good = (
                                (q == 0) &
                                np.isfinite(sigma) &
                                (sigma < 0.5)
                            )


                            h.extend(list(g['h_li'][good]))
                            lat.extend(list(g['latitude'][good]))
                            lon.extend(list(g['longitude'][good]))

                    os.remove(fpath)   # delete raw file
                except (OSError, IOError):
                    print(f"⚠️ Corrupted file detected, deleting and skipping: {os.path.basename(fpath)}")
                    os.remove(fpath)
                    continue

            if len(h) == 0:
                print("⚠️ No valid ICESat-2 points after filtering.")
                continue

            # Convert to arrays
            h = np.array(h)
            lat = np.array(lat)
            lon = np.array(lon)

            # Grid onto mask raster
            grid = bin_points_to_grid(
                lon, lat, h,
                mask_crs,
                x_coords, y_coords,
                res_x, res_y
            )

            # Build DataArray for this time slice
            timestamp = np.datetime64(season_timestamp(year, season))

            da_time = xr.DataArray(
                grid[np.newaxis, :, :],
                dims=("time", "y", "x"),
                coords={"time": [timestamp], "y": y_coords, "x": x_coords},
                name="h"
            )

            slices.append(da_time)
            print(f"✅ Created gridded slice for {season} {year}")
    

    # ---------------------------------------------------------
    # 7. Save final NetCDF
    # ---------------------------------------------------------
    out_nc = f"../data/initial/icesat-2/icesat-2_grid_region-{region_id}.nc"

    if slices:
        da_out = xr.concat(slices, dim="time", coords="minimal")
        da_out = da_out.sortby("time")

        # Apply mask if mask>0 are valid pixels
        da_out = da_out.where(mask > 0)

        da_out.to_netcdf(out_nc)

        return print(f"\n🎉 DONE — wrote gridded dataset to: {out_nc}")
    else:
        return print("⚠️ No slices were produced. Nothing to save.")

