import numpy as np
import xarray as xr
import rioxarray
import geopandas
import pandas as pd
import os
from scipy.stats import linregress
import matplotlib.pyplot as plt 
from scipy.ndimage import distance_transform_edt
from scipy.signal.windows import tukey
import matplotlib.colors as mcolors
import pickle
import string



#################################################
#####       Save prediction as netCDF       #####
#################################################

def make_blend_window(size, kind="distance", min_weight=0.1):

    if kind == "mean":

        window = np.ones((size, size), dtype="float32")

    elif kind == "hann":

        w1d = np.hanning(size)
        window = np.outer(w1d, w1d)

        # Normalize
        window = window / window.max()

        # Make sure edges are not exactly zero
        window = min_weight + (1 - min_weight) * window

    elif kind == "distance":

        # Pixel coordinates
        y, x = np.indices((size, size))

        # Distance of every pixel to the nearest tile edge
        distance = np.minimum.reduce([
            x,
            y,
            size - 1 - x,
            size - 1 - y,
        ]).astype("float32")

        # Normalize so centre has weight 1
        window = distance / distance.max()

        # Give the tile edge a small non-zero weight
        window = min_weight + (1 - min_weight) * window

    else:
        raise ValueError(
            "kind must be 'mean', 'hann', or 'distance'"
        )

    return window.astype("float32")



def zscore(df):
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    z = (df - mean) / std
    return z, std, mean

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


def predictions_to_netcdf(
    X_region,
    projectDir,
    region_id,
    experiment,
    type,
    clip_px=0,
    blend="distance",   # "hann", "mean" or "distance"
):

    mask_dir = f"{projectDir}/data/interim/region-{region_id}/masks/"
    mask = rioxarray.open_rasterio(os.path.join(mask_dir, "mask_100m.tif")).squeeze()

    tile_dir = f"{projectDir}/data/interim/region-{region_id}/masks/"
    tiles = geopandas.read_file(f"{tile_dir}tiles_{experiment}.gpkg")

    # time
    X_region["time"] = pd.to_datetime(X_region["time"] + "_15", format="%Y_%m_%d")
    X_region["tile_id"] = X_region["tile"].astype(str).str.extract(r"(\d+)").astype(int)

    H, W = mask.shape
    times = np.sort(X_region["time"].unique())
    nt = len(times)

    sum_arr = np.zeros((nt, H, W), dtype="float32")
    wgt_arr = np.zeros((nt, H, W), dtype="float32")


    T = X_region.pred.iloc[0].shape[0] 
    inner_T = T - 2 * clip_px   


    if blend == "hann":
        blend_window = make_blend_window(
            inner_T,
            kind="hann",
            min_weight=0.1
        )

    elif blend == "distance":
        blend_window = make_blend_window(
            inner_T,
            kind="distance",
            min_weight=0.1
        )

    elif blend == "mean":
        blend_window = np.ones(
            (inner_T, inner_T),
            dtype="float32"
        )

    else:
        raise ValueError(
            "blend must be 'hann', 'distance', or 'mean'"
        )


    # --------------------------------------------------
    # main loop
    # --------------------------------------------------
    for ti, t in enumerate(times):
        df = X_region[X_region["time"] == t]

        for _, row in df.iterrows():
            tile_id = row["tile_id"]
            pred = row["pred"]  

            # ---- clip edges ----
            pred = pred[
                clip_px : T - clip_px,
                clip_px : T - clip_px,
            ]  # now (inner_T, inner_T)
            row_tile = tiles[tiles["tile_id"] == tile_id]
            top = row_tile["top"].values[0]
            left = row_tile["left"].values[0]

            top = row_tile["top"].values[0] + clip_px
            left = row_tile["left"].values[0] + clip_px

            y0 = top
            y1 = top + inner_T
            x0 = left
            x1 = left + inner_T

            # bounds check
            gy0 = max(y0, 0)
            gx0 = max(x0, 0)
            gy1 = min(y1, H)
            gx1 = min(x1, W)

            py0 = gy0 - y0
            px0 = gx0 - x0
            py1 = py0 + (gy1 - gy0)
            px1 = px0 + (gx1 - gx0)

            pred_slice = pred[py0:py1, px0:px1]
            wgt_slice = blend_window[py0:py1, px0:px1]

            sum_arr[ti, gy0:gy1, gx0:gx1] += pred_slice * wgt_slice
            wgt_arr[ti, gy0:gy1, gx0:gx1] += wgt_slice

    # --------------------------------------------------
    # weighted mean
    # --------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_arr = sum_arr / wgt_arr

    mean_arr[wgt_arr == 0] = np.nan

    # --------------------------------------------------
    # mask
    # --------------------------------------------------

    ds = xr.Dataset(
        {"udem": (("time", "y", "x"), mean_arr)},
        coords={"time": times, "y": mask["y"], "x": mask["x"]},
    )

    ds = ds.rio.write_crs(mask.rio.crs)
    ds = ds.rio.write_transform(mask.rio.transform())

    # --------------------------------------------------
    # unnormalize 
    # --------------------------------------------------
    cs = xr.open_dataset(
        f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_gathered_region-{region_id}.nc"
    )["cs-elevation"].rio.write_crs(mask.rio.crs)

    mosaic = rioxarray.open_rasterio(
        f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif",
        masked=True,
    )

    mosaic = fill_na_nc(mosaic,50)
    cs = cs - mosaic

    cs_mean = cs.mean(skipna=True)
    cs_std = cs.std(skipna=True)

    ds["udem"] = ds["udem"] * cs_std + cs_mean + mosaic
    ds["udem"].attrs["_FillValue"] = np.nan

    out_dir = f"{projectDir}/data/final/"
    os.makedirs(out_dir, exist_ok=True)

    out_nc = f"{out_dir}udem_region-{region_id}_{experiment}_{type}.nc"
    ds.to_netcdf(out_nc)

    print("Saved:", out_nc)


#################################################
#####     Common arrays of two datasets     #####
#################################################

def common_array(ds1,ds2):
    common_times = np.intersect1d(ds1.time.values, ds2.time.values)

    udem_common = ds1.sel(time=common_times)
    adem_common = ds2.sel(time=common_times)

    # Boolean mask: True where BOTH datasets have data
    valid_mask = (~udem_common.isnull()) & (~adem_common.isnull())

    # Apply mask
    udem_masked = udem_common.where(valid_mask)
    adem_masked = adem_common.where(valid_mask)

    adem_arr = adem_masked.values.flatten()
    udem_arr = udem_masked.values.flatten()

    adem_array = adem_arr[~np.isnan(adem_arr)]
    udem_array = udem_arr[~np.isnan(adem_arr)]

    return udem_array, adem_array

#################################################
#####     Common arrays of three datasets     #####
#################################################

def common_array_three(ds1, ds2, ds3):
    # Find common times across all three
    common_times = np.intersect1d(
        np.intersect1d(ds1.time.values, ds2.time.values),
        ds3.time.values
    )

    ds1_common = ds1.sel(time=common_times)
    ds2_common = ds2.sel(time=common_times)
    ds3_common = ds3.sel(time=common_times)

    # Valid where ALL three have data
    valid_mask = (
        (~ds1_common.isnull()) &
        (~ds2_common.isnull()) &
        (~ds3_common.isnull())
    )

    # Apply mask (only to ds2 and ds3, since those are returned)
    ds1_masked = ds1_common.where(valid_mask)
    ds2_masked = ds2_common.where(valid_mask)
    ds3_masked = ds3_common.where(valid_mask)

    # Flatten and remove NaNs
    ds1_arr = ds1_masked.values.flatten()
    ds2_arr = ds2_masked.values.flatten()
    ds3_arr = ds3_masked.values.flatten()

    ds1_array = ds1_arr[~np.isnan(ds1_arr)]
    ds2_array = ds2_arr[~np.isnan(ds2_arr)]
    ds3_array = ds3_arr[~np.isnan(ds3_arr)]

    return ds1_array, ds2_array, ds3_array

# #################################################
# #####           Plotting ICESat-2           #####
# #################################################

def calculate_statistics(x,y):

    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # --- Statistics ---
    rmse = np.sqrt(np.mean((y - x)**2))
    bias = np.mean(y - x)
    r2 = np.corrcoef(x, y)[0, 1]**2
    return rmse, bias, r2

def plot_icesat2_validation_overall(udem_all,cs_all,is2_all,projectDir,experiment):
    udem_rmse, udem_bias, udem_r2 = calculate_statistics(is2_all, udem_all)
    cs_rmse, cs_bias, cs_r2 = calculate_statistics(is2_all, cs_all)

    # --- Figure with marginal histograms ---
    fig,ax = plt.subplots(1,2,figsize=(12, 6), constrained_layout=True)


    norm = mcolors.LogNorm(vmin=1, vmax=1e5)

    # --- Main hexbin density plot ---
    hb = ax[0].hexbin(is2_all, udem_all,gridsize=500,norm=norm,mincnt=1, cmap='hot')
    cax = ax[0].inset_axes([0.73, 0.02, 0.25, 0.04])  # adjust to taste
    cb = fig.colorbar(hb, cax=cax, orientation='horizontal',ticks=[10**0, 10**2, 10**4])
    cb.set_label("log10(count)", fontsize=7)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False,labelsize=7)

    # 1:1 line
    lims = [0,2500]
    ax[0].plot(lims, lims, 'k--', lw=1, label="1:1 line")


    # --- Statistics annotation (shifted DOWN a bit) ---
    stats_text = (
        f"RMSE = {udem_rmse:.2f} m\n"
        f"Bias = {udem_bias:.2f} m\n"
        f"R²   = {udem_r2:.5f}"
    )


    ax[0].text(
        0.05, 0.9,          # moved from 0.95 → 0.70
        stats_text,
        transform=ax[0].transAxes,
        va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax[0].text(
        0.05, 0.95,          # moved from 0.95 → 0.70
        "(a) U-DEM",
        transform=ax[0].transAxes,
        va='top', ha='left',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )


    # Labels
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    ax[0].set_xlabel(f"ICESat-2 elevations [m]")
    ax[0].set_ylabel(f"U-DEM elevations [m]")



    # --- Main hexbin density plot ---
    hb = ax[1].hexbin(is2_all, cs_all,gridsize=500,norm=norm,mincnt=1, cmap='hot')
    cax = ax[1].inset_axes([0.73, 0.02, 0.25, 0.04])  # adjust to taste [x0, y0, width, height]
    cb = fig.colorbar(hb, cax=cax, orientation='horizontal',ticks=[10**0, 10**2, 10**4])
    cb.set_label("log10(count)", fontsize=7)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False,labelsize=7)

    #cax.tick_params(labelsize=7)

    # 1:1 line
    lims = [0,2500]
    ax[1].plot(lims, lims, 'k--', lw=1, label="1:1 line")


    # --- Statistics annotation (shifted DOWN a bit) ---
    stats_text = (
        f"RMSE = {cs_rmse:.2f} m\n"
        f"Bias = {cs_bias:.2f} m\n"
        f"R²   = {cs_r2:.5f}"
    )

    ax[1].text(
        0.05, 0.9,          # moved from 0.95 → 0.70
        stats_text,
        transform=ax[1].transAxes,
        va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax[1].text(
        0.05, 0.95,          # moved from 0.95 → 0.70
        "(b) CryoSat-2",
        transform=ax[1].transAxes,
        va='top', ha='left',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Labels
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)
    ax[1].set_xlabel(f"ICESat-2 elevations [m]")
    ax[1].set_ylabel(f"CryoSat-2 elevations [m]")



    fig.savefig(f"{projectDir}/figures/{experiment}_validation_ICESat-2.png",dpi=300)


def plot_seasonal_icesat2_validation(region_ids_full,projectDir,experiment):
    # ------------------------------------------------------------
    # Seasonal definitions
    # ------------------------------------------------------------
    seasons = {
        "Winter": 1,    # YYYY-01-15
        "Spring": 4,    # YYYY-04-15
        "Summer": 7,    # YYYY-07-15
        "Fall": 10,     # YYYY-10-15
    }


    # ------------------------------------------------------------
    # Store data from all regions
    #
    # Each product is matched independently with ICESat-2.
    # This avoids discarding valid U-DEM points simply because
    # CryoSat-2 is missing, and vice versa.
    # ------------------------------------------------------------
    seasonal_data = {
        season: {
            "udem": [],
            "cs": [],
            "is2": [],
        }
        for season in seasons
    }


    # ------------------------------------------------------------
    # Load and collect the data
    # ------------------------------------------------------------
    for region_id in region_ids_full:

        print(f"Processing region {region_id}")

        udem = xr.load_dataset(
            f"{projectDir}/data/final/"
            f"udem_region-{region_id}_{experiment}_all.nc"
        )["udem"]

        is2 = xr.load_dataset(
            f"{projectDir}/data/initial/icesat-2/"
            f"icesat-2_grid_region-{region_id}.nc"
        )["h"]

        cs = xr.load_dataset(
            f"{projectDir}/data/interim/region-{region_id}/"
            f"cryoswath_seasonal/"
            f"cs_raw_gathered_region-{region_id}.nc"
        )["cs-elevation"]

        arcticdem_path = (
            f"{projectDir}/data/initial/ArcticDEM/"
            f"region-{region_id}/"
            f"arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif"
        )

        mosaic = rioxarray.open_rasterio(
            arcticdem_path,
            masked=True
        ).squeeze(drop=True)

        # Remove ICESat-2 observations differing by more than 100 m
        # from the ArcticDEM mosaic.
        is2_diff = is2 - mosaic
        is2_filtered = is2.where(np.abs(is2_diff) <= 100)

        for season, month in seasons.items():

            # Select all years belonging to the relevant season
            udem_season = udem.where(
                udem.time.dt.month == month,
                drop=True
            )

            is2_season = is2_filtered.where(
                is2_filtered.time.dt.month == month,
                drop=True
            )

            cs_season = cs.where(
                cs.time.dt.month == month,
                drop=True
            )

            # Only retain observations that are valid in all three datasets
            udem_arr, is2_arr, cs_arr = common_array_three(
                udem_season,
                is2_season,
                cs_season
            )

            if len(udem_arr) > 0:

                seasonal_data[season]["udem"].append(
                    np.asarray(udem_arr).ravel()
                )

                seasonal_data[season]["cs"].append(
                    np.asarray(cs_arr).ravel()
                )

                # Both products now use exactly the same ICESat-2 observations
                seasonal_data[season]["is2"].append(
                    np.asarray(is2_arr).ravel()
                )




    # ------------------------------------------------------------
    # Concatenate arrays from all regions
    # ------------------------------------------------------------
    for season in seasons:

        for key in seasonal_data[season]:

            arrays = seasonal_data[season][key]

            if len(arrays) > 0:
                seasonal_data[season][key] = np.concatenate(arrays)
            else:
                seasonal_data[season][key] = np.array([])


    # ------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        2,
        4,
        figsize=(12, 6),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.91,
        bottom=0.10,
        top=0.92,
        wspace=0.10,
        hspace=0.13
    )

    lims = [0, 2500]

    # Shared normalization across all panels.
    norm = mcolors.LogNorm(
        vmin=1,
        vmax=1e5
    )




    # ------------------------------------------------------------
    # Plot each season
    # ------------------------------------------------------------
    for col, season in enumerate(seasons):

        # ========================================================
        # First row: U-DEM versus ICESat-2
        # ========================================================
        is2_values = seasonal_data[season]["is2"]
        product_values = seasonal_data[season]["udem"]

        udem_rmse, udem_bias, udem_r2 = calculate_statistics(
            is2_values,
            product_values
        )

        hb = ax[0, col].hexbin(
            is2_values,
            product_values,
            gridsize=300,
            norm=norm,
            mincnt=1,
            cmap="hot"
        )

        ax[0, col].plot(
            lims,
            lims,
            "k--",
            linewidth=1
        )

        panel_letter = string.ascii_lowercase[col]

        if col == 0:

            cax = ax[0, 0].inset_axes([0.08, 0.78, 0.28, 0.05])

            cb = fig.colorbar(
                hb,
                cax=cax,
                orientation="horizontal",
                ticks=[1, 1e2, 1e4]
            )

            cb.set_label("log$_{10}$(count)", fontsize=8)

            cb.ax.xaxis.set_label_position("bottom")
            cb.ax.tick_params(
                top=False,
                bottom=True,
                labeltop=False,
                labelbottom=True,
                labelsize=8
            )

        ax[0, col].text(
            0.04,
            0.96,
            f"({panel_letter})",
            transform=ax[0, col].transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=2
            )
        )

        stats_text = (
            f"RMSE = {udem_rmse:.2f} m\n"
            f"Bias = {udem_bias:.2f} m\n"
            f"R² = {udem_r2:.5f}"
        )

        ax[0, col].text(
            0.96,
            0.04,
            stats_text,
            transform=ax[0, col].transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=2
            )
        )

        ax[0, col].set_title(
            season,
            fontweight="bold"
        )

        # ========================================================
        # Second row: CryoSat-2 versus ICESat-2
        # ========================================================
        is2_values = seasonal_data[season]["is2"]
        product_values = seasonal_data[season]["cs"]

        cs_rmse, cs_bias, cs_r2 = calculate_statistics(
            is2_values,
            product_values
        )

        ax[1, col].hexbin(
            is2_values,
            product_values,
            gridsize=300,
            norm=norm,
            mincnt=1,
            cmap="hot"
        )

        ax[1, col].plot(
            lims,
            lims,
            "k--",
            linewidth=1
        )

        panel_letter = string.ascii_lowercase[4 + col]

        ax[1, col].text(
            0.04,
            0.96,
            f"({panel_letter})",
            transform=ax[1, col].transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=2
            )
        )

        stats_text = (
            f"RMSE = {cs_rmse:.2f} m\n"
            f"Bias = {cs_bias:.2f} m\n"
            f"R² = {cs_r2:.5f}"
        )

        ax[1, col].text(
            0.96,
            0.04,
            stats_text,
            transform=ax[1, col].transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=2
            )
        )


    # ------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------
    for row in range(2):
        for col in range(4):

            ax[row, col].set_xlim(lims)
            ax[row, col].set_ylim(lims)

            ax[row, col].set_aspect("equal")

            ax[row, col].tick_params(
                direction="out",
                labelsize=8
            )


    # Row-specific y-axis labels
    ax[0, 0].set_ylabel("U-DEM elevation [m]")
    ax[1, 0].set_ylabel("CryoSat-2 elevation [m]")

    # X-axis labels on the lower row only
    for col in range(4):
        ax[1, col].set_xlabel("ICESat-2 elevation [m]")



def plot_true_vs_prediction_poster_gathered(fig, x, y, region_id, ax, letter):

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    rmse = np.sqrt(np.mean((y - x)**2))
    bias = np.mean(y - x)
    r2 = np.corrcoef(x, y)[0, 1]**2

    norm = mcolors.LogNorm(vmin=1, vmax=1e4)

    hb = ax.hexbin(x, y,gridsize=500,norm=norm,mincnt=1, cmap='hot')

    ax.set_facecolor('none')

    # colorbar inside
    if region_id == "01":
        cax = ax.inset_axes([0.1, 0.8, 0.25, 0.04])  # adjust to taste [x0, y0, width, height]
        cb = fig.colorbar(hb, cax=cax, orientation='horizontal',ticks=[10**0, 10**2, 10**4])
        cb.set_label("log10(count)", fontsize=8)
        cb.ax.xaxis.set_label_position('bottom')
        cb.ax.tick_params(top=False,bottom=True,labeltop=False,labelbottom=True,labelsize=8)


    lims = [0,2500]
    ax.plot(lims, lims, '--', lw=0.4, color="#1D2731")

    stats_text = (
        f"RMSE={rmse:.2f}m\n"
        f"Bias={bias:.2f}m\n"
        f"R²={r2:.4f}"
    )


    ax.text(
        0.53, 0.05,
        stats_text,
        transform=ax.transAxes,
        fontsize=10
    )

    ax.text(
        0.08, 0.9,
        letter,
        fontweight='bold',
        transform=ax.transAxes,
        fontsize=10
    )



    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"Region {region_id}")
    ax.set_aspect("equal")


def plot_regional_icesat2_validation(product,region_ids_full,projectDir,experiment):

    fig, axs = plt.subplots(
        4, 4,
        figsize=(12,12),
        constrained_layout=True
    )


    axs = axs.flatten()



    letters = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)","(j)","(k)","(l)","(m)","(n)","(o)",]


    # --------------------------------------------------
    # Region validation plots
    # --------------------------------------------------

    for i, region_id in enumerate(region_ids_full):

        ax = axs[i]

        udem = xr.load_dataset(
            f"{projectDir}/data/final/udem_region-{region_id}_{experiment}_all.nc"
        )["udem"]

        is2 = xr.load_dataset(
            f"{projectDir}/data/initial/icesat-2/icesat-2_grid_region-{region_id}.nc"
        )["h"]

        cs = xr.load_dataset(
            f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/cs_raw_gathered_region-{region_id}.nc"
        )["cs-elevation"]

        arcticdem_path = f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif"

        mosaic = rioxarray.open_rasterio(
            arcticdem_path,
            masked=True
        ).squeeze()

        is2_diff = is2 - mosaic
        is2_filtered = is2.where(np.abs(is2_diff) <= 100)

        udem_arr, is2_arr, cs_arr = common_array_three(
            udem,
            is2_filtered,
            cs
        )

        if product == "U-DEM":
            plot_true_vs_prediction_poster_gathered(
                fig,
                is2_arr,
                udem_arr,
                region_id,
                ax,
                letters[i]
            )
        elif product == "CryoSat-2":
            plot_true_vs_prediction_poster_gathered(
                fig,
                is2_arr,
                cs_arr,
                region_id,
                ax,
                letters[i]
            )
        else:
            raise ValueError(
                f"Invalid product '{product}'. Valid inputs are 'U-DEM' and 'CryoSat-2'."
            )


    # Remove inner labels
    for ax in axs:
        ax.label_outer()


    #axs[1].tick_params(labelleft=True)

    axs[0].set_ylabel(f'{product} elevation [m]')
    axs[4].set_ylabel(f'{product} elevation [m]')
    axs[8].set_ylabel(f'{product} elevation [m]')
    axs[12].set_ylabel(f'{product} elevation [m]')

    axs[12].set_xlabel('ICESat-2 elevation [m]')
    axs[13].set_xlabel('ICESat-2 elevation [m]')
    axs[14].set_xlabel('ICESat-2 elevation [m]')
    axs[11].set_xlabel('ICESat-2 elevation [m]')

    axs[-1].set_axis_off()



    fig.set_constrained_layout_pads(wspace=0.02, hspace=0.02)

    axs[11].tick_params(labelbottom=True)
    axs[11].xaxis.set_visible(True)

    fig.canvas.draw()
    fig.savefig(f"{projectDir}/figures/{experiment}_validation_ICESat-2_{product}_allregions.png",dpi=300,bbox_inches="tight")


def plot_slope_elevation_validation(region_ids_full,projectDir,experiment,project_crs):

    def calculate_slope(dem):
        dzdy, dzdx = np.gradient(dem, dy, dx)
        slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
        return slope


    def collect_product_arrays(da_masked, is2_filt):
        common_times = np.intersect1d(
            is2_filt.time.values,
            da_masked.time.values
        )

        is2_common = is2_filt.sel(time=common_times)
        da_common = da_masked.sel(time=common_times)

        dh = da_common - is2_common

        dx = float(da_common.x.diff("x").mean())
        dy = float(da_common.y.diff("y").mean())

        def calculate_slope_local(dem):
            dzdy, dzdx = np.gradient(dem, dy, dx)
            return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

        slope_da = xr.apply_ufunc(
            calculate_slope_local,
            da_common,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        x = dh.values.ravel()
        elev = da_common.values.ravel()
        slope = slope_da.values.ravel()

        valid = np.isfinite(x) & np.isfinite(elev) & np.isfinite(slope)

        return x[valid], elev[valid], slope[valid]


    udem_dh_all = []
    udem_elev_all = []
    udem_slope_all = []

    cs_dh_all = []
    cs_elev_all = []
    cs_slope_all = []

    for region_id in region_ids_full:

        print(f"Processing region {region_id}")

        # Load ICESat-2 and mosaic
        is2 = xr.load_dataset(
            f"{projectDir}/data/initial/icesat-2/icesat-2_grid_region-{region_id}.nc"
        )["h"]

        arcticdem_path = (
            f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/"
            f"arcticdem_mosaic_100m_v4.1_dem_region-{region_id}.tif"
        )

        mosaic = rioxarray.open_rasterio(arcticdem_path, masked=True).squeeze()

        is2 = is2.rio.write_crs(project_crs).rio.reproject_match(mosaic)
        is2_diff = is2 - mosaic
        is2_filt = is2.where(np.abs(is2_diff) <= 100)

        # Load mask
        mask_dir = f"{projectDir}/data/interim/region-{region_id}/masks/"
        mask = rioxarray.open_rasterio(
            os.path.join(mask_dir, "mask_100m.tif"),
            masked=True
        ).squeeze()

        # Load U-DEM
        udem = xr.load_dataset(
            f"{projectDir}/data/final/udem_region-{region_id}_{experiment}_all.nc"
        )["udem"]

        udem_masked = udem.where(mask == 1)
        udem_masked = udem_masked.rio.write_crs(project_crs)

        # Load CryoSat-2
        cs = xr.load_dataset(
            f"{projectDir}/data/interim/region-{region_id}/cryoswath_seasonal/"
            f"cs_raw_gathered_region-{region_id}.nc"
        )["cs-elevation"]

        cs_masked = cs.where(mask == 1)
        cs_masked = cs_masked.rio.write_crs(project_crs)

        # Collect U-DEM arrays
        dh, elev, slope = collect_product_arrays(udem_masked, is2_filt)
        udem_dh_all.append(dh)
        udem_elev_all.append(elev)
        udem_slope_all.append(slope)

        # Collect CryoSat-2 arrays
        dh, elev, slope = collect_product_arrays(cs_masked, is2_filt)
        cs_dh_all.append(dh)
        cs_elev_all.append(elev)
        cs_slope_all.append(slope)


    # Combine all regions
    udem_dh_all = np.concatenate(udem_dh_all)
    udem_elev_all = np.concatenate(udem_elev_all)
    udem_slope_all = np.concatenate(udem_slope_all)

    cs_dh_all = np.concatenate(cs_dh_all)
    cs_elev_all = np.concatenate(cs_elev_all)
    cs_slope_all = np.concatenate(cs_slope_all)

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    shared_norm = mcolors.LogNorm(vmin=1, vmax=1e5)

    hb0 = ax[0, 0].hexbin(
        udem_dh_all,
        udem_elev_all,
        gridsize=500,
        norm=shared_norm,
        mincnt=1,
        cmap="hot"
    )

    hb1 = ax[0, 1].hexbin(
        cs_dh_all,
        cs_elev_all,
        gridsize=500,
        norm=shared_norm,
        mincnt=1,
        cmap="hot"
    )

    hb2 = ax[1, 0].hexbin(
        udem_dh_all,
        udem_slope_all,
        gridsize=500,
        norm=shared_norm,
        mincnt=1,
        cmap="hot"
    )

    hb3 = ax[1, 1].hexbin(
        cs_dh_all,
        cs_slope_all,
        gridsize=500,
        norm=shared_norm,
        mincnt=1,
        cmap="hot"
    )

    ax[0, 0].set_title("U-DEM")
    ax[0, 1].set_title("CryoSat-2")

    ax[0, 0].set_ylabel("Elevation [m]")
    ax[1, 0].set_ylabel("Slope [$\degree$]")

    ax[1, 0].set_xlabel("Residual [m]")
    ax[1, 1].set_xlabel("Residual [m]")

    for a in ax.ravel():
        a.set_xlim(-300, 300)

    ax[1,0].set_ylim(0,70)
    ax[1,1].set_ylim(0,70)
    ax[0,0].set_ylim(0,2500)
    ax[0,1].set_ylim(0,2500)


    ax[0,0].text(0.06,0.9,"(a)",fontweight='bold',transform=ax[0,0].transAxes,fontsize=10,bbox=dict(facecolor="white",alpha=0.8,edgecolor="none",boxstyle="square,pad=0.15"))
    ax[0,1].text(0.06,0.9,"(b)",fontweight='bold',transform=ax[0,1].transAxes,fontsize=10,bbox=dict(facecolor="white",alpha=0.8,edgecolor="none",boxstyle="square,pad=0.15"))
    ax[1,0].text(0.06,0.9,"(c)",fontweight='bold',transform=ax[1,0].transAxes,fontsize=10,bbox=dict(facecolor="white",alpha=0.8,edgecolor="none",boxstyle="square,pad=0.15"))
    ax[1,1].text(0.06,0.9,"(d)",fontweight='bold',transform=ax[1,1].transAxes,fontsize=10,bbox=dict(facecolor="white",alpha=0.8,edgecolor="none",boxstyle="square,pad=0.15"))



    # Leave some room on the right for the colorbar
    fig.subplots_adjust(right=0.88)

    # [left, bottom, width, height] in figure coordinates
    cax = fig.add_axes([0.90, 0.11, 0.02, 0.77])

    cb = fig.colorbar(hb3, cax=cax)
    cb.set_label(r'$\log_{10}(\mathrm{count})$')

    #plt.tight_layout()

    fig.savefig(f"{projectDir}/figures/{experiment}_validation_ICESat-2_slope_elevation_allregions.png",dpi=300,bbox_inches="tight")

