
import argparse
import os
import subprocess
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

parser = argparse.ArgumentParser(description="Download and process ArcticDEM strips.")
parser.add_argument("--region_id", type=str, required=True,
                    help="Region ID, e.g. '05'")
parser.add_argument("--project_crs", type=str, default="EPSG:3413",
                    help="Target CRS, e.g. 'EPSG:3413'")
parser.add_argument("--n_threads", type=int, default=8,
                    help="Number of parallel threads")
parser.add_argument("--projectDir", type=str, required=True,
                    help="Base project directory")

args = parser.parse_args()

# Assign to variables used below
region_id = args.region_id
project_crs = args.project_crs
n_threads = args.n_threads
projectDir = args.projectDir

#region_id = "05"
#project_crs = "EPSG:3413"
#n_threads = 8  # number of parallel downloads
downloader = "aria2c"  # or "wget"

#projectDir = "/Users/rfk471/Dropbox/elevation-canada"
baseDir = f"{projectDir}/data/initial/ArcticDEM/region-{region_id}/"
demDir = os.path.join(baseDir, "ArcticDEM_50m_strips")
bitDir = os.path.join(baseDir, "ArcticDEM_50m_strips_bitmask")
metaDir = os.path.join(baseDir, "metadataFiles")
tmpDir = os.path.join(baseDir, "tmp")

for d in [demDir, bitDir, metaDir, tmpDir]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def process_file(file_url):
    """Download, extract, resample, and clean up one ArcticDEM strip."""
    file_name = os.path.basename(file_url)
    tar_path = os.path.join(tmpDir, file_name)
    base_name = file_name.replace(".tar.gz", "").replace(".tar", "")

    demFile = os.path.join(demDir, f"{base_name}_dem_50m.tif")
    bitFile = os.path.join(bitDir, f"{base_name}_bitmask_50m.tif")

    # Skip if both files exist
    if os.path.exists(demFile) and os.path.exists(bitFile):
        return f"✅ Skipped {file_name} (already processed)"

    # -----------------------------
    # Download
    # -----------------------------
    try:
        if downloader == "aria2c":
            cmd = ["aria2c", "-x", "8", "-s", "8", "-d", tmpDir, "-o", file_name, file_url]
        else:
            cmd = ["wget", "-q", "-O", tar_path, file_url]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return f"❌ Download failed: {file_name}"

    # -----------------------------
    # Extract
    # -----------------------------
    try:
        subprocess.run(["tar", "-xf", tar_path, "-C", tmpDir], check=True)
    except subprocess.CalledProcessError:
        return f"❌ Extraction failed: {file_name}"

    # -----------------------------
    # Move metadata and resample
    # -----------------------------
    try:
        # Move metadata
        meta_src = os.path.join(tmpDir, f"{base_name}_mdf.txt")
        if os.path.exists(meta_src):
            meta_dst = os.path.join(metaDir, f"{base_name}_mdf.txt")
            os.replace(meta_src, meta_dst)

        # Resample DEM and bitmask
        dem_src = os.path.join(tmpDir, f"{base_name}_dem.tif")
        bit_src = os.path.join(tmpDir, f"{base_name}_bitmask.tif")

        if os.path.exists(dem_src):
            subprocess.run([
                "gdalwarp", "-tr", "50", "50", "-r", "near",
                dem_src, demFile
            ], check=True)

        if os.path.exists(bit_src):
            subprocess.run([
                "gdalwarp", "-tr", "50", "50", "-r", "near",
                bit_src, bitFile
            ], check=True)
    except subprocess.CalledProcessError:
        return f"❌ GDAL processing failed: {file_name}"

    # -----------------------------
    # Cleanup
    # -----------------------------
    for f in os.listdir(tmpDir):
        if f.startswith(base_name) or f == "index":
            path = os.path.join(tmpDir, f)
            if os.path.isdir(path):
                subprocess.run(["rm", "-rf", path])
            else:
                os.remove(path)
    if os.path.exists(tar_path):
        os.remove(tar_path)

    return f"✅ Done: {file_name}"

# --------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------
if __name__ == "__main__":
    print("🔹 Loading RGI and ArcticDEM index...")

    rgi = gpd.read_file(
        f"{projectDir}/data/initial/RGI/RGI2000-v7.0-C-03_arctic_canada_north/RGI2000-v7.0-C-03_arctic_canada_north.shp"
    ).to_crs(project_crs)

    region = gpd.read_file(f"{projectDir}/data/initial/regions/region-{region_id}.shp")
    rgi_region = gpd.clip(rgi, region)

    strips = gpd.read_file(
        f"{projectDir}/data/initial/ArcticDEM/ArcticDEM_Strip_Index_s2s041.shp"
    ).to_crs(project_crs)
    strips_region = gpd.clip(strips, rgi_region)
    strips_region = strips_region[
        (strips_region.sensor1 != "GEO1") & (strips_region.sensor2 != "GEO1")
    ]
    urls = strips_region.fileurl.tolist()

    print(f"🚀 Starting parallel processing with {n_threads} threads...")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_file, url): url for url in urls}
        for future in as_completed(futures):
            print(future.result())

    print("🏁 All downloads completed.")

