import os
import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# ARGUMENT PARSING
# -------------------------
parser = argparse.ArgumentParser(description="DEM visual quality reviewer")
parser.add_argument("--region_id", required=True, help="Region ID (e.g., 01)")
parser.add_argument("--label_tagger", required=True, help="Name of labeler")

args = parser.parse_args()

region_id = args.region_id
label_tagger = args.label_tagger

# -------------------------
# SETTINGS
# -------------------------
folder = f"/Users/rfk471/Dropbox/elevation-canada/data/interim/region-{region_id}/ArcticDEM/co-registered-strips"
output_csv = f"/Users/rfk471/Dropbox/elevation-canada/data/interim/region-{region_id}/ArcticDEM/labels_{label_tagger}_region-{region_id}.csv"
cmap = "viridis"

# -------------------------
# FILE LIST
# -------------------------
all_files = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])

if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
else:
    df = pd.DataFrame(columns=["file", "label"])

reviewed = list(df["file"])
files = all_files

current = len(reviewed)

if current >= len(files):
    print("Nothing left to review.")
    exit()

print(f"Resuming at file {current+1} / {len(files)}")

# -------------------------
# PLOTTING SETUP
# -------------------------
fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
#plt.tight_layout()

cbar = None
im = None

def review_file(i):
    global cbar, im

    path = os.path.join(folder, files[i])
    
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(1, 600, 600),
            resampling=rasterio.enums.Resampling.average
        )

    vmin, vmax = np.nanpercentile(data, (2, 98))

    if im is None:
        # First time: create image + colorbar
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    else:
        # Update existing image + colorbar
        im.set_data(data)
        im.set_clim(vmin, vmax)
        cbar.update_normal(im)

    ax.set_title(
        f"{files[i]}\n"
        f"[1]=Good  [2]=MedGood  [3]=MedBad  [4]=Bad\n"
        f"[z]=Back one    [esc]=Exit\n"
        f"File {i+1} / {len(files)}",
        fontsize=10
    )
    ax.axis("off")

    plt.draw()

review_file(current)

# -------------------------
# KEYBOARD CONTROL
# -------------------------
def on_key(event):
    global current, df

    if event.key in ["1", "2", "3", "4"]:
        label = int(event.key)

        if current < len(files):
            df.loc[len(df)] = [files[current], label]
            df.to_csv(output_csv, index=False)

            current += 1

            if current < len(files):
                review_file(current)
            else:
                print("Review complete.")
                plt.close()

    elif event.key == "z":
        if current > 0:
            current -= 1

            if len(df) > 0:
                df = df.iloc[:-1]
                df.to_csv(output_csv, index=False)

            review_file(current)
        else:
            print("Already at first file.")

    elif event.key == "escape":
        df.to_csv(output_csv, index=False)
        print("Exiting early.")
        plt.close()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()