import json
import os
from pathlib import Path
import random
from PIL import Image
import numpy as np
import cv2
import argparse
                        
US_LAT_MIN, US_LAT_MAX = 24.0, 49.0
US_LON_MIN, US_LON_MAX = -125.0, -66.0 


records = []

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Raw data folder path", type=str, default="images")
    parser.add_argument("--output", help="Output folder path", type=str, default="images" )
    return parser.parse_args()

def parse_month(date_str):
    """Convert '2019-07' -> 7"""
    try:
        return int(date_str.split("-")[1])
    except:
        return None
    
def month_to_season(m):
    """
    0 = Winter (12,1,2)
    1 = Spring (3,4,5)
    2 = Summer (6,7,8)
    3 = Fall   (9,10,11)
    """
    if m in (12, 1, 2): return 0
    if m in (3, 4, 5): return 1
    if m in (6, 7, 8): return 2
    return 3

def compute_haze_index(arr):
    """Lower = hazier. Using Laplacian variance."""
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    haze = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(haze)

def load_and_clean(data_path):
    meta_path = data_path + "/meta.jsonl"
    with open(meta_path, "r") as f:
        for line in f:
            rec = json.loads(line)

            # skip missing date
            if not rec.get("date"):
                continue
            
            # skip missing lat/lon
            if "lat" not in rec or "lon" not in rec:
                continue
            
            # skip image that doesn't exist
            img_path = Path(data_path + "/" + rec["img"])
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            # brightness
            brightness = float(arr.mean())
            # green ratio
            r = arr[...,0].astype(np.float32)
            g = arr[...,1].astype(np.float32)
            green_ratio = float(((g - r) / (g + r + 1e-6)).mean())
            # haze
            # haze = compute_haze_index(arr)

            # sky ratio ( including white and blue sky)
            sky_ratio = compute_sky_ratio(arr)
            rec["brightness"] = brightness
            rec["green_ratio"] = green_ratio
            # rec["haze"] = haze
            rec["sky_ratio"] = sky_ratio
           

            records.append(rec)
    print(f"Loaded {len(records)} valid samples") 

def create_id_for_region():
    cities = sorted(set(r["city"] for r in records))
    states = sorted(set(r["state"] for r in records))

    city2id = {c: i for i, c in enumerate(cities)}
    state2id = {s: i for i, s in enumerate(states)}

    for r in records:
        r["city_id"] = city2id[r["city"]]
        r["state_id"] = state2id[r["state"]]


def compute_sky_ratio(arr):
    """
    Robust sky detector using HSV + Blue-Channel dominance + edge masking.
    Works for bright blue sky, pale sky, cloudy sky, partial sky.
    """

    # Convert to HSV
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]


    mask_bright = v > 130

    mask_blue_hsv = ((h > 85) & (h < 140))           
    b = arr[...,2]
    g = arr[...,1]
    r = arr[...,0]
    mask_blue_rgb = (b > g + 10) & (b > r + 10)      # blue clearly dominant

    # Combine blue-related rules
    mask_blue = mask_blue_hsv | mask_blue_rgb


    edges = cv2.Canny(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 80, 160)
    edges = cv2.dilate(edges, None)  # enlarge edge areas
    mask_no_edges = edges < 10       

    # Final sky mask
    sky_mask = mask_bright & mask_blue & mask_no_edges

    ratio = sky_mask.mean()
    return float(ratio)

def add_month_season():
    global records
    cleaned = []
    for r in records:
        m = parse_month(r["date"])
        if m is None:
            continue

        r["month"] = m
        r["season"] = month_to_season(m)
        cleaned.append(r)

    records = cleaned
    print(f"After month/season filtering: {len(records)} samples")

def latlon_to_grid(lat, lon, grid_size=20):\
    
    grid_x = int((lat - US_LAT_MIN) / (US_LAT_MAX - US_LAT_MIN) * grid_size)
    grid_y = int((lon - US_LON_MIN) / (US_LON_MAX - US_LON_MIN) * grid_size)

    grid_x = max(0, min(grid_size - 1, grid_x))
    grid_y = max(0, min(grid_size - 1, grid_y))
    
    grid_id = grid_x * grid_size + grid_y
    return grid_x, grid_y, grid_id

def normalize_elevation():
    """Min-max normalize elevation to [0, 1] and store as 'elev_norm'."""
    # Change the key name here if your field name is different
    key = "elevation"

    vals = [r[key] for r in records if key in r]
    if not vals:
        print("No elevation field found, skip elevation normalization.")
        return

    min_e = min(vals)
    max_e = max(vals)
    if max_e == min_e:
        print("All elevation values are the same, skip normalization.")
        for r in records:
            if key in r:
                r["elev_norm"] = 0.5     # or 0.0, doesn't matter much
        return

    for r in records:
        if key in r:
            e = r[key]
            r["elev_norm"] = (e - min_e) / (max_e - min_e)

    print(f"Elevation normalized: min={min_e:.2f}, max={max_e:.2f}")

def reformat_lat_lon(GRID_SIZE =20):
    CELL_LAT_SIZE = (US_LAT_MAX - US_LAT_MIN) / GRID_SIZE  
    CELL_LON_SIZE = (US_LON_MAX - US_LON_MIN) / GRID_SIZE 
    
    for r in records:
        lat = r["lat"]
        lon = r["lon"]
        grid_x, grid_y, grid_id = latlon_to_grid(lat, lon)

        cell_lat_min = US_LAT_MIN + grid_x * CELL_LAT_SIZE
        cell_lon_min = US_LON_MIN + grid_y * CELL_LON_SIZE
        offset_x = (lat - cell_lat_min) / CELL_LAT_SIZE
        offset_y = (lon - cell_lon_min) / CELL_LON_SIZE

        r["offset_x"] = offset_x
        r["offset_y"] = offset_y
        r["grid_x"] = grid_x
        r["grid_y"] = grid_y
        r["grid_id"] = grid_id

def split_train_val(output_path):
    random.shuffle(records)
    n = len(records)
    train_records = records[: int(n * 0.8)]
    val_records = records[int(n * 0.8):]

    with open(os.path.join(output_path, "train_meta.jsonl"), "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(output_path, "val_meta.jsonl"), "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")



# Main execution block to run all steps in order
if __name__ == "__main__":
    args = get_args()
    load_and_clean(args.data)
    create_id_for_region()
    add_month_season()
    reformat_lat_lon()
    normalize_elevation()
    split_train_val(args.output)