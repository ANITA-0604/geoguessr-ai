import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import math

# Import your code
from model import GeoGuessorModel
from geoguessr_dataset import GeoGuessrDataset
import data_preprocess # To get the MIN/MAX constants

# --- CONSTANTS FROM THE PREPROCESS SCRIPT ---
# Ensure these match exactly what was in data_preprocess.py
US_LAT_MIN, US_LAT_MAX = 24.0, 49.0
US_LON_MIN, US_LON_MAX = -125.0, -66.0
GRID_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grid_to_latlon(gx, gy, off_x, off_y):
    """
    Reverses the normalization to get back real Lat/Lon
    """
    # 1. Size of one cell
    lat_step = (US_LAT_MAX - US_LAT_MIN) / GRID_SIZE
    lon_step = (US_LON_MAX - US_LON_MIN) / GRID_SIZE

    # 2. Base coordinate of the grid cell
    lat_base = US_LAT_MIN + (gx * lat_step)
    lon_base = US_LON_MIN + (gy * lon_step)

    # 3. Add the offset
    # predicted offset is -1 to 1 (Tanh), so we map it back to 0 to 1
    # If the model outputs Tanh, off_x is -1..1. 
    # But in data_preprocess, offset was 0..1 relative to cell.
    # We need to be careful here. 
    # Let's assume the model learned to map Tanh(-1..1) to the relative position.
    # To map [-1, 1] -> [0, 1]: (x + 1) / 2

    off_x_norm = (off_x + 1) / 2
    off_y_norm = (off_y + 1) / 2

    pred_lat = lat_base + (off_x_norm * lat_step)
    pred_lon = lon_base + (off_y_norm * lon_step)

    return pred_lat, pred_lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance in KM between two lat/lon points
    """
    R = 6371 # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def evaluate(model, loader):
    model.eval()
    distances = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc = "Evaluating Distances"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get predictions
            outputs = model(images, use_teacher_forcing = False)

            # Get predicted Grids and Offsets
            pred_gx = torch.argmax(outputs['gx_logits'], dim=1).cpu().numpy()
            pred_gy = torch.argmax(outputs['gy_logits'], dim=1).cpu().numpy()
            pred_offset = outputs['offset'].cpu().numpy() # [B, 2] (TanH values)
            
            # Get True Lat/Lon (we have to reverse engineer this from targets or load raw meta)
            # Targets: [state, gx, gy, off_x, off_y...]
            true_gx = targets[:, 1].cpu().numpy()
            true_gy = targets[:, 2].cpu().numpy()
            # The target offset in dataset is 0..1. We need to use that.
            true_off_raw = targets[:, 3:5].cpu().numpy() 

            for i in range(len(images)):
                # 1. Calculate Predicted Lat/Lon
                p_lat, p_lon = grid_to_latlon(pred_gx[i], pred_gy[i], pred_offset[i][0], pred_offset[i][1])
                
                # 2. Calculate True Lat/Lon
                # Note: true_off_raw is 0..1, so we map 0..1 -> 0..1 (no change needed)
                # But grid_to_latlon expects -1..1 input for offset logic we wrote above.
                # Let's adjust logic manually for Truth:
                
                lat_step = (US_LAT_MAX - US_LAT_MIN) / GRID_SIZE
                lon_step = (US_LON_MAX - US_LON_MIN) / GRID_SIZE
                t_lat = US_LAT_MIN + (true_gx[i] * lat_step) + (true_off_raw[i][0] * lat_step)
                t_lon = US_LON_MIN + (true_gy[i] * lon_step) + (true_off_raw[i][1] * lon_step)
                
                # 3. Compute Distance
                dist = haversine_distance(p_lat, p_lon, t_lat, t_lon)
                distances.append(dist)
    return distances

def main():
    # Load Data
    val_ds = GeoGuessrDataset("./images", json_file='val_meta.jsonl', is_train=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # Load Model
    model = GeoGuessorModel(num_grid_x=20, num_grid_y=20, d_model=512).to(DEVICE)
    model.load_state_dict(torch.load("best_geoguessr_model.pth"))
    
    print("Calculating distances...")
    distances = evaluate(model, val_loader)
    
    # Statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    
    print(f"\n--- RESULTS ---")
    print(f"Mean Error:   {mean_dist:.2f} km")
    print(f"Median Error: {median_dist:.2f} km")
    
    # "Street Guess" Accuracy (Within 25km)
    acc_25km = np.sum(np.array(distances) < 25) / len(distances)
    # "City Guess" Accuracy (Within 100km)
    acc_100km = np.sum(np.array(distances) < 100) / len(distances)
    # "State Guess" Accuracy (Within 500km)
    acc_500km = np.sum(np.array(distances) < 500) / len(distances)
    
    print(f"Within 25km:  {acc_25km:.2%}")
    print(f"Within 100km: {acc_100km:.2%}")
    print(f"Within 500km: {acc_500km:.2%}")

if __name__ == "__main__":
    main()    
