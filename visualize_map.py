import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import your code
from model import GeoGuessorModel
from geoguessr_dataset import GeoGuessrDataset
from evaluate_distance import grid_to_latlon # Reuse function

# Constants
US_LAT_MIN, US_LAT_MAX = 24.0, 49.0
US_LON_MIN, US_LON_MAX = -125.0, -66.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Load Data & Model
    print("Loading Validation Data...")
    val_ds = GeoGuessrDataset("./images", json_file='val_meta.jsonl', is_train=False)
    # Shuffle=True to get a random mix of locations for the plot
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4)

    model = GeoGuessorModel(num_grid_x=20, num_grid_y=20, d_model=512).to(DEVICE)
    model.load_state_dict(torch.load("best_geoguessr_model.pth", map_location=DEVICE))
    model.eval()

    # 2. Collect Predictions (Limit to first 100 points to keep map clean)
    true_coords = []
    pred_coords = []
    
    print("Running Inference...")
    count = 0
    MAX_POINTS = 100 # Change this to show more/less lines

    with torch.no_grad():
        for images, targets in val_loader:
            if count >= MAX_POINTS: break
            
            images = images.to(DEVICE)
            outputs = model(images, use_teacher_forcing=False)
            
            # Predictions
            pred_gx = torch.argmax(outputs['gx_logits'], dim=1).cpu().numpy()
            pred_gy = torch.argmax(outputs['gy_logits'], dim=1).cpu().numpy()
            pred_offset = outputs['offset'].cpu().numpy()
            
            # Truth
            true_gx = targets[:, 1].numpy()
            true_gy = targets[:, 2].numpy()
            true_off = targets[:, 3:5].numpy()

            for i in range(len(images)):
                if count >= MAX_POINTS: break
                
                # Convert Grid -> Lat/Lon
                p_lat, p_lon = grid_to_latlon(pred_gx[i], pred_gy[i], pred_offset[i][0], pred_offset[i][1])
                
                # Convert Truth -> Lat/Lon (Reusing logic)
                lat_step = (US_LAT_MAX - US_LAT_MIN) / 20
                lon_step = (US_LON_MAX - US_LON_MIN) / 20
                t_lat = US_LAT_MIN + (true_gx[i] * lat_step) + (true_off[i][0] * lat_step)
                t_lon = US_LON_MIN + (true_gy[i] * lon_step) + (true_off[i][1] * lon_step)

                pred_coords.append((p_lon, p_lat))
                true_coords.append((t_lon, t_lat))
                count += 1

    # 3. Plot Map
    print("Plotting Map...")
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([US_LON_MIN - 2, US_LON_MAX + 2, US_LAT_MIN - 2, US_LAT_MAX + 2])

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')

    # Draw lines
    for i in range(len(true_coords)):
        t_lon, t_lat = true_coords[i]
        p_lon, p_lat = pred_coords[i]
        
        # Color line based on distance (Red = Bad, Green = Good)
        dist = np.sqrt((t_lat-p_lat)**2 + (t_lon-p_lon)**2)
        color = 'green' if dist < 2.0 else ('orange' if dist < 5.0 else 'red')
        
        plt.plot([t_lon, p_lon], [t_lat, p_lat], color=color, linewidth=1, transform=ccrs.PlateCarree())
        plt.plot(t_lon, t_lat, 'bo', markersize=3, transform=ccrs.PlateCarree()) # Blue Dot = Truth

    plt.title(f"GeoGuessr Predictions (First {MAX_POINTS} samples)\nBlue=True, Red Line=Error")
    plt.savefig('results_map.png', dpi=300)
    print("Saved 'results_map.png'")

if __name__ == "__main__":
    main()