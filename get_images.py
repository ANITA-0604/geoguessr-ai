import requests
from tqdm import tqdm
import os
import json
from random import randint
import argparse
import random


STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"

cities = []

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", help="Batch number for processing", type=int, required=True)
    parser.add_argument("--cities", help="The file full of addresses per city to read and extract GPS coordinates from", required=True, type=str)
    parser.add_argument("--output", help="The output folder where the images will be stored, (defaults to: images/)", default='images/', type=str)
    parser.add_argument("--icount", help="Images per city", default=200, type=int)
    parser.add_argument("--key", help="Your Google Street View API Key", type=str, required=True)
    return parser.parse_args()

args = get_args()

def load_cities(batch, BATCH_SIZE):
    start_line = (batch - 1) * BATCH_SIZE

    with open(args.cities, 'r') as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue  
            
            if idx >= start_line + BATCH_SIZE:
                break  
            
            data = json.loads(line)
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                cities.append([lon, lat])

def jitter(coord, radius=0.035):
    lat, lon = coord
    return lat + random.uniform(-radius, radius), lon + random.uniform(-radius, radius)

def main():
    # Open and create all the necessary files & folders
    os.makedirs(args.output, exist_ok=True)
    execute_batch = args.batch
    BATCH_SIZE = 50 # 50 * 200 * 2 = 20000 ( retry 2 times)
    load_cities(execute_batch, BATCH_SIZE)
   
    
    num_cities = len(cities)
    total_images = num_cities * args.icount
    
    elevations = []
    for coord in cities:
        el_params = {"locations": f"{coord[1]},{coord[0]}", "key": args.key}
        el_resp = requests.get(ELEVATION_URL, params=el_params).json()
        elevations.append(el_resp["results"][0]["elevation"])
    
    geocodes = []
    for coord in cities:
        geo_params = {"latlng": f"{coord[1]},{coord[0]}", "key": args.key}
        geo_resp = requests.get(GEOCODING_URL, params=geo_params).json()
        city_name, state_name, area_name = "", "", ""
        if geo_resp.get("results"):
            components = geo_resp["results"][0].get("address_components", [])
            for comp in components:
                if "locality" in comp["types"]:
                    city_name = comp["long_name"]
                if "administrative_area_level_1" in comp["types"]:
                    state_name = comp["long_name"]
                if "sublocality" in comp["types"] or "neighborhood" in comp["types"]:
                    area_name = comp["long_name"]
        geocodes.append([city_name, state_name, area_name])
    meta_output = open(os.path.join(args.output, 'meta.jsonl'), 'a')
    
    for i in tqdm(range(total_images)):
        city_idx = i // args.icount

        # Retry loop to avoid ZERO_RESULTS and ensure valid pano
        success = False
        for attempt in range(2):
            # jitter around city center
            lat_j, lon_j = jitter((cities[city_idx][1], cities[city_idx][0]))

            # check metadata first to avoid wasting money on image API
            meta_params = {
                "location": f"{lat_j},{lon_j}",
                "radius": 80,   # let Google search within 80m for nearest pano
                "key": args.key
            }
            meta_resp = requests.get(STREETVIEW_METADATA_URL, params=meta_params).json()

            if meta_resp.get("status") == "OK":
                success = True
                break  # valid pano found, proceed
        if not success:
            # skip this image and move on
            continue

        # actual pano location (Google may snap slightly)
        pano_lat = meta_resp["location"]["lat"]
        pano_lon = meta_resp["location"]["lng"]
        date = meta_resp.get("date", "")

        elev = elevations[city_idx]

        # Street View image parameters
        params = {
            'key': args.key,
            'size': '640x640',
            'location': f"{pano_lat},{pano_lon}",
            'heading': str((randint(0, 3) * 90) + randint(-15, 15)),
            'pitch': '20',
            'fov': '90',
            'radius': 80
        }
        global_i = (execute_batch - 1) * BATCH_SIZE * args.icount + i
        response = requests.get(STREETVIEW_URL, params)

        # Save image
        with open(os.path.join(args.output, f'street_view_{global_i}.jpg'), "wb") as file:
            file.write(response.content)

        record = {
            "id": global_i,
            "img": f"street_view_{global_i}.jpg",
            "lat": pano_lat,
            "lon": pano_lon,
            "elevation": elev,
            "date": date,
            "city": geocodes[city_idx][0],
            "state": geocodes[city_idx][1],
            "area": geocodes[city_idx][2]
        }
        meta_output.write(json.dumps(record) + "\n")

    meta_output.close()

if __name__ == '__main__':
    main()
