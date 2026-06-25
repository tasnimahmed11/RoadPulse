import os
import csv
import json
import time
import argparse
import requests

# Configuration Constants
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
TRACKER_FILE = "download_tracker.csv"
OUTPUT_DIR = "streetview_images"


def load_tracker():
    """Loads already processed feature IDs to avoid redundant downloads."""
    completed_ids = set()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                if row:
                    completed_ids.add(row[0])
    return completed_ids


def mark_completed(feature_id):
    """Appends a successfully processed feature ID to the tracking CSV."""
    file_exists = os.path.exists(TRACKER_FILE)
    with open(TRACKER_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "timestamp"])
        writer.writerow([feature_id, time.strftime("%Y-%m-%d %H:%M:%S")])


def download_streetview_images(lat, lng, feature_id):
    """Downloads 4 directional images for a specific location."""
    headings = [0, 90, 180, 270]
    
    for heading in headings:
        filename = f"{feature_id}_heading_{heading}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # Skip individual image download if it already exists locally
        if os.path.exists(save_path):
            continue

        params = {
            "size": "640x640",
            "location": f"{lat},{lng}",
            "heading": str(heading),
            "pitch": "0",
            "fov": "90",
            "key": API_KEY,
            "return_error_code": "true"  # Crucial: Returns 404 instead of a paid blank gray tile
        }

        retries = 3
        while retries > 0:
            try:
                response = requests.get(BASE_URL, params=params, timeout=15)
                
                if response.status_code == 200:
                    with open(save_path, "wb") as img_file:
                        img_file.write(response.content)
                    break  # Success, exit retry loop
                    
                elif response.status_code == 404:
                    print(f"  [404 Not Found] No imagery for heading {heading} at {lat},{lng}")
                    break  # No image available, don't waste retries
                    
                elif response.status_code == 429:
                    print("  [429 Rate Limit] Hit API rate limit. Backing off for 2 seconds...")
                    time.sleep(2)
                    retries -= 1
                    
                else:
                    print(f"  [Error {response.status_code}] Failed fetching heading {heading}. Retrying...")
                    retries -= 1
                    time.format(1)
            except Exception as e:
                print(f"  [Connection Error] {e}. Retrying...")
                retries -= 1
                time.sleep(1)
        
        # A tiny safety sleep between rapid-fire calls to stay safely below aggressive bursts
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Download 4-directional Google Street View images for valid GeoJSON entries."
    )
    parser.add_argument("filepath", help="Path to the .geojson / .json file")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional max number of entries to process this session (default: all)"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data and progress tracker
    print("Loading GeoJSON dataset...")
    with open(args.filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    total_features = len(features)
    print(f"Total features in file: {total_features}")

    completed_ids = load_tracker()
    print(f"Found {len(completed_ids)} already processed entries in tracker.")

    processed_count = 0

    for feature in features:
        # Enforce session limit if one was set via CLI arguments
        if args.limit and processed_count >= args.limit:
            print(f"\nReached session limit of {args.limit} entries. Stopping.")
            break

        properties = feature.get("properties", {}) or {}
        analysis_status = properties.get("AnalysisStatus")

        # Condition 1: Check if status is explicitly Valid
        if analysis_status != "Valid":
            continue

        # Extract unique identifier
        feature_id = str(feature.get("id") or properties.get("OBJECTID") or properties.get("OvertureID"))
        
        # Condition 2: Skip if already verified in tracker CSV
        if feature_id in completed_ids:
            continue

        street_image_link = properties.get("StreetImageLink")
        if not street_image_link:
            print(f"Row ID {feature_id}: Missing 'StreetImageLink' coordinates. Skipping.")
            continue

        # Coordinate parsing from 'StreetImageLink' string (expects: lng1,lat1,lng2,lat2)
        try:
            coord_tokens = [c.strip() for c in street_image_link.split(",") if c.strip()]
            lng = coord_tokens[0]
            lat = coord_tokens[1]
        except IndexError:
            print(f"Row ID {feature_id}: 'StreetImageLink' parsing failed ('{street_image_link}'). Skipping.")
            continue

        print(f"Processing Row ID: {feature_id} | Location: {lat}, {lng}")
        
        # Run the down loader loop
        download_streetview_images(lat, lng, feature_id)
        
        # Log entry to CSV tracker to prevent re-processing next run
        mark_completed(feature_id)
        processed_count += 1

    print(f"\nSession complete! Successfully processed {processed_count} new valid entries.")


if __name__ == "__main__":
    main()