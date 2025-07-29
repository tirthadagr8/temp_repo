import os
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
import time

# --- Configuration ---
META_FILE = 'OID-rated-image-captions.v2.train.tsv.meta'
CAPTIONS_FILE = 'OID-rated-image-captions.v2.train.tsv'
IMAGE_DIR = 'images'
PROGRESS_FILE = 'image_filenames.json'
OUTPUT_CSV = 'image_captions.csv'

# Step 1: Read the metadata and captions TSV files
print("--- Initializing ---")
print(f"Reading metadata from {META_FILE}...")
try:
    meta_df = pd.read_csv(META_FILE, sep='\t')
    print(f"Reading captions data from {CAPTIONS_FILE}...")
    tsv_df = pd.read_csv(CAPTIONS_FILE, sep='\t')
except FileNotFoundError as e:
    print(f"Error: Could not find a required file: {e.filename}. Please ensure all files are in the correct directory.")
    exit()

# Step 2: Create the images directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Step 3: Define a mapping from PIL format names to file extensions
FORMAT_TO_EXTENSION = {
    'JPEG': 'jpg',
    'PNG': 'png',
    'GIF': 'gif',
    'BMP': 'bmp',
    'WEBP': 'webp'
}

# Step 4: Load existing progress if available
image_filenames = {}
try:
    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
        image_filenames = json.load(f)
    print(f"Resuming from previous session. Loaded {len(image_filenames)} completed downloads.")
except (FileNotFoundError, json.JSONDecodeError):
    print("Starting a new download session or progress file was corrupted.")
    pass # If file doesn't exist or is empty/corrupt, start fresh

stop_index = len(meta_df)//2

# --- Main Download Loop ---
print("\n--- Starting Image Download Process ---")
# Use tqdm to create a progress bar for the download loop
with tqdm(total=len(meta_df), desc='Downloading images') as pbar:
    pbar.update(len(image_filenames)) # Initialize progress bar to show already completed downloads
    for index, row in meta_df.iterrows():
		
	if index>=stop_index:
		break
		
        image_key = row['IMAGE_KEY']

        # --- Check if already downloaded ---
        if image_key in image_filenames:
            continue # Skip to the next image if already processed

        original_url = row['OriginalURL']

        # Skip if URL is invalid
        if not isinstance(original_url, str) or not original_url.startswith('http'):
            pbar.update(1) # Still need to advance the progress bar
            continue

        try:
            # Download the image with a timeout
            response = requests.get(original_url, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Verify that the response content is actually an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                pbar.update(1)
                continue

            # Use PIL to verify the image and get its format
            with Image.open(BytesIO(response.content)) as img:
                img.verify() # Verifies that the file is a valid image
                img_format = img.format

            if not img_format:
                pbar.update(1)
                continue

            # Map the format to a file extension, defaulting to 'jpg'
            file_ext = FORMAT_TO_EXTENSION.get(img_format.upper(), 'jpg')
            filename = f"{image_key}.{file_ext}"
            filepath = os.path.join(IMAGE_DIR, filename)

            # Save the image content to a file
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # --- Record the successful download and save progress immediately ---
            image_filenames[image_key] = filename
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(image_filenames, f, indent=4)

        except Exception:
            # We'll just skip this image on error and let the loop continue
            pass
        finally:
            # IMPORTANT: Update the progress bar regardless of success or failure
            pbar.update(1)


# --- Final CSV Creation ---
print("\n--- Generating Final Captions CSV ---")
csv_rows = []
skipped_count = 0

for index, row in tqdm(tsv_df.iterrows(), total=len(tsv_df), desc="Building CSV"):
    image_key = row['IMAGE_KEY']
    caption = row['CAPTION_PRED']

    if image_key in image_filenames:
        csv_rows.append([image_filenames[image_key], caption])
    else:
        skipped_count += 1

if skipped_count > 0:
    print(f"Note: Skipped {skipped_count} captions because their images could not be downloaded.")

# Write the CSV file
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'caption'])  # Header row
    writer.writerows(csv_rows)

print(f"\n✅ All Done! Downloaded a total of {len(image_filenames)} images.")
print(f"Final captions file saved to: {OUTPUT_CSV}")
