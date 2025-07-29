import os
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
import csv
from tqdm import tqdm

# Step 1: Read the TSV files
meta_df = pd.read_csv('OID-rated-image-captions.v2.train.tsv.meta', sep='\t')
tsv_df = pd.read_csv('OID-rated-image-captions.v2.train.tsv', sep='\t')

# Step 2: Create the images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Step 3: Map to convert PIL format names to file extensions
FORMAT_TO_EXTENSION = {
    'JPEG': 'jpg',
    'PNG': 'png',
    'GIF': 'gif',
    'BMP': 'bmp',
    'WEBP': 'webp'
}

# Dictionary to hold mapping from IMAGE_KEY to saved image filename
image_filenames = {}

# Step 4: Download and save each image
for index, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Downloading images'):
    image_key = row['IMAGE_KEY']
    original_url = row['OriginalURL']

    if not isinstance(original_url, str) or not original_url.startswith('http'):
        print(f"Invalid or missing URL for {image_key}")
        continue

    try:
        print(f"Downloading: {image_key}")
        response = requests.get(original_url, timeout=10)
        response.raise_for_status()

        # Verify that the response is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            print(f"URL {original_url} is not an image (Content-Type: {content_type})")
            continue

        # Load image data into memory
        data = BytesIO(response.content)
        image = Image.open(data)
        image.verify()  # Verifies that the file is indeed an image

        if not image.format:
            print(f"Could not determine image format for {image_key}")
            continue

        # Map the format to a file extension
        file_ext = FORMAT_TO_EXTENSION.get(image.format.upper(), 'jpg')

        # Save the image with the correct format
        filename = f"{image_key}.{file_ext}"
        filepath = os.path.join('images', filename)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        image_filenames[image_key] = filename
        print(f"Saved: {filepath}")

    except Exception as e:
        print(f"Error downloading {original_url}: {e}")

# Step 5: Build the image-to-caption CSV
csv_rows = []

for index, row in tsv_df.iterrows():
    image_key = row['IMAGE_KEY']
    caption = row['CAPTION_PRED']

    if image_key in image_filenames:
        csv_rows.append([image_filenames[image_key], caption])
    else:
        print(f"Skipping caption for {image_key} — image not downloaded")

# Step 6: Write the CSV file
output_csv = 'image_captions.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'caption'])  # Header row
    writer.writerows(csv_rows)

print(f"\n✅ Done! Captions saved to: {output_csv}")
