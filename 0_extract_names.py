import os
import csv
import pandas as pd
from pathlib import Path
import shutil

def extract_image_names(bridge_name):
    
    folder_path = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\Input images"
    output_csv = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_image_list.csv"

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
    # Get only image files (not directories)
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in image_exts
    ]

    # Write the list to a CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename"])  # Header
        for file in files:
            writer.writerow([file])

    print(f"Saved {len(files)} image filenames to {output_csv}")


folders = []
for entry in os.listdir(r"C:\Users\juanj\Desktop\Bridges"):
    if os.path.isdir(os.path.join(r"C:\Users\juanj\Desktop\Bridges", entry)):
        folders.append(entry)

for folder in folders:
    bridge_name = folder #"2_Verrières Viaduct"
    origin_dir = r"C:\Users\juanj\Desktop\Bridges"

    source_dir = Path(origin_dir) / bridge_name
    images_dir = Path(origin_dir) / bridge_name / "Input images"

    images_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
    
    for item in source_dir.iterdir():
        if item.is_file() and item.suffix.lower() in image_exts:
            dst = images_dir / item.name
            shutil.move(item, dst)
    extract_image_names(folder)

# extract_image_names("1_Avignon Viaducts")