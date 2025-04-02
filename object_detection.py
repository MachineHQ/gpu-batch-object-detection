import os
import time
import csv
import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import pipeline
from datasets import load_dataset
from urllib.parse import urlparse

# Define folder paths
IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs("detection_output", exist_ok=True)

# Load metadata for the first 1000 samples from phiyodr/coco2017
print("Loading metadata from phiyodr/coco2017 (first 1000 samples)...")
dataset = load_dataset("phiyodr/coco2017", split="train[:1000]")

def download_image(example):
    # Determine image URL from available keys
    if "url" in example:
        image_url = example["url"]
    elif "coco_url" in example:
        image_url = example["coco_url"]
    elif "flickr_url" in example:
        image_url = example["flickr_url"]
    else:
        raise KeyError("No image URL found in the example.")

    # Derive filename from URL
    file_name = os.path.basename(urlparse(image_url).path)
    image_path = os.path.join("images", file_name)

    # Ensure directory exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    # Download and save image
    response = requests.get(image_url)
    with open(image_path, "wb") as f:
        f.write(response.content)

    # Save image path under a key that DETR expects
    example["image_path"] = image_path
    return example

# Download images one-by-one as needed
dataset = dataset.map(download_image)

# Diagnostic: Check GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA is available. Using GPU: {gpu_name}")
else:
    print("CUDA is not available. Using CPU (inference will be much slower).")

# Load DETR object detection pipeline (using GPU if available)
print("Loading DETR model...")
start_time = time.time()
device = 0 if torch.cuda.is_available() else -1
object_detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
print(f"DETR model loaded in {time.time() - start_time:.1f} seconds.")

def annotate_image(image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        # Try to retrieve the bounding box from either "box" or "bbox"
        if "box" in detection:
            bbox = detection["box"]
        elif "bbox" in detection:
            bbox = detection["bbox"]
        else:
            continue

        # If bbox is a dictionary, extract numeric values
        if isinstance(bbox, dict):
            try:
                xmin = int(float(bbox.get("xmin", 0)))
                ymin = int(float(bbox.get("ymin", 0)))
                xmax = int(float(bbox.get("xmax", 0)))
                ymax = int(float(bbox.get("ymax", 0)))
            except Exception as e:
                print("Error converting bbox dict:", bbox, e)
                continue
        # Otherwise, if bbox is a list or tuple
        else:
            try:
                # Attempt to convert each element to a float and then int
                coords = [int(float(x)) for x in bbox]
            except Exception as e:
                print("Error converting bbox list:", bbox, e)
                continue
            if len(coords) != 4:
                print("Unexpected bbox format:", coords)
                continue
            xmin, ymin, xmax, ymax = coords

        # Draw the rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # Optionally, draw label and score
        label = detection.get("label", "object")
        score = detection.get("score", 0)
        draw.text((xmin, ymin), f"{label}: {score:.2f}", fill="red")

    return image

# Prepare CSV to record detection results
csv_rows = []
csv_header = ["image_name", "detection_index", "label", "score", "xmin", "ymin", "xmax", "ymax"]

# Process each downloaded image
for example in dataset:
    img_path = example["image_path"]
    print(f"Processing {img_path}...")
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image {img_path}: {e}")
        continue

    detections = object_detector(image)
    print(f"Found {len(detections)} objects in {example['file_name']}.")

    # Annotate and save the image
    # Construct the full save path based on your desired output structure.
    save_path = os.path.join("detection_output", f"detected_{example['file_name']}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    annotated_image = annotate_image(image.copy(), detections)
    annotated_image.save(os.path.join("detection_output", f"detected_{example['file_name']}"))

    # Record detection details for CSV
    for idx, det in enumerate(detections):
        xmin, ymin, xmax, ymax = det["box"]
        csv_rows.append([
            example["file_name"], idx, det["label"], det["score"],
            xmin, ymin, xmax, ymax
        ])

# Write detection results to CSV
csv_path = os.path.join("detection_output", "detections.csv")
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)
print(f"Detection results saved to {csv_path}")
print("Detection complete.")
