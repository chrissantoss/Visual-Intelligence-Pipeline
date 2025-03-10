#!/usr/bin/env python3
"""
Download sample images for testing the Apple Visual Intelligence system.

This script downloads sample images from the COCO dataset for testing.

Usage:
    python download_sample_images.py
"""

import os
import sys
import argparse
import requests
import zipfile
import io
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("download_samples")

# COCO sample images URL
COCO_SAMPLE_URL = "http://images.cocodataset.org/zips/val2017.zip"

def download_coco_samples(num_images: int = 10):
    """Download sample images from the COCO dataset."""
    logger.info(f"Downloading {num_images} sample images from COCO dataset")
    
    # Create data directory
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have enough images
    existing_images = list(data_dir.glob("*.jpg"))
    if len(existing_images) >= num_images:
        logger.info(f"Already have {len(existing_images)} images, skipping download")
        return
    
    # Download a small subset of COCO images
    try:
        # Download the first part of the zip file
        logger.info(f"Downloading from {COCO_SAMPLE_URL}")
        response = requests.get(COCO_SAMPLE_URL, stream=True)
        response.raise_for_status()
        
        # Extract images from the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Get all jpg files
            jpg_files = [f for f in z.namelist() if f.endswith(".jpg")]
            
            # Extract a subset of images
            for i, jpg_file in enumerate(jpg_files[:num_images]):
                z.extract(jpg_file, data_dir)
                logger.info(f"Extracted {jpg_file} ({i+1}/{num_images})")
        
        logger.info(f"Downloaded {num_images} sample images to {data_dir}")
    except Exception as e:
        logger.error(f"Error downloading sample images: {str(e)}")
        
        # If download fails, create synthetic images
        logger.info("Creating synthetic images instead")
        import cv2
        import numpy as np
        
        for i in range(num_images):
            # Create a synthetic image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some random shapes
            for _ in range(3):
                # Random color
                color = tuple(np.random.randint(0, 256, 3).tolist())
                
                # Random shape type
                shape_type = np.random.choice(["rectangle", "circle", "triangle"])
                
                if shape_type == "rectangle":
                    x1 = np.random.randint(0, 540)
                    y1 = np.random.randint(0, 380)
                    x2 = x1 + np.random.randint(50, 100)
                    y2 = y1 + np.random.randint(50, 100)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                
                elif shape_type == "circle":
                    x = np.random.randint(100, 540)
                    y = np.random.randint(100, 380)
                    radius = np.random.randint(30, 80)
                    cv2.circle(image, (x, y), radius, color, -1)
                
                elif shape_type == "triangle":
                    x1 = np.random.randint(100, 540)
                    y1 = np.random.randint(100, 380)
                    x2 = x1 + np.random.randint(-50, 50)
                    y2 = y1 + np.random.randint(-50, 50)
                    x3 = x1 + np.random.randint(-50, 50)
                    y3 = y1 + np.random.randint(-50, 50)
                    pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
                    cv2.fillPoly(image, [pts], color)
            
            # Save the image
            cv2.imwrite(str(data_dir / f"synthetic_{i:04d}.jpg"), image)
            logger.info(f"Created synthetic_{i:04d}.jpg ({i+1}/{num_images})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sample images for Apple Visual Intelligence")
    parser.add_argument("--num-images", type=int, help="Number of images to download", default=10)
    args = parser.parse_args()
    
    download_coco_samples(args.num_images) 