#!/usr/bin/env python3
"""
Batch processing script for the Apple Visual Intelligence system.

This script processes a directory of images and saves the results.

Usage:
    python batch_process.py --input-dir data/sample --output-dir results
"""

import os
import sys
import time
import argparse
import requests
import base64
import cv2
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("batch_process")

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_image(image_path: str) -> Dict[str, Any]:
    """Process an image using the API."""
    logger.info(f"Processing {image_path}")
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Send request to API
        response = requests.post(
            f"{API_URL}/api/analyze_scene",
            json={"image": base64_image},
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        logger.info(f"Processed {image_path}: {len(result['detected_objects'])} objects detected")
        
        return {
            "image_path": image_path,
            "success": True,
            "result": result,
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        
        return {
            "image_path": image_path,
            "success": False,
            "error": str(e),
        }

def visualize_result(image_path: str, result: Dict[str, Any], output_path: str):
    """Visualize the result of processing an image."""
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw bounding boxes for detected objects
    for obj in result["detected_objects"]:
        bbox = obj["bounding_box"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, vis_image.shape[1] - 1))
        y1 = max(0, min(y1, vis_image.shape[0] - 1))
        x2 = max(0, min(x2, vis_image.shape[1] - 1))
        y2 = max(0, min(y2, vis_image.shape[0] - 1))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw label
        label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Save visualization
    cv2.imwrite(output_path, vis_image)
    
    logger.info(f"Visualization saved to {output_path}")

def batch_process(input_dir: str, output_dir: str, num_workers: int = 4):
    """Process all images in a directory."""
    logger.info(f"Batch processing images in {input_dir}")
    
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    vis_dir = output_dir_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    json_dir = output_dir_path / "json"
    json_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images in parallel
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(process_image, str(path)): path
            for path in image_files
        }
        
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results and create visualizations
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    logger.info(f"Processed {len(results)} images in {total_time:.2f}s")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")
    
    # Create visualizations and save JSON results
    for result in successful_results:
        image_path = result["image_path"]
        image_name = Path(image_path).name
        base_name = Path(image_path).stem
        
        # Save visualization
        vis_path = str(vis_dir / f"{base_name}_result.jpg")
        visualize_result(image_path, result["result"], vis_path)
        
        # Save JSON result
        json_path = str(json_dir / f"{base_name}_result.json")
        with open(json_path, "w") as f:
            json.dump(result["result"], f, indent=2)
    
    # Save summary
    summary = {
        "total_images": len(results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "total_time": total_time,
        "average_time_per_image": total_time / len(results) if results else 0,
        "failed_images": [r["image_path"] for r in failed_results],
    }
    
    with open(str(output_dir_path / "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    return len(successful_results) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images with Apple Visual Intelligence")
    parser.add_argument("--api-url", help="API URL (default: http://localhost:8000)", default="http://localhost:8000")
    parser.add_argument("--input-dir", help="Input directory containing images", required=True)
    parser.add_argument("--output-dir", help="Output directory for results", required=True)
    parser.add_argument("--num-workers", type=int, help="Number of worker threads", default=4)
    args = parser.parse_args()
    
    API_URL = args.api_url
    
    success = batch_process(args.input_dir, args.output_dir, args.num_workers)
    sys.exit(0 if success else 1) 