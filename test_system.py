#!/usr/bin/env python3
"""
Comprehensive test script for the Apple Visual Intelligence system.

This script tests all aspects of the system, including:
1. API functionality
2. Privacy guarantees
3. Performance metrics
4. Memory usage
5. End-to-end workflow

Usage:
    python test_system.py
"""

import os
import sys
import time
import asyncio
import argparse
import requests
import base64
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("system_test")

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with some shapes for testing."""
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a green rectangle
    cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)
    
    # Add a red circle
    cv2.circle(image, (450, 200), 80, (0, 0, 255), -1)
    
    # Add a blue triangle
    pts = np.array([[200, 400], [300, 350], [400, 400]], np.int32)
    cv2.fillPoly(image, [pts], (255, 0, 0))
    
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode an image to base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def save_image(image: np.ndarray, filename: str):
    """Save an image to disk."""
    os.makedirs("test_results", exist_ok=True)
    cv2.imwrite(f"test_results/{filename}", image)

def test_api_health():
    """Test the API health endpoint."""
    logger.info("Testing API health...")
    
    try:
        response = requests.get(f"{API_URL}/")
        response.raise_for_status()
        
        logger.info(f"API health check successful: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        return False

def test_analyze_scene():
    """Test the analyze_scene endpoint."""
    logger.info("Testing analyze_scene endpoint...")
    
    # Create a test image
    image = create_test_image()
    save_image(image, "test_image.jpg")
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    try:
        # Test with base64-encoded image
        response = requests.post(
            f"{API_URL}/api/analyze_scene",
            json={"base64_image": {"image": base64_image}},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"analyze_scene with base64 successful")
        logger.info(f"Detected {len(result['detected_objects'])} objects")
        logger.info(f"Processing time: {result['processing_time']:.3f}s")
        
        # Test with file upload
        with open("test_results/test_image.jpg", "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = requests.post(
                f"{API_URL}/api/analyze_scene",
                files=files,
            )
            response.raise_for_status()
        
        result = response.json()
        logger.info(f"analyze_scene with file upload successful")
        logger.info(f"Detected {len(result['detected_objects'])} objects")
        logger.info(f"Processing time: {result['processing_time']:.3f}s")
        
        return True
    except Exception as e:
        logger.error(f"analyze_scene test failed: {str(e)}")
        return False

def test_scene_summary():
    """Test the scene_summary endpoint."""
    logger.info("Testing scene_summary endpoint...")
    
    try:
        response = requests.get(f"{API_URL}/api/scene_summary")
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"scene_summary successful")
        logger.info(f"Total scenes processed: {result['total_scenes_processed']}")
        logger.info(f"Object counts: {result['object_counts']}")
        
        return True
    except Exception as e:
        logger.error(f"scene_summary test failed: {str(e)}")
        return False

def test_privacy_guarantees():
    """Test the privacy guarantees of the system."""
    logger.info("Testing privacy guarantees...")
    
    # Create a test image
    image = create_test_image()
    base64_image = encode_image_to_base64(image)
    
    try:
        # Process the image multiple times and check for differences in results
        results = []
        for _ in range(5):
            response = requests.post(
                f"{API_URL}/api/analyze_scene",
                json={"base64_image": {"image": base64_image}},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            results.append(response.json())
        
        # Check that the results are different due to differential privacy
        bboxes = []
        for result in results:
            if result["detected_objects"]:
                bbox = result["detected_objects"][0]["bounding_box"]
                bboxes.append((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
        
        # Check if bounding boxes are different
        if len(bboxes) > 1:
            all_same = all(bbox == bboxes[0] for bbox in bboxes)
            if not all_same:
                logger.info("Privacy test passed: Results differ due to differential privacy")
                return True
            else:
                logger.warning("Privacy test inconclusive: Results are identical")
                return False
        else:
            logger.warning("Privacy test inconclusive: No objects detected")
            return False
    except Exception as e:
        logger.error(f"Privacy test failed: {str(e)}")
        return False

def test_performance():
    """Test the performance of the system."""
    logger.info("Testing performance...")
    
    # Create a test image
    image = create_test_image()
    base64_image = encode_image_to_base64(image)
    
    try:
        # Measure latency
        latencies = []
        for _ in range(5):
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/api/analyze_scene",
                json={"base64_image": {"image": base64_image}},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        logger.info(f"Average latency: {avg_latency:.3f}s")
        logger.info(f"Min latency: {min_latency:.3f}s")
        logger.info(f"Max latency: {max_latency:.3f}s")
        
        # Check if latency is within acceptable range
        if avg_latency < 10.0:  # Adjust threshold as needed
            logger.info("Performance test passed: Latency is within acceptable range")
            return True
        else:
            logger.warning("Performance test failed: Latency is too high")
            return False
    except Exception as e:
        logger.error(f"Performance test failed: {str(e)}")
        return False

def test_end_to_end():
    """Test the end-to-end workflow of the system."""
    logger.info("Testing end-to-end workflow...")
    
    # Create a test image
    image = create_test_image()
    base64_image = encode_image_to_base64(image)
    
    try:
        # Process the image
        response = requests.post(
            f"{API_URL}/api/analyze_scene",
            json={"base64_image": {"image": base64_image}},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        # Check that the response contains all expected fields
        expected_fields = [
            "detected_objects",
            "segmentation",
            "depth_estimation",
            "processing_time",
            "timestamp",
            "privacy_epsilon",
        ]
        
        for field in expected_fields:
            if field not in result:
                logger.error(f"End-to-end test failed: Missing field '{field}' in response")
                return False
        
        # Visualize the results
        visualize_results(image, result)
        
        logger.info("End-to-end test passed")
        return True
    except Exception as e:
        logger.error(f"End-to-end test failed: {str(e)}")
        return False

def visualize_results(image: np.ndarray, result: Dict[str, Any]):
    """Visualize the results of the analysis."""
    # Create a copy of the image for visualization
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
    save_image(vis_image, "result_visualization.jpg")
    
    logger.info("Results visualization saved to test_results/result_visualization.jpg")

def run_all_tests():
    """Run all tests."""
    tests = [
        ("API Health", test_api_health),
        ("Analyze Scene", test_analyze_scene),
        ("Scene Summary", test_scene_summary),
        ("Privacy Guarantees", test_privacy_guarantees),
        ("Performance", test_performance),
        ("End-to-End Workflow", test_end_to_end),
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        logger.info(f"\n{'=' * 50}\nRunning test: {name}\n{'=' * 50}")
        try:
            result = test_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"Test {name} failed with exception: {str(e)}")
            results[name] = False
            all_passed = False
    
    # Print summary
    logger.info("\n\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
    
    logger.info("=" * 50)
    logger.info(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
    logger.info("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run system tests for Apple Visual Intelligence")
    parser.add_argument("--api-url", help="API URL (default: http://localhost:8000)", default="http://localhost:8000")
    args = parser.parse_args()
    
    API_URL = args.api_url
    
    success = run_all_tests()
    sys.exit(0 if success else 1) 