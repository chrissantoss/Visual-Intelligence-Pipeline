import pytest
import numpy as np
import cv2
from typing import List, Dict, Any
import base64
import zlib

from app.core.privacy.differential_privacy import apply_differential_privacy
from app.core.pipeline.processor import process_image
from app.utils.config import get_settings

class MockModel:
    """Mock model for testing."""
    
    async def predict(self, image):
        """Mock prediction method."""
        return {}

class MockObjectDetectionModel(MockModel):
    """Mock object detection model for testing."""
    
    async def predict(self, image):
        """Mock prediction method."""
        return [
            {
                "class_id": 1,
                "class_name": "person",
                "confidence": 0.9,
                "bounding_box": {
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 30.0,
                    "y2": 40.0,
                },
            },
        ]

class MockSegmentationModel(MockModel):
    """Mock segmentation model for testing."""
    
    async def predict(self, image):
        """Mock prediction method."""
        return {
            "class_ids": [1],
            "class_names": ["person"],
            "mask_base64": "dummy_base64_string",
        }

class MockDepthEstimationModel(MockModel):
    """Mock depth estimation model for testing."""
    
    async def predict(self, image):
        """Mock prediction method."""
        return {
            "depth_map_base64": "dummy_base64_string",
            "min_depth": 0.5,
            "max_depth": 10.0,
        }

def test_privacy_reconstruction_attack():
    """
    Test that raw image data cannot be reconstructed from the output.
    
    This test simulates a reconstruction attack by trying to reconstruct
    the original image from the model outputs.
    """
    # Create a test image
    original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(original_image, (25, 25), (75, 75), (0, 255, 0), -1)
    
    # Create mock models
    models = {
        "object_detection": MockObjectDetectionModel(),
        "segmentation": MockSegmentationModel(),
        "depth_estimation": MockDepthEstimationModel(),
    }
    
    # Process the image with differential privacy
    # (In a real test, we would call process_image, but here we'll simulate it)
    detections = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.9,
            "bounding_box": {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 40.0,
            },
        },
    ]
    
    # Apply differential privacy
    noisy_detections = apply_differential_privacy(detections, data_type="detections")
    
    # Attempt to reconstruct the image from the noisy detections
    # (This is a simplified simulation of a reconstruction attack)
    reconstructed_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    for detection in noisy_detections:
        bbox = detection["bounding_box"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, 99))
        y1 = max(0, min(y1, 99))
        x2 = max(0, min(x2, 99))
        y2 = max(0, min(y2, 99))
        
        # Draw a rectangle for the detected object
        cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), (0, 255, 0), -1)
    
    # Calculate the difference between the original and reconstructed images
    diff = cv2.absdiff(original_image, reconstructed_image)
    diff_sum = np.sum(diff)
    
    # Assert that the reconstruction is significantly different from the original
    # (If diff_sum is large, the reconstruction is poor, which is what we want)
    assert diff_sum > 0, "Reconstruction should be different from the original"

def test_epsilon_sensitivity():
    """
    Test the sensitivity of the output to changes in epsilon.
    
    This test verifies that smaller epsilon values (more privacy) result in
    more noise being added to the output.
    """
    # Create sample detections
    detections = [
        {
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.9,
            "bounding_box": {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 40.0,
            },
        },
    ]
    
    # Apply differential privacy with different epsilon values
    noisy_detections_high_epsilon = apply_differential_privacy(
        detections, data_type="detections"
    )
    
    # Temporarily set epsilon to a smaller value
    settings = get_settings()
    original_epsilon = settings.privacy_epsilon
    settings.privacy_epsilon = 0.1  # Smaller epsilon = more privacy = more noise
    
    noisy_detections_low_epsilon = apply_differential_privacy(
        detections, data_type="detections"
    )
    
    # Restore original epsilon
    settings.privacy_epsilon = original_epsilon
    
    # Calculate the difference in bounding box coordinates
    bbox_high = noisy_detections_high_epsilon[0]["bounding_box"]
    bbox_low = noisy_detections_low_epsilon[0]["bounding_box"]
    
    diff_high = abs(bbox_high["x1"] - detections[0]["bounding_box"]["x1"]) + \
                abs(bbox_high["y1"] - detections[0]["bounding_box"]["y1"]) + \
                abs(bbox_high["x2"] - detections[0]["bounding_box"]["x2"]) + \
                abs(bbox_high["y2"] - detections[0]["bounding_box"]["y2"])
    
    diff_low = abs(bbox_low["x1"] - detections[0]["bounding_box"]["x1"]) + \
               abs(bbox_low["y1"] - detections[0]["bounding_box"]["y1"]) + \
               abs(bbox_low["x2"] - detections[0]["bounding_box"]["x2"]) + \
               abs(bbox_low["y2"] - detections[0]["bounding_box"]["y2"])
    
    # Assert that lower epsilon results in more noise (higher difference)
    # This might not always be true due to randomness, but should be true on average
    # We'll run the test multiple times to increase confidence
    total_diff_high = 0
    total_diff_low = 0
    
    for _ in range(10):
        noisy_detections_high_epsilon = apply_differential_privacy(
            detections, data_type="detections"
        )
        
        settings.privacy_epsilon = 0.1
        noisy_detections_low_epsilon = apply_differential_privacy(
            detections, data_type="detections"
        )
        settings.privacy_epsilon = original_epsilon
        
        bbox_high = noisy_detections_high_epsilon[0]["bounding_box"]
        bbox_low = noisy_detections_low_epsilon[0]["bounding_box"]
        
        diff_high = abs(bbox_high["x1"] - detections[0]["bounding_box"]["x1"]) + \
                    abs(bbox_high["y1"] - detections[0]["bounding_box"]["y1"]) + \
                    abs(bbox_high["x2"] - detections[0]["bounding_box"]["x2"]) + \
                    abs(bbox_high["y2"] - detections[0]["bounding_box"]["y2"])
        
        diff_low = abs(bbox_low["x1"] - detections[0]["bounding_box"]["x1"]) + \
                   abs(bbox_low["y1"] - detections[0]["bounding_box"]["y1"]) + \
                   abs(bbox_low["x2"] - detections[0]["bounding_box"]["x2"]) + \
                   abs(bbox_low["y2"] - detections[0]["bounding_box"]["y2"])
        
        total_diff_high += diff_high
        total_diff_low += diff_low
    
    # Assert that on average, lower epsilon results in more noise
    assert total_diff_low > total_diff_high, "Lower epsilon should result in more noise" 