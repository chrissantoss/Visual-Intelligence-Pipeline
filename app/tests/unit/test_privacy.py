import pytest
import numpy as np
from typing import Dict, Any, List

from app.core.privacy.differential_privacy import (
    apply_differential_privacy,
    _apply_laplace_noise,
    _apply_noise_to_detections,
    _apply_noise_to_depth,
)

def test_apply_laplace_noise():
    """Test applying Laplace noise to feature vectors."""
    # Create a sample feature vector
    features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    # Apply noise
    noisy_features = _apply_laplace_noise(features, epsilon=1.0)
    
    # Check that the shape is preserved
    assert noisy_features.shape == features.shape
    
    # Check that the values have changed
    assert not np.array_equal(noisy_features, features)

def test_apply_noise_to_detections():
    """Test applying noise to object detection results."""
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
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8,
            "bounding_box": {
                "x1": 50.0,
                "y1": 60.0,
                "x2": 70.0,
                "y2": 80.0,
            },
        },
    ]
    
    # Apply noise
    noisy_detections = _apply_noise_to_detections(detections, epsilon=1.0)
    
    # Check that the number of detections is preserved
    assert len(noisy_detections) == len(detections)
    
    # Check that the class IDs and names are preserved
    for i in range(len(detections)):
        assert noisy_detections[i]["class_id"] == detections[i]["class_id"]
        assert noisy_detections[i]["class_name"] == detections[i]["class_name"]
    
    # Check that the bounding box coordinates have changed
    for i in range(len(detections)):
        bbox_original = detections[i]["bounding_box"]
        bbox_noisy = noisy_detections[i]["bounding_box"]
        
        assert bbox_noisy["x1"] != bbox_original["x1"] or \
               bbox_noisy["y1"] != bbox_original["y1"] or \
               bbox_noisy["x2"] != bbox_original["x2"] or \
               bbox_noisy["y2"] != bbox_original["y2"]

def test_apply_noise_to_depth():
    """Test applying noise to depth estimation results."""
    # Create sample depth estimation results
    depth = {
        "depth_map_base64": "dummy_base64_string",
        "min_depth": 0.5,
        "max_depth": 10.0,
    }
    
    # Apply noise
    noisy_depth = _apply_noise_to_depth(depth, epsilon=1.0)
    
    # Check that the depth map is preserved
    assert noisy_depth["depth_map_base64"] == depth["depth_map_base64"]
    
    # Check that the min and max depth values have changed
    assert noisy_depth["min_depth"] != depth["min_depth"]
    assert noisy_depth["max_depth"] != depth["max_depth"]
    
    # Check that min_depth < max_depth
    assert noisy_depth["min_depth"] < noisy_depth["max_depth"]

def test_apply_differential_privacy_features():
    """Test applying differential privacy to feature vectors."""
    # Create a sample feature vector
    features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    # Apply differential privacy
    noisy_features = apply_differential_privacy(features, data_type="features")
    
    # Check that the shape is preserved
    assert noisy_features.shape == features.shape
    
    # Check that the values have changed
    assert not np.array_equal(noisy_features, features)

def test_apply_differential_privacy_detections():
    """Test applying differential privacy to object detection results."""
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
    
    # Apply differential privacy
    noisy_detections = apply_differential_privacy(detections, data_type="detections")
    
    # Check that the number of detections is preserved
    assert len(noisy_detections) == len(detections)
    
    # Check that the class IDs and names are preserved
    assert noisy_detections[0]["class_id"] == detections[0]["class_id"]
    assert noisy_detections[0]["class_name"] == detections[0]["class_name"] 