import numpy as np
from loguru import logger
from typing import Dict, Any, Union, List
import torch

from app.utils.config import get_settings

def apply_differential_privacy(
    data: Union[np.ndarray, List[Dict[str, Any]], Dict[str, Any]],
    data_type: str = "features",
) -> Union[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply differential privacy to data.
    
    Args:
        data: Data to apply differential privacy to
        data_type: Type of data (features, detections, segmentation, depth)
        
    Returns:
        Data with differential privacy applied
    """
    settings = get_settings()
    epsilon = settings.privacy_epsilon
    delta = settings.privacy_delta
    
    try:
        if data_type == "features":
            # Apply Laplace noise to feature vectors
            return _apply_laplace_noise(data, epsilon)
        
        elif data_type == "detections":
            # Apply noise to object detection results
            return _apply_noise_to_detections(data, epsilon)
        
        elif data_type == "segmentation":
            # Apply noise to segmentation mask
            return _apply_noise_to_segmentation(data, epsilon)
        
        elif data_type == "depth":
            # Apply noise to depth map
            return _apply_noise_to_depth(data, epsilon)
        
        else:
            logger.warning(f"Unknown data type: {data_type}, returning original data")
            return data
    
    except Exception as e:
        logger.error(f"Error applying differential privacy: {str(e)}")
        # In case of error, return original data
        return data

def _apply_laplace_noise(features: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Apply Laplace noise to feature vectors.
    
    Args:
        features: Feature vectors
        epsilon: Privacy parameter
        
    Returns:
        Feature vectors with Laplace noise applied
    """
    # Calculate sensitivity (L1 norm)
    sensitivity = 2.0  # Assuming normalized features in [-1, 1]
    
    # Calculate scale parameter
    scale = sensitivity / epsilon
    
    # Generate Laplace noise
    noise = np.random.laplace(0, scale, features.shape)
    
    # Add noise to features
    noisy_features = features + noise
    
    return noisy_features

def _apply_noise_to_detections(detections: List[Dict[str, Any]], epsilon: float) -> List[Dict[str, Any]]:
    """Apply Laplace noise to object detection results."""
    noisy_detections = []
    
    for detection in detections:
        # Deep copy the detection to avoid modifying the original
        noisy_detection = detection.copy()
        noisy_detection["bounding_box"] = detection["bounding_box"].copy()
        
        # Add noise to bounding box coordinates
        sensitivity = 10.0  # Assuming pixel coordinates can change by up to 10 pixels
        noise_scale = sensitivity / epsilon
        
        for key in ["x1", "y1", "x2", "y2"]:
            noise = np.random.laplace(0, noise_scale)
            noisy_detection["bounding_box"][key] = float(detection["bounding_box"][key] + noise)
        
        # Add noise to confidence score
        confidence_sensitivity = 0.1
        confidence_noise = np.random.laplace(0, confidence_sensitivity / epsilon)
        noisy_detection["confidence"] = max(0.0, min(1.0, float(detection["confidence"] + confidence_noise)))
        
        noisy_detections.append(noisy_detection)
    
    return noisy_detections

def _apply_noise_to_segmentation(segmentation: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
    """Apply Laplace noise to segmentation results."""
    # Deep copy the segmentation to avoid modifying the original
    noisy_segmentation = segmentation.copy()
    
    # Add noise to class probabilities (if available)
    if "class_probabilities" in segmentation:
        sensitivity = 0.1
        noise_scale = sensitivity / epsilon
        noisy_probabilities = []
        
        for prob in segmentation["class_probabilities"]:
            noise = np.random.laplace(0, noise_scale)
            noisy_prob = max(0.0, min(1.0, float(prob + noise)))
            noisy_probabilities.append(noisy_prob)
        
        noisy_segmentation["class_probabilities"] = noisy_probabilities
    
    return noisy_segmentation

def _apply_noise_to_depth(depth: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
    """Apply Laplace noise to depth estimation results."""
    # Deep copy the depth to avoid modifying the original
    noisy_depth = depth.copy()
    
    # Add noise to min/max depth values
    sensitivity = 1.0  # Assuming depth values can change by up to 1 unit
    noise_scale = sensitivity / epsilon
    
    for key in ["min_depth", "max_depth"]:
        if key in depth:
            noise = np.random.laplace(0, noise_scale)
            noisy_depth[key] = float(depth[key] + noise)
    
    return noisy_depth 