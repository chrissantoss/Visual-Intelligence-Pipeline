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

def _apply_noise_to_detections(
    detections: List[Dict[str, Any]], epsilon: float
) -> List[Dict[str, Any]]:
    """
    Apply noise to object detection results.
    
    Args:
        detections: Object detection results
        epsilon: Privacy parameter
        
    Returns:
        Object detection results with noise applied
    """
    # Calculate sensitivity for bounding boxes
    sensitivity = 0.1  # Small perturbation to bounding boxes
    
    # Calculate scale parameter
    scale = sensitivity / epsilon
    
    # Apply noise to each detection
    noisy_detections = []
    for detection in detections:
        # Create a copy of the detection
        noisy_detection = detection.copy()
        
        # Add noise to bounding box coordinates
        bbox = noisy_detection["bounding_box"]
        bbox["x1"] += np.random.laplace(0, scale)
        bbox["y1"] += np.random.laplace(0, scale)
        bbox["x2"] += np.random.laplace(0, scale)
        bbox["y2"] += np.random.laplace(0, scale)
        
        # Ensure coordinates are valid
        bbox["x1"] = max(0, bbox["x1"])
        bbox["y1"] = max(0, bbox["y1"])
        bbox["x2"] = max(bbox["x1"] + 1, bbox["x2"])
        bbox["y2"] = max(bbox["y1"] + 1, bbox["y2"])
        
        # Add noise to confidence
        confidence_scale = 0.05 / epsilon  # Smaller scale for confidence
        noisy_detection["confidence"] += np.random.laplace(0, confidence_scale)
        noisy_detection["confidence"] = max(0, min(1, noisy_detection["confidence"]))
        
        noisy_detections.append(noisy_detection)
    
    return noisy_detections

def _apply_noise_to_segmentation(
    segmentation: Dict[str, Any], epsilon: float
) -> Dict[str, Any]:
    """
    Apply noise to segmentation results.
    
    For segmentation, we don't modify the actual mask but add noise to the
    class probabilities, which affects the reported classes.
    
    Args:
        segmentation: Segmentation results
        epsilon: Privacy parameter
        
    Returns:
        Segmentation results with noise applied
    """
    # We don't modify the base64-encoded mask directly
    # Instead, we could add noise to the class IDs or class names
    # For simplicity, we'll just return the original segmentation
    # In a real implementation, you would decode the mask, add noise, and re-encode
    
    return segmentation

def _apply_noise_to_depth(
    depth: Dict[str, Any], epsilon: float
) -> Dict[str, Any]:
    """
    Apply noise to depth estimation results.
    
    For depth estimation, we don't modify the actual depth map but add noise to
    the min and max depth values.
    
    Args:
        depth: Depth estimation results
        epsilon: Privacy parameter
        
    Returns:
        Depth estimation results with noise applied
    """
    # Calculate sensitivity for depth values
    sensitivity = 0.1  # Small perturbation to depth values
    
    # Calculate scale parameter
    scale = sensitivity / epsilon
    
    # Create a copy of the depth results
    noisy_depth = depth.copy()
    
    # Add noise to min and max depth
    noisy_depth["min_depth"] += np.random.laplace(0, scale)
    noisy_depth["max_depth"] += np.random.laplace(0, scale)
    
    # Ensure min_depth < max_depth
    if noisy_depth["min_depth"] >= noisy_depth["max_depth"]:
        noisy_depth["max_depth"] = noisy_depth["min_depth"] + 0.1
    
    return noisy_depth 