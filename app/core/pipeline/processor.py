import numpy as np
import cv2
from loguru import logger
from typing import Dict, Any, List
import time
import asyncio

from app.core.privacy import apply_differential_privacy
from app.utils.config import get_settings

async def process_image(image_data: bytes, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an image through the visual intelligence pipeline.
    
    Args:
        image_data: Raw image data as bytes
        models: Dictionary of loaded models
        
    Returns:
        Dictionary with processing results
    """
    settings = get_settings()
    
    try:
        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Process image with models
        tasks = [
            _run_object_detection(image, models["object_detection"]),
            _run_segmentation(image, models["segmentation"]),
            _run_depth_estimation(image, models["depth_estimation"]),
        ]
        
        # Run tasks concurrently
        object_detection_result, segmentation_result, depth_estimation_result = await asyncio.gather(*tasks)
        
        # Apply differential privacy to results
        noisy_detections = apply_differential_privacy(
            object_detection_result, data_type="detections"
        )
        noisy_segmentation = apply_differential_privacy(
            segmentation_result, data_type="segmentation"
        )
        noisy_depth = apply_differential_privacy(
            depth_estimation_result, data_type="depth"
        )
        
        # Combine results
        result = {
            "detected_objects": noisy_detections,
            "segmentation": noisy_segmentation,
            "depth_estimation": noisy_depth,
            "privacy_epsilon": settings.privacy_epsilon,
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise
    finally:
        # Ensure image data is deleted from memory
        if 'image' in locals():
            del image
        if 'nparr' in locals():
            del nparr

async def _run_object_detection(image: np.ndarray, model) -> List[Dict[str, Any]]:
    """Run object detection on an image."""
    try:
        return await model.predict(image)
    except Exception as e:
        logger.error(f"Error in object detection: {str(e)}")
        return []

async def _run_segmentation(image: np.ndarray, model) -> Dict[str, Any]:
    """Run semantic segmentation on an image."""
    try:
        return await model.predict(image)
    except Exception as e:
        logger.error(f"Error in segmentation: {str(e)}")
        return {
            "class_ids": [],
            "class_names": [],
            "mask_base64": "",
        }

async def _run_depth_estimation(image: np.ndarray, model) -> Dict[str, Any]:
    """Run depth estimation on an image."""
    try:
        return await model.predict(image)
    except Exception as e:
        logger.error(f"Error in depth estimation: {str(e)}")
        return {
            "depth_map_base64": "",
            "min_depth": 0.0,
            "max_depth": 1.0,
        } 