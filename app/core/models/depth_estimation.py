import os
import numpy as np
import torch
import cv2
import base64
import zlib
from loguru import logger
from typing import Dict, Any
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

from app.utils.config import get_settings

class DepthEstimationModel:
    """MiDaS depth estimation model."""
    
    def __init__(self, model, feature_extractor, device):
        """Initialize the model."""
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.settings = get_settings()
        logger.info(f"Initialized depth estimation model: {self.settings.depth_estimation_model}")
    
    @classmethod
    async def create(cls):
        """Create a new instance of the model."""
        settings = get_settings()
        
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load MiDaS model
            model_name = "Intel/dpt-large"  # Use DPT-Large for best quality
            feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
            model = DPTForDepthEstimation.from_pretrained(model_name)
            
            model.eval()
            model.to(device)
            
            return cls(model, feature_extractor, device)
        except Exception as e:
            logger.error(f"Error loading depth estimation model: {str(e)}")
            raise
    
    async def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate depth from an image.
        
        Args:
            image: NumPy array containing the image data
            
        Returns:
            Dictionary with depth estimation results
        """
        try:
            # Convert image to RGB if it's BGR (OpenCV format)
            if image.shape[2] == 3:
                image_rgb = image[..., ::-1] if image.shape[2] == 3 else image
            else:
                image_rgb = image
            
            # Prepare image for the model
            inputs = self.feature_extractor(images=image_rgb, return_tensors="pt").to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            
            # Normalize depth map for visualization
            depth_min = prediction.min()
            depth_max = prediction.max()
            normalized_depth = (prediction - depth_min) / (depth_max - depth_min) * 255.0
            normalized_depth = normalized_depth.astype(np.uint8)
            
            # Compress and encode depth map
            compressed_depth = zlib.compress(normalized_depth.tobytes())
            depth_base64 = base64.b64encode(compressed_depth).decode("utf-8")
            
            return {
                "depth_map_base64": depth_base64,
                "min_depth": float(depth_min),
                "max_depth": float(depth_max),
            }
        
        except Exception as e:
            logger.error(f"Error in depth estimation: {str(e)}")
            raise 