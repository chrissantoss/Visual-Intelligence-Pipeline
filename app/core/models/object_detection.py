import os
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger
from typing import List, Dict, Any, Tuple

from app.utils.config import get_settings

class ObjectDetectionModel:
    """YOLOv8 object detection model."""
    
    def __init__(self, model):
        """Initialize the model."""
        self.model = model
        self.settings = get_settings()
        logger.info(f"Initialized object detection model: {self.settings.object_detection_model}")
    
    @classmethod
    async def create(cls):
        """Create a new instance of the model."""
        settings = get_settings()
        
        try:
            # Load YOLOv8 model
            model = YOLO(settings.object_detection_model)
            return cls(model)
        except Exception as e:
            logger.error(f"Error loading object detection model: {str(e)}")
            raise
    
    async def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: NumPy array containing the image data
            
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0].item())
                    
                    # Add detection
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    })
            
            return detections
        
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            raise 