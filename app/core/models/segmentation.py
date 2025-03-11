import os
import numpy as np
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import base64
import zlib
from loguru import logger
from typing import Dict, Any, List, Tuple

from app.utils.config import get_settings

class SegmentationModel:
    """DeepLabV3 semantic segmentation model."""
    
    def __init__(self, model, device):
        """Initialize the model."""
        self.model = model
        self.device = device
        self.settings = get_settings()
        
        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # COCO class names
        self.class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        logger.info(f"Initialized segmentation model: {self.settings.segmentation_model}")
    
    @classmethod
    async def create(cls):
        """Create a new instance of the model."""
        settings = get_settings()
        
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load DeepLabV3 model
            model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
            model.eval()
            model.to(device)
            
            return cls(model, device)
        except Exception as e:
            logger.error(f"Error loading segmentation model: {str(e)}")
            raise
    
    async def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: NumPy array containing the image data
            
        Returns:
            Dictionary with segmentation results
        """
        try:
            # Convert image to RGB if it's BGR (OpenCV format)
            if image.shape[2] == 3:
                image_rgb = image[..., ::-1].copy() if image.shape[2] == 3 else image.copy()
            else:
                image_rgb = image.copy()
            
            # Preprocess image
            input_tensor = self.preprocess(image_rgb)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_batch)["out"][0]
            
            # Get segmentation mask
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Get unique class IDs in the mask
            unique_class_ids = np.unique(output_predictions)
            unique_class_ids = unique_class_ids[unique_class_ids != 0]  # Remove background
            
            # Get class names for the unique class IDs
            class_names = [self.class_names[class_id] for class_id in unique_class_ids]
            
            # Compress and encode mask
            compressed_mask = zlib.compress(output_predictions.tobytes())
            mask_base64 = base64.b64encode(compressed_mask).decode("utf-8")
            
            return {
                "class_ids": unique_class_ids.tolist(),
                "class_names": class_names,
                "mask_base64": mask_base64,
            }
        
        except Exception as e:
            logger.error(f"Error in segmentation: {str(e)}")
            raise 