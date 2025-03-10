"""Models package for the Apple Visual Intelligence system."""

from app.core.models.object_detection import ObjectDetectionModel
from app.core.models.segmentation import SegmentationModel
from app.core.models.depth_estimation import DepthEstimationModel

# Global models dictionary
_MODELS = None

async def load_models():
    """Load all models."""
    global _MODELS
    
    # Initialize models
    object_detection = await ObjectDetectionModel.create()
    segmentation = await SegmentationModel.create()
    depth_estimation = await DepthEstimationModel.create()
    
    # Store models in global dictionary
    _MODELS = {
        "object_detection": object_detection,
        "segmentation": segmentation,
        "depth_estimation": depth_estimation,
    }
    
    return _MODELS

def get_models():
    """Get loaded models."""
    global _MODELS
    
    if _MODELS is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    
    return _MODELS 