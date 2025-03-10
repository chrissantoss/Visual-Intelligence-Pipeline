from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time

class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str

class SceneAnalysisRequest(BaseModel):
    """Request schema for scene analysis."""
    image: Optional[str] = Field(
        None,
        description="Base64-encoded image data",
    )

class BoundingBox(BaseModel):
    """Bounding box schema for object detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    
class DetectedObject(BaseModel):
    """Detected object schema."""
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox

class SegmentationResult(BaseModel):
    """Semantic segmentation result schema."""
    class_ids: List[int]
    class_names: List[str]
    mask_base64: str = Field(
        description="Base64-encoded segmentation mask (compressed)",
    )

class DepthEstimationResult(BaseModel):
    """Depth estimation result schema."""
    depth_map_base64: str = Field(
        description="Base64-encoded depth map (compressed)",
    )
    min_depth: float
    max_depth: float

class SceneAnalysisResponse(BaseModel):
    """Response schema for scene analysis."""
    detected_objects: List[DetectedObject]
    segmentation: Optional[SegmentationResult] = None
    depth_estimation: Optional[DepthEstimationResult] = None
    processing_time: float
    timestamp: float = Field(default_factory=time.time)
    privacy_epsilon: float = Field(
        description="Differential privacy parameter used for this analysis",
    )

class SceneSummaryResponse(BaseModel):
    """Response schema for scene summary."""
    total_scenes_processed: int
    object_counts: Dict[str, int]
    average_objects_per_scene: float
    timestamp: float = Field(default_factory=time.time) 