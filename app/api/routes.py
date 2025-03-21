from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Body, Request
from fastapi.responses import JSONResponse
import base64
from typing import Optional, List, Dict, Any
import io
import time
from loguru import logger
import json

from app.core.pipeline import process_image
from app.core.models import get_models
from app.utils.config import get_settings
from app.api.schemas import (
    SceneAnalysisRequest,
    SceneAnalysisResponse,
    SceneSummaryResponse,
    ErrorResponse,
)

router = APIRouter()

@router.post(
    "/analyze_scene",
    response_model=SceneAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_scene(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    base64_image: Optional[SceneAnalysisRequest] = None,
):
    """
    Analyze a scene from an image.
    
    The image can be provided either as a file upload or as a base64-encoded string.
    Raw image data is never stored or logged, preserving user privacy.
    """
    start_time = time.time()
    
    logger.info(f"Received analyze_scene request: file={file is not None}, base64_image={base64_image is not None}")
    
    try:
        # Get image data
        if file:
            logger.info(f"Processing file: {file.filename}")
            image_data = await file.read()
        elif base64_image:
            logger.info("Processing base64 image")
            try:
                image_data = base64.b64decode(base64_image.image)
            except Exception as e:
                logger.error(f"Error decoding base64 image: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid base64 image data",
                )
        else:
            # Try to get the image from the request body directly
            logger.info("Attempting to get image from request body directly")
            try:
                body = await request.json()
                if "image" in body:
                    image_data = base64.b64decode(body["image"])
                else:
                    raise ValueError("No image field in request body")
            except Exception as e:
                logger.error(f"Error parsing request body: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail="No image provided. Please upload a file or provide a base64-encoded image.",
                )
        
        # Process the image
        try:
            models = get_models()
            result = await process_image(image_data, models)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add processing time to the response
            result["processing_time"] = processing_time
            
            return result
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )

# Add a simple test endpoint for debugging
@router.post("/test_image")
async def test_image(image_base64: str = Body(..., embed=True)):
    """
    Test endpoint for debugging image upload.
    
    Simply returns the length of the base64-encoded image string.
    """
    try:
        # Decode the base64 string to verify it's valid
        image_data = base64.b64decode(image_base64)
        return {
            "success": True,
            "image_length": len(image_base64),
            "decoded_length": len(image_data),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

@router.get(
    "/scene_summary",
    response_model=SceneSummaryResponse,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def scene_summary():
    """
    Get aggregated, anonymized scene data.
    
    Returns summary statistics about recently processed scenes without exposing
    any raw visual data.
    """
    try:
        # In a real implementation, this would query the database for aggregated data
        # For now, we'll return mock data
        return {
            "total_scenes_processed": 10,
            "object_counts": {
                "person": 15,
                "chair": 8,
                "table": 3,
                "car": 2,
            },
            "average_objects_per_scene": 2.8,
            "timestamp": time.time(),
        }
    
    except Exception as e:
        logger.error(f"Error retrieving scene summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving scene summary: {str(e)}",
        ) 