import pytest
from fastapi.testclient import TestClient
import base64
import os
import cv2
import numpy as np
from pathlib import Path
from app.main import app
from app.core.models import load_models

# Create test client
client = TestClient(app)

@pytest.fixture(autouse=True, scope="module")
async def setup_models():
    """Load models before running tests."""
    await load_models()

def get_test_image():
    """Get a test image for API testing."""
    # Create a simple test image (100x100 with a rectangle)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (75, 75), (0, 255, 0), -1)
    
    # Encode image to bytes
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    
    return image_bytes

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs" in response.json()

def test_analyze_scene_with_file():
    """Test the analyze_scene endpoint with a file upload."""
    # Get test image
    image_bytes = get_test_image()
    
    # Create a test file
    files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
    
    # Make request
    response = client.post("/api/analyze_scene", files=files)
    
    # Check response
    assert response.status_code == 200
    assert "detected_objects" in response.json()
    assert "processing_time" in response.json()
    assert "privacy_epsilon" in response.json()

def test_analyze_scene_with_base64():
    """Test the analyze_scene endpoint with a base64-encoded image."""
    # Get test image and encode as base64
    image_bytes = get_test_image()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Make request
    response = client.post(
        "/api/analyze_scene",
        json={"image": base64_image},
    )
    
    # Check response
    assert response.status_code == 200
    assert "detected_objects" in response.json()
    assert "processing_time" in response.json()
    assert "privacy_epsilon" in response.json()

def test_analyze_scene_without_image():
    """Test the analyze_scene endpoint without an image."""
    # Make request without an image
    response = client.post("/api/analyze_scene")
    
    # Check response
    assert response.status_code == 400
    assert "detail" in response.json()

def test_scene_summary():
    """Test the scene_summary endpoint."""
    # Make request
    response = client.get("/api/scene_summary")
    
    # Check response
    assert response.status_code == 200
    assert "total_scenes_processed" in response.json()
    assert "object_counts" in response.json()
    assert "average_objects_per_scene" in response.json()
    assert "timestamp" in response.json() 