# Visual Intelligence Demo

This project demonstrates privacy-preserving visual intelligence capabilities including object detection, semantic segmentation, and depth estimation.

## Features

- Object Detection: Identifies and localizes objects in images
- Semantic Segmentation: Provides pixel-level classification of image content
- Depth Estimation: Generates depth maps from single images
- Privacy Protection: Applies differential privacy to protect sensitive information

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model:
```bash
python download_sample_images.py
```

## Running the Demo

1. Start the API server:
```bash
python -m uvicorn app.main:app --reload
```

2. In a new terminal, run the demo script:
```bash
python demo.py
```

The demo will:
- Create a sample image with multiple objects
- Send it to the API for analysis
- Display the results including:
  - Object detection with bounding boxes
  - Semantic segmentation mask
  - Depth estimation map
  - Privacy metrics

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Privacy Features

The system implements differential privacy to protect sensitive information:
- Adds calibrated noise to object detection results
- Protects exact locations and dimensions
- Configurable privacy level (epsilon parameter)
- No raw image data is stored

## Example Output

The demo will show:
1. Console output with detected objects and metrics
2. Visualization window with three panels:
   - Left: Original image with detected objects
   - Middle: Semantic segmentation visualization
   - Right: Depth map visualization
