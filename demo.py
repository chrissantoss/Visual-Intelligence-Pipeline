#!/usr/bin/env python3
"""
Demo script for Apple Visual Intelligence project.
Shows object detection, segmentation, and depth estimation with privacy features.
"""

import asyncio
import cv2
import numpy as np
import base64
import requests
import matplotlib.pyplot as plt
import zlib
from PIL import Image, ImageDraw
import io

def create_demo_image():
    """Create a demo image with multiple objects."""
    # Create a 640x480 image with a light blue sky and green ground
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    image[:240, :] = [255, 230, 180]  # Sky
    image[240:, :] = [100, 180, 100]  # Ground
    
    # Add a house (brown rectangle with a triangle roof)
    cv2.rectangle(image, (100, 180), (250, 350), (70, 100, 150), -1)  # House body
    pts = np.array([[75, 180], [175, 100], [275, 180]], np.int32)
    cv2.fillPoly(image, [pts], (70, 100, 150))  # Roof
    
    # Add a tree (brown trunk and green circle)
    cv2.rectangle(image, (400, 250), (430, 350), (70, 100, 150), -1)  # Trunk
    cv2.circle(image, (415, 200), 60, (60, 160, 60), -1)  # Leaves
    
    # Add a car (blue rectangle with wheels)
    cv2.rectangle(image, (300, 380), (500, 440), (200, 0, 0), -1)  # Car body
    cv2.circle(image, (340, 440), 20, (50, 50, 50), -1)  # Left wheel
    cv2.circle(image, (460, 440), 20, (50, 50, 50), -1)  # Right wheel
    
    # Add a person (stick figure)
    cv2.circle(image, (150, 400), 15, (0, 0, 0), -1)  # Head
    cv2.line(image, (150, 415), (150, 460), (0, 0, 0), 3)  # Body
    cv2.line(image, (150, 430), (130, 450), (0, 0, 0), 3)  # Left arm
    cv2.line(image, (150, 430), (170, 450), (0, 0, 0), 3)  # Right arm
    cv2.line(image, (150, 460), (130, 480), (0, 0, 0), 3)  # Left leg
    cv2.line(image, (150, 460), (170, 480), (0, 0, 0), 3)  # Right leg
    
    return image

def visualize_results(image, results):
    """Visualize the detection, segmentation, and depth results."""
    # Convert image to PIL for drawing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Draw bounding boxes for detected objects
    for obj in results["detected_objects"]:
        bbox = obj["bounding_box"]
        draw.rectangle(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
            outline="red",
            width=2
        )
        # Draw label
        label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        draw.text((bbox["x1"], bbox["y1"] - 10), label, fill="red")
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Original image with detections
    plt.subplot(131)
    plt.imshow(pil_image)
    plt.title("Object Detection")
    plt.axis("off")
    
    # Segmentation visualization
    if "segmentation" in results:
        seg_data = results["segmentation"]
        if "mask_base64" in seg_data:
            # Decode and decompress segmentation mask
            mask_bytes = base64.b64decode(seg_data["mask_base64"])
            mask = np.frombuffer(zlib.decompress(mask_bytes), dtype=np.uint8)
            mask = mask.reshape(image.shape[:2])
            
            plt.subplot(132)
            plt.imshow(mask, cmap="tab20")
            plt.title("Segmentation")
            plt.axis("off")
    
    # Depth visualization
    if "depth_estimation" in results:
        depth_data = results["depth_estimation"]
        if "depth_map_base64" in depth_data:
            # Decode and decompress depth map
            depth_bytes = base64.b64decode(depth_data["depth_map_base64"])
            depth = np.frombuffer(zlib.decompress(depth_bytes), dtype=np.uint8)
            depth = depth.reshape(image.shape[:2])
            
            plt.subplot(133)
            plt.imshow(depth, cmap="plasma")
            plt.title("Depth Estimation")
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def load_sample_image():
    """Load a sample image from the COCO dataset."""
    # URL of a sample image from COCO dataset - street scene with cars and people
    url = "http://images.cocodataset.org/val2017/000000000285.jpg"
    
    try:
        # Download the image
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert to numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Resize to a reasonable size if needed
        if image.shape[1] > 800:
            scale = 800 / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        return image
    except Exception as e:
        print(f"Error loading sample image: {str(e)}")
        print("Falling back to synthetic image...")
        return create_demo_image()

async def main():
    """Run the demo."""
    print("Apple Visual Intelligence Demo")
    print("=============================")
    
    # Create or load image
    print("\nLoading image...")
    use_real_image = True  # Set to False to use synthetic image
    image = load_sample_image() if use_real_image else create_demo_image()
    
    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    
    # Send request to API
    print("\nSending image to API for analysis...")
    try:
        response = requests.post(
            "http://localhost:8000/api/analyze_scene",
            json={"image": base64_image},
            timeout=30
        )
        response.raise_for_status()
        results = response.json()
        
        # Print results
        print("\nResults:")
        print("--------")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Privacy epsilon: {results['privacy_epsilon']}")
        print("\nDetected objects:")
        for obj in results["detected_objects"]:
            print(f"- {obj['class_name']} (confidence: {obj['confidence']:.2f})")
        
        if "segmentation" in results:
            print("\nSegmented classes:")
            for class_name in results["segmentation"]["class_names"]:
                print(f"- {class_name}")
        
        if "depth_estimation" in results:
            depth_data = results["depth_estimation"]
            print("\nDepth estimation:")
            print(f"- Min depth: {depth_data['min_depth']:.2f}")
            print(f"- Max depth: {depth_data['max_depth']:.2f}")
        
        # Visualize results
        print("\nVisualizing results...")
        visualize_results(image, results)
        
    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to API: {str(e)}")
        print("Make sure the API server is running (python -m uvicorn app.main:app --reload)")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 