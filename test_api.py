#!/usr/bin/env python3
"""
Simple test script to test the API directly.
"""

import requests
import base64
import cv2
import numpy as np
import json

# Create a simple test image
def create_test_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (0, 255, 0), -1)
    return img

# Encode image to base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Test the test_image endpoint
def test_test_image_endpoint():
    print("Testing test_image endpoint...")
    
    # Create a test image
    image = create_test_image()
    base64_image = encode_image_to_base64(image)
    
    # Send request to test_image endpoint
    response = requests.post(
        "http://127.0.0.1:8000/api/test_image",
        json={"image_base64": base64_image},
    )
    
    print(f"Response status code: {response.status_code}")
    print(f"Response: {response.json()}")

# Test the analyze_scene endpoint
def test_analyze_scene_endpoint():
    print("Testing analyze_scene endpoint...")
    
    # Create a test image
    image = create_test_image()
    base64_image = encode_image_to_base64(image)
    
    # Create the request payload
    payload = {
        "base64_image": {
            "image": base64_image
        }
    }
    
    # Print the payload for debugging
    print(f"Payload: {json.dumps(payload)[:100]}...")
    
    # Send request to analyze_scene endpoint
    response = requests.post(
        "http://127.0.0.1:8000/api/analyze_scene",
        json=payload,
    )
    
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_test_image_endpoint()
    print("\n" + "=" * 50 + "\n")
    test_analyze_scene_endpoint() 