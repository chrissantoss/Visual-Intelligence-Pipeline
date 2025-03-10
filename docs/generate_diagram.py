#!/usr/bin/env python3
"""
Generate an architecture diagram image from the text-based diagram.

This script converts the text-based architecture diagram to a PNG image.

Usage:
    python generate_diagram.py
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_diagram():
    """Generate an architecture diagram image from the text-based diagram."""
    # Read the text-based diagram
    with open("docs/architecture.txt", "r") as f:
        diagram_text = f.read()
    
    # Calculate dimensions
    lines = diagram_text.split("\n")
    width = max(len(line) for line in lines) * 10  # 10 pixels per character
    height = len(lines) * 20  # 20 pixels per line
    
    # Create a blank image
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Try to load a monospace font
    try:
        font = ImageFont.truetype("Courier", 15)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Draw the diagram
    for i, line in enumerate(lines):
        draw.text((10, i * 20), line, fill=(0, 0, 0), font=font)
    
    # Save the image
    image.save("docs/architecture.png")
    print("Architecture diagram saved to docs/architecture.png")

if __name__ == "__main__":
    generate_diagram() 