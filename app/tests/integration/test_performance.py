import pytest
import time
import asyncio
import numpy as np
import cv2
from typing import List, Dict, Any
import base64
import statistics

from app.core.pipeline.processor import process_image
from app.core.models import load_models, get_models

class TestPerformance:
    """Performance tests for the visual intelligence system."""
    
    @pytest.mark.asyncio
    async def test_latency(self):
        """Test the latency of the image processing pipeline."""
        # Load models
        await load_models()
        models = get_models()
        
        # Create a test image
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Measure latency for multiple runs
        latencies = []
        for _ in range(5):
            start_time = time.time()
            result = await process_image(image_bytes, models)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Log results
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"Min latency: {min_latency:.3f}s")
        print(f"Max latency: {max_latency:.3f}s")
        
        # Assert that latency is within acceptable range
        # Note: This threshold might need adjustment based on the hardware
        assert avg_latency < 10.0, "Average latency should be less than 10 seconds"
    
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test the throughput of the image processing pipeline."""
        # Load models
        await load_models()
        models = get_models()
        
        # Create a test image
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Number of images to process
        num_images = 10
        
        # Measure throughput
        start_time = time.time()
        
        # Process images concurrently
        tasks = [process_image(image_bytes, models) for _ in range(num_images)]
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput (images per second)
        throughput = num_images / total_time
        
        # Log results
        print(f"Throughput: {throughput:.2f} images/second")
        print(f"Total time for {num_images} images: {total_time:.3f}s")
        
        # Assert that throughput is within acceptable range
        # Note: This threshold might need adjustment based on the hardware
        assert throughput > 0.1, "Throughput should be greater than 0.1 images/second"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test that memory usage is reasonable and no leaks occur."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Load models
        await load_models()
        models = get_models()
        
        # Create a test image
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Measure memory before processing
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple images
        for _ in range(5):
            result = await process_image(image_bytes, models)
            # Force garbage collection
            import gc
            gc.collect()
        
        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate memory increase
        memory_increase = memory_after - memory_before
        
        # Log results
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after: {memory_after:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Assert that memory increase is within acceptable range
        # Note: This threshold might need adjustment based on the implementation
        assert memory_increase < 500, "Memory increase should be less than 500 MB" 