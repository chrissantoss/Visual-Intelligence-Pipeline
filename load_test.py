#!/usr/bin/env python3
"""
Load testing script for the Apple Visual Intelligence system.

This script tests the system under high load by sending multiple concurrent requests.

Usage:
    python load_test.py --num-clients 10 --num-requests 100
"""

import os
import sys
import time
import asyncio
import argparse
import aiohttp
import base64
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("load_test")

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with some shapes for testing."""
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a green rectangle
    cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)
    
    # Add a red circle
    cv2.circle(image, (450, 200), 80, (0, 0, 255), -1)
    
    # Add a blue triangle
    pts = np.array([[200, 400], [300, 350], [400, 400]], np.int32)
    cv2.fillPoly(image, [pts], (255, 0, 0))
    
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode an image to base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

async def send_request(session: aiohttp.ClientSession, base64_image: str, request_id: int) -> Dict[str, Any]:
    """Send a request to the API."""
    start_time = time.time()
    
    try:
        async with session.post(
            f"{API_URL}/api/analyze_scene",
            json={"image": base64_image},
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "latency": latency,
                "num_objects": len(result["detected_objects"]),
                "processing_time": result["processing_time"],
            }
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        return {
            "request_id": request_id,
            "success": False,
            "latency": latency,
            "error": str(e),
        }

async def client(client_id: int, num_requests: int, base64_image: str, results: List[Dict[str, Any]]):
    """Simulate a client sending multiple requests."""
    logger.info(f"Client {client_id} starting with {num_requests} requests")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            request_id = client_id * num_requests + i
            result = await send_request(session, base64_image, request_id)
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Client {client_id} completed {i + 1}/{num_requests} requests")
    
    logger.info(f"Client {client_id} completed all {num_requests} requests")

async def run_load_test(num_clients: int, num_requests: int):
    """Run the load test with multiple clients."""
    logger.info(f"Starting load test with {num_clients} clients, {num_requests} requests per client")
    
    # Create a test image
    image = create_test_image()
    base64_image = encode_image_to_base64(image)
    
    # Create shared results list
    results = []
    
    # Start clients
    start_time = time.time()
    
    tasks = [
        client(i, num_requests, base64_image, results)
        for i in range(num_clients)
    ]
    
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    total_requests = num_clients * num_requests
    successful_requests = sum(1 for r in results if r["success"])
    failed_requests = total_requests - successful_requests
    
    if successful_requests > 0:
        latencies = [r["latency"] for r in results if r["success"]]
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        processing_times = [r["processing_time"] for r in results if r["success"] and "processing_time" in r]
        if processing_times:
            avg_processing_time = statistics.mean(processing_times)
        else:
            avg_processing_time = 0
    else:
        avg_latency = min_latency = max_latency = p95_latency = avg_processing_time = 0
    
    throughput = total_requests / total_time
    
    # Print results
    logger.info("\n\n" + "=" * 50)
    logger.info("LOAD TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Successful requests: {successful_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    logger.info(f"Success rate: {successful_requests / total_requests * 100:.2f}%")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Throughput: {throughput:.2f} requests/second")
    
    if successful_requests > 0:
        logger.info(f"Average latency: {avg_latency:.3f}s")
        logger.info(f"Min latency: {min_latency:.3f}s")
        logger.info(f"Max latency: {max_latency:.3f}s")
        logger.info(f"95th percentile latency: {p95_latency:.3f}s")
        logger.info(f"Average processing time: {avg_processing_time:.3f}s")
    
    logger.info("=" * 50)
    
    # Save results to file
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/load_test_results.txt", "w") as f:
        f.write("LOAD TEST RESULTS\n")
        f.write(f"Total requests: {total_requests}\n")
        f.write(f"Successful requests: {successful_requests}\n")
        f.write(f"Failed requests: {failed_requests}\n")
        f.write(f"Success rate: {successful_requests / total_requests * 100:.2f}%\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Throughput: {throughput:.2f} requests/second\n")
        
        if successful_requests > 0:
            f.write(f"Average latency: {avg_latency:.3f}s\n")
            f.write(f"Min latency: {min_latency:.3f}s\n")
            f.write(f"Max latency: {max_latency:.3f}s\n")
            f.write(f"95th percentile latency: {p95_latency:.3f}s\n")
            f.write(f"Average processing time: {avg_processing_time:.3f}s\n")
    
    logger.info("Results saved to test_results/load_test_results.txt")
    
    return successful_requests == total_requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run load tests for Apple Visual Intelligence")
    parser.add_argument("--api-url", help="API URL (default: http://localhost:8000)", default="http://localhost:8000")
    parser.add_argument("--num-clients", type=int, help="Number of concurrent clients", default=5)
    parser.add_argument("--num-requests", type=int, help="Number of requests per client", default=10)
    args = parser.parse_args()
    
    API_URL = args.api_url
    
    asyncio.run(run_load_test(args.num_clients, args.num_requests)) 