#!/usr/bin/env python3
"""
Test imports to identify any issues.
"""

try:
    print("Testing imports...")
    
    # Test FastAPI imports
    print("Importing FastAPI...")
    from fastapi import FastAPI
    print("FastAPI imported successfully.")
    
    # Test app imports
    print("Importing app modules...")
    from app.main import app
    print("app.main imported successfully.")
    
    print("All imports successful!")
except Exception as e:
    print(f"Error importing modules: {str(e)}") 