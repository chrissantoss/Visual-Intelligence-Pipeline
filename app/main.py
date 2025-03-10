import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Set environment variables directly
os.environ["PRIVACY_EPSILON"] = "1.0"
os.environ["PRIVACY_DELTA"] = "0.00001"

from app.api.routes import router as api_router
from app.core.models import load_models
from app.utils.config import get_settings

# Configure logger
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level=os.getenv("LOG_LEVEL", "INFO"),
)

app = FastAPI(
    title="Apple Visual Intelligence",
    description="Privacy-preserving visual data processing system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup."""
    logger.info("Starting Apple Visual Intelligence API")
    settings = get_settings()
    
    # Load ML models
    await load_models()
    
    logger.info(f"API running in {'debug' if settings.debug else 'production'} mode")
    logger.info(f"Privacy epsilon set to {settings.privacy_epsilon}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Apple Visual Intelligence API")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Apple Visual Intelligence API",
        "docs": "/docs",
    } 