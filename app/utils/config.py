import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Privacy Settings
    privacy_epsilon: float = Field(default=1.0, env="PRIVACY_EPSILON")
    privacy_delta: float = Field(default=1e-5, env="PRIVACY_DELTA")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic: str = Field(default="visual-data-stream", env="KAFKA_TOPIC")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="visual_intelligence", env="POSTGRES_DB")
    
    # Model Configuration
    object_detection_model: str = Field(default="yolov8n.pt", env="OBJECT_DETECTION_MODEL")
    segmentation_model: str = Field(default="deeplabv3_resnet101", env="SEGMENTATION_MODEL")
    depth_estimation_model: str = Field(default="DPT_Large", env="DEPTH_ESTIMATION_MODEL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings() 