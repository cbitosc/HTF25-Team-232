"""
Configuration management using environment variables
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List
import os

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # API Info
    api_title: str = "AI Traffic Violation Backend"
    api_version: str = "1.0.0"
    api_description: str = "Backend API for AI Traffic Violation Detection System"
    
    # Paths
    violation_root: str = "output/violations"
    logs_dir: str = "output/logs"
    fallback_log: str = "output/logs/fallback.json"
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:8501"
    
    # File limits
    max_file_size_mb: int = 100
    
    # Security
    api_key: str = "your-secret-api-key-here"
    enable_api_key: bool = False
    
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute"""
        return os.path.abspath(relative_path)

@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()
