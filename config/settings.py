"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the dart-ai system.

    Values are read from the .env file or environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://localhost/dartai"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Camera
    camera_source: str = "0"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # Vision
    yolo_model_path: str = "models/yolo11n.pt"
    detection_confidence: float = 0.5
    detection_device: str = "mps"

    # Game
    max_players: int = 4


# Single instance used across the entire application
settings = Settings()
