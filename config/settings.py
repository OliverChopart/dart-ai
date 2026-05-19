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

    # Vision — YOLO model
    yolo_model_path: str = "models/yolo11n.pt"
    yolo_num_classes: int = 5              # dart + cal_20 + cal_6 + cal_3 + cal_11
    detection_confidence: float = 0.35     # lowered from 0.5 to improve cal_3 detection recall
    yolo_cal_confidence: float = 0.3       # lowered from 0.4 for same reason
    detection_device: str = "mps"

    # Homography
    homography_output_size: int = 800
    homography_fifo_size: int = 5
    homography_fifo_min_hits: int = 3

    # Game
    max_players: int = 4


# Single instance used across the entire application
settings = Settings()
