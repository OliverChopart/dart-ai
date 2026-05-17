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
    # Update yolo_model_path to the 5-class model once trained:
    #   models/dart_5class.pt
    # Until then, the 1-class model can still be used with manual calibration.
    yolo_model_path: str = "models/yolo11n.pt"
    yolo_num_classes: int = 5              # dart + cal_20 + cal_6 + cal_3 + cal_11
    detection_confidence: float = 0.5      # minimum confidence for dart tips
    yolo_cal_confidence: float = 0.4       # minimum confidence for cal points (lower = more recall)
    detection_device: str = "mps"

    # Homography
    homography_output_size: int = 800      # side length of canonical top-down board image
    homography_fifo_size: int = 5          # sliding window length for FIFO debouncing
    homography_fifo_min_hits: int = 3      # min frames a tip must appear in to be stable

    # Game
    max_players: int = 4


# Single instance used across the entire application
settings = Settings()
