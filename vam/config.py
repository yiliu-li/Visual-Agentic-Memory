from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Visual Agentic Memory"
    app_version: str = "0.1.0"
    websocket_path: str = "/ws/agent"
    
    # Core
    openrouter_api_key: str = ""
    llm_model: str = "google/gemini-3-flash-preview"
    embedding_model: str = "google/gemini-embedding-2-preview"
    
    # Settings
    vision_embedding_backend: str = "openrouter_gemini"
    frame_store_path: str = "data/frame_store.sqlite3"

    # Overrides
    openrouter_model_id: str = ""
    openrouter_model_id_main: str = ""
    openrouter_model_id_light: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_model: str = ""
    openrouter_embedding_dimensions: int = 0
    
    # Internal Defaults
    qwen3_vl_embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    qwen3_vl_device: str = "auto"
    self_qwen_embedding_api: str = ""
    self_qwen_embedding_key: str = ""
    enable_retrieval: bool = True

    # Video Processing & Filtering Presets
    video_fps: float = 1.0
    video_max_frames: Optional[int] = None
    video_laplacian_min: float = 20.0
    video_diff_threshold: float = 20.0
    video_ssim_threshold: float = 0.92
    video_hist_threshold: float = 0.0
    video_similarity_threshold: float = 0.88  # For Embedding deduplication
    video_event_threshold: float = 0.80       # Trigger VLM event analysis if similarity < this
    video_event_max_duration_s: float = 300.0

    # Layered memory retention
    memory_recent_window_s: float = 60 * 60.0
    memory_mid_window_s: float = 24 * 60 * 60.0
    memory_recent_min_gap_s: float = 1.0
    memory_mid_min_gap_s: float = 20.0
    memory_long_min_gap_s: float = 120.0
    memory_mid_max_side: int = 768
    memory_long_max_side: int = 512
    memory_mid_jpeg_quality: int = 70
    memory_long_jpeg_quality: int = 45

    # LLM Multi-modal limits
    llm_max_image_size_mb: float = 5.0
    llm_max_image_pixels: int = 1280 * 1280

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
