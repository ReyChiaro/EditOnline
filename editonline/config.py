from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by API adapters."""

    app_name: str = "EditOnline"
    storage_dir: Path = Field(default=Path("runtime"))

    default_base_model: str = "Qwen/Qwen-Image-Edit-2511"
    default_lora_model: str = "inference/pretrained_weights/qwenimage_mask_flow_demo.safetensors"
    default_lora_adapter_name: str = "default"
    default_device: str = "cuda:0"

    edit_api_base_url: str | None = None
    edit_api_key: str | None = None
    mask_api_base_url: str | None = None
    mask_api_key: str | None = None

    sam_checkpoint: Path = Path("mask_generator/model_hub/sam_vit_h_4b8939.pth")
    sam_model_type: str = "vit_h"
    sam_device: str = "cuda:0"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EDITONLINE_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

