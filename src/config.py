"""Configuration centrale du projet.

Charge les secrets depuis `.env` (pydantic-settings) et les paramètres
non sensibles depuis `config.yaml`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = ROOT_DIR / "config.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    chroma_persist_dir: str = Field(default="./data/vector_db", alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")

    mistral_ocr_model: str = Field(default="mistral-ocr-latest", alias="MISTRAL_OCR_MODEL")
    mistral_vlm_model: str = Field(default="pixtral-large-latest", alias="MISTRAL_VLM_MODEL")
    mistral_llm_model: str = Field(default="mistral-large-latest", alias="MISTRAL_LLM_MODEL")

    ban_api_url: str = Field(default="https://api-adresse.data.gouv.fr", alias="BAN_API_URL")
    gpu_api_url: str = Field(default="https://apicarto.ign.fr/api/gpu", alias="GPU_API_URL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_yaml_config() -> dict[str, Any]:
    if not CONFIG_YAML_PATH.exists():
        return {}
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
