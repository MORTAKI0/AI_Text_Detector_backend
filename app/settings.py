import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",),
    }

    app_name: str = "PFA AI Text Detector"
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    database_url: str = "sqlite:///./pfa.db"
    model_dir: str = "./model"
    AI_THRESHOLD: float = float(os.getenv("AI_THRESHOLD", "0.10"))
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        default_origins = list(cls.model_fields["cors_origins"].default or [])
        if value is None:
            return default_origins
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return default_origins
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
            items = [item.strip() for item in raw.split(",") if item.strip()]
            return items or default_origins
        return value

    @field_validator("model_dir")
    @classmethod
    def normalize_model_dir(cls, value: str) -> str:
        path = os.path.abspath(os.path.expanduser(value))
        return os.path.normpath(path)


@lru_cache
def get_settings() -> Settings:
    return Settings()
