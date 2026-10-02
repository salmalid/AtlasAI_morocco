"""Runtime configuration.

Every setting has a default that lets the platform boot and serve real
satellite analysis with an empty environment. Optional subsystems (database,
LLM assistant, neural backends) advertise their availability through the
`has_*` properties rather than failing at import time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, EnvSettingsSource, DotEnvSettingsSource

REPO_ROOT = Path(__file__).resolve().parents[3]


class _OriginsSource:
    def prepare_field_value(self, field_name, field, value, value_is_complex):
        if field_name == "cors_origins" and isinstance(value, str) and not value.lstrip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class _EnvSource(_OriginsSource, EnvSettingsSource):
    pass


class _DotEnvSource(_OriginsSource, DotEnvSettingsSource):
    pass


class Settings(BaseSettings):
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, _EnvSource(settings_cls),
                _DotEnvSource(settings_cls, env_file=dotenv_settings.env_file), file_secret_settings)

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", Path(".env")),
        env_prefix="ATLASAI_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # --- Earth Observation --------------------------------------------------
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    pc_subscription_key: str = ""
    #: Hard ceiling on pixels read per analysis, protecting the worker from
    #: an AOI the size of a continent. 4M px ~ 2000x2000.
    max_pixels: int = 4_000_000
    #: Ground sample distance in metres used when the caller does not pin one.
    default_resolution: int = 20

    # --- Cache --------------------------------------------------------------
    cache_dir: Path = Path(".cache")
    cache_ttl_seconds: int = 86_400

    # --- Database (optional) -------------------------------------------------
    database_url: str = ""

    # --- Assistant (optional) ------------------------------------------------
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    assistant_model: str = "gemini-2.0-flash"

    # --- Neural backends (optional) ------------------------------------------
    model_dir: Path = Path("models")
    torch_device: str = "auto"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def has_database(self) -> bool:
        return bool(self.database_url)

    @property
    def has_llm(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def cache_path(self) -> Path:
        path = self.cache_dir
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
