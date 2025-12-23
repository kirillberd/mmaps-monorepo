from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class PreprocessConfig(BaseModel):
    """Image preprocessing that mirrors the common torchvision pipeline:

    Resize -> ToTensor -> Normalize(mean, std)
    """

    image_size: int = 224
    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])


class TritonConfig(BaseModel):
    # Triton Inference Server URL, e.g. "localhost:8000" for HTTP.
    url: str
    model_name: str = "efficientnet"
    model_version: str | None = "1"
    input_name: str = "images"
    output_name: str = "embeddings"
    timeout_s: float = 10.0
    ssl: bool = False

    # pydantic v2 treats fields starting with "model_" as protected by default.
    model_config = {
        "protected_namespaces": (),
    }


class OpenSearchConfig(BaseModel):
    host: str
    port: int
    index: str

    # Secrets should be provided via env vars, see config.yml (${VAR}).
    username: str | None = None
    password: str | None = None

    use_ssl: bool = False
    verify_certs: bool = False
    http_compress: bool = True

    # OpenSearch client-level socket/read timeout.
    timeout_s: int = 600
    # Per-search request timeout.
    request_timeout_s: int = 60

    # HNSW ef_search for KNN queries.
    ef_search: int = 128

    # What fields to return from _source. If empty/null -> full _source.
    source_includes: list[str] | None = None


class SearchConfig(BaseModel):
    default_k: int = 10
    max_k: int = 100
    # Using innerproduct in OpenSearch + L2-normalization makes it equivalent to cosine similarity.
    l2_normalize: bool = True


class Settings(BaseSettings):
    """App settings.

    Loads the base config from YAML and substitutes secrets from env vars.
    """

    server: ServerConfig
    preprocess: PreprocessConfig
    triton: TritonConfig
    opensearch: OpenSearchConfig
    search: SearchConfig = SearchConfig()

    # You can still override settings from env if you want, but for secrets we prefer
    # YAML placeholders like ${OPENSEARCH_PASSWORD}.
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        with open(path, "rb") as f:
            yaml_data = yaml.safe_load(f) or {}
        yaml_data = _replace_env_vars(yaml_data)
        return cls(**yaml_data)


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _replace_env_vars(obj: Any) -> Any:
    """Recursively substitutes ${ENV_VAR} or ${ENV_VAR:default} inside YAML values."""

    if isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(_env_replacer, obj)
    return obj


def _env_replacer(match: re.Match[str]) -> str:
    var_name = match.group(1)
    default = match.group(2)
    value = os.getenv(var_name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(
        f"Environment variable '{var_name}' is required by config but not set",
    )
