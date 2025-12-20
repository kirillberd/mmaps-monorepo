from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pathlib import Path
import yaml


class ServerConfig(BaseModel):
    host: str
    port: int
    reload: bool


class Settings(BaseSettings):
    server: ServerConfig

    model_config = SettingsConfigDict()

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "rb") as f:
            yaml_data = yaml.safe_load(f)
        return cls(**yaml_data)


@lru_cache
def get_settings() -> Settings:
    path = Path(__file__).resolve().parent.parent / "config" / "config.yml"

    return Settings.from_yaml(Path(path))
