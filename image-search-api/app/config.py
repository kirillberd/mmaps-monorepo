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
