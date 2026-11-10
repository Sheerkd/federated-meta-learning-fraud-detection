"""Validated configuration with project-local storage boundaries."""

from pathlib import Path
from typing import Literal
import tomllib

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Paths(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: str = "data"
    clients: str = "clients"
    runs: str = "runs"

    @field_validator("data", "clients", "runs")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = Path(value)
        if not value.strip() or path.is_absolute() or path.drive or ".." in path.parts or path == Path("."):
            raise ValueError("Storage paths must be nonempty project-relative subdirectories")
        return value


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    device: Literal["cpu", "cuda"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    client_ids: list[str] = Field(default_factory=lambda: ["bank_a", "bank_b", "bank_c", "bank_d"])
    paths: Paths = Field(default_factory=Paths)

    @field_validator("client_ids")
    @classmethod
    def four_clients(cls, value: list[str]) -> list[str]:
        if sorted(value) != ["bank_a", "bank_b", "bank_c", "bank_d"]:
            raise ValueError("Exactly bank_a, bank_b, bank_c and bank_d are required")
        return value


def load_config(path: Path) -> AppConfig:
    with path.open("rb") as stream:
        return AppConfig.model_validate(tomllib.load(stream))


def initialize_storage(root: Path, config: AppConfig) -> list[Path]:
    root = root.resolve()
    targets = [root / config.paths.data / "raw", root / config.paths.runs]
    targets.extend(root / config.paths.clients / bank / part
                   for bank in config.client_ids
                   for part in ("data", "preprocessing", "models", "explanations"))
    # Validate every target before creating anything, including existing symlinks.
    for target in targets:
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Storage escapes project root: {target}")
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
    return targets
