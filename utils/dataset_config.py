"""Read dataset configuration and command-line options."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import tomllib


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "dataset_config.toml"


class DatasetError(ValueError):
    """Raised when configuration or a token-file row is invalid."""


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Resolved settings read from dataset_config.toml."""

    languages: tuple[str, ...]
    roots: tuple[Path, ...]
    language_roots: dict[str, Path]
    file_pattern: str
    encoding: str
    output_dir: Path
    train_fraction: float
    random_seed: int
    max_iterations: int
    convergence_tolerance: float


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def read_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> DatasetSettings:
    """Read and validate settings, resolving paths from the config file."""

    path = Path(config_path).resolve()
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)

    try:
        data = config["data"]
        training = config["training"]
        languages = tuple(data["languages"])
        root_values = tuple(data["roots"])
        file_pattern = data["file_pattern"]
    except (KeyError, TypeError) as exc:
        raise DatasetError(f"Invalid dataset config in {path}: missing {exc}") from exc

    if not languages or not all(isinstance(value, str) and value for value in languages):
        raise DatasetError("data.languages must be a non-empty list of strings")
    if not root_values or not all(isinstance(value, str) and value for value in root_values):
        raise DatasetError("data.roots must be a non-empty list of paths")
    if not isinstance(file_pattern, str) or "{language}" not in file_pattern:
        raise DatasetError("data.file_pattern must contain the {language} placeholder")
    if not file_pattern.endswith(".tokens"):
        raise DatasetError("data.file_pattern must select only .tokens files")
    pattern_without_placeholder = file_pattern.replace("{language}", "")
    if any(char in pattern_without_placeholder for char in "*?[]"):
        raise DatasetError("data.file_pattern must not contain wildcard characters")

    config_dir = path.parent
    roots = tuple(_resolve_path(value, config_dir) for value in root_values)
    raw_language_roots = data.get("language_roots", {})
    if not isinstance(raw_language_roots, dict):
        raise DatasetError("data.language_roots must be a table of language/path pairs")
    language_roots = {
        language: _resolve_path(root, config_dir)
        for language, root in raw_language_roots.items()
    }

    return DatasetSettings(
        languages=languages,
        roots=roots,
        language_roots=language_roots,
        file_pattern=file_pattern,
        encoding=data.get("encoding", "utf-8"),
        output_dir=_resolve_path(training["output_dir"], config_dir),
        train_fraction=float(training["train_fraction"]),
        random_seed=int(training["random_seed"]),
        max_iterations=int(training["max_iterations"]),
        convergence_tolerance=float(training["convergence_tolerance"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse loader command-line options."""

    parser = argparse.ArgumentParser(
        description="Load configured wd_<language>.normalized.aligned.tokens datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"TOML config path (default: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Language/alphabet to load; repeat to load several. Overrides config.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N processed pairs for each language.",
    )
    return parser.parse_args(argv)


def parse_training_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse cost-matrix training options."""

    parser = argparse.ArgumentParser(description="Train phonetic cost matrices.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument(
        "--direction",
        action="append",
        choices=("english_to_target", "target_to_english"),
        dest="directions",
    )
    return parser.parse_args(argv)
