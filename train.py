"""Train and save directional phonetic cost matrices."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

from load_datasets import TransliterationPair, load_datasets
from utils.dataset_config import parse_training_args, read_settings
from utils.training import TrainingResult, train_cost_matrix


DIRECTIONS = ("english_to_target", "target_to_english")


def _training_records(
    records: list[TransliterationPair], fraction: float, seed: int, language: str
) -> list[TransliterationPair]:
    Random(f"{seed}:{language}").shuffle(records)
    return records[: max(1, int(len(records) * fraction))]


def _directional_pairs(
    records: list[TransliterationPair], direction: str
) -> list[tuple[str, str]]:
    if direction == "english_to_target":
        return [(record.source, record.target) for record in records]
    return [(record.target, record.source) for record in records]


def _output_path(output_dir: Path, language: str, direction: str) -> Path:
    if direction == "english_to_target":
        return output_dir / f"english_to_{language}.json"
    return output_dir / f"{language}_to_english.json"


def _save_result(
    path: Path,
    result: TrainingResult,
    language: str,
    direction: str,
    pair_count: int,
) -> None:
    source_language, target_language = (
        ("english", language)
        if direction == "english_to_target"
        else (language, "english")
    )
    payload = {
        "source_language": source_language,
        "target_language": target_language,
        "source_symbols": result.source_symbols,
        "target_symbols": result.target_symbols,
        "costs": result.costs.tolist(),
        "training_pairs": pair_count,
        "iterations": result.iterations,
        "converged": result.converged,
        "total_cost": result.total_cost,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_training_args()
    settings = read_settings(args.config)
    languages = tuple(args.languages) if args.languages else settings.languages
    directions = tuple(args.directions) if args.directions else DIRECTIONS

    for language in languages:
        records = load_datasets(args.config, [language])[language]
        training_records = _training_records(
            records, settings.train_fraction, settings.random_seed, language
        )

        for direction in directions:
            print(f"{language} {direction}: {len(training_records):,} pairs")
            result = train_cost_matrix(
                _directional_pairs(training_records, direction),
                max_iterations=settings.max_iterations,
                tolerance=settings.convergence_tolerance,
                on_iteration=lambda iteration, normalized_cost: print(
                    f"  iteration {iteration}: "
                    f"row_normalized_cost={normalized_cost:.6f}"
                ),
            )
            path = _output_path(settings.output_dir, language, direction)
            _save_result(path, result, language, direction, len(training_records))
            print(f"  saved: {path}")


if __name__ == "__main__":
    main()
