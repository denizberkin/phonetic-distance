"""Load normalized named-entity transliteration pairs from ``*.tokens`` files.

The token files contain three tab-separated fields per row:

    latin_source<TAB>space separated target characters<TAB>occurrence count

The configured language name is substituted into an exact token filename.
This makes the loader usable for every alphabet present in either configured
data directory, not only the four selected for the phonetic-distance study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import unicodedata

from utils.dataset_config import (
    DEFAULT_CONFIG_PATH,
    DatasetError,
    DatasetSettings,
    parse_args,
    read_settings,
)


@dataclass(frozen=True, slots=True)
class TransliterationPair:
    """One processed source/target transliteration pair."""

    language: str
    source: str
    target: str
    target_tokens: tuple[str, ...]
    count: int
    path: Path
    line_number: int


def find_token_files(language: str, settings: DatasetSettings) -> tuple[Path, ...]:
    """Find the exact ``.tokens`` file for one language/alphabet."""

    if not language or any(char in language for char in "\\/*?[]{}"):
        raise DatasetError(f"Invalid language/alphabet name: {language!r}")

    pattern = settings.file_pattern.format(language=language)
    preferred_root = settings.language_roots.get(language)
    roots = (preferred_root,) if preferred_root is not None else settings.roots

    for root in roots:
        if not root.is_dir():
            raise DatasetError(f"Dataset root does not exist: {root}")
        candidate = root / pattern
        if candidate.is_file() and candidate.suffix == ".tokens":
            return (candidate.resolve(),)

    searched = ", ".join(str(root) for root in roots)
    raise DatasetError(
        f"No .tokens file matched {pattern!r} for {language!r} in: {searched}"
    )


def iter_token_file(
    path: str | Path,
    language: str,
    *,
    encoding: str = "utf-8",
) -> Iterator[TransliterationPair]:
    """Parse single token file yield pairs."""

    token_path = Path(path).resolve()
    skipped = 0
    with token_path.open("r", encoding=encoding) as token_file:
        for line_number, raw_line in enumerate(token_file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                skipped += 1
                continue

            fields = line.split("\t")
            if len(fields) != 3:
                skipped += 1
                continue

            source, target_field, count_field = fields
            source = unicodedata.normalize("NFC", source)
            target_tokens = tuple(
                unicodedata.normalize("NFC", token) for token in target_field.split()
            )
            target = unicodedata.normalize("NFC", "".join(target_tokens))
            try:
                count = int(count_field)
            except ValueError as _:
                skipped += 1
                continue

            if not source or not target_tokens or count < 1:
                skipped += 1
                continue

            yield TransliterationPair(
                language=language,
                source=source,
                target=target,
                target_tokens=target_tokens,
                count=count,
                path=token_path,
                line_number=line_number,
            )


def iter_language(
    language: str,
    settings: DatasetSettings,
) -> Iterator[TransliterationPair]:
    """Yield all records for one configured or discoverable language."""

    for path in find_token_files(language, settings):
        yield from iter_token_file(
            path,
            language,
            encoding=settings.encoding,
        )


def load_datasets(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    languages: Sequence[str] | None = None,
) -> dict[str, list[TransliterationPair]]:
    """Load records into a mapping keyed by language/alphabet name."""

    settings = read_settings(config_path)
    selected = tuple(languages) if languages is not None else settings.languages
    if not selected:
        raise DatasetError("At least one language/alphabet must be selected")
    return {
        language: list(iter_language(language, settings))
        for language in selected
    }


def main() -> None:
    args = parse_args()
    if args.preview < 0:
        raise SystemExit("--preview must be non-negative")

    settings = read_settings(args.config)
    selected = tuple(args.languages) if args.languages else settings.languages
    for language in selected:
        files = find_token_files(language, settings)
        records = list(iter_language(language, settings))
        total_count = sum(record.count for record in records)
        print(
            f"{language}: {len(records):,} pairs, "
            f"{total_count:,} weighted occurrences"
        )
        for path in files:
            print(f"  file: {path}")
        for record in records[: args.preview]:
            print(f"  {record.source} -> {record.target} (count={record.count})")


if __name__ == "__main__":
    main()
