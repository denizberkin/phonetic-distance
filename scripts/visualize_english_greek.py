"""Render the paper-style English-to-Greek cost matrix as SVG."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_config import DEFAULT_CONFIG_PATH, read_settings


ENGLISH = tuple("'abcdefghijklmnopqrstuvwxyz")
GREEK = tuple("'αβγδεζηθικλμνξοπρςστυφχψω")
CELL = 28
LEFT = 48
TOP = 12


def render_matrix(matrix_path: Path, output_path: Path) -> None:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    source_index = {symbol: i for i, symbol in enumerate(matrix["source_symbols"])}
    target_index = {symbol: i for i, symbol in enumerate(matrix["target_symbols"])}
    costs = [
        [matrix["costs"][source_index[source]][target_index[target]] for target in GREEK]
        for source in ENGLISH
    ]
    low = min(map(min, costs))
    high = max(map(max, costs))
    scale = high - low or 1.0
    width = LEFT + len(GREEK) * CELL + 12
    height = TOP + len(ENGLISH) * CELL + 42

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g shape-rendering="crispEdges">',
    ]
    for row, source in enumerate(ENGLISH):
        for column, target in enumerate(GREEK):
            cost = costs[row][column]
            shade = round(255 * (cost - low) / scale)
            x, y = LEFT + column * CELL, TOP + row * CELL
            svg.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="rgb({shade},{shade},{shade})" stroke="#999" stroke-width="0.5">'
                f'<title>{escape(source)} → {escape(target)}: {cost:.6f}</title></rect>'
            )
    svg.append("</g>")

    svg.append('<g font-family="sans-serif" font-size="15" fill="black">')
    for row, source in enumerate(ENGLISH):
        y = TOP + (row + 0.5) * CELL + 5
        svg.append(f'<text x="{LEFT - 10}" y="{y}" text-anchor="end">{escape(source)}</text>')
    for column, target in enumerate(GREEK):
        x = LEFT + (column + 0.5) * CELL
        y = TOP + len(ENGLISH) * CELL + 24
        svg.append(f'<text x="{x}" y="{y}" text-anchor="middle">{escape(target)}</text>')
    svg.extend(("</g>", "</svg>"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    settings = read_settings(DEFAULT_CONFIG_PATH)
    matrix_path = settings.output_dir / "english_to_greek.json"
    output_path = Path(__file__).parent.parent / "assets" / "english_to_greek.svg"
    render_matrix(matrix_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
