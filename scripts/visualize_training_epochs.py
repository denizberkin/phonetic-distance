'''Animate English-to-Greek cost learning and one example DTW path.'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from load_datasets import load_datasets
from train import _directional_pairs, _training_records
from utils.dataset_config import DEFAULT_CONFIG_PATH, read_settings
from utils.dtw import dtw
from utils.training import train_cost_matrix


ROOT = Path(__file__).parent.parent
ENGLISH = tuple(chr(39) + 'abcdefghijklmnopqrstuvwxyz')
GREEK = tuple(chr(39) + '\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03bf\u03c0\u03c1\u03c2\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9')
FPS = 2


def _visible_axis(
    symbols: tuple[str, ...], preferred: tuple[str, ...]
) -> tuple[tuple[str, ...], np.ndarray]:
    indices = {symbol: index for index, symbol in enumerate(symbols)}
    visible = tuple(symbol for symbol in preferred if symbol in indices)
    return visible, np.array([indices[symbol] for symbol in visible])


def save_matrix_animation(
    history: list[tuple[int, np.ndarray]],
    source_symbols: tuple[str, ...],
    target_symbols: tuple[str, ...],
    output_path: Path,
) -> None:
    sources, source_indices = _visible_axis(source_symbols, ENGLISH)
    targets, target_indices = _visible_axis(target_symbols, GREEK)
    frames = [matrix[np.ix_(source_indices, target_indices)] for _, matrix in history]
    low = min(float(frame.min()) for frame in frames)
    high = max(float(frame.max()) for frame in frames)
    high = high if high > low else low + 1

    figure, axis = plt.subplots(figsize=(9, 7.5), layout='constrained')
    image = axis.imshow(
        frames[0], cmap='gray', vmin=low, vmax=high, aspect='auto'
    )
    axis.set_xticks(range(len(targets)), labels=targets)
    axis.set_yticks(range(len(sources)), labels=sources)
    axis.set_xlabel('Greek character')
    axis.set_ylabel('English character')
    title = axis.set_title('')
    figure.colorbar(image, ax=axis, label='phonetic cost (darker = lower)')

    def update(frame_index: int):
        epoch, _ = history[frame_index]
        image.set_data(frames[frame_index])
        title.set_text(f'English → Greek cost matrix · epoch {epoch}')
        return image, title

    animation = FuncAnimation(figure, update, frames=len(frames), interval=500)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=FPS), dpi=100)
    plt.close(figure)


def save_pair_animation(
    history: list[tuple[int, np.ndarray]],
    source_symbols: tuple[str, ...],
    target_symbols: tuple[str, ...],
    source: str,
    target: str,
    output_path: Path,
) -> None:
    source_index = {symbol: index for index, symbol in enumerate(source_symbols)}
    target_index = {symbol: index for index, symbol in enumerate(target_symbols)}
    try:
        encoded_source = np.array([source_index[char] for char in source])
        encoded_target = np.array([target_index[char] for char in target])
    except KeyError as error:
        raise ValueError(
            f'Example character absent from training matrix: {error.args[0]!r}'
        ) from error

    frames = []
    for epoch, matrix in history:
        pair_costs = matrix[encoded_source[:, None], encoded_target]
        total, path = dtw(
            encoded_source,
            encoded_target,
            matrix,
            return_path=True,
            return_positions=True,
        )
        frames.append((epoch, pair_costs, total, path))

    low = min(float(frame[1].min()) for frame in frames)
    high = max(float(frame[1].max()) for frame in frames)
    high = high if high > low else low + 1
    epochs = [frame[0] for frame in frames]
    totals = [frame[2] for frame in frames]

    figure, (path_axis, cost_axis) = plt.subplots(
        1,
        2,
        figsize=(10, 4.6),
        gridspec_kw={'width_ratios': (1.2, 1)},
        layout='constrained',
    )
    image = path_axis.imshow(
        frames[0][1], cmap='viridis_r', vmin=low, vmax=high
    )
    path_axis.set_xticks(range(len(target)), labels=target)
    path_axis.set_yticks(range(len(source)), labels=source)
    path_axis.set_xlabel('Greek target')
    path_axis.set_ylabel('English source')
    path_axis.set_xticks(np.arange(-0.5, len(target), 1), minor=True)
    path_axis.set_yticks(np.arange(-0.5, len(source), 1), minor=True)
    path_axis.grid(which='minor', color='white', linewidth=0.6, alpha=0.65)
    path_axis.tick_params(which='minor', bottom=False, left=False)
    path_line, = path_axis.plot([], [], 'o-', color='#ef476f', linewidth=2.5)
    labels = [
        path_axis.text(column, row, '', ha='center', va='center', fontsize=8)
        for row in range(len(source))
        for column in range(len(target))
    ]
    title = path_axis.set_title('')
    figure.colorbar(
        image, ax=path_axis, label='local character cost', shrink=0.82
    )

    cost_line, = cost_axis.plot([], [], 'o-', color='#118ab2', linewidth=2)
    cursor, = cost_axis.plot([], [], 'o', color='#ef476f', markersize=8)
    cost_axis.set_xlim(min(epochs), max(epochs) or 1)
    padding = max((max(totals) - min(totals)) * 0.08, 0.1)
    cost_axis.set_ylim(min(totals) - padding, max(totals) + padding)
    cost_axis.set_xlabel('epoch')
    cost_axis.set_ylabel('DTW cost')
    cost_axis.set_title('Pair cost over training')
    cost_axis.grid(alpha=0.25)

    def update(frame_index: int):
        epoch, pair_costs, total, path = frames[frame_index]
        image.set_data(pair_costs)
        for label, value in zip(labels, pair_costs.flat):
            label.set_text(f'{value:.2f}')
        path_line.set_data(
            [target_position for _, target_position, _ in path],
            [source_position for source_position, _, _ in path],
        )
        title.set_text(
            f'{source} → {target} · epoch {epoch}\nDTW cost = {total:.3f}'
        )
        cost_line.set_data(epochs[: frame_index + 1], totals[: frame_index + 1])
        cursor.set_data([epoch], [total])
        return image, path_line, title, cost_line, cursor, *labels

    animation = FuncAnimation(figure, update, frames=len(frames), interval=500)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=FPS), dpi=100)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Animate English-to-Greek cost-matrix training.'
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets')
    args = parser.parse_args()

    settings = read_settings(args.config)
    records = load_datasets(args.config, ['greek'])['greek']
    example = next(record for record in records if record.source == 'allegro')
    training_records = _training_records(
        records.copy(), settings.train_fraction, settings.random_seed, 'greek'
    )
    history: list[tuple[int, np.ndarray]] = []
    result = train_cost_matrix(
        _directional_pairs(training_records, 'english_to_target'),
        max_iterations=settings.max_iterations,
        tolerance=settings.convergence_tolerance,
        on_iteration=lambda epoch, cost: print(
            f'epoch {epoch}: row_normalized_cost={cost:.6f}'
        ),
        on_epoch=lambda epoch, matrix: history.append((epoch, matrix)),
    )

    matrix_path = args.output_dir / 'english_to_greek_epochs.gif'
    pair_path = args.output_dir / 'allegro_to_greek_path.gif'
    save_matrix_animation(
        history, result.source_symbols, result.target_symbols, matrix_path
    )
    save_pair_animation(
        history,
        result.source_symbols,
        result.target_symbols,
        example.source,
        example.target,
        pair_path,
    )
    print(matrix_path)
    print(pair_path)


if __name__ == '__main__':
    main()
