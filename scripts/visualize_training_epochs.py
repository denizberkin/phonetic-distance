'''Animate English-to-target cost learning and one example DTW path.'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from load_datasets import load_datasets
from train import _directional_pairs, _training_records
from utils.dataset_config import DEFAULT_CONFIG_PATH, read_settings
from utils.dtw import dtw
from utils.training import train_cost_matrix


ROOT = Path(__file__).parent.parent
ENGLISH = tuple(chr(39) + 'abcdefghijklmnopqrstuvwxyz')
TARGET_ALPHABETS = {
    'greek': tuple(chr(39) + '\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03bf\u03c0\u03c1\u03c2\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9'),
    'russian': tuple(chr(39) + '\u0430\u0431\u0432\u0433\u0434\u0435\u0451\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f'),
    'katakana': tuple(
        chr(39)
        + ''.join(chr(codepoint) for codepoint in range(0x30A1, 0x30FB))
        + '\u30fc'
    ),
}
EXAMPLE_SOURCES = {
    'greek': 'allegro',
    'russian': 'ashenfelter',
    'katakana': 'mozart',
}
STEP_COLORS = {'one': '#6c757d', 'many_to_one': '#ef476f', 'one_to_many': '#118ab2'}
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
    target_alphabet: tuple[str, ...],
    target_language: str,
    output_path: Path,
) -> None:
    sources, source_indices = _visible_axis(source_symbols, ENGLISH)
    targets, target_indices = _visible_axis(target_symbols, target_alphabet)
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
    target_name = target_language.title()
    axis.set_xlabel(f'{target_name} character')
    axis.set_ylabel('English character')
    title = axis.set_title('')
    figure.colorbar(image, ax=axis, label='phonetic cost (darker = lower)')

    def update(frame_index: int):
        epoch, _ = history[frame_index]
        image.set_data(frames[frame_index])
        title.set_text(f'English → {target_name} cost matrix · epoch {epoch}')
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
    target_language: str,
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
    path_axis.set_xlabel(f'{target_language.title()} target')
    path_axis.set_ylabel('English source')
    path_axis.set_xticks(np.arange(-0.5, len(target), 1), minor=True)
    path_axis.set_yticks(np.arange(-0.5, len(source), 1), minor=True)
    path_axis.grid(which='minor', color='white', linewidth=0.6, alpha=0.65)
    path_axis.tick_params(which='minor', bottom=False, left=False)
    path_segments = LineCollection([], linewidths=2.8, zorder=3)
    path_axis.add_collection(path_segments)
    path_dots, = path_axis.plot([], [], 'o', color='black', markersize=4, zorder=4)
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
    cost_axis.legend(
        handles=[
            Line2D([], [], color=STEP_COLORS['one'], label='one-to-one'),
            Line2D([], [], color=STEP_COLORS['many_to_one'], label='many Latin → one target'),
            Line2D([], [], color=STEP_COLORS['one_to_many'], label='one Latin → many target'),
        ],
        loc='upper right',
        fontsize=8,
    )

    def update(frame_index: int):
        epoch, pair_costs, total, path = frames[frame_index]
        image.set_data(pair_costs)
        for label, value in zip(labels, pair_costs.flat):
            label.set_text(f'{value:.2f}')
        points = np.array(
            [(target_position, source_position) for source_position, target_position, _ in path]
        )
        segments = np.stack((points[:-1], points[1:]), axis=1)
        steps = np.diff(points, axis=0)
        path_segments.set_segments(segments)
        path_segments.set_color(
            [
                STEP_COLORS['many_to_one']
                if target_step == 0
                else STEP_COLORS['one_to_many']
                if source_step == 0
                else STEP_COLORS['one']
                for target_step, source_step in steps
            ]
        )
        path_dots.set_data(points[:, 0], points[:, 1])
        title.set_text(
            f'{source} → {target} · epoch {epoch}\nDTW cost = {total:.3f}'
        )
        cost_line.set_data(epochs[: frame_index + 1], totals[: frame_index + 1])
        cursor.set_data([epoch], [total])
        return image, path_segments, path_dots, title, cost_line, cursor, *labels

    animation = FuncAnimation(figure, update, frames=len(frames), interval=500)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=FPS), dpi=100)
    plt.close(figure)


def save_pair_progression(
    history: list[tuple[int, np.ndarray]],
    source_symbols: tuple[str, ...],
    target_symbols: tuple[str, ...],
    source: str,
    target: str,
    target_language: str,
    output_path: Path,
) -> None:
    """Save the first and final pair-cost/path frames as a vector PDF."""
    source_index = {symbol: index for index, symbol in enumerate(source_symbols)}
    target_index = {symbol: index for index, symbol in enumerate(target_symbols)}
    encoded_source = np.array([source_index[char] for char in source])
    encoded_target = np.array([target_index[char] for char in target])

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
    colors = plt.get_cmap('viridis_r')(np.linspace(0, 1, 256))
    colors[:, :3] = 0.78 * colors[:, :3] + 0.22
    light_cmap = matplotlib.colors.ListedColormap(colors)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.9),
        gridspec_kw={'wspace': 0.02},
        layout='constrained',
    )

    for index, (axis, frame) in enumerate(
        zip(axes, (frames[0], frames[-1]))
    ):
        epoch, pair_costs, total, path = frame
        axis.pcolormesh(
            pair_costs,
            cmap=light_cmap,
            vmin=low,
            vmax=high,
            edgecolors='white',
            linewidth=0.7,
        )
        axis.set_xlim(0, len(target))
        axis.set_ylim(len(source), 0)
        axis.set_aspect('equal')
        axis.set_xticks(np.arange(len(target)) + 0.5, labels=target, fontsize=11)
        axis.set_yticks(np.arange(len(source)) + 0.5, labels=source, fontsize=11)
        axis.set_xlabel(f'{target_language.title()} target', fontsize=12)
        if index == 0:
            axis.set_ylabel('English source', fontsize=12)
        else:
            axis.tick_params(axis='y', left=False, labelleft=False)
        for (row, column), value in np.ndenumerate(pair_costs):
            axis.text(
                column + 0.5,
                row + 0.5,
                f'{value:.2f}',
                ha='center',
                va='center',
                color='black',
                fontsize=8.2,
                fontweight='normal',
            )
        points = np.array(
            [
                (target_position, source_position)
                for source_position, target_position, _ in path
            ]
        ) + 0.5
        axis.plot(
            points[:, 0],
            points[:, 1],
            '-o',
            color='#ef476f',
            linewidth=2.5,
            markersize=3.5,
            zorder=3,
        )
        label = 'Initialization' if index == 0 else 'Converged'
        axis.set_title(
            f'{label} · epoch {epoch}\nDTW cost = {total:.3f}',
            fontsize=13,
            fontweight='normal',
        )

    figure.suptitle(
        f'{source} → {target}: first and final DTW paths',
        fontsize=16,
        fontweight='bold',
        color='#17324D',
    )
    figure.canvas.draw()
    figure.set_layout_engine(None)
    left_box = axes[0].get_position()
    right_box = axes[1].get_position()
    gap = right_box.x0 - left_box.x1
    y_center = (left_box.y0 + left_box.y1) / 2
    arrow = FancyArrowPatch(
        (left_box.x1 + 0.22 * gap, y_center),
        (right_box.x0 - 0.22 * gap, y_center),
        transform=figure.transFigure,
        arrowstyle='-|>',
        mutation_scale=12,
        linewidth=2,
        shrinkA=0,
        shrinkB=0,
        color='#2E6F9E',
    )
    arrow.set_in_layout(False)
    figure.add_artist(arrow)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Animate English-to-target cost-matrix training.'
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets')
    parser.add_argument(
        '--language', choices=TARGET_ALPHABETS, default='greek'
    )
    parser.add_argument('--source', help='Dataset source example to animate')
    parser.add_argument(
        '--static-only',
        action='store_true',
        help='write only the vector first/final pair panel',
    )
    args = parser.parse_args()

    settings = read_settings(args.config)
    language = args.language
    if language == 'katakana':
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        japanese_font = next(
            (
                name
                for name in ('Noto Sans JP', 'Yu Gothic', 'Meiryo', 'MS Gothic')
                if name in available_fonts
            ),
            None,
        )
        if japanese_font is None:
            raise SystemExit('Install a Japanese font such as Noto Sans JP')
        matplotlib.rcParams['font.family'] = japanese_font
    records = load_datasets(args.config, [language])[language]
    example_source = args.source or EXAMPLE_SOURCES[language]
    try:
        example = next(record for record in records if record.source == example_source)
    except StopIteration:
        raise SystemExit(f'No {language} pair has source {example_source!r}')
    training_records = _training_records(
        records.copy(), settings.train_fraction, settings.random_seed, language
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

    progression_path = (
        args.output_dir / f'{example.source}_to_{language}_progression.pdf'
    )
    save_pair_progression(
        history,
        result.source_symbols,
        result.target_symbols,
        example.source,
        example.target,
        language,
        progression_path,
    )
    print(progression_path)
    if args.static_only:
        return

    matrix_path = args.output_dir / f'english_to_{language}_epochs.gif'
    pair_path = args.output_dir / f'{example.source}_to_{language}_path.gif'
    save_matrix_animation(
        history,
        result.source_symbols,
        result.target_symbols,
        TARGET_ALPHABETS[language],
        language,
        matrix_path,
    )
    save_pair_animation(
        history,
        result.source_symbols,
        result.target_symbols,
        example.source,
        example.target,
        language,
        pair_path,
    )
    print(matrix_path)
    print(pair_path)


if __name__ == '__main__':
    main()
