"""Dynamic Time Warping for indexed character sequences."""

from __future__ import annotations

import numpy as np


DTWPath = list[tuple[int, int, float]]


def dtw(
    source: np.ndarray,
    target: np.ndarray,
    costs: np.ndarray,
    *,
    return_path: bool = False,
    return_positions: bool = False,
) -> float | tuple[float, DTWPath]:
    """Return minimum cost and optionally its forward ``(s, t, c)`` path."""

    pair_costs = costs[source[:, None], target].tolist()
    width = len(target)
    back = bytearray(len(source) * width) if return_path else None
    previous = [0.0] * width
    previous[0] = pair_costs[0][0]

    for j in range(1, width):
        previous[j] = previous[j - 1] + pair_costs[0][j]
        if back is not None:
            back[j] = 1

    for i in range(1, len(source)):
        current = [0.0] * width
        current[0] = previous[0] + pair_costs[i][0]
        offset = i * width

        for j in range(1, width):
            up, left, diagonal = previous[j], current[j - 1], previous[j - 1]
            if up <= left and up <= diagonal:
                best, step = up, 0
            elif left <= diagonal:
                best, step = left, 1
            else:
                best, step = diagonal, 2
            current[j] = pair_costs[i][j] + best
            if back is not None:
                back[offset + j] = step
        previous = current

    total = float(previous[-1])
    if back is None:
        return total

    path: DTWPath = []
    i, j = len(source) - 1, width - 1
    while True:
        source_step, target_step = (
            (i, j) if return_positions else (int(source[i]), int(target[j]))
        )
        path.append((source_step, target_step, pair_costs[i][j]))
        if i == 0 and j == 0:
            break
        step = back[i * width + j]
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return total, path
