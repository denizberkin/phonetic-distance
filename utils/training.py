"""DTW and hard-EM training from the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from utils.dtw import DTWPath, dtw


@dataclass(frozen=True, slots=True)
class TrainingResult:
    source_symbols: tuple[str, ...]
    target_symbols: tuple[str, ...]
    costs: np.ndarray
    iterations: int
    converged: bool
    total_cost: float


def _costs_from_weights(weights: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.maximum(
            weights / weights.sum(axis=1, keepdims=True),
            weights / weights.sum(axis=0, keepdims=True),
        )
        costs = -np.log(probabilities)
    finite = np.isfinite(costs)
    costs[~finite] = costs[finite].max(initial=0.0)
    return costs


def _accumulate_path(weights: np.ndarray, path: DTWPath) -> None:
    for source, target, _ in path:
        weights[source, target] += 1


def train_cost_matrix(
    pairs: Sequence[tuple[str, str]],
    *,
    max_iterations: int,
    tolerance: float,
    on_iteration: Callable[[int, float], None] | None = None,
) -> TrainingResult:
    """Learn a directional character cost matrix with Algorithms 1 and 2."""

    source_i2c = tuple(sorted({char for source, _ in pairs for char in source}))
    target_i2c = tuple(sorted({char for _, target in pairs for char in target}))
    source_c2i = {char: i for i, char in enumerate(source_i2c)}
    target_c2i = {char: i for i, char in enumerate(target_i2c)}
    encoded = [
        (
            np.fromiter((source_c2i[char] for char in source), dtype=np.intp),
            np.fromiter((target_c2i[char] for char in target), dtype=np.intp),
        )
        for source, target in pairs
    ]

    weights = np.zeros((len(source_i2c), len(target_i2c)), dtype=np.int64)
    for source, target in encoded:
        np.add.at(weights, (source[:, None], target[None, :]), 1)
    costs = _costs_from_weights(weights)

    previous_normalized: float | None = None
    converged = False
    total_cost = 0.0

    # train loop
    for iteration in range(1, max_iterations + 1):
        next_weights = np.zeros_like(weights)
        total_cost = 0.0
        for source, target in encoded:
            pair_cost, path = dtw(source, target, costs, return_path=True)
            total_cost += pair_cost
            _accumulate_path(next_weights, path)
        normalized_cost = total_cost / len(encoded)

        if on_iteration:
            on_iteration(iteration, normalized_cost)
        costs = _costs_from_weights(next_weights)

        if (
            previous_normalized is not None
            and abs(normalized_cost - previous_normalized) < tolerance
        ):
            converged = True
            break
        previous_normalized = normalized_cost

    return TrainingResult(
        source_symbols=source_i2c,
        target_symbols=target_i2c,
        costs=costs,
        iterations=iteration,
        converged=converged,
        total_cost=total_cost,
    )
