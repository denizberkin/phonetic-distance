"""Small check for the DTW recurrence and trainer."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dtw import dtw
from utils.training import _accumulate_path, train_cost_matrix


def test_training() -> None:
    weights = np.zeros((2, 2), dtype=np.int64)
    source = np.array((0, 1))
    target = np.array((0, 1))
    costs = np.array(((1.0, 5.0), (5.0, 1.0)))
    assert dtw(source, target, costs) == 2.0
    total, path = dtw(source, target, costs, return_path=True)
    assert path == [(0, 0, 1.0), (1, 1, 1.0)]
    _accumulate_path(weights, path)
    assert total == 2.0
    assert np.array_equal(weights, ((1, 0), (0, 1)))

    normalized_costs = []
    result = train_cost_matrix(
        [("ab", "αβ"), ("ac", "αγ"), ("db", "δβ"), ("dc", "δγ")],
        max_iterations=5,
        tolerance=1e-5,
        on_iteration=lambda _, cost: normalized_costs.append(cost),
    )
    assert normalized_costs
    assert result.costs.shape == (
        len(result.source_symbols),
        len(result.target_symbols),
    )
    assert np.isfinite(result.costs).all()


if __name__ == "__main__":
    test_training()
    print("training check passed")
