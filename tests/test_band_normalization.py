import numpy as np
import pytest
from band_normalization import normalize_band


def test_heldout_rows_cannot_change_fitted_training_representation():
    b = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    mask = np.array([True, True, False, False])
    expected = normalize_band(b, mask)
    changed = b.copy()
    changed[~mask] += 10000
    actual = normalize_band(changed, mask)
    np.testing.assert_array_equal(actual[mask], expected[mask])
    mu, sd = b[mask].mean(0), b[mask].std(0) + 1e-6
    np.testing.assert_allclose(expected, ((b-mu)/sd).mean(1))
    assert not np.allclose(actual[~mask], expected[~mask])


def test_no_implicit_full_data_or_empty_training_fit():
    b = np.ones((4, 2, 3), dtype=np.float32)
    for mask in [None, [0, 1, 0, 1], np.zeros(4, dtype=bool)]:
        with pytest.raises(ValueError):
            normalize_band(b, mask)
