import numpy as np
import pytest

from analysis.championship_preview import field_quality


def test_preview_quality_preserves_driver_alignment_and_unknown_field_mean():
    actual = field_quality(np.array(["a", "b"]), np.array([[1., 2.], [3., 7.]]),
                           np.array(["b", "unknown", "a"]), np.array([1, 0]))
    np.testing.assert_allclose(actual, [[2., 0., -2.], [.5, 0., -.5]])
    np.testing.assert_allclose(actual.mean(1), 0.)
    with pytest.raises(ValueError, match="No current drivers"):
        field_quality(np.array(["a"]), np.array([[1.]]), np.array(["unknown"]), np.array([0]))
