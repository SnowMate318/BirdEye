import numpy as np

from wide_fov_supervision_v2.pipeline import _select_final_query_depth


def test_continuous_queries_use_base_depth_when_enabled():
    model_depth = np.array([12.0, 3.0, 8.0], dtype=np.float32)
    base_depth = np.array([10.0, 4.0, 9.0], dtype=np.float32)
    continuous = np.array([True, False, True])

    final_depth, applied = _select_final_query_depth(
        model_depth,
        base_depth,
        continuous,
        use_base_for_continuous=True,
    )

    np.testing.assert_allclose(final_depth, np.array([10.0, 3.0, 9.0], dtype=np.float32))
    np.testing.assert_array_equal(applied, np.array([True, False, True]))


def test_continuous_base_depth_policy_keeps_model_when_base_invalid():
    model_depth = np.array([12.0, 3.0, 8.0], dtype=np.float32)
    base_depth = np.array([np.nan, 4.0, 0.0], dtype=np.float32)
    continuous = np.array([True, False, True])

    final_depth, applied = _select_final_query_depth(
        model_depth,
        base_depth,
        continuous,
        use_base_for_continuous=True,
    )

    np.testing.assert_allclose(final_depth, model_depth)
    np.testing.assert_array_equal(applied, np.array([False, False, False]))


def test_continuous_base_depth_policy_can_be_disabled():
    model_depth = np.array([12.0, 3.0], dtype=np.float32)
    base_depth = np.array([10.0, 4.0], dtype=np.float32)
    continuous = np.array([True, True])

    final_depth, applied = _select_final_query_depth(
        model_depth,
        base_depth,
        continuous,
        use_base_for_continuous=False,
    )

    np.testing.assert_allclose(final_depth, model_depth)
    np.testing.assert_array_equal(applied, np.array([False, False]))
