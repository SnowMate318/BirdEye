import numpy as np

from wide_fov_supervision_v2.config import BevConfig
from wide_fov_supervision_v2.modules.surface_rasterization import rasterize_floor_surfaces


def test_floor_surface_rasterization_fills_area_between_corner_points():
    config = BevConfig(
        center_xy=(0.0, 0.0),
        size_m=1.0,
        meters_per_pixel=0.05,
        floor_surface_fill_height_margin_m=0.1,
        floor_surface_fill_max_corner_z_range_m=0.02,
    )
    rgb = np.full((2, 2, 3), [20, 40, 80], dtype=np.uint8)
    points = np.array(
        [
            [[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]],
            [[-0.2, 0.2, 0.0], [0.2, 0.2, 0.0]],
        ],
        dtype=np.float32,
    )

    result = rasterize_floor_surfaces(rgb, points, np.ones((2, 2), dtype=bool), config)

    source_point_cells = 4
    assert result.source_cell_mask.sum() == 1
    assert np.count_nonzero(result.bev_valid) > source_point_cells
    assert result.bev_rgb[..., 3].max() == 255


def test_floor_surface_rasterization_rejects_large_z_discontinuity():
    config = BevConfig(
        center_xy=(0.0, 0.0),
        size_m=1.0,
        meters_per_pixel=0.05,
        floor_surface_fill_height_margin_m=0.1,
        floor_surface_fill_max_corner_z_range_m=0.02,
    )
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    points = np.array(
        [
            [[-0.4, -0.2, 0.0], [0.0, -0.2, 0.0], [0.4, -0.2, 0.5]],
            [[-0.4, 0.2, 0.0], [0.0, 0.2, 0.0], [0.4, 0.2, 0.5]],
        ],
        dtype=np.float32,
    )

    result = rasterize_floor_surfaces(rgb, points, np.ones((2, 3), dtype=bool), config)

    np.testing.assert_array_equal(result.source_cell_mask, np.array([[True, False]]))
