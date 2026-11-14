"""Tiled inference: window planning and merging (shared/yolo_platform_shared/tiling.py).

The same module and tests live in lost-found and InsightEye.
"""

import numpy as np
import pytest

from yolo_platform_shared.tiling import TileConfig, crops, merge, parse_grid, plan_windows

AUTO = TileConfig()


def row(x1, y1, x2, y2, conf, cls):
    return [x1, y1, x2, y2, conf, cls]


def dets(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


# --------------------------------------------------------------------- windows
def test_auto_leaves_substreams_alone():
    assert plan_windows(720, 480, AUTO) == [(0, 0, 720, 480)]
    assert plan_windows(640, 480, AUTO) == [(0, 0, 640, 480)]


def test_auto_tiles_1080p_into_whole_frame_plus_2x2():
    windows = plan_windows(1920, 1080, AUTO)
    assert windows[0] == (0, 0, 1920, 1080)
    tiles = windows[1:]
    assert len(tiles) == 4
    sizes = {(x2 - x1, y2 - y1) for x1, y1, x2, y2 in tiles}
    assert len(sizes) == 1, "tiles must share one size so they batch together"
    (tw, th), = sizes
    assert (tw, th) == (1067, 600)
    # Flush with the frame's edges, overlapping by about 20% of a tile.
    assert max(x2 for _, _, x2, _ in tiles) == 1920 and max(y2 for *_, y2 in tiles) == 1080
    xs = sorted({x1 for x1, *_ in tiles})
    assert xs == [0, 853] and 1067 - 853 == pytest.approx(0.2 * tw, abs=2)


def test_every_pixel_is_covered_by_a_tile():
    for w, h, grid in [(1920, 1080, (2, 2)), (2560, 1440, (3, 2)), (3840, 2160, (4, 3)), (1001, 997, (3, 3))]:
        cfg = TileConfig(mode="on", grid=grid)
        seen = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in plan_windows(w, h, cfg)[1:]:
            seen[y1:y2, x1:x2] = True
        assert seen.all(), (w, h, grid)
        assert len(plan_windows(w, h, cfg)) == 1 + grid[0] * grid[1]


def test_modes():
    assert plan_windows(1920, 1080, TileConfig(mode="off")) == [(0, 0, 1920, 1080)]
    assert len(plan_windows(720, 480, TileConfig(mode="on"))) == 5
    assert plan_windows(1920, 1080, TileConfig(grid=(1, 1))) == [(0, 0, 1920, 1080)]
    assert len(plan_windows(959, 540, AUTO)) == 1 and len(plan_windows(960, 540, AUTO)) == 5


def test_crops_are_views():
    frame = np.arange(1080 * 1920 * 3, dtype=np.uint32).reshape(1080, 1920, 3)
    windows = plan_windows(1920, 1080, AUTO)
    parts = crops(frame, windows)
    assert parts[0].base is frame or parts[0].base is frame.base
    x1, y1, x2, y2 = windows[4]
    assert np.array_equal(parts[4], frame[y1:y2, x1:x2]) and parts[4].shape == (600, 1067, 3)


# ---------------------------------------------------------------------- config
def test_overrides_from_camera_settings():
    cfg = AUTO.with_overrides({"tile_mode": "ON", "tile_grid": "3x2", "tile_min_side": "1280",
                               "tile_overlap": 0.25, "abandon_seconds": 30})
    assert (cfg.mode, cfg.grid, cfg.min_side, cfg.overlap) == ("on", (3, 2), 1280, 0.25)
    assert AUTO.with_overrides(None) is AUTO and AUTO.with_overrides({"tile_mode": None}) is AUTO


@pytest.mark.parametrize("settings", [
    {"tile_mode": "sometimes"}, {"tile_grid": "9x9"}, {"tile_grid": "2by2"}, {"tile_grid": 4},
    {"tile_overlap": 0.9}, {"tile_overlap": "lots"}, {"tile_min_side": -1}, {"tile_min_side": True},
    {"tile_merge_iou": 1.0},
])
def test_bad_overrides_are_rejected(settings):
    with pytest.raises(ValueError):
        AUTO.with_overrides(settings)


def test_parse_grid():
    assert parse_grid("2x2") == (2, 2) and parse_grid(" 3 X 2 ") == (3, 2) and parse_grid([4, 3]) == (4, 3)


# ----------------------------------------------------------------------- merge
WINDOWS = plan_windows(1920, 1080, AUTO)  # full, (0,0)-(1067,600), (853,0)-(1920,600), (0,480)-(1067,1080), ...


def test_untouched_detections_shift_into_frame_pixels():
    out = merge([dets(), dets(row(10, 20, 50, 80, 0.9, 1)), dets(), dets(), dets()], WINDOWS)
    assert out.tolist() == [row(10, 20, 50, 80, pytest.approx(0.9), 1)]
    out = merge([dets(), dets(), dets(), dets(), dets(row(10, 20, 50, 80, 0.9, 1))], WINDOWS)
    x1, y1 = WINDOWS[4][:2]
    assert out[0, :4].tolist() == [10 + x1, 20 + y1, 50 + x1, 80 + y1]


def test_small_object_found_only_in_a_tile_is_kept():
    out = merge([dets(row(100, 100, 400, 500, 0.8, 5)), dets(row(900, 300, 920, 315, 0.4, 7)),
                 dets(), dets(), dets()], WINDOWS)
    assert sorted(out[:, 5].tolist()) == [5, 7]


def test_object_in_two_overlapping_tiles_becomes_one_box():
    # A bag at x 900-1000 sits in the overlap of the two top tiles.
    a = dets(row(900, 100, 1000, 200, 0.7, 1))
    b = dets(row(900 - 853, 100, 1000 - 853, 200, 0.8, 1))
    out = merge([dets(), a, b, dets(), dets()], WINDOWS)
    assert len(out) == 1 and out[0, 4] == pytest.approx(0.8)


def test_big_object_cut_by_tiles_keeps_the_whole_frame_box():
    # A person 500-1300 x 200-1000 spans all four tiles; each tile sees a cut piece
    # (running to the tile's inner edge), the whole frame sees it whole.
    full = dets(row(500, 200, 1300, 1000, 0.6, 5))
    pieces = [
        dets(row(500, 200, 1067, 600, 0.9, 5)),            # top-left: cut right and bottom
        dets(row(0, 200, 1300 - 853, 600, 0.85, 5)),        # top-right: cut left and bottom
        dets(row(500, 0, 1067, 1000 - 480, 0.88, 5)),        # bottom-left: cut right and top
        dets(row(0, 0, 1300 - 853, 1000 - 480, 0.8, 5)),     # bottom-right: cut left and top
    ]
    out = merge([full, *pieces], WINDOWS)
    assert out.tolist() == [row(500, 200, 1300, 1000, pytest.approx(0.6), 5)]


def test_cut_box_with_no_whole_view_is_kept():
    out = merge([dets(), dets(row(1000, 100, 1067, 200, 0.5, 1)), dets(), dets(), dets()], WINDOWS)
    assert len(out) == 1


def test_frame_border_is_not_a_cut():
    # Touching the top-left tile's outer edges (the frame border) is not a cut,
    # so the whole-frame duplicate is merged by NMS, not dropped as a piece.
    full = dets(row(0, 0, 60, 60, 0.5, 3))
    tile = dets(row(0, 0, 60, 60, 0.9, 3))
    out = merge([full, tile, dets(), dets(), dets()], WINDOWS)
    assert len(out) == 1 and out[0, 4] == pytest.approx(0.9)


def test_wallet_in_a_hand_survives_the_person_box():
    # A wallet inside a person box, both from the same window: the model decided.
    # The same pair also seen from a tile collapses to one person and one wallet.
    person, wallet = row(100, 100, 300, 500, 0.9, 5), row(200, 250, 230, 280, 0.6, 7)
    out = merge([dets(person, wallet), dets(person, wallet), dets(), dets(), dets()], WINDOWS)
    assert sorted(out[:, 5].tolist()) == [5, 7]


def test_same_window_overlaps_are_left_to_the_model():
    # Two people overlapping (IoU ~0.6) in one window stay two.
    a, b = row(100, 100, 300, 500, 0.9, 5), row(160, 100, 360, 500, 0.8, 5)
    out = merge([dets(a, b), dets(), dets(), dets(), dets()], WINDOWS)
    assert len(out) == 2


def test_agnostic_merges_a_class_flip_between_views():
    full = dets(row(900, 100, 1000, 200, 0.5, 1))  # bags
    tile = dets(row(900, 100, 1000, 200, 0.7, 7))  # wallet
    assert len(merge([full, tile, dets(), dets(), dets()], WINDOWS, agnostic=True)) == 1
    assert len(merge([full, tile, dets(), dets(), dets()], WINDOWS, agnostic=False)) == 2


def test_extra_columns_and_empty_input():
    seven = np.array([[10, 10, 50, 50, 3, 0.9, 1]], dtype=np.float32)  # with a track id column
    out = merge([seven, None, None, None, None], WINDOWS)
    assert out.shape == (1, 7) and out[0, 5] == pytest.approx(0.9)
    assert merge([None] * 5, WINDOWS).shape == (0, 6)
    with pytest.raises(ValueError):
        merge([None], WINDOWS)


def test_max_det_caps_the_result():
    many = dets(*[row(i * 30, 10, i * 30 + 20, 30, 0.5 + i / 1000, 1) for i in range(50)])
    assert len(merge([many, None, None, None, None], WINDOWS, max_det=10)) == 10
