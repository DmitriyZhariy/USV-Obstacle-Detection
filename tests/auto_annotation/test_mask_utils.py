import numpy as np
import pytest
from usv.auto_annotation.postprocess.mask_utils import (
    mask_area, mask_iou, bbox_iou, mask_to_polygon,
)


# mask_area

def test_mask_area_zeros():
    assert mask_area(np.zeros((10, 10), dtype=np.uint8)) == 0

def test_mask_area_full():
    assert mask_area(np.ones((10, 10), dtype=np.uint8)) == 100

def test_mask_area_partial():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 3:7] = 1
    assert mask_area(m) == 12


# mask_iou

def test_mask_iou_identical():
    m = np.eye(5, dtype=np.uint8)
    assert mask_iou(m, m) == pytest.approx(1.0)

def test_mask_iou_no_overlap():
    a = np.zeros((4, 4), dtype=np.uint8); a[0, 0] = 1
    b = np.zeros((4, 4), dtype=np.uint8); b[3, 3] = 1
    assert mask_iou(a, b) == pytest.approx(0.0)

def test_mask_iou_partial():
    a = np.zeros((4, 4), dtype=np.uint8); a[0:2, 0:2] = 1  # 4 px
    b = np.zeros((4, 4), dtype=np.uint8); b[1:3, 1:3] = 1  # 4 px, 1 px overlap
    # intersection=1, union=7
    assert mask_iou(a, b) == pytest.approx(1 / 7)

def test_mask_iou_empty_both():
    a = np.zeros((4, 4), dtype=np.uint8)
    assert mask_iou(a, a) == pytest.approx(0.0)


# bbox_iou

def test_bbox_iou_identical():
    box = (0.0, 0.0, 10.0, 10.0)
    assert bbox_iou(box, box) == pytest.approx(1.0)

def test_bbox_iou_no_overlap():
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == pytest.approx(0.0)

def test_bbox_iou_half_overlap():
    # a=(0,0,2,2)=4, b=(1,0,3,2)=4, inter=(1,0,2,2)=2, union=6
    assert bbox_iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(2 / 6)

def test_bbox_iou_contained():
    # inner fully inside outer: inter=inner_area, union=outer_area
    assert bbox_iou((0, 0, 4, 4), (1, 1, 3, 3)) == pytest.approx(4 / 16)


# mask_to_polygon

def test_mask_to_polygon_square():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[5:15, 5:15] = 1
    pts = mask_to_polygon(m)
    assert pts is not None
    assert len(pts) >= 3
    # All points must be within bounds
    for x, y in pts:
        assert 0 <= x <= 20
        assert 0 <= y <= 20

def test_mask_to_polygon_too_small():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[5:7, 5:7] = 1  # 4 px — below min_area=64
    assert mask_to_polygon(m) is None

def test_mask_to_polygon_binary_01():
    # Accept 0/1 masks, not just 0/255
    m = np.zeros((20, 20), dtype=np.uint8)
    m[2:12, 2:12] = 1
    pts = mask_to_polygon(m)
    assert pts is not None

def test_mask_to_polygon_float32():
    m = np.zeros((20, 20), dtype=np.float32)
    m[2:12, 2:12] = 1.0
    pts = mask_to_polygon(m)
    assert pts is not None

def test_mask_to_polygon_empty():
    assert mask_to_polygon(np.zeros((20, 20), dtype=np.uint8)) is None
