import pytest
from src.annotator_dashboard.constants import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
    KEYPOINT_COLORS,
    CAMERA_KEYS,
)
from PyQt6.QtGui import QColor

def test_coco_keypoints_count_and_names():
    assert len(COCO_KEYPOINTS) == 17
    assert COCO_KEYPOINTS[0] == "nose"
    assert COCO_KEYPOINTS[5] == "left_shoulder"
    assert COCO_KEYPOINTS[6] == "right_shoulder"
    assert COCO_KEYPOINTS[15] == "left_ankle"
    assert COCO_KEYPOINTS[16] == "right_ankle"

def test_coco_skeleton_structure():
    assert len(COCO_SKELETON) > 0
    # Verify all indices in skeleton pairs are valid keypoint indices (0..16)
    for k1, k2 in COCO_SKELETON:
        assert 0 <= k1 < 17
        assert 0 <= k2 < 17

def test_keypoint_colors_mapping():
    assert len(KEYPOINT_COLORS) == 17
    for idx in range(17):
        assert idx in KEYPOINT_COLORS
        assert isinstance(KEYPOINT_COLORS[idx], QColor)

def test_camera_keys_list():
    assert len(CAMERA_KEYS) == 8
    assert "Camera1_M11139" in CAMERA_KEYS
    assert "Camera8_M11463" in CAMERA_KEYS
