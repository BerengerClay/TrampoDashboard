import pytest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor
from src.annotator_dashboard.items import KeypointItem, SkeletonItem, BBoxItem

def test_keypoint_item_init(qapp):
    kp = KeypointItem(x=100.0, y=150.0, point_id=0, name="nose", parent_widget=None, kv=0.95)
    assert kp.pos() == QPointF(100.0, 150.0)
    assert kp.point_id == 0
    assert kp.name == "nose"
    assert kp.kv == 0.95
    assert "nose" in kp.toolTip()

def test_keypoint_item_set_radius(qapp):
    kp = KeypointItem(x=10.0, y=20.0, point_id=5, name="left_shoulder", parent_widget=None, kv=1.0)
    kp.set_radius(10.0)
    assert kp.rect() == QRectF(-10.0, -10.0, 20.0, 20.0)

def test_skeleton_item_update_position(qapp):
    kp1 = KeypointItem(x=10.0, y=20.0, point_id=5, name="left_shoulder", parent_widget=None, kv=1.0)
    kp2 = KeypointItem(x=50.0, y=60.0, point_id=7, name="left_elbow", parent_widget=None, kv=1.0)
    
    skel = SkeletonItem(kp1, kp2, color=QColor(0, 200, 255))
    line = skel.line()
    assert line.x1() == 10.0
    assert line.y1() == 20.0
    assert line.x2() == 50.0
    assert line.y2() == 60.0

def test_bbox_item_init(qapp):
    rect = QRectF(10.0, 20.0, 100.0, 200.0)
    bbox = BBoxItem(rect, parent_widget=None)
    assert bbox.rect() == rect
    assert "top_left" in bbox.handles
    assert "bottom_right" in bbox.handles
