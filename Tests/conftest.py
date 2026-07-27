import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Force Qt offscreen rendering for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure root directory, src directory, and annotator_dashboard directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
dashboard_dir = src_dir / "annotator_dashboard"

for p in [str(root_dir), str(src_dir), str(dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

@pytest.fixture(scope="session")
def qapp():
    """Provides a persistent QApplication instance for testing PyQt6 widgets and graphics items."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture
def sample_3d_trajectory():
    """Generates a synthetic 3D pose trajectory (20 frames, 17 keypoints, 3D coordinates)."""
    n_frames = 20
    n_kpts = 17
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, n_frames)
    coords = np.zeros((n_frames, n_kpts, 3), dtype=np.float64)
    for k in range(n_kpts):
        coords[:, k, 0] = np.sin(t) * 100.0 + k * 10.0 + np.random.normal(0, 1.0, n_frames)
        coords[:, k, 1] = np.cos(t) * 100.0 + k * 5.0 + np.random.normal(0, 1.0, n_frames)
        coords[:, k, 2] = t * 50.0 + np.random.normal(0, 1.0, n_frames)
    return coords

@pytest.fixture
def sample_coco_pose_coords():
    """
    Generates a single-frame COCO pose (17 keypoints) positioned in a standard standing position (Straight posture).
    Coordinates in mm.
    """
    coords = np.zeros((1, 17, 3), dtype=np.float64)
    coords[0, 0] = [0, 0, 1700]
    coords[0, 1] = [-30, 0, 1720]
    coords[0, 2] = [30, 0, 1720]
    coords[0, 3] = [-60, 0, 1710]
    coords[0, 4] = [60, 0, 1710]
    coords[0, 5] = [-200, 0, 1500]
    coords[0, 6] = [200, 0, 1500]
    coords[0, 7] = [-250, 0, 1200]
    coords[0, 8] = [250, 0, 1200]
    coords[0, 9] = [-280, 0, 900]
    coords[0, 10] = [280, 0, 900]
    coords[0, 11] = [-100, 0, 1000]
    coords[0, 12] = [100, 0, 1000]
    coords[0, 13] = [-100, 0, 500]
    coords[0, 14] = [100, 0, 500]
    coords[0, 15] = [-100, 0, 0]
    coords[0, 16] = [100, 0, 0]
    return coords
