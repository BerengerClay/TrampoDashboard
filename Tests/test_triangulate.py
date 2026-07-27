import os
import pytest
import numpy as np
from src.utils.triangulate import (
    computeP,
    retrieve_calib_params,
    extract_keypoints_from_preds,
)

CALIB_PATH = os.path.abspath("configs/Calib.toml")

def test_compute_p():
    if not os.path.exists(CALIB_PATH):
        pytest.skip("Calib.toml not found")

    cams = ["M11139", "M11140"]
    P_matrices = computeP(CALIB_PATH, undistort=False, cams=cams)
    
    assert len(P_matrices) == 2
    for P in P_matrices:
        assert P.shape == (3, 4)
        assert not np.isnan(P).any()

def test_compute_p_undistort():
    if not os.path.exists(CALIB_PATH):
        pytest.skip("Calib.toml not found")

    cams = ["M11139"]
    P_matrices = computeP(CALIB_PATH, undistort=True, cams=cams)
    assert len(P_matrices) == 1
    assert P_matrices[0].shape == (3, 4)

def test_retrieve_calib_params():
    if not os.path.exists(CALIB_PATH):
        pytest.skip("Calib.toml not found")

    cams = ["M11139", "M11140"]
    params = retrieve_calib_params(CALIB_PATH, cams=cams)
    
    assert "S" in params
    assert "K" in params
    assert "dist" in params
    assert "R_mat" in params
    assert "T" in params
    
    assert len(params["S"]) == 2
    assert params["K"][0].shape == (3, 3)
    assert params["R_mat"][0].shape == (3, 3)

def test_extract_keypoints_from_preds():
    preds = [
        {
            "img_path": "/data/seq1-Camera1/frame_0000.png",
            "pred_instances": {
                "keypoints": np.ones((1, 17, 2), dtype=np.float32) * 100.0,
                "keypoint_scores": np.ones((1, 17), dtype=np.float32) * 0.9,
            },
        },
        {
            "img_path": "/data/seq1-Camera2/frame_0000.png",
            "pred_instances": {
                "keypoints": np.ones((1, 17, 2), dtype=np.float32) * 150.0,
                "keypoint_scores": np.ones((1, 17), dtype=np.float32) * 0.85,
            },
        },
    ]

    extracted = extract_keypoints_from_preds(preds, cams=["seq1-Camera1", "seq1-Camera2"], use_gt=False)
    assert len(extracted) == 1  # 1 sequence
    keypoints_tuple, frames, base_path, seq = extracted[0]
    
    x_files, y_files, likelihood_files = keypoints_tuple
    assert len(frames) == 1
    assert seq == "seq1"
    assert len(x_files) == 1  # 1 frame
