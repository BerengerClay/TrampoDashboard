import pytest
import numpy as np
from src.annotator_dashboard.acrobatics import (
    normalize,
    calculate_local_frame,
    calculate_acrobatics_rotations,
    detect_trampoline_impacts,
    calculate_angle,
    detect_acrobatic_position,
    calculate_acrobatics_summary,
    format_fig_trampoline_code,
)

def test_normalize():
    v = np.array([3.0, 4.0, 0.0])
    normed = normalize(v)
    np.testing.assert_allclose(normed, [0.6, 0.8, 0.0])
    assert np.isclose(np.linalg.norm(normed), 1.0)

    # Zero vector handling
    zero_v = np.array([0.0, 0.0, 0.0])
    normed_zero = normalize(zero_v)
    assert not np.isnan(normed_zero).any()

def test_calculate_local_frame():
    Z_target = np.array([0.0, 0.0, 1.0])
    X_rough = np.array([1.0, 0.0, 0.0])
    
    X_body, Y_body, Z_body = calculate_local_frame(Z_target, X_rough)
    
    # Check orthogonality
    assert np.isclose(np.dot(X_body, Y_body), 0.0)
    assert np.isclose(np.dot(Y_body, Z_body), 0.0)
    assert np.isclose(np.dot(X_body, Z_body), 0.0)
    
    # Check unit lengths
    assert np.isclose(np.linalg.norm(X_body), 1.0)
    assert np.isclose(np.linalg.norm(Y_body), 1.0)
    assert np.isclose(np.linalg.norm(Z_body), 1.0)

def test_calculate_angle():
    v1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v2 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    
    angles = calculate_angle(v1, v2)
    np.testing.assert_allclose(angles, [0.0, 180.0])

def test_detect_acrobatic_position_straight(sample_coco_pose_coords):
    results, hip_angles, knee_angles = detect_acrobatic_position(sample_coco_pose_coords)
    assert len(results) == 1
    assert results[0]["position"] == "Straight"
    assert results[0]["base_deduction"] == 0.0
    assert results[0]["knee_deduction"] == 0.0

def test_detect_acrobatic_position_tuck():
    # Construct a posture where hip_angle < 135 and knee_angle < 135
    coords = np.zeros((1, 17, 3), dtype=np.float64)
    # Shoulder at (0, 0, 1000), Hip at (0, 0, 500), Knee at (300, 0, 500), Ankle at (300, 0, 800)
    coords[0, 5] = [-100, 0, 1000]
    coords[0, 6] = [100, 0, 1000]
    coords[0, 11] = [-100, 0, 500]
    coords[0, 12] = [100, 0, 500]
    coords[0, 13] = [-100, 300, 500]
    coords[0, 14] = [100, 300, 500]
    coords[0, 15] = [-100, 300, 800]
    coords[0, 16] = [100, 300, 800]
    
    results, hip_angles, knee_angles = detect_acrobatic_position(coords)
    assert results[0]["position"] == "Tuck"

def test_format_fig_trampoline_code():
    # Single Salto, no twist, Tuck
    assert format_fig_trampoline_code(1.0, 0.0, posture="Tuck") == "4-o"
    
    # Single Salto, 1/2 twist (1 half-twist), Pike
    assert format_fig_trampoline_code(1.0, 0.5, posture="Pike") == "41<"
    
    # Double Salto (8 quarters), 1/2 twist in 2nd salto, Straight
    assert format_fig_trampoline_code(2.0, 0.5, posture="Straight") == "801/"

    # Double Salto, 1 full twist (2 half-twists) in 1st salto, 1/2 twist in 2nd salto, Tuck
    vrilles_per_salto = [1.0, 0.5]
    assert format_fig_trampoline_code(2.0, 1.5, posture="Tuck", vrilles_per_salto=vrilles_per_salto) == "821o"

    # Zero salto -> empty string
    assert format_fig_trampoline_code(0.0, 0.0, posture="Straight") == ""

def test_detect_trampoline_impacts():
    n_frames = 50
    coords = np.zeros((n_frames, 17, 3), dtype=np.float64)
    # Ankle height dips at frame 20 (simulating impact with trampoline bed)
    z_height = np.sin(np.linspace(0, np.pi, n_frames)) * 1000.0 + 50.0
    coords[:, 15, 2] = z_height
    coords[:, 16, 2] = z_height
    coords[:, 13, 2] = z_height + 400.0
    coords[:, 14, 2] = z_height + 400.0

    impacts = detect_trampoline_impacts(coords)
    assert isinstance(impacts, np.ndarray)
    assert len(impacts) >= 2
    assert impacts[0] == 0
    assert impacts[-1] == n_frames - 1

def test_calculate_acrobatics_summary(sample_3d_trajectory):
    s_jump, v_jump, s_cumul, v_cumul, impacts, acro_res = calculate_acrobatics_summary(sample_3d_trajectory)
    
    assert len(s_jump) == len(sample_3d_trajectory)
    assert len(v_jump) == len(sample_3d_trajectory)
    assert len(s_cumul) == len(sample_3d_trajectory)
    assert len(v_cumul) == len(sample_3d_trajectory)
    assert len(acro_res) == len(sample_3d_trajectory)
    assert isinstance(impacts, np.ndarray)
