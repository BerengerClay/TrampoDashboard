import os
import pytest
import numpy as np
from src.annotator_dashboard.read_trc_files import (
    expand_header,
    extract_coordinates,
    save_trc_file,
)

def test_expand_header():
    header_input = ["Frame#", "Time", "Nose", "", "", "L_Eye", "", ""]
    expanded = expand_header(header_input)
    assert expanded == [
        "Frame#",
        "Time",
        "X0_Nose",
        "Y0_Nose",
        "Z0_Nose",
        "X1_L_Eye",
        "Y1_L_Eye",
        "Z1_L_Eye",
    ]

def test_trc_roundtrip(tmp_path, sample_3d_trajectory):
    trc_path = os.path.join(tmp_path, "test_output.trc")
    marker_names = [f"Marker_{i}" for i in range(sample_3d_trajectory.shape[1])]
    
    # Coordinates in save_trc_file are expected in meters (or standard units)
    coords_m = sample_3d_trajectory / 1000.0
    save_trc_file(trc_path, coords_m, fps=60.0, marker_names=marker_names)
    
    assert os.path.exists(trc_path)
    
    # Read back with to_mm=True
    loaded_coords, frames, loaded_markers, times = extract_coordinates(
        trc_path, to_mm=True, return_time=True
    )
    
    assert loaded_coords.shape == sample_3d_trajectory.shape
    assert len(frames) == sample_3d_trajectory.shape[0]
    assert len(loaded_markers) == len(marker_names)
    assert len(times) == sample_3d_trajectory.shape[0]
    
    # Verify values are close (considering floating point format in text file)
    np.testing.assert_allclose(loaded_coords, sample_3d_trajectory, rtol=1e-4, atol=1e-3)

def test_trc_to_mm_false(tmp_path, sample_3d_trajectory):
    trc_path = os.path.join(tmp_path, "test_output_m.trc")
    coords_m = sample_3d_trajectory / 1000.0
    save_trc_file(trc_path, coords_m, fps=30.0)
    
    loaded_coords, _, _ = extract_coordinates(trc_path, to_mm=False, return_time=False)
    np.testing.assert_allclose(loaded_coords, coords_m, rtol=1e-4, atol=1e-3)

def test_trc_with_nan(tmp_path):
    trc_path = os.path.join(tmp_path, "test_nan.trc")
    coords = np.array([
        [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]],
        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    ])
    marker_names = ["M1", "M2"]
    save_trc_file(trc_path, coords, fps=30.0, marker_names=marker_names)
    
    loaded_coords, _, _ = extract_coordinates(trc_path, to_mm=False)
    assert np.isnan(loaded_coords[0, 1]).all()
    assert not np.isnan(loaded_coords[1, 1]).any()
    np.testing.assert_allclose(loaded_coords[1, 1], [7.0, 8.0, 9.0], rtol=1e-4)
