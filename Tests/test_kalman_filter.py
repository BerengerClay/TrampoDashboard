import pytest
import numpy as np
from src.annotator_dashboard.kalman_filter import (
    apply_kalman_filter,
    apply_kalman_smoothing_3d,
)

def test_kalman_filter_basic_shape(sample_3d_trajectory):
    filtered = apply_kalman_filter(sample_3d_trajectory)
    assert filtered.shape == sample_3d_trajectory.shape
    assert not np.isnan(filtered).any()

def test_kalman_filter_none_and_empty():
    assert apply_kalman_filter(None) is None
    empty_arr = np.array([])
    assert apply_kalman_filter(empty_arr).size == 0

def test_kalman_filter_single_frame():
    single_frame = np.ones((1, 17, 3), dtype=np.float64) * 50.0
    res = apply_kalman_filter(single_frame)
    assert res.shape == single_frame.shape
    np.testing.assert_array_almost_equal(res, single_frame)

def test_kalman_filter_all_nans():
    all_nans = np.full((10, 17, 3), np.nan)
    res = apply_kalman_filter(all_nans)
    assert res.shape == all_nans.shape
    assert np.isnan(res).all()

def test_kalman_filter_nan_imputation(sample_3d_trajectory):
    # Introduce NaN gaps in middle of trajectory
    corrupted = sample_3d_trajectory.copy()
    corrupted[5:10, 3, :] = np.nan
    
    filtered = apply_kalman_filter(corrupted, use_rts_smoothing=True)
    
    assert not np.isnan(filtered[:, 3, :]).any()
    # Ensure imputed values are reasonable (not zeros or infs)
    assert np.all(np.abs(filtered[5:10, 3, :]) < 1000.0)

def test_kalman_filter_rts_smoothing_flag(sample_3d_trajectory):
    filt_rts = apply_kalman_filter(sample_3d_trajectory, use_rts_smoothing=True)
    filt_forward = apply_kalman_filter(sample_3d_trajectory, use_rts_smoothing=False)

    assert filt_rts.shape == sample_3d_trajectory.shape
    assert filt_forward.shape == sample_3d_trajectory.shape
    assert not np.isnan(filt_rts).any()
    assert not np.isnan(filt_forward).any()
    # RTS smoothing should produce smoother trajectory (different results from forward only)
    assert not np.array_equal(filt_rts, filt_forward)

def test_kalman_filter_parameters(sample_3d_trajectory):
    filtered_custom = apply_kalman_filter(
        sample_3d_trajectory,
        process_noise=1e-2,
        measurement_noise=1e-1,
        dt=0.5,
        use_rts_smoothing=True,
    )
    assert filtered_custom.shape == sample_3d_trajectory.shape
    assert not np.isnan(filtered_custom).any()

def test_apply_kalman_smoothing_3d_alias(sample_3d_trajectory):
    res = apply_kalman_smoothing_3d(sample_3d_trajectory, process_noise_q=1e-3, measurement_noise_r=1e-2)
    assert res.shape == sample_3d_trajectory.shape
    assert not np.isnan(res).any()
