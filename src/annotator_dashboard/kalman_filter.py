"""
Linear 3D Constant-Velocity Kalman Filter for Pose Trajectories & Imputation.
State vector: 6D x = [x, y, z, vx, vy, vz]^T
Parameters:
- process_noise (Q): 1e-2 default (Q[:3,:3] = 0.1 * Q, Q[3:,3:] = Q)
- measurement_noise (R): 1e-1 default (3x3 diagonal matrix)
- P_init: 10.0 * I_6
- dt: 1.0 (unit time step per frame)
Handles missing marker coordinates (NaN) by performing pure motion state prediction and imputation.
"""
import numpy as np

def apply_kalman_filter(coords_3d, process_noise=1e-4, measurement_noise=2e-3, dt=1.0, use_rts_smoothing=True):
    """
    Applies 6D Constant-Velocity Kalman Filtering & NaN Imputation to 3D keypoint trajectories.

    INPUTS:
    - coords_3d: np.ndarray of shape (n_frames, n_keypoints, 3)
    - process_noise: float, process noise scale Q (default: 0.0001)
    - measurement_noise: float, measurement noise scale R (default: 0.002)
    - dt: float, frame time step (default: 1.0)
    - use_rts_smoothing: bool, whether to apply Rauch-Tung-Striebel backward pass

    OUTPUTS:
    - filtered_coords: np.ndarray of shape (n_frames, n_keypoints, 3)
    """
    if coords_3d is None or len(coords_3d) == 0:
        return coords_3d

    n_frames, n_kpts, _ = coords_3d.shape
    if n_frames < 2:
        return coords_3d.copy()

    dt_val = float(dt)
    filtered_coords = coords_3d.copy()

    # 6D State Transition Matrix F: x_t = F * x_{t-1}
    F = np.array([
        [1.0, 0.0, 0.0, dt_val, 0.0,    0.0],
        [0.0, 1.0, 0.0, 0.0,    dt_val, 0.0],
        [0.0, 0.0, 1.0, 0.0,    0.0,    dt_val],
        [0.0, 0.0, 0.0, 1.0,    0.0,    0.0],
        [0.0, 0.0, 0.0, 0.0,    1.0,    0.0],
        [0.0, 0.0, 0.0, 0.0,    0.0,    1.0]
    ], dtype=np.float64)

    # 3x6 Measurement Matrix H: z_t = H * x_t
    H = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64)

    # 6x6 Process Noise Matrix Q (position block set to 0.1 * process_noise)
    Q = np.zeros((6, 6), dtype=np.float64)
    Q[:3, :3] = np.eye(3) * (0.1 * float(process_noise))
    Q[3:, 3:] = np.eye(3) * float(process_noise)

    # 3x3 Measurement Noise Matrix R
    R = np.eye(3) * float(measurement_noise)

    for k in range(n_kpts):
        pts = coords_3d[:, k, :]
        valid_mask = ~np.isnan(pts).any(axis=1)

        if not np.any(valid_mask):
            continue

        first_v = int(np.argmax(valid_mask))

        x_pred = np.zeros((n_frames, 6))
        P_pred = np.zeros((n_frames, 6, 6))
        x_filt = np.zeros((n_frames, 6))
        P_filt = np.zeros((n_frames, 6, 6))

        # Initial state x0 and initial covariance P0 = 10.0 * I_6
        x_init = np.array([pts[first_v, 0], pts[first_v, 1], pts[first_v, 2], 0.0, 0.0, 0.0])
        P_init = np.eye(6) * 10.0

        x_filt[first_v] = x_init
        P_filt[first_v] = P_init

        # Forward Kalman Filter Loop
        for t in range(first_v + 1, n_frames):
            # Predict step
            x_p = F @ x_filt[t-1]
            P_p = F @ P_filt[t-1] @ F.T + Q
            x_pred[t] = x_p
            P_pred[t] = P_p

            # Measurement update or prediction imputation for missing NaN
            z = pts[t]
            if not np.isnan(z).any():
                y = z - H @ x_p
                S = H @ P_p @ H.T + R
                K = P_p @ H.T @ np.linalg.inv(S)
                x_f = x_p + K @ y
                P_f = (np.eye(6) - K @ H) @ P_p
            else:
                x_f = x_p
                P_f = P_p

            x_filt[t] = x_f
            P_filt[t] = P_f

        if use_rts_smoothing:
            # Backward RTS Smoothing Pass
            x_smooth = np.zeros((n_frames, 6))
            x_smooth[-1] = x_filt[-1]

            for t in range(n_frames - 2, first_v - 1, -1):
                det = np.linalg.det(P_pred[t+1])
                if abs(det) < 1e-12:
                    x_smooth[t] = x_filt[t]
                    continue
                C = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t+1])
                x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - x_pred[t+1])

            filtered_coords[first_v:, k, :] = x_smooth[first_v:, :3]
        else:
            filtered_coords[first_v:, k, :] = x_filt[first_v:, :3]

        if first_v > 0:
            filtered_coords[:first_v, k, :] = filtered_coords[first_v, k, :]

    return filtered_coords

# Backward compatibility alias
def apply_kalman_smoothing_3d(coords_3d, process_noise_q=1e-4, measurement_noise_r=2e-3, fps=30.0):
    return apply_kalman_filter(coords_3d, process_noise=process_noise_q, measurement_noise=measurement_noise_r, dt=1.0)
