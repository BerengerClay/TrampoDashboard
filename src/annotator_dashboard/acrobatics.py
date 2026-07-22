import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize(v):
    """Normalizes array of vectors along last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-6, norm)
    return v / norm

def calculate_local_frame(Z_target, X_target_rough):
    """
    Computes a strict orthonormal local coordinate frame [X, Y, Z].
    Z_target: head-hip axis (upward/longitudinal)
    X_target_rough: right-left hip axis
    """
    Z_t = normalize(Z_target)
    X_t = np.cross(X_target_rough, Z_t)
    X_t = normalize(X_t)
    Y_t = np.cross(Z_t, X_t)
    return X_t, Y_t, Z_t

def calculate_acrobatics_rotations(X_t, Y_t, Z_t, num_frames):
    """
    Calculates continuous cumulative somersaults (Salto) and twists (Vrille) without Gimbal Lock,
    without curve drops during twists, and referenced to stable flip orientation.
    - Vrille: incremental rotation of lateral hip axis Y around longitudinal body axis Z.
    - Salto: incremental rotation magnitude of body axis Z signed by un-twisted lateral axis Y_ref.
    """
    d_salto = np.zeros(num_frames)
    d_vrille = np.zeros(num_frames)

    cum_vrille = 0.0

    for i in range(1, num_frames):
        if np.isnan(Z_t[i-1]).any() or np.isnan(Z_t[i]).any() or np.isnan(Y_t[i-1]).any() or np.isnan(Y_t[i]).any():
            continue

        # 1. Vrille: rotation of Y around Z
        cross_y = np.cross(Y_t[i-1], Y_t[i])
        dot_y = np.clip(np.dot(Y_t[i-1], Y_t[i]), -1.0, 1.0)
        d_vrille[i] = np.arctan2(np.dot(cross_y, Z_t[i]), dot_y)
        cum_vrille += d_vrille[i]

        # Un-rotate Y[i] by accumulated vrille angle around Z[i] to get stable anatomical flip axis
        rot_z = R.from_rotvec(-cum_vrille * Z_t[i])
        Y_ref = rot_z.apply(Y_t[i])

        # 2. Salto: magnitude of Z rotation signed by stable Y_ref (prevents drops during vrilles)
        cross_z = np.cross(Z_t[i-1], Z_t[i])
        dot_z = np.clip(np.dot(Z_t[i-1], Z_t[i]), -1.0, 1.0)
        mag_z = np.linalg.norm(cross_z)
        ang_z = np.arctan2(mag_z, dot_z)

        proj_y = np.dot(cross_z, Y_ref)
        sign_s = np.sign(proj_y) if abs(proj_y) > 1e-6 else 1.0

        d_salto[i] = sign_s * ang_z

    saltos_turns = np.cumsum(d_salto) / (2.0 * np.pi)
    vrilles_turns = np.cumsum(d_vrille) / (2.0 * np.pi)

    return saltos_turns, vrilles_turns

def detect_trampoline_impacts(coords, ankle_idxs=(15, 16), knee_idxs=(13, 14)):
    """Detects trampoline impact frames using dynamic vertical axis detection."""
    num_frames = coords.shape[0]
    l_ankle_idx, r_ankle_idx = ankle_idxs
    l_knee_idx, r_knee_idx = knee_idxs

    # Dynamically detect vertical axis (Y=1 or Z=2) based on coordinate variance
    std_y = np.nanstd(coords[:, :, 1])
    std_z = np.nanstd(coords[:, :, 2])
    v_axis = 1 if std_y >= std_z else 2

    ankles_height = (coords[:, l_ankle_idx, v_axis] + coords[:, r_ankle_idx, v_axis]) / 2.0
    knees_height = (coords[:, l_knee_idx, v_axis] + coords[:, r_knee_idx, v_axis]) / 2.0

    bed_height = np.nanpercentile(ankles_height, 2)
    contact_margin = 0.40 if bed_height < 50.0 else 400.0
    contact_threshold = bed_height + contact_margin
    in_contact = ankles_height < contact_threshold

    transitions = np.diff(in_contact.astype(int))
    contact_starts = np.where(transitions == 1)[0] + 1
    contact_ends = np.where(transitions == -1)[0] + 1

    if in_contact[0]:
        contact_starts = np.insert(contact_starts, 0, 0)
    if in_contact[-1]:
        contact_ends = np.append(contact_ends, num_frames - 1)

    raw_impacts = []
    for start, end in zip(contact_starts, contact_ends):
        if end > start:
            window = knees_height[start : end + 1]
            if len(window) > 0:
                raw_impacts.append(start + np.nanargmin(window))

    filtered_impacts = []
    min_frames_between = 15
    for p in raw_impacts:
        if not filtered_impacts:
            filtered_impacts.append(p)
        elif p - filtered_impacts[-1] < min_frames_between:
            if knees_height[p] < knees_height[filtered_impacts[-1]]:
                filtered_impacts[-1] = p
        else:
            filtered_impacts.append(p)

    impacts = np.array(filtered_impacts)
    if len(impacts) == 0 or impacts[0] > 10:
        impacts = np.insert(impacts, 0, 0)
    if len(impacts) > 0 and impacts[-1] < num_frames - 1:
        impacts = np.append(impacts, num_frames - 1)

    return impacts

def calculate_angle(v1, v2):
    """Computes angle in degrees between vector batches (N, 3)."""
    v1_n = normalize(v1)
    v2_n = normalize(v2)
    dot = np.sum(v1_n * v2_n, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def detect_acrobatic_position(coords, idx_dict=None):
    """
    Detects Tuck, Pike, Straight positions based on joint angles.
    coords: array of shape (num_frames, 17, 3)
    """
    if idx_dict is None:
        idx_dict = {
            "shoulder": [5, 6],
            "hip": [11, 12],
            "knee": [13, 14],
            "ankle": [15, 16],
        }

    shoulder = np.mean(coords[:, idx_dict["shoulder"], :], axis=1)
    hip = np.mean(coords[:, idx_dict["hip"], :], axis=1)
    knee = np.mean(coords[:, idx_dict["knee"], :], axis=1)
    ankle = np.mean(coords[:, idx_dict["ankle"], :], axis=1)

    v_hip_shoulder = shoulder - hip
    v_hip_knee = knee - hip
    hip_angles = calculate_angle(v_hip_shoulder, v_hip_knee)

    v_knee_hip = hip - knee
    v_knee_ankle = ankle - knee
    knee_angles = calculate_angle(v_knee_hip, v_knee_ankle)

    results = []
    for h_ang, k_ang in zip(hip_angles, knee_angles):
        position = "Transition"
        base_deduction = 0.0
        knee_deduction = 0.0

        if h_ang >= 135 and k_ang >= 135:
            position = "Straight"
            if h_ang < 170:
                base_deduction = 0.1 if h_ang >= 150 else 0.2
            if k_ang < 170:
                knee_deduction = 0.1 if k_ang >= 150 else 0.2
        elif h_ang < 135 and k_ang >= 135:
            position = "Pike"
            if h_ang > 55:
                base_deduction = 0.1 if h_ang <= 90 else 0.2
            if k_ang < 170:
                knee_deduction = 0.1 if k_ang >= 150 else 0.2
        elif h_ang < 135 and k_ang < 135:
            position = "Tuck"
            max_ang = max(h_ang, k_ang)
            if max_ang > 55:
                base_deduction = 0.1 if max_ang <= 90 else 0.2

        if base_deduction + knee_deduction > 0.5:
            knee_deduction = 0.5 - base_deduction

        results.append(
            {
                "position": position,
                "base_deduction": round(base_deduction, 1),
                "knee_deduction": round(knee_deduction, 1),
            }
        )

    return results, hip_angles, knee_angles

def calculate_acrobatics_summary(coords_3d):
    """
    Computes per-jump Somersaults, Twists, impacts, and Postures matching src_old/visualize/dashboard.py.
    """
    num_frames = len(coords_3d)
    
    pelvis_center = (coords_3d[:, 12, :] + coords_3d[:, 11, :]) / 2.0
    head_hip_axis = coords_3d[:, 0, :] - pelvis_center
    hip_axis = coords_3d[:, 12, :] - coords_3d[:, 11, :]

    head_hip_norm = normalize(head_hip_axis)
    hip_norm = normalize(hip_axis)

    X_t, Y_t, Z_t = calculate_local_frame(head_hip_norm, hip_norm)
    saltos_turns, vrilles_turns = calculate_acrobatics_rotations(X_t, Y_t, Z_t, num_frames)

    impacts = detect_trampoline_impacts(coords_3d)

    s_turns = np.array(saltos_turns)
    v_turns = np.array(vrilles_turns)
    saltos_per_jump = np.zeros_like(s_turns)
    vrilles_per_jump = np.zeros_like(v_turns)

    if len(impacts) >= 2:
        for i in range(len(impacts) - 1):
            s, e = impacts[i], impacts[i + 1]
            saltos_per_jump[s:e] = np.abs(s_turns[s:e] - s_turns[s])
            vrilles_per_jump[s:e] = np.abs(v_turns[s:e] - v_turns[s])
        saltos_per_jump[-1] = np.abs(s_turns[-1] - s_turns[impacts[-2]])
        vrilles_per_jump[-1] = np.abs(v_turns[-1] - v_turns[impacts[-2]])
    else:
        saltos_per_jump = np.abs(s_turns - s_turns[0])
        vrilles_per_jump = np.abs(v_turns - v_turns[0])

    saltos_cumul = np.abs(s_turns - s_turns[0])
    vrilles_cumul = np.abs(v_turns - v_turns[0])

    acro_dict = {
        "shoulder": [5, 6],
        "hip": [11, 12],
        "knee": [13, 14],
        "ankle": [15, 16],
    }
    acro_results, _, _ = detect_acrobatic_position(coords_3d, acro_dict)

    return saltos_per_jump, vrilles_per_jump, saltos_cumul, vrilles_cumul, impacts, acro_results


def format_fig_trampoline_code(salto_turns, vrille_turns, posture="Tuck"):
    """
    Converts Somersault (Salto) turns, Twist (Vrille) turns, and posture into official FIG Trampoline Short Code.
    Examples:
    - 3.0 Saltos, 0.5 Vrilles, Tuck     -> FIG: 12001o (Triffis)
    - 2.0 Saltos, 0.5 Vrilles, Tuck     -> FIG: 801o (Double Half Out)
    - 1.5 Saltos, 1.0 Vrille, Pike       -> FIG: 611<
    - 1.0 Salto, 0.5 Vrille, Straight   -> FIG: 41/ (Barani)
    - 1.0 Salto, 1.0 Vrille, Straight   -> FIG: 42/ (Full)
    """
    q_salto = int(round(salto_turns * 4))
    h_vrille = int(round(vrille_turns * 2))

    if q_salto == 0:
        return ""

    posture_code = "o" if "Tuck" in str(posture) else ("<" if "Pike" in str(posture) else "/")

    # Single Salto (4 quarters)
    if q_salto <= 5:
        if h_vrille == 0:
            code = f"{q_salto}-"
        else:
            code = f"{q_salto}{h_vrille}"
        return f"{code}{posture_code}"

    # Double Salto (8 quarters)
    elif q_salto <= 9:
        if h_vrille == 0:
            code = "800"
        elif h_vrille == 1:
            code = "801"  # Half Out / Barani Out
        elif h_vrille == 2:
            code = "811"  # Half In Half Out
        elif h_vrille == 3:
            code = "821"  # Full In Half Out
        else:
            code = f"8{h_vrille}"
        return f"{code}{posture_code}"

    # Triple Salto (12 quarters)
    else:
        if h_vrille == 0:
            code = "12000"
        elif h_vrille == 1:
            code = "12001"  # Triffis / Triple Half Out
        elif h_vrille == 2:
            code = "12011"
        else:
            code = f"12_{h_vrille}"
        return f"{code}{posture_code}"
