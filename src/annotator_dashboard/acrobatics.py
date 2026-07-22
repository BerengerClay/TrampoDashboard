import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize(v):
    """Normalizes array of vectors along last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-6, norm)
    return v / norm

def calculate_local_frame(Z_target, X_target_rough):
    """
    Computes a strict orthonormal local coordinate frame [X_body, Y_body, Z_body]:
    - X_body (col 0): Lateral hip-to-hip axis (Salto / Somersault / Pitch)
    - Y_body (col 1): Antero-Posterior axis (Front-Back / Roll)
    - Z_body (col 2): Longitudinal pelvis-to-head axis (Vrille / Twist / Yaw)
    """
    Z_body = normalize(Z_target)
    X_raw = normalize(X_target_rough)
    Y_body = normalize(np.cross(Z_body, X_raw))
    X_body = normalize(np.cross(Y_body, Z_body))
    return X_body, Y_body, Z_body

def calculate_acrobatics_rotations(X_t, Y_t, Z_t, num_frames):
    """
    Calculates cumulative somersaults and twists using strict body frame Euler angles (XYZ):
    - angles[0]: rotation around X_t (Lateral axis -> Somersault / Salto)
    - angles[2]: rotation around Z_t (Longitudinal axis -> Twist / Vrille)
    """
    saltos_history = []
    vrilles_history = []

    last_salto = 0.0
    last_vrille = 0.0

    for i in range(num_frames):
        frame_curr = np.column_stack((X_t[i], Y_t[i], Z_t[i]))

        if np.isnan(frame_curr).any():
            saltos_history.append(last_salto)
            vrilles_history.append(last_vrille)
        else:
            R_curr = R.from_matrix(frame_curr)
            angles = R_curr.as_euler("XYZ")

            last_salto = angles[0]
            last_vrille = angles[2]

            saltos_history.append(last_salto)
            vrilles_history.append(last_vrille)

    saltos_turns = np.unwrap(saltos_history) / (2 * np.pi)
    vrilles_turns = np.unwrap(vrilles_history) / (2 * np.pi)

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
            raw_s = np.abs(s_turns[s:e] - s_turns[s])
            raw_v = np.abs(v_turns[s:e] - v_turns[s])
            saltos_per_jump[s:e] = np.maximum.accumulate(raw_s)
            vrilles_per_jump[s:e] = np.maximum.accumulate(raw_v)
        
        last_s = impacts[-2]
        saltos_per_jump[-1] = max(saltos_per_jump[-2], np.abs(s_turns[-1] - s_turns[last_s]))
        vrilles_per_jump[-1] = max(vrilles_per_jump[-2], np.abs(v_turns[-1] - v_turns[last_s]))
    else:
        saltos_per_jump = np.maximum.accumulate(np.abs(s_turns - s_turns[0]))
        vrilles_per_jump = np.maximum.accumulate(np.abs(v_turns - v_turns[0]))

    saltos_cumul = np.maximum.accumulate(np.abs(s_turns - s_turns[0]))
    vrilles_cumul = np.maximum.accumulate(np.abs(v_turns - v_turns[0]))

    acro_dict = {
        "shoulder": [5, 6],
        "hip": [11, 12],
        "knee": [13, 14],
        "ankle": [15, 16],
    }
    acro_results, _, _ = detect_acrobatic_position(coords_3d, acro_dict)

    return saltos_per_jump, vrilles_per_jump, saltos_cumul, vrilles_cumul, impacts, acro_results


def format_fig_trampoline_code(salto_turns, vrille_turns, posture="Tuck", vrilles_per_salto=None):
    """
    Dynamically converts Somersault (Salto) turns, Twist (Vrille) turns, and posture into official FIG Trampoline Short Code.
    Generates codes dynamically (e.g. 41o, 801o, 803o, 812o, 821o, 12001o) without hardcoded cases.
    """
    q_salto = int(round(salto_turns * 4))
    h_total = int(round(vrille_turns * 2))

    if q_salto == 0:
        return ""

    posture_code = "o" if "Tuck" in str(posture) else ("<" if "Pike" in str(posture) else "/")

    # Single Salto (4 quarters = 1 salto)
    if q_salto <= 5:
        if h_total == 0:
            code = f"{q_salto}-"
        else:
            code = f"{q_salto}{h_total}"
        return f"{code}{posture_code}"

    # Multiple Saltos (Double = 8 quarters, Triple = 12 quarters, Quad = 16 quarters)
    n_saltos = max(2, int(round(q_salto / 4)))
    
    if vrilles_per_salto is not None and len(vrilles_per_salto) >= (n_saltos - 1):
        h_list = []
        rem_h = h_total
        for k in range(n_saltos - 1):
            v_k = float(vrilles_per_salto[k])
            hk = int(round(v_k * 2))
            hk = max(0, min(rem_h, hk))
            h_list.append(hk)
            rem_h -= hk
        h_list.append(max(0, rem_h))
        vrille_digits = "".join(str(h) for h in h_list)
    else:
        # Default fallback: attribute all twists to the last salto
        h_list = [0] * (n_saltos - 1) + [h_total]
        vrille_digits = "".join(str(h) for h in h_list)

    return f"{q_salto}{vrille_digits}{posture_code}"
