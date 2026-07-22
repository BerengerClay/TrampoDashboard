import os
import numpy as np
import pandas as pd

def expand_header(header_line):
    new_header = []
    i = 0
    j = 0
    while i < len(header_line):
        entry = header_line[i]
        if entry not in ['', None]:
            if entry in ['Frame#', 'Time']:
                new_header.append(entry)
                i += 1
            else:
                new_header.extend([f"X{j}_{entry}", f"Y{j}_{entry}", f"Z{j}_{entry}"])
                j += 1
                i += 3  # Skip the next 2 empty strings
        else:
            i += 1  # Just skip if it's empty
    return new_header

def extract_coordinates(filename, to_mm=True, return_time=False):
    """
    Extract joint coordinates from a .trc file and return them as a numpy array of shape (frames, keypoints, 3).
        - filename: path to the .trc file
        - to_mm: whether to convert coordinates from meters to millimeters (default: True)
        - return_time: whether to return the 'Time' column array as a 4th return item (default: False)
    Returns:
        - coords: numpy array of shape (frames, keypoints, 3) contains coordinates
        - frame_numbers: numpy array containing the frame numbers
        - marker_names: list of marker names
        - (Optional) times: numpy array of timestamps (seconds)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Extraire les noms de colonnes depuis la 4e ligne (index 3)
    column_names = lines[3].strip().split('\t')
    column_names = expand_header(column_names)

    # Charger le reste des données en DataFrame
    df = pd.read_csv(filename, 
                    sep='\t', 
                    skiprows=5, 
                    names=column_names)
    df = df.dropna(axis=1, how='all')  # Supprimer les colonnes complètement vides

    # Extraire les noms des marqueurs
    marker_names, marker_indices = [], []
    for name in column_names[2:]:  # Ignorer 'Frame#' et 'Time'
        if name.startswith('X'):
            marker_indices.append(name[1:].split('_')[0])
            marker_names.append(name.split('_')[-1])

    # -- Convertir en numpy array (frames, keypoints, 3) --
    num_frames = df.shape[0]
    num_markers = len(marker_names)
    coords = np.zeros((num_frames, num_markers, 3))

    for (i, marker_id), marker_name in zip(enumerate(marker_indices), marker_names):
        coords[:, i, 0] = df[f'X{marker_id}_{marker_name}'].values
        coords[:, i, 1] = df[f'Y{marker_id}_{marker_name}'].values
        coords[:, i, 2] = df[f'Z{marker_id}_{marker_name}'].values
    
    if to_mm:
        coords *= 1000 #convert to mm

    frame_numbers = df['Frame#'].values
    times = df['Time'].values

    if return_time:
        return coords, frame_numbers, marker_names, times
    return coords, frame_numbers, marker_names


def save_trc_file(filename, coords, fps=30.0, marker_names=None):
    """
    Saves a 3D coordinate array of shape (frames, keypoints, 3) back to a standard .trc file.
    """
    if marker_names is None:
        marker_names = [
            "Nose", "L Eye", "R Eye", "L Ear", "R Ear",
            "L Shoulder", "R Shoulder", "L Elbow", "R Elbow",
            "L Wrist", "R Wrist", "L Hip", "R Hip",
            "L Knee", "R Knee", "L Ankle", "R Ankle"
        ]
    num_frames, num_markers, _ = coords.shape
    dt = 1.0 / float(fps)
    
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(filename)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:g}\t{fps:g}\t{num_frames}\t{num_markers}\tm\t{fps:g}\t0\t{num_frames}\n")
        
        header1 = ["Frame#", "Time"]
        for m in marker_names:
            header1.extend([m, "", ""])
        f.write("\t".join(header1) + "\n")
        
        header2 = ["", ""]
        for i in range(1, num_markers + 1):
            header2.extend([f"X{i}", f"Y{i}", f"Z{i}"])
        f.write("\t".join(header2) + "\n")
        
        for frame_idx in range(num_frames):
            time_val = frame_idx * dt
            row = [str(frame_idx), f"{time_val:.4f}"]
            for m_idx in range(num_markers):
                pt = coords[frame_idx, m_idx]
                if np.isnan(pt).any():
                    row.extend(["nan", "nan", "nan"])
                else:
                    row.extend([f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])
            f.write("\t".join(row) + "\n")
