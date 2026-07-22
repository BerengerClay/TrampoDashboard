#!/usr/bin/env python3
"""
Utility script to run YOLO in Python to detect bounding boxes,
generate a temporary COCO JSON, run ViTPose via standard MMPose test.py,
and then run Pose2Sim 3D triangulation.
"""

import os
import sys
import json
import pickle
import subprocess
import toml
import numpy as np

def load_env():
    env_vars = {}
    env_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env_vars[k.strip()] = v.strip()
    return env_vars

def load_local_settings():
    settings_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "local_settings.json"))
    if not os.path.exists(settings_path):
        settings_path = "configs/local_settings.json"
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def generate_predictions_and_triangulate(camera_paths, pkl_output, config_file, calib_file, trc_output, gt_file=None):
    env_vars = load_env()
    local_settings = load_local_settings()

    # 1. Resolve YOLO weights path (.env -> local_settings.json -> weights/ candidates)
    yolo_path = env_vars.get("YOLO_WEIGHTS_PATH")
    if not yolo_path or not os.path.exists(yolo_path):
        yolo_path = local_settings.get("yolo_path")
    if not yolo_path or not os.path.exists(yolo_path):
        candidates = [
            "weights/YOLO26s_best.pt",
            "src/weights/YOLO26s_best.pt",
            "weights/yolov8s.pt"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                yolo_path = cand
                break

    if not yolo_path or not os.path.exists(yolo_path):
        raise FileNotFoundError(
            "YOLO weights file not found. "
            "Please specify 'yolo_path' in configs/local_settings.json, set YOLO_WEIGHTS_PATH in .env, or place weights in 'weights/'."
        )

    # 2. Resolve ViTPose weights path (.env -> local_settings.json -> weights/ candidates)
    vitpose_weights_path = env_vars.get("VITPOSE_WEIGHTS_PATH")
    if not vitpose_weights_path or not os.path.exists(vitpose_weights_path):
        vitpose_weights_path = local_settings.get("vitpose_path")
    if not vitpose_weights_path or not os.path.exists(vitpose_weights_path):
        candidates = [
            "weights/best_ViTPose-s_AP731.pth",
            "weights/best_mvssl_AP713_iter290.pth",
            "weights/best_coco_AP_epoch_298_AP0705.pth"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                vitpose_weights_path = cand
                break

    if not vitpose_weights_path or not os.path.exists(vitpose_weights_path):
        raise FileNotFoundError(
            "ViTPose weights file not found. "
            "Please specify 'vitpose_path' in configs/local_settings.json, set VITPOSE_WEIGHTS_PATH in .env, or place weights in 'weights/'."
        )

    # 1. Resolve camera folders and files
    camera_folders = sorted(list(camera_paths))
    cams = [os.path.basename(p.rstrip('/')) for p in camera_folders]
    seq_name = cams[0].split('-Camera')[0]
    
    print(f"Sequence: {seq_name}", flush=True)
    print(f"Cameras: {cams}", flush=True)
    
    all_files_set = set()
    for cam_folder in camera_folders:
        if os.path.isdir(cam_folder):
            for f in os.listdir(cam_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files_set.add(f)
    all_files = sorted(list(all_files_set))
    print(f"Total unique frames across all cameras: {len(all_files)}", flush=True)
    
    # 2. Run YOLO to get accurate bounding boxes
    print("--- 1. Running YOLO to detect bounding boxes ---", flush=True)
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"YOLO running on device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}", flush=True)
    
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    
    yolo_model = YOLO(yolo_path)
    yolo_model.to(device)
    
    image_paths_to_run = []
    for cam_folder in camera_folders:
        for filename in all_files:
            full_p = os.path.join(cam_folder, filename)
            if os.path.exists(full_p):
                image_paths_to_run.append(full_p)
            
    total_yolo_images = len(image_paths_to_run)
    print(f"Detecting bounding boxes for {total_yolo_images} images...", flush=True)
    batch_size = 64
    total_yolo_batches = (total_yolo_images + batch_size - 1) // batch_size
    detected_bboxes = {}
    default_bbox = [300.0, 100.0, 1300.0, 950.0]
    
    def load_single_image(path):
        try:
            img = cv2.imread(path)
            return path, img
        except Exception:
            return path, None
            
    for idx, i in enumerate(range(0, total_yolo_images, batch_size)):
        batch = image_paths_to_run[i:i+batch_size]
        print(f"[YOLO] Loading batch {idx+1}/{total_yolo_batches} in parallel (images {i} to {min(i+batch_size, total_yolo_images)})...", flush=True)
        
        # Parallel image loading
        with ThreadPoolExecutor(max_workers=8) as executor:
            load_results = list(executor.map(load_single_image, batch))
            
        batch_paths = []
        batch_imgs = []
        for path, img in load_results:
            if img is not None:
                batch_paths.append(path)
                batch_imgs.append(img)
            else:
                detected_bboxes[path] = default_bbox
                
        if not batch_imgs:
            continue
            
        print(f"[YOLO] Running GPU detection on batch {idx+1}/{total_yolo_batches}...", flush=True)
        results = yolo_model(batch_imgs, verbose=False, conf=0.20, device=device)
        for path, res in zip(batch_paths, results):
            bbox = default_bbox
            if len(res.boxes) > 0:
                best_idx = 0
                best_score = -1e9
                h_orig, w_orig = res.orig_shape
                center_x, center_y = w_orig / 2.0, h_orig / 2.0
                diag = np.sqrt(w_orig**2 + h_orig**2)
                
                for b_idx in range(len(res.boxes)):
                    xyxy = res.boxes.xyxy[b_idx].cpu().numpy()
                    conf = float(res.boxes.conf[b_idx])
                    box_cx = (xyxy[0] + xyxy[2]) / 2.0
                    box_cy = (xyxy[1] + xyxy[3]) / 2.0
                    dist = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2) / diag
                    # Priority score: high confidence + closeness to image center (where athlete is located)
                    score = conf - 0.6 * dist
                    if score > best_score:
                        best_score = score
                        best_idx = b_idx

                xyxy = res.boxes.xyxy[best_idx].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x1 = max(0.0, min(float(w_orig), float(x1)))
                y1 = max(0.0, min(float(h_orig), float(y1)))
                x2 = max(0.0, min(float(w_orig), float(x2)))
                y2 = max(0.0, min(float(h_orig), float(y2)))
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    bbox = [x1, y1, w, h]
            detected_bboxes[path] = bbox

    # Clean up YOLO model and release GPU memory before running ViTPose
    del yolo_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
    # Load GT bboxes dynamically if gt_file parameter or nearby GT json is available
    gt_bbox_map = {}
    gt_candidates = []
    if gt_file and os.path.exists(gt_file):
        gt_candidates.append(gt_file)
    
    # Auto-scan parent directory of first camera folder for any .json GT files
    first_cam_folder = camera_folders[0]
    parent_dir = os.path.dirname(first_cam_folder)
    if os.path.exists(parent_dir):
        for candidate in os.listdir(parent_dir):
            if candidate.endswith('.json') and not candidate.startswith('temp_'):
                gt_candidates.append(os.path.join(parent_dir, candidate))

    for gt_cand in gt_candidates:
        if os.path.exists(gt_cand):
            try:
                with open(gt_cand, 'r') as gf:
                    gt_j = json.load(gf)
                id_map = {img['id']: img['file_name'] for img in gt_j.get('images', [])}
                for ann in gt_j.get('annotations', []):
                    fn = id_map.get(ann['image_id'], '')
                    if fn and 'bbox' in ann and ann['bbox'] and ann['bbox'][2] > 0:
                        gt_bbox_map[fn] = ann['bbox']
                        gt_bbox_map[os.path.basename(fn)] = ann['bbox']
                        gt_bbox_map[fn.replace('\\', '/')] = ann['bbox']
                        parts = fn.replace('\\', '/').replace('-', '/').split('/')
                        c_p = [p for p in parts if 'Camera' in p]
                        f_p = [p for p in parts if 'frame_' in p]
                        if c_p and f_p:
                            gt_bbox_map[f"{c_p[0]}/{f_p[0]}"] = ann['bbox']
                            gt_bbox_map[f"{c_p[0]}-{f_p[0]}"] = ann['bbox']
            except Exception:
                pass

    # 3. Generate COCO JSON
    print("--- 2. Creating temporary COCO JSON ---", flush=True)
    images_list = []
    annotations_list = []
    image_id = 1
    ann_id = 1
    abs_data_root = os.path.abspath("Data")
    
    for filename in all_files:
        for cam_folder in camera_folders:
            full_path = os.path.join(cam_folder, filename)
            if not os.path.exists(full_path):
                continue
            rel_path = os.path.relpath(full_path, abs_data_root)
            cam_b = os.path.basename(cam_folder)
            
            # Check GT bbox map first
            bbox = None
            for key in [f"{cam_b}/{filename}", f"{cam_b}-{filename}", full_path, filename]:
                if key in gt_bbox_map:
                    bbox = gt_bbox_map[key]
                    break
            if bbox is None:
                bbox = detected_bboxes.get(full_path, default_bbox)
            
            # Standard COCO keypoints dummy values to prevent dataset loader from filtering them out
            dummy_kpts = []
            for _ in range(17):
                dummy_kpts.extend([960.0, 540.0, 1])
                
            images_list.append({
                "id": image_id,
                "file_name": rel_path,
                "width": 1920,
                "height": 1080
            })
            
            annotations_list.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "keypoints": dummy_kpts,
                "num_keypoints": 17,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            image_id += 1
            ann_id += 1
            
    # Define complete COCO categories structure
    categories = [{
        'id': 1, 
        'name': 'person', 
        'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'], 
        'skeleton': []
    }]

    coco_data = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories
    }
    
    temp_json_path = "Data/temp_coco.json"
    os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)
    with open(temp_json_path, 'w') as jf:
        json.dump(coco_data, jf)
        
    # 4. Run MMPose test.py
    print("--- 3. Running ViTPose via MMPose test.py ---", flush=True)
    python_bin = sys.executable
    
    cfg_candidate = os.path.join("configs", "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py")
    if not os.path.exists(cfg_candidate):
        cfg_candidate = "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"
    abs_config_file = os.path.abspath(cfg_candidate)
    abs_weights = os.path.abspath(vitpose_weights_path)
    abs_temp_json = os.path.abspath(temp_json_path)
    abs_dump = os.path.abspath("temp_predictions.pkl")
    mmpose_dir = os.path.abspath("mmpose_src")
    
    args = [
        "-u",
        "-s",
        "tools/test.py",
        abs_config_file,
        abs_weights,
        "--cfg-options",
        f"test_dataloader.dataset.data_root={abs_data_root}",
        f"test_dataloader.dataset.ann_file={abs_temp_json}",
        "test_dataloader.dataset.data_prefix.img=",
        f"test_evaluator.0.ann_file={abs_temp_json}",
        "test_dataloader.batch_size=32",
        "test_dataloader.num_workers=2",
        "--dump", abs_dump
    ]
    
    print(f"Command: {python_bin} " + " ".join(args), flush=True)
    
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    process = subprocess.Popen(
        [python_bin] + args,
        cwd=mmpose_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip(), flush=True)
            
    rc = process.poll()
    if rc != 0:
        raise RuntimeError(f"ViTPose inference failed with exit code {rc}")
        
    # Move temp_predictions.pkl to pkl_output
    if os.path.exists("temp_predictions.pkl"):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(pkl_output)), exist_ok=True)
        shutil.move("temp_predictions.pkl", pkl_output)
        
    # Clean up temp JSON
    try:
        os.remove(temp_json_path)
    except Exception:
        pass
        
    # 5. Run Triangulation
    print("--- 4. Running Pose2Sim Triangulation ---", flush=True)
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from utils.triangulate import run_triangulation
    
    run_triangulation(pkl_output, config_file, calib_file, trc_output, n_cams=8)
    print(f"Triangulation complete! Saved to {trc_output}", flush=True)


if __name__ == '__main__':
    env_vars = load_env()
    pkl_output = env_vars.get("PKL_FILE_PATH", "output/predictions.pkl")
    trc_output = env_vars.get("TRC_OUTPUT_PATH", "output/pose-3d/triangulated.trc")
    import Pose2Sim
    pose2sim_dir = os.path.dirname(Pose2Sim.__file__)
    default_config = os.path.join(pose2sim_dir, "Demo_SinglePerson", "Config.toml")
    config_file = env_vars.get("CONFIG_FILE_PATH", default_config)
    
    calib_file = env_vars.get("CALIB_FILE_PATH", "configs/Calib.toml")
    
    gt_arg = None
    camera_paths = []
    skip_next = False
    args_list = sys.argv[1:]
    for i, arg in enumerate(args_list):
        if skip_next:
            skip_next = False
            continue
        if arg in ['--gt', '--gt-path']:
            if i + 1 < len(args_list):
                gt_arg = args_list[i + 1]
                skip_next = True
        elif not arg.startswith('--'):
            if os.path.isdir(arg):
                camera_paths.append(arg)

    if camera_paths:
        first_dir = os.path.basename(camera_paths[0].rstrip('/'))
        seq_name = first_dir.split('-Camera')[0]
        pkl_output = f"output/{seq_name}/predictions.pkl"
        trc_output = f"output/{seq_name}/pose-3d/triangulated.trc"
    else:
        import glob
        camera_paths = sorted(glob.glob("Data/*Camera*"))
        if not camera_paths:
            print("Usage: python3 generate_predictions.py Data/sequence-Camera* [--gt GT_FILE.json]")
            sys.exit(1)
        
    generate_predictions_and_triangulate(camera_paths, pkl_output, config_file, calib_file, trc_output, gt_file=gt_arg)
