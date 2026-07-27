import pytest
import numpy as np
import torch
import os
from src.annotator_dashboard.backend import ModelWrapper

def test_model_wrapper_init():
    wrapper = ModelWrapper(weights_dir="/tmp/test_weights", device="cpu")
    assert wrapper.weights_dir == "/tmp/test_weights"
    assert wrapper.device == "cpu"
    assert wrapper.yolo_model is None
    assert wrapper.vitpose_model is None

def test_model_wrapper_missing_weights_raises():
    wrapper = ModelWrapper(yolo_path="/non_existent/yolo.pt", vitpose_path="/non_existent/vitpose.pth")
    with pytest.raises(FileNotFoundError):
        wrapper.init_yolo()
    with pytest.raises(FileNotFoundError):
        wrapper.init_vitpose()

def test_resize_and_pad_keep_aspect():
    wrapper = ModelWrapper()
    # Create dummy crop of shape 100x200x3 (H=100, W=200)
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    resized_padded, scale, (pad_left, pad_top) = wrapper.resize_and_pad_keep_aspect(crop, target_size=(256, 192))
    
    assert resized_padded.shape == (256, 192, 3)
    assert scale > 0
    assert pad_left >= 0
    assert pad_top >= 0

def test_map_keypoints_to_bbox():
    wrapper = ModelWrapper()
    keypoints = torch.tensor([128.0, 96.0])  # Center of model input
    scale = 0.5
    pad = (14, 16)
    
    x_bbox, y_bbox = wrapper.map_keypoints_to_bbox(keypoints, scale, pad)
    
    assert torch.isclose(x_bbox, torch.tensor((128.0 - 14.0) / 0.5))
    assert torch.isclose(y_bbox, torch.tensor((96.0 - 16.0) / 0.5))
