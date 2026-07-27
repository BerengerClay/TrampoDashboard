import os
import json
import pytest
from src.utils.generate_predictions import load_env, load_local_settings

def test_load_env():
    env_vars = load_env()
    assert isinstance(env_vars, dict)

def test_load_local_settings():
    settings = load_local_settings()
    assert isinstance(settings, dict)
