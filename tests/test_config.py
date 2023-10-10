import json
from pathlib import Path

import pytest

from monlam_ocr.config import MODELS_PATH, fix_models_path, get_models_path


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("/root/machine/Models/model.onnx", MODELS_PATH / "model.onnx"),
        ("Models/model.onnx", MODELS_PATH / "model.onnx"),
    ],
)
def test_fix_models_path(test_input, expected):
    fixed_model_path = fix_models_path(test_input)
    assert str(expected) == fixed_model_path


def test_get_models_path():
    line_model_config_path, ocr_model_config_path = get_models_path()
    assert line_model_config_path.exists()
    assert str(line_model_config_path).startswith("/")
    assert ocr_model_config_path.exists()
    assert str(ocr_model_config_path).startswith("/")

    line_model_config = json.load(open(line_model_config_path))
    assert Path(line_model_config["cpu-model"]).exists()
    assert line_model_config["cpu-model"].startswith("/")
    assert Path(line_model_config["gpu-model"]).exists()
    assert line_model_config["gpu-model"].startswith("/")
    assert Path(line_model_config["model"]).exists()
    assert line_model_config["model"].startswith("/")

    ocr_model_config = json.load(open(ocr_model_config_path))
    assert Path(ocr_model_config["cpu-model"]).exists()
    assert ocr_model_config["cpu-model"].startswith("/")
    assert Path(ocr_model_config["gpu-model"]).exists()
    assert ocr_model_config["gpu-model"].startswith("/")
    assert Path(ocr_model_config["model"]).exists()
    assert ocr_model_config["model"].startswith("/")
