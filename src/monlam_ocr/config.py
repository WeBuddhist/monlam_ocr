import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR_NAME = "Models"
MODELS_PATH = ROOT_DIR / MODELS_DIR_NAME


def fix_models_path(path: str) -> str:
    path = str(path)
    if path.startswith("/"):
        _, model_path = path.split("/" + MODELS_DIR_NAME + "/")
    elif path.startswith(MODELS_DIR_NAME):
        model_path = path[len(MODELS_DIR_NAME) + 1 :]  # noqa E203
    else:
        raise ValueError(
            f"Model path should start with '/' or '{MODELS_DIR_NAME}' but got {path}"
        )
    return str(MODELS_PATH / model_path)


def get_models_path():
    # setup line model path
    line_model_path = MODELS_PATH / "LineModels"
    line_model_config_path = line_model_path / "line_model_config.json"
    line_model_config = json.load(open(line_model_config_path))

    line_model_config["cpu-model"] = fix_models_path(line_model_config["cpu-model"])
    assert Path(line_model_config["cpu-model"]).exists()
    line_model_config["gpu-model"] = fix_models_path(line_model_config["gpu-model"])
    assert Path(line_model_config["gpu-model"]).exists()
    line_model_config["model"] = fix_models_path(line_model_config["model"])
    assert Path(line_model_config["model"]).exists()
    json.dump(line_model_config, open(line_model_config_path, "w"))

    # setup ocr model path
    ocr_model_path = MODELS_PATH / "OCRModels" / "LhasaKanjur"
    ocr_model_config_path = ocr_model_path / "ocr_model_config.json"
    ocr_model_config = json.load(open(ocr_model_config_path))
    ocr_model_config["cpu-model"] = fix_models_path(ocr_model_config["cpu-model"])
    assert Path(ocr_model_config["cpu-model"]).exists()
    ocr_model_config["gpu-model"] = fix_models_path(ocr_model_config["gpu-model"])
    assert Path(ocr_model_config["gpu-model"]).exists()
    ocr_model_config["model"] = fix_models_path(ocr_model_config["model"])
    assert Path(ocr_model_config["model"]).exists()
    json.dump(ocr_model_config, open(ocr_model_config_path, "w"))

    return line_model_config_path, ocr_model_config_path
