import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def setup_models_path():

    # setup line model path
    line_model_path = ROOT_DIR / "Models" / "LineModels"
    line_model_config_path = line_model_path / "line_model_config.json"
    line_model_config = json.load(open(line_model_config_path))
    resolved_line_model_path = ROOT_DIR / Path(line_model_config["model"])
    line_model_config["model"] = str(resolved_line_model_path)
    assert resolved_line_model_path.exists()
    json.dump(line_model_config, open(line_model_config_path, "w"))

    # setup ocr model path
    ocr_model_path = ROOT_DIR / "Models" / "OCRModels" / "LhasaKanjur"
    ocr_model_config_path = ocr_model_path / "model_config.json"
    ocr_model_config = json.load(open(ocr_model_config_path))
    resolved_ocr_model_path = ROOT_DIR / Path(ocr_model_config["model"])
    ocr_model_config["model"] = str(resolved_ocr_model_path)
    assert resolved_ocr_model_path.exists()
    json.dump(ocr_model_config, open(ocr_model_config_path, "w"))


setup_models_path()
