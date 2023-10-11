import json
import logging
import os
import zipfile
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).parent.parent
PKG_DATA = Path.home() / ".monlam_ocr"
MODELS_DIR_NAME = "Models"
MODELS_PATH = PKG_DATA / MODELS_DIR_NAME
MODELS_VERSION_TAG = os.environ.get("MODELS_VERSION_TAG", "v0.0.2")


def download_models():
    token = os.environ["GITHUB_TOKEN"]
    asset_name = "Models.zip"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # 1. Get the list of releases
    releases_url = "https://api.github.com/repos/OpenPecha/monlam_ocr/releases"
    releases = requests.get(releases_url, headers=headers).json()

    # find asset id from the specified release with tag
    asset_id = None
    found_release = False
    for release in releases:
        if release["tag_name"] != MODELS_VERSION_TAG:
            continue
        for asset in release["assets"]:
            if asset["name"] == asset_name:
                asset_id = asset["id"]
                found_release = True
                break

        if found_release:
            break

    if asset_id is None:
        print("Asset not found!")
        return

    # 2. Download the asset using its ID
    asset_url = (
        f"https://api.github.com/repos/OpenPecha/monlam_ocr/releases/assets/{asset_id}"
    )
    headers["Accept"] = "application/octet-stream"
    response = requests.get(asset_url, headers=headers, stream=True)

    with open(asset_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile("Models.zip", "r") as zip_ref:
        zip_ref.extractall(str(PKG_DATA))
    os.remove("Models.zip")
    logging.info(f"Models is saved at {MODELS_PATH}")

    return MODELS_PATH


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
    download_models()

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
