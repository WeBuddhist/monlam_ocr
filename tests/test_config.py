import tempfile
from pathlib import Path
from unittest.mock import patch

from monlam_ocr.config import get_models_path


def generate_fake_model_files(models_path):
    # setup line model path
    line_model_path = models_path / "LineModels"
    line_model_path.mkdir(parents=True, exist_ok=True)
    line_model_config_path = line_model_path / "line_model_config.json"
    line_model_config_path.write_text("{}")

    # setup ocr model path
    ocr_model_path = models_path / "OCRModels" / "DergeTenjur"
    ocr_model_path.mkdir(parents=True, exist_ok=True)
    ocr_model_config_path = ocr_model_path / "ocr_model_config.json"
    ocr_model_config_path.write_text("{}")


@patch("monlam_ocr.config.download_models")
def test_get_models_path(mock_download_models):
    with tempfile.TemporaryDirectory() as tmpdirname:
        models_path = Path(tmpdirname)
        generate_fake_model_files(models_path)
        mock_download_models.return_value = models_path
        line_model_config_path, ocr_model_config_path = get_models_path()

        assert line_model_config_path.exists()
        assert ocr_model_config_path.exists()
