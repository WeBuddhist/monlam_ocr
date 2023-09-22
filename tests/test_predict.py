import os
from monlam_ocr.predict import run_ocr



def test_predict():
    image_path = "tests/data/test.jpg"
    line_model_config_path = "Models/LineModels/line_model_config.json"
    ocr_model_config_path = "Models/OCRModels/LhasaKanjur/model_config.json"
    predected_text = run_ocr(image_path, ocr_model_config_path, line_model_config_path)
    assert len(predected_text[0]) == 6