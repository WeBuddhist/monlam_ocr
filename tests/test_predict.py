from monlam_ocr.config import get_models_path
from monlam_ocr.line_detection import LineDetection
from monlam_ocr.ocr import OCRInference
from monlam_ocr.predict import run_ocr



def test_predict():
    image_path = "tests/data/test.jpg"
    line_model_config_path, ocr_model_config_path = get_models_path()
    line_inference = LineDetection(line_model_config_path)
    ocr_inference = OCRInference(ocr_model_config_path)
    predected_text = run_ocr(image_path, line_inference, ocr_inference)
    assert len(predected_text) == 6

