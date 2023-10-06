"""
A minimalistic interface to run OCR on a given set of images stored locally.
See the Predict Notebook for a simple walkthrough.

Note: As of now the line_dilation has to be parameterized until this part of the
pipeline has been more generalized. See examples in the Notebook for choosing parameters.

Run e.g. python predict.py --input_dir "Data/W30125" --line_dilate 10 --line_iteration 10
"""

import cv2

from monlam_ocr.config import get_models_path
from monlam_ocr.line_detection import LineDetection
from monlam_ocr.ocr import OCRInference


def run_ocr(
    image_path: str,
    ocr_model_config_path: str = None,
    line_model_config_path: str = None,
    line_dilate: int = 32,
    line_iteration: int = 30,
) -> None:
    if ocr_model_config_path is None or line_model_config_path is None:
        line_model_config_path, ocr_model_config_path = get_models_path()

    image = cv2.imread(image_path)
    line_inference = LineDetection(
        config_file=line_model_config_path,
        dilate_kernel=line_dilate,
        dilate_iterations=line_iteration,
        binarize_output=False,
    )
    ocr_inference = OCRInference(config_file=ocr_model_config_path)
    prediction, line_images, sorted_contours, peaks = line_inference.predict(image, 0)
    predicted_text = ocr_inference.run(line_images)
    return predicted_text
