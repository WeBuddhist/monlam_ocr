import cv2

from monlam_ocr.exceptions import FailedToLoadImage
from monlam_ocr.ocr import OCRInference


def run_line_ocr(image_path: str, ocr_inference: OCRInference) -> None:
    image = [cv2.imread(image_path)]
    if image is None:
        FailedToLoadImage(f"Failed to load image from {image_path}")

    predicted_text, prediction = ocr_inference.run(image)
    return predicted_text
