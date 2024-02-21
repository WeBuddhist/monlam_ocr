"""
A minimalistic interface to run OCR on a given set of images stored locally.
See the Predict Notebook for a simple walkthrough.

Note: As of now the line_dilation has to be parameterized until this part of the
pipeline has been more generalized. See examples in the Notebook for choosing parameters.

Run e.g. python predict.py --input_dir "Data/W30125" --line_dilate 10 --line_iteration 10
"""

import cv2

from monlam_ocr.exceptions import ImageReadError, InvalidDimensionError
from monlam_ocr.line_detection import LineDetection
from monlam_ocr.ocr import OCRInference


def run_ocr(
    image_path: str, line_inference: LineDetection, ocr_inference: OCRInference
) -> None:
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ImageReadError(f"Failed to load image from {image_path}")
        (
            back_sized_image,
            line_images,
            sorted_contours,
            bbox,
            peaks,
        ) = line_inference.predict(image, 0)
        predicted_text, prediction = ocr_inference.run(line_images)
    except cv2.error as e:
        raise ImageReadError(f"OpenCV error in processing {image_path}: {e}")
    except IndexError as e:
        raise InvalidDimensionError(f"Dimension error in processing {image_path}: {e}")
    return predicted_text
