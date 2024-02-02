import cv2
import argparse
import logging
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from monlam_ocr.ocr import OCRInference
from monlam_ocr.utils import binarize_line

model_db = {
    "LhasaKanjur" : "Models/OCRModels/LhasaKanjur/model_config.json",
    "DergeTenjur" : "Models/OCRModels/DergeTenjur/model_config.json"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-f", "--format", choices=["jpg", "tif", "tiff"], required=False, default="jpg")
    parser.add_argument("-b", "--binarize", choices=["yes", "no"], required=False, default="no")
    parser.add_argument("-m", "--model", choices=["LhasaKanjur", "DergeTenjur"], required=False, default="LhasaKanjur")

    args = parser.parse_args()
    input_dir = args.input_dir
    input_format = args.format
    input_binarize = True if args.binarize == "yes" else False
    model_config = model_db[args.model] # maybe wrap this into proper exception handling etc.

    logging.info("Loading OCR from Model: {ocr_model}...")

    ocr_inference = OCRInference(model_config)

    line_images = natsorted(glob(f"{input_dir}/*.{input_format}"))

    if len(line_images) > 0:
        img_batch = [cv2.imread(x) for x in line_images]

        if input_binarize:
            img_batch = [binarize_line(x) for x in img_batch]

        predicted_text, _ = ocr_inference.run(img_batch)

        print(predicted_text)

 