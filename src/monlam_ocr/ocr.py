import json
import numpy as np
import onnxruntime as ort
from fast_ctc_decode import viterbi_search
from monlam_ocr.utils import preprocess_image_tf


class OCRInference:
    def __init__(self, config_file) -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self._can_run = False
        self.ocr_session = None

        self._init()

    def _init(self) -> None:
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())
        self._onnx_model_file = json_content["model"]
        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self.charset = self.get_model_characters(json_content["charset"])
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=["CPUExecutionProvider"]
        )

    def get_model_characters(self, characters):
        characters = [c for c in characters]
        characters.append("[BLK]")
        characters.insert(0, "[UNK]")

        return characters

    def run(self, line_images: list, replace_blank: str = ""):
        """
        See comment in Utils.py. So preprocess_image_tf() is used for time being.
        img_batch = [prepare_ocr_image(x) for x in line_images]
        """
        img_batch = [preprocess_image_tf(x) for x in line_images]
        img_batch = np.array(img_batch)
        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            ["Final"], {"the_input": ort_batch}
        )
        prediction = ocr_results[0].numpy()

        predicted_text = []
        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            text, _ = viterbi_search(pred_line, self.charset)
            text = text.replace("[BLK]", replace_blank)

            if len(text) > 0:
                predicted_text.append(text)
            else:
                predicted_text.append("")

        return predicted_text, prediction