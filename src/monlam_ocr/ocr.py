import json
import os

import numpy as np
import onnxruntime as ort
import pyewts
from fast_ctc_decode import viterbi_search

from monlam_ocr.exceptions import FailedToLoadModel
from monlam_ocr.utils import prepare_ocr_image

converter = pyewts.pyewts()


class OCRInference:
    def __init__(self, config_file) -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self.ocr_session = None
        self.execution_providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        self._init()

    def _init(self) -> None:
        model_dir = os.path.dirname(self.config)

        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())
        self._onnx_model_file = f"{model_dir}/{json_content['model']}"

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._squeeze_channel_dim = (
            True if json_content["squeeze_channel_dim"] == "yes" else False
        )
        self._swap_hw = True if json_content["swap_hw"] == "yes" else False
        self._characters = self.get_charset(json_content["charset"])
        print(len(self._characters))

        try:
            self.ocr_session = ort.InferenceSession(
                self._onnx_model_file, providers=self.execution_providers
            )
        except Exception as e:
            raise FailedToLoadModel(f"Failed to load model: {e}")

    def get_charset(self, charset: str) -> str:
        return f"ß{charset} "

    def run(self, line_images: list, replace_tsheg: str = " "):

        img_batch = [
            prepare_ocr_image(
                x, target_width=self._input_width, target_height=self._input_height
            )
            for x in line_images
        ]
        img_batch = np.array(img_batch)

        if self._squeeze_channel_dim:
            img_batch = np.squeeze(img_batch, axis=1)

        if self._swap_hw:
            img_batch = np.transpose(img_batch, axes=[0, 2, 1])
        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )
        prediction = ocr_results[0].numpy()

        if not self._swap_hw:
            prediction = np.transpose(prediction, axes=[1, 0, 2])

        predicted_text = []

        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            text, _ = viterbi_search(pred_line, self._characters)

            if len(text) > 0:
                text = text.replace(" ", "")
                text = text.replace("§", replace_tsheg)
                bo_text = converter.toUnicode(text)
                predicted_text.append(bo_text)
            else:
                predicted_text.append("")

        return predicted_text, prediction
