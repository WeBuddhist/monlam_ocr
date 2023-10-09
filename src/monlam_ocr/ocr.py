import json
import pyewts
import numpy as np
import onnxruntime as ort
from fast_ctc_decode import viterbi_search
from monlam_ocr.utils import prepare_ocr_image


converter = pyewts.pyewts()

class OCRInference:
    def __init__(self, config_file, mode: str = "cpu") -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self._can_run = False
        self.ocr_session = None
        self.mode = mode
        self._init()

    def _init(self) -> None:
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())

        if self.mode == "cuda":
            self._onnx_model_file = json_content["gpu-model"]
        else:
            self._onnx_model_file = json_content["cpu-model"]

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._add_channel_dim = json_content["add_channel_dim"]
        self.charset = json_content["charset"]

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=execution_providers
        )

    def run(self, line_images: list, replace_blank: str = ""):
        img_batch = [
            prepare_ocr_image(
                x, target_width=self._input_width, target_height=self._input_height
            )
            for x in line_images
        ]
        img_batch = np.array(img_batch)

        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )
        prediction = ocr_results[0].numpy()
        prediction = np.transpose(prediction, axes=[1, 0, 2])
        # print(f"Raw prediction: {prediction.shape}")

        predicted_text = []
        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            text, _ = viterbi_search(pred_line, self.charset)
            bo_text = converter.toUnicode(text)

            if len(text) > 0:
                predicted_text.append(bo_text)
            else:
                predicted_text.append("")

        return predicted_text, prediction