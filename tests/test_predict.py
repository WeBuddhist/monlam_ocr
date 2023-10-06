from monlam_ocr.predict import run_ocr


def test_predict():
    image_path = "tests/data/test.jpg"
    predected_text = run_ocr(image_path)
    assert len(predected_text[0]) == 6
