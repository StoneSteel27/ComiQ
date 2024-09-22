import numpy as np
from paddleocr import PaddleOCR
import easyocr


def perform_ocr(image: np.ndarray, methods: list[str]) -> list[dict]:
    """Perform OCR using specified methods."""
    results = []
    for method in methods:
        if method == "paddleocr":
            results += detect_text_pad(image)
        elif method == "easyocr":
            results += detect_text_easy(image)
    return results


def detect_text_pad(image: np.ndarray) -> None:
    """Detect text using PaddleOCR."""
    # Implementation of PaddleOCR

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        det_limit_side_len=2560,
        det_db_thresh=0.1,
        det_db_box_thresh=0.2,
        use_space_char=True,
        use_gpu=True,
        enable_mkldnn=True,
        show_log=False,
    )

    result = ocr.ocr(image, cls=True)
    if result[0] is None:
        return None
    data = []
    for idx, line in enumerate(result[0]):
        box = line[0]
        text = line[1][0]
        xmin, ymin = map(int, box[0])
        xmax, ymax = map(int, box[2])
        data.append({"text_box": [ymin, xmin, ymax, xmax], "text": text})
    return data


def detect_text_easy(image: np.ndarray) -> list[dict]:
    """Detect text using EasyOCR."""
    # Implementation of EasyOCR

    reader = easyocr.Reader(["en"], gpu=True)

    result_preprocessed = reader.readtext(
        image,
        paragraph=False,
        decoder="beamsearch",
        beamWidth=5,
        batch_size=8,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.6,
        mag_ratio=2.0,
    )

    # Process the combined results

    data = []
    for detection in result_preprocessed:
        box, text, confidence = detection
        xmin, ymin = map(int, box[0])
        xmax, ymax = map(int, box[2])
        data.append(
            {
                "text_box": [ymin, xmin, ymax, xmax],
                "text": text,
            }
        )
    return data
