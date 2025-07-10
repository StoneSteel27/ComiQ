import numpy as np
from typing import List, Dict, Any, Callable
from paddleocr import PaddleOCR
import easyocr

# OCR engine registry
_ocr_engines: Dict[str, Callable] = {}


def register_ocr_engine(name: str, engine: Callable):
    """
    Registers a custom OCR engine. The engine function must accept a numpy.ndarray
    and **kwargs, and return a list of dictionaries, each with 'text_box' and 'text'.
    """
    if not callable(engine):
        raise TypeError("The provided engine must be a callable function.")
    _ocr_engines[name] = engine


def get_available_ocr_engines() -> List[str]:
    """Returns a list of available OCR engines."""
    return list(_ocr_engines.keys())


def perform_ocr(
    image: np.ndarray, methods: List[str], **kwargs
) -> List[Dict[str, Any]]:
    """Perform OCR using specified methods from the registry."""
    results = []
    for method in methods:
        if method not in _ocr_engines:
            raise ValueError(
                f"OCR engine '{method}' is not registered. "
                f"Available engines: {get_available_ocr_engines()}"
            )
        engine_func = _ocr_engines[method]
        results.extend(engine_func(image, **kwargs.get(method, {})))
    return results


# --- Built-in OCR Implementations ---

def _detect_text_pad(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """Internal implementation for PaddleOCR, optimized for v2.x."""
    paddle_config = {
        "use_angle_cls": True,
        "lang": "en",
        "det_limit_side_len": 2560,
        "det_db_thresh": 0.1,
        "det_db_box_thresh": 0.2,
        "use_space_char": True,
        "use_gpu": True,
        "enable_mkldnn": True,
        "show_log": False,
    }
    paddle_config.update(kwargs)
    ocr = PaddleOCR(**paddle_config)

    result = ocr.ocr(image, cls=True)
    if not result or not result[0]:
        return []

    data = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        xmin, ymin = map(int, box[0])
        xmax, ymax = map(int, box[2])
        data.append({"text_box": [ymin, xmin, ymax, xmax], "text": text})
    return data


def _detect_text_easy(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """Internal implementation for EasyOCR."""
    reader_config = {"gpu": True}
    readtext_config = {
        "paragraph": False, "decoder": "beamsearch", "beamWidth": 5,
        "batch_size": 8, "contrast_ths": 0.1, "adjust_contrast": 0.5,
        "text_threshold": 0.7, "low_text": 0.4, "link_threshold": 0.6,
        "mag_ratio": 2.0,
    }
    reader_config.update(kwargs.get("reader", {}))
    readtext_config.update(kwargs.get("readtext", {}))

    reader = easyocr.Reader(["en"], **reader_config)
    result = reader.readtext(image, **readtext_config)

    data = []
    for detection in result:
        box, text, _ = detection
        xmin, ymin = map(int, box[0])
        xmax, ymax = map(int, box[2])
        data.append({"text_box": [ymin, xmin, ymax, xmax], "text": text})
    return data


# Register the built-in engines
register_ocr_engine("paddleocr", _detect_text_pad)
register_ocr_engine("easyocr", _detect_text_easy)
