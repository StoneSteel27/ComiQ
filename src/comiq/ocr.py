import numpy as np
import warnings
import sys
from typing import List, Dict, Any, Callable

# Try importing PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    warnings.warn(
        "PaddleOCR is not installed. "
        "It will be automatically installed for Python 3.8-3.12.\n"
        "For Python 3.13+, PaddleOCR 2.x is not available. Use EasyOCR instead:\n"
        "  comiq.extract(image, ocr='easyocr')\n"
        "Or register a custom PaddleOCR 3.x engine - see README for details.",
        UserWarning
    )

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

def _detect_text_paddleocr(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """Internal implementation for PaddleOCR 2.x (PP-OCRv4).
    
    Supports Python 3.8-3.12 only. For Python 3.13+, use EasyOCR or register
    a custom PaddleOCR 3.x engine.
    """
    if not PADDLEOCR_AVAILABLE:
        raise ImportError(
            "PaddleOCR is not installed.\n\n"
            "For Python 3.8-3.12, install with:\n"
            "  pip install 'comiq[paddleocr2]'\n\n"
            "For Python 3.13+, PaddleOCR 2.x is not available. Options:\n"
            "  1. Use EasyOCR: comiq.extract(image, ocr='easyocr')\n"
            "  2. Register custom PaddleOCR 3.x engine (see README)\n"
        )
    
    # Check Python version at runtime
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "PaddleOCR 2.x does not support Python 3.13+.\n\n"
            "Options:\n"
            "  1. Use EasyOCR (recommended):\n"
            "     comiq.extract(image, ocr='easyocr')\n"
            "  2. Downgrade to Python 3.8-3.12\n"
            "  3. Register custom PaddleOCR 3.x engine:\n"
            "     See README.md for custom engine registration example\n\n"
            "Note: PaddleOCR 3.x is unstable on Windows. We recommend EasyOCR for Python 3.13+."
        )
    
    paddle_config = {
        "use_angle_cls": True,
        "lang": "en",
        "det_limit_side_len": 2560,
        "det_db_thresh": 0.1,
        "det_db_box_thresh": 0.2,
        "use_gpu": False,  # Default to CPU for stability
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
    """Internal implementation for EasyOCR.
    
    Works on all Python versions (3.8+) and supports CUDA 11.x-13.x.
    """
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
# "paddleocr" is the main engine (PaddleOCR 2.x)
register_ocr_engine("paddleocr", _detect_text_paddleocr)

# "paddleocr2" is an alias for explicit version control
register_ocr_engine("paddleocr2", _detect_text_paddleocr)

# Always register EasyOCR (it's a required dependency)
register_ocr_engine("easyocr", _detect_text_easy)
