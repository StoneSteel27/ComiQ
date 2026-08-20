import warnings
from typing import List, Dict, Any, Callable

import numpy as np

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


def _paddleocr_version() -> tuple:
    """Returns the installed paddleocr version as a tuple, e.g. (3, 7, 0).

    Returns (0, 0, 0) if paddleocr is not installed.
    """
    try:
        import paddleocr
    except ImportError:
        return (0, 0, 0)
    version = getattr(paddleocr, "__version__", "0")
    parts = []
    for piece in str(version).split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _paddle_version() -> tuple:
    """Returns the installed paddlepaddle framework version as a tuple, e.g. (3, 2, 2).

    Returns (0, 0, 0) if paddlepaddle is not installed.
    """
    try:
        import paddle
    except ImportError:
        return (0, 0, 0)
    version = getattr(paddle, "__version__", "0")
    parts = []
    for piece in str(version).split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# --- Built-in OCR Implementations ---

_ppocrv6_instances: Dict[tuple, Any] = {}


def _detect_text_ppocrv6(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """PP-OCRv6 engine (default). Requires PaddleOCR 3.7+ and PaddlePaddle 3.x.

    PP-OCRv6 unifies 50 languages in a single model and offers three tiers:
    'tiny' (1.5M params), 'small' (7.7M), and 'medium' (34.5M, default).
    Select via the 'tier' kwarg, e.g. perform_ocr(..., tier='small').

    Instances are cached per configuration, so switching tiers (or other
    settings) between calls creates a new instance only when needed.
    """
    global _ppocrv6_instances
    version = _paddleocr_version()
    if version == (0, 0, 0):
        raise ImportError(
            "PaddleOCR is not installed. Install with:\n"
            "  pip install 'paddleocr>=3.7.0' 'paddlepaddle>=3.0.0'"
        )
    if version < (3, 7):
        raise RuntimeError(
            f"PP-OCRv6 requires PaddleOCR >= 3.7.0, but "
            f"{'.'.join(map(str, version))} is installed. Upgrade with:\n"
            "  pip install 'paddleocr>=3.7.0' 'paddlepaddle>=3.0.0'\n\n"
            "Note: ComiQ 0.2.0+ dropped support for PaddleOCR 2.x."
        )

    from paddleocr import PaddleOCR
    import paddle

    # Auto-select GPU if paddlepaddle-gpu with CUDA is installed, else CPU
    default_device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

    tier = kwargs.pop("tier", "medium")
    config = {
        "text_detection_model_name": f"PP-OCRv6_{tier}_det",
        "text_recognition_model_name": f"PP-OCRv6_{tier}_rec",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "text_det_limit_side_len": 2560,
        "text_det_limit_type": "max",  # 'min' (Paddle 3.x default) upscales small images
        "text_det_thresh": 0.1,
        "text_det_box_thresh": 0.2,
        "device": default_device,
    }
    if str(config["device"]).lower() == "cpu":
        # Off by default for stability: oneDNN/PIR errors
        # ("ConvertPirAttribute2RuntimeAttribute") affect PaddlePaddle 3.3.x.
        # Users on paddlepaddle<=3.2.2 can opt-in to enable_mkldnn=True for
        # up to 3.6x speedup on CPU.
        config["enable_mkldnn"] = False
    config.update(kwargs)

    if str(config.get("device", "cpu")).lower() == "cpu" and config.get("enable_mkldnn"):
        paddle_ver = _paddle_version()
        if (3, 3, 0) <= paddle_ver <= (3, 3, 1):
            warnings.warn(
                f"enable_mkldnn=True is set with PaddlePaddle {'.'.join(map(str, paddle_ver))}, "
                "which has a known upstream oneDNN regression bug ('ConvertPirAttribute2RuntimeAttribute'). "
                "If inference fails with NotImplementedError, downgrade to paddlepaddle==3.2.2 "
                "or set enable_mkldnn=False.",
                UserWarning,
                stacklevel=2,
            )

    cache_key = tuple(sorted((k, str(v)) for k, v in config.items()))
    ocr = _ppocrv6_instances.get(cache_key)
    if ocr is None:
        ocr = PaddleOCR(**config)
        _ppocrv6_instances[cache_key] = ocr

    results = ocr.predict(image)
    data = []
    for res in results:
        # PaddleOCR 3.x results are dict-like; attribute access is unreliable
        def _get(key):
            try:
                return res[key]
            except (KeyError, TypeError, IndexError):
                return getattr(res, key, None)

        texts = _get("rec_texts")
        polys = _get("rec_polys")
        if texts is None or polys is None:
            continue
        for text, poly in zip(texts, polys):
            if not text:
                continue
            poly = np.asarray(poly)
            xmin, xmax = int(poly[:, 0].min()), int(poly[:, 0].max())
            ymin, ymax = int(poly[:, 1].min()), int(poly[:, 1].max())
            data.append({"text_box": [ymin, xmin, ymax, xmax], "text": text})
    return data


_easyocr_reader = None


def _detect_text_easy(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """EasyOCR engine.

    Works on all Python versions (3.8+) and supports CUDA 11.x-13.x.
    The reader is cached after the first call, so only the reader
    configuration passed on the first call takes effect.
    """
    global _easyocr_reader
    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "EasyOCR is not installed. Install with:\n"
            "  pip install easyocr"
        )

    reader_config = {"gpu": True}
    readtext_config = {
        "paragraph": False, "decoder": "beamsearch", "beamWidth": 5,
        "batch_size": 8, "contrast_ths": 0.1, "adjust_contrast": 0.5,
        "text_threshold": 0.7, "low_text": 0.4, "link_threshold": 0.6,
        "mag_ratio": 2.0,
    }
    reader_config.update(kwargs.get("reader", {}))
    readtext_config.update(kwargs.get("readtext", {}))

    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], **reader_config)
    result = _easyocr_reader.readtext(image, **readtext_config)

    data = []
    for detection in result:
        box, text, _ = detection
        xmin, ymin = map(int, box[0])
        xmax, ymax = map(int, box[2])
        data.append({"text_box": [ymin, xmin, ymax, xmax], "text": text})
    return data


# Register the built-in engines
# "paddleocr" is the default engine (PP-OCRv6)
register_ocr_engine("paddleocr", _detect_text_ppocrv6)

# Explicit aliases for the PP-OCRv6 engine
register_ocr_engine("paddleocr6", _detect_text_ppocrv6)
register_ocr_engine("ppocrv6", _detect_text_ppocrv6)

# EasyOCR works on all Python versions
register_ocr_engine("easyocr", _detect_text_easy)
