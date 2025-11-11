from .comiq import ComiQ
from .ocr import register_ocr_engine, get_available_ocr_engines

__version__ = "0.1.3"
__all__ = ["ComiQ", "register_ocr_engine", "get_available_ocr_engines"]
