from .comiq import ComiQ
from .ocr import register_ocr_engine, get_available_ocr_engines
from .models import ComicAnalysis, Group

__version__ = "0.2.0"
__all__ = [
    "ComiQ",
    "register_ocr_engine",
    "get_available_ocr_engines",
    "ComicAnalysis",
    "Group",
]
