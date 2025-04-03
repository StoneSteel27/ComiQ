import cv2
import os
from typing import List, Union
from .ocr import perform_ocr
from .ai_processing import process_with_ai
from .preprocessing import preprocess_image
from .utils import ai2norm, norm2ai, merge_box_groups, assign_ids_to_bounds, cv2pil

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_API_KEY = None

available_ocr = ["easyocr", "paddleocr"]


def set_api_key(api_key: str):
    """Set the API key for the Comiq module."""
    global _API_KEY
    _API_KEY = api_key


def extract(
        image: Union[str, "numpy.ndarray"], ocr: Union[str, List[str]] = "paddleocr"
):
    """
    Extract text from the given image using specified OCR method(s) and process with AI.

    Args:
        image (str or numpy.ndarray): Path to the image file or numpy array of the image.
        ocr (str or list): OCR method(s) to use. Can be "paddleocr", "easyocr", or a list containing both.

    Returns:
        dict: Processed data containing text extractions and their locations.
    """
    if _API_KEY is None:
        raise ValueError("API key not set. Use comiq.set_api_key() to set the API key.")
    # Load and preprocess the image

    if isinstance(image, str):
        image = cv2.imread(image)
    processed_image = preprocess_image(image)
    height, width = processed_image.shape[:2]

    # Perform OCR

    if isinstance(ocr, str):
        ocr = [ocr]
    ocr = list(filter(lambda x: x in available_ocr, ocr))

    if len(ocr) == 0:
        raise ValueError(
            f"The OCR engine provided is invalid, The Available OCR engines:{available_ocr}"
        )
    # Perform OCR on the Image

    ocr_results = perform_ocr(processed_image, ocr)

    # Make the data AI-friendly

    ocr_results = norm2ai(ocr_results, height, width)
    ocr_bound_ids = assign_ids_to_bounds(ocr_results)

    # Process data with AI into possible text groups

    predicted_groups = process_with_ai(cv2pil(processed_image), ocr_bound_ids, _API_KEY)

    # Merge boxes into text bubbles

    results = merge_box_groups(predicted_groups, ocr_bound_ids)
    final_results = ai2norm(results, height, width)

    return final_results
