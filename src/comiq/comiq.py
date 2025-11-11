import cv2
import os
from typing import List, Union, Dict, Any
import numpy as np
from dotenv import load_dotenv
from .ocr import perform_ocr, get_available_ocr_engines
from .ai_processing import process_with_ai
from .utils import ai2norm, norm2ai, merge_box_groups, assign_ids_to_bounds, cv2pil

load_dotenv()


class ComiQ:
    """
    A class to extract text from comic images, process it, and return structured data.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/",
        **kwargs,
    ):
        """
        Initializes the ComiQ instance.

        Args:
            api_key (str, optional): The API key for the AI service. If not provided,
                                     it's sourced from the GEMINI_API_KEY environment variable.
            model_name (str): The name of the AI model to use.
            base_url (str): The base URL for the AI service.
            **kwargs: Additional configuration for AI and OCR.
        """
        self.api_key = api_key or os.getenv("MLLM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Please pass it to the constructor or set the "
                "MLLM_API_KEY environment variable."
            )
        self.model_name = model_name
        self.base_url = base_url
        self.config = kwargs

    def extract(
        self,
        image: Union[str, np.ndarray],
        ocr: Union[str, List[str]] = "paddleocr",
    ) -> Dict[str, Any]:
        """
        Extracts text from the given image using specified OCR method(s) and processes it with AI.

        Args:
            image (str or numpy.ndarray): Path to the image file or a numpy array of the image.
            ocr (str or list): The OCR method(s) to use.

        Returns:
            dict: Processed data containing text extractions and their locations.
        """
        # Load the image
        if isinstance(image, str):
            image = cv2.imread(image)

        height, width = image.shape[:2]

        # Perform OCR
        if isinstance(ocr, str):
            ocr = [ocr]

        available_ocr = get_available_ocr_engines()
        ocr_methods = [method for method in ocr if method in available_ocr]
        if not ocr_methods:
            raise ValueError(
                f"Invalid OCR engine requested. Available options: {available_ocr}"
            )

        ocr_config = self.config.get("ocr", {})
        ocr_results = perform_ocr(image, ocr_methods, **ocr_config)

        # Normalize data for AI processing
        ocr_results_ai = norm2ai(ocr_results, height, width)
        ocr_bound_ids = assign_ids_to_bounds(ocr_results_ai)

        # Process with AI
        ai_config = self.config.get("ai", {})
        predicted_groups = process_with_ai(
            image=cv2pil(image),
            ocr_results=ocr_bound_ids,
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            **ai_config,
        )

        # Merge results and convert coordinates back
        merged_results = merge_box_groups(predicted_groups, ocr_bound_ids)
        final_results = ai2norm(merged_results, height, width)

        return final_results
