import cv2
import os
from typing import Any, List, Optional, Type, Union, Dict
from pydantic import BaseModel
from typing import get_args
import numpy as np
from dotenv import load_dotenv
from .ocr import perform_ocr, get_available_ocr_engines
from .ai_processing import process_with_ai
from .models import ComicAnalysis
from .utils import ai2norm, norm2ai, merge_box_groups, assign_ids_to_bounds, cv2pil

load_dotenv()


def _validate_response_model(response_model: Any) -> None:
    """Ensure a custom response model can drive the extraction pipeline.

    The model must have a 'groups' field whose items contain at least
    'box_ids' and 'cleaned_text' — these drive box merging. All other
    fields are free-form and flow through to the output.
    """
    if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
        raise TypeError(
            "response_model must be a Pydantic BaseModel subclass, got "
            f"{response_model!r}. See README: Custom Response Models."
        )
    if "groups" not in response_model.model_fields:
        raise ValueError(
            "response_model must have a 'groups' field (a list of group "
            "models). See README: Custom Response Models."
        )
    group_model = next(
        (
            arg for arg in get_args(response_model.model_fields["groups"].annotation)
            if isinstance(arg, type) and issubclass(arg, BaseModel)
        ),
        None,
    )
    if group_model is None:
        raise ValueError(
            "response_model's 'groups' field must be a list of Pydantic models."
        )
    missing = {"box_ids", "cleaned_text"} - set(group_model.model_fields)
    if missing:
        raise ValueError(
            f"response_model's group model is missing required field(s): "
            f"{sorted(missing)}. 'box_ids' and 'cleaned_text' are required "
            "so detected text boxes can be merged. Extend comiq.Group to "
            "add your own fields. See README: Custom Response Models."
        )


class ComiQ:
    """
    A class to extract text from comic images, process it, and return structured data.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-3.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/",
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ):
        """
        Initializes the ComiQ instance.

        Args:
            api_key (str, optional): The API key for the AI service. If not provided,
                                     it's sourced from the MLLM_API_KEY environment variable.
            model_name (str): The name of the AI model to use. Defaults to "gemini-3.5-flash-lite".
            base_url (str): The base URL for the AI service.
            response_model (Type[BaseModel], optional): A custom Pydantic model for the
                MLLM response. Must have a 'groups' field whose items contain at least
                'box_ids' and 'cleaned_text'; all other fields are free-form and are
                included in the extract() output. Defaults to comiq.ComicAnalysis.
                See README: Custom Response Models.
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
        if response_model is not None:
            _validate_response_model(response_model)
        self.response_model = response_model or ComicAnalysis
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
            mllm_api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            response_model=self.response_model,
            **ai_config,
        )

        # Merge results and convert coordinates back
        merged_results = merge_box_groups(predicted_groups, ocr_bound_ids)
        final_results = ai2norm(merged_results, height, width)

        return final_results
