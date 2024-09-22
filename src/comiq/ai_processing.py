import json
from PIL import Image
from typing import Dict, List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from .prompts import comic_prompt


def process_with_ai(
    image: Image, ocr_results: Dict[str, List[Dict]], api_key: str
) -> Dict:
    """Process OCR results with AI."""
    model = configure_genai(api_key)
    prompt = comic_prompt.format(ocr_results)
    response = model.generate_content([image, prompt])
    data = json.loads(response.text)
    return data


def configure_genai(api_key: str) -> genai.GenerativeModel:
    """Configure and return a GenerativeModel instance."""

    safety_ratings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=GenerationConfig(
            response_mime_type="application/json", temperature=0, top_k=1, top_p=0
        ),
        safety_settings=safety_ratings,
    )
