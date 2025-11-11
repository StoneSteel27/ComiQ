import base64
import io
import json
from openai import OpenAI
from PIL import Image
from typing import Dict, List
from pydantic import ValidationError
from .prompts import comic_prompt
from .models import ComicAnalysis, Group


def get_base64_image(image: Image) -> str:
    """Get base64 representation of an image."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_with_ai(
    image: Image,
    ocr_results: Dict[str, List[Dict]],
    mllm_api_key: str,
    model_name: str,
    base_url: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    **kwargs,
) -> ComicAnalysis:
    """Process OCR results with AI and validate with Pydantic."""
    client = configure_openai(mllm_api_key, base_url)
    base64_image = get_base64_image(image)
    prompt = comic_prompt.format(ocr_results)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        top_p=top_p,
        **kwargs,
    )

    response_text = response.choices[0].message.content
    cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        # First, try to validate the expected object structure: {"groups": [...]}
        return ComicAnalysis.model_validate_json(cleaned_text)
    except ValidationError as e:
        # If that fails, check if the AI returned a raw list: [...]
        try:
            parsed_json = json.loads(cleaned_text)
            if isinstance(parsed_json, list):
                groups = [Group.model_validate(item) for item in parsed_json]
                return ComicAnalysis(groups=groups)
            else:
                # If it's not a list, the structure is unexpected.
                raise e
        except (json.JSONDecodeError, ValidationError):
            # If it's not valid JSON or doesn't match the Group model, re-raise the original error.
            raise e


def configure_openai(mllm_api_key: str, base_url: str) -> OpenAI:
    """Configure and return an OpenAI client."""
    return OpenAI(
        api_key=mllm_api_key,
        base_url=base_url,
    )
