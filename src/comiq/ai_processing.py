import base64
import io
import json
from typing import Any, Dict, List, Optional

import instructor
from openai import OpenAI
from PIL import Image

from .prompts import comic_prompt
from .models import ComicAnalysis

# Cached instructor clients, keyed by (api_key, base_url, mode)
_client_cache: Dict[tuple, instructor.Instructor] = {}


def get_base64_image(image: Image) -> str:
    """Get base64 representation of an image."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _resolve_mode(instructor_mode) -> instructor.Mode:
    """Resolve an instructor mode from an enum value or string name."""
    if instructor_mode is None:
        return instructor.Mode.JSON
    if isinstance(instructor_mode, instructor.Mode):
        return instructor_mode
    if isinstance(instructor_mode, str):
        try:
            return instructor.Mode[instructor_mode.upper()]
        except KeyError:
            valid = ", ".join(m.name for m in instructor.Mode)
            raise ValueError(
                f"Unknown instructor mode '{instructor_mode}'. Valid modes: {valid}"
            )
    raise TypeError(f"instructor_mode must be an instructor.Mode or str, got {type(instructor_mode)}")


def _get_client(mllm_api_key: str, base_url: str, mode: instructor.Mode) -> instructor.Instructor:
    """Return a cached instructor-patched OpenAI client."""
    cache_key = (mllm_api_key, base_url, mode)
    client = _client_cache.get(cache_key)
    if client is None:
        client = instructor.from_openai(
            OpenAI(api_key=mllm_api_key, base_url=base_url),
            mode=mode,
        )
        _client_cache[cache_key] = client
    return client


def process_with_ai(
    image: Image,
    ocr_results: Dict[str, List[Dict]],
    mllm_api_key: str,
    model_name: str,
    base_url: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    response_model: Optional[Any] = None,
    max_retries: int = 3,
    instructor_mode: Optional[Any] = None,
    **kwargs,
) -> Any:
    """Process OCR results with AI and validate with Pydantic.

    Uses instructor to enforce the response schema: if the model returns
    invalid output, the validation error is fed back to the model for
    retry (up to max_retries), which keeps even small LLMs on-schema.

    response_model must have a 'groups' field whose items contain at least
    'box_ids' and 'cleaned_text' (see ComiQ.__init__ for validation).
    """
    response_model = response_model or ComicAnalysis
    mode = _resolve_mode(instructor_mode)
    client = _get_client(mllm_api_key, base_url, mode)
    base64_image = get_base64_image(image)
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    prompt = comic_prompt.format(ocr_results, schema)

    return client.chat.completions.create(
        model=model_name,
        response_model=response_model,
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
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        **kwargs,
    )
