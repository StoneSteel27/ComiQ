# ComiQ: Comic-Focused Hybrid OCR Library

**ComiQ** is a Python library built specifically for reading comics and manga. It pairs state-of-the-art OCR detection with an AI vision model (MLLM) to solve the hardest problems in comic text extraction: grouping fragmented word boxes into coherent speech bubbles, classifying bubble types (dialogue, thought, narration, SFX), and cleaning up recognition errors.

For visual capability demonstrations, check the [examples directory](examples/ReadME.md).

---

## How It Works

```
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────────────┐       ┌──────────────────────┐
│   Comic Image   │ ────► │       OCR Engine       │ ────► │       MLLM + Instructor       │ ────► │  Structured Bubbles  │
│  (file / array) │       │ (PP-OCRv6 / EasyOCR)   │       │ (Bubble Grouping & Cleaning)  │       │ (text, boxes, meta)  │
└─────────────────┘       └────────────────────────┘       └───────────────────────────────┘       └──────────────────────┘
```

1. **High-Precision OCR:** Detects raw word bounding boxes across panels using **PP-OCRv6** (unified 50-language SOTA model) or EasyOCR.
2. **AI Layout & Bubble Grouping:** An MLLM (via Gemini or any OpenAI-compatible vision endpoint) analyzes the visual page layout and groups scattered word boxes into complete bubbles.
3. **Structured Extraction via Instructor:** Schema enforcement with automatic retries guarantees valid JSON and allows extracting custom metadata (character speakers, emotional tone, translations).

---

## Features

- 🎯 **SOTA Comic OCR:** Powered by **PP-OCRv6** with 3 scalable model tiers (`tiny`, `small`, `medium`) supporting 50 languages in one model.
- 💬 **Intelligent Bubble Grouping:** Combines individual word boxes into coherent dialogue, thought bubbles, and narration panels.
- ✨ **OCR Error Correction:** The vision model cleans split words, misrecognized punctuation, and manga-specific font quirks.
- 🏷️ **Custom Pydantic Schemas:** Powered by [Instructor](https://python.useinstructor.com) — easily extract speakers, emotion, translation, or narrative tags alongside text.
- ⚡ **GPU & MKL-DNN Accelerated:** Fast inference on NVIDIA GPUs (CUDA) or multi-threaded CPU.
- 🔌 **Extensible:** Register custom OCR engines (Tesseract, RapidOCR, cloud APIs) with a simple decorator.
- 🖼️ **Flexible Input:** Works directly with file paths (`.png`, `.jpg`) or in-memory OpenCV / NumPy arrays.

---

## Installation

Install ComiQ with pip:

```bash
pip install comiq
```

This automatically installs:
- ✅ **PaddleOCR 3.x with PP-OCRv6** — SOTA accuracy, 50 languages in a unified model (Python 3.8+)
- ✅ **EasyOCR** — Multi-engine fallback supporting CUDA 11.x–13.x
- ✅ **Instructor & OpenAI SDK** — Structured MLLM extraction with validation retries

### GPU Acceleration (Optional)

ComiQ runs on CPU by default, but NVIDIA GPU acceleration is **10–50× faster**.

#### PP-OCRv6 GPU Support:
1. Install the PaddlePaddle GPU build matching your CUDA version (CUDA 11.8+ or 12.x) from the [PaddlePaddle install guide](https://www.paddlepaddle.org.cn/en/install/quick), e.g.:
   ```bash
   pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
   ```
2. Pass `device="gpu"` in your configuration. Windows GPU is supported (including RTX 30, 40, and 50 series).

#### EasyOCR GPU Support:
Install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Minimal Installation (EasyOCR Only)

To install without PaddleOCR:
```bash
pip install --no-deps comiq
pip install openai instructor python-dotenv pydantic easyocr
```

---

## Quick Start

### 1. Set your API Key
ComiQ requires an MLLM API key (Gemini by default). You can pass it directly to `ComiQ()` or create a `.env` file in your project root:

```env
MLLM_API_KEY="your-api-key-here"
```

### 2. Extract Text from a Comic

```python
import cv2
from comiq import ComiQ

# Initialize ComiQ (loads MLLM_API_KEY from .env)
comiq = ComiQ()

# Process from an image path
data = comiq.extract("path/to/comic_page.png")

# Or process directly from a NumPy array
image_array = cv2.imread("path/to/comic_page.png")
data = comiq.extract(image_array)

for bubble in data:
    print(f"[{bubble['type']}] Panel {bubble['panel_id']}: {bubble['text']}")
```

---

## OCR Engines & PP-OCRv6

ComiQ supports multiple built-in OCR engines:

```python
# Use default PP-OCRv6 (best accuracy, 50 unified languages)
data = comiq.extract(image_path, ocr="paddleocr")

# Use EasyOCR
data = comiq.extract(image_path, ocr="easyocr")

# Use both engines for maximum coverage
data = comiq.extract(image_path, ocr=["paddleocr", "easyocr"])
```

### PP-OCRv6 Model Tiers

PP-OCRv6 offers three model tiers configurable via the `tier` parameter:

| Tier | Parameters | Speed (GPU) | Best For |
|---|---|---|---|
| `tiny` | 1.5M | ~0.2s | Edge devices, real-time preview, high-speed batching |
| `small` | 7.7M | ~0.5s | Balanced speed and accuracy |
| **`medium`** (default) | 34.5M | ~1.5s | Production quality, complex typography, stylized text |

```python
config = {
    "ocr": {
        "paddleocr": {
            "tier": "medium",  # tiny | small | medium
            "device": "gpu",   # cpu | gpu
        }
    }
}
comiq = ComiQ(**config)
```

### Performance & Caching

OCR engine instances are **cached per configuration**, so the initial model download and initialization only happen once.

Measured inference time on a sample comic page (RTX 4050 GPU / 8-core CPU):

| Setup | Warm Latency | Detections |
|---|---|---|
| **GPU, `medium`** | **~0.2–1.5s** | **High (SOTA)** |
| **CPU, `tiny` (MKL-DNN)** | **~0.5s** | Good |
| **CPU, `small` (MKL-DNN)** | **~1.4s** | Great |
| **CPU, `medium`** | **~13s** | High (SOTA) |

**Tuning tips:**
- **`tier`**: The most effective speed dial on CPU (`tiny` is 7× faster, `small` is 3× faster than `medium`).
- **`enable_mkldnn: True`**: Opt-in CPU acceleration (0.5s tiny / 1.4s small). Kept `False` by default for stability because `paddlepaddle 3.3.x` has an upstream oneDNN bug. Users on `paddlepaddle<=3.2.2` can safely pass `enable_mkldnn: True`.
- **`enable_hpi: True`**: PaddleOCR's High-Performance Inference (auto TensorRT/OpenVINO). *Linux only*.

---

## Custom Response Models

Powered by [Instructor](https://python.useinstructor.com), ComiQ allows you to extract **structured data beyond text** by passing your own Pydantic model. 

Simply extend `comiq.Group` (or define a model with a `groups` list). Any extra fields you define are filled by the vision model from context and included in the output:

```python
from pydantic import BaseModel, Field
from comiq import ComiQ, Group

# 1. Extend the Group model with custom fields
class RichGroup(Group):
    speaker: str = Field("unknown", description="Name of the speaking character, or 'narrator'.")
    emotion: str = Field("neutral", description="Emotional tone: angry, shouting, whisper, calm, etc.")
    translation: str = Field("", description="English translation if the original text is Japanese/foreign.")

class RichAnalysis(BaseModel):
    groups: list[RichGroup]

# 2. Pass your schema to ComiQ
comiq = ComiQ(
    model_name="gemini-3.5-flash-lite",
    response_model=RichAnalysis
)

results = comiq.extract("manga_page.png")

for bubble in results:
    print(f"{bubble['speaker']} ({bubble['emotion']}): {bubble['text']}")
    # Output: Orihime (concerned): BE CAREFUL, CHAD...
```

---

## Custom Configuration

You can customize AI parameters, OCR settings, and retries:

```python
config = {
    "ocr": {
        "paddleocr": {
            "tier": "medium",
            "device": "gpu",
        },
        "easyocr": {
            "reader": {"gpu": True},
        }
    },
    "ai": {
        "temperature": 0.2,       # Lower = more deterministic
        "max_retries": 3,          # Instructor retry count on schema mismatch
        "instructor_mode": "JSON", # JSON (default), TOOLS, or MD_JSON
    }
}

comiq = ComiQ(
    model_name="gemini-3.5-flash-lite",
    base_url="https://generativelanguage.googleapis.com/v1beta/",
    **config
)
```

---

## Registering a Custom OCR Engine

You can plug in any third-party OCR library (e.g., Tesseract, RapidOCR, cloud APIs):

```python
import cv2
import numpy as np
import pytesseract
import comiq

# 1. Define the engine function (accepts BGR image + **kwargs)
def pytesseract_engine(image: np.ndarray, **kwargs) -> list:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb_image,
        output_type=pytesseract.Output.DICT,
        config=kwargs.get("config", "--psm 6")
    )
    
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and data['text'][i].strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            results.append({
                "text_box": [y, x, y + h, x + w],  # [ymin, xmin, ymax, xmax]
                "text": data['text'][i]
            })
    return results

# 2. Register with ComiQ
comiq.register_ocr_engine("pytesseract", pytesseract_engine)

# 3. Use it in extract()
my_comiq = comiq.ComiQ()
data = my_comiq.extract("image.png", ocr="pytesseract")
```

---

## API Reference

### `ComiQ`

```python
ComiQ(
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/",
    response_model: Optional[Type[BaseModel]] = None,
    **kwargs
)
```

- **`api_key`**: MLLM API key. If omitted, loaded from `MLLM_API_KEY` in environment / `.env`.
- **`model_name`**: Vision model identifier (defaults to `"gemini-3.5-flash-lite"`).
- **`base_url`**: Endpoint URL for the OpenAI-compatible vision service.
- **`response_model`**: Optional custom Pydantic response schema (see [Custom Response Models](#custom-response-models)).
- **`**kwargs`**: Nested configuration dictionaries (`ocr`, `ai`).

---

### `comiq.extract()`

```python
extract(
    image: Union[str, np.ndarray],
    ocr: Union[str, List[str]] = "paddleocr"
) -> List[Dict[str, Any]]
```

- **`image`**: File path (`str`) or loaded image as a NumPy array (`np.ndarray` in BGR format).
- **`ocr`**: Engine name (`"paddleocr"`, `"easyocr"`, or custom registered name) or list of names (`["paddleocr", "easyocr"]`).

#### Return Schema

Returns a list of dictionaries, each representing an extracted speech bubble:

```python
[
  {
    "panel_id": "1",                                # Panel number
    "text_bubble_id": "1-1",                        # Bubble identifier within panel
    "text_box": [31, 25, 87, 97],                   # Pixel coordinates [ymin, xmin, ymax, xmax]
    "text": "BE CAREFUL, CHAD...",                  # Cleaned & reconstructed text
    "type": "dialogue",                             # dialogue | thought | narration | sound_effect | background
    "style": "normal",                              # normal | emphasized | angled | split
    "notes": "none",                                # AI notes, uncertainties, or SFX justification
    "original_text": "BE CAREFUL, CHAD...",         # Raw OCR text before AI correction
    # + Any extra fields defined in your custom response_model
  },
  ...
]
```

---

### OCR Registry Functions

#### `register_ocr_engine(name: str, engine: Callable)`
Registers a custom OCR function. The engine must accept `(image: np.ndarray, **kwargs)` and return a list of `{"text_box": [ymin, xmin, ymax, xmax], "text": str}` dicts.

#### `get_available_ocr_engines() -> List[str]`
Returns the list of currently registered OCR engines (default: `['paddleocr', 'paddleocr6', 'ppocrv6', 'easyocr']`).

---

## Contributing

Contributions are welcome! Please check our [Contributing Guide](CONTRIBUTING.md) and [Changelog](CHANGELOG.md) for details on development workflows and guidelines.

---

## License

ComiQ is licensed under the [MIT License](LICENSE).
