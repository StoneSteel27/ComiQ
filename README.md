# ComiQ: Comic-Focused Hybrid OCR Library

ComiQ is an advanced Optical Character Recognition (OCR) library specifically designed for comics. It combines traditional OCR engines with a powerful AI model to provide accurate text detection and grouping in comic images.

For examples of ComiQ's capabilities, visit the [examples directory](examples/ReadME.md).

## Features

- **Hybrid OCR:** Combines multiple OCR engines for improved accuracy and robustness.
- **Extensible:** Register your own custom OCR engines.
- **AI-Powered Grouping:** Uses an AI model to intelligently group detected text into coherent bubbles and captions.
- **Configurable:** Easily configure the AI model, OCR engines, and other parameters.
- **Flexible:** Supports both file paths and in-memory image arrays.
- **Environment-Friendly:** Automatically loads your API key from environment variables.

## Installation

Install ComiQ with a single command:

```bash
pip install comiq
```

This automatically installs:
- ✅ **EasyOCR** - Works on all Python versions (3.8+) with CUDA 11.x-13.x
- ✅ **PaddleOCR 2.x** - For Python 3.8-3.12 (PP-OCRv4, most stable, CUDA 10.2-12.0)

### GPU Acceleration (Optional)

ComiQ works with CPU by default, but GPU acceleration is **10-50x faster**.

**EasyOCR GPU Support:**
- Requires PyTorch with CUDA support (not included by default)
- Install PyTorch with GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Supports CUDA 11.x, 12.x, and 13.x

**PaddleOCR 2.x GPU Support** (Python 3.8-3.12):
- Requires: CUDA 10.2, 11.2, 11.6, **11.7** (recommended), or 12.0
- Install: `pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/`
- ⚠️ **Python 3.13+:** PaddleOCR 2.x doesn't support Python 3.13. Use EasyOCR or register a custom PaddleOCR 3.x engine (see Custom Engine example below)

### Advanced Installation Options

**Force PaddleOCR 2.x installation:**
```bash
# Explicitly install PaddleOCR 2.x (requires Python 3.8-3.12)
pip install comiq[paddleocr2]
```

**Install without PaddleOCR** (minimal install, EasyOCR only):
```bash
pip install --no-deps comiq
pip install openai python-dotenv pydantic easyocr
```

**Note:** If you upgrade your Python version (e.g., 3.12 → 3.13), reinstall ComiQ to get the correct PaddleOCR version:
```bash
pip install --force-reinstall comiq
```

## Quick Start

1.  **Set your API Key:** ComiQ requires an MLLM API key. You can either pass it to the `ComiQ` constructor or set it as an environment variable named `MLLM_API_KEY`.

    You can create a `.env` file in your project's root directory:
    ```
    MLLM_API_KEY="your-api-key-here"
    ```

2.  **Use the `ComiQ` class:**

```python
from comiq import ComiQ
import cv2

# Initialize ComiQ. It will automatically load the API key from the .env file.
comiq = ComiQ()

# Process an image from a file path
image_path = "path/to/your/comic/image.jpg"
data = comiq.extract(image_path)

# Or process an image from a numpy array
image_array = cv2.imread(image_path)
data_from_array = comiq.extract(image_array)

# 'data' now contains a list of text bubbles with their text and locations.
print(data)
```

## Choosing an OCR Engine

ComiQ supports multiple OCR engines. Choose based on your needs:

```python
# Use PaddleOCR 2.x (Python 3.8-3.12 only)
data = comiq.extract(image_path, ocr="paddleocr")

# Use only EasyOCR (works on all Python versions, CUDA 11.x-13.x)
data = comiq.extract(image_path, ocr="easyocr")

# Explicitly use PaddleOCR 2.x
data = comiq.extract(image_path, ocr="paddleocr2")

# Use multiple engines for better coverage
data = comiq.extract(image_path, ocr=["paddleocr", "easyocr"])
```

**Recommendation:**
- **Python 3.8-3.12:** Use `"paddleocr"` (most stable, excellent accuracy)
- **Python 3.13+:** Use `"easyocr"` (PaddleOCR 2.x doesn't support Python 3.13)
- **CUDA 12.1+ or 13.x:** Use `"easyocr"` (PaddleOCR 2.x max CUDA is 12.0)
- **Maximum accuracy:** Use `["paddleocr", "easyocr"]` (slower but more thorough)

## API Reference

### `ComiQ(api_key: str = None, model_name: str = "gemini-2.5-flash", base_url: str = "https://generativelanguage.googleapis.com/v1beta/", **kwargs)`

Initializes the ComiQ instance.

- **`api_key` (str, optional):** Your MLLM API key. If not provided, it will be loaded from the `MLLM_API_KEY` environment variable.
- **`model_name` (str, optional):** The name of the AI model to use. Defaults to `"gemini-2.5-flash"`.
- **`base_url` (str, optional):** The base URL for the AI service. Defaults to Google's Generative AI endpoint.
- **`**kwargs`:** Additional configuration for the OCR and AI models. See "Custom Configuration" for more details.

### `extract(image: Union[str, 'numpy.ndarray'], ocr: Union[str, List[str]] = "paddleocr")`

Extracts and groups text from the given comic image.

- **`image` (str or numpy.ndarray):** The path to the image file or the image as a NumPy array.
- **`ocr` (str or list, optional):** The OCR engine(s) to use. Available engines:
  - `"paddleocr"` - PaddleOCR 2.x with PP-OCRv4 (Python 3.8-3.12 only)
  - `"paddleocr2"` - Alias for `"paddleocr"`
  - `"easyocr"` - EasyOCR (all Python versions, CUDA 11.x-13.x)
  - Or a list like `["paddleocr", "easyocr"]` to use multiple engines
  - Or custom registered engines (see below)
  
  Defaults to `"paddleocr"`.

**Returns:**
- `dict`: A dictionary containing the processed data, including text bubbles, their locations, and other metadata.

### `register_ocr_engine(name: str, engine: Callable)`
Registers a new OCR engine.

- **`name` (str):** The name to identify the engine.
- **`engine` (Callable):** The function that implements the engine. See "Advanced Usage" for details.

### `get_available_ocr_engines() -> List[str]`
Returns a list of all registered OCR engine names.

## Advanced Usage: Registering a Custom OCR Engine

You can extend ComiQ by adding your own OCR engine. Your custom engine must be a function that adheres to the following contract:

- **Input:** It must accept two arguments:
    1.  `image` (a `numpy.ndarray` in BGR format).
    2.  `**kwargs` (a dictionary for any configuration your engine needs).
- **Output:** It must return a list of dictionaries, where each dictionary represents a detected text box and has two keys:
    1.  `"text_box"`: A list of four integers `[ymin, xmin, ymax, xmax]`.
    2.  `"text"`: The detected text as a string.

**Example:**

Here is how you could wrap the `pytesseract` library as a custom engine:

```python
import comiq
import pytesseract
import cv2
import numpy as np

# 1. Define the custom engine function
def pytesseract_engine(image: np.ndarray, **kwargs) -> list:
    """
    A custom OCR engine using pytesseract.
    It expects a 'config' kwarg, e.g., pytesseract_engine(image, config="--psm 6")
    """
    # pytesseract works with RGB images
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use pytesseract to get structured data
    data = pytesseract.image_to_data(
        rgb_image,
        output_type=pytesseract.Output.DICT,
        config=kwargs.get("config", "")
    )
    
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60: # Filter by confidence
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text = data['text'][i]
            if text.strip():
                results.append({
                    "text_box": [y, x, y + h, x + w],
                    "text": text
                })
    return results

# 2. Register the new engine with ComiQ
comiq.register_ocr_engine("pytesseract", pytesseract_engine)

# 3. Now you can use it!
my_comiq = comiq.ComiQ()
tesseract_config = {"ocr": {"pytesseract": {"config": "--psm 6"}}}
data = my_comiq.extract(
    "path/to/image.png",
    ocr="pytesseract",
    **tesseract_config
)

print(comiq.get_available_ocr_engines())
# Output: ['paddleocr', 'paddleocr2', 'easyocr', 'pytesseract']
```

### Example: Registering PaddleOCR 3.x as a Custom Engine

For Python 3.13+ users who want to use PaddleOCR 3.x (note: unstable on Windows):

```python
import comiq
import numpy as np

# First, install PaddleOCR 3.x manually:
# pip install paddleocr>=3.0 paddlepaddle>=3.0

def paddleocr3_engine(image: np.ndarray, **kwargs) -> list:
    """
    Custom OCR engine for PaddleOCR 3.x (PP-OCRv5).
    Note: PaddleOCR 3.x is unstable on Windows with GPU.
    """
    from paddleocr import PaddleOCR
    
    # PaddleOCR 3.x configuration
    paddle_config = {
        "device": "cpu",  # Use CPU (GPU unstable on Windows)
        "lang": "en",
        "text_det_limit_side_len": 2560,
        "text_det_thresh": 0.3,
        "text_det_box_thresh": 0.6,
        "enable_mkldnn": False,  # Disabled to avoid 3.3.0 oneDNN bug
    }
    paddle_config.update(kwargs)
    
    ocr = PaddleOCR(**paddle_config)
    result = ocr.predict(image)
    
    if not result:
        return []
    
    data = []
    for res in result:
        if hasattr(res, 'rec_polys') and hasattr(res, 'rec_texts'):
            polys = res.rec_polys
            texts = res.rec_texts
            for poly, text in zip(polys, texts):
                xmin = int(poly[:, 0].min())
                ymin = int(poly[:, 1].min())
                xmax = int(poly[:, 0].max())
                ymax = int(poly[:, 1].max())
                data.append({"text_box": [ymin, xmin, ymax, xmax], "text": text})
    
    return data

# Register the custom engine
comiq.register_ocr_engine("paddleocr3", paddleocr3_engine)

# Now you can use it
my_comiq = comiq.ComiQ()
data = my_comiq.extract("path/to/image.png", ocr="paddleocr3")

print(comiq.get_available_ocr_engines())
# Output: ['paddleocr', 'paddleocr2', 'easyocr', 'paddleocr3']
```

## Custom Configuration

You can pass additional configuration to the `ComiQ` constructor to customize the behavior of the OCR and AI models.

```python
from comiq import ComiQ

# Custom configuration
config = {
    "ocr": {
        "paddleocr": {
            "lang": "japan", # Use Japanese language model for PaddleOCR
        },
        "easyocr": {
            "reader": {"gpu": False}, # Disable GPU for EasyOCR
        }
    },
    "ai": {
        "temperature": 0.5, # Make the AI model more creative
    }
}

comiq = ComiQ(
    model_name="gemini-2.5-flash",
    base_url="https://your-custom-endpoint.com/v1/",
    **config
)

data = comiq.extract("path/to/manga.jpg", ocr="paddleocr")
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

ComiQ is licensed under the [MIT License](LICENSE).
