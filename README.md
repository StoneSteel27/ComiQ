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

You can install ComiQ and its dependencies with a single pip command:

```bash
pip install comiq
```

This will install the CPU-versions of the required OCR libraries. For GPU acceleration, please see the section below.

### Optional: GPU Acceleration

For significantly better performance, you can install the GPU-enabled versions of PyTorch and PaddlePaddle. It is highly recommended to do this in a clean virtual environment.

**1. Uninstall CPU Versions (if necessary):**
If you have already installed the CPU versions, uninstall them first:
```bash
pip uninstall torch paddlepaddle
```

**2. Install GPU-Enabled PyTorch:**
Visit the official [PyTorch website](https://pytorch.org/get-started/locally/) and use the command generator to find the correct installation command for your specific system (OS, package manager, and CUDA version).

*Example for Windows with an NVIDIA GPU (CUDA 12.1):*
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Install GPU-Enabled PaddlePaddle:**
To ensure compatibility, install a version below 3.0:
```bash
pip install "paddlepaddle-gpu<3.0"
```

After installing the GPU versions, the OCR engines in ComiQ will automatically use them.

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
- **`ocr` (str or list, optional):** The OCR engine(s) to use. Can be `"paddleocr"`, `"easyocr"`, or a list like `["paddleocr", "easyocr"]`. Defaults to `"paddleocr"`.

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
# Output: ['paddleocr', 'easyocr', 'pytesseract']
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
