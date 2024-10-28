# ComiQ: Comic-Focused Hybrid OCR Library

ComiQ is an advanced Optical Character Recognition (OCR) library specifically designed for comics. It combines traditional OCR engines like EasyOCR and PaddleOCR with Google's Gemini Flash-1.5 model to provide accurate text detection and translation in comic images.

For, observing the capabilities of ComiQ, Visit: [examples/ReadME.md](https://github.com/StoneSteel27/ComiQ/blob/c27f94f0b987ec3df0b60d1863872efd7bd84eef/examples/ReadME.md)

## Features

- Hybrid OCR approach for improved accuracy
- Specialized in detecting text within comic bubbles and panels
- Integration with Google's Gemini Flash-1.5 model for enhanced performance
- Support for multiple OCR engines
- Easy-to-use Python interface

## Installation

Install ComiQ using pip:

```bash
pip install comiq
```

**Important Notes:** 
- For GPU-accelerated processing, please visit the [PyTorch website](https://pytorch.org/get-started/locally/) to install `torch` and `torchvision` with CUDA support.
- ComiQ uses `opencv-python-headless` as a dependency. If your project requires the full `opencv-python` package, you may need to manage these dependencies carefully to avoid conflicts. Choose the appropriate version based on your project's needs:
  - For headless environments or when GUI features are not required, ComiQ's default `opencv-python-headless` is sufficient.
  - If you need GUI features, you may need to uninstall `opencv-python-headless` and install `opencv-python` separately.

## Quick Start

```python
import comiq

# Set up your Gemini API key
comiq.set_api_key("<GEMINI_API_KEY>")

# Process an image
image_path = "path/to/your/comic/image.jpg"
data = comiq.extract(image_path)

# 'data' now contains a list of bounding boxes for each text bubble in the image
```

## API Reference

### `set_api_key(api_key: str)`

Sets the API key for the ComiQ module, which is required for using the Gemini AI model.

**Parameters:**
- `api_key` (str): The API key for accessing the Gemini AI service.

**Usage:**
```python
import comiq

comiq.set_api_key("your-api-key-here")
```

**Note:**
- You must call this function and set a valid API key before using any other ComiQ functions.
- Keep your API key confidential and do not share it publicly.

### `extract(image: Union[str, 'numpy.ndarray'], ocr: Union[str, List[str]] = "paddleocr")`

Extracts text from the given image using specified OCR method(s) and processes it with the Gemini AI model.

**Parameters:**
- `image` (str or numpy.ndarray): 
  - If str: Path to the image file.
  - If numpy.ndarray: Numpy array representation of the image.
- `ocr` (str or list of str, optional): 
  - OCR engine(s) to use. Default is "paddleocr".
  - Possible values: "paddleocr", "easyocr", or a list containing both.

**Returns:**
- dict: Processed data containing text extractions and their locations.

**Usage:**
```python
import comiq

# Using default OCR (PaddleOCR)
result = comiq.extract("path/to/your/comic/image.jpg")

# Using a specific OCR engine
result = comiq.extract("path/to/your/comic/image.jpg", ocr="easyocr")

# Using multiple OCR engines
result = comiq.extract("path/to/your/comic/image.jpg", ocr=["paddleocr", "easyocr"])

# Using a numpy array instead of an image path
import cv2
image_array = cv2.imread("path/to/your/comic/image.jpg")
result = comiq.extract(image_array)
```

**Notes:**
- Ensure you've set the API key using `set_api_key()` before calling this function.
- The function automatically preprocesses the image for optimal OCR performance.
- When using multiple OCR engines, the results are combined for improved accuracy.
- The returned dictionary contains bounding box coordinates and extracted text for each detected text region in the image.

## Advanced Usage

### Selecting OCR Engines

ComiQ supports two OCR engines: PaddleOCR and EasyOCR. You can specify which engine(s) to use:

```python
# Use a single OCR engine
data = comiq.extract(image_path, ocr="paddleocr")

# Use multiple OCR engines
data = comiq.extract(image_path, ocr=["paddleocr", "easyocr"])
```

### OCR Engine Comparison

| Feature       | EasyOCR                                                | PaddleOCR                                             |
|---------------|--------------------------------------------------------|-------------------------------------------------------|
| Strengths     | - Detects styled text<br>- Handles directional text<br>- Accurate bounding box positioning | - Higher true positive rate<br>- Better text quality |
| Weaknesses    | - Lower text quality<br>- Higher false positive rate   | - Struggles with styled text<br>- Limited directional text support<br>- Less accurate positioning |

## Contributing

We welcome contributions to ComiQ! Please see our [Contributing Guide](CONTRIBUTING.md) for more information on how to get started.

## License

ComiQ is released under the [MIT License](LICENSE).

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Google Gemini](https://deepmind.google/technologies/gemini/)

## Contact

For questions, issues, or suggestions, please [open an issue](https://github.com/yourusername/comiq/issues) on our GitHub repository.
