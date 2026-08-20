import os
import sys

from comiq import ComiQ

# ComiQ 0.2.0 - Sample Test
# Engines: paddleocr (PP-OCRv6, default), easyocr

# Custom configuration for OCR engines (optional)
# By default, both PaddleOCR and EasyOCR will automatically use GPU if CUDA is available.
config = {
    "ocr": {
        # PP-OCRv6 configuration (PaddleOCR 3.7+)
        "paddleocr": {
            "tier": "medium",  # tiny | small | medium
        },
        # EasyOCR configuration
        "easyocr": {
            # "reader": {"gpu": True},  # Defaults to True
        }
    },
    "ai": {
        "temperature": 0.5,  # AI creativity level
    }
}

# Initialize ComiQ
# It will automatically load MLLM_API_KEY from .env file
comiq = ComiQ(
    model_name="gemini-3.5-flash-lite",
    # base_url="https://your-custom-endpoint.com/v1/",  # Optional: custom endpoint
    **config
)

# Path to a bundled example image
image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "examples", "images", "normal-text--normal-orientation.png",
)

print("ComiQ 0.2.0 - Sample Test")
print("=" * 50)
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("=" * 50)

# Example 1: Use PP-OCRv6 (default engine)
print("\n1. Using PP-OCRv6 engine (paddleocr):")
try:
    data = comiq.extract(image_path, ocr="paddleocr")
    print(f"   ✓ Success! Detected {len(data)} text bubbles")
    if data:
        print(f"   First bubble: {data[0]['text'][:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Example 2: Use EasyOCR
print("\n2. Using EasyOCR engine:")
try:
    data = comiq.extract(image_path, ocr="easyocr")
    print(f"   ✓ Success! Detected {len(data)} text bubbles")
    if data:
        print(f"   First bubble: {data[0]['text'][:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Example 3: Use multiple engines for maximum coverage
print("\n3. Using multiple engines (paddleocr + easyocr):")
try:
    data = comiq.extract(image_path, ocr=["paddleocr", "easyocr"])
    print(f"   ✓ Success! Detected {len(data)} text bubbles (combined)")
    if data:
        print(f"   First bubble: {data[0]['text'][:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 50)
print("Test complete!")

# Show available OCR engines
from comiq import get_available_ocr_engines
print(f"\nAvailable OCR engines: {get_available_ocr_engines()}")
