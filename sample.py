from comiq import ComiQ
import sys

# ComiQ 0.1.4 - Sample Test
# PaddleOCR 2.x is automatically installed for Python 3.8-3.12
# Python 3.13+ users: Use EasyOCR (PaddleOCR 2.x doesn't support Python 3.13)

# Custom configuration for OCR engines
config = {
    "ocr": {
        # PaddleOCR 2.x configuration (Python 3.8-3.12)
        "paddleocr": {
            "lang": "japan",  # Use Japanese language model
            "use_gpu": False,  # Use CPU for stability
        },
        # EasyOCR configuration
        "easyocr": {
            "reader": {"gpu": False},  # Disable GPU for testing
        }
    },
    "ai": {
        "temperature": 0.5,  # AI creativity level
    }
}

# Initialize ComiQ
# It will automatically load MLLM_API_KEY from .env file
comiq = ComiQ(
    model_name="gemini-3-flash-preview",
    # base_url="https://your-custom-endpoint.com/v1/",  # Optional: custom endpoint
    **config
)

# Path to your test image
image_path = r"C:\Users\Kanishqvijay\Pictures\Screenshots\bub.png"

print("ComiQ 0.1.4 - Sample Test")
print("=" * 50)
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("=" * 50)

# Example 1: Use PaddleOCR (Python 3.8-3.12 only)
if sys.version_info < (3, 13):
    print("\n1. Using PaddleOCR 2.x engine:")
    try:
        data = comiq.extract(image_path, ocr="paddleocr")
        print(f"   ✓ Success! Detected {len(data)} text bubbles")
        if data:
            print(f"   First bubble: {data[0]['text'][:50]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("\n1. Skipping PaddleOCR test (Python 3.13+ not supported)")
    print("   → PaddleOCR 2.x requires Python 3.8-3.12")

# Example 2: Use EasyOCR (works on all Python versions)
print("\n2. Using EasyOCR engine:")
try:
    data = comiq.extract(image_path, ocr="easyocr")
    print(f"   ✓ Success! Detected {len(data)} text bubbles")
    if data:
        print(f"   First bubble: {data[0]['text'][:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Example 3: Use multiple engines for maximum coverage
if sys.version_info < (3, 13):
    print("\n3. Using both PaddleOCR + EasyOCR:")
    try:
        data = comiq.extract(image_path, ocr=["paddleocr", "easyocr"])
        print(f"   ✓ Success! Detected {len(data)} text bubbles (combined)")
        if data:
            print(f"   First bubble: {data[0]['text'][:50]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("\n3. Skipping multi-engine test (PaddleOCR not available on Python 3.13+)")

print("\n" + "=" * 50)
print("Test complete!")

# Show available OCR engines
from comiq import get_available_ocr_engines
print(f"\nAvailable OCR engines: {get_available_ocr_engines()}")
