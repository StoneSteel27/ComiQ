import cv2
import numpy as np
from PIL import Image, ImageEnhance


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the input image for better OCR results."""
    # Convert to PIL Image

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Enhance sharpness

    enhancer = ImageEnhance.Sharpness(pil_img)
    sharpened = enhancer.enhance(2.0)

    # Enhance contrast

    enhancer = ImageEnhance.Contrast(sharpened)
    contrasted = enhancer.enhance(1.5)

    # Convert back to OpenCV format

    cv_img = cv2.cvtColor(np.array(contrasted), cv2.COLOR_RGB2BGR)

    # Convert to grayscale

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Denoise

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    denoised_bw = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    return denoised_bw
