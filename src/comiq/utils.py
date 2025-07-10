from typing import List, Dict
import cv2
import numpy as np
from PIL import Image
from .models import ComicAnalysis


def cv2pil(image: np.ndarray) -> Image.Image:
    """Convert Processed OpenCV image to PIL Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def ai2norm(bounds: List[Dict], height: int, width: int) -> List[Dict]:
    """Converts AI-supported bounds (0-1000 scale) to normal image coordinates."""
    for bound in bounds:
        box = bound["text_box"]
        x_min = int((box[1] / 1000) * width)
        y_min = int((box[0] / 1000) * height)
        x_max = int((box[3] / 1000) * width)
        y_max = int((box[2] / 1000) * height)
        bound["text_box"] = [y_min, x_min, y_max, x_max]
    return bounds


def norm2ai(bounds: List[Dict], height: int, width: int) -> List[Dict]:
    """Converts normal image coordinates to AI-supported bounds (0-1000 scale)."""
    for bound in bounds:
        box = bound["text_box"]
        x_min = int((box[1] / width) * 1000)
        y_min = int((box[0] / height) * 1000)
        x_max = int((box[3] / width) * 1000)
        y_max = int((box[2] / height) * 1000)
        bound["text_box"] = [y_min, x_min, y_max, x_max]
    return bounds


def assign_ids_to_bounds(bounds: List[Dict]) -> List[Dict]:
    """Assigns unique string IDs to each bounding box."""
    return [{**bound, "id": str(i)} for i, bound in enumerate(bounds)]


def merge_box_groups(
    predicted_groups: ComicAnalysis, bounds_with_ids: List[Dict]
) -> List[Dict]:
    """Merges OCR-detected boxes into text bubbles based on the AI's analysis."""
    id_to_bound = {bound["id"]: bound for bound in bounds_with_ids}
    merged = []

    for group in predicted_groups.groups:
        relevant_boxes = [
            id_to_bound[box_id]["text_box"]
            for box_id in group.box_ids
            if box_id in id_to_bound
        ]

        if relevant_boxes:
            ymin = min(box[0] for box in relevant_boxes)
            xmin = min(box[1] for box in relevant_boxes)
            ymax = max(box[2] for box in relevant_boxes)
            xmax = max(box[3] for box in relevant_boxes)
            merged.append(
                {
                    "text_box": [ymin, xmin, ymax, xmax],
                    "text": group.cleaned_text,
                    "panel_id": group.panel_id,
                    "text_bubble_id": group.text_bubble_id,
                    "type": group.type,
                    "style": group.style,
                }
            )
    return merged