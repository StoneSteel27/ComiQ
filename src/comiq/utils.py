from typing import List, Dict
import cv2
import numpy as np
from PIL import Image


def cv2pil(image: np.ndarray) -> Image.Image:
    """Convert Processed OpenCV image to PIL Image."""
    return Image.fromarray(image)


def ai2norm(bounds, height, width):
    """Converts AI supported bounds to Normal Image Bounds"""
    for bound in bounds:
        box = bound["text_box"]
        x_min, y_min = int((box[1] / 1000) * width), int((box[0] / 1000) * height)
        x_max, y_max = int((box[3] / 1000) * width), int((box[2] / 1000) * height)
        bound["text_box"] = [y_min, x_min, y_max, x_max]
    return bounds


def norm2ai(bounds, height, width):
    """Converts Normal Image bounds to AI supported Bounds"""
    for bound in bounds:
        box = bound["text_box"]
        x_min, y_min = int((box[1] / width) * 1000), int((box[0] / height) * 1000)
        x_max, y_max = int((box[3] / width) * 1000), int((box[2] / height) * 1000)
        bound["text_box"] = [y_min, x_min, y_max, x_max]
    return bounds


def assign_ids_to_bounds(bounds):
    """Assigns Unique IDs for Bounds"""
    return [{**bound, "id": str(i)} for i, bound in enumerate(bounds)]


def merge_box_groups(groups, bounds_with_ids):
    """Merges Bounds into Text Bubbles, Based on AI Response"""
    id_to_bound = {bound["id"]: bound for bound in bounds_with_ids}
    merged = []
    for group in groups.get("groups", []):
        box_ids = group.get("box_ids", [])
        text = group.get("cleaned_text", "")
        panel_id = group.get("panel_id", "")

        relevant_boxes = [
            id_to_bound[id]["text_box"] for id in box_ids if id in id_to_bound
        ]

        if relevant_boxes:
            ymin = min(box[0] for box in relevant_boxes)
            xmin = min(box[1] for box in relevant_boxes)
            ymax = max(box[2] for box in relevant_boxes)
            xmax = max(box[3] for box in relevant_boxes)
            merged.append(
                {
                    "text_box": [ymin, xmin, ymax, xmax],
                    "text": text,
                    "panel_id": panel_id,
                    "type": group.get("type", ""),
                }
            )
    return merged
