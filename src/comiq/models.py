from pydantic import BaseModel, Field
from typing import List, Literal


class Group(BaseModel):
    """Represents a single text bubble or caption in a comic panel."""
    panel_id: str = Field(..., description="The panel number for the group.")
    text_bubble_id: str = Field(..., description="The unique ID for the text bubble or caption within the panel.")
    box_ids: List[str] = Field(..., description="A list of OCR-detected word box IDs that form this group.")
    original_text: str = Field(..., description="The raw OCR output for the group before cleaning.")
    cleaned_text: str = Field(..., description="The corrected and cleaned text for the group.")
    type: Literal["dialogue", "thought", "narration", "sound_effect", "background"] = Field(
        ..., description="The type of text content."
    )
    style: Literal["normal", "emphasized", "angled", "split"] = Field(
        ..., description="The visual style of the text."
    )
    notes: str = Field(..., description="Notes on corrections, uncertainties, or justifications for inclusion.")


class ComicAnalysis(BaseModel):
    """The root model for the AI's analysis of the comic image."""
    groups: List[Group] = Field(..., description="A list of all text groups found in the comic.")
