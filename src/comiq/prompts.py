

comic_prompt = '''
Analyze the comic image, the provided OCR text locations, and the AI text extraction result carefully. Your task is to group the OCR-detected word boxes into coherent text bubbles or captions within comic panels, while also cleaning and correcting OCR errors. Pay special attention to the necessity of including background text and sound effects.

Key Instructions:

1. Word Grouping and Text Cleaning:
   - Group individual word boxes (identified by their IDs) into complete text bubbles or captions.
   - While grouping, clean and correct the OCR text to accurately represent the text in the image.
   - Fix common OCR errors such as misrecognized characters, incorrectly split or joined words, misinterpreted punctuation, and case errors.

2. Panel and Bubble Identification:
   - Assign a unique panel number to each group.
   - Within each panel, assign a unique text bubble or caption number.

3. Text Reconstruction:
   - Provide the complete, cleaned, and corrected text for each group as it should appear in the image.
   - Ensure the text is coherent, grammatically correct, and matches the content visible in the comic.

4. Spatial and Visual Analysis:
   - Use the spatial relationships between word boxes to determine groupings.
   - Consider the visual layout of the comic, including panel borders and bubble shapes.

5. Selective Inclusion of Background Text and Sound Effects:
   - Include background text (signs, posters, etc.) ONLY if it is crucial for understanding the story or context of the comic.
   - Include sound effects ONLY if they are integral to the narrative or significantly impact the scene's interpretation.
   - If background text or sound effects are purely decorative or do not add meaningful information to the story, DO NOT include them in the groupings.
   - When in doubt, err on the side of exclusion for background text and sound effects.

6. Text Types and Styles:
   - Dialogue: Usually in speech bubbles with a pointer to the speaker.
   - Thoughts: Often in cloud-like bubbles or italicized text.
   - Narration: Typically in rectangular boxes, often at the top or bottom of panels.
   - Sound Effects (if crucial): Can be stylized, vary in size, and placed near the source of the sound.
   - Background Text (if crucial): Signs, posters, or other environmental text within the comic world that impacts the story.

7. Accuracy Priority:
   - Prioritize accuracy in grouping, text reconstruction, and error correction.
   - If uncertain about a correction or inclusion, provide your best judgment but flag it in the notes.

OCR Text Locations:
```json
{0}
```

Output Format:
{{
  "groups": [
    {{
      "panel_id": "1",
      "text_bubble_id": "1-1",
      "box_ids": ["1", "2", "3"],
      "original_text": """The OCR output before cleaning""",
      "cleaned_text": """The corrected and cleaned text""",
      "type": "dialogue|thought|narration|sound_effect|background",
      "style": "normal|emphasized|angled|split",
      "notes": "Justification for inclusion if background or sound effect, any significant corrections or uncertainties|none"
    }},
    ...
  ]
}}

Additional Guidelines:
- Respect panel boundaries: Never group text from different panels.
- Maintain bubble integrity: Each group should correspond to a single text bubble, caption, or crucial sound effect/background text element.
- Use context clues to resolve ambiguities in text order, bubble assignment, or OCR errors.
- For included sound effects or background text, describe their significance to the story in the "notes" field.
- If you make significant corrections to the OCR text, briefly explain your reasoning in the "notes" field.
- Be particularly selective with sound effects and background text. Only include them if they are necessary for understanding the comic's narrative or context.

Analyze the image and OCR data thoroughly to produce accurate and contextually appropriate groupings with cleaned and corrected text that reflects the comic's essential narrative elements. Remember to include background text and sound effects only when they are crucial to the story or scene interpretation.
If sound effects and background text are purely decorative or do not add meaningful information, exclude them from your groupings.
Properly format the output json
'''