# Changelog

All notable changes to ComiQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-20

### Added
- PP-OCRv6 engine (SOTA, 50 languages in one model), now the default: `ocr="paddleocr"` (aliases `"paddleocr6"`, `"ppocrv6"`)
- Configurable PP-OCRv6 model tiers via the `tier` option (`tiny` / `small` / `medium`, default `medium`); switching tiers between calls is supported (per-configuration instance cache)
- Custom `response_model` on `ComiQ(...)`: pass your own Pydantic model to extract more than text (speaker, emotion, translation, ...). Only `groups` → `box_ids` + `cleaned_text` are required; extra fields flow into `extract()` output. `comiq.ComicAnalysis` and `comiq.Group` are now exported for extension
- MLLM responses are now schema-enforced via [Instructor](https://python.useinstructor.com) with retries (`ai={"max_retries": 3}`), and the JSON schema in the prompt is generated from the response model
- Updated default MLLM model to `gemini-3.5-flash-lite` for high-throughput, low-latency extraction
- Cached OCR instances for Paddle and EasyOCR (no re-initialization on every call)

### Changed
- **BREAKING:** Dropped PaddleOCR 2.x support — the `"paddleocr"` engine now requires PaddleOCR 3.7+ and PaddlePaddle 3.x (installed automatically as base dependencies); `"paddleocr2"` is no longer registered
- Base dependencies switched from `paddleocr>=2.9,<3.0` / `paddlepaddle>=2.5,<3.0` (Python < 3.13 only) to `paddleocr>=3.7.0` / `paddlepaddle>=3.0.0,<=3.2.2` (all supported Pythons; pinned `<=3.2.2` to avoid upstream 3.3.x oneDNN regression); added `instructor>=1.0.0`
- OCR engines now import their backends lazily; importing `comiq` no longer warns or requires PaddleOCR at import time
- `sample.py` now uses the bundled example images instead of a hardcoded local path
- `extract()` output now includes `notes` and `original_text` per bubble, plus any custom `response_model` fields

### Fixed
- PP-OCRv6 engine: parse PaddleOCR 3.x results via dict access (attribute access silently returned 0 detections)
- PP-OCRv6 engine: never pass `enable_mkldnn` on GPU (breaks inference on paddlepaddle-gpu 3.3.1); default `text_det_limit_type="max"` so small images are not upscaled 6x

### Notes
- Upgrading from 0.1.x: remove the old Paddle packages first (`pip uninstall paddlepaddle paddlepaddle-gpu`), then `pip install comiq>=0.2.0` — PaddleOCR 3.x replaces 2.x within the same environment
- Performance: OCR instances are cached per configuration; on CPU the `tier` option is the main speed dial (`small` ~3x, `tiny` ~7x faster than `medium`), while `enable_hpi=True` (High-Performance Inference) is Linux-only. `enable_mkldnn` defaults to `False` on CPU for stability; users with `paddlepaddle<=3.2.2` can opt in to `enable_mkldnn=True` for up to 3.6x speedup (0.5s tiny, 1.4s small), with a warning emitted if enabled on 3.3.x.
- PP-OCRv6 defaults to CPU (`device="cpu"`; pass `device="gpu"` with a paddlepaddle-gpu 3.x install — Windows GPU is supported since PaddleOCR 3.2.0, incl. RTX 50 series)
- On CPU, `enable_mkldnn` defaults to `False`: oneDNN/PIR errors (`ConvertPirAttribute2RuntimeAttribute`) still affect Paddle 3.x builds (verified on paddlepaddle 3.3.1); on GPU the flag is not passed at all (breaks inference, verified on paddlepaddle-gpu 3.3.1)
- If a paddlepaddle-gpu wheel warns about a cuDNN version mismatch at startup, fix it before trusting results — with cuDNN 9.5 against a Paddle build expecting 9.9, GPU inference silently returned zero detections (upgrade via `pip install "nvidia-cudnn-cu12==9.9.*"`)
- Removed stray junk file `examples/images/e`

## [0.1.5] - 2025-01-29

### Documentation
- Removed CUDA_SUPPORT_EXPLAINED.md (unnecessary complexity)
- Clarified that EasyOCR GPU requires separate PyTorch installation
- Added response schema documentation showing output structure
- Simplified GPU setup instructions

## [0.1.4] - 2025-01-29

### Fixed
- Fixed critical bug: `mllm_api_key` parameter name mismatch in `comiq.py:89` causing `TypeError` when calling `process_with_ai()`
- Fixed sample.py return format handling (extract() returns List, not Dict)

### Changed
- **PaddleOCR is now included in base installation** for Python 3.8-3.12
- Python 3.13+ users: Use EasyOCR (PaddleOCR 2.x doesn't support Python 3.13)
- Simplified installation - just `pip install comiq` works on Python 3.8-3.12
- Removed PaddleOCR 3.x support due to instability (especially on Windows)
- Simplified codebase by removing version detection logic (~130 lines removed)

### Added
- Clear error message when PaddleOCR is used on Python 3.13+ with helpful alternatives
- Documentation for registering custom OCR engines (including PaddleOCR 3.x example)
- `paddleocr2` engine name as explicit alias for `paddleocr`

### Removed
- PaddleOCR 3.x automatic support (users can register custom engine if needed)
- `paddleocr3` optional dependency from pyproject.toml
- Complex version detection and auto-selection logic
- Windows GPU warnings for PaddleOCR 3.x (no longer relevant)

### Notes
- **Python 3.8-3.12:** Full support with PaddleOCR 2.x + EasyOCR
- **Python 3.13+:** EasyOCR only (or custom engines via `register_ocr_engine()`)
- **Breaking Changes:** None for Python 3.8-3.12 users
- **For Python 3.13+ users:** If you were using auto-installed PaddleOCR 3.x, switch to EasyOCR or register a custom PaddleOCR 3.x engine (see README)

## [0.1.3] - 2025-01-28

### Changed
- Added PaddlePaddle 2.x version constraints for stability
- Updated README with CUDA compatibility information

## [0.1.2] - 2025-01-27

### Initial Release
- Basic OCR functionality with PaddleOCR and EasyOCR
- AI-powered text grouping
- Support for multiple MLLM providers
