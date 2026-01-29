# Changelog

All notable changes to ComiQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
