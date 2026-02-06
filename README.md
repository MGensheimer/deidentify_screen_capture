# Deidentify Screen Capture Media (PHI Redaction)

Author: Michael Gensheimer (michael.gensheimer@gmail.com) and Codex CLI.

Text block removal with OpenCV was adapted from [this OpenCV tutorial](https://opencv.org/blog/text-detection-and-removal-using-opencv/).

## What This Repo Does

This repo removes on-screen PHI from images/videos and can deidentify subtitle text.

- Image redaction with OpenCV: `remove_text_from_image.py`
- Video redaction with OpenCV + whitelist protection: `remove_text_from_video.py`
- Image redaction with PaddleOCR + whitelist protection: `remove_text_from_image_paddleocr.py`
- Video redaction with PaddleOCR + whitelist protection: `remove_text_from_video_paddleocr.py`
- Subtitle deidentification (Ollama or Gemini): `deidentify_subtitles.py`

Sample output:

![Sample output](sample_output.jpg)

The process is not 100% reliable. Continue treating output as potentially containing PHI.

## Prerequisites

Required for core scripts:

- `uv`
- `ffmpeg` (for video recompression in scripts that use `--target-bitrate`)
- Python deps from `pyproject.toml` (installed via `uv` workflow)

Needed for specific workflows:

- `pytesseract` + system Tesseract install for OpenCV helper workflows that OCR text
- `paddleocr` for Paddle scripts (intentionally not installed by default in this repo)
- `ollama` model runtime for local subtitle deidentification
- Google Vertex AI access for `deidentify_subtitles.py --use-gemini`

## Whitelist Files

- `whitelist_terms.txt`
- `whitelist_regex.txt`

When whitelist matching is enabled by a script, detected text boxes matching either file are preserved (not blacked out).

Current whitelist usage:

- `remove_text_from_video.py`: uses whitelist files
- `remove_text_from_image_paddleocr.py`: uses whitelist files
- `remove_text_from_video_paddleocr.py`: uses whitelist files
- `remove_text_from_image.py`: does not use whitelist files

Matching behavior is case-sensitive substring/regex matching in script logic. Very short OCR text (length `<= 3`) is treated as keep.

## Quick Start

### OpenCV Image Redaction

```bash
uv run remove_text_from_image.py -i input/tps.jpeg -o output/output_image_db.png
```

Useful flags:

- `-c/--color` fill color (default `black`)
- `--outline` draw outlines only
- `--tile-overlap` default `0.5`
- `-p/--phrase` redact only boxes matching phrase(s)
- `--redact_dates_times`
- `--redact_digits N`
- `-v/--verbose`

### OpenCV Video Redaction (with whitelist protection)

```bash
uv run remove_text_from_video.py \
  -i input/test_recording.mov \
  -o output/output_video_opencv.mp4 \
  --interval 2 \
  --extra-keyframes 1 \
  --target-bitrate 1500k
```

Useful flags:

- `--interval` seconds between keyframe detections
- `--extra-keyframes` propagate detections to adjacent slots
- `--target-bitrate` ffmpeg bitrate string
- `--only_first_seconds N` test on first N seconds
- `--tile WIDTH HEIGHT`
- `--tile-overlap`
- `--detector {DB50,DB18,EAST}`
- `--tesseract-min-conf`
- `-v/--verbose`

### PaddleOCR Image Redaction (with whitelist protection)

```bash
uv run remove_text_from_image_paddleocr.py \
  -i input/tps.jpeg \
  -o output/output_image_paddleocr.png
```

### PaddleOCR Video Redaction (with whitelist protection)

```bash
uv run remove_text_from_video_paddleocr.py \
  -i input/test_recording.mov \
  -o output/output_video_paddleocr.mp4 \
  --interval 2 \
  --extra-keyframes 1 \
  --target-bitrate 1500k
```

Useful flags:

- `--interval`
- `--extra-keyframes`
- `--target-bitrate`
- `--only_first_seconds N`
- `-v/--verbose`

## Subtitle Deidentification

Basic (local Ollama):

```bash
uv run deidentify_subtitles.py -i input.srt -o output_cleaned.srt
```

Gemini (Vertex AI):

```bash
uv run deidentify_subtitles.py \
  -i input.srt \
  -o output_cleaned.srt \
  --use-gemini \
  --google-project YOUR_PROJECT
```

## Batch Processing Script

`process_study_videos.py` is a project-specific batch runner tied to local filesystem paths in that script. Update constants before use in another environment.

## Notes

- Default `uv run ...` execution is expected for all scripts.
- Video scripts remove audio when re-encoding with ffmpeg in current implementation (`-an`).
- For quick test cycles on videos, use `--only_first_seconds`.
