"""Utilities for removing detected text from images in memory."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np
import pytesseract


# Config flag to choose which detector to run: "DB18" or "EAST"
DETECTOR = "DB18"


Color = Tuple[int, int, int]


def build_detector(input_size: Tuple[int, int], detector_name: str = DETECTOR):
    detector_name = detector_name.upper()
    if detector_name not in {"DB18", "EAST"}:
        raise ValueError("detector_name must be either 'DB18' or 'EAST'")

    if detector_name == "EAST":
        model = cv2.dnn_TextDetectionModel_EAST(
            "weights/frozen_east_text_detection.pb"
        )
        conf_thresh = 0.8
        nms_thresh = 0.4
        model.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
        model.setInputParams(1.0, input_size, (123.68, 116.78, 103.94), True)
    else:
        model = cv2.dnn_TextDetectionModel_DB("weights/DB_TD500_resnet18.onnx")
        bin_thresh = 0.3
        poly_thresh = 0.5
        mean = (122.67891434, 116.66876762, 104.00698793)
        model.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
        model.setInputParams(1.0 / 255, input_size, mean, True)

    return model


def detect_text_with_tiling(
    detector, full_image: np.ndarray, tile_size: Tuple[int, int]
) -> list[np.ndarray]:
    """Run a detector on fixed-size tiles and merge detections to image coords."""

    tile_w, tile_h = tile_size
    height, width = full_image.shape[:2]
    all_boxes: list[np.ndarray] = []

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            tile = full_image[y : min(y + tile_h, height), x : min(x + tile_w, width)]
            if tile.size == 0:
                continue

            boxes, _ = detector.detect(tile)
            if boxes is None:
                continue

            for box in boxes:
                adjusted = np.array(box, dtype=np.float32)
                adjusted[:, 0] += x
                adjusted[:, 1] += y
                all_boxes.append(adjusted)

    return all_boxes


def ocr_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Extract text from a region of the image using Tesseract OCR."""
    # Ensure bounds are within image
    img_h, img_w = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return ""
    
    region = image[y:y+h, x:x+w]
    if region.size == 0:
        return ""
    
    # Convert to RGB for pytesseract (OpenCV uses BGR)
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    
    # Run OCR (PSM 6 = assume a single uniform block of text)
    try:
        text = pytesseract.image_to_string(region_rgb, config='--psm 6').strip()
    except Exception:
        text = ""
    
    return text


def normalize_for_matching(text: str) -> str:
    """Normalize text for phrase matching: lowercase, remove spaces and punctuation."""
    import re
    # Remove everything except alphanumeric characters
    return re.sub(r'[^a-z0-9]', '', text.lower())


def matches_any_phrase(text: str, phrases: list[str]) -> bool:
    """Check if normalized text contains any of the normalized phrases."""
    normalized_text = normalize_for_matching(text)
    for phrase in phrases:
        normalized_phrase = normalize_for_matching(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            return True
    return False


def contains_date_or_time(text: str) -> bool:
    """Check if text contains any date or time pattern."""
    import re
    
    # Date patterns (comprehensive list)
    date_patterns = [
        # ISO format: YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b',
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY (day first)
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b',
        # MM-DD-YY, M-D-YY, MM/DD/YY, M/D/YY (2-digit year)
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2}\b',
        # Month name formats: Jan 25, 2024 / January 25, 2024 / 25 Jan 2024
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
        r'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
        r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,?\s+\d{2,4})?\b',
        # Month/day patterns: M/D, M-D, MM/DD, MM-DD (e.g., 1/7, 01-07)
        # Month 1-12, day 1-31, using / or - as separator
        r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])\b',
        # Standalone 4-digit years 2000-2040 (plausible CT scan years)
        # Not preceded or followed by digit or decimal point (to avoid matching 20000 or 2000.1)
        r'(?<![0-9.])(?:20[0-3][0-9]|2040)(?![0-9.])',
    ]
    
    # Time patterns (comprehensive list)
    time_patterns = [
        # 24-hour format: HH:MM, HH:MM:SS, H:MM
        r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
        # 12-hour format with AM/PM: 11:23 AM, 7:11pm, 11:23:45 AM
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b',
        # Hour only with AM/PM: 7 AM, 11pm
        r'\b\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b',
    ]
    
    all_patterns = date_patterns + time_patterns
    
    for pattern in all_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def has_min_consecutive_digits(text: str, min_digits: int) -> bool:
    """Check if text contains at least min_digits consecutive digits after removing spaces/punctuation."""
    import re
    # Remove spaces and punctuation, keep only alphanumeric
    stripped = re.sub(r'[^a-zA-Z0-9]', '', text)
    # Find longest run of consecutive digits
    digit_runs = re.findall(r'\d+', stripped)
    for run in digit_runs:
        if len(run) >= min_digits:
            return True
    return False


def should_redact_box(
    text: str,
    redact_phrases: list[str] | None,
    redact_dates_times: bool,
    redact_min_digits: int | None,
) -> bool:
    """Determine if a text box should be redacted based on filtering criteria."""
    # If no filters are active, don't redact selectively
    if not redact_phrases and not redact_dates_times and redact_min_digits is None:
        return False  # Caller handles this case separately
    
    # Check phrase matching
    if redact_phrases and matches_any_phrase(text, redact_phrases):
        return True
    
    # Check date/time patterns
    if redact_dates_times and contains_date_or_time(text):
        return True
    
    # Check minimum consecutive digits
    if redact_min_digits is not None and has_min_consecutive_digits(text, redact_min_digits):
        return True
    
    return False


def draw_boxes(
    boxes: Iterable[Sequence[Sequence[float]]],
    canvas: np.ndarray,
    color: Color,
    outline_only: bool,
    redact_phrases: list[str] | None = None,
    redact_dates_times: bool = False,
    redact_min_digits: int | None = None,
    verbose: bool = False,
    original_image: np.ndarray | None = None,
) -> None:
    """Draw rectangles over detected text regions.
    
    If any redaction filter is active (redact_phrases, redact_dates_times, or
    redact_min_digits), only boxes matching a filter are redacted (filled black
    with white border and "[REDACTED]" text). Non-matching boxes are left untouched.
    
    If verbose is True, prints OCR'd text from each detected text box.
    """
    thickness = 2 if outline_only else -1
    has_filters = redact_phrases or redact_dates_times or redact_min_digits is not None
    
    for i, box in enumerate(boxes):
        points = np.array(box, np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        if has_filters and original_image is not None:
            # OCR the region from the original image
            text = ocr_region(original_image, x, y, w, h)
            
            if verbose:
                redacted = should_redact_box(text, redact_phrases, redact_dates_times, redact_min_digits)
                status = "[REDACTED]" if redacted else "[kept]"
                # Show text with escaped newlines for readability
                text_display = text.replace('\n', '\\n')
                print(f"Box {i+1} @ ({x},{y}) {w}x{h}: {status} \"{text_display}\"")
            
            # Check if text matches any redaction criteria
            if should_redact_box(text, redact_phrases, redact_dates_times, redact_min_digits):
                # Fill the box with black and draw white border
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), -1)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)
                
                # Draw "[REDACTED]" in white
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                text_color = (255, 255, 255)  # White
                
                (_, line_height), _ = cv2.getTextSize(
                    '[REDACTED]', font, font_scale, font_thickness
                )
                text_x = x + 2
                text_y = y + line_height + 2
                
                cv2.putText(
                    canvas, '[REDACTED]', (text_x, text_y),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA
                )
            # Non-matching boxes are left untouched
        else:
            # No filtering - apply fill/outline to all boxes
            if verbose and original_image is not None:
                text = ocr_region(original_image, x, y, w, h)
                text_display = text.replace('\n', '\\n')
                print(f"Box {i+1} @ ({x},{y}) {w}x{h}: [filled] \"{text_display}\"")
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)


def remove_text_in_memory(
    image: np.ndarray,
    *,
    fill_color: Color,
    outline_only: bool = False,
    redact_phrases: list[str] | None = None,
    redact_dates_times: bool = False,
    redact_min_digits: int | None = None,
    verbose: bool = False,
    input_size: Tuple[int, int] = (320, 320),
    detector_name: str = DETECTOR,
    detector=None,
):
    """Return a copy of `image` with detected text regions filled or outlined.
    
    If any redaction filter is active (redact_phrases, redact_dates_times, or
    redact_min_digits), only boxes matching a filter are redacted.
    Non-matching boxes are left untouched.
    
    If verbose is True, prints OCR'd text from each detected text box.
    """

    if image is None:
        raise ValueError("image cannot be None")

    local_detector = detector or build_detector(input_size, detector_name=detector_name)
    boxes = detect_text_with_tiling(local_detector, image, input_size)
    
    if verbose:
        print(f"Detected {len(boxes)} text box(es)")
    
    output_image = image.copy()
    draw_boxes(
        boxes,
        output_image,
        fill_color,
        outline_only,
        redact_phrases=redact_phrases,
        redact_dates_times=redact_dates_times,
        redact_min_digits=redact_min_digits,
        verbose=verbose,
        original_image=image,
    )
    return output_image
