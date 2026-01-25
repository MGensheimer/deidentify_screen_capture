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


def draw_boxes(
    boxes: Iterable[Sequence[Sequence[float]]],
    canvas: np.ndarray,
    color: Color,
    outline_only: bool,
    ocr_mode: bool = False,
    original_image: np.ndarray | None = None,
) -> None:
    """Draw rectangles over detected text regions.
    
    If ocr_mode is True, fills boxes with black and overlays OCR'd text in white.
    """
    thickness = 2 if outline_only else -1
    for box in boxes:
        points = np.array(box, np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        if ocr_mode and original_image is not None:
            # OCR the region from the original image before we modify it
            text = ocr_region(original_image, x, y, w, h)
            
            # Fill the box with black and draw white border
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)
            
            # Draw the OCR'd text in white inside the box
            if text:
                # Use a small font that fits reasonably in the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                text_color = (255, 255, 255)  # White
                
                # Split text into lines and draw each one
                lines = text.split('\n')
                
                # Get line height from a sample character
                (_, line_height), baseline = cv2.getTextSize(
                    'Ay', font, font_scale, font_thickness
                )
                line_spacing = line_height + 2
                
                # Position first line at top-left of box with small padding
                text_x = x + 2
                text_y = y + line_height + 2
                
                for line in lines:
                    line = line.strip()
                    if line:
                        cv2.putText(
                            canvas, line, (text_x, text_y),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA
                        )
                    text_y += line_spacing
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)


def remove_text_in_memory(
    image: np.ndarray,
    *,
    fill_color: Color,
    outline_only: bool = False,
    ocr_mode: bool = False,
    input_size: Tuple[int, int] = (320, 320),
    detector_name: str = DETECTOR,
    detector=None,
):
    """Return a copy of `image` with detected text regions filled or outlined.
    
    If ocr_mode is True, fills regions with black and overlays OCR'd text in white.
    """

    if image is None:
        raise ValueError("image cannot be None")

    local_detector = detector or build_detector(input_size, detector_name=detector_name)
    boxes = detect_text_with_tiling(local_detector, image, input_size)
    output_image = image.copy()
    draw_boxes(boxes, output_image, fill_color, outline_only, ocr_mode=ocr_mode, original_image=image)
    return output_image
