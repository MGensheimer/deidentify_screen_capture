# Example usage:
# uv run remove_text_from_image.py -i images/tps.jpeg -o output/output_image.png -c "red" --outline

import argparse
import os
import math

import cv2
import numpy as np


# Config flag to choose which detector to run: "DB18" or "EAST"
DETECTOR = "DB18"

COLOR_MAP = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "gray": (127, 127, 127),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and remove text from an image")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="Path to save the processed image (default: output/output_image_<detector>.png)",
    )
    parser.add_argument(
        "-c",
        "--color",
        dest="color",
        default="black",
        help="Fill/outline color (name, #RRGGBB, or R,G,B). Default: black",
    )
    parser.add_argument(
        "--outline",
        action="store_true",
        help="Draw only the outline of detected text regions",
    )
    return parser.parse_args()


def parse_color(color_str):
    """Parse color string to BGR tuple for OpenCV."""
    normalized = color_str.strip().lower()
    if normalized in COLOR_MAP:
        return COLOR_MAP[normalized]

    if normalized.startswith("#") and len(normalized) == 7:
        r = int(normalized[1:3], 16)
        g = int(normalized[3:5], 16)
        b = int(normalized[5:7], 16)
        return (b, g, r)

    if "," in normalized:
        parts = [p.strip() for p in normalized.split(",")]
        if len(parts) == 3:
            r, g, b = (int(p) for p in parts)
            if all(0 <= value <= 255 for value in (r, g, b)):
                return (b, g, r)

    raise ValueError(
        "Color must be a named color, #RRGGBB hex, or comma-separated R,G,B values"
    )


def build_detector(input_size):
    detector_name = DETECTOR.upper()
    if detector_name not in {"DB18", "EAST"}:
        raise ValueError("DETECTOR must be either 'DB18' or 'EAST'")

    if detector_name == "EAST":
        model = cv2.dnn_TextDetectionModel_EAST("weights/frozen_east_text_detection.pb")
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


def detect_text_with_tiling(detector, full_image, tile_size):
    """Run a detector on 320x320 tiles and merge detections to image coords."""
    tile_w, tile_h = tile_size
    height, width = full_image.shape[:2]
    all_boxes = []

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


def draw_boxes(boxes, canvas, color, outline_only):
    thickness = 2 if outline_only else -1
    for box in boxes:
        points = np.array(box, np.int32)
        x, y, w, h = cv2.boundingRect(points)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)


def main():
    args = parse_args()

    image = cv2.imread(args.input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.input_path}")

    fill_color = parse_color(args.color)
    output_image = image.copy()

    input_size = (320, 320)
    tile_w, tile_h = input_size
    tiles_x = math.ceil(image.shape[1] / tile_w)
    tiles_y = math.ceil(image.shape[0] / tile_h)
    print(
        f"Processing image in {tiles_x} tiles wide x {tiles_y} tiles high (tile size {tile_w}x{tile_h})"
    )

    detector = build_detector(input_size)
    default_filename = "output_image.png"
    boxes = detect_text_with_tiling(detector, image, input_size)
    draw_boxes(boxes, output_image, fill_color, args.outline)

    os.makedirs("output", exist_ok=True)
    output_path = args.output_path or os.path.join("output", default_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Saved processed image to {output_path}")


if __name__ == "__main__":
    main()
