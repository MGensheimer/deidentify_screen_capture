# Uses DB18 deep learning model to detect text and either fill in the area with color (default) or outline it.
#
# Example usage:
# uv run remove_text_from_image.py -i input/tps.jpeg -o output/output_image.png -c "red" --outline
#
# Adapted from https://opencv.org/blog/text-detection-and-removal-using-opencv/

import argparse
import os
import math

import cv2

from text_removal_helper import DETECTOR, remove_text_in_memory

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
    parser.add_argument(
        "-p",
        "--phrase",
        dest="phrases",
        action="append",
        default=None,
        help="Phrase to redact (can be used multiple times). Matching is case-insensitive "
             "and ignores spaces/punctuation. Only matching text boxes are redacted; "
             "non-matching boxes are left untouched.",
    )
    parser.add_argument(
        "--redact_dates_times",
        action="store_true",
        help="Redact all dates and times (e.g., YYYY-MM-DD, MM/DD/YYYY, 11:23, 7:11 AM, etc.)",
    )
    parser.add_argument(
        "--redact_digits",
        type=int,
        metavar="N",
        default=None,
        help="Redact text boxes containing at least N consecutive digits "
             "(spaces/punctuation are ignored when counting).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print OCR'd text from each detected text box (useful for debugging).",
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


def main():
    args = parse_args()

    image = cv2.imread(args.input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.input_path}")

    fill_color = parse_color(args.color)

    input_size = (320, 320)
    tile_w, tile_h = input_size
    tiles_x = math.ceil(image.shape[1] / tile_w)
    tiles_y = math.ceil(image.shape[0] / tile_h)
    print(
        f"Processing image in {tiles_x} tiles wide x {tiles_y} tiles high (tile size {tile_w}x{tile_h})"
    )

    output_image = remove_text_in_memory(
        image,
        fill_color=fill_color,
        outline_only=args.outline,
        redact_phrases=args.phrases,
        redact_dates_times=args.redact_dates_times,
        redact_min_digits=args.redact_digits,
        verbose=args.verbose,
        input_size=input_size,
        detector_name=DETECTOR,
    )
    default_filename = f"output_image_{DETECTOR.lower()}.png"

    os.makedirs("output", exist_ok=True)
    output_path = args.output_path or os.path.join("output", default_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Saved processed image to {output_path}")


if __name__ == "__main__":
    main()
