"""Deidentify text in an image using PaddleOCR by blacking out detected text regions.

Example:
  uv run remove_text_from_image_paddleocr.py -i input/tps.jpeg -o output/output_image_paddleocr.png
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

Detection = tuple[np.ndarray, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deidentify an image using PaddleOCR text detection",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="Path to save deidentified image",
    )
    return parser.parse_args()


def _as_polygon(points: object) -> np.ndarray | None:
    """Normalize candidate points into an Nx2 int32 polygon array."""
    try:
        arr = np.asarray(points, dtype=np.float32)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return None
    return np.round(arr).astype(np.int32)


def load_whitelist_terms(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Whitelist terms file not found: {path}")
    terms: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            terms.append(text.lstrip("\ufeff"))
    return terms


def load_whitelist_regexes(path: Path) -> list[re.Pattern]:
    if not path.exists():
        raise FileNotFoundError(f"Whitelist regex file not found: {path}")
    patterns: list[re.Pattern] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                patterns.append(re.compile(text))
            except re.error as exc:
                raise ValueError(f"Invalid regex in {path}: {text}") from exc
    return patterns


def line_matches_whitelist(text: str, terms: list[str], regexes: list[re.Pattern]) -> bool:
    if len(text) <= 3:
        return True
    for term in terms:
        if term in text:
            return True
    for pattern in regexes:
        if pattern.search(text):
            return True
    return False


def _extract_text(entry: object) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (list, tuple)) and entry:
        if isinstance(entry[0], str):
            return entry[0]
    return ""


def _extract_detections_from_item(item: object) -> list[Detection]:
    """Extract (polygon, text) detections from one prediction item."""
    detections: list[Detection] = []

    # PaddleOCR v3 style: dict with rec_polys / dt_polys
    if isinstance(item, dict):
        polygons: list[np.ndarray] = []
        for key in ("rec_polys", "dt_polys", "polys", "points"):
            value = item.get(key)
            if value is None:
                continue
            parsed: list[np.ndarray] = []
            try:
                for candidate in value:
                    poly = _as_polygon(candidate)
                    if poly is not None:
                        parsed.append(poly)
            except TypeError:
                poly = _as_polygon(value)
                if poly is not None:
                    parsed.append(poly)
            if parsed:
                polygons = parsed
                break

        texts_raw = item.get("rec_texts", item.get("texts", []))
        texts: list[str] = []
        if isinstance(texts_raw, (list, tuple)):
            texts = [_extract_text(value) for value in texts_raw]

        for i, poly in enumerate(polygons):
            text = texts[i] if i < len(texts) else ""
            detections.append((poly, text))

    # PaddleOCR v2 style: [ [box, (text, conf)], ... ]
    if isinstance(item, list):
        for entry in item:
            if isinstance(entry, (list, tuple)) and entry:
                poly = _as_polygon(entry[0])
                if poly is not None:
                    text = _extract_text(entry[1]) if len(entry) > 1 else ""
                    detections.append((poly, text))

    return detections


def extract_detections(results: object) -> list[Detection]:
    """Collect text detections from PaddleOCR `predict` return payload."""
    detections: list[Detection] = []

    if results is None:
        return detections

    if isinstance(results, (list, tuple)):
        for item in results:
            detections.extend(_extract_detections_from_item(item))
    else:
        detections.extend(_extract_detections_from_item(results))

    return detections


def redact_image(image: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Fill each detected polygon with black pixels."""
    output = image.copy()
    for poly in polygons:
        cv2.fillPoly(output, [poly], color=(0, 0, 0))
    return output


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.input_path}")

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError(
            "paddleocr is required for this script. Install it on the target machine, "
            "then rerun."
        ) from exc

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # Prefer the newer predict API from PaddleOCR sample docs.
    try:
        results = ocr.predict(input=args.input_path)
    except TypeError:
        # Backward compatibility with environments exposing only ocr(...).
        results = ocr.ocr(args.input_path)

    whitelist_terms_path = Path(__file__).resolve().parent / "whitelist_terms.txt"
    whitelist_regex_path = Path(__file__).resolve().parent / "whitelist_regex.txt"
    whitelist_terms = load_whitelist_terms(whitelist_terms_path)
    whitelist_regexes = load_whitelist_regexes(whitelist_regex_path)

    detections = extract_detections(results)
    polygons_to_redact: list[np.ndarray] = []
    whitelisted_count = 0
    for poly, text in detections:
        if line_matches_whitelist(text, whitelist_terms, whitelist_regexes):
            whitelisted_count += 1
            continue
        polygons_to_redact.append(poly)

    if not detections:
        print("No text detected; writing original image.")

    output_image = redact_image(image, polygons_to_redact)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output_image):
        raise RuntimeError(f"Failed to write output image to {output_path}")

    print(
        f"Redacted {len(polygons_to_redact)} text region(s); "
        f"skipped {whitelisted_count} whitelisted region(s)"
    )
    print(f"Saved deidentified image to {output_path}")


if __name__ == "__main__":
    main()
