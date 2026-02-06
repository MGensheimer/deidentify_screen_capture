"""Deidentify text in a video using PaddleOCR with interval-based keyframe reuse.

Example:
  uv run remove_text_from_video_paddleocr.py \
    -i input/test_recording.mov \
    -o output/output_video_paddleocr.mp4 \
    --interval 2 \
    --extra-keyframes 1 \
    --target-bitrate 1500k
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Deque, Optional

import cv2
import numpy as np


SlotBufferEntry = dict[str, Any]
Detection = tuple[np.ndarray, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove text in a video using PaddleOCR and reuse detections across fixed intervals"
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="Path to save the processed video (default: output/output_video_paddleocr.mp4)",
    )
    parser.add_argument(
        "--interval",
        dest="interval_seconds",
        type=float,
        default=2.0,
        help=(
            "Seconds between reference detections. Frames within each interval reuse "
            "the detection results from its midpoint frame."
        ),
    )
    parser.add_argument(
        "--extra-keyframes",
        dest="extra_keyframes",
        type=int,
        default=0,
        help=(
            "Number of additional keyframe intervals to keep each detection active "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--target-bitrate",
        type=str,
        default="1500k",
        help="Explicit ffmpeg bitrate string (e.g., 1500k) used for recompression.",
    )
    parser.add_argument(
        "--only_first_seconds",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Only process the first N seconds of the video (useful for testing).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print keyframe OCR region counts for debugging.",
    )
    return parser.parse_args()


def _as_polygon(points: object) -> np.ndarray | None:
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
    detections: list[Detection] = []

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

    if isinstance(item, list):
        for entry in item:
            if isinstance(entry, (list, tuple)) and entry:
                poly = _as_polygon(entry[0])
                if poly is not None:
                    text = _extract_text(entry[1]) if len(entry) > 1 else ""
                    detections.append((poly, text))

    return detections


def extract_detections(results: object) -> list[Detection]:
    detections: list[Detection] = []

    if results is None:
        return detections

    if isinstance(results, (list, tuple)):
        for item in results:
            detections.extend(_extract_detections_from_item(item))
    else:
        detections.extend(_extract_detections_from_item(results))

    return detections


def detect_text_polygons(
    ocr: Any,
    frame: np.ndarray,
    whitelist_terms: list[str],
    whitelist_regexes: list[re.Pattern],
    verbose: bool = False,
) -> tuple[list[np.ndarray], int, int]:
    """Run OCR over a frame and return redaction polygons plus whitelist stats."""
    results = None
    last_error: Exception | None = None

    # Prefer newer predict API. PaddleOCR builds differ in accepted argument style.
    for call in (
        lambda: ocr.predict(input=frame),
        lambda: ocr.predict(frame),
        lambda: ocr.ocr(frame),
    ):
        try:
            results = call()
            break
        except Exception as exc:
            last_error = exc

    if results is None and last_error is not None:
        raise RuntimeError("PaddleOCR failed to run on video frame") from last_error

    detections = extract_detections(results)
    polygons_to_redact: list[np.ndarray] = []
    whitelisted_count = 0
    for idx, (poly, text) in enumerate(detections, start=1):
        keep = line_matches_whitelist(text, whitelist_terms, whitelist_regexes)
        if keep:
            whitelisted_count += 1
        else:
            polygons_to_redact.append(poly)

        if verbose:
            status = "[KEEP]" if keep else "[REDACT]"
            text_display = text.replace("\n", "\\n")
            print(f"  Box {idx}: {status} \"{text_display}\"")

    return polygons_to_redact, whitelisted_count, len(detections)


def redact_frame(frame: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    output = frame.copy()
    for poly in polygons:
        cv2.fillPoly(output, [poly], color=(0, 0, 0))
    return output


def maybe_recompress_video(video_path: str, target_bitrate: Optional[str]) -> None:
    """Recompress the written video via ffmpeg using a constant bitrate."""
    if not target_bitrate:
        return

    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary is None:
        print("Skipping recompression: ffmpeg not found on PATH.")
        return

    output_file = Path(video_path)
    if not output_file.exists():
        raise FileNotFoundError(f"Expected output video at {video_path} before recompression")

    temp_output = output_file.with_suffix(".recompressed.mp4")

    cmd = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        target_bitrate,
        "-maxrate",
        target_bitrate,
        "-bufsize",
        target_bitrate,
        "-movflags",
        "+faststart",
        "-an",
        str(temp_output),
    ]

    print(f"Recompressing with bitrate {target_bitrate}")
    print("Running ffmpeg to recompress output:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stdout:\n" + result.stdout)
        print("ffmpeg stderr:\n" + result.stderr)
        raise RuntimeError("ffmpeg recompression failed")

    os.replace(temp_output, video_path)
    print("Recompression complete; output overwritten with compressed version.")


def main() -> None:
    args = parse_args()

    if args.interval_seconds <= 0:
        raise ValueError("--interval must be greater than 0")
    if args.extra_keyframes < 0:
        raise ValueError("--extra-keyframes must be 0 or a positive integer")
    if args.only_first_seconds is not None and args.only_first_seconds <= 0:
        raise ValueError("--only_first_seconds must be greater than 0")

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
    whitelist_terms_path = Path(__file__).resolve().parent / "whitelist_terms.txt"
    whitelist_regex_path = Path(__file__).resolve().parent / "whitelist_regex.txt"
    whitelist_terms = load_whitelist_terms(whitelist_terms_path)
    whitelist_regexes = load_whitelist_regexes(whitelist_regex_path)

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {args.input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Unable to determine video FPS from input stream")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("output", exist_ok=True)
    output_path = args.output_path or os.path.join("output", "output_video_paddleocr.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    total_frames_value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_value if total_frames_value > 0 else None

    if args.only_first_seconds is not None and total_frames is not None:
        limited_frames = int(args.only_first_seconds * fps)
        total_frames = min(total_frames, limited_frames)

    if total_frames:
        total_duration = total_frames / fps
        duration_note = ""
        if args.only_first_seconds is not None:
            duration_note = f" (limited to first {args.only_first_seconds}s)"
        print(
            f"Processing {total_frames} frames (~{total_duration:.1f}s) from {args.input_path}{duration_note}"
        )
    else:
        print(f"Processing video {args.input_path} (unknown frame count)")

    interval = args.interval_seconds
    frame_duration = 1.0 / fps
    frame_idx = 0
    last_percent_bucket = -1
    last_frames_log = 0
    percent_step = 5

    slot_index: Optional[int] = None
    slot_sample_time = interval / 2
    slot_frames: list[tuple[np.ndarray, float]] = []
    slot_polygons: Optional[list[np.ndarray]] = None
    slot_whitelisted_count: Optional[int] = None
    slot_detection_count: Optional[int] = None
    slot_reference_timestamp: Optional[float] = None

    slot_queue: Deque[SlotBufferEntry] = deque()
    coverage_map: DefaultDict[int, list[np.ndarray]] = defaultdict(list)
    highest_finalized_slot: Optional[int] = None

    def flush_ready_slots(force: bool = False) -> None:
        nonlocal highest_finalized_slot
        if not slot_queue:
            return

        if force:
            threshold = float("inf")
        else:
            if highest_finalized_slot is None:
                return
            threshold = highest_finalized_slot - args.extra_keyframes

        while slot_queue and slot_queue[0]["index"] <= threshold:
            slot_entry = slot_queue.popleft()
            idx = slot_entry["index"]
            frames = slot_entry["frames"]
            polygons = coverage_map.pop(idx, [])
            for frame in frames:
                writer.write(redact_frame(frame, polygons) if polygons else frame)

    def finalize_slot(sample_time: float) -> None:
        nonlocal slot_frames
        nonlocal slot_polygons
        nonlocal slot_whitelisted_count
        nonlocal slot_detection_count
        nonlocal slot_reference_timestamp
        nonlocal highest_finalized_slot

        if not slot_frames:
            return

        polygons = slot_polygons
        whitelisted_count = slot_whitelisted_count
        detection_count = slot_detection_count
        reference_time = slot_reference_timestamp

        if polygons is None:
            sample_frame, reference_time = min(
                slot_frames, key=lambda item: abs(item[1] - sample_time)
            )
            polygons, whitelisted_count, detection_count = detect_text_polygons(
                ocr,
                sample_frame,
                whitelist_terms,
                whitelist_regexes,
                verbose=args.verbose,
            )
            slot_reference_timestamp = reference_time

        slot_queue.append({"index": slot_index, "frames": [frame for frame, _ in slot_frames]})

        if polygons:
            for offset in range(-args.extra_keyframes, args.extra_keyframes + 1):
                target_slot = slot_index + offset
                if target_slot < 0:
                    continue
                coverage_map[target_slot].extend(polygons)

        frame_count = len(slot_frames)
        polygon_count = len(polygons) if polygons else 0
        whitelisted_total = whitelisted_count if whitelisted_count is not None else 0
        detected_total = detection_count if detection_count is not None else 0
        reference_desc = f"{reference_time:.2f}s" if reference_time is not None else "unknown time"
        print(
            f"Processed slot {slot_index} ({frame_count} frames) using reference at "
            f"{reference_desc} with {polygon_count} redacted region(s), "
            f"{whitelisted_total} whitelisted region(s) out of {detected_total} detected"
        )

        highest_finalized_slot = slot_index
        flush_ready_slots()

        slot_frames = []
        slot_polygons = None
        slot_whitelisted_count = None
        slot_detection_count = None
        slot_reference_timestamp = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx * frame_duration
            if args.only_first_seconds is not None and timestamp >= args.only_first_seconds:
                break

            frame_idx += 1
            processed_frames = frame_idx

            if total_frames:
                percent_complete = (processed_frames / total_frames) * 100
                percent_bucket = int(percent_complete // percent_step)
                if percent_bucket != last_percent_bucket and percent_complete < 100:
                    print(
                        f"Progress: {percent_complete:5.1f}% "
                        f"({processed_frames}/{total_frames} frames)"
                    )
                    last_percent_bucket = percent_bucket
            else:
                if processed_frames - last_frames_log >= 150:
                    print(f"Processed {processed_frames} frames...")
                    last_frames_log = processed_frames

            current_slot = int(timestamp / interval)

            if slot_index is None:
                slot_index = current_slot
                slot_sample_time = slot_index * interval + (interval / 2)
            elif current_slot != slot_index:
                finalize_slot(slot_sample_time)
                slot_index = current_slot
                slot_sample_time = slot_index * interval + (interval / 2)

            slot_frames.append((frame.copy(), timestamp))

            if slot_polygons is None and timestamp >= slot_sample_time:
                (
                    slot_polygons,
                    slot_whitelisted_count,
                    slot_detection_count,
                ) = detect_text_polygons(
                    ocr,
                    frame,
                    whitelist_terms,
                    whitelist_regexes,
                    verbose=args.verbose,
                )
                slot_reference_timestamp = timestamp
                if args.verbose:
                    print(
                        f"Keyframe at {timestamp:.2f}s: detected {slot_detection_count} region(s), "
                        f"whitelisted {slot_whitelisted_count}, redacting {len(slot_polygons)}"
                    )

        finalize_slot(slot_sample_time)
        flush_ready_slots(force=True)

    finally:
        cap.release()
        writer.release()

    maybe_recompress_video(output_path, args.target_bitrate)

    if total_frames:
        print(f"Progress: 100.0% ({frame_idx}/{total_frames} frames)")

    print(
        f"Saved processed video to {output_path} "
        f"({frame_idx} frames, target interval {interval} s)"
    )


if __name__ == "__main__":
    main()
