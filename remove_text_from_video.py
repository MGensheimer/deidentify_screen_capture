# Detect text on representative video frames and apply results across intervals.
#
# Example usage:
# uv run remove_text_from_video.py -i recording.mp4 -o output/study_recording.mp4 --interval 2 --extra-keyframes 1 --target-bitrate 1500k


from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from text_removal_helper import (
    build_detector,
    detect_text_with_tiling,
    draw_boxes,
    preprocess_for_text_detection,
)


SlotBufferEntry = Dict[str, Any]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Remove text in a video by reusing detections across fixed intervals"
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
        help="Path to save the processed video (default: output/output_video.mp4)",
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
        "--detector",
        dest="detector_name",
        choices=["DB50", "DB18", "EAST"],
        default="DB50",
        help="Text detector backend to use (default: %(default)s)",
    )
    parser.add_argument(
        "--tesseract-min-conf",
        type=float,
        default=-1.0,
        help=(
            "Minimum Tesseract confidence to keep a word when building line boxes "
            "(default: -1, include all non-empty text)."
        ),
    )
    parser.add_argument(
        "--tile",
        dest="tile_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(736, 736),
        help="Tile size fed to the detector (default: 736 736)",
    )
    parser.add_argument(
        "--tile-overlap",
        dest="tile_overlap",
        type=float,
        default=0.25,
        help="Tile overlap as a fraction of tile size (0.0-<1.0). Default: 0.25",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Print OCR'd text from each detected text box on keyframes (useful for debugging).",
    )
    parser.add_argument(
        "--only_first_seconds",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Only process the first N seconds of the video (useful for testing).",
    )
    return parser.parse_args()


def load_whitelist_terms(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Whitelist terms file not found: {path}")
    terms: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            terms.append(text.lstrip("\ufeff"))
    return terms


def load_whitelist_regexes(path: Path) -> List[re.Pattern]:
    if not path.exists():
        raise FileNotFoundError(f"Whitelist regex file not found: {path}")
    patterns: List[re.Pattern] = []
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


def line_matches_whitelist(
    line_text: str,
    terms: List[str],
    regexes: List[re.Pattern],
) -> bool:
    if len(line_text) <= 3:
        return True
    for term in terms:
        if term in line_text:
            return True
    for pattern in regexes:
        if pattern.search(line_text):
            return True
    return False


def detect_tesseract_whitelist_boxes(
    frame: np.ndarray,
    *,
    min_confidence: float,
    whitelist_terms: List[str],
    whitelist_regexes: List[re.Pattern],
    verbose: bool,
) -> List[np.ndarray]:
    """Run Tesseract on a full frame and return line-level boxes."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    config = "--psm 11 -c load_system_dawg=0 -c load_freq_dawg=0"
    ocr_data = pytesseract.image_to_data(
        rgb,
        output_type=Output.DICT,
        lang="eng",
        config=config,
    )

    lines: Dict[Tuple[int, int, int], List[Tuple[str, int, int, int, int]]] = {}
    num_items = len(ocr_data["text"])
    for i in range(num_items):
        text = ocr_data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(ocr_data["conf"][i])
        except ValueError:
            conf = -1.0
        if conf < min_confidence:
            continue
        key = (
            int(ocr_data["block_num"][i]),
            int(ocr_data["par_num"][i]),
            int(ocr_data["line_num"][i]),
        )
        left = int(ocr_data["left"][i])
        top = int(ocr_data["top"][i])
        width = int(ocr_data["width"][i])
        height = int(ocr_data["height"][i])
        lines.setdefault(key, []).append((text, left, top, width, height))

    if verbose:
        print(f"Tesseract detected {len(lines)} line(s)")

    boxes: List[np.ndarray] = []

    for idx, items in enumerate(lines.values(), start=1):
        texts = [item[0] for item in items]
        lefts = [item[1] for item in items]
        tops = [item[2] for item in items]
        rights = [item[1] + item[3] for item in items]
        bottoms = [item[2] + item[4] for item in items]
        left = min(lefts)
        top = min(tops)
        right = max(rights)
        bottom = max(bottoms)
        line_text = " ".join(texts)
        keep_line = line_matches_whitelist(
            line_text, whitelist_terms, whitelist_regexes
        )
        if verbose:
            status = "[KEEP]" if keep_line else "[REDACT]"
            text_display = line_text.replace("\n", "\\n")
            print(
                f"  Line {idx} @ ({left},{top}) {right-left}x{bottom-top}: "
                f"{status} \"{text_display}\""
            )
        if not keep_line:
            continue

        box = np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float32,
        )
        boxes.append(box)

    if verbose:
        print(f"Tesseract kept {len(boxes)} line(s) after whitelisting")

    return boxes


def maybe_recompress_video(video_path: str, target_bitrate: Optional[str]):
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

    print(f"Recompressing with bitrate {target_bitrate}")

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

    print("Running ffmpeg to recompress output:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stdout:\n" + result.stdout)
        print("ffmpeg stderr:\n" + result.stderr)
        raise RuntimeError("ffmpeg recompression failed")

    os.replace(temp_output, video_path)
    print("Recompression complete; output overwritten with compressed version.")


def main():
    args = parse_args()

    if args.interval_seconds <= 0:
        raise ValueError("--interval must be greater than 0")
    if args.extra_keyframes < 0:
        raise ValueError("--extra-keyframes must be 0 or a positive integer")
    if not (0.0 <= args.tile_overlap < 1.0):
        raise ValueError("--tile-overlap must be >= 0 and < 1")

    video_path = args.input_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Unable to determine video FPS from input stream")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tile_size = tuple(args.tile_size)
    tile_overlap = args.tile_overlap
    detector = build_detector(tile_size, detector_name=args.detector_name)
    whitelist_terms_path = Path(__file__).resolve().parent / "whitelist_terms.txt"
    whitelist_regex_path = Path(__file__).resolve().parent / "whitelist_regex.txt"
    whitelist_terms = load_whitelist_terms(whitelist_terms_path)
    whitelist_regexes = load_whitelist_regexes(whitelist_regex_path)

    os.makedirs("output", exist_ok=True)
    default_name = "output_video.mp4"
    output_path = args.output_path or os.path.join("output", default_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    total_frames_value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_value if total_frames_value > 0 else None
    
    # Adjust total frames if --only_first_seconds is set
    if args.only_first_seconds is not None and total_frames:
        limited_frames = int(args.only_first_seconds * fps)
        total_frames = min(total_frames, limited_frames)
    
    if total_frames:
        total_duration = total_frames / fps
        duration_note = ""
        if args.only_first_seconds is not None:
            duration_note = f" (limited to first {args.only_first_seconds}s)"
        print(
            f"Processing {total_frames} frames (~{total_duration:.1f}s) from {video_path}{duration_note}"
        )
    else:
        print(f"Processing video {video_path} (unknown frame count)")

    interval = args.interval_seconds
    frame_idx = 0
    slot_index: Optional[int] = None
    slot_frames: List[Tuple[np.ndarray, float]] = []
    slot_boxes: Optional[List[Sequence[Sequence[float]]]] = None
    slot_whitelist_boxes: Optional[List[Sequence[Sequence[float]]]] = None
    slot_sample_time: float = interval / 2
    slot_reference_timestamp: Optional[float] = None
    last_percent_bucket = -1
    last_frames_log = 0
    percent_step = 5
    slot_queue: Deque[SlotBufferEntry] = deque()
    coverage_map: DefaultDict[int, List[Sequence[Sequence[float]]]] = defaultdict(list)
    whitelist_map: DefaultDict[int, List[Sequence[Sequence[float]]]] = defaultdict(list)
    highest_finalized_slot: Optional[int] = None

    def detect_opencv_boxes(frame):
        detection_frame = preprocess_for_text_detection(frame)
        boxes = detect_text_with_tiling(
            detector, detection_frame, tile_size, overlap=tile_overlap
        )
        if args.verbose:
            print(f"Detected {len(boxes)} text box(es)")
        return boxes

    def detect_whitelist_boxes(frame):
        return detect_tesseract_whitelist_boxes(
            frame,
            min_confidence=args.tesseract_min_conf,
            whitelist_terms=whitelist_terms,
            whitelist_regexes=whitelist_regexes,
            verbose=args.verbose,
        )

    def restore_boxes(
        boxes: List[Sequence[Sequence[float]]],
        source_frame: np.ndarray,
        target_frame: np.ndarray,
    ) -> None:
        for box in boxes:
            points = np.array(box, np.int32)
            x, y, w, h = cv2.boundingRect(points)
            if w <= 0 or h <= 0:
                continue
            target_frame[y : y + h, x : x + w] = source_frame[
                y : y + h, x : x + w
            ]

    def rect_from_box(box: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
        points = np.array(box, np.int32)
        x, y, w, h = cv2.boundingRect(points)
        return (x, y, w, h)

    def intersection_rect(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0:
            return None
        return (left, top, w, h)

    def union_area(rects: List[Tuple[int, int, int, int]]) -> int:
        if not rects:
            return 0
        x_coords = set()
        for x, y, w, h in rects:
            x_coords.add(x)
            x_coords.add(x + w)
        xs = sorted(x_coords)
        area = 0
        for i in range(len(xs) - 1):
            x_left = xs[i]
            x_right = xs[i + 1]
            if x_right <= x_left:
                continue
            y_intervals = []
            for x, y, w, h in rects:
                if x <= x_left and (x + w) >= x_right:
                    y_intervals.append((y, y + h))
            if not y_intervals:
                continue
            y_intervals.sort()
            merged = []
            cur_start, cur_end = y_intervals[0]
            for start, end in y_intervals[1:]:
                if start <= cur_end:
                    cur_end = max(cur_end, end)
                else:
                    merged.append((cur_start, cur_end))
                    cur_start, cur_end = start, end
            merged.append((cur_start, cur_end))
            covered_y = sum(end - start for start, end in merged)
            area += (x_right - x_left) * covered_y
        return area

    def filter_opencv_boxes(
        opencv_boxes: List[Sequence[Sequence[float]]],
        whitelist_boxes: List[Sequence[Sequence[float]]],
        coverage_threshold: float = 0.9,
    ) -> List[Sequence[Sequence[float]]]:
        if not opencv_boxes or not whitelist_boxes:
            return opencv_boxes
        whitelist_rects = [rect_from_box(box) for box in whitelist_boxes]
        kept = []
        for box in opencv_boxes:
            rect = rect_from_box(box)
            _, _, w, h = rect
            if w <= 0 or h <= 0:
                continue
            intersections = []
            for wrect in whitelist_rects:
                intersect = intersection_rect(rect, wrect)
                if intersect is not None:
                    intersections.append(intersect)
            if not intersections:
                kept.append(box)
                continue
            covered = union_area(intersections)
            if covered < coverage_threshold * (w * h):
                kept.append(box)
        return kept

    def flush_ready_slots(force: bool = False):
        nonlocal slot_queue, coverage_map, whitelist_map, highest_finalized_slot
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
            slot_idx = slot_entry["index"]
            frames_only = slot_entry["frames"]
            combined_boxes = coverage_map.pop(slot_idx, [])
            combined_whitelist_boxes = whitelist_map.pop(slot_idx, [])
            if combined_boxes and combined_whitelist_boxes:
                combined_boxes = filter_opencv_boxes(
                    combined_boxes, combined_whitelist_boxes, coverage_threshold=0.9
                )

            for frame in frames_only:
                output_frame = frame.copy()
                if combined_boxes:
                    draw_boxes(combined_boxes, output_frame, (0, 0, 0), False)
                if combined_whitelist_boxes:
                    restore_boxes(combined_whitelist_boxes, frame, output_frame)
                writer.write(output_frame)

    def finalize_slot(sample_time: float):
        nonlocal slot_frames
        nonlocal slot_boxes
        nonlocal slot_whitelist_boxes
        nonlocal slot_reference_timestamp
        nonlocal highest_finalized_slot
        nonlocal slot_queue
        if not slot_frames:
            return

        boxes = slot_boxes
        whitelist_boxes = slot_whitelist_boxes
        reference_time = slot_reference_timestamp

        if boxes is None or whitelist_boxes is None:
            # Fall back to the frame closest to the intended midpoint.
            sample_frame, reference_time = min(
                slot_frames, key=lambda item: abs(item[1] - sample_time)
            )
            boxes = detect_opencv_boxes(sample_frame)
            whitelist_boxes = detect_whitelist_boxes(sample_frame)
            slot_reference_timestamp = reference_time
            if args.verbose:
                print(
                    f"Keyframe (fallback) at {reference_time:.2f}s: "
                    f"OpenCV {len(boxes)} box(es), whitelist {len(whitelist_boxes)} line(s)"
                )

        slot_queue.append(
            {"index": slot_index, "frames": [frame for frame, _ in slot_frames]}
        )

        if boxes:
            for offset in range(-args.extra_keyframes, args.extra_keyframes + 1):
                target_slot = slot_index + offset
                if target_slot < 0:
                    continue
                coverage_map[target_slot].extend(boxes)
        if whitelist_boxes:
            for offset in range(-args.extra_keyframes, args.extra_keyframes + 1):
                target_slot = slot_index + offset
                if target_slot < 0:
                    continue
                whitelist_map[target_slot].extend(whitelist_boxes)

        frame_count = len(slot_frames)
        detection_count = len(boxes) if boxes else 0
        whitelist_count = len(whitelist_boxes) if whitelist_boxes else 0
        reference_desc = (
            f"{reference_time:.2f}s" if reference_time is not None else "unknown time"
        )
        print(
            f"Processed slot {slot_index} ({frame_count} frames) "
            f"using reference at {reference_desc} with {detection_count} boxes, "
            f"{whitelist_count} whitelisted line(s)"
        )

        highest_finalized_slot = slot_index
        flush_ready_slots()

        slot_frames = []
        slot_boxes = None
        slot_whitelist_boxes = None
        slot_reference_timestamp = None

    frame_duration = 1.0 / fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx * frame_duration
            
            # Stop early if --only_first_seconds is set
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
                    print(f"Processed {processed_frames} frames…")
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

            if slot_boxes is None and timestamp >= slot_sample_time:
                slot_boxes = detect_opencv_boxes(frame)
                slot_whitelist_boxes = detect_whitelist_boxes(frame)
                slot_reference_timestamp = timestamp
                if args.verbose:
                    print(
                        f"Keyframe at {timestamp:.2f}s: OpenCV {len(slot_boxes)} box(es), "
                        f"whitelist {len(slot_whitelist_boxes)} line(s)"
                    )

        # Flush the final slot and any buffered slots still awaiting future detections
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
