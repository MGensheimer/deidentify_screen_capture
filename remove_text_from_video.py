# Detect text on representative video frames and apply results across intervals.
#
# Example usage:
# uv run remove_text_from_video.py -i recording.mp4 -o output/study_recording.mp4 --interval 2 --extra-keyframes 1 --target-bitrate 1500k


from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from text_removal_helper import (
    DETECTOR,
    build_detector,
    detect_text_with_tiling,
    draw_boxes,
)


Color = Tuple[int, int, int]
SlotBufferEntry = Dict[str, Any]


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
    parser = argparse.ArgumentParser(
        description=(
            "Remove or outline text in a video by reusing detections across fixed intervals"
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
        choices=["DB18", "EAST"],
        default=DETECTOR,
        help="Text detector backend to use (default: %(default)s)",
    )
    parser.add_argument(
        "--tile",
        dest="tile_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(320, 320),
        help="Tile size fed to the detector (default: 320 320)",
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
    return parser.parse_args()


def parse_color(color_str: str) -> Color:
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

    video_path = args.input_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Unable to determine video FPS from input stream")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fill_color = parse_color(args.color)
    tile_size = tuple(args.tile_size)
    detector = build_detector(tile_size, detector_name=args.detector_name)

    os.makedirs("output", exist_ok=True)
    default_name = "output_video.mp4"
    output_path = args.output_path or os.path.join("output", default_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    total_frames_value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_value if total_frames_value > 0 else None
    if total_frames:
        total_duration = total_frames / fps
        print(
            f"Processing {total_frames} frames (~{total_duration:.1f}s) from {video_path}"
        )
    else:
        print(f"Processing video {video_path} (unknown frame count)")

    interval = args.interval_seconds
    frame_idx = 0
    slot_index: Optional[int] = None
    slot_frames: List[Tuple[np.ndarray, float]] = []
    slot_boxes: Optional[List[Sequence[Sequence[float]]]] = None
    slot_sample_time: float = interval / 2
    slot_reference_timestamp: Optional[float] = None
    last_percent_bucket = -1
    last_frames_log = 0
    percent_step = 5
    slot_queue: Deque[SlotBufferEntry] = deque()
    coverage_map: DefaultDict[int, List[Sequence[Sequence[float]]]] = defaultdict(list)
    highest_finalized_slot: Optional[int] = None

    def detect_boxes(frame):
        return detect_text_with_tiling(detector, frame, tile_size)

    def flush_ready_slots(force: bool = False):
        nonlocal slot_queue, coverage_map, highest_finalized_slot
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

            for frame in frames_only:
                output_frame = frame.copy()
                if combined_boxes:
                    draw_boxes(combined_boxes, output_frame, fill_color, args.outline)
                writer.write(output_frame)

    def finalize_slot(sample_time: float):
        nonlocal slot_frames, slot_boxes, slot_reference_timestamp, highest_finalized_slot, slot_queue
        if not slot_frames:
            return

        boxes = slot_boxes
        reference_time = slot_reference_timestamp

        if boxes is None:
            # Fall back to the frame closest to the intended midpoint.
            sample_frame, reference_time = min(
                slot_frames, key=lambda item: abs(item[1] - sample_time)
            )
            boxes = detect_boxes(sample_frame)
            slot_reference_timestamp = reference_time

        slot_queue.append(
            {"index": slot_index, "frames": [frame for frame, _ in slot_frames]}
        )

        if boxes:
            for offset in range(-args.extra_keyframes, args.extra_keyframes + 1):
                target_slot = slot_index + offset
                if target_slot < 0:
                    continue
                coverage_map[target_slot].extend(boxes)

        frame_count = len(slot_frames)
        detection_count = len(boxes) if boxes else 0
        reference_desc = (
            f"{reference_time:.2f}s" if reference_time is not None else "unknown time"
        )
        print(
            f"Processed slot {slot_index} ({frame_count} frames) "
            f"using reference at {reference_desc} with {detection_count} boxes"
        )

        highest_finalized_slot = slot_index
        flush_ready_slots()

        slot_frames = []
        slot_boxes = None
        slot_reference_timestamp = None

    frame_duration = 1.0 / fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx * frame_duration
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
                slot_boxes = detect_boxes(frame)
                slot_reference_timestamp = timestamp

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
