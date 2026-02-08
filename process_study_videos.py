#!/usr/bin/env python3
"""Finalize study deidentification by renaming ready videos and deidentifying subtitles.

Expected workflow:
- Deidentified video files already exist in `DEID_VIDEOS_READY_FOR_RENAME_DIR`.
  Filenames should match the source PHI video stem plus `_deid`, e.g.
  `AA_20250101_1_deid.mp4`.
- Original `.srt` subtitle files are read from `PHI_VIDEOS_DIR` and are expected
  to share the same stem as the source `.mp4` videos.

What this script does:
- Iterates source PHI videos in `PHI_VIDEOS_DIR`.
- Maps each source video to an anonymized output basename (from `patients.csv`).
- Copies/renames matching deidentified videos from
  `DEID_VIDEOS_READY_FOR_RENAME_DIR` into `DEID_VIDEOS_DIR`.
- Runs subtitle deidentification and writes anonymized `.srt` files into
  `DEID_VIDEOS_DIR`.

Subtitle deidentification only runs when the corresponding final deidentified
video is ready (already present in `DEID_VIDEOS_DIR` or copied there by this
script during the current run).
"""

import argparse
import os
import subprocess
import shutil

import pandas as pd
from termcolor import colored

PHI_VIDEOS_DIR = "/Users/michael/Box Sync/Michael Gensheimer's Files/research/lesion ident segment/data/recordings/ID Recordings/ID Recordings (PHI)"
DEID_VIDEOS_READY_FOR_RENAME_DIR = "/Users/michael/Box Sync/Michael Gensheimer's Files/research/lesion ident segment/data/recordings/ID Recordings/deid_ready_for_rename"
DEID_VIDEOS_DIR = "/Users/michael/Box Sync/Michael Gensheimer's Files/research/lesion ident segment/data/recordings/deid_recordings"
PATIENTS_CSV = "/Users/michael/Box Sync/Michael Gensheimer's Files/research/lesion ident segment/data/recordings/patients.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process study videos to de-identify them by removing PHI text."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: no limit).",
    )
    return parser.parse_args()


def build_subtitle_command(input_filepath, output_filepath):
    """Build the command to de-identify subtitles."""
    return [
        "uv", "run", "deidentify_subtitles.py",
        "-i", input_filepath,
        "-o", output_filepath,
        "--use-gemini",
        "--google-project", "som-nero-phi-mgens-starr",
    ]


def find_ready_video(ready_dir, source_filename):
    """Find a ready _deid video that matches the source filename stem."""
    source_stem = os.path.splitext(source_filename)[0]
    preferred = [
        os.path.join(ready_dir, f"{source_stem}_deid.mp4"),
        os.path.join(ready_dir, f"{source_stem}_deid.mov"),
    ]
    for candidate in preferred:
        if os.path.exists(candidate):
            return candidate

    prefix = f"{source_stem.lower()}_deid."
    for name in sorted(os.listdir(ready_dir)):
        lower = name.lower()
        if not (lower.endswith(".mp4") or lower.endswith(".mov")):
            continue
        if lower.startswith(prefix):
            return os.path.join(ready_dir, name)
    return None


def main():
    args = parse_args()
    
    try:
        patients = pd.read_csv(PATIENTS_CSV)
    except Exception as exc:
        print(colored(
            f"Error: Failed to read patients CSV at {PATIENTS_CSV}: {exc}",
            "yellow",
        ))
        return
    
    # video_name is MP4 filename without the .mp4. video_name has the format:
    # [attending initials]_[mrn]_[anon_patient_id][optional .x where x is the video number]
    # So, for example, MG_123456_1.2.mp4 for the 2nd video for anonymized patient 1.
    # last_name and first_name are the patient's last and first name
    # anon_pt_id is the anonymized patient ID

    if not os.path.isdir(DEID_VIDEOS_READY_FOR_RENAME_DIR):
        print(colored(
            "Error: ready-for-rename directory does not exist: "
            f"{DEID_VIDEOS_READY_FOR_RENAME_DIR}",
            "yellow",
        ))
        return

    # Create output directory if it doesn't exist
    if not args.dry_run:
        os.makedirs(DEID_VIDEOS_DIR, exist_ok=True)

    # Build a lookup dictionary from video_name to patient info
    patient_lookup = {}
    for _, row in patients.iterrows():
        patient_lookup[row['video_name']] = {
            'first_name': row['first_name'].lower(),
            'last_name': row['last_name'].lower(),
            'anon_pt_id': row['anon_pt_id']
        }

    videos_processed = 0

    # Process each MP4 file in PHI_VIDEOS_DIR
    for filename in sorted(os.listdir(PHI_VIDEOS_DIR)):
        if not filename.lower().endswith('.mp4'):
            continue

        srt_filename = f"{filename[:-4]}.srt"
        srt_input_filepath = os.path.join(PHI_VIDEOS_DIR, srt_filename)
        if not os.path.exists(srt_input_filepath):
            print(colored(
                f"Skipping {filename}: missing subtitle file {srt_filename}",
                "yellow",
            ))
            continue
        
        # Check if we've reached the max_videos limit
        if args.max_videos is not None and videos_processed >= args.max_videos:
            print(f"\nReached max_videos limit ({args.max_videos}), stopping.")
            break
        
        video_name = filename[:-4]  # Remove .mp4 extension
        
        if video_name not in patient_lookup:
            print(colored(
                f"Warning: No patient info found for {filename}, skipping...",
                "yellow",
            ))
            continue
        
        patient_info = patient_lookup[video_name]
        
        # Extract optional video number suffix from video_name
        # Format: [initials]_[mrn]_[anon_pt_id][.x] where .x is optional
        # We need to get the anon_pt_id part with its optional suffix
        parts = video_name.split('_')
        anon_part = parts[2] if len(parts) >= 3 else str(patient_info['anon_pt_id'])
        
        # The anon_part may be like "1" or "1.2" - use it directly for output filename
        output_filepath = os.path.join(DEID_VIDEOS_DIR, f"{anon_part}.mp4")
        output_srt_filepath = os.path.join(DEID_VIDEOS_DIR, f"{anon_part}.srt")
        
        subtitle_cmd = build_subtitle_command(srt_input_filepath, output_srt_filepath)
        output_video_exists = os.path.exists(output_filepath)
        ready_video_path = None
        if not output_video_exists and os.path.isdir(DEID_VIDEOS_READY_FOR_RENAME_DIR):
            ready_video_path = find_ready_video(DEID_VIDEOS_READY_FOR_RENAME_DIR, filename)
        
        if args.dry_run:
            print(f"\n[DRY RUN] Would process: {filename}")
            if output_video_exists:
                print(f"  Video ready: final output already exists at {output_filepath}")
            elif ready_video_path:
                print(f"  Would copy video: {ready_video_path} -> {output_filepath}")
            else:
                print("  Skipping video copy: no matching _deid file in ready-for-rename directory")
            if os.path.exists(output_srt_filepath):
                print(f"  Skipping subtitles: output already exists at {output_srt_filepath}")
            elif output_video_exists or ready_video_path:
                print(f"  Subtitle command: {' '.join(subtitle_cmd)}")
            else:
                print("  Skipping subtitles: deid video is not ready")
            videos_processed += 1
        else:
            print(f"\nProcessing: {filename}")
            print(f"  Output: {output_filepath}")
            video_ready_for_subtitles = output_video_exists

            if output_video_exists:
                print(f"  Video ready: final output already exists at {output_filepath}")
            elif ready_video_path:
                print(f"  Copying video: {ready_video_path} -> {output_filepath}")
                try:
                    shutil.copy2(ready_video_path, output_filepath)
                    print(f"  Success: copied deid video to {output_filepath}")
                    video_ready_for_subtitles = True
                except Exception as exc:
                    print(f"  Error: failed to copy video: {exc}")
            else:
                print("  Skipping video copy: no matching _deid file in ready-for-rename directory")

            if os.path.exists(output_srt_filepath):
                print(f"  Skipping subtitles: output already exists at {output_srt_filepath}")
            elif video_ready_for_subtitles:
                print(f"  Running subtitles: {' '.join(subtitle_cmd)}")
                subtitle_result = subprocess.run(
                    subtitle_cmd, cwd=os.path.dirname(os.path.abspath(__file__))
                )
                if subtitle_result.returncode != 0:
                    print(
                        "  Error: Subtitle command failed with return code "
                        f"{subtitle_result.returncode}"
                    )
                else:
                    print(f"  Success: De-identified subtitles saved to {output_srt_filepath}")
            else:
                print("  Skipping subtitles: deid video is not ready")
            
            videos_processed += 1

    print(f"\nDone. Processed {videos_processed} videos.")


if __name__ == "__main__":
    main()
