#!/usr/bin/env python3
"""Process study videos to de-identify them by removing PHI text."""

import argparse
import os
import subprocess

import pandas as pd
from termcolor import colored

PHI_VIDEOS_DIR = "/Users/michael/Box Sync/Michael Gensheimer's Files/research/lesion ident segment/data/recordings/ID Recordings/ID Recordings (PHI)"
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


def build_command(input_filepath, output_filepath, first_name, last_name):
    """Build the command to de-identify a video."""
    return [
        "uv", "run", "remove_text_from_video.py",
        "-i", input_filepath,
        "-o", output_filepath,
        "--interval", "4",
        "--extra-keyframes", "1",
        "--target-bitrate", "1500k",
        "-p", first_name,
        "-p", last_name,
        "--redact_dates_times",
        "--redact_digits", "6",
        "--verbose",
        #"--only_first_seconds", "20"
    ]


def build_subtitle_command(input_filepath, output_filepath):
    """Build the command to de-identify subtitles."""
    return [
        "uv", "run", "deidentify_subtitles.py",
        "-i", input_filepath,
        "-o", output_filepath,
        "--use-gemini",
        "--google-project", "som-nero-phi-mgens-starr",
    ]


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
        first_name = patient_info['first_name']
        last_name = patient_info['last_name']
        
        input_filepath = os.path.join(PHI_VIDEOS_DIR, filename)
        
        # Extract optional video number suffix from video_name
        # Format: [initials]_[mrn]_[anon_pt_id][.x] where .x is optional
        # We need to get the anon_pt_id part with its optional suffix
        parts = video_name.split('_')
        anon_part = parts[2] if len(parts) >= 3 else str(patient_info['anon_pt_id'])
        
        # The anon_part may be like "1" or "1.2" - use it directly for output filename
        output_filepath = os.path.join(DEID_VIDEOS_DIR, f"{anon_part}.mp4")
        output_srt_filepath = os.path.join(DEID_VIDEOS_DIR, f"{anon_part}.srt")
        
        cmd = build_command(input_filepath, output_filepath, first_name, last_name)
        subtitle_cmd = build_subtitle_command(srt_input_filepath, output_srt_filepath)
        
        if args.dry_run:
            print(f"\n[DRY RUN] Would process: {filename}")
            if os.path.exists(output_filepath):
                print(f"  Skipping video: output already exists at {output_filepath}")
            else:
                print(f"  Video command: {' '.join(cmd)}")
            if os.path.exists(output_srt_filepath):
                print(f"  Skipping subtitles: output already exists at {output_srt_filepath}")
            else:
                print(f"  Subtitle command: {' '.join(subtitle_cmd)}")
            videos_processed += 1
        else:
            print(f"\nProcessing: {filename}")
            print(f"  Patient: {first_name} {last_name}")
            print(f"  Output: {output_filepath}")
            if os.path.exists(output_filepath):
                print(f"  Skipping video: output already exists at {output_filepath}")
            else:
                print(f"  Running video: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                if result.returncode != 0:
                    print(f"  Error: Video command failed with return code {result.returncode}")
                else:
                    print(f"  Success: De-identified video saved to {output_filepath}")

            if os.path.exists(output_srt_filepath):
                print(f"  Skipping subtitles: output already exists at {output_srt_filepath}")
            else:
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
            
            videos_processed += 1

    print(f"\nDone. Processed {videos_processed} videos.")


if __name__ == "__main__":
    main()
