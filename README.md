# Remove protected health information (PHI) from screen capture videos

Author: Michael Gensheimer (michael.gensheimer@gmail.com) and Codex CLI.

Text block removal from images was adapted from [this OpenCV tutorial](https://opencv.org/blog/text-detection-and-removal-using-opencv/).

## About

Use this repository to remove protected health information (PHI) such as patient name, ID, dates, etc. from screen capture videos. Text in the video is replaced by black boxes, and audio is replaced with a subtitles file that contains the transcript with any PHI removed.

Sample output:

![Sample output](sample_output.jpg)

Note that large text blocks have been redacted but some small text snippets are still visible. _The process is not 100% reliable so continue to assume there is PHI in the output!_

## Prerequisites (tested on Mac but should work on other platforms)

- uv package manager
- ffmpeg
- Tesseract OCR (required for selective redaction features; install with `brew install tesseract` on Mac)
- Ollama with gemma3 model installed (4b version is smart enough and runs smoothly on Mac)
- A transcription tool that can make a .srt subtitles file. Mac Whisper Pro with Parakeet v3 model is good, it is able to separate the different speakers. Or can try FFTrans Parakeet which is free, but I found the quality to be lower.

## Steps

1. Make subtitles file from the video file

Use a transcription tool to make a .srt subtitles file (see Prerequisites).

2. Deidentify the subtitles file (removes patient names, IDs, dates, etc.)

`uv run deidentify_subtitles.py -i subtitles.srt -o subtitles_deid.srt`

If you prefer to call Google Vertex AI Gemini instead of a local Ollama model, supply your Vertex AI project and add `--use-gemini`:

`uv run deidentify_subtitles.py -i subtitles.srt -o subtitles_deid.srt --use-gemini --google-project YOUR_PROJECT`

The tool uses the `gemini-2.5-flash-lite` model by default; adjust `--google-location` or `--google-model` if your deployment needs different settings.

3. Remove audio and on-screen text from the video

`uv run remove_text_from_video.py -i recording.mp4 -o recording_no_text.mp4 --interval 2 --extra-keyframes 1 --target-bitrate 1500k`

This will find any letters/numbers in the video and change them to black boxes. I recommend keeping interval and extra-keyframes this way but you can try different values.

### Selective redaction with OCR

Instead of redacting all text, you can use OCR to selectively redact only specific content. This is useful when you want to preserve some on-screen text while removing PHI.

**Redact specific phrases** (case-insensitive, ignores spaces/punctuation):

`uv run remove_text_from_video.py -i recording.mp4 -o output.mp4 -p "John Smith" -p "Patient Name"`

**Redact dates and times** (various formats like YYYY-MM-DD, MM/DD/YYYY, 11:23 AM, etc.):

`uv run remove_text_from_video.py -i recording.mp4 -o output.mp4 --redact_dates_times`

**Redact long numbers** (e.g., patient IDs, phone numbers with 7+ digits):

`uv run remove_text_from_video.py -i recording.mp4 -o output.mp4 --redact_digits 7`

**Combine multiple filters:**

`uv run remove_text_from_video.py -i recording.mp4 -o output.mp4 --redact_dates_times --redact_digits 5 -p "Confidential"`

Use `-v` (verbose) to see what text is being detected and redacted on each keyframe—helpful for debugging.

### Testing with a short clip

Use `--only_first_seconds` to process only the beginning of a video, which is useful for testing settings:

`uv run remove_text_from_video.py -i recording.mp4 -o test_output.mp4 --redact_dates_times -v --only_first_seconds 10`

## Result

The result is a video file with the audio and on-screen text removed, and a subtitle file with the transcription. Name the .srt file with the same name as the video (as in, video.mp4 and video.srt). You can test out the finished product by loading the video in VLC and turning subtitles on.
