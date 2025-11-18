"""Deidentify SRT subtitle files by redacting sensitive text via Ollama."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


DEFAULT_PROMPT_TEMPLATE = (
    "Please return the following text, redacting any patient identifiers (names, IDs, dates, etc.) with [redacted identifier]. "
    "For numbers, only redact patient IDs and dates. You do not need to redact other numbers such as radiation dose levels. "
    "Return ONLY the deidentified text, no other text or explanations. Here is the text to clean: '{text}'"
)

TIME_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass
class SRTEntry:
    index: int
    start: str
    end: str
    text_lines: List[str]

    def text_block(self) -> str:
        return "\n".join(self.text_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove patient identifiers from an SRT file by calling Ollama."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the input .srt file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="Path for the cleaned .srt file (default: <input>_cleaned.srt)",
    )
    parser.add_argument(
        "--model",
        default="gemma3",
        help="Ollama model name to run (default: gemma3)",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help=(
            "Python format string used for each entry. "
            "Must include '{text}' placeholder for the subtitle text."
        ),
    )
    parser.add_argument(
        "--ollama-bin",
        default="ollama",
        help="Binary to invoke for Ollama (default: ollama)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout (seconds) applied to each Ollama call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip calling Ollama and simply copy each entry's text (useful for tests).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-entry progress messages.",
    )
    return parser.parse_args()


def parse_srt_entries(content: str) -> List[SRTEntry]:
    entries: List[SRTEntry] = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        try:
            index = int(line)
        except ValueError as exc:
            raise ValueError(f"Expected numeric index on line {i + 1}: {lines[i]!r}") from exc
        i += 1

        if i >= len(lines):
            raise ValueError("Unexpected end of file while reading timecode line.")

        time_line = lines[i].strip()
        match = TIME_PATTERN.fullmatch(time_line)
        if not match:
            raise ValueError(f"Invalid timecode on line {i + 1}: {time_line!r}")
        start = match.group("start")
        end = match.group("end")
        i += 1

        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        entries.append(SRTEntry(index=index, start=start, end=end, text_lines=text_lines))

    return entries


def serialize_entries(entries: Sequence[SRTEntry]) -> str:
    blocks = []
    for entry in entries:
        block_lines = [str(entry.index), f"{entry.start} --> {entry.end}"]
        block_lines.extend(entry.text_lines or [""])
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks) + "\n"


def run_ollama_prompt(
    *,
    text: str,
    model: str,
    prompt_template: str,
    ollama_bin: str,
    timeout: float | None,
    dry_run: bool,
) -> str:
    if dry_run or not text.strip():
        return text

    prompt = prompt_template.format(text=text)
    cmd = [ollama_bin, "run", model, prompt]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Unable to find Ollama binary '{ollama_bin}'. Is it installed?"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Ollama call timed out after {timeout} seconds for prompt starting with "
            f"{prompt[:60]!r}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Ollama exited with status {exc.returncode}. stderr: {exc.stderr.strip()}"
        ) from exc

    cleaned = result.stdout.strip()
    return cleaned if cleaned else text


def determine_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser()
    stem = input_path.stem + "_cleaned"
    return input_path.with_name(stem + input_path.suffix)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = determine_output_path(input_path, args.output_path)

    try:
        content = input_path.read_text(encoding="utf-8")
        content = content.lstrip("\ufeff")
    except UnicodeDecodeError:
        content = input_path.read_text(encoding="utf-8-sig")

    entries = parse_srt_entries(content)
    cleaned_entries: List[SRTEntry] = []

    for entry in entries:
        original_text = entry.text_block()
        if not args.quiet:
            print(f"Deidentifying entry {entry.index} ({len(original_text)} chars)...")

        cleaned_text = run_ollama_prompt(
            text=original_text,
            model=args.model,
            prompt_template=args.prompt_template,
            ollama_bin=args.ollama_bin,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        new_lines = cleaned_text.splitlines()
        entry.text_lines = new_lines if new_lines else [""]
        cleaned_entries.append(entry)

    output_path.write_text(serialize_entries(cleaned_entries), encoding="utf-8")

    if not args.quiet:
        print(f"Wrote cleaned subtitles to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surface readable message to CLI
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
