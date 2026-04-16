"""
MP4 to WAV Converter for opensauce-python preprocessing
Converts MP4 files to 16-bit PCM WAV (mono, 16kHz) using ffmpeg directly.
Compatible with Python 3.13+. No pydub required.

Requirements:
    ffmpeg must be installed and in PATH.
    Download from: https://ffmpeg.org/download.html

Usage:
    # Convert a single file:
    python mp4_to_wav.py audio.mp4

    # Convert all MP4s in a folder:
    python mp4_to_wav.py "my audio folder/"

    # Specify an output directory:
    python mp4_to_wav.py "my audio folder/" --output converted/

    # Keep original sample rate (don't resample to 16kHz):
    python mp4_to_wav.py audio.mp4 --no-resample
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.")
        print("  Download from: https://ffmpeg.org/download.html")
        print("  Extract it, then add the ffmpeg/bin folder to your Windows PATH.")
        sys.exit(1)


def convert_file(mp4_path: Path, output_dir: Path, resample: bool) -> bool:
    wav_path = output_dir / (mp4_path.stem + ".wav")
    print(f"  {mp4_path.name}  ->  {wav_path.name}")

    cmd = ["ffmpeg", "-y", "-i", str(mp4_path), "-ac", "1", "-acodec", "pcm_s16le"]
    if resample:
        cmd += ["-ar", "16000"]
    cmd.append(str(wav_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.splitlines()[-1] if result.stderr else 'unknown error'}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP4 files to WAV for use with opensauce-python."
    )
    parser.add_argument("input", help="Path to an MP4 file or a folder containing MP4 files.")
    parser.add_argument("--output", "-o", default=None, help="Output directory for WAV files.")
    parser.add_argument("--no-resample", action="store_true", help="Skip resampling to 16kHz.")
    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() != ".mp4":
            print(f"ERROR: '{input_path}' is not an MP4 file.")
            sys.exit(1)
        mp4_files = [input_path]
        default_output = input_path.parent
    elif input_path.is_dir():
        mp4_files = sorted(input_path.rglob("*.mp4")) + sorted(input_path.rglob("*.MP4"))
        if not mp4_files:
            print(f"No MP4 files found in '{input_path}'.")
            sys.exit(0)
        default_output = input_path
    else:
        print(f"ERROR: '{input_path}' does not exist.")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    resample = not args.no_resample
    print(f"\nFound {len(mp4_files)} MP4 file(s).")
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Resample to 16kHz: {'yes' if resample else 'no'}\n")

    success, failed = 0, []
    for i, mp4_path in enumerate(mp4_files, 1):
        print(f"[{i}/{len(mp4_files)}]", end=" ")
        if convert_file(mp4_path, output_dir, resample):
            success += 1
        else:
            failed.append(mp4_path.name)

    print(f"\nDone. {success}/{len(mp4_files)} converted.", end="")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    else:
        print()


if __name__ == "__main__":
    main()
