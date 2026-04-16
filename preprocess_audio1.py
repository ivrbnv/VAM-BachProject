"""
preprocess_audio.py  (v2 - Python 3.13 compatible)
-------------------
Preprocessing script for opensauce-python.

Pipeline: WAV -> MP3 (lossy compression) -> WAV (16-bit PCM, mono)
The output WAVs are ready to be fed into opensauce-python.

Requirements:
    pip install soundfile numpy
    ffmpeg must be installed and on your PATH.
    Download ffmpeg for Windows: https://www.gyan.dev/ffmpeg/builds/

Usage:
    python preprocess_audio.py --input_dir path\to\wavs --output_dir path\to\output
    python preprocess_audio.py --input_dir path\to\wavs  # outputs to 'preprocessed' subfolder

Optional flags:
    --mp3_bitrate   MP3 bitrate in kbps (default: 128)
    --keep_mp3      Keep intermediate MP3 files (default: deleted after conversion)
    --mp3_dir       Custom folder to store intermediate MP3s
"""

import argparse
import os
import shutil
import subprocess
import sys


def find_ffmpeg():
    """Check that ffmpeg is available on PATH."""
    path = shutil.which("ffmpeg")
    if path is None:
        sys.exit(
            "ERROR: ffmpeg was not found on your PATH.\n"
            "Download it from https://www.gyan.dev/ffmpeg/builds/\n"
            "and add the 'bin' folder to your Windows PATH environment variable."
        )
    return path


def wav_to_mp3(ffmpeg_path, wav_path, mp3_path, bitrate="128k"):
    """Convert a WAV file to MP3 using ffmpeg."""
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg WAV->MP3 failed:\n{result.stderr}")


def mp3_to_wav(ffmpeg_path, mp3_path, wav_path):
    """
    Convert an MP3 back to WAV (16-bit PCM mono) using ffmpeg.
    opensauce-python requires 16-bit integer PCM format.
    """
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", mp3_path,
        "-ac", "1",
        "-sample_fmt", "s16",
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg MP3->WAV failed:\n{result.stderr}")


def preprocess_directory(ffmpeg_path, input_dir, output_dir, mp3_dir, mp3_bitrate="128k", keep_mp3=False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mp3_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No WAV files found in '{input_dir}'.")
        return

    print(f"Found {len(wav_files)} WAV file(s) in '{input_dir}'.\n")

    succeeded = 0
    failed = 0

    for i, filename in enumerate(wav_files, start=1):
        base = os.path.splitext(filename)[0]
        input_wav = os.path.join(input_dir, filename)
        intermediate_mp3 = os.path.join(mp3_dir, base + ".mp3")
        output_wav = os.path.join(output_dir, filename)

        print(f"[{i}/{len(wav_files)}] Processing: {filename}")

        try:
            print(f"  WAV -> MP3  ({mp3_bitrate}): {intermediate_mp3}")
            wav_to_mp3(ffmpeg_path, input_wav, intermediate_mp3, bitrate=mp3_bitrate)

            print(f"  MP3 -> WAV  (16-bit PCM mono): {output_wav}")
            mp3_to_wav(ffmpeg_path, intermediate_mp3, output_wav)

            if not keep_mp3:
                os.remove(intermediate_mp3)

            print(f"  Done.\n")
            succeeded += 1

        except Exception as e:
            print(f"  ERROR: {e}\n")
            failed += 1

    print(f"Preprocessing complete: {succeeded} succeeded, {failed} failed.")
    print(f"Output WAVs saved to: {os.path.abspath(output_dir)}")
    if keep_mp3:
        print(f"Intermediate MP3s saved to: {os.path.abspath(mp3_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess WAV files for opensauce-python via WAV -> MP3 -> WAV."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing your original WAV files.")
    parser.add_argument("--output_dir", default=None, help="Folder to save preprocessed WAV files.")
    parser.add_argument("--mp3_bitrate", default="128", help="MP3 bitrate in kbps (default: 128).")
    parser.add_argument("--keep_mp3", action="store_true", help="Keep intermediate MP3 files.")
    parser.add_argument("--mp3_dir", default=None, help="Folder to store intermediate MP3 files.")

    args = parser.parse_args()

    ffmpeg_path = find_ffmpeg()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        sys.exit(f"ERROR: input_dir '{input_dir}' does not exist.")

    output_dir = args.output_dir or os.path.join(input_dir, "preprocessed")
    mp3_dir = args.mp3_dir or os.path.join(input_dir, "mp3_temp")
    mp3_bitrate = args.mp3_bitrate.rstrip("k") + "k"

    preprocess_directory(
        ffmpeg_path=ffmpeg_path,
        input_dir=input_dir,
        output_dir=output_dir,
        mp3_dir=mp3_dir,
        mp3_bitrate=mp3_bitrate,
        keep_mp3=args.keep_mp3,
    )


if __name__ == "__main__":
    main()
