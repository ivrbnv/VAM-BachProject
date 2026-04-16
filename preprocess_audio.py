"""
preprocess_audio.py
-------------------
Preprocessing script for opensauce-python.

Pipeline: WAV -> MP3 (lossy compression) -> WAV (16-bit PCM, mono)
The output WAVs are ready to be fed into opensauce-python.

Requirements:
    pip install pydub
    ffmpeg must be installed and on your PATH.
    Download ffmpeg for Windows: https://ffmpeg.org/download.html
    (Recommended: use the "ffmpeg-release-essentials" build from gyan.dev)

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
import sys

# --- Dependency check ---
try:
    from pydub import AudioSegment
except ImportError:
    sys.exit(
        "ERROR: pydub is not installed.\n"
        "Run:  pip install pydub\n"
        "Also make sure ffmpeg is installed and on your PATH."
    )


def find_ffmpeg():
    """Warn the user if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "WARNING: ffmpeg was not found on your PATH.\n"
            "Download it from https://ffmpeg.org/download.html\n"
            "and add the 'bin' folder to your Windows PATH environment variable.\n"
        )


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "128k") -> None:
    """Convert a WAV file to MP3 at the given bitrate."""
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)


def mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    """
    Convert an MP3 file back to WAV.
    Output is forced to 16-bit PCM mono, as required by opensauce-python.
    """
    audio = AudioSegment.from_mp3(mp3_path)
    # opensauce-python requires 16-bit integer PCM
    audio = audio.set_sample_width(2)  # 2 bytes = 16-bit
    audio = audio.set_channels(1)      # mono
    audio.export(wav_path, format="wav")


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    mp3_dir: str,
    mp3_bitrate: str = "128k",
    keep_mp3: bool = False,
) -> None:
    """
    Find all WAV files in input_dir, run them through the
    WAV -> MP3 -> WAV pipeline, and save results to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mp3_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No WAV files found in '{input_dir}'.")
        return

    print(f"Found {len(wav_files)} WAV file(s) in '{input_dir}'.\n")

    for i, filename in enumerate(wav_files, start=1):
        base = os.path.splitext(filename)[0]
        input_wav = os.path.join(input_dir, filename)
        intermediate_mp3 = os.path.join(mp3_dir, base + ".mp3")
        output_wav = os.path.join(output_dir, filename)

        print(f"[{i}/{len(wav_files)}] Processing: {filename}")

        try:
            # Step 1: WAV -> MP3
            print(f"  WAV  -> MP3  ({mp3_bitrate}): {intermediate_mp3}")
            wav_to_mp3(input_wav, intermediate_mp3, bitrate=mp3_bitrate)

            # Step 2: MP3 -> WAV (16-bit PCM mono)
            print(f"  MP3  -> WAV  (16-bit PCM mono): {output_wav}")
            mp3_to_wav(intermediate_mp3, output_wav)

            # Optionally clean up the MP3
            if not keep_mp3:
                os.remove(intermediate_mp3)

            print(f"  Done.\n")

        except Exception as e:
            print(f"  ERROR processing '{filename}': {e}\n")

    print(f"Preprocessing complete.")
    print(f"Output WAVs are in: {os.path.abspath(output_dir)}")
    if keep_mp3:
        print(f"Intermediate MP3s are in: {os.path.abspath(mp3_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess WAV files for opensauce-python via WAV -> MP3 -> WAV."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing your original WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Folder to save preprocessed WAV files (default: 'preprocessed' inside input_dir).",
    )
    parser.add_argument(
        "--mp3_bitrate",
        default="128",
        help="MP3 bitrate in kbps, e.g. 64, 128, 192, 320 (default: 128).",
    )
    parser.add_argument(
        "--keep_mp3",
        action="store_true",
        help="Keep intermediate MP3 files instead of deleting them.",
    )
    parser.add_argument(
        "--mp3_dir",
        default=None,
        help="Folder to store intermediate MP3 files (default: 'mp3_temp' inside input_dir).",
    )

    args = parser.parse_args()

    find_ffmpeg()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        sys.exit(f"ERROR: input_dir '{input_dir}' does not exist.")

    output_dir = args.output_dir or os.path.join(input_dir, "preprocessed")
    mp3_dir = args.mp3_dir or os.path.join(input_dir, "mp3_temp")

    # Append 'k' to bitrate if not already there
    mp3_bitrate = args.mp3_bitrate.rstrip("k") + "k"

    preprocess_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        mp3_dir=mp3_dir,
        mp3_bitrate=mp3_bitrate,
        keep_mp3=args.keep_mp3,
    )


if __name__ == "__main__":
    main()