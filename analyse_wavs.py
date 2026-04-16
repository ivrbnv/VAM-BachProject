#!/usr/bin/env python3
"""
Batch voice analyser — openSMILE eGeMAPSv02
============================================
Extracts the 88-feature eGeMAPSv02 feature set from all WAV files
in a folder and writes results to a single CSV.

Install dependency:
    pip install opensmile

Usage:
    python analyse_wavs.py --wav-dir C:\path\to\wavs --output results.csv

    # Analyse AI and human folders separately, then compare
    python analyse_wavs.py --wav-dir C:\wavs\ai    --output ai_results.csv
    python analyse_wavs.py --wav-dir C:\wavs\human --output human_results.csv
"""

import argparse
import glob
import os
import sys

try:
    import opensmile
except ImportError:
    sys.exit("ERROR: opensmile not found. Run: pip install opensmile")

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas not found. Run: pip install pandas")


# ── Core ──────────────────────────────────────────────────────────────────────

def build_extractor():
    """Initialise the openSMILE eGeMAPSv02 extractor (file-level summary)."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,  # one row per file
    )


def find_wav_files(wav_dir):
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not files:
        files = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    return files


def run(wav_dir, output_path):
    wav_files = find_wav_files(wav_dir)
    if not wav_files:
        sys.exit(f"ERROR: No .wav files found in {wav_dir}")

    print(f"Found {len(wav_files)} WAV file(s) in '{wav_dir}'")
    print("Initialising openSMILE eGeMAPSv02...\n")

    smile = build_extractor()
    rows = []
    failed = []

    for i, wav_path in enumerate(wav_files, 1):
        name = os.path.basename(wav_path)
        print(f"[{i}/{len(wav_files)}] {name}")
        try:
            features = smile.process_file(wav_path)
            # process_file returns a DataFrame with one row and a MultiIndex;
            # reset_index drops the file/start/end index into plain columns
            features = features.reset_index(drop=True)
            features.insert(0, "filename", name)
            rows.append(features)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(wav_path)

    if not rows:
        sys.exit("No results collected — check errors above.")

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"  Done! {len(rows)} file(s) analysed.")
    if failed:
        print(f"  {len(failed)} failed: {', '.join(os.path.basename(f) for f in failed)}")
    print(f"  Output  : {output_path}")
    print(f"  Features: {result.shape[1] - 1} eGeMAPSv02 features + filename")
    print(f"{'='*50}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch voice analyser using openSMILE eGeMAPSv02"
    )
    parser.add_argument("--wav-dir", "-d", required=True, help="Folder containing WAV files")
    parser.add_argument("--output",  "-o", default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    if not os.path.isdir(args.wav_dir):
        sys.exit(f"ERROR: Directory not found: {args.wav_dir}")

    run(args.wav_dir, args.output)


if __name__ == "__main__":
    main()
