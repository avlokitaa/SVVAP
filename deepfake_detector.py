"""
Deepfake Detector — Dynamic Pseudo-Fake Generation Strategy
============================================================
Role    : Hackathon audio-visual sync detector
Strategy: Train on 50-100 real videos; generate pseudo-fakes on-the-fly
          so the model learns physics of speech, not just memorized frames.

Dependencies
------------
    pip install librosa opencv-python-headless numpy scikit-learn torch

Usage
-----
    # Train:
    from deepfake_detector import train_on_seed_data
    train_on_seed_data("path/to/real_videos/")

    # Demo a single file:
    from deepfake_detector import analyze_single_file
    score = analyze_single_file("path/to/test_video.mp4")
    print(f"Sync score: {score:.3f}  ({'REAL' if score > 0.5 else 'FAKE'})")
"""

import sys

from avsync.audio import analyze_single_file, train_on_seed_data


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "train":
        train_on_seed_data(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "analyze":
        score = analyze_single_file(sys.argv[2])
        sys.exit(0 if score > 0.5 else 1)
    else:
        print(__doc__)
        print("CLI usage:")
        print("  python deepfake_detector.py train  <video_folder>")
        print("  python deepfake_detector.py analyze <video_file>")


if __name__ == "__main__":
    main()
