"""Robust mono audio loading for the embedded backing tracks.

`librosa.load` goes through soundfile, and that path has failed twice on real
mp3s:

  * soundfile **silently under-decodes** certain mono VBR mp3s -- a 253s
    recording decoding as 154s with no exception raised.
  * "Pennywise - Fuck Authority"'s embedded track raises a hard
    `LibsndfileError: Unspecified internal error` instead.

A silent 40% truncation is much worse than a crash here: the alignment would
still produce a confident, monotonic, entirely wrong map. So decoding goes
through ffmpeg when it is on PATH (else librosa), and the result is
length-checked against ffprobe's reported duration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class AudioLoadError(RuntimeError):
    pass


def probe_duration(path: Path | str) -> float | None:
    """Container-reported duration in seconds, or None if ffprobe is absent."""
    if not FFPROBE:
        return None
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            # ffprobe's JSON is UTF-8; text=True alone decodes as the Windows
            # code page and crashes on non-ASCII mp3 tags (3 of 7 Sabaton tabs).
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, check=True).stdout
        return float(json.loads(out)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        return None


def load_mono(path: Path | str, sr: int, tolerance: float = 0.5) -> np.ndarray:
    """Decode `path` to mono float32 at `sr`, preferring ffmpeg.

    Raises AudioLoadError if the decode comes back more than `tolerance`
    seconds shorter than the container claims -- the silent-truncation failure
    mode, which must never be allowed to reach the aligner.
    """
    path = Path(path)
    expected = probe_duration(path)
    y = _load_ffmpeg(path, sr) if FFMPEG else _load_librosa(path, sr)

    if y is None or len(y) == 0:
        raise AudioLoadError(f"decoded no samples from {path.name}")

    got = len(y) / sr
    if expected and got < expected - tolerance:
        raise AudioLoadError(
            f"{path.name}: decoded {got:.1f}s but container reports "
            f"{expected:.1f}s -- refusing a truncated decode")
    return y


def _load_ffmpeg(path: Path, sr: int) -> np.ndarray | None:
    proc = subprocess.run(
        [FFMPEG, "-v", "quiet", "-i", str(path), "-vn",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return _load_librosa(path, sr)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_librosa(path: Path, sr: int) -> np.ndarray | None:
    import librosa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception as e:
        raise AudioLoadError(f"{path.name}: {type(e).__name__}: {e}")
