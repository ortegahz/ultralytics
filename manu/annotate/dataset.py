# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Recursive video/frame discovery plus cached frame decoding for the annotation server.

Two sequence kinds share one interface so the front-end never learns which one it is looking at:

    VideoSequence     an .mp4/.avi/... file, decoded lazily with a JPEG ring buffer
    ImageDirSequence  a directory of frame_XXXXXX.jpg, read straight off disk

Frame access is sequential in practice (an annotator walks one frame at a time), so the decoder keeps
a single open capture positioned near the last request, decodes forward without seeking, and keeps the
last ``cache_limit`` frames as JPEG bytes. Stepping backwards is a cache hit; a large jump forces one
``CAP_PROP_POS_FRAMES`` seek.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".webm")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Frames shorter than this are too small for the annotator to click; a directory with only a handful of
# such files is a crop/export artefact rather than a sequence.
MIN_SEQUENCE_FRAMES = 2
# A seek is only worth it when the gap exceeds what sequential decode can absorb cheaply.
SEEK_THRESHOLD = 240
JPEG_QUALITY = 90


@dataclass(frozen=True)
class SequenceInfo:
    """Immutable description handed to the browser; mirrors ``VideoSequence``/``ImageDirSequence``."""

    seq_id: str
    name: str
    kind: str
    source: str
    frame_count: int
    width: int
    height: int
    fps: float
    group: str

    def to_dict(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "group": self.group,
        }


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame")
    return buffer.tobytes()


def _sanitize(value: str) -> str:
    """Make a filesystem path safe to use as a URL segment and as a label directory name."""
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return cleaned.strip("._") or "seq"


class _FrameCache:
    """Thread-safe LRU of encoded frames shared by both sequence kinds."""

    def __init__(self, limit: int) -> None:
        self._limit = max(8, limit)
        self._items: OrderedDict[int, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, index: int) -> bytes | None:
        with self._lock:
            payload = self._items.get(index)
            if payload is not None:
                self._items.move_to_end(index)
            return payload

    def put(self, index: int, payload: bytes) -> None:
        with self._lock:
            self._items[index] = payload
            self._items.move_to_end(index)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class VideoSequence:
    """A single video file exposed as an indexable frame sequence."""

    kind = "video"

    def __init__(self, path: Path, group: str = "", cache_limit: int = 192) -> None:
        self.path = path
        self.group = group
        self.seq_id = f"{_sanitize(group)}__{_sanitize(path.stem)}" if group else _sanitize(path.stem)
        self.name = path.stem
        self._cache = _FrameCache(cache_limit)
        self._decode_lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._position = 0  # index the next ``cap.read()`` will return

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open video: {path}")
        self.frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"video reports no frame size: {path}")
        # A container may over-report; trust the codec only when it gives something smaller and sane.
        if self.frame_count <= 0:
            self.frame_count = 0

    # -- decoding ---------------------------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open video: {self.path}")
            self._position = 0
        return self._cap

    def _seek(self, index: int) -> None:
        capture = self._open()
        self._cache.clear()
        if capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._position = index
            return
        # Some containers ignore POS_FRAMES; fall back to a plain rewind plus grab-forward.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._position = 0
        while self._position < index:
            if not capture.grab():
                break
            self._position += 1

    def _decode_until(self, index: int) -> bytes | None:
        capture = self._open()
        if index < self._position or index - self._position > SEEK_THRESHOLD:
            self._seek(index)
        while self._position <= index:
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            payload = _encode_jpeg(frame)
            self._cache.put(self._position, payload)
            self._position += 1
            if self._position == index + 1:
                return payload
        return None

    def frame_jpeg(self, index: int) -> bytes:
        index = max(0, int(index))
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        with self._decode_lock:
            cached = self._cache.get(index)
            if cached is not None:
                return cached
            payload = self._decode_until(index)
        if payload is None:
            raise IndexError(f"{self.name}: frame {index} not decodable (count={self.frame_count})")
        return payload

    def close(self) -> None:
        with self._decode_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def info(self) -> SequenceInfo:
        return SequenceInfo(
            seq_id=self.seq_id,
            name=self.name,
            kind=self.kind,
            source=str(self.path),
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
            fps=self.fps,
            group=self.group,
        )


class ImageDirSequence:
    """A directory of extracted frames; ``frame_%06d.jpg`` and friends all work because we sort by name."""

    kind = "frames"

    def __init__(self, directory: Path, group: str = "", cache_limit: int = 192) -> None:
        self.directory = directory
        self.group = group
        self.name = directory.name
        self.seq_id = f"{_sanitize(group)}__{_sanitize(directory.name)}" if group else _sanitize(directory.name)
        paths = sorted(
            (item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: item.name,
        )
        if len(paths) < MIN_SEQUENCE_FRAMES:
            raise RuntimeError(f"not enough frames in {directory}")
        self._paths = paths
        self._cache = _FrameCache(cache_limit)
        probe = cv2.imread(str(self._paths[0]), cv2.IMREAD_COLOR)
        if probe is None:
            raise RuntimeError(f"unreadable frame: {self._paths[0]}")
        self.width, self.height = probe.shape[1], probe.shape[0]
        self.fps = 25.0

    @property
    def frame_count(self) -> int:
        return len(self._paths)

    def frame_jpeg(self, index: int) -> bytes:
        index = max(0, int(index))
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        if index >= len(self._paths):
            raise IndexError(f"{self.name}: frame {index} out of range (count={len(self._paths)})")
        raw = self._paths[index].read_bytes()
        self._cache.put(index, raw)
        return raw

    def close(self) -> None:  # symmetry with VideoSequence
        return None

    def info(self) -> SequenceInfo:
        return SequenceInfo(
            seq_id=self.seq_id,
            name=self.name,
            kind=self.kind,
            source=str(self.directory),
            frame_count=len(self._paths),
            width=self.width,
            height=self.height,
            fps=self.fps,
            group=self.group,
        )


def _dedupe(seq_id: str, taken: dict[str, int]) -> str:
    if seq_id not in taken:
        taken[seq_id] = 1
        return seq_id
    taken[seq_id] += 1
    return f"{seq_id}__{taken[seq_id]}"


def scan_root(root: str | os.PathLike, max_sequences: int = 0, min_frames: int = MIN_SEQUENCE_FRAMES) -> list:
    """Recursively collect every video file and every frame directory under ``root``.

    A directory is treated as a frame sequence and is *not* descended into, so ``frames_ir_jpg/<seq>/frame_*.jpg``
    yields one sequence per ``<seq>`` rather than one per directory. Hidden directories and the tool's own
    output tree are skipped so re-scanning an annotated root is idempotent.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    sequences: list = []
    taken: dict[str, int] = {}
    skip_names = {".git", "__pycache__", ".cache", "labels", "annotations", ".annot"}

    for current, directories, files in os.walk(root_path):
        current_path = Path(current)
        directories[:] = sorted(item for item in directories if not item.startswith(".") and item not in skip_names)

        # ``os.walk`` yields the root itself first, whose parent is *not* under the root, so
        # ``relative_to`` would raise; fall back to an empty group instead.
        try:
            group = str(current_path.parent.relative_to(root_path)) if current_path.parent != root_path else ""
        except ValueError:
            group = ""

        videos = sorted(item for item in files if Path(item).suffix.lower() in VIDEO_SUFFIXES)
        if videos:
            for name in videos:
                try:
                    sequence = VideoSequence(current_path / name, group=group)
                except Exception as error:  # a single unreadable file must not kill the scan
                    print(f"[WARN] skipping {name}: {error}")
                    continue
                if max_sequences and len(sequences) >= max_sequences:
                    return sequences
                sequence.seq_id = _dedupe(sequence.seq_id, taken)
                sequences.append(sequence)
            # videos do not imply sibling frame dirs are part of this sequence; keep scanning
            continue

        images = sorted(item for item in files if Path(item).suffix.lower() in IMAGE_SUFFIXES)
        if len(images) >= min_frames:
            try:
                sequence = ImageDirSequence(current_path, group=group)
            except Exception as error:
                print(f"[WARN] skipping {current_path.name}: {error}")
            else:
                if max_sequences and len(sequences) >= max_sequences:
                    return sequences
                sequence.seq_id = _dedupe(sequence.seq_id, taken)
                sequences.append(sequence)
            directories[:] = []  # never descend into a directory already consumed as a sequence
    return sequences
