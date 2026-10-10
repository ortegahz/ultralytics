# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO label storage, review state and atomic saves.

Layout under the workspace root handed to the tool::

    labels/<seq_id>/<frame:06d>.txt    human-edited annotations, ``class cx cy w h`` normalised
    prelabels/<seq_id>/<frame:06d>.txt model proposals, same format, never overwritten by edits
    review/<seq_id>.json               per-frame review flags

Keeping proposals and human edits in separate trees is what makes "unreviewed" meaningful: a frame that
still matches its proposal byte-for-byte has not been looked at, which is exactly the work queue the
annotator needs. Duplicate rows are dropped on read because this project has already shipped 22 label
files that contained literally repeated lines, and any row count taken before dedup is inflated.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

LABEL_SUFFIX = ".txt"


def _index_name(index: int) -> str:
    return f"{max(0, int(index)):06d}{LABEL_SUFFIX}"


def parse_yolo(text: str, width: int, height: int) -> list[dict]:
    """Parse ``class cx cy w h`` (normalised) into pixel-space ``xyxy`` boxes, dropping duplicates.

    Raises on malformed rows rather than silently skipping them: a label file that cannot be parsed is a
    real defect and the annotator must see it rather than annotate on top of a truncated view.
    """
    boxes: list[dict] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"line {lineno}: expected 5 fields, got {len(parts)}: {raw!r}")
        try:
            klass = int(float(parts[0]))
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
        except ValueError as error:
            raise ValueError(f"line {lineno}: {error}") from error
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        half_w, half_h = bw * width / 2.0, bh * height / 2.0
        # Clamp to the same [0, width] x [0, height] box that ``format_yolo`` clips to. These two have to
        # be inverses: clamping one pixel tighter here made every edge-touching box shrink by a pixel on
        # each save/reload cycle, so a box drawn against the frame border crept inward while annotating.
        x1 = max(0.0, min(cx * width - half_w, float(width)))
        y1 = max(0.0, min(cy * height - half_h, float(height)))
        x2 = min(float(width), max(x1 + 1.0, cx * width + half_w))
        y2 = min(float(height), max(y1 + 1.0, cy * height + half_h))
        # Sub-pixel precision is kept on purpose. Rounding to whole pixels here made the pipeline lossy
        # on every save/reload cycle: the file stores six decimals, the UI re-sent the rounded integers,
        # and the next save wrote visibly different numbers. For a 6x6 px IR target that is ~10% of the
        # box. Two decimals is far finer than any annotation needs and keeps the JSON compact.
        pixel = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
        key = (klass, *pixel)
        if key in seen:
            continue
        seen.add(key)
        boxes.append({"cls": klass, "x1": pixel[0], "y1": pixel[1], "x2": pixel[2], "y2": pixel[3]})
    return boxes


def format_yolo(boxes: list[dict], width: int, height: int, min_visible: float = 0.10) -> str:
    """Serialise pixel ``xyxy`` boxes back to normalised ``class cx cy w h``, clipped to the image.

    A box is dropped when less than ``min_visible`` of its area falls inside the frame. Plain clamping
    alone would turn a target that has flown out of view into a thin sliver welded to the border, which
    pollutes the label set exactly where a tracking target is leaving; a mostly-outside box is not a
    label. Boxes that are merely clipped at an edge are kept and clipped, as usual.

    Visibility is measured against the **unclamped** box — clamping first would make every box look
    fully visible and the check would never fire.

    Identical rows are collapsed here so a stored label file is canonical: the on-disk line count is
    then the exact box count, which lets ``summary()`` skip re-parsing every file.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for box in boxes:
        x1, x2 = float(box["x1"]), float(box["x2"])
        y1, y2 = float(box["y1"]), float(box["y2"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        original_area = (x2 - x1) * (y2 - y1)
        if original_area <= 0:
            continue
        # Intersect with the frame; do not clamp, so a fully-outside box collapses to zero area.
        ix1, iy1 = max(x1, 0.0), max(y1, 0.0)
        ix2, iy2 = min(x2, float(width)), min(y2, float(height))
        visible_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if visible_area <= 0 or visible_area < min_visible * original_area:
            continue
        bw, bh = ix2 - ix1, iy2 - iy1
        if bw <= 0 or bh <= 0:
            continue
        line = (
            f"{int(box.get('cls', 0))} {(ix1 + bw / 2) / width:.6f} {(iy1 + bh / 2) / height:.6f} "
            f"{bw / width:.6f} {bh / height:.6f}"
        )
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _run_length(flags: list[bool]) -> list[list[int]]:
    """Compress a per-frame boolean list as ``[[value, count], ...]`` for a compact timeline payload."""
    runs: list[list[int]] = []
    for value in flags:
        if runs and runs[-1][0] == int(value):
            runs[-1][1] += 1
        else:
            runs.append([int(value), 1])
    return runs


class LabelStore:
    """Owns the label/prelabel/review trees for one annotation workspace.

    ``extra_prelabel_roots`` are **read-only** directories of YOLO proposals produced elsewhere —
    typically the project's existing ``preannot_v1`` tree, which already holds one proposal file per
    frame for every 龙泉山 sequence. Pointing at them costs nothing instead of copying ~165k small files
    across the SSHFS mount, and human edits can never reach them because the tool only reads them.

    Both layouts are understood, because pre-existing annotation sets come in both:
      per-sequence ``<root>/<seq_id>/<frame:06d>.txt``
      flat          ``<root>/<seq_id>__<frame:06d>.txt``
    """

    def __init__(self, root: str | os.PathLike, extra_prelabel_roots: list | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.labels_root = self.root / "labels"
        self.prelabel_root = self.root / "prelabels"
        self.review_root = self.root / "review"
        for directory in (self.labels_root, self.prelabel_root, self.review_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.extra_prelabel_roots = [Path(item).expanduser() for item in (extra_prelabel_roots or [])]
        self._external_index: dict[str, dict[str, set[int]]] = {}
        self._lock = threading.RLock()
        self._review_cache: dict[str, dict] = {}

    # -- paths ------------------------------------------------------------------------------------
    def label_path(self, seq_id: str, index: int) -> Path:
        return self.labels_root / seq_id / _index_name(index)

    def prelabel_path(self, seq_id: str, index: int) -> Path:
        return self.prelabel_root / seq_id / _index_name(index)

    def review_path(self, seq_id: str) -> Path:
        return self.review_root / f"{seq_id}.json"

    def external_prelabel_path(self, seq_id: str, index: int) -> Path | None:
        """Resolve a proposal in a read-only external tree, or ``None`` when no tree provides it."""
        name = _index_name(index)
        for base in self.extra_prelabel_roots:
            candidate = base / seq_id / name
            if candidate.exists():
                return candidate
            candidate = base / f"{seq_id}__{name}"
            if candidate.exists():
                return candidate
        return None

    def _external_indices(self, seq_id: str) -> set[int]:
        """Indices available in the external trees, cached per sequence.

        Built with one listing per root rather than one ``stat`` per frame: the flat layout holds
        165k files, and probing each of them per request would be hopeless over a network mount.
        """
        cached = self._external_index.get(seq_id)
        if cached is not None:
            return cached.get("indices", set())
        found: set[int] = set()
        for base in self.extra_prelabel_roots:
            if not base.is_dir():
                continue
            directory = base / seq_id
            if directory.is_dir():
                for path in directory.glob(f"*{LABEL_SUFFIX}"):
                    if path.stem.isdigit():
                        found.add(int(path.stem))
                continue
            prefix = f"{seq_id}__"
            try:
                entries = list(base.glob(f"{prefix}*{LABEL_SUFFIX}"))
            except OSError:
                continue
            for path in entries:
                stem = path.stem[len(prefix):]
                if stem.isdigit():
                    found.add(int(stem))
        self._external_index[seq_id] = {"indices": found}
        return found

    def invalidate_external_cache(self) -> None:
        with self._lock:
            self._external_index.clear()

    # -- reads ------------------------------------------------------------------------------------
    def _read_boxes(self, path: Path, width: int, height: int) -> list[dict]:
        if not path.exists():
            return []
        return parse_yolo(path.read_text(encoding="utf-8"), width, height)

    def read_labels(self, seq_id: str, index: int, width: int, height: int) -> list[dict]:
        with self._lock:
            return self._read_boxes(self.label_path(seq_id, index), width, height)

    def read_prelabels(self, seq_id: str, index: int, width: int, height: int,
                       classes: list[int] | None = None) -> list[dict]:
        """Proposals for one frame: the tool's own tree first, then any read-only external tree."""
        with self._lock:
            path = self.prelabel_path(seq_id, index)
            external = None
            if not path.exists():
                external = self.external_prelabel_path(seq_id, index)
                path = external if external is not None else path
            boxes = self._read_boxes(path, width, height)
        if classes:
            boxes = [box for box in boxes if int(box.get("cls", 0)) in classes]
        return boxes

    # -- writes -----------------------------------------------------------------------------------
    def write_labels(self, seq_id: str, index: int, boxes: list[dict], width: int, height: int) -> list[dict]:
        """Persist boxes for one frame. Returns the round-tripped boxes so the client sees what was stored."""
        with self._lock:
            text = format_yolo(boxes, width, height)
            _atomic_write(self.label_path(seq_id, index), text)
            return parse_yolo(text, width, height) if text else []

    def write_prelabels(self, seq_id: str, index: int, boxes: list[dict], width: int, height: int) -> None:
        with self._lock:
            _atomic_write(self.prelabel_path(seq_id, index), format_yolo(boxes, width, height))

    def delete_labels(self, seq_id: str, index: int) -> None:
        with self._lock:
            path = self.label_path(seq_id, index)
            if path.exists():
                path.unlink()

    # -- review state -----------------------------------------------------------------------------
    def _review(self, seq_id: str) -> dict:
        cached = self._review_cache.get(seq_id)
        if cached is not None:
            return cached
        path = self.review_path(seq_id)
        state: dict = {"reviewed": []}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("reviewed"), list):
                    state = {"reviewed": sorted({int(item) for item in loaded["reviewed"]})}
            except (json.JSONDecodeError, ValueError, TypeError):
                # A corrupt review file must not block annotation; start a fresh one rather than crash.
                state = {"reviewed": []}
        self._review_cache[seq_id] = state
        return state

    def is_reviewed(self, seq_id: str, index: int) -> bool:
        with self._lock:
            return int(index) in set(self._review(seq_id)["reviewed"])

    def mark_reviewed(self, seq_id: str, indices) -> None:
        with self._lock:
            state = self._review(seq_id)
            merged = sorted(set(state["reviewed"]) | {int(item) for item in indices})
            state["reviewed"] = merged
            _atomic_write(self.review_path(seq_id), json.dumps({"reviewed": merged}))

    def summary(self, seq_id: str, frame_count: int, count_boxes: bool = False) -> dict:
        """Per-sequence work-queue state, run-length encoded so a 16k-frame sequence stays a few KB.

        ``count_boxes`` reads every label file to total the boxes. That is the only expensive part and it
        is linear in sequence length, so it is opt-in: at 12k frames it measured 3.3 s, which would
        stall the UI on every sequence click. Because ``format_yolo`` collapses duplicate rows on write,
        counting non-empty lines is exact without a full parse.
        """
        with self._lock:
            label_dir = self.labels_root / seq_id
            prelabel_dir = self.prelabel_root / seq_id
            labelled = set()
            if label_dir.is_dir():
                for path in label_dir.glob(f"*{LABEL_SUFFIX}"):
                    stem = path.stem
                    if stem.isdigit():
                        labelled.add(int(stem))
            prelabelled = set()
            if prelabel_dir.is_dir():
                for path in prelabel_dir.glob(f"*{LABEL_SUFFIX}"):
                    stem = path.stem
                    if stem.isdigit():
                        prelabelled.add(int(stem))
            if self.extra_prelabel_roots:
                prelabelled |= self._external_indices(seq_id)
            reviewed = set(self._review(seq_id)["reviewed"])

            labelled_flag = [index in labelled for index in range(frame_count)]
            prelabel_flag = [index in prelabelled for index in range(frame_count)]
            reviewed_flag = [index in reviewed for index in range(frame_count)]

            boxes_total = None
            if count_boxes:
                boxes_total = 0
                for index in sorted(labelled):
                    if index >= frame_count:
                        continue
                    try:
                        text = self.label_path(seq_id, index).read_text(encoding="utf-8")
                    except OSError:
                        continue
                    boxes_total += sum(1 for line in text.splitlines() if line.strip() and not line.startswith("#"))
            return {
                "frame_count": frame_count,
                "labelled": _run_length(labelled_flag),
                "prelabelled": _run_length(prelabel_flag),
                "reviewed": _run_length(reviewed_flag),
                "boxes_total_known": count_boxes,
                "counts": {
                    "labelled": sum(1 for index in labelled_flag if index),
                    "prelabelled": sum(1 for index in prelabel_flag if index),
                    "reviewed": sum(1 for index in reviewed_flag if index),
                    "empty": sum(1 for flag in labelled_flag if not flag),
                    "boxes": boxes_total,
                },
            }
