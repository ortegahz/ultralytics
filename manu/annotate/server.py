#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stdlib HTTP server for the video annotation tool — no fastapi/uvicorn required.

The workstation's torch environment is broken (torch 1.13.1 against numpy 2.x) and no interpreter here
has a web framework, so the server is written against ``http.server`` alone and keeps cv2/numpy as its
only real dependencies. Model inference never happens in this process; it is delegated to the training
server by :mod:`manu.annotate.jobs` when the annotator presses the button.

    python -m manu.annotate.server --root /path/to/videos --port 8777
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from manu.annotate.dataset import scan_root
from manu.annotate.jobs import INFERENCE_MODES, DEFAULT_SSH_TARGET, JobManager, resolve_mode, to_server_path
from manu.annotate.labels import LabelStore

STATIC_ROOT = Path(__file__).resolve().parent / "static"
_SEQ_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_FRAME_RE = re.compile(r"^/api/seq/([^/]+)/frame/(\d+)$")
_LABELS_RE = re.compile(r"^/api/seq/([^/]+)/labels/(\d+)$")


class AnnotatorState:
    """Everything the request handlers share: the sequence table, the label store and the job queue."""

    def __init__(self, root: str | Path, workspace: str | Path, max_sequences: int = 0,
                 prelabel_sources: list | None = None, prelabel_classes: list | None = None,
                 inference_mode: str = "auto") -> None:
        self.root = Path(root).expanduser().resolve()
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_sequences = max_sequences
        # Read-only proposal trees produced elsewhere (e.g. the project's existing preannot_v1 set).
        self.prelabel_sources = [Path(item) for item in (prelabel_sources or [])]
        self.prelabel_classes = prelabel_classes or None
        self.store = LabelStore(self.workspace, extra_prelabel_roots=self.prelabel_sources)
        self.jobs = JobManager(self.workspace, inference_mode=inference_mode)
        self._lock = threading.Lock()
        self.sequences: dict[str, object] = {}
        self.order: list[str] = []
        self.scan_error = ""
        self.score_cache: dict[str, tuple[int, dict]] = {}
        self.rescan()

    def rescan(self) -> None:
        try:
            found = scan_root(self.root, max_sequences=self.max_sequences)
        except Exception as error:
            self.scan_error = f"{type(error).__name__}: {error}"
            traceback.print_exc()
            return
        with self._lock:
            self.sequences = {item.seq_id: item for item in found}
            self.order = [item.seq_id for item in found]
            self.scan_error = ""

    def get(self, seq_id: str):
        with self._lock:
            return self.sequences.get(seq_id)

    def listing(self) -> list[dict]:
        with self._lock:
            return [self.sequences[key].info().to_dict() for key in self.order]


class Handler(BaseHTTPRequestHandler):
    server_version = "AnnotateTool/1.0"
    state: AnnotatorState  # injected by serve()

    # -- plumbing ---------------------------------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # keep the console for real errors only
        pass

    def _send(self, code: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser aborted a frame request; normal while scrubbing fast

    def _json(self, payload, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _sequence(self, seq_id: str):
        sequence = self.state.get(seq_id)
        if sequence is None:
            self._error(404, f"unknown sequence: {seq_id}")
        return sequence

    # -- routing ----------------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = posixpath.normpath(unquote(parsed.path))
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path == "/api/state":
                return self._json(
                    {
                        "root": str(self.state.root),
                        "workspace": str(self.state.workspace),
                        "sequences": self.state.listing(),
                        "scan_error": self.state.scan_error,
                    }
                )
            if path == "/api/jobs":
                return self._json({"jobs": self.state.jobs.all_jobs()})
            match = _FRAME_RE.match(path)
            if match:
                return self._frame(match.group(1), int(match.group(2)), query)
            match = _LABELS_RE.match(path)
            if match:
                return self._labels(match.group(1), int(match.group(2)))
            if re.match(r"^/api/seq/[^/]+/summary$", path):
                return self._summary(unquote(path.split("/")[3]))
            self._error(404, "not found")
        except Exception as error:  # a bad request must not take the server down
            traceback.print_exc()
            self._error(500, f"{type(error).__name__}: {error}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = posixpath.normpath(unquote(parsed.path))
        try:
            if re.match(r"^/api/seq/[^/]+/labels/\d+$", path):
                parts = path.split("/")
                return self._save_labels(parts[3], int(parts[5]))
            if re.match(r"^/api/seq/[^/]+/batch$", path):
                return self._batch(unquote(path.split("/")[3]))
            if path == "/api/preannotate":
                return self._preannotate()
            if re.match(r"^/api/jobs/[^/]+/cancel$", path):
                return self._json({"cancelled": self.state.jobs.cancel(unquote(path.split("/")[3]))})
            if path == "/api/rescan":
                self.state.rescan()
                return self._json({"sequences": self.state.listing(), "scan_error": self.state.scan_error})
            self._error(404, "not found")
        except Exception as error:
            traceback.print_exc()
            self._error(500, f"{type(error).__name__}: {error}")

    # -- endpoints --------------------------------------------------------------------------------
    def _static(self, relative: str) -> None:
        target = (STATIC_ROOT / relative).resolve()
        if not str(target).startswith(str(STATIC_ROOT)) or not target.is_file():
            return self._error(404, "asset not found")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type, cache="no-cache")

    def _frame(self, seq_id: str, index: int, query: dict) -> None:
        if not _SEQ_RE.match(seq_id):
            return self._error(400, "bad sequence id")
        sequence = self._sequence(seq_id)
        if sequence is None:
            return
        total = sequence.frame_count
        if total and index >= total:
            return self._error(404, f"frame {index} out of range (count={total})")
        payload = sequence.frame_jpeg(index)
        self._send(200, payload, "image/jpeg", cache="public, max-age=3600")

    def _labels(self, seq_id: str, index: int) -> None:
        sequence = self._sequence(seq_id)
        if sequence is None:
            return
        store = self.state.store
        scores = self._scores(seq_id, index)
        prelabels = store.read_prelabels(
            seq_id, index, sequence.width, sequence.height, classes=self.state.prelabel_classes
        )
        if self.state.prelabel_classes:
            # An external mixed set carries its own class ids (e.g. 0=bbox, 1=heatmap). Once the user has
            # said which classes are "the object", accepting them must produce class-0 labels, because
            # the label set is single-class — otherwise accepting a proposal silently writes class 1.
            prelabels = [dict(box, cls=0) for box in prelabels]
        self._json(
            {
                "index": index,
                "labels": store.read_labels(seq_id, index, sequence.width, sequence.height),
                "prelabels": prelabels,
                "scores": scores,
                "reviewed": store.is_reviewed(seq_id, index),
                "width": sequence.width,
                "height": sequence.height,
            }
        )

    def _scores(self, seq_id: str, index: int) -> list[float]:
        """Peak confidence per proposal, when the proposals came from this tool's own worker.

        External trees have no score sidecar, so their proposals render without one; the UI treats a
        missing score as "always visible" rather than hiding them.
        """
        key = str(self.state.store.prelabel_root / seq_id / "scores.json")
        cache = self.state.score_cache
        entry = cache.get(key)
        try:
            if entry is None or entry[0] != self.state.store.prelabel_root.joinpath(seq_id, "scores.json").stat().st_mtime_ns:
                path = self.state.store.prelabel_root / seq_id / "scores.json"
                entry = (path.stat().st_mtime_ns, json.loads(path.read_text(encoding="utf-8")))
                cache.clear()
                cache[key] = entry
        except (OSError, json.JSONDecodeError):
            return []
        return entry[1].get(f"{index:06d}", [])

    def _summary(self, seq_id: str) -> None:
        sequence = self._sequence(seq_id)
        if sequence is None:
            return
        # The exact box total is linear in sequence length, so the UI asks for it in a second,
        # background request instead of blocking the timeline on it.
        query = parse_qs(urlparse(self.path).query)
        want_boxes = query.get("boxes", ["0"])[0] in {"1", "true", "yes"}
        info = sequence.info().to_dict()
        info["summary"] = self.state.store.summary(seq_id, sequence.frame_count, count_boxes=want_boxes)
        self._json(info)

    def _save_labels(self, seq_id: str, index: int) -> None:
        sequence = self._sequence(seq_id)
        if sequence is None:
            return
        payload = self._body()
        boxes = payload.get("boxes") or []
        if not isinstance(boxes, list):
            return self._error(400, "boxes must be a list")
        stored = self.state.store.write_labels(seq_id, index, boxes, sequence.width, sequence.height)
        if payload.get("reviewed", True):
            self.state.store.mark_reviewed(seq_id, [index])
        self._json({"index": index, "boxes": stored, "reviewed": True})

    def _batch(self, seq_id: str) -> None:
        """Write many frames in one round trip — track propagation writes dozens at a time."""
        sequence = self._sequence(seq_id)
        if sequence is None:
            return
        payload = self._body()
        entries = payload.get("entries") or []
        if not isinstance(entries, list):
            return self._error(400, "entries must be a list")
        store = self.state.store
        written = []
        for entry in entries:
            try:
                index = int(entry["index"])
                boxes = entry.get("boxes") or []
            except (KeyError, TypeError, ValueError):
                continue
            store.write_labels(seq_id, index, boxes, sequence.width, sequence.height)
            written.append(index)
        if payload.get("reviewed", True) and written:
            store.mark_reviewed(seq_id, written)
        self._json({"written": written})

    def _preannotate(self) -> None:
        payload = self._body()
        raw_ids = payload.get("seq_ids")
        if not raw_ids:
            single = str(payload.get("seq_id") or "")
            raw_ids = [single] if single else []
        seq_ids = [str(item) for item in raw_ids if str(item)]
        if not seq_ids:
            return self._error(400, "seq_id or seq_ids is required")

        extra = []
        for key, flag in (("scales", "--scales"), ("main_threshold", "--main-threshold"), ("max_frames", "--max-frames")):
            if payload.get(key) not in (None, ""):
                extra += [flag, str(payload[key])]

        devices = [str(item) for item in (payload.get("devices") or []) if str(item).strip()]
        requests, skipped, invisible = [], [], []
        for seq_id in seq_ids:
            sequence = self.state.get(seq_id)
            if sequence is None:
                skipped.append(seq_id)
                continue
            local_source = sequence.info().source
            # In local mode the worker runs right here, so the source needs no cross-machine translation
            # at all — running this check with the workstation PATH_MAP would reject every sequence.
            if to_server_path(local_source, translate=not self.state.jobs.local) is None:
                # A local-only video can still be labelled by hand; it just has nothing to propose.
                invisible.append(local_source)
                continue
            requests.append((seq_id, local_source))

        if not requests:
            return self._error(
                400,
                f"序列不在服务器可见范围内（{', '.join(invisible[:3])}）；"
                "请把视频放在 SSHFS 挂载内，或先人工标注",
            )
        try:
            jobs = self.state.jobs.submit_many(requests, extra, devices or None)
        except RuntimeError as error:
            return self._error(409, str(error))
        status_code = 202 if len(jobs) == 1 else 202
        self._json(
            {
                "jobs": [job.to_dict() for job in jobs],
                "skipped": skipped,
                "not_visible_to_server": invisible,
                "gpus": sorted({job.device for job in jobs}),
                "inference": self.state.jobs.describe(),
            },
            code=status_code,
        )


def serve(root: str, workspace: str, host: str, port: int, max_sequences: int = 0,
          prelabel_sources: list | None = None, prelabel_classes: list | None = None,
          inference_mode: str = "auto") -> None:
    state = AnnotatorState(root, workspace, max_sequences, prelabel_sources, prelabel_classes, inference_mode)
    Handler.state = state
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"[annotate] root      = {state.root}")
    print(f"[annotate] workspace = {state.workspace}")
    print(f"[annotate] sequences = {len(state.order)}")
    if state.prelabel_sources:
        classes = ",".join(str(item) for item in state.prelabel_classes) if state.prelabel_classes else "全部"
        print(f"[annotate] prelabel sources (read-only, classes={classes}):")
        for item in state.prelabel_sources:
            print(f"             - {item}{'' if item.is_dir() else '  [不存在]'}")
    if state.scan_error:
        print(f"[annotate] scan error: {state.scan_error}")
    print(f"[annotate] open http://{host}:{port}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[annotate] shutting down")
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Web video annotation tool with multi-scale SOTA pre-annotation")
    parser.add_argument("--root", required=True, help="Folder recursively scanned for videos / frame directories")
    parser.add_argument("--workspace", default="", help="Label workspace; defaults to <root>/../annotate_workspace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--prelabel-source", action="append", default=[],
        help="Read-only YOLO proposal tree produced elsewhere; repeatable. Reuse it instead of copying.",
    )
    parser.add_argument(
        "--prelabel-class", default="",
        help="Comma-separated class ids kept from external proposals, e.g. '1' to take only the "
             "heatmap branch of a mixed set. Empty keeps every class.",
    )
    parser.add_argument(
        "--inference-mode", choices=list(INFERENCE_MODES), default="auto",
        help="How the pre-annotation worker is started. 'auto' (default) runs it in place when this host "
             "is the SSH target and over SSH otherwise; 'local' forces in-place; 'ssh' forces the hop.",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else root.parent / "annotate_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    classes = [int(item) for item in args.prelabel_class.split(",") if item.strip()] or None
    print(f"[annotate] inference  = {'in-place' if resolve_mode(args.inference_mode, DEFAULT_SSH_TARGET) else 'ssh ' + DEFAULT_SSH_TARGET}")
    serve(
        str(root), str(workspace), args.host, args.port, args.max_sequences,
        prelabel_sources=args.prelabel_source, prelabel_classes=classes,
        inference_mode=args.inference_mode,
    )


if __name__ == "__main__":
    main()
