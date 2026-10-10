# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Web video annotation tool: recursive video discovery, multi-scale SOTA pre-annotation, LabelMe-style editing.

The package is split so the HTTP layer stays dependency-free:

    dataset.py    recursive discovery + frame decoding (cv2 only)
    labels.py     YOLO label read/write, review state, atomic saves
    jobs.py       pre-annotation job manager (spawns worker_infer.py)
    worker_infer.py  multi-scale Trial 0474 inference, needs torch (run in the `uav` env)
    server.py     stdlib ThreadingHTTPServer, no fastapi/uvicorn required
    static/       canvas front-end

``server.py`` and ``dataset.py`` import nothing beyond cv2/numpy so the server can start in any
interpreter, while ``worker_infer.py`` is launched as a subprocess in the torch-capable environment.
"""

__all__ = ["dataset", "labels", "jobs", "server"]
