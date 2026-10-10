# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pre-annotation job manager: spawns the torch worker on the training server.

The web server normally runs on the workstation, but the only working torch environment is the training
server's ``uav`` conda env (the local ``uav`` env pairs torch 1.13.1 with numpy 2.x, so every tensor
interop call raises ``RuntimeError: Numpy is not available``). Inference is therefore remote.

Three consequences shape this module:

* Paths must be translated. The workstation sees the server through SSHFS mounts, so a local prefix maps
  to a different server prefix; ``PATH_MAP`` holds both ends of every mount the tool relies on.
* The model only loads when the annotator presses the button. Nothing here runs at import or start-up,
  which keeps the server free of the compute the annotator did not ask for.
* **The server can host the web UI too.** Deploying there is the better layout — frame reads and GPU
  inference stop crossing a network hop — but then ``ssh host -> ssh host`` is a self-connection, and the
  server has no private key for itself, so it is rejected with ``Permission denied (publickey)``. Hence
  ``inference_mode``: in ``local`` mode the worker is spawned in place instead of being shipped over SSH.
  That also flips path translation off, since there is no longer a second filesystem view to translate to.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Local (workstation, SSHFS view) prefix -> server prefix. Longest matching prefix wins.
PATH_MAP: list[tuple[str, str]] = [
    ("/home/manu/mnt/pycharm_project_10ae9e2e", "/tmp/pycharm_project_10ae9e2e"),
    ("/home/manu/mnt/data", "/mnt/data"),
]

DEFAULT_SSH_TARGET = "huangzhe@192.168.99.40"
DEFAULT_SSH_PORT = 32222
DEFAULT_SERVER_REPO = "/tmp/pycharm_project_10ae9e2e"
DEFAULT_CONDA_ENV = "uav"

INFERENCE_MODES = ("auto", "ssh", "local")


def local_ipv4() -> set[str]:
    """Every IPv4 address bound on this host.

    ``socket.gethostbyname(socket.gethostname())`` is useless for this: on Debian/Ubuntu it answers
    ``127.0.1.1``, not the real address, so the server would never recognise itself as the SSH target.
    Reading each interface through the kernel instead keeps the answer truthful without shelling out or
    pulling in a dependency.
    """
    found = {"127.0.0.1"}
    try:
        import fcntl
        import struct

        for interface in os.listdir("/sys/class/net"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                request = struct.pack("256s", interface.encode()[:15])
                packed = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
                found.add(socket.inet_ntoa(packed[20:24]))
            finally:
                sock.close()
    except (ImportError, OSError):
        pass
    return found


def ssh_host(target: str) -> str:
    """``huangzhe@192.168.99.40`` -> ``192.168.99.40``."""
    return target.rsplit("@", 1)[-1].strip()


def resolve_mode(mode: str, target: str) -> bool:
    """Decide whether the worker runs in place. Returns ``True`` for local mode.

    ``auto`` compares the SSH target against this host's own addresses, so the same deployment works on
    either machine without anyone having to remember a flag. ``ssh``/``local`` pin the answer, which is
    what you want when auto-detection is being second-guessed.
    """
    if mode == "local":
        return True
    if mode == "ssh":
        return False
    if mode not in INFERENCE_MODES:
        raise ValueError(f"unknown inference mode {mode!r}; expected one of {INFERENCE_MODES}")
    host = ssh_host(target)
    try:
        return host in local_ipv4() or host == socket.gethostname()
    except OSError:
        return False


def to_server_path(local_path: str | Path, *, translate: bool = True) -> str | None:
    """Translate a workstation path into its server-side twin, or ``None`` when it is not shared.

    ``translate=False`` is the local-mode case: the caller already runs where the worker will run, so
    there is nothing to translate. Passing it through ``PATH_MAP`` anyway would be actively wrong — a
    server-side ``/mnt/data/...`` path matches no workstation prefix and comes back ``None``, which the
    caller reports as "not reachable from the server" even though it is right there.
    """
    text = str(Path(local_path).expanduser())
    if not translate:
        return text
    for local_prefix, server_prefix in PATH_MAP:
        if text == local_prefix or text.startswith(local_prefix.rstrip("/") + "/"):
            return server_prefix + text[len(local_prefix) :]
    return None


@dataclass
class Job:
    seq_id: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    message: str = ""
    done: int = 0
    total: int = 0
    detections: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    returncode: int | None = None
    log: list[str] = field(default_factory=list)
    # Server-side paths, resolved at submit time so the UI thread never does path translation.
    server_source: str = ""
    server_prelabel: str = ""
    server_status: str = ""
    device: str = "0"
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "seq_id": self.seq_id,
            "status": self.status,
            "message": self.message,
            "done": self.done,
            "total": self.total,
            "detections": self.detections,
            "device": self.device,
            "elapsed": round(elapsed, 1),
            "returncode": self.returncode,
            "log": self.log[-40:],
        }


class JobManager:
    """Tracks at most one running pre-annotation job and streams its output into the UI."""

    def __init__(
        self,
        workspace: Path,
        ssh_target: str = DEFAULT_SSH_TARGET,
        ssh_port: int = DEFAULT_SSH_PORT,
        server_repo: str = DEFAULT_SERVER_REPO,
        conda_env: str = DEFAULT_CONDA_ENV,
        device: str = "0",
        max_concurrent: int = 4,
        gpu_devices: list[str] | None = None,
        inference_mode: str = "auto",
    ) -> None:
        self.workspace = Path(workspace)
        self.ssh_target = ssh_target
        self.ssh_port = ssh_port
        self.server_repo = server_repo
        self.conda_env = conda_env
        self.device = device
        self.gpu_devices = gpu_devices or [device]
        self.inference_mode = inference_mode
        self.local = resolve_mode(inference_mode, ssh_target)
        # Default to one worker per GPU: the training server carries four idle 4090s.
        self.max_concurrent = max_concurrent if max_concurrent else len(self.gpu_devices)
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._runner = threading.Thread(target=self._run_queue, daemon=True)
        self._runner.start()

    # -- public API --------------------------------------------------------------------------------
    def submit(self, seq_id: str, source: str, extra: list[str] | None = None, device: str | None = None) -> Job:
        """Queue one sequence.

        ``source`` is a **workstation** path; the translation to the server twin happens here, once, so
        callers cannot accidentally hand over an already-translated path and have it translated twice.
        In local mode there is nothing to translate and the path is used verbatim.

        Job status alone tracks liveness — an extra bookkeeping set used to drift out of sync with it
        and silently stranded every job in ``queued``.
        """
        translate = not self.local
        server_prelabel = to_server_path(self.workspace / "prelabels", translate=translate)
        server_status = to_server_path(self.workspace / "jobs", translate=translate)
        if server_prelabel is None or server_status is None:
            raise RuntimeError(
                f"workspace {self.workspace} is not reachable from the server; "
                f"place it under one of: {[local for local, _ in PATH_MAP]}"
            )
        server_source = to_server_path(source, translate=translate)
        if server_source is None:
            # Refuse here rather than queue a job that could only ever fail on the far side.
            raise RuntimeError(f"source {source} is not reachable from the server")
        with self._lock:
            active = self._jobs.get(seq_id)
            if active is not None and active.status in {"queued", "running"}:
                raise RuntimeError(f"{seq_id} is already running")
            job = Job(
                seq_id=seq_id,
                status="queued",
                message="queued",
                server_source=server_source,
                server_prelabel=server_prelabel,
                server_status=server_status,
                device=device if device is not None else self.device,
                extra_args=list(extra or []),
            )
            self._jobs[seq_id] = job
            if seq_id in self._order:
                self._order.remove(seq_id)
            self._order.append(seq_id)
            self._trim_locked()
        return job

    def submit_many(self, requests: list[tuple[str, str]], extra: list[str] | None = None,
                    devices: list[str] | None = None) -> list[Job]:
        """Queue several sequences, spreading them round-robin over ``devices``.

        The training server carries four 4090s, so one click can pre-annotate the whole folder instead
        of forcing one click and one GPU per sequence.
        """
        pool = devices or [self.device]
        jobs = []
        for index, (seq_id, server_source) in enumerate(requests):
            jobs.append(self.submit(seq_id, server_source, extra, device=pool[index % len(pool)]))
        return jobs

    def status(self, seq_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(seq_id)

    def all_jobs(self) -> list[dict]:
        with self._lock:
            return [self._jobs[key].to_dict() for key in reversed(self._order)]

    def cancel(self, seq_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(seq_id)
            if job is None or job.status not in {"queued", "running"}:
                return False
            job.status = "cancelled"
            job.message = "cancelled by user"
            return True

    def _trim_locked(self) -> None:
        while len(self._order) > 200:
            oldest = self._order.pop(0)
            if self._jobs.get(oldest) and self._jobs[oldest].status not in {"queued", "running"}:
                self._jobs.pop(oldest, None)

    # -- execution ---------------------------------------------------------------------------------
    def _worker_command(self, seq_id: str, source: str, prelabel_dir: str, status_dir: str,
                        device: str, extra: list[str]) -> str:
        """The shell line that actually runs the worker, identical on either host.

        Conda is sourced explicitly in both cases: the server's default ``python`` is 2.7, so relying on
        a login shell to have picked the env up is what makes workers die with an unrelated syntax error.
        """
        return (
            f"source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || "
            f"source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; "
            f"conda activate {self.conda_env} && "
            f"cd {shlex.quote(self.server_repo)} && "
            f"PYTHONPATH={shlex.quote(self.server_repo)} python manu/annotate/worker_infer.py "
            f"--source {shlex.quote(source)} "
            f"--seq-id {shlex.quote(seq_id)} "
            f"--prelabel-dir {shlex.quote(prelabel_dir)} "
            f"--status-file {shlex.quote(status_dir + '/' + seq_id + '.json')} "
            f"--device {shlex.quote(device)} " + " ".join(shlex.quote(item) for item in extra)
        )

    def _build_command(self, seq_id: str, source: str, prelabel_dir: str, status_dir: str,
                       device: str, extra: list[str]) -> list[str]:
        """Wrap the worker line in an SSH hop, or run it in place when this host *is* the server."""
        remote = self._worker_command(seq_id, source, prelabel_dir, status_dir, device, extra)
        if self.local:
            # `bash -lc` rather than the caller's shell so conda activation sees a login environment
            # without depending on how the web server was itself started.
            return ["bash", "-lc", remote]
        return [
            "ssh",
            "-p",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ServerAliveInterval=30",
            self.ssh_target,
            remote,
        ]

    def describe(self) -> str:
        """One line for the UI so the operator can see which path inference will take."""
        if self.local:
            return f"in-place on this host ({socket.gethostname()}), mode={self.inference_mode}"
        return f"ssh {self.ssh_target}:{self.ssh_port}, mode={self.inference_mode}"

    def _run_queue(self) -> None:
        """Dispatch queued jobs, one thread each, honouring ``max_concurrent``.

        The only state consulted is ``Job.status``: ``queued`` is dispatchable, ``running`` counts
        against the concurrency budget, anything else is terminal.
        """
        while True:
            with self._lock:
                running = sum(1 for key in self._order if self._jobs[key].status == "running")
                free = self.max_concurrent - running
                candidates = [key for key in self._order if self._jobs[key].status == "queued"][: max(0, free)]
                for key in candidates:
                    self._jobs[key].status = "running"
                    self._jobs[key].message = (
                        "starting worker in place" if self.local else "connecting to training server"
                    )
            for seq_id in candidates:
                threading.Thread(target=self._launch, args=(seq_id,), daemon=True).start()
            time.sleep(0.3)

    def _launch(self, seq_id: str) -> None:
        with self._lock:
            job = self._jobs.get(seq_id)
            if job is None or job.status != "running":
                return
            source, prelabel_dir = job.server_source, job.server_prelabel
            status_dir, extra = job.server_status, list(job.extra_args)
            device = job.device

        command = self._build_command(seq_id, source, prelabel_dir, status_dir, device, extra)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            with self._lock:
                job.status = "failed"
                job.message = f"cannot spawn {'bash' if self.local else 'ssh'}: {error}"
                job.finished_at = time.time()
            return

        for line in process.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if not line:
                continue
            self._absorb_progress(job, line)
        process.wait()
        with self._lock:
            job.returncode = process.returncode
            job.finished_at = time.time()
            if job.status != "cancelled":
                if process.returncode == 0:
                    job.status = "done"
                    job.message = job.message or "completed"
                else:
                    job.status = "failed"
                    tail = job.log[-1] if job.log else ""
                    job.message = f"worker exited with code {process.returncode}" + (f": {tail}" if tail else "")

    def _absorb_progress(self, job: Job, line: str) -> None:
        """Fold one line of worker output, plus its status file, into the job record."""
        progress = self.poll_status_file(job.seq_id)
        with self._lock:
            job.log.append(line)
            if len(job.log) > 400:
                del job.log[:200]
            if progress:
                job.done = int(progress.get("done", job.done) or 0)
                job.total = int(progress.get("total", job.total) or 0)
                job.detections = int(progress.get("detections", job.detections) or 0)
            if line.startswith("[DONE]"):
                job.message = line
            elif line.startswith("[INFO]") or line.startswith("[WARN]") or line.startswith("[ERROR]"):
                job.message = line
            elif progress.get("state") == "running" and job.total:
                job.message = f"{job.done}/{job.total} frames, {job.detections} proposals"

    def poll_status_file(self, seq_id: str) -> dict:
        """Read the worker's own progress file through the shared mount, when it exists."""
        path = self.workspace / "jobs" / f"{seq_id}.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
