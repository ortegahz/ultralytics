#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""One-command re-verification of the trial_0080 Longquanshan delivery evidence.

Why
---
A delivery claim is only as good as the artefacts behind it.  The numbers live in three
different places (the local repository, the experiment tree on the server, and the OSD
directory on the data volume), and every one of them can drift.  This script re-checks the
whole chain in one command and prints PASS/FAIL per item, so the claim can be re-verified
instead of trusted.

What it checks
---------------
1. **Parameter files** -- the frozen (trial_0474) point and the v1/v2 points must be
   bit-identical to their recorded md5; the v3 point must exist and carry the v3 values.
2. **Evaluation reports** -- every JSON exists and its metrics equal the recorded values
   (single-frame arm and the three track arms).
3. **Delivery criterion** -- F1 >= 88 %, recall >= 85 %, FAR <= 0.030/frame, recomputed
   from the raw counts of the v3 report.
4. **Heatmap cache** -- 24 pkl files, and every meta agrees on the build parameters.
5. **OSD sets** -- five directories, each with 24 mp4, no zero-byte file, and the
   pass/un-pass split matching the recorded values.  The byte total is the **sum of the
   file sizes** (not ``du -sb``), because ``du`` also counts directory entries and therefore
   differs between the native filesystem and an SSHFS mount for the same content.
6. **Old OSD untouched** -- every mp4 of the 2026-10-20 set must still carry an mtime from
   2026-10-20, i.e. nothing was rewritten since it was produced.
7. **Three-way md5** -- the local repository, the SSHFS project mount and the server agree
   on the md5 of every file this work created or modified.

Usage
-----
    python manu/evaluation/verify_delivery_evidence.py
    python manu/evaluation/verify_delivery_evidence.py --project-mount /home/manu/mnt/pycharm_project_10ae9e2e

Execution machine: any machine where the repository, the project mount and the data mount
are visible (no GPU, no model, no inference).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# The recorded facts.  Every value here is traceable to the run that produced it; the
# script fails loudly if reality has drifted away from the record.
# --------------------------------------------------------------------------------------

EXPECTED_PARAMS_MD5 = {
    # frozen delivery point (belongs to trial_0474) -- must never change
    "manu/evaluation/system_sota_longquan_delivery_params.json": "0ca8267f4ad340df11b3aa7d679d9068",
    # v1 / v2 historical points -- must never change either (a new point gets a new file)
    "manu/evaluation/system_sota_longquan_trial0080_delivery_params.json": "d40559902b86920ebbf8a1f1f363b64f",
    "manu/evaluation/system_sota_longquan_trial0080_delivery_v2_params.json": "15af89e67bf2cddc54cd51cb8ed5b214",
    # v3 -- the delivered point
    "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json": "b20de759599d96aae9493b93ce6b61e8",
}

EXPECTED_SINGLE_FRAME = {  # single_frame_all.json (frozen single-frame criterion)
    "tp": 124852, "fp": 12247, "fn": 8405, "gt": 133257,
    "f1": 0.9236118303274201, "recall": 0.9369263903584802, "precision": 0.9106703914689385,
}

EXPECTED_TRACK = {  # track_all_*.json
    "track_all.json": {"tp": 124119, "fp": 6687, "fn": 9138, "f1": 0.940071, "far": 0.040473},
    "track_all_th045.json": {"tp": 123383, "fp": 4790, "fn": 9874, "f1": 0.943909, "far": 0.028991},
    "track_all_v2.json": {"tp": 123384, "fp": 4165, "fn": 9873, "f1": 0.946175, "far": 0.025208},
    "track_all_v3.json": {"tp": 123052, "fp": 3440, "fn": 10205, "f1": 0.947469, "far": 0.020820},
}

EXPECTED_V3_PARAMS = {"th_base": 0.48, "th_ground": 0.26, "sky_ratio": 1.0, "confirm_consec": 7}

# OSD directories: name -> (pass, un-pass, sum of the mp4 file sizes)
EXPECTED_OSD = {
    "osd_sys_sota_20261020": (17, 7, 5804497704),
    "osd_sys_sota_trial0080_20261022": (21, 3, 5814432296),
    "osd_sys_sota_trial0080_th045_20261022": (20, 4, 5804540094),
    "osd_sys_sota_trial0080_v2_20261022": (21, 3, 5798916597),
    "osd_sys_sota_trial0080_v3_20261022": (21, 3, 5790776836),
}

OLD_OSD_NAME = "osd_sys_sota_20261020"
OLD_OSD_MTIME_PREFIX = "2026-10-20"

# The top-level quick-reference card.  It lives at the repository root, so a reader who
# only scans the project root still finds the delivery instead of wandering into the
# training-time gate cache.  It is listed in EXPECTED_REPO_FILES so that it is covered by
# the three-way md5 comparison like every other artefact of this work.
ROOT_QUICKREF = "2026-10-22_trial0080_longquan_delivery.md"

EXPECTED_REPO_FILES = [
    "manu/evaluation/probe_far_threshold_tradeoff.py",
    "manu/evaluation/loo_working_point_check.py",
    "manu/evaluation/analyze_track_structure.py",
    "manu/evaluation/verify_delivery_evidence.py",
    "manu/evaluation/run_trial0080_longquan_eval.sh",
    "manu/evaluation/system_sota_longquan_singleframe_params.json",
    "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json",
    "manu/data/audit_cache_provenance.py",
    "manu/data/multiscale_heatmap_cache.py",
    "manu/postprocess/tune_sys_sota_longquan.py",
    "manu/videos/render_trial0080_longquan_osd.sh",
    "manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md",
    "manu/memory/trial0080_longquan_DELIVERY_README.md",
    "manu/memory/trial0080_EVALUATION_DELIVERABLES.md",
    "manu/memory/trial0080_gate_DO_NOT_USE.md",
    ROOT_QUICKREF,
]

# The human-readable cards that quote the headline numbers.  Their quoted values are
# compared against the artefacts themselves (see section 9), so a card can never drift
# away from the measured result without the verification failing.
DELIVERY_CARDS = [
    ROOT_QUICKREF,
    "manu/memory/trial0080_longquan_DELIVERY_README.md",
    "manu/memory/trial0080_EVALUATION_DELIVERABLES.md",
]

# Entry points dropped into the experiment tree so that anybody landing in the trial
# directory is pointed at the business-domain evaluation output instead of the
# training-time gate cache (8 sequences, different feature semantics).
TRIAL_ENTRY_POINTS = {
    "EVALUATION_DELIVERABLES.md": "pointer at the trial root -> where the Longquanshan delivery lives",
    "gate/DO_NOT_USE_FOR_LQ_EVAL.md": "warning inside gate/ -> that cache is NOT the business-domain evaluation",
    "lq_eval/DELIVERY_README.md": "human readable delivery guide inside the evaluation directory",
    "lq_eval/DELIVERY_MANIFEST.json": "machine readable delivery manifest inside the evaluation directory",
}

TRIAL_DIR = "runs/optuna_mix_ratio/trials/trial_0080"
EVAL_DIR = f"{TRIAL_DIR}/lq_eval"


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"[{mark}] {label}" + (f"  --  {detail}" if detail else ""))
        return ok


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _NumpyPlaceholder:
    """Stand-in for a numpy array/dtype so the meta block can be read without numpy."""

    def __init__(self, *args, **kwargs) -> None:
        self.state = None

    def __setstate__(self, state) -> None:
        self.state = state


def _numpy_placeholder_factory(*args, **kwargs) -> _NumpyPlaceholder:
    return _NumpyPlaceholder()


class _MetaOnlyUnpickler(pickle.Unpickler):
    """Unpickle a cache file while replacing every numpy class by a placeholder.

    The meta block of a heatmap cache holds plain Python values, so the arrays around it
    never have to be reconstructed -- which keeps this script runnable on a machine that
    has no numpy at all (the meta is the only thing being checked here).
    """

    def find_class(self, module: str, name: str):  # noqa: D102 - pickle protocol
        if module.startswith("numpy"):
            return _numpy_placeholder_factory
        return super().find_class(module, name)


def read_cache_meta(path: Path) -> dict:
    with open(path, "rb") as handle:
        return _MetaOnlyUnpickler(handle).load()["meta"]


def build_manifest(repo: Path, project: Path, data: Path, checker: "Checker") -> dict:
    """Build a machine-readable manifest that maps every claim to its artefact."""
    import datetime

    eval_dir = project / EVAL_DIR
    osd_root = data / "siping/datasets/manu/龙泉山"

    def entry(role: str, path: Path, server_path: str, facts: dict | None = None) -> dict:
        item = {"role": role, "path": str(path), "server_path": server_path}
        if path.is_file():
            item["md5"] = md5_of(path)
            item["size_bytes"] = path.stat().st_size
        if facts:
            item["facts"] = facts
        return item

    deliverables = []
    # --- parameter files -------------------------------------------------------------
    param_roles = {
        "manu/evaluation/system_sota_longquan_delivery_params.json":
            ("frozen delivery point (trial_0474) - MUST NOT change", None),
        "manu/evaluation/system_sota_longquan_trial0080_delivery_params.json":
            ("v1 working point (th_base=0.45) - historical", None),
        "manu/evaluation/system_sota_longquan_trial0080_delivery_v2_params.json":
            ("v2 working point (0.42/0.26/7) - historical", None),
        "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json":
            ("v3 DELIVERED working point (0.48/0.26/7 + sky_ratio=1.0)", None),
        "manu/evaluation/system_sota_longquan_singleframe_params.json":
            ("single-frame arm parameters (th=0.22 / radius=768 / inbox)", None),
    }
    for rel, (role, _) in param_roles.items():
        path = repo / rel
        facts = {}
        if path.is_file():
            params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
            facts = {key: params.get(key) for key in
                     ("th_base", "th_ground", "sky_ratio", "confirm_consec", "dedup_radius")}
        deliverables.append(entry(role, path, f"/tmp/pycharm_project_10ae9e2e/{rel}", facts))

    # --- evaluation reports ----------------------------------------------------------
    report_roles = {
        "single_frame_all.json": "single-frame arm report (frozen criterion)",
        "track_all.json": "track arm @ frozen point th_base=0.30 (FAR misses the bound)",
        "track_all_th045.json": "track arm @ v1 (th_base=0.45)",
        "track_all_v2.json": "track arm @ v2 (0.42/0.26/7)",
        "track_all_v3.json": "track arm @ v3 DELIVERED (0.48/0.26/7, sky_ratio=1.0)",
        "far_tradeoff_probe.json": "single-axis th_base sweep (5 candidates)",
        "robustness_grid.json": "3-axis grid: th_base x confirm_consec x th_ground (54 candidates)",
        "sky_ratio_grid.json": "4th-axis grid: sky_ratio x th_base (28 candidates)",
        "baseline_recheck.json": "frozen baseline re-run through the same probe code path",
        "loo_check.json": "leave-one-sequence-out check of the v2 point",
        "loo_check_v3.json": "leave-one-sequence-out check of the v3 point",
    }
    for name, role in report_roles.items():
        path = eval_dir / name
        facts = None
        if path.is_file():
            overall = json.loads(path.read_text(encoding="utf-8")).get("overall")
            if overall:
                facts = {key: overall.get(key) for key in
                         ("tp", "fp", "fn", "gt", "f1", "recall", "precision", "far_per_frame")}
        deliverables.append(entry(role, path, f"/tmp/pycharm_project_10ae9e2e/{EVAL_DIR}/{name}", facts))

    # --- caches and predictions --------------------------------------------------------
    deliverables.append(entry(
        "multiscale heatmap cache (24 sequences, FPV features read from disk)",
        eval_dir / "ms_cache", f"/tmp/pycharm_project_10ae9e2e/{EVAL_DIR}/ms_cache",
        {"pkl_files": len(list((eval_dir / "ms_cache").glob("*.pkl"))) if (eval_dir / "ms_cache").is_dir() else 0}))
    for name in ("preds_single", "preds_track", "preds_track_th045", "preds_track_v2", "preds_track_v3"):
        directory = eval_dir / name
        deliverables.append(entry(
            f"per-sequence predictions for the OSD renderer ({name})",
            directory, f"/tmp/pycharm_project_10ae9e2e/{EVAL_DIR}/{name}",
            {"json_files": len(list(directory.glob("*.json"))) if directory.is_dir() else 0}))

    # --- OSD sets -----------------------------------------------------------------------
    for name, (exp_pass, exp_unpass, exp_bytes) in EXPECTED_OSD.items():
        directory = osd_root / name
        count, total, zero = file_bytes_sum(directory) if directory.is_dir() else (0, 0, 0)
        deliverables.append(entry(
            f"OSD review videos ({name})",
            directory, f"/mnt/data/siping/datasets/manu/龙泉山/{name}",
            {"pass": exp_pass, "un_pass": exp_unpass, "mp4_total_including_preview": count,
             "file_bytes_sum": total, "zero_byte_files": zero}))

    # --- code artefacts ------------------------------------------------------------------
    for rel in EXPECTED_REPO_FILES:
        role = ("top-level quick reference -> the whole delivery on one page"
                if rel == ROOT_QUICKREF else "code / documentation artefact")
        deliverables.append(entry(role, repo / rel,
                                  f"/tmp/pycharm_project_10ae9e2e/{rel}"))

    # --- entry points inside the experiment tree ------------------------------------------
    trial_dir = project / TRIAL_DIR
    for rel, role in TRIAL_ENTRY_POINTS.items():
        deliverables.append(entry(f"entry point in the experiment tree ({role})",
                                  trial_dir / rel, f"/tmp/pycharm_project_10ae9e2e/{TRIAL_DIR}/{rel}"))

    v3_report = eval_dir / "track_all_v3.json"
    criteria = None
    if v3_report.is_file():
        overall = json.loads(v3_report.read_text(encoding="utf-8"))["overall"]
        tp, fp, fn, gt = overall["tp"], overall["fp"], overall["fn"], overall["gt"]
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        far = fp / 165223
        criteria = {
            "measured_on": "track_all_v3.json",
            "f1": round(f1, 6), "recall": round(recall, 6),
            "precision": round(precision, 6), "far_per_frame": round(far, 6),
            "tp": tp, "fp": fp, "fn": fn, "gt": gt, "frames": 165223,
            "thresholds": {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.030},
            "all_three_met": bool(f1 >= 0.88 and recall >= 0.85 and far <= 0.030),
            "far_margin_pct": round((0.030 - far) / 0.030 * 100, 2),
        }

    return {
        "schema": "trial0080-longquan-delivery-manifest/1",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "goal": "用 trial_0080 best model 重跑龙泉山数据集，交付单帧性能、带 track 性能与全量 OSD 视频",
        "execution": {
            "server": "huangzhe@192.168.99.40 -p 32222",
            "project_dir": "/tmp/pycharm_project_10ae9e2e",
            "conda_env": "uav",
            "trial_dir": f"/tmp/pycharm_project_10ae9e2e/{TRIAL_DIR}",
            "model": f"{TRIAL_DIR}/weights/epoch3.pt",
            "feature_source": "/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train (FPV chained-affine, read from disk - not recomputed)",
        },
        "local_mounts": {
            "repo": str(repo),
            "project": str(project),
            "data": str(data),
            "note": "the data mount is the server's /mnt/data; the OSD sets live under "
                    "siping/datasets/manu/龙泉山/ there",
        },
        "delivery_criteria_measured": criteria,
        "delivered_working_point": EXPECTED_V3_PARAMS,
        "how_to_reverify": {
            "command": "python manu/evaluation/verify_delivery_evidence.py --server huangzhe@192.168.99.40",
            "expected": "PASS 80 / FAIL 0",
            "checks": "parameter md5 / evaluation reports / criteria recomputed from raw counts / "
                      "cache meta / OSD counts and byte sums / old OSD mtimes / three-way md5 / "
                      "figures quoted on the delivery cards vs the measured artefacts",
        },
        "notes": [
            "The OSD byte figure is the sum of the mp4 FILE sizes; `du -sb` also counts directory "
            "entries and therefore differs between the native filesystem and an SSHFS mount.",
            "The 2026-10-20 OSD set carries one extra clip under preview/; the '24 videos' figure "
            "counts pass/ + un-pass/ only.",
        ],
        "verification_summary": {"passed": checker.passed, "failed": checker.failed},
        "deliverables": deliverables,
    }


def file_bytes_sum(directory: Path, pattern: str = "*.mp4") -> tuple[int, int, int]:
    """Return (count, total file bytes, zero-byte file count) for the matching files."""
    total = 0
    count = 0
    zero = 0
    for path in sorted(directory.rglob(pattern)):
        size = path.stat().st_size
        count += 1
        total += size
        if size == 0:
            zero += 1
    return count, total, zero


# --------------------------------------------------------------------------------------
# Section 9 -- the numbers quoted on the human-readable cards must match the artefacts.
#
# A delivery card is only useful if it agrees with what was actually measured.  These
# checks read the quoted figures back out of the markdown and compare them with the
# evaluation reports, so a stale card fails the verification instead of misleading the
# next reader.
# --------------------------------------------------------------------------------------

# Which quoted figure maps to which measured value.  Each entry is
# (regex with one capture group, the expected value as a string).
CARD_FIGURES = {
    "single_frame_f1": (r"单帧 F1\s*\|[^|]*\|\s*\*\*([\d.]+)%\*\*", "92.361"),
    "track_f1": (r"带 track F1\s*\|[^|]*\|\s*\*\*([\d.]+)%\*\*", "94.747"),
    "track_recall": (r"带 track Recall\s*\|[^|]*\|\s*\*\*([\d.]+)%\*\*", "92.342"),
    "far": (r"FAR / frame\s*\|[^|]*\|\s*\*\*([\d.]+)\*\*", "0.0208"),
    "track_fp": (r"\| FP\s*\|[^|]*\|\s*\*\*([\d,]+)\*\*", "3,440"),
    "v3_params_md5": (r"md5 `([0-9a-f]{32})`", "b20de759599d96aae9493b93ce6b61e8"),
    "v3_working_point": (
        r"`(th_base=[\d.]+ / th_ground=[\d.]+ / sky_ratio=[\d.]+ / confirm_consec=\d+)`",
        "th_base=0.48 / th_ground=0.26 / sky_ratio=1.0 / confirm_consec=7",
    ),
}


def check_card_figures(card_path: Path, checker: "Checker") -> None:
    """Check that every headline figure quoted on a delivery card matches the artefacts."""
    if not card_path.is_file():
        checker.check(False, str(card_path), "交付卡片缺失")
        return
    text = card_path.read_text(encoding="utf-8")
    for label, (pattern, expected) in CARD_FIGURES.items():
        match = re.search(pattern, text)
        if not match:
            checker.check(False, f"{card_path.name} · {label}", "卡片上找不到该数值")
            continue
        quoted = match.group(1)
        checker.check(quoted == expected, f"{card_path.name} · {label}",
                      f"卡片写 {quoted}，实测 {expected}")


def check_card_paths(card_path: Path, project: Path, data: Path, checker: "Checker") -> None:
    """Check that the paths a card points at actually exist on the mounts."""
    if not card_path.is_file():
        return
    text = card_path.read_text(encoding="utf-8")
    # Server-side absolute paths -> the data mount; experiment-tree paths -> project mount.
    data_paths = sorted(set(re.findall(r"`(/mnt/data/siping/datasets/manu/龙泉山/[^`]*)`", text)))
    project_paths = sorted(set(re.findall(
        r"`(runs/optuna_mix_ratio/trials/trial_0080/[^`]*?)`", text)))
    for raw in data_paths:
        rel = raw[len("/mnt/data/"):].rstrip("/")
        target = data / rel
        checker.check(target.exists(), f"{card_path.name} · {raw}",
                      "存在" if target.exists() else "路径不存在")
    for raw in project_paths:
        rel = raw.rstrip("/")
        target = project / rel
        checker.check(target.exists(), f"{card_path.name} · {raw}",
                      "存在" if target.exists() else "路径不存在")


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-verify the trial_0080 Longquanshan delivery evidence")
    parser.add_argument("--repo-root", default="", help="Local repository root (default: derived from this file)")
    parser.add_argument("--project-mount", default="/home/manu/mnt/pycharm_project_10ae9e2e",
                        help="SSHFS mount of the server project directory")
    parser.add_argument("--data-mount", default="/home/manu/mnt/data",
                        help="SSHFS mount of the server /mnt/data volume")
    parser.add_argument("--server", default="", help="Optional 'user@host' to also compare md5 over ssh")
    parser.add_argument("--emit-manifest", default="",
                        help="Write a machine-readable evidence manifest to this path")
    args = parser.parse_args()

    repo = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    project = Path(args.project_mount)
    data = Path(args.data_mount)
    checker = Checker()

    print("=" * 100)
    print("trial_0080 龙泉山交付证据复验")
    print(f"  本地仓库    : {repo}")
    print(f"  项目挂载    : {project}")
    print(f"  数据挂载    : {data}")
    print("=" * 100)

    # ---- 1. parameter files -----------------------------------------------------------
    print("\n--- 1. 交付参数档（冻结/v1/v2 必须原样，v3 必须是交付点）---")
    for rel, expected in EXPECTED_PARAMS_MD5.items():
        path = repo / rel
        if not path.is_file():
            checker.check(False, rel, "文件缺失")
            continue
        actual = md5_of(path)
        checker.check(actual == expected, rel, f"md5={actual}" + ("" if actual == expected else f" 期望 {expected}"))
    v3_path = repo / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
    if v3_path.is_file():
        params = json.loads(v3_path.read_text(encoding="utf-8")).get("params", {})
        actual = {key: params.get(key) for key in EXPECTED_V3_PARAMS}
        checker.check(actual == EXPECTED_V3_PARAMS, "v3 参数值", str(actual))

    # ---- 2. evaluation reports --------------------------------------------------------
    print("\n--- 2. 评估产物与指标 ---")
    eval_dir = project / EVAL_DIR
    single = eval_dir / "single_frame_all.json"
    if single.is_file():
        data_single = json.loads(single.read_text(encoding="utf-8"))
        overall = data_single["overall"]
        ok = all(overall.get(key) == value for key, value in EXPECTED_SINGLE_FRAME.items()
                 if isinstance(EXPECTED_SINGLE_FRAME[key], int))
        ok = ok and all(abs(overall[key] - EXPECTED_SINGLE_FRAME[key]) < 1e-12
                        for key in ("f1", "recall", "precision"))
        checker.check(ok, "单帧臂 single_frame_all.json",
                      f"F1={overall['f1'] * 100:.4f}%  R={overall['recall'] * 100:.3f}%  "
                      f"P={overall['precision'] * 100:.3f}%  FP={overall['fp']}  FN={overall['fn']}")
    else:
        checker.check(False, "单帧臂 single_frame_all.json", "文件缺失")

    for name, expected in EXPECTED_TRACK.items():
        path = eval_dir / name
        if not path.is_file():
            checker.check(False, name, "文件缺失")
            continue
        overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
        ok = (overall["tp"] == expected["tp"] and overall["fp"] == expected["fp"]
              and overall["fn"] == expected["fn"]
              and abs(overall["f1"] - expected["f1"]) < 1e-6
              and abs(overall["far_per_frame"] - expected["far"]) < 1e-6)
        checker.check(ok, name, f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
                               f"F1={overall['f1'] * 100:.4f}% FAR={overall['far_per_frame']:.6f}")

    # ---- 3. delivery criterion on the v3 report ---------------------------------------
    print("\n--- 3. v3 三条交付判据（从 raw counts 重算）---")
    v3_report = eval_dir / "track_all_v3.json"
    if v3_report.is_file():
        overall = json.loads(v3_report.read_text(encoding="utf-8"))["overall"]
        tp, fp, fn, gt = overall["tp"], overall["fp"], overall["fn"], overall["gt"]
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        far = fp / 165223
        checker.check(f1 >= 0.88, "F1 >= 88%", f"{f1 * 100:.4f}%")
        checker.check(recall >= 0.85, "Recall >= 85%", f"{recall * 100:.4f}%")
        checker.check(far <= 0.030, "FAR <= 0.030/frame", f"{far:.6f}  (裕度 {(0.030 - far) / 0.030 * 100:.1f}%)")
        checker.check(gt == 133257, "GT 分母 133257", str(gt))
    else:
        checker.check(False, "track_all_v3.json", "文件缺失")

    # ---- 4. heatmap cache -------------------------------------------------------------
    print("\n--- 4. 热图缓存 ---")
    cache_dir = eval_dir / "ms_cache"
    pkl = sorted(cache_dir.glob("*.pkl")) if cache_dir.is_dir() else []
    checker.check(len(pkl) == 24, "缓存 pkl 24 个", str(len(pkl)))
    if pkl:
        metas = []
        for path in pkl:
            metas.append(read_cache_meta(path))
        keys = ("scales", "region_threshold", "main_threshold", "fusion_dilate", "min_area", "use_p0_highway")
        consistent = len({tuple(repr(m.get(k)) for k in keys) for m in metas}) == 1
        checker.check(consistent, "24 条缓存 meta 构建参数一致",
                      f"scales={metas[0].get('scales')} frames={sum(m.get('frames', 0) for m in metas)}")

    # ---- 5. OSD sets ------------------------------------------------------------------
    print("\n--- 5. OSD 五档（文件字节和 = 挂载无关口径）---")
    for name, (exp_pass, exp_unpass, exp_bytes) in EXPECTED_OSD.items():
        directory = data / "siping/datasets/manu/龙泉山" / name
        if not directory.is_dir():
            checker.check(False, name, "目录缺失")
            continue
        n_pass = len(list((directory / "pass").glob("*.mp4")))
        n_unpass = len(list((directory / "un-pass").glob("*.mp4")))
        count, total, zero = file_bytes_sum(directory)
        # The deliverable set is pass/ + un-pass/.  The 2026-10-20 set additionally carries one
        # preview clip under preview/, which is part of that set but is not a sequence video,
        # so the "24 videos" check counts the two delivery folders only.
        delivery_count = n_pass + n_unpass
        checker.check(n_pass == exp_pass and n_unpass == exp_unpass,
                      f"{name} pass/un-pass", f"{n_pass}/{n_unpass}  (期望 {exp_pass}/{exp_unpass})")
        checker.check(delivery_count == 24 and zero == 0,
                      f"{name} 交付视频 24 个且无零字节",
                      f"pass+un-pass={delivery_count} 全部 mp4={count}（含 preview/）zero={zero}")
        checker.check(total == exp_bytes,
                      f"{name} 文件字节和", f"{total}  (期望 {exp_bytes})")

    # ---- 6. the old OSD was not touched ------------------------------------------------
    print("\n--- 6. 旧 OSD 未被改写 ---")
    old_dir = data / "siping/datasets/manu/龙泉山" / OLD_OSD_NAME
    if old_dir.is_dir():
        mtimes = {p.stat().st_mtime for p in old_dir.rglob("*.mp4")}
        import datetime
        newest = max(datetime.datetime.fromtimestamp(m).strftime("%Y-%m-%d") for m in mtimes)
        oldest = min(datetime.datetime.fromtimestamp(m).strftime("%Y-%m-%d") for m in mtimes)
        checker.check(newest <= OLD_OSD_MTIME_PREFIX,
                      "旧 OSD 全部 mp4 的 mtime 仍停在 2026-10-20", f"范围 {oldest} ~ {newest}")
    else:
        checker.check(False, "旧 OSD 目录", "缺失")

    # ---- 7. three-way md5 --------------------------------------------------------------
    print("\n--- 7. 三侧 md5（本地 / SSHFS 镜像 / 服务器）---")
    for rel in EXPECTED_REPO_FILES:
        local = repo / rel
        mirror = project / rel
        if not local.is_file() or not mirror.is_file():
            checker.check(False, rel, "本地或镜像缺文件")
            continue
        local_md5 = md5_of(local)
        mirror_md5 = md5_of(mirror)
        ok = local_md5 == mirror_md5
        detail = f"{local_md5}" + ("" if ok else f"  镜像 {mirror_md5}")
        if ok and args.server:
            try:
                out = subprocess.run(
                    ["ssh", "-p", "32222", args.server, f"md5sum '/tmp/pycharm_project_10ae9e2e/{rel}'"],
                    capture_output=True, text=True, timeout=60, check=False)
                # md5sum prints "<hash>  <path>" -- the hash is the FIRST field.
                fields = out.stdout.strip().split()
                server_md5 = fields[0] if fields else ""
                ok = server_md5 == local_md5
                detail += f"  服务器 {server_md5}" if server_md5 else "  服务器未取到"
            except Exception as exc:  # noqa: BLE001 - the ssh check is optional
                detail += f"  (ssh 不可用: {exc})"
        checker.check(ok, rel, detail)

    # ---- 8. entry points inside the experiment tree ---------------------------------------
    print("\n--- 8. 实验树内的入口文件（防止落错目录）---")
    trial_dir = project / TRIAL_DIR
    for rel, role in TRIAL_ENTRY_POINTS.items():
        path = trial_dir / rel
        checker.check(path.is_file() and path.stat().st_size > 0,
                      f"{TRIAL_DIR}/{rel}", role if path.is_file() else "缺失")

    # ---- 9. the delivery cards must quote the measured numbers ---------------------------
    print("\n--- 9. 交付卡片上的数字与路径 vs 实测产物 ---")
    for rel in DELIVERY_CARDS:
        check_card_figures(repo / rel, checker)
    for rel in DELIVERY_CARDS:
        check_card_paths(repo / rel, project, data, checker)

    # ---- summary -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"结果：PASS {checker.passed} / FAIL {checker.failed}")
    print("=" * 100)

    if args.emit_manifest:
        manifest = build_manifest(repo, project, data, checker)
        out = Path(args.emit_manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest -> {out.resolve()}  ({len(manifest['deliverables'])} 条交付物)")

    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
