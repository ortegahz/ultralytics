#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Audit every shipped mosaic-SDK artifact for the cross-scale candidate merge.

Why
---
The board's per-frame FAR was 12.8x over the red line (0.3832 vs 0.03) because the
SDK fed the track-fusion engine the **raw** candidate stream: one target that fired on
several scales became several tracks.  ``manu/pipeline/npu`` now implements the missing
step (``mosaic::MergeCandidatesByRadius`` + ``mosaic_pipeline_set_candidate_dedup_radius``)
and the C++ result is **bit-identical** to the server's Python mirror
(``manu/evaluation/board_cand_merge_verify.py``).

**A fixed source tree is not a fixed deliverable.**  This script answers the question
that actually matters for handover: *does any artifact a customer can install carry
the fix?*  It checks, per package tree:

  * the shipped binaries -- does ``libmosaic_pipeline.so`` export the new API, do the
    two probes carry the ``--cand-dedup-radius`` switch;
  * the shipped sources -- do they match the fixed sources in the repo;
  * the MANIFEST -- do its md5 rows still match the files on disk.

Re-run it after a rebuild: the ``contains_merge`` flags flip to true and the tree is
ready to ship.  Read-only, no board, no NPU, no training.

Usage
-----
    python manu/evaluation/artifact_audit_20261120.py \
        --out-json manu/memory/board_far_20261120/artifact_audit.json \
        --out-md   manu/memory/board_far_20261120/artifact_audit_20261120.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

WORKSPACE = Path("/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace")
REPO_SRC = WORKSPACE / "ultralytics/manu/pipeline/npu"

# The source of truth for the fix (repo tree) and the files that carry it.
MERGE_SOURCES = (
    "mosaic_track_fusion.cpp",
    "mosaic_track_fusion.h",
    "mosaic_pipeline.cpp",
    "mosaic_pipeline.h",
    "mosaic_track_dump.cpp",
    "mosaic_cadence_probe.cpp",
)

# (package tree, label).  "delivery" = what the customer gets; "staging" = the
# packaging script's DEST; "mirror" = the app-side copy of the package.
TREES: dict[str, Path] = {
    "delivery:mosaic_sdk_20261120": WORKSPACE / "build/mosaic_sdk_20261120",
    "delivery:mosaic_sdk_20261120_ocv48": WORKSPACE / "build/mosaic_sdk_20261120_ocv48",
    "delivery:mosaic_sdk_20261120_ocv42": WORKSPACE / "build/mosaic_sdk_20261120_ocv42",
    "staging:ultralytics/manu/pipeline/npu/build": REPO_SRC / "build",
    "mirror:rk3588_app_base/mosaic_sdk": WORKSPACE / "rk3588_app_base/mosaic_sdk",
}

BINARIES = ("out/libmosaic_pipeline.so", "out/mosaic_track_dump", "out/mosaic_cadence_probe")


def md5(path: Path) -> str:
    """Return the md5 of a file, or "" when it does not exist."""
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


def _run(cmd: list[str]) -> str:
    """Run a command and return stdout ("" on failure)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout


def binary_contains_merge(path: Path) -> dict[str, Any]:
    """Report whether a shipped binary carries the cross-scale candidate merge."""
    info: dict[str, Any] = {"exists": path.is_file()}
    if not path.is_file():
        info["contains_merge"] = False
        return info

    info["md5"] = md5(path)
    info["bytes"] = path.stat().st_size
    info["mtime"] = path.stat().st_mtime
    if path.suffix == ".so" or path.name.endswith(".so"):
        # The public C ABI entry point (the C++ helper is mangled, so check both).
        symbols = _run(["nm", "-D", "--defined-only", str(path)])
        info["has_api_symbol"] = "mosaic_pipeline_set_candidate_dedup_radius" in symbols
        info["has_cpp_symbol"] = "MergeCandidatesByRadius" in _run(["nm", "-C", "-D", str(path)])
    else:
        info["has_api_symbol"] = False
        info["has_cpp_symbol"] = False

    strings = _run(["strings", "-a", str(path)])
    info["has_cli_flag"] = "--cand-dedup-radius" in strings
    info["has_log_marker"] = "candidate dedup ON" in strings
    info["contains_merge"] = bool(
        info.get("has_api_symbol") or info.get("has_cpp_symbol") or info.get("has_cli_flag")
    )
    return info


def manifest_rows(manifest: Path) -> dict[str, str]:
    """Extract the (path -> md5) rows of a MANIFEST.md md5 table."""
    if not manifest.is_file():
        return {}
    text = manifest.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{32})` \|", text, re.M)
    return {path: digest for path, digest in rows}


def audit_tree(label: str, root: Path) -> dict[str, Any]:
    """Audit one package tree: binaries, sources and MANIFEST consistency."""
    tree: dict[str, Any] = {
        "label": label,
        "path": str(root),
        "exists": root.is_dir(),
        "binaries": {},
        "sources": {},
        "manifest": {},
    }
    if not root.is_dir():
        return tree

    for rel in BINARIES:
        tree["binaries"][rel] = binary_contains_merge(root / rel)

    for name in MERGE_SOURCES:
        shipped = root / "src/npu" / name
        repo = REPO_SRC / name
        shipped_md5, repo_md5 = md5(shipped), md5(repo)
        if not shipped.is_file():
            status = "missing"
        elif shipped_md5 == repo_md5:
            status = "match"
        else:
            status = "stale"
        tree["sources"][name] = {
            "shipped_md5": shipped_md5,
            "repo_md5": repo_md5,
            "status": status,
            "matches_repo": status == "match",
        }

    manifest = root / "MANIFEST.md"
    rows = manifest_rows(manifest)
    mismatched = sorted(p for p, digest in rows.items() if not (root / p).is_file() or md5(root / p) != digest)
    tree["manifest"] = {
        "path": str(manifest),
        "md5": md5(manifest),
        "rows": len(rows),
        "mismatched_rows": mismatched,
        "consistent": not mismatched,
    }
    return tree


def summarize(trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the per-tree audit to the numbers that decide the handover."""
    binaries = [b for t in trees for b in t["binaries"].values()]
    with_merge = [b for b in binaries if b.get("contains_merge")]
    stale_sources = [
        f"{t['label']}:{name}" for t in trees for name, s in t["sources"].items() if s["status"] != "match"
    ]
    return {
        "trees": len(trees),
        "binaries_checked": len(binaries),
        "binaries_with_merge": len(with_merge),
        "binaries_without_merge": len(binaries) - len(with_merge),
        "any_artifact_carries_merge": bool(with_merge),
        "stale_source_slots": len(stale_sources),
        "stale_source_slots_detail": stale_sources,
        "manifest_inconsistent_trees": [t["label"] for t in trees if t["manifest"] and not t["manifest"]["consistent"]],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    """Render the audit as a markdown report."""
    has_merge = audit["summary"]["any_artifact_carries_merge"]
    lines = [
        "# 交付件审计：跨尺度候选合并（cross-scale candidate merge）是否进了**二进制**",
        "",
        "> 由 `manu/evaluation/artifact_audit_20261120.py` 生成（只读，无板卡 / 无 NPU / 无训练）。",
        "> 判定依据：`libmosaic_pipeline.so` 是否导出 `mosaic_pipeline_set_candidate_dedup_radius`、",
        "> 两个探针是否带 `--cand-dedup-radius` 开关。",
        "",
        "## 结论",
        "",
        f"- 审计二进制 **{audit['summary']['binaries_checked']}** 个，"
        f"**含合并修复 {audit['summary']['binaries_with_merge']}** 个"
        f"（不含 {audit['summary']['binaries_without_merge']} 个）。",
        f"- **任何交付件带上修复了吗：{'是' if has_merge else '否'}**。",
        f"- 源码槽位与仓库修复版不一致的：**{audit['summary']['stale_source_slots']}** 处。",
        "",
        "## 逐包明细",
        "",
        "| 包 | 二进制 | md5 | 含合并修复 |",
        "|:--|:--|:--|:--|",
    ]
    for tree in audit["trees"]:
        for rel, info in tree["binaries"].items():
            lines.append(
                f"| {tree['label']} | `{rel}` | `{(info.get('md5') or '—')[:12]}` | "
                f"{'✅ 是' if info.get('contains_merge') else '❌ 否'} |"
            )
    lines += ["", "## 源码是否与仓库修复版一致", "", "| 包 | 文件 | 与仓库一致 |", "|:--|:--|:--|"]
    for tree in audit["trees"]:
        for name, info in tree["sources"].items():
            mark = {"match": "✅", "stale": "❌ 旧版", "missing": "❌ 缺失"}.get(info["status"], "?")
            lines.append(f"| {tree['label']} | `{name}` | {mark} |")
    lines += ["", "## MANIFEST 自洽性", "", "| 包 | md5 行数 | 不一致行 |", "|:--|--:|:--|"]
    for tree in audit["trees"]:
        man = tree["manifest"]
        if not man:
            continue
        bad = ", ".join(f"`{p}`" for p in man["mismatched_rows"]) or "—"
        lines.append(f"| {tree['label']} | {man['rows']} | {bad} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Audit every known package tree and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    out_dir = WORKSPACE / "ultralytics/manu/memory/board_far_20261120"
    parser.add_argument("--out-json", default=str(out_dir / "artifact_audit.json"))
    parser.add_argument("--out-md", default=str(out_dir / "artifact_audit_20261120.md"))
    args = parser.parse_args()

    trees = [audit_tree(label, root) for label, root in TREES.items()]
    audit = {
        "generated_by": "manu/evaluation/artifact_audit_20261120.py",
        "repo_source_of_truth": str(REPO_SRC),
        "summary": summarize(trees),
        "trees": trees,
    }
    outputs = (
        (args.out_json, json.dumps(audit, indent=2, ensure_ascii=False)),
        (args.out_md, render_markdown(audit)),
    )
    for path, payload in outputs:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + ("\n" if path.endswith(".md") else ""), encoding="utf-8")
        print(f"[OUT] {out}")

    summary = audit["summary"]
    print(f"二进制 {summary['binaries_checked']} 个；含修复 {summary['binaries_with_merge']} 个")
    print(f"任何交付件带上修复：{'是' if summary['any_artifact_carries_merge'] else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
