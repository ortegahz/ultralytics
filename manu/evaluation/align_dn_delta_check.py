#!/usr/bin/env python3
"""C++ <-> Python 时空融合的 **delta-n（跳帧）口径** 对齐闸门.

为什么需要这个脚本
------------------
2026-11-19 留下的 `manu/memory/cadence_ab_20261119/vid09_align/` 只有 **delta-n == 1**
的候选流（190 个帧间隔全是 1），因此它证明的是「逐帧不跳」时 C++ 与 Python 一致。
但交付参数组 `cc2_md10` 的**全部意义**就是跳帧工况（板卡实测 delta-n 均值 ~3.1），
`trial0080_board_fusion_preset_20261119.md` §2.1 也点明最大的风险是
「物理帧号差」vs「更新次数」两种口径。

本脚本补上这个洞：在**同一份候选流**上按多种 delta-n 图案抽帧，
用各参数组分别跑

  * C++   : `mosaic::TrackFusionEngine`（`rk3588_app_base/mosaic_sdk/src/npu/track_fusion_align_test.cpp`）
  * Python: `manu/evaluation/sdk_track_fusion_ref.py --physical-frame-index`

然后 **逐字节** 比对两份输出。任何不一致都会被定位到第一处不同的帧/字段。

用法
----
    python align_dn_delta_check.py \
        --cands   <candidates.tsv> \
        --cpp-bin <track_fusion_align_test> \
        --workdir /tmp/align_dn \
        --params delivery=manu/evaluation/xxx_params.json \
        [--dedup-radius 20]

退出码: 0 = 全部逐字节一致；1 = 有不一致；2 = 用法/环境错误。

跨尺度候选合并
--------------
`--dedup-radius R` 会**同时**打开两侧的「候选构造阶段跨尺度合并」：

  * C++   : `track_fusion_align_test --dedup-radius R`（= `mosaic::MergeCandidatesByRadius`）
  * Python: `sdk_track_fusion_ref.py --dedup-radius R`（= 同规则的 Python 镜像）

`dedup_radius` **不是**融合引擎的参数，而是候选构造参数，所以它**不进**引擎的 `--params`
文件：C++ 的 `LoadParams()` 会用参数文件里的值覆盖掉命令行传的半径，两侧就会走不同的
路径。参数 JSON 里带的 `dedup_radius`（服务器交付档是 768，板端的正确值是 16~24）默认
**不采用**，要显式传 `--dedup-radius`，或用 `--dedup-radius-from-params` 明确表示采用。
R = 0 表示关闭这一步 ⇒ 与本脚本引入前的行为逐字节一致。
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# C++ 侧 `LoadParams()` 认识的全部键（见 mosaic_track_fusion.h 的 TrackFusionParams
# 与 track_fusion_align_test.cpp 的 LoadParams）。不在表里的键 C++ 会打印
# "[warn] unknown param key" 并 **忽略** —— 那就是一条真实的移植缺口。
CPP_KNOWN_KEYS = frozenset(
    {
        "th_base", "th_salvage", "th_ground", "sky_ratio", "img_h",
        "max_age", "min_hits", "match_dist", "max_match_dist",
        "min_track_score", "instant_conf", "min_displacement",
        "th_deep", "min_hits_deep",
        "stitch_gap", "stitch_max_dist",
        "infill_gap", "min_hits_infill",
        "min_rigid_disp", "max_rigid_var", "min_hits_prune", "min_rigid_disp_sky",
        "confirm_consec", "confirm_gap_tol", "confirm_trim_prefix",
        "max_age_phys", "r_slope", "alpha_decay",
        # `dedup_radius` 是**候选构造阶段**的跨尺度合并半径，不是引擎参数：
        # C++ 的 `track_fusion_align_test` 用 `--dedup-radius` 单独传（`LoadParams` 也认这个
        # 键，但参数文件里的值会覆盖命令行，所以本脚本一律走命令行）。见 `DEDUP_KEY`。
        "dedup_radius",
    }
)

# 候选构造阶段的跨尺度合并半径（在引擎参数文件之外单独传递）。
DEDUP_KEY = "dedup_radius"

# Python 侧 `OnlineAdaptiveTracker` 的缺省值（eval_bidirectional_track_fusion.py）。
# JSON 里没有的键，Python 用这里的值；C++ 的头文件缺省值 **可能不同**（例如
# r_slope：Python 缺省 0.0，C++ 头文件缺省 5.0）⇒ 落盘时必须显式写死。
PY_DEFAULTS = {
    "max_age_phys": None,
    "r_slope": 0.0,
    "alpha_decay": 1.0,
    "min_displacement": 2.5,
    "max_age": 3,
    "min_hits": 3,
    "max_match_dist": 24.0,
    "min_track_score": 0.08,
    "instant_conf": 0.25,
    "th_deep": 0.0,
    "min_hits_deep": 3,
    "stitch_gap": 6,
    "stitch_max_dist": 25.0,
    "infill_gap": 3,
    "min_hits_infill": 5,
    "min_rigid_disp": 2.0,
    "max_rigid_var": 1.0,
    "min_hits_prune": 4,
    "min_rigid_disp_sky": -1.0,
    "confirm_trim_prefix": 1,
}

# Python 的 None（=「不设」）在 C++ 侧的等价写法。C++ 的 max_age_phys < 0 才是
# 「关闭」；写 0 会让航迹下一帧就死（mosaic_track_fusion.h:77-79）。
NONE_TO_CPP = {"max_age_phys": -1, "min_rigid_disp_sky": -1.0}

# Python 侧的键名与 C++ 侧不一致的别名（Python build_tracker_config 用 min_disp，
# C++ TrackFusionParams 用 min_displacement）。
PY_TO_CPP_ALIAS = {"min_disp": "min_displacement"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frames(path: Path) -> list[list[str]]:
    """读候选流，返回每帧的 token 列表（去掉帧号，便于重新编号）。"""
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            frames.append(parts[1:])
    return frames


def write_candidates(frames: list[list[str]], frame_ids: list[int], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for fid, body in zip(frame_ids, frames):
            count = int(body[0]) if body else 0
            tail = "".join(" " + v for v in body[1:]) if body else ""
            handle.write(f"{fid} {count}{tail}\n")


def dn_patterns(n_frames: int, real_deltas: list[int] | None) -> dict[str, list[tuple[int, int]]]:
    """构造各种 delta-n 图案。

    返回 {名称: [(物理帧号, 基础流下标), ...]}。
    * 物理帧号 = 引擎真正看到的相机帧号（**带跳空** ⇒ dt = 相邻物理帧号之差）
    * 基础流下标 = 从 v09_cands.tsv 取哪一帧的候选

    `dnN`      : 每 N 帧送一次（物理帧号 0, N, 2N, ... ⇒ dt 恒为 N）
    `dn_board_cadence` : 用板卡实测 cadence 的间隔序列（cad_on.tsv 的相邻 fid 之差）
                  依次累加出物理帧号，候选按顺序取基础流的前若干帧。
    """
    patterns: dict[str, list[tuple[int, int]]] = {
        "dn1": [(i, i) for i in range(n_frames)]
    }
    for step in (2, 3, 5, 7):
        patterns[f"dn{step}"] = [(k * step, k * step) for k in range(n_frames // step + 1)]
    if real_deltas:
        pairs: list[tuple[int, int]] = []
        cur = 0
        for k, delta in enumerate(real_deltas):
            if k >= n_frames:
                break
            pairs.append((cur, k))
            cur += delta
        patterns["dn_board_cadence"] = pairs
    return patterns


def load_cadence_deltas(cadence_tsv: Path) -> list[int]:
    """从板卡 cadence TSV（t_collect_ms / fid / n_dets）取相邻 fid 之差。"""
    fids = []
    with cadence_tsv.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                fids.append(int(parts[1]))
    return [b - a for a, b in zip(fids, fids[1:])]


def cpp_params_text(params: dict) -> tuple[str, list[str]]:
    """把 Python 侧的 params 字典转成 C++ 的 `key value` 参数文件文本。

    返回 (文本, C++ 不认识的键列表)。
    """
    merged = dict(PY_DEFAULTS)
    merged.update(params)
    # 跨尺度合并半径不是引擎参数：它必须走 `--dedup-radius` 命令行，不能进这个参数文件
    # （C++ 的 `LoadParams()` 会用文件里的值覆盖命令行的值，两侧就会走不同的路径）。
    merged.pop(DEDUP_KEY, None)
    for py_key, cpp_key in PY_TO_CPP_ALIAS.items():
        if py_key in merged and cpp_key not in params:
            merged[cpp_key] = merged[py_key]
    lines = ["# generated by align_dn_delta_check.py -- do not edit"]
    unknown = []
    for key in sorted(merged):
        value = merged[key]
        if value is None:
            if key in NONE_TO_CPP:
                value = NONE_TO_CPP[key]
            else:
                continue
        if key in PY_TO_CPP_ALIAS or key in PY_TO_CPP_ALIAS.values():
            continue  # 别名：Python 的键名与 C++ 的键名不同，已转换
        if key not in CPP_KNOWN_KEYS:
            unknown.append(key)
            continue
        if isinstance(value, bool):
            value = int(value)
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n", unknown


def split_dedup_radius(params: dict) -> tuple[dict, float | None]:
    """把候选构造参数 `dedup_radius` 从引擎参数里摘出来。

    返回 (引擎参数, 半径或 None)。半径是**候选构造阶段**的跨尺度合并半径，两侧都用命令行
    单独传（C++ `--dedup-radius` / Python `--dedup-radius`），所以它不能留在引擎的
    `--params` 文件里。
    """
    engine_params = dict(params)
    radius = engine_params.pop(DEDUP_KEY, None)
    return engine_params, (None if radius is None else float(radius))


def resolve_radius(args, params_radius: float | None, name: str) -> float:
    """决定这一组参数用多大的跨尺度合并半径（两侧必须完全一样）。

    * `--dedup-radius R` 显式给出 => 用它（命令行优先于 `--dedup-radius-from-params`）；
    * `--dedup-radius-from-params` => 用参数 JSON 里的 `dedup_radius`；
    * 都没有 => 0.0（关闭这一步，与本脚本引入前逐字节一致）。

    参数 JSON 里带的 `dedup_radius` 默认**不采用**并明确提示：服务器交付档的值是 768 px
    （板端不能用，见 mosaic_track_fusion.h 的注释），默默采用会让闸门跑在一个既不是服务器
    口径也不是板端口径的中间态上。
    """
    if args.dedup_radius is not None:
        if args.dedup_radius_from_params:
            print(f"[!] {name}: --dedup-radius 与 --dedup-radius-from-params 同时给出，命令行优先")
        return float(args.dedup_radius)
    if args.dedup_radius_from_params:
        if params_radius is None:
            print(f"[i] {name}: 参数组里没有 {DEDUP_KEY} => 跨尺度合并关闭")
            return 0.0
        return params_radius
    if params_radius is not None:
        print(
            f"[i] {name}: 参数组的 {DEDUP_KEY}={params_radius:g} 未采用"
            f"（候选构造参数，只走 --dedup-radius；服务器值 768 不是板端值）"
        )
    return 0.0


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800, check=False
    )


def first_diff(a: Path, b: Path) -> str:
    """定位两份 TSV 的第一处差异。"""
    with a.open("r", encoding="utf-8") as fa, b.open("r", encoding="utf-8") as fb:
        for lineno, (la, lb) in enumerate(zip(fa, fb), start=1):
            if la != lb:
                return f"line {lineno}: cpp={la.strip()!r} py={lb.strip()!r}"
    la = a.read_text().splitlines()
    lb = b.read_text().splitlines()
    if len(la) != len(lb):
        return f"length differs: cpp={len(la)} lines, py={len(lb)} lines"
    return "identical"


def main() -> int:
    parser = argparse.ArgumentParser(description="delta-n C++/Python alignment gate")
    parser.add_argument("--cands", required=True, type=Path, help="base candidates TSV")
    parser.add_argument("--cpp-bin", required=True, type=Path, help="track_fusion_align_test binary")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, help="python with numpy+scipy")
    parser.add_argument("--repo-root", default=None, help="ultralytics repo root (for PYTHONPATH)")
    parser.add_argument("--params", action="append", default=[], metavar="NAME=PATH",
                        help="参数组名=params.json 路径（可重复）")
    parser.add_argument("--cadence-tsv", default=None, type=Path, help="板卡 cadence TSV")
    parser.add_argument("--dedup-radius", type=float, default=None,
                        help="跨尺度候选合并半径(px)：同时传给 C++ (--dedup-radius) 与 Python "
                             "(sdk_track_fusion_ref.py --dedup-radius)；不传 = 关闭该步骤 "
                             "(与本脚本引入前逐字节一致)")
    parser.add_argument("--dedup-radius-from-params", action="store_true",
                        help="采用参数 JSON 里的 dedup_radius（默认不采用：服务器交付档的 768 "
                             "不是板端值）")
    args = parser.parse_args()

    if not args.cands.is_file():
        print(f"[ERROR] candidates not found: {args.cands}", file=sys.stderr)
        return 2
    if not args.cpp_bin.is_file():
        print(f"[ERROR] cpp binary not found: {args.cpp_bin}", file=sys.stderr)
        return 2

    repo_root = args.repo_root or str(Path(__file__).resolve().parents[2])
    ref_script = Path(repo_root) / "manu" / "evaluation" / "sdk_track_fusion_ref.py"
    if not ref_script.is_file():
        print(f"[ERROR] python reference not found: {ref_script}", file=sys.stderr)
        return 2

    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)

    frames = read_frames(args.cands)
    real_deltas = None
    if args.cadence_tsv and args.cadence_tsv.is_file():
        real_deltas = load_cadence_deltas(args.cadence_tsv)
        print(f"[i] 板卡 cadence: {len(real_deltas)} 个间隔, 均值 {sum(real_deltas)/len(real_deltas):.3f}")
    patterns = dn_patterns(len(frames), real_deltas)

    param_sets = []
    for spec in args.params:
        if "=" not in spec:
            print(f"[ERROR] bad --params spec: {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        param_sets.append((name, Path(path)))

    results = []
    for name, params_path in param_sets:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        params = params.get("params", params)
        params, params_radius = split_dedup_radius(params)
        radius = resolve_radius(args, params_radius, name)
        text, unknown = cpp_params_text(params)
        if unknown:
            print(f"[!] {name}: C++ 不实现这些参数（移植缺口候选）: {sorted(unknown)}")
        params_txt = work / f"params_{name}.txt"
        params_txt.write_text(text, encoding="utf-8")

        for pat_name, pairs in patterns.items():
            pairs = [(fid, k) for fid, k in pairs if k < len(frames)]
            frame_ids = [fid for fid, _ in pairs]
            cands = work / f"cands_{name}_{pat_name}.tsv"
            write_candidates([frames[k] for _, k in pairs], frame_ids, cands)

            out_cpp = work / f"fused_cpp_{name}_{pat_name}.tsv"
            out_py = work / f"fused_py_{name}_{pat_name}.tsv"

            r1 = run([str(args.cpp_bin), "--cands", str(cands), "--out", str(out_cpp),
                      "--params", str(params_txt), "--dedup-radius", repr(radius)])
            if r1.returncode != 0:
                print(f"[FAIL] {name}/{pat_name}: cpp rc={r1.returncode} {r1.stderr.strip()[:300]}")
                results.append((name, pat_name, False, "cpp failed", radius))
                continue

            env = dict(os.environ)
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
            r2 = run([args.python, str(ref_script), "--cands", str(cands), "--params", str(params_path),
                      "--out", str(out_py), "--physical-frame-index",
                      "--dedup-radius", repr(radius)], cwd=Path(repo_root), env=env)
            if r2.returncode != 0:
                print(f"[FAIL] {name}/{pat_name}: python rc={r2.returncode} {r2.stderr.strip()[:300]}")
                results.append((name, pat_name, False, "python failed", radius))
                continue

            # 审计行：Python 侧的合并统计（候选池进/出/丢弃）。每个参数组只打一次。
            if pat_name == next(iter(patterns)):
                for line in r2.stdout.splitlines():
                    if "[i] dedup" in line:
                        print(f"       {name:12s} {line.strip()}")

            same = filecmp.cmp(out_cpp, out_py, shallow=False)
            detail = "byte-identical" if same else first_diff(out_cpp, out_py)
            results.append((name, pat_name, same, detail, radius))
            print(f"[{'OK  ' if same else 'DIFF'}] {name:12s} {pat_name:16s} r={radius:<6g} "
                  f"frames={len(frame_ids):4d} {detail}")
            if not same:
                print(f"        cpp sha256={sha256(out_cpp)[:16]}  py sha256={sha256(out_py)[:16]}")

    print("\n===== 汇总 =====")
    n_ok = sum(1 for r in results if r[2])
    for name, pat_name, same, detail, radius in results:
        print(f"  {'PASS' if same else 'FAIL'}  {name:12s} {pat_name:16s} r={radius:<6g} {detail}")
    print(f"\n{n_ok}/{len(results)} 个 (参数组 x delta-n 图案) 组合逐字节一致")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
