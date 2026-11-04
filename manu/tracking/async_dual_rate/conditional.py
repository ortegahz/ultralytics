"""条件化对比：退回重推演**到底有没有用**（臂 3 的正结论）。

全序列打平掩盖了一个正结论——在**重推演跑通**的帧上，臂 3 明显优于臂 1；
但**尝试重推演的帧里有 20% 跑不通**（824/4117，弃权），跑不通时比纯检测更差，
这正是"条件化对比"要拆开的机制。

⚠️ 口径修正：上面这两张条件化表只覆盖 **success + abstain** 两个子集，
它们**不足以**解释全序列——剩下的"未被条件化覆盖"的帧上，臂 3 的命中数
可能比臂 1 **更少**（旧口径把这些帧静默排除，"相抵才是全序列打平"的说法
在算术上并不成立）。真正的机制是"重推只在约 1/3 的帧上被触发"。
因此本模块同时给出两样东西：

* ``full_sequence_decomposition``：把"目标存在且两臂都有输出"的全序列 GT 帧
  切成**互斥**的四桶（``success`` / ``abstain`` / ``replay_fallback`` /
  ``uncovered``），逐桶给出两臂各自的 hits / n / 命中率；四桶相加 = 全序列，
  分母自洽、可用 ``sum_check`` 直接核对。``replay_fallback`` 桶收的是
  ``fusion_kind == "hard_reinit"`` 的帧（弃权后硬拉回兜底）——旧口径下这些帧
  既不在 success 也不在 abstain，被静默排除。
* ``z_value`` / ``p_value``：配对方向的**符号检验**（z = (better − worse) /
  √(better + worse)，双侧 p 值用**正态近似**，非精确二项检验），回答
  "臂 3 逐帧方向上到底是不是更好"。

本模块用**与验收表完全相同的官方口径**（``servo_metrics`` 的 GT 中心 + 有效位置）
重算这张条件化对照表，产出可复现的 JSON / Markdown。

⚠️ 口径提醒：``evidence_*.tsv`` 的 ``valid`` 列与"位置存在"**不等价**，
且 ``gt_x/gt_y`` 在无 GT 帧也会被写入 ⇒ **不要**用 TSV 手工复算本表。

用法::

    PYTHONPATH=. python -m manu.tracking.async_dual_rate.conditional \\
        --sequence VIDEO00007_19700101_001324 --out-dir runs/tracking_async
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .arms import JpegFrameSource, SlowStream, run_lockstep
from .servo_metrics import _gt_centre
from .three_arm import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_FRAMES_ROOT,
    DEFAULT_LABELS_ROOT,
    _arm_specs,
)
from .tracker import TrackerConfig

__all__ = ["run_conditional", "main"]


# --------------------------------------------------------------------------
# 口径：与 servo_metrics 完全一致
# --------------------------------------------------------------------------


def _positions(records: list[dict]) -> dict[int, np.ndarray]:
    """记录 → ``frame_index -> (x, y)``（位置为 NaN 或缺失的帧直接丢掉）。"""
    out: dict[int, np.ndarray] = {}
    for record in records:
        x = record.get("x", float("nan"))
        y = record.get("y", float("nan"))
        if math.isnan(x) or math.isnan(y):
            continue
        out[int(record["frame_index"])] = np.asarray([x, y], dtype=np.float64)
    return out


def _cle_map(
    positions: dict[int, np.ndarray], ground_truth: dict[int, np.ndarray]
) -> dict[int, float]:
    """``frame_index -> CLE``（只在**目标存在**的帧上评测，与验收表同口径）。"""
    out: dict[int, float] = {}
    for frame, boxes in ground_truth.items():
        position = positions.get(int(frame))
        if position is None:
            continue
        centre = _gt_centre(boxes)
        out[int(frame)] = float(np.linalg.norm(position - centre))
    return out


def _stats(values: list[float], tolerance: float = 5.0) -> dict:
    if not values:
        # 空帧集也返回**完整键集**——否则下游（render_markdown / verify_docs）会 KeyError。
        return {
            "n": 0,
            "cle_p50": float("nan"),
            "cle_p90": float("nan"),
            "cle_p95": float("nan"),
            "hit_rate_pct": float("nan"),
        }
    array = np.asarray(sorted(values), dtype=np.float64)
    return {
        "n": int(array.size),
        "cle_p50": float(np.percentile(array, 50)),
        "cle_p90": float(np.percentile(array, 90)),
        "cle_p95": float(np.percentile(array, 95)),
        "hit_rate_pct": float((array <= tolerance).mean() * 100.0),
    }


def _hit_stats(values: list[float], tolerance: float = 5.0) -> dict:
    """命中数 / 参与评测帧数 / 命中率（分桶分解用）。

    命中判据与 :func:`_stats` 完全一致（``CLE <= tolerance``），但只给**计数**
    字段：分桶分解要回答的是"命中数 + 分母"，不是分位数。

    分母 = ``n`` = 传入的帧数（**不是**序列总帧数）。
    """
    array = np.asarray(values, dtype=np.float64)
    hits = int((array <= tolerance).sum()) if array.size else 0
    return {
        "hits": hits,
        "n": int(array.size),
        "hit_rate_pct": float(hits / array.size * 100.0) if array.size else float("nan"),
    }


def _sign_test(better: int, worse: int) -> tuple[float, float]:
    """配对方向的符号检验：z 值与双侧 p 值。

    ``z = (better - worse) / sqrt(better + worse)``，其中 ``better`` /
    ``worse`` 是"臂 3 在这些帧上更好 / 更差"的计数（平局不参与）。
    ``z > 0`` ⇒ 臂 3 **逐帧方向上**占优。

    p 值用**正态近似**（``p = erfc(|z| / sqrt(2))``）——**不是**精确二项检验。
    ``better + worse`` 很小时近似偏差明显，只用来回答"方向上是不是更好"
    这种粗判断，别当成严格显著性证明。

    Returns
    -------
    (z_value, p_value)
        没有非平局配对时两者都是 NaN（方向无从判断）。
    """
    total = int(better) + int(worse)
    if total == 0:
        return float("nan"), float("nan")
    z_value = (int(better) - int(worse)) / math.sqrt(total)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return float(z_value), float(p_value)


def _split_frame_sets(records: list[dict]) -> tuple[set[int], set[int], dict[str, int]]:
    """按臂 3 自己的融合状态把帧切成「重推演成功 / 弃权」两堆。

    Returns
    -------
    (success, abstain, abstain_reasons)
        ``abstain_reasons`` 的键是弃权原因（``lost_target:<frame>`` 只取前缀）。
    """
    success: set[int] = set()
    abstain: set[int] = set()
    reasons: dict[str, int] = {}
    for record in records:
        kind = record.get("fusion_kind")
        frame = int(record["frame_index"])
        if kind == "replay":
            success.add(frame)
        elif kind == "replay_abstain":
            abstain.add(frame)
            reason = str(record.get("replay_reason", "")).split(":")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
    return success, abstain, reasons


def _paired(
    base_cle: dict[int, float], other_cle: dict[int, float], frames: set[int]
) -> dict:
    """同帧配对比较：谁更好、谁更差、中位差。"""
    diffs: list[float] = []
    better = worse = tie = 0
    for frame in sorted(frames):
        if frame not in base_cle or frame not in other_cle:
            continue
        delta = base_cle[frame] - other_cle[frame]
        diffs.append(delta)
        if delta < -1e-9:
            better += 1
        elif delta > 1e-9:
            worse += 1
        else:
            tie += 1
    median = float(np.median(diffs)) if diffs else float("nan")
    z_value, p_value = _sign_test(better, worse)
    return {
        "paired_frames": len(diffs),
        "better": better,
        "worse": worse,
        "tie": tie,
        "median_diff_px": median,
        "z_value": z_value,
        "p_value": p_value,
    }


def _replay_fallback_frames(outcomes: dict) -> tuple[set[int], dict[str, int]]:
    """所有臂里 ``fusion_kind == "hard_reinit"`` 的帧（弃权后的硬拉回兜底）。

    ``hard_reinit`` 只由 ``replay_fallback="hard"`` 的消融臂产生：臂 3 的
    TRACK / LOST 分支弃权后，直接用迟到检测框把位置硬拉回去。臂 3 自身的
    ``replay_fallback="none"`` 永远不产生这个 kind ⇒ 旧口径（帧集切分只看
    ``replay`` / ``replay_abstain``）下这些帧**既不在 success 也不在
    abstain**，被静默排除。

    Returns
    -------
    (frames, by_arm)
        ``frames`` 是全部臂的 ``hard_reinit`` 帧并集；``by_arm`` 是各臂各自
        贡献的帧数（只有真正产生该 kind 的臂才会出现）。
    """
    frames: set[int] = set()
    by_arm: dict[str, int] = {}
    for name, outcome in outcomes.items():
        count = 0
        for record in outcome.records:
            if record.get("fusion_kind") == "hard_reinit":
                frames.add(int(record["frame_index"]))
                count += 1
        if count:
            by_arm[name] = count
    return frames, by_arm


def _bucket_block(
    base_cle: dict[int, float],
    other_cle: dict[int, float],
    frames: set[int],
    base_name: str,
    other_name: str,
    tolerance_px: float,
) -> dict:
    """一个桶的对照块：两臂各自的 hits/n/命中率 + 配对方向统计。

    ``frames`` 必须已经限定在"两臂都有输出"的帧上 ⇒ 两臂的 ``n`` 相同，
    ``hits_delta_arm3_minus_other`` 才是可比的净命中差。
    """
    base_values = [base_cle[f] for f in sorted(frames) if f in base_cle]
    other_values = [other_cle[f] for f in sorted(frames) if f in other_cle]
    base_stats = _hit_stats(base_values, tolerance_px)
    other_stats = _hit_stats(other_values, tolerance_px)
    return {
        base_name: base_stats,
        other_name: other_stats,
        "hits_delta_arm3_minus_other": int(base_stats["hits"]) - int(other_stats["hits"]),
        "paired_arm3_vs_other": _paired(base_cle, other_cle, frames),
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run_conditional(
    sequence: str,
    cache_root: str = DEFAULT_CACHE_ROOT,
    frames_root: str = DEFAULT_FRAMES_ROOT,
    labels_root: str = DEFAULT_LABELS_ROOT,
    det_hz: float = 12.5,
    latency: int = 8,
    camera_hz: float = 25.0,
    total_frames: int = 0,
    score_threshold: float = 0.22,
    dedup_radius: float = 768.0,
    tolerance_px: float = 5.0,
    out_dir: Path | None = None,
    base_config: TrackerConfig | None = None,
) -> dict:
    """跑一次锁步遍历，产出条件化对照表（退回重推演成功 / 弃权两种帧集）。"""
    from manu.evaluation.eval_multiscale_cache_th import load_ground_truth

    base = base_config or TrackerConfig()
    base.latency_frames = latency
    base.camera_hz = camera_hz

    stream = SlowStream(
        cache_root, sequence, dedup_radius=dedup_radius, score_threshold=score_threshold
    )
    frames_total = int(total_frames or stream.frames)
    source = JpegFrameSource(Path(frames_root) / sequence)
    specs = _arm_specs(det_hz, latency)

    started = time.perf_counter()
    outcomes = run_lockstep(
        source,
        stream,
        specs,
        base_config=base,
        total_frames=frames_total,
        camera_hz=camera_hz,
    )
    elapsed = time.perf_counter() - started

    ground_truth = load_ground_truth(Path(labels_root), sequence, stream.width, stream.height)
    if frames_total < stream.frames:
        ground_truth = {
            frame: boxes for frame, boxes in ground_truth.items() if frame < frames_total
        }

    cle = {
        name: _cle_map(_positions(outcome.records), ground_truth)
        for name, outcome in outcomes.items()
    }

    # ---- 帧集：按臂 3 自己的融合状态切分 ----
    arm3_records = outcomes["arm3_rollback_replay"].records
    replay_ok, replay_abstain, abstain_reasons = _split_frame_sets(arm3_records)
    # ``hard_reinit``（弃权后硬拉回兜底）只出现在 replay_fallback="hard" 的
    # 消融臂上；旧口径的帧集切分不认识这个 kind ⇒ 这些帧既不在 success 也不在
    # abstain，被静默排除。与 success/abstain 重叠的仍按臂 3 自己的状态归类，
    # 剩下的单独成桶。
    hard_reinit, hard_reinit_by_arm = _replay_fallback_frames(outcomes)
    fallback_frames = hard_reinit - replay_ok - replay_abstain

    comparisons = ("arm1_detector_only", "ablation_fused_alpha")
    payload: dict = {
        "sequence": sequence,
        "frames": frames_total,
        "tolerance_px": tolerance_px,
        "latency_frames": latency,
        "elapsed_seconds": elapsed,
        "frame_sets": {
            "replay_success": len(replay_ok),
            "replay_abstain": len(replay_abstain),
            "replay_attempted": len(replay_ok) + len(replay_abstain),
            "abstain_rate_pct": (
                len(replay_abstain) / (len(replay_ok) + len(replay_abstain)) * 100.0
                if (len(replay_ok) + len(replay_abstain))
                else 0.0
            ),
            "abstain_reasons": abstain_reasons,
            "replay_fallback": len(fallback_frames),
            "replay_fallback_total": len(hard_reinit),
            "replay_fallback_overlap": len(hard_reinit & (replay_ok | replay_abstain)),
            "replay_fallback_by_arm": hard_reinit_by_arm,
        },
        "on_replay_success": {},
        "on_replay_abstain": {},
        "full_sequence_decomposition": {
            "note": (
                "全序列分解：把'目标存在且两臂都有输出'的全序列 GT 帧，按是否落在"
                "条件帧集内切成**互斥**的四桶（success / abstain / replay_fallback / "
                "uncovered）。每个桶的 n = 该桶内两臂都有输出的帧数，命中率 = hits / n。"
                "四桶相加 = total_frames（见 sum_check，可直接核对）。"
            ),
            "bucket_definitions": {
                "success": "臂 3 重推演成功的帧（臂 3 的 fusion_kind == 'replay'）",
                "abstain": (
                    "臂 3 尝试重推演但弃权的帧（臂 3 的 fusion_kind == 'replay_abstain'）"
                ),
                "replay_fallback": (
                    "弃权后硬拉回兜底的帧（fusion_kind == 'hard_reinit'，只有 "
                    "replay_fallback='hard' 的消融臂会产生）；与 success/abstain 重叠的"
                    "帧仍归 success/abstain，这里只收剩下的"
                ),
                "uncovered": (
                    "**未被条件化覆盖**的帧：臂 3 在这些帧上没有发生任何重推演事件"
                    "（没有迟到检测到达 / 尚未初始化等），臂 3 只是继续前推"
                ),
            },
            "hits_definition": (
                "hits = CLE ≤ tolerance_px 的帧数；n = 该桶内两臂都有输出的帧数"
            ),
            "sign_test_note": (
                "z = (better - worse) / sqrt(better + worse)，双侧 p 值用正态近似"
                "（erfc），**非**精确二项检验；better/worse 均以臂 3 为准"
            ),
            "by_comparison": {},
        },
    }

    for name in comparisons:
        if name not in cle:
            continue
        success_frames = {
            frame
            for frame in replay_ok
            if frame in cle["arm3_rollback_replay"] and frame in cle[name]
        }
        payload["on_replay_success"][name] = {
            "arm3_rollback_replay": _stats(
                [cle["arm3_rollback_replay"][f] for f in success_frames], tolerance_px
            ),
            name: _stats([cle[name][f] for f in success_frames], tolerance_px),
            "paired_arm3_vs_other": _paired(
                cle["arm3_rollback_replay"], cle[name], success_frames
            ),
        }

        abstain_frames = {
            frame
            for frame in replay_abstain
            if frame in cle["arm3_rollback_replay"] and frame in cle[name]
        }
        payload["on_replay_abstain"][name] = {
            "arm3_rollback_replay": _stats(
                [cle["arm3_rollback_replay"][f] for f in abstain_frames], tolerance_px
            ),
            name: _stats([cle[name][f] for f in abstain_frames], tolerance_px),
            "paired_arm3_vs_other": _paired(
                cle["arm3_rollback_replay"], cle[name], abstain_frames
            ),
        }

        # ---- 全序列分解：success / abstain / replay_fallback / uncovered ----
        universe = set(cle["arm3_rollback_replay"]) & set(cle[name])
        bucket_frames = {
            "success": {frame for frame in replay_ok if frame in universe},
            "abstain": {frame for frame in replay_abstain if frame in universe},
            "replay_fallback": {frame for frame in fallback_frames if frame in universe},
        }
        covered = set().union(*bucket_frames.values())
        bucket_frames["uncovered"] = universe - covered
        buckets_n_sum = sum(len(frames) for frames in bucket_frames.values())
        payload["full_sequence_decomposition"]["by_comparison"][name] = {
            "total_frames": len(universe),
            "covered_frames": len(covered),
            "coverage_pct": (len(covered) / len(universe) * 100.0) if universe else 0.0,
            "buckets": {
                bucket: _bucket_block(
                    cle["arm3_rollback_replay"],
                    cle[name],
                    frames,
                    "arm3_rollback_replay",
                    name,
                    tolerance_px,
                )
                for bucket, frames in bucket_frames.items()
            },
            "sum_check": {
                "buckets_n_sum": buckets_n_sum,
                "total_frames": len(universe),
                "consistent": buckets_n_sum == len(universe),
                "note": (
                    "四桶互斥且并集 = total_frames（目标存在且两臂都有输出的 GT 帧数）"
                ),
            },
        }

    if out_dir is not None:
        out_dir = Path(out_dir) / sequence
        out_dir.mkdir(parents=True, exist_ok=True)
        payload["artifacts"] = str(out_dir)
        # 先渲染再落盘：渲染失败就不会留下"只写了一半"的产物。
        markdown = render_markdown(payload)
        (out_dir / "conditional_comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "conditional_comparison.md").write_text(markdown, encoding="utf-8")
    return payload


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append(f"# 条件化对比 — {payload['sequence']}")
    lines.append("")
    sets = payload["frame_sets"]
    reasons = "、".join(
        f"`{key}` {value}" for key, value in sorted(sets["abstain_reasons"].items())
    )
    lines.append(
        f"帧集：重推演成功 **{sets['replay_success']}** 帧；弃权 **{sets['replay_abstain']}** 帧"
        f"（{reasons}）"
    )
    lines.append(
        f"尝试重推演的帧共 **{sets.get('replay_attempted', 0)}** 帧，"
        f"弃权率 **{sets.get('abstain_rate_pct', 0.0):.2f}%**"
    )
    by_arm = sets.get("replay_fallback_by_arm", {})
    arms_text = "、".join(f"`{key}` {value}" for key, value in sorted(by_arm.items()))
    lines.append(
        f"弃权硬拉回兜底（``hard_reinit``）**{sets.get('replay_fallback', 0)}** 帧"
        f"（全部臂共 {sets.get('replay_fallback_total', 0)} 帧，其中 "
        f"{sets.get('replay_fallback_overlap', 0)} 帧与 success/abstain 重叠；"
        f"来源：{arms_text or '无'}）"
    )
    lines.append("")

    lines.append("## 仅在「重推演成功」的帧上同帧配对")
    lines.append("")
    for name, block in payload["on_replay_success"].items():
        lines.append(f"### vs {name}")
        lines.append("")
        lines.append("| 臂 | CLE p50 | CLE p90 | CLE p95 | Hit@5px | n |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for label, key in (
            ("臂 3 退回重推演", "arm3_rollback_replay"),
            (name, name),
        ):
            stats = block[key]
            if not stats.get("n"):
                lines.append(f"| {label} | — | — | — | — | 0 |")
                continue
            lines.append(
                f"| {label} | {stats['cle_p50']:.2f} px | {stats['cle_p90']:.2f} px | "
                f"{stats['cle_p95']:.2f} px | {stats['hit_rate_pct']:.2f}% | {stats['n']} |"
            )
        paired = block["paired_arm3_vs_other"]
        lines.append("")
        lines.append(
            f"- 配对 臂3 vs {name}：更好 **{paired['better']}** / 更差 **{paired['worse']}** / "
            f"持平 {paired['tie']}，中位差 **{paired['median_diff_px']:.3f} px**"
            f"（配对 {paired['paired_frames']} 帧）"
        )
        lines.append("")

    lines.append("## 在「弃权」的帧上")
    lines.append("")
    for name, block in payload["on_replay_abstain"].items():
        lines.append(f"### vs {name}")
        lines.append("")
        lines.append("| 臂 | CLE p50 | Hit@5px | n |")
        lines.append("| --- | --- | --- | --- |")
        for label, key in (
            ("臂 3 退回重推演", "arm3_rollback_replay"),
            (name, name),
        ):
            stats = block[key]
            if not stats.get("n"):
                lines.append(f"| {label} | — | — | 0 |")
                continue
            lines.append(
                f"| {label} | {stats['cle_p50']:.2f} px | {stats['hit_rate_pct']:.2f}% | {stats['n']} |"
            )
        lines.append("")

    decomposition = payload.get("full_sequence_decomposition")
    if decomposition:
        lines.append("## 全序列分解（条件帧集覆盖 vs 未覆盖）")
        lines.append("")
        lines.append(
            "分母：**目标存在且两臂都有输出**的全序列 GT 帧；四桶互斥、相加 = 分母。"
        )
        lines.append("")
        for name, block in decomposition["by_comparison"].items():
            lines.append(f"### vs {name}")
            lines.append("")
            lines.append(
                f"全序列 **{block['total_frames']}** 帧；条件化覆盖 "
                f"**{block['covered_frames']}** 帧（{block['coverage_pct']:.1f}%）"
            )
            lines.append("")
            lines.append(
                "| 桶 | n | 臂 3 hits | "
                + name
                + " hits | Δhits | 臂 3 命中率 | "
                + name
                + " 命中率 | 配对 z | p |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for bucket in ("success", "abstain", "replay_fallback", "uncovered"):
                item = block["buckets"][bucket]
                base = item["arm3_rollback_replay"]
                other = item[name]
                paired = item["paired_arm3_vs_other"]
                if not base["n"]:
                    lines.append(f"| {bucket} | 0 | — | — | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| {bucket} | {base['n']} | {base['hits']} | {other['hits']} | "
                    f"{item['hits_delta_arm3_minus_other']:+d} | "
                    f"{base['hit_rate_pct']:.2f}% | {other['hit_rate_pct']:.2f}% | "
                    f"{paired['z_value']:+.2f} | {paired['p_value']:.3g} |"
                )
            lines.append("")
        lines.append(
            "配对 z = (better − worse) / √(better + worse)，双侧 p 用正态近似"
            "（**非**精确二项检验）；better / worse 均以臂 3 为准。"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="条件化对比（退回重推演成功 / 弃权）")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--det-hz", type=float, default=12.5)
    parser.add_argument("--latency", type=int, default=8)
    parser.add_argument("--camera-hz", type=float, default=25.0)
    parser.add_argument("--total-frames", type=int, default=0)
    parser.add_argument("--tolerance-px", type=float, default=5.0)
    parser.add_argument("--out-dir", default="runs/tracking_async")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    payload = run_conditional(
        args.sequence,
        cache_root=args.cache_root,
        frames_root=args.frames_root,
        labels_root=args.labels_root,
        det_hz=args.det_hz,
        latency=args.latency,
        camera_hz=args.camera_hz,
        total_frames=args.total_frames,
        tolerance_px=args.tolerance_px,
        out_dir=None if args.no_write else Path(args.out_dir),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
