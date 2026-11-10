#!/usr/bin/env python3
"""联合门禁：板卡实测延迟 × NanoBench 分层召回（纯 CPU，只读缓存）。

═══ 为什么需要 ═══
RK3588 板端交付有两条互相拉扯的硬门槛：

* **速度**：int8 推理 ≤ 30 ms（RK3588 NPU **板卡实测**延迟，毫秒）
* **精度**：**微小目标召回不能下降**（nano 桶：目标 ≤4px 或面积 ≤16px²）

单看任何一条都会骗人：

* 只看速度 ⇒ 会选到一个 nano 召回掉了 3 pp 的"快模型"；
* 只看 nano 召回 ⇒ 会放行一个 55 ms 的模型（板端根本跑不动）。

本脚本把两者合到**一张判定表**：每臂给 SPEED / RECALL 两个门各自的结论 + 联合总判定。

═══ 口径（不重写，只复用）═══
全部精度数字来自 ``manu/evaluation/tiny_target_subset.py`` 的 ``eval_cache``：
no-GMC 口径、双容差（8px 主档 / 4px 严格档）、工作点 = 整体 F1 峰值阈值，
分桶按物理像素尺寸（nano / micro / small / regular）。

参照系（base_frozen / RC2 / cs_t0043 / delivery）**优先用缓存实测**；
缓存拿不到时（本机没挂服务器数据）回落到**内置常量表**，并在 ``source`` 字段标注
``builtin_constant``，**绝不因此报错或中断**。

═══ 用法 ═══
::

    python manu/evaluation/dual_constraint_gate.py \
      --candidate name=/path/cache.pkl=21.998 ... \
      --baseline /path/base.pkl \
      --ref-labels base_frozen,RC2,delivery \
      --budget-ms 30 --nano-min 0.6978 --out out.json

    # 口径自检（纯内置常量，不碰任何缓存，秒级）
    python manu/evaluation/dual_constraint_gate.py --selftest

退出码：有 ERROR 臂 ⇒ 2；否则 0（PASS/FAIL **不**影响退出码）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

# --- 复用 NanoBench 评测台（唯一真值来源，本脚本不重写匹配口径）---------------
try:  # 包内导入（python -m manu.evaluation.dual_constraint_gate）
    from manu.evaluation.tiny_target_subset import eval_cache
except ImportError:  # pragma: no cover - 以脚本方式直接运行时兜底
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from manu.evaluation.tiny_target_subset import eval_cache

# ==============================================================================
# 内置参照常量表（缓存缺失时兜底；数值来自 NanoBench 实测，no-GMC 口径）
# ==============================================================================
BUILTIN_REFS: dict[str, dict[str, Any]] = {
    "base_frozen": {
        "cache": "/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl",
        "nano": 0.6978, "micro": 0.8111, "regular": 0.9614, "f1": 0.901312,
        "desc": "冻结基线（P1 320 头，板卡 55.48 ms）",
    },
    "RC2": {
        "cache": "/mnt/data/siping/runs/npu20ms/tiny_ref/rc2_trial0080.pkl",
        "nano": 0.6678, "micro": 0.8239, "regular": 0.9503, "f1": 0.893119,
        "desc": "RC2 = trial_0080（板卡 100 ms）",
    },
    "cs_t0043": {
        "cache": "/mnt/data/siping/runs/npu20ms/tiny_ref/cs_t0043.pkl",
        "nano": 0.6498, "micro": 0.8300, "regular": 0.9352, "f1": 0.890972,
        "desc": "Model Soup 主装 cs_t0043",
    },
    "delivery": {
        "cache": "/mnt/data/siping/runs/npu20ms/tiny_ref/trial_0031.pkl",
        "nano": 0.6666, "micro": 0.8370, "regular": 0.9445, "f1": 0.887122,
        "desc": "在位交付件 trial_0031（板卡 19.306 ms）",
    },
}

# 表格里 Δ 列的短标签（b / r / d / c）
REF_SHORT: dict[str, str] = {"base_frozen": "b", "RC2": "r", "delivery": "d", "cs_t0043": "c"}

# 默认参照系顺序
DEFAULT_REF_LABELS = ("base_frozen", "RC2", "delivery")

# 服务器数据在本机的只读 SSHFS 镜像（若存在则优先用它，省一次网络往返）
MIRROR_PREFIX = "/mnt/data/"
LOCAL_MIRROR_PREFIX = "/home/manu/mnt/data/"

# 判定 token
SPEED_PASS, SPEED_FAIL, SPEED_UNKNOWN = "SPEED_PASS", "SPEED_FAIL", "SPEED_UNKNOWN"
RECALL_PASS, RECALL_FAIL = "RECALL_PASS", "RECALL_FAIL"
VERDICT_PASS, VERDICT_FAIL, VERDICT_GAP, VERDICT_ERROR = "PASS", "FAIL", "PASS_WITH_GAP", "ERROR"

DEFAULT_BUDGET_MS = 30.0
DEFAULT_NANO_MIN = 0.6978  # 冻结 P1 基线水平

# 召回门比较容差 = 阈值末位（4 位小数）ULP 的一半 = 0.005 pp
RECALL_EPS = 5e-5


# ==============================================================================
# 基础工具
# ==============================================================================
def resolve_cache_path(path: str) -> Optional[str]:
    """把缓存路径解析成本机真实存在的路径；都不存在则返回 None。

    先试原路径（服务器绝对路径），再试本机只读 SSHFS 镜像的等价路径。
    """
    if not path:
        return None
    if os.path.exists(path):
        return path
    if path.startswith(MIRROR_PREFIX):
        mirrored = LOCAL_MIRROR_PREFIX + path[len(MIRROR_PREFIX):]
        if os.path.exists(mirrored):
            return mirrored
    return None


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    """除法防零：分母为 0（或非有限）时返回 None。"""
    if denominator == 0 or denominator != denominator:
        return None
    return numerator / denominator


def parse_tols(text: str) -> tuple[float, ...]:
    """解析 --tols；必须同时含 4.0 与 8.0（eval_cache 的双容差口径）。"""
    try:
        tols = tuple(float(x) for x in str(text).split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit(f"[参数错误] --tols 解析失败: {text!r}（{exc}）") from exc
    if not tols:
        raise SystemExit("[参数错误] --tols 不能为空")
    for need in (4.0, 8.0):
        if need not in tols:
            raise SystemExit(f"[参数错误] --tols 必须同时含 {need:g} 与 8.0（双容差口径），当前 {tols}")
    return tols


def _is_float(text: str) -> bool:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def parse_candidate_spec(spec: str, index: int) -> tuple[str, str, Optional[float]]:
    """解析 --candidate 的 ``name=path`` / ``name=path=ms`` / ``name=path:ms``。

    返回 (name, cache_path, board_ms)；board_ms 为 None 表示"未测速"。
    路径里含 '=' 是允许的：只有**末尾**那段能解析成数字时才当作板卡 ms。
    """
    text = str(spec).strip()
    if "=" in text:
        name, rest = text.split("=", 1)
    else:
        name, rest = "", text
    name, rest = name.strip(), rest.strip()
    if not rest:
        raise SystemExit(f"[参数错误] --candidate 缺少缓存路径: {spec!r}")
    if not name:
        name = Path(rest).stem or f"cand_{index}"

    board_ms: Optional[float] = None
    for sep in ("=", ":"):
        pos = rest.rfind(sep)
        if pos > 0 and _is_float(rest[pos + 1:]):
            board_ms = float(rest[pos + 1:])
            rest = rest[:pos].strip()
            break
    return name, rest, board_ms


def _disp_width(text: str) -> int:
    """终端显示宽度（东亚全角字符按 2 列算）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    """按显示宽度补齐，保证中英混排的表格能对齐。"""
    gap = max(0, width - _disp_width(text))
    return (" " * gap + text) if align == "right" else (text + " " * gap)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# ==============================================================================
# 指标抽取：把 eval_cache 的返回收敛成统一的 metrics dict
# ==============================================================================
def metrics_from_eval_result(result: dict) -> dict[str, Any]:
    """从 eval_cache 的返回值抽出本脚本需要的字段（tol8 主档 + tol4 严格档）。"""
    p8 = result["peak_tol8"]
    p4 = result["peak_tol4"]
    r8 = p8["recall_by_bucket"]
    r4 = p4["recall_by_bucket"]
    return {
        "nano": float(r8["nano"]),
        "micro": float(r8["micro"]),
        "small": float(r8["small"]),
        "regular": float(r8["regular"]),
        "f1": float(p8["f1"]),
        "precision": float(p8["precision"]),
        "recall": float(p8["recall"]),
        "fp": int(p8["fp"]),
        "fn": int(p8["fn"]),
        "th": float(p8["th"]),
        "nano_tol4": float(r4["nano"]),
        "micro_tol4": float(r4["micro"]),
        "frames": int(result.get("frames", 0)),
        "total_gt": int(result.get("total_gt", 0)),
    }


_EVAL_MEMO: dict[tuple, dict[str, Any]] = {}


def evaluate_cache(cache_path: str, semantics: str, tols: tuple[float, ...],
                   img_w: int, img_h: int, label: str) -> dict[str, Any]:
    """跑一次 NanoBench 评测（no-GMC 口径），返回统一 metrics dict。

    同一个缓存只算一次（每份 31613 帧缓存的阈值扫描约 8 s，参照系与候选臂常会重合）。
    """
    key = (cache_path, semantics, tols, img_w, img_h)
    if key not in _EVAL_MEMO:
        result = eval_cache(cache_path, semantics, tols=tols, img_w=img_w, img_h=img_h, label=label)
        _EVAL_MEMO[key] = metrics_from_eval_result(result)
    return _EVAL_MEMO[key]


def load_ref_metrics(label: str, semantics: str, tols: tuple[float, ...],
                     img_w: int, img_h: int, cache_override: str = "") -> dict[str, Any]:
    """取一个参照系的指标：缓存可读 ⇒ 实测；否则 ⇒ 内置常量（source=builtin_constant）。"""
    builtin = BUILTIN_REFS.get(label, {})
    cache = cache_override or builtin.get("cache", "")
    resolved = resolve_cache_path(cache)
    if resolved:
        try:
            metrics = evaluate_cache(resolved, semantics, tols, img_w, img_h, label)
            return {
                "label": label, "cache": cache, "resolved_cache": resolved,
                "source": "cache_eval", "desc": builtin.get("desc", ""), **metrics,
            }
        except Exception as exc:  # 缓存损坏 ⇒ 回落常量，不中断整轮门禁
            print(f"[警告] 参照 {label} 缓存评测失败（{type(exc).__name__}: {exc}）⇒ 回落到内置常量",
                  file=sys.stderr)
    return {
        "label": label, "cache": cache, "resolved_cache": None,
        "source": "builtin_constant", "desc": builtin.get("desc", ""),
        "nano": float(builtin.get("nano", 0.0)), "micro": float(builtin.get("micro", 0.0)),
        "regular": float(builtin.get("regular", 0.0)), "f1": float(builtin.get("f1", 0.0)),
        "fp": None, "fn": None, "th": None, "small": None,
        "nano_tol4": None, "micro_tol4": None, "frames": None, "total_gt": None,
    }


# ==============================================================================
# 判定逻辑
# ==============================================================================
def judge_speed(board_ms: Optional[float], budget_ms: float) -> str:
    """速度门：给了 ms 且 ≤ 预算 ⇒ PASS；给了 ms 且超预算 ⇒ FAIL；没给 ⇒ UNKNOWN。"""
    if board_ms is None:
        return SPEED_UNKNOWN
    return SPEED_PASS if board_ms <= budget_ms else SPEED_FAIL


def judge_recall(nano: Optional[float], nano_min: float, eps: float = RECALL_EPS) -> str:
    """召回门：nano 召回 ≥ 下限 ⇒ PASS；否则 FAIL。

    ⚠️ ``eps``（默认 5e-5 = 0.005 pp）是**阈值表示精度的一半**：``--nano-min`` 只给到
    4 位小数（0.6978），而实测值是 0.69777883…，两者相差 2.1e-5 纯属四舍五入。
    若用严格 ``>=``，冻结基线**自己**会被自己的下限判死（差 -0.0021 pp）⇒ 门禁失去
    可用性。取半个末位 ULP 作容差，既不放过真实掉点，也不被表示误差误伤。
    """
    if nano is None:
        return RECALL_FAIL
    return RECALL_PASS if nano >= nano_min - eps else RECALL_FAIL


def judge_verdict(speed_verdict: str, recall_verdict: str) -> str:
    """联合总判定。

    ⚠️ **最容易搞错的地方**：``FAIL`` 的判定必须**优先于** ``SPEED_UNKNOWN``。
    若写成"速度不是 PASS 就给 PASS_WITH_GAP"，那么一个 nano 召回不达标、
    又恰好没测速的臂会被误判成 PASS_WITH_GAP（= 放行）⇒ 硬门槛被绕过。
    正确次序：先查有没有 FAIL，再查 PASS，最后才是 PASS_WITH_GAP。
    """
    if speed_verdict == SPEED_FAIL or recall_verdict == RECALL_FAIL:
        return VERDICT_FAIL
    if speed_verdict == SPEED_PASS and recall_verdict == RECALL_PASS:
        return VERDICT_PASS
    if speed_verdict == SPEED_UNKNOWN and recall_verdict == RECALL_PASS:
        return VERDICT_GAP
    return VERDICT_FAIL


def compute_deltas(metrics: dict[str, Any], refs: dict[str, dict[str, Any]],
                   ref_order: list[str]) -> dict[str, dict[str, float]]:
    """算相对各参照系的差值（百分点，保留 2 位小数）。"""
    deltas: dict[str, dict[str, float]] = {}
    for label in ref_order:
        ref = refs.get(label)
        if not ref:
            continue
        deltas[label] = {
            "nano_pp": round((metrics["nano"] - ref["nano"]) * 100.0, 2),
            "micro_pp": round((metrics["micro"] - ref["micro"]) * 100.0, 2),
            "f1_pp": round((metrics["f1"] - ref["f1"]) * 100.0, 2),
        }
    return deltas


def build_row(name: str, cache: str, board_ms: Optional[float], metrics: dict[str, Any],
              baseline_micro: Optional[float], budget_ms: float, nano_min: float,
              refs: dict[str, dict[str, Any]], ref_order: list[str],
              recall_eps: float = RECALL_EPS) -> dict[str, Any]:
    """把一个臂的指标 + 判定 + 参照系差值组装成一行（JSON 与表格共用）。"""
    speed_verdict = judge_speed(board_ms, budget_ms)
    recall_verdict = judge_recall(metrics.get("nano"), nano_min, recall_eps)
    verdict = judge_verdict(speed_verdict, recall_verdict)
    deltas = compute_deltas(metrics, refs, ref_order)

    ms_per_nano = None
    if board_ms is not None:
        ms_per_nano = safe_div(board_ms, metrics["nano"] * 100.0)

    micro_vs_baseline = None
    if baseline_micro is not None:
        micro_vs_baseline = round(metrics["micro"] - baseline_micro, 6)

    nano = metrics.get("nano")
    recall_margin_pp = None if nano is None else round((nano - nano_min) * 100.0, 4)

    return {
        "name": name,
        "cache": cache,
        "board_ms": board_ms,
        "speed_verdict": speed_verdict,
        "recall_verdict": recall_verdict,
        "verdict": verdict,
        "nano": metrics.get("nano"),
        "micro": metrics.get("micro"),
        "regular": metrics.get("regular"),
        "small": metrics.get("small"),
        "f1": metrics.get("f1"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "th": metrics.get("th"),
        "nano_tol4": metrics.get("nano_tol4"),
        "micro_tol4": metrics.get("micro_tol4"),
        "frames": metrics.get("frames"),
        "total_gt": metrics.get("total_gt"),
        "deltas": deltas,
        "ms_per_nano_pp": None if ms_per_nano is None else round(ms_per_nano, 6),
        "micro_vs_baseline": micro_vs_baseline,
        "micro_ge_baseline": None if micro_vs_baseline is None else (micro_vs_baseline >= 0.0),
        "recall_margin_pp": recall_margin_pp,
    }


# ==============================================================================
# 表格输出
# ==============================================================================
def format_table(rows: list[dict[str, Any]], ref_order: list[str]) -> str:
    """打印对齐的联合门禁表（只印摘要行，不印大 dict）。"""
    header = ["模型", "板卡ms", "SPEED", "nano", "micro", "regular", "T1_F1", "RECALL", "总判定", "Δnano vs 参照"]
    aligns = ["left", "right", "left", "right", "right", "right", "right", "left", "left", "left"]
    widths = [22, 10, 8, 8, 8, 9, 10, 8, 15, 38]

    lines = [" ".join(_pad(h, w, a) for h, w, a in zip(header, widths, aligns))]
    lines.append("-" * _disp_width(lines[0]))

    for row in rows:
        if row.get("verdict") == VERDICT_ERROR:
            cells = [row["name"],
                     f"{row['board_ms']:.3f}" if row.get("board_ms") is not None else "-",
                     "-", "-", "-", "-", "-", "-", VERDICT_ERROR, "缓存不可用"]
        else:
            speed = {SPEED_PASS: "PASS", SPEED_FAIL: "FAIL", SPEED_UNKNOWN: "UNKNOWN"}[row["speed_verdict"]]
            recall = {RECALL_PASS: "PASS", RECALL_FAIL: "FAIL"}[row["recall_verdict"]]
            delta_parts = []
            for label in ref_order:
                value = row.get("deltas", {}).get(label, {}).get("nano_pp")
                short = REF_SHORT.get(label, label[:1].lower())
                delta_parts.append(f"{short}:{value:+.1f}pp" if value is not None else f"{short}:-")
            cells = [
                row["name"],
                f"{row['board_ms']:.3f}" if row.get("board_ms") is not None else "-",
                speed,
                _fmt(row.get("nano")),
                _fmt(row.get("micro")),
                _fmt(row.get("regular")),
                _fmt(row.get("f1"), 6),
                recall,
                row["verdict"],
                " ".join(delta_parts),
            ]
        lines.append(" ".join(_pad(c, w, a) for c, w, a in zip(cells, widths, aligns)))
    return "\n".join(lines)


def format_conclusion(rows: list[dict[str, Any]], budget_ms: float, nano_min: float,
                      refs: dict[str, dict[str, Any]], ref_order: list[str]) -> str:
    """表下面的文字结论：谁 PASS、谁 FAIL、失败原因分别是什么。"""
    out: list[str] = []
    passed = [r for r in rows if r["verdict"] == VERDICT_PASS]
    gaps = [r for r in rows if r["verdict"] == VERDICT_GAP]
    failed = [r for r in rows if r["verdict"] == VERDICT_FAIL]
    errors = [r for r in rows if r["verdict"] == VERDICT_ERROR]

    out.append("结论：")
    if passed:
        out.append(f"  ✅ PASS（{len(passed)} 臂）：" + "、".join(
            f"{r['name']}（板卡 {_fmt(r['board_ms'], 3) if r.get('board_ms') is not None else '-'} ms "
            f"≤ {budget_ms:g} ms，nano {r['nano']:.4f} ≥ {nano_min:g}）" for r in passed))
    else:
        out.append("  ✅ PASS（0 臂）")
    if gaps:
        out.append(f"  ⚠️ PASS_WITH_GAP（{len(gaps)} 臂，召回达标但**未测速**，不构成放行）：" + "、".join(
            f"{r['name']}（nano {r['nano']:.4f}，板卡延迟待测）" for r in gaps))
    if failed:
        out.append(f"  ❌ FAIL（{len(failed)} 臂）：")
        for r in failed:
            reasons = []
            if r["speed_verdict"] == SPEED_FAIL:
                reasons.append(f"板卡 {r['board_ms']:.3f} ms 超预算 {budget_ms:g} ms"
                               f"（超 {r['board_ms'] - budget_ms:.3f} ms）")
            if r["recall_verdict"] == RECALL_FAIL:
                reasons.append(f"nano 召回 {r['nano']:.4f} < 下限 {nano_min:g}"
                               f"（差 {(r['nano'] - nano_min) * 100:.4f} pp）")
            out.append(f"     - {r['name']}：" + "；".join(reasons))
    if errors:
        out.append(f"  ❗ ERROR（{len(errors)} 臂，缓存不可用/损坏）：" + "、".join(
            f"{r['name']}（{r.get('error', '未知原因')}）" for r in errors))

    # micro 辅助提示（**不参与硬判定**）：是否 ≥ 基线 micro
    hints = [r for r in rows if r.get("micro_ge_baseline") is not None]
    if hints:
        out.append("辅助提示（不参与硬判定）：micro 召回 vs 基线 micro —— " + "、".join(
            f"{r['name']} {'≥' if r['micro_ge_baseline'] else '<'} 基线"
            f"（{r['micro_vs_baseline'] * 100:+.2f} pp）" for r in hints))

    if ref_order:
        out.append("参照系（差值基准，nano/micro/F1 均按百分点给出）：")
        for label in ref_order:
            ref = refs.get(label)
            if not ref:
                continue
            out.append(f"  - {label}：nano {ref['nano']:.4f} / micro {ref['micro']:.4f} / "
                       f"F1 {ref['f1']:.6f}（来源 {ref['source']}）"
                       + (f" —— {ref['desc']}" if ref.get("desc") else ""))
    return "\n".join(out)


# ==============================================================================
# 自检（纯内置常量，不碰任何缓存，秒级）
# ==============================================================================
def run_selftest(budget_ms: float, nano_min: float, out_path: str = "",
                 recall_eps: float = RECALL_EPS) -> int:
    """用内置常量构造假想臂，跑通**同一套**判定逻辑并断言结果。"""
    print("=== 联合门禁自检（内置常量构造假想臂，不读任何缓存）===")
    print(f"预算：板卡 ≤ {budget_ms:g} ms；召回下限：nano ≥ {nano_min:g}"
          f"（比较容差 {recall_eps:g} = {recall_eps * 100:.4f} pp）")
    print()

    # 假想臂：(name, nano, micro, regular, f1, board_ms, 期望总判定, 说明)
    fake_arms = [
        ("selftest_pass", 0.6978, 0.8111, 0.9614, 0.901312, 21.998, VERDICT_PASS,
         "nano 达标 + 速度达标 ⇒ PASS"),
        ("selftest_fail_recall", 0.6498, 0.8300, 0.9352, 0.890972, 21.998, VERDICT_FAIL,
         "nano 不达标 ⇒ FAIL（召回门）"),
        ("selftest_fail_speed", 0.6978, 0.8111, 0.9614, 0.901312, 55.480, VERDICT_FAIL,
         "速度超标 ⇒ FAIL（速度门）"),
        ("selftest_gap", 0.6990, 0.8150, 0.9620, 0.901500, None, VERDICT_GAP,
         "召回达标但未测速 ⇒ PASS_WITH_GAP"),
        ("selftest_fail_recall_noms", 0.6666, 0.8370, 0.9445, 0.887122, None, VERDICT_FAIL,
         "召回不达标 + 未测速 ⇒ **仍须 FAIL**（FAIL 优先于 UNKNOWN）"),
        ("selftest_eps_boundary", 0.69779, 0.8111, 0.9614, 0.901312, 21.998, VERDICT_PASS,
         "低于下限 1e-5（< 半个末位 ULP）⇒ PASS（阈值表示精度容差）"),
    ]

    ref_order = list(DEFAULT_REF_LABELS)
    refs = {
        label: {
            "label": label, "cache": BUILTIN_REFS[label]["cache"], "source": "builtin_constant",
            "desc": BUILTIN_REFS[label]["desc"],
            "nano": BUILTIN_REFS[label]["nano"], "micro": BUILTIN_REFS[label]["micro"],
            "regular": BUILTIN_REFS[label]["regular"], "f1": BUILTIN_REFS[label]["f1"],
        }
        for label in ref_order
    }
    baseline_micro = BUILTIN_REFS["base_frozen"]["micro"]

    rows = []
    for name, nano, micro, regular, f1, board_ms, expected, note in fake_arms:
        metrics = {"nano": nano, "micro": micro, "regular": regular, "small": None, "f1": f1,
                   "precision": None, "recall": None, "fp": None, "fn": None, "th": None,
                   "nano_tol4": None, "micro_tol4": None, "frames": None, "total_gt": None}
        row = build_row(name, "<builtin_constant>", board_ms, metrics, baseline_micro,
                        budget_ms, nano_min, refs, ref_order, recall_eps)
        row["expected_verdict"] = expected
        row["note"] = note
        rows.append(row)

    print(format_table(rows, ref_order))
    print()
    print(format_conclusion(rows, budget_ms, nano_min, refs, ref_order))
    print()

    bad = 0
    print("断言：")
    for row in rows:
        ok = row["verdict"] == row["expected_verdict"]
        bad += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {row['name']}：期望 {row['expected_verdict']}，"
              f"实得 {row['verdict']}（{row['note']}）")
    print(f"\n自检：{len(rows)} 臂 ⇒ 不一致 {bad} 处 ⇒ "
          + ("PASS（判定逻辑正确）" if bad == 0 else "FAIL（判定逻辑有 bug）"))

    if out_path:
        payload = {
            "budget_ms": budget_ms, "nano_min": nano_min, "recall_eps": recall_eps,
            "mode": "selftest", "rows": rows, "refs": refs,
            "summary": {
                "pass": [r["name"] for r in rows if r["verdict"] == VERDICT_PASS],
                "fail": [r["name"] for r in rows if r["verdict"] == VERDICT_FAIL],
                "pass_with_gap": [r["name"] for r in rows if r["verdict"] == VERDICT_GAP],
                "error": [],
            },
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written {out_path}")
    return 0 if bad == 0 else 1


# ==============================================================================
# 主流程
# ==============================================================================
def resolve_ref_labels(text: str) -> list[str]:
    """解析 --ref-labels；大小写不敏感匹配内置参照名。"""
    labels: list[str] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if item in BUILTIN_REFS:
            labels.append(item)
            continue
        hit = next((k for k in BUILTIN_REFS if k.lower() == item.lower()), None)
        if hit is None:
            print(f"[警告] 未知参照名 {item!r}（已知: {', '.join(BUILTIN_REFS)}）⇒ 跳过", file=sys.stderr)
            continue
        labels.append(hit)
    return labels


def run_gate(args: argparse.Namespace) -> int:
    """跑一轮联合门禁，返回进程退出码（有 ERROR 臂 ⇒ 2，否则 0）。"""
    tols = parse_tols(args.tols)
    ref_order = resolve_ref_labels(args.ref_labels)
    print(f"门禁参数：速度预算 {args.budget_ms:g} ms；nano 召回下限 {args.nano_min:g}"
          f"（比较容差 {args.recall_eps:g} = {args.recall_eps * 100:.4f} pp）")

    # --- 参照系：缓存实测优先，缺失 ⇒ 内置常量 --------------------------
    refs: dict[str, dict[str, Any]] = {}
    for label in ref_order:
        refs[label] = load_ref_metrics(label, args.semantics, tols, args.img_w, args.img_h)
        ref = refs[label]
        print(f"[参照] {label}：nano {ref['nano']:.4f} / micro {ref['micro']:.4f} / "
              f"F1 {ref['f1']:.6f}（来源 {ref['source']}"
              + (f"，缓存 {ref['resolved_cache']}" if ref.get("resolved_cache") else "") + "）")

    # --- 基线：用于 micro 辅助提示（不给 / 不存在 ⇒ 用 base_frozen 内置常量）--
    baseline_ref = refs.get("base_frozen")
    baseline_cache = args.baseline or (baseline_ref or {}).get("cache", "")
    baseline_micro: Optional[float] = None
    baseline_info: dict[str, Any] = {"cache": baseline_cache, "source": "builtin_constant"}
    resolved_baseline = resolve_cache_path(baseline_cache) if baseline_cache else None
    if resolved_baseline:
        try:
            base_metrics = evaluate_cache(resolved_baseline, args.semantics, tols,
                                          args.img_w, args.img_h, "baseline")
            baseline_micro = base_metrics["micro"]
            baseline_info = {"cache": baseline_cache, "resolved_cache": resolved_baseline,
                             "source": "cache_eval", "nano": base_metrics["nano"],
                             "micro": base_metrics["micro"], "f1": base_metrics["f1"]}
        except Exception as exc:
            print(f"[警告] --baseline 缓存评测失败（{type(exc).__name__}: {exc}）⇒ 回落到内置常量",
                  file=sys.stderr)
    if baseline_micro is None:
        baseline_micro = BUILTIN_REFS["base_frozen"]["micro"]
        baseline_info = {"cache": baseline_cache, "resolved_cache": None, "source": "builtin_constant",
                         "nano": BUILTIN_REFS["base_frozen"]["nano"], "micro": baseline_micro,
                         "f1": BUILTIN_REFS["base_frozen"]["f1"]}
    print(f"[基线] micro 辅助门槛 {baseline_micro:.4f}（来源 {baseline_info['source']}）")

    # --- 逐臂评测 -------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(args.candidate):
        name, cache, board_ms = parse_candidate_spec(spec, index)
        resolved = resolve_cache_path(cache)
        if not resolved:
            rows.append({"name": name, "cache": cache, "board_ms": board_ms,
                         "speed_verdict": SPEED_UNKNOWN, "recall_verdict": RECALL_FAIL,
                         "verdict": VERDICT_ERROR, "nano": None, "micro": None, "regular": None,
                         "small": None, "f1": None, "fp": None, "fn": None, "th": None,
                         "nano_tol4": None, "micro_tol4": None, "frames": None, "total_gt": None,
                         "deltas": {}, "ms_per_nano_pp": None, "micro_vs_baseline": None,
                         "micro_ge_baseline": None,
                         "error": f"缓存不存在或不可读: {cache}"})
            print(f"[{name}] ERROR：缓存不存在或不可读：{cache}", file=sys.stderr)
            continue
        try:
            metrics = evaluate_cache(resolved, args.semantics, tols, args.img_w, args.img_h, name)
        except Exception as exc:
            rows.append({"name": name, "cache": cache, "board_ms": board_ms,
                         "speed_verdict": SPEED_UNKNOWN, "recall_verdict": RECALL_FAIL,
                         "verdict": VERDICT_ERROR, "nano": None, "micro": None, "regular": None,
                         "small": None, "f1": None, "fp": None, "fn": None, "th": None,
                         "nano_tol4": None, "micro_tol4": None, "frames": None, "total_gt": None,
                         "deltas": {}, "ms_per_nano_pp": None, "micro_vs_baseline": None,
                         "micro_ge_baseline": None,
                         "error": f"缓存评测失败 {type(exc).__name__}: {exc}"})
            print(f"[{name}] ERROR：缓存评测失败 {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        row = build_row(name, cache, board_ms, metrics, baseline_micro, args.budget_ms,
                        args.nano_min, refs, ref_order, args.recall_eps)
        rows.append(row)
        print(f"[{name}] nano {row['nano']:.4f} / micro {row['micro']:.4f} / "
              f"F1 {row['f1']:.6f} ⇒ {row['verdict']}")

    # --- 输出 -------------------------------------------------------------
    print()
    print(format_table(rows, ref_order))
    print()
    print(format_conclusion(rows, args.budget_ms, args.nano_min, refs, ref_order))

    errors = [r["name"] for r in rows if r["verdict"] == VERDICT_ERROR]
    summary = {
        "pass": [r["name"] for r in rows if r["verdict"] == VERDICT_PASS],
        "fail": [r["name"] for r in rows if r["verdict"] == VERDICT_FAIL],
        "pass_with_gap": [r["name"] for r in rows if r["verdict"] == VERDICT_GAP],
        "error": errors,
    }
    if args.out:
        payload = {
            "budget_ms": args.budget_ms, "nano_min": args.nano_min, "recall_eps": args.recall_eps,
            "semantics": args.semantics, "tols": list(tols),
            "img_w": args.img_w, "img_h": args.img_h,
            "baseline": baseline_info,
            "rows": rows, "refs": refs, "summary": summary,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten {args.out}")

    return 2 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI。"""
    parser = argparse.ArgumentParser(
        description="联合门禁：板卡实测延迟 × NanoBench 分层召回（纯 CPU）")
    parser.add_argument("--candidate", action="append", default=[],
                        help="候选臂，形如 name=cache.pkl 或 name=cache.pkl=21.998（板卡 ms，可选）")
    parser.add_argument("--baseline", default="",
                        help="recall 基线缓存路径（缺省/不存在 ⇒ 用内置 base_frozen 常量）")
    parser.add_argument("--ref-labels", default=",".join(DEFAULT_REF_LABELS),
                        help="参照系名，逗号分隔（默认 base_frozen,RC2,delivery）")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS, help="板卡速度预算（ms）")
    parser.add_argument("--nano-min", type=float, default=DEFAULT_NANO_MIN, help="nano 召回下限")
    parser.add_argument("--recall-eps", type=float, default=RECALL_EPS,
                        help="召回门比较容差（默认 5e-05 = 0.005 pp，阈值末位半个 ULP；"
                             "设 0 即为严格 >=）")
    parser.add_argument("--out", default="", help="输出 JSON 路径（可选）")
    parser.add_argument("--semantics", default="nogmc", choices=["nogmc", "gmc"])
    parser.add_argument("--tols", default="4,8", help="匹配容差（px），必须含 4 与 8")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--selftest", action="store_true", help="跑内置常量的判定逻辑自检")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    # stdout 走管道时是块缓冲，stderr 不是 ⇒ 行缓冲化，否则 ERROR 行会跑到进度行前面
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - 非标准流（如 pytest 捕获）
        pass
    if args.selftest:
        return run_selftest(args.budget_ms, args.nano_min, args.out, args.recall_eps)
    if not args.candidate:
        print("[提示] 没有 --candidate，可用 --selftest 跑判定逻辑自检。", file=sys.stderr)
        return 0
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
