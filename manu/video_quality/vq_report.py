#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render the Chinese leadership report from the comparison artefacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

GOOD_LABEL = "好视频组（朗基天香20260903）"
BAD_LABEL = "坏视频组（纵横挂飞）"
PAIR_TITLE = {
    "ir": "红外 IR（源视频）",
    "vis": "可见光 VIS（源视频）",
    "osd": "地面站「挂飞」录屏（参考组，非源视频）",
}
BIAS_CLIPS = [
    "bad IR  120BT-04-红外02=/mnt/data/xxxu/纵横挂飞/120BT-04-红外02.mp4",
    "bad IR  130CT-02红外_IRT=/mnt/data/xxxu/纵横挂飞/130CT-02红外_IRT.mp4",
    "good IR VIDEO00007=/mnt/data/xxxu/朗基天香20260903/VIDEO00007_19700101_012038IR.h264",
    "good IR VIDEO00008=/mnt/data/xxxu/朗基天香20260903/VIDEO00008_19700101_012152IR.h264",
    "bad VIS 130CT-02可见光_OSDT=/mnt/data/xxxu/纵横挂飞/osd/130CT-02可见光_OSDT.mp4",
    "bad VIS 120BT-04-可见光=/mnt/data/xxxu/纵横挂飞/osd/120BT-04-可见光.mp4",
    "good VIS VIDEO00011=/mnt/data/xxxu/朗基天香20260903/VIDEO00011_19700101_010155VIS.h264",
    "good VIS VIDEO00015=/mnt/data/xxxu/朗基天香20260903/VIDEO00015_19700101_010552VIS.h264",
    "good VIS VIDEO00019=/mnt/data/xxxu/朗基天香20260903/VIDEO00019_19700101_011317VIS.h264",
]
STRONG = 0.70
WEAK = 0.55
REVERSE_REASON = {
    "snr_db": "糊图更平滑→信噪比反而更高，与画质直觉相反",
    "grad_energy_ratio": "糊图梯度能量更低，ρ 不是「脏」而是「糊」",
    "shift_frac": "坏片是机头朝地/悬停，比航拍好片更稳，「晃」不是失效原因",
    "shift_rate": "同上：坏片更稳",
    "shift_frac_max": "同上：红外坏片比航拍好片更稳，最大位移也更低",
    "stripe_idx_row": "条纹指数随画面变糊而下降，方向相反",
    "col_bias": "方向相反，且判别力弱",
    "dog_snr_max": "糊图上最像目标的区域对比度反而更高，方向相反",
    "dog_snr_top8": "同上",
    "sat_ratio": "坏片更灰更暗→近白像素更少，方向相反",
}
REVERSE_REASON_PAIR = {
    ("ir", "noise_sigma"): "糊图高频少→实测噪声更小；按 2.6 节折算回原生尺度后差距 4.5×，方向仍是「坏组更小」",
    ("vis", "noise_sigma"): "糊图高频少→实测噪声更小；按 2.6 节折算回原生尺度后方向不变",
}
METRIC_CN = {
    "blockiness": "块效应(压缩伪影)",
    "global_contrast": "全局对比度",
    "entropy": "灰度熵",
    "edge_density": "边缘密度",
    "lap_var": "拉普拉斯清晰度方差",
    "frame_mean": "平均亮度",
    "noise_sigma": "噪声 sigma",
    "snr_db": "图像信噪比(dB)",
    "grad_energy_ratio": "梯度能量比 rho",
    "shift_frac": "帧间位移(占画幅宽)",
    "shift_frac_max": "帧间最大位移(占画幅宽)",
    "shift_rate": "帧间位移速率(占画幅宽/秒)",
    "rot_deg": "帧间旋转(度/帧)",
    "rot_rate": "旋转速率(度/秒)",
    "scale_pct": "帧间缩放(%)",
    "illum_jump": "帧间亮度跳变",
    "stripe_idx_row": "行条纹指数",
    "stripe_idx_col": "列条纹指数",
    "row_bias": "行向非均匀性",
    "col_bias": "列向非均匀性",
    "dog_snr_max": "最强小目标对比度(SNR)",
    "dog_snr_top8": "前8强小目标对比度(SNR)",
    "sat_ratio": "过曝像素比",
    "dark_ratio": "死黑像素比",
    "flow_med_frac": "光流整体位移(占画幅宽)",
    "flow_resid_frac": "光流非刚性残差(占画幅宽)",
}


def fmt(value: float, digits: int = 4) -> str:
    if value is None or value != value:
        return "-"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def fmt_p(value: float) -> str:
    """Format an exact permutation p-value, keeping the small-sample resolution visible."""
    if value is None or value != value:
        return "-"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def fmt_ci(low: float, high: float) -> str:
    if low != low or high != high:
        return "-"
    return f"[{low:.3f}, {high:.3f}]"


def load(report_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Read the artefacts: the comparison rows already carry every statistic the report quotes."""
    comparison = json.loads((report_dir / "comparison.json").read_text(encoding="utf-8"))["comparison"]
    with (report_dir / "metrics_table.csv").open(encoding="utf-8") as handle:
        clips = list(csv.DictReader(handle))
    tail = []
    tail_path = report_dir / "tail_compare.csv"
    if tail_path.exists():
        with tail_path.open(encoding="utf-8") as handle:
            tail = list(csv.DictReader(handle))
    return comparison, clips, tail


def load_bias(report_dir: Path) -> dict[str, dict[str, float]]:
    """Parse ``resample_bias.txt`` into ``{clip file: {metric: canonical/native ratio}}``."""
    path = report_dir / "resample_bias.txt"
    bias: dict[str, dict[str, float]] = {}
    if not path.exists():
        return bias
    clip = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("--- "):
            head = line.split("(crop ->")[0].strip()
            clip = head.split()[-1] if head else ""
            bias.setdefault(clip, {})
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0] in ("noise_sigma", "snr_db", "grad_energy_ratio", "lap_var"):
            index = next((i for i, token in enumerate(parts) if token.startswith("ratio=")), None)
            if index is not None and index + 1 < len(parts):
                bias.setdefault(clip, {})[parts[0]] = float(parts[index + 1])
    return bias


MIXED_CLIP = "130CT-02可见光_OSDT"


def load_modality_facts(metrics_dir: Path) -> dict:
    """Read the modality probe facts of the mixed infrared/visible-light clip from the artefact.

    The report quotes how much of that clip is visible light and when the visible part starts.
    Those numbers have to come from the artefact (``out/<clip>.json``) instead of being typed in
    here, otherwise they cannot be checked against the data they describe.
    """
    path = metrics_dir / f"{MIXED_CLIP}.json"
    if not path.exists():
        raise SystemExit(f"[FAIL] 缺少 {path}：模态占比必须来自产物，不允许写死在报告里")
    data = json.loads(path.read_text(encoding="utf-8"))
    ratio = float(data["modality_vis_ratio"])
    first = data.get("modality_first_vis_seconds")
    return {
        "vis_pct": ratio * 100.0,
        "ir_pct": (1.0 - ratio) * 100.0,
        "samples": int(data.get("modality_samples", 0)),
        "coloured": int(data.get("modality_coloured", 0)),
        "first_vis_seconds": float(first) if first is not None else None,
        "first_vis_phrase": (
            f"开头约 {float(first) / 60.0:.0f} 分钟是红外灰度"
            if first is not None
            else "全片未出现彩色"
        ),
    }


def corrected_gap(
    pair: str,
    metric: str,
    comparison: list[dict],
    clips: list[dict],
    bias: dict[str, dict[str, float]],
) -> tuple[float, float] | None:
    """Group gap after undoing the canonical-640 scale bias, per clip.

    The canonical width *raises* the scale sensitive metrics; the raise differs per clip because it
    depends on the native width and on the content's own spectrum.  Undoing it per clip therefore
    needs the measured ratio of every clip, which ``resample_bias.txt`` provides.
    """
    row = next((r for r in comparison if r["pair"] == pair and r["metric"] == metric), None)
    if row is None or not bias:
        return None
    good = [c for c in clips if c["group"] == "good" and c["modality"] == pair and c["file"] in bias]
    bad = [c for c in clips if c["group"] == "bad" and c["kind"] == "source" and c["modality"] == pair
           and c["file"] in bias]
    if not good or not bad:
        return None
    good_factor = float(np.mean([bias[c["file"]][metric] for c in good]))
    bad_native = [float(c[metric]) / bias[c["file"]][metric] for c in bad]
    good_native = float(row["good_median"]) / good_factor
    native_bad_median = float(np.median(bad_native))
    if not native_bad_median or not good_native:
        return None
    return good_native / native_bad_median, good_factor

def table(rows: list[dict], columns: list[tuple[str, str]], align_right: set[str] | None = None) -> str:
    align_right = align_right or set()
    head = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def classify(row: dict) -> str:
    """Verdict from the window level cluster bootstrap and the video level exact permutation test.

    Two independent pieces of evidence have to agree before a metric is called effective:
    the *cluster* bootstrap over clips (the CI resamples clips and keeps all of their windows, so
    the correlation inside a clip is respected - the plain window level Mann-Whitney p would be
    pseudo-replication) and the *exact permutation* test on the video level numbers (the bad group
    holds only a handful of clips, so every label assignment is enumerated instead of assuming a
    normal approximation).
    """
    disc = row.get("window_discrimination", float("nan"))
    low = row.get("window_auc_cluster_ci_low", float("nan"))
    high = row.get("window_auc_cluster_ci_high", float("nan"))
    p_value = row.get("clip_p_exact", float("nan"))
    ratio = row.get("ratio", float("nan"))

    cluster_significant = low == low and high == high and (low > 0.5 or high < 0.5)
    clip_significant = p_value == p_value and p_value < 0.05
    if disc >= STRONG and (cluster_significant or clip_significant):
        # the reference method's practical-usability rule: a metric whose two group medians sit
        # within 1.2x of each other carries little operational value even when the distributions
        # happen not to overlap - the red line then has to be read off the distribution gap.
        aligned = direction_aligned(row)
        if aligned is False:
            # the two groups separate perfectly, but the bad group sits on the *good* side of the
            # metric (e.g. the blurry bad clips have LESS sensor noise).  Such a metric cannot
            # become an acceptance red line - it would reject the good footage.
            return "⚠️ 方向相反·不可作判据"
        if aligned is None:
            return "✅ 有效（描述性）"
        if ratio == ratio and ratio <= 1.2:
            return "⚠️ 统计有效·区分倍数小"
        return "✅ 有效"
    if disc == disc and disc >= WEAK:
        return "⚠️ 弱"
    return "❌ 无区分度（剔除）"


def direction_aligned(row: dict) -> bool | None:
    """Is the bad group on the physically worse side of the metric?

    ``expected_direction`` is +1 when a *higher* value means a worse clip (blockiness, noise,
    motion) and -1 when a *lower* value means a worse clip (sharpness, contrast, entropy).
    ``None`` means the metric has no expected direction (average brightness) and can only be used
    descriptively.
    """
    expected = row.get("expected_direction")
    try:
        expected = int(expected)
    except (TypeError, ValueError):
        return None
    if expected == 0:
        return None
    bad_high = row.get("separation") == "bad_high"
    return bad_high if expected == 1 else not bad_high


def render(report_dir: Path, out_path: Path, metrics_dir: Path) -> None:
    comparison, clips, tail = load(report_dir)
    modality = load_modality_facts(metrics_dir)
    good_n = sum(1 for c in clips if c["group"] == "good")
    bad_n = sum(1 for c in clips if c["group"] == "bad")
    total_frames = sum(int(c["frames"]) for c in clips)

    lines: list[str] = []
    lines.append("# 视频质量指标挖掘报告：好视频组 vs 坏视频组")
    lines.append("")
    lines.append(
        f"> **样本**：好视频 {good_n} 条 + 坏视频 {bad_n} 条，合计 {total_frames:,} 帧；"
        f"**执行机** `huangzhe@192.168.99.40:32222`（conda env `uav`）；"
        f"**产物目录** `/tmp/pycharm_project_10ae9e2e/manu/video_quality/report/`"
    )
    lines.append("")

    # ---------------- 0. 数据构成 ----------------
    lines.append("## 〇、数据构成（先把话说清楚）")
    lines.append("")
    bad_sources = [c for c in clips if c["group"] == "bad" and c["kind"] == "source"]
    bad_osd = [c for c in clips if c["group"] == "bad" and c["kind"] == "osd"]
    good_ir = [c for c in clips if c["group"] == "good" and c["modality"] == "ir"]
    good_vis = [c for c in clips if c["group"] == "good" and c["modality"] == "vis"]
    bad_ir = [c for c in bad_sources if c["modality"] == "ir"]
    bad_vis = [c for c in bad_sources if c["modality"] == "vis"]
    lines.append(
        f"- **好视频组**：{len(clips) - len(bad_sources) - len(bad_osd)} 条源视频"
        f"（红外 {len(good_ir)} 条 + 可见光 {len(good_vis)} 条），全部为 30 fps 裸 H.264。"
    )
    lines.append(
        f"- **坏视频组**：真正可比的**源视频只有 {len(bad_sources)} 条**"
        f"（红外 {len(bad_ir)} 条 + 可见光 {len(bad_vis)} 条，25 fps mp4）；"
        f"另有 **{len(bad_osd)} 条是操作员桌面的屏幕录像**，不是视频源。"
    )
    lines.append("")
    lines.append(
        "> **重要澄清 1**：`纵横挂飞/osd/` 下 3 个「挂飞」文件实测是**地面站操作员的桌面录屏**"
        "（画面里是 Windows 桌面、地面站 UI、MobaXterm 窗口），而不是摄像头视频流。"
        "证据见 `evidence_挂飞录屏实为桌面录屏_t300s.png`。"
        "因此它们**不参与主结论的统计**，只作为操作现场的佐证材料。"
    )
    lines.append("")
    lines.append(
        "> **重要澄清 2**：坏组文件名不可信——`130CT-02可见光_OSDT.mp4` 文件名写「可见光」，"
        f"实测是**红外与可见光来回切换的地面站录制**（{modality['first_vis_phrase']}，"
        f"**全片 {modality['vis_pct']:.1f}% 的时长是可见光**；采样口径：每 5 秒 1 帧，"
        f"{modality['coloured']}/{modality['samples']} 个采样点判为彩色，"
        f"见 `manu/video_quality/out/{MIXED_CLIP}.json` 的 `modality_vis_ratio`）。"
        "本报告的模态分组**按像素饱和度在全片时长上多数表决**自动判定，"
        "既不看文件名，也不只看开头几帧。"
    )
    lines.append("")
    lines.append(
        "> **重要澄清 3**：坏组红外源片存在**大面积黑边（pillarbox）**，"
        "例如 `120BT-04-红外02.mp4` 的有效画面只占 1920×1080 中间一条。"
        "黑边硬边缘会大幅抬高拉普拉斯/梯度能量，**不裁边就会把坏片「测成」清晰**——这是本次分析最关键的一个坑。"
    )
    lines.append("")

    # ---------------- 1. 结论 ----------------
    lines.append("## 一、一句话结论")
    lines.append("")
    effective = [r for r in comparison if r["pair"] in ("ir", "vis") and classify(r) == "✅ 有效"]
    reversed_rows = [r for r in comparison
                     if r["pair"] in ("ir", "vis") and classify(r) == "⚠️ 方向相反·不可作判据"]
    pairs_seen = {}
    for row in effective:
        pairs_seen.setdefault(row["metric"], set()).add(row["pair"])
    common = sorted(
        (metric for metric, pairs in pairs_seen.items() if pairs == {"ir", "vis"}),
        key=lambda metric: -min(r["window_discrimination"] for r in effective if r["metric"] == metric),
    )
    ir_only = sorted(
        (metric for metric, pairs in pairs_seen.items() if pairs == {"ir"}),
        key=lambda metric: -min(r["window_discrimination"] for r in effective if r["metric"] == metric),
    )
    if common:
        names = [f"**{METRIC_CN[metric]}**" for metric in common]
        lines.append(
            f"**真正把好视频和坏视频分开的不是「晃」，而是「糊 + 灰 + 压」**——"
            f"即 {'、'.join(names)} 这 {len(common)} 个指标，"
            f"它们在**红外与可见光两个模态上同时成立**（跨模态一致），"
            f"共同刻画的是同一件事：**画面被压糊了、对比度被压平了、细节被磨没了**。"
        )
        lines.append("")
        if ir_only:
            ir_names = "、".join(f"**{METRIC_CN[metric]}**" for metric in ir_only)
            lines.append(
                f"**红外模态另有 {len(ir_only)} 个同样有效的指标**：{ir_names}"
                f"（在可见光上区分度不足，故只用于红外）。"
                f"合计 **{len(common) + len(ir_only)} 个有效指标（核心红线）**可作准入红线（另有 2 条可见光弱红线只做告警），"
                f"全部落在第三节的红线表里。"
            )
            lines.append("")
        lines.append(
            f"**红线**：坏片在这几个指标上全部落在「更差」的一侧，红外一组的块效应/熵/对比度"
            f"**两组分布零重叠**（见第三节的阈值表）；按第三节的分级规则越线即判「质量不达标」，"
            f"不必先跑跟踪算法。"
        )
        lines.append("")
        def pick(pair: str, metric: str) -> dict:
            return next(r for r in comparison if r["pair"] == pair and r["metric"] == metric)

        lap_vis = pick("vis", "lap_var")
        shift_ir = pick("ir", "shift_frac")
        shift_max_ir = pick("ir", "shift_frac_max")
        shift_vis = pick("vis", "shift_frac")
        shift_max_vis = pick("vis", "shift_frac_max")
        rho_vis = pick("vis", "grad_energy_ratio")
        rho_ir = pick("ir", "grad_energy_ratio")
        lines.append(
            "**同时要向领导说明的一点**：参考方法脚本里那 **4 个指标**"
            "（帧间位移 P95、**帧间最大位移**、拉普拉斯清晰度 P5、红外粗糙度 ρ），"
            f"本批实测**只有一个原样站得住**——"
            f"**拉普拉斯清晰度方差 P5**（可见光组中位数 {fmt(lap_vis['good_median'])} → "
            f"{fmt(lap_vis['bad_median'])}，表观 **{lap_vis['ratio']:.1f}×**；"
            f"**其中约 1.7× 来自归一化尺度偏置，折算回原生尺度约 29×**，对外请用 29×，见 2.6）；"
            f"**两个位移口径（帧间位移 P95、帧间最大位移）都不能做准入判据，但原因分模态不同**："
            f"**红外侧是方向与画质直觉相反**（坏片比航拍好片**更稳**，位移 P95 判别力 "
            f"{shift_ir['window_discrimination']:.3f}、最大位移 {shift_max_ir['window_discrimination']:.3f}）；"
            f"**可见光侧方向是对的**（坏片整段确实更晃，最大位移 {shift_max_vis['ratio']:.2f}×），"
            f"但窗口级判别力只有 {shift_vis['window_discrimination']:.3f} / "
            f"{shift_max_vis['window_discrimination']:.3f}、**都未达 0.70 门槛**——"
            f"**这是「窗口级」口径的否决，不是说它分不开**：可见光坏样本只有 2 条，"
            f"视频级「完全可分」很可能是偶然；而且准入阈值还要用于**在线逐秒告警**（见 2.4 结论），"
            f"必须先过窗口级这一关。"
            f"**红外粗糙度 ρ** 同样分模态：**两个模态实测都是「坏组更低」**"
            f"（ρ 越低 = 梯度能量越少 = 画面越糊），"
            f"而 ρ 在本报告的方向口径是「**值越高越差**」，"
            f"**按 2.5 的判据（坏组须落在「更差」一侧），两个模态的 ρ 都属「方向相反」**；"
            f"**两模态剔除表象不同，差别在判别力门槛**："
            f"可见光 ρ 判别力 {rho_vis['window_discrimination']:.3f} 达标，"
            f"2.2 表判「⚠️ 方向相反·不可作判据」；"
            f"红外 ρ 判别力只有 {rho_ir['window_discrimination']:.3f}、**未达 0.70 门槛**，"
            f"2.1 表只判「⚠️ 弱」，因此没进 2.5 的方向相反清单。"
            f"**两个模态的 ρ 都不进红线**：ρ 测的是「糊」不是「脏」，"
            f"与已进入红线的清晰度 / 灰度类指标（{METRIC_CN['lap_var']}、{METRIC_CN['edge_density']}、"
            f"{METRIC_CN['global_contrast']}、{METRIC_CN['entropy']}）**信息冗余**——"
            f"四个指标都**不能拿来做准入判据**；"
            "**局部信杂比 SCR / 归一化位移**因两组目录里**没有任何 GT 框**而**无法按原样计算**（见第五节局限 2）。"
            f"本批额外补测的**噪声 sigma、图像信噪比**同样是**方向相反**"
            f"（归一化偏置已实测，但**不是**剔除原因，详见 2.5 / 2.6）。"
            f"合计 **{len(reversed_rows)} 个「指标对但方向反」的组合被剔除**。"
            f"**「晃」要分模态看**：红外的坏片其实比航拍好片**更稳**（方向相反）；"
            f"可见光的坏片整段确实**更晃**（{shift_vis['ratio']:.2f}×、视频级完全可分），"
            f"但窗口级判别力只有 {shift_vis['window_discrimination']:.3f}、未达 0.70 门槛，"
            f"因此只作参考、不进红线。**跨两个模态真正共同的失效特征是「糊 / 灰 + 压」**，"
            f"也就是上面那 {len(common)} 个跨模态一致的指标所刻画的。"
        )
        lines.append("")
        lines.append("各项定量阈值见第三节，分组统计见第二节。")
    lines.append("")

    # ---------------- 2. 分组统计 ----------------
    lines.append("## 二、分组统计与区分度")
    lines.append("")
    lines.append(
        "口径说明：两组分辨率/帧率都不同（好组 640×512@30 与 1920×1080@30，坏组 1920×1080@25/20），"
        "因此**清晰度/噪声类指标统一在裁掉黑边后缩放到宽 640 的归一化画幅上计算**，"
        "**运动类指标同时给出『占画幅宽比例』与『每秒』两个口径**。"
        "坏组 IR 源片有大面积黑边（pillarbox），若不裁剪，黑边硬边缘会把拉普拉斯能量抬高，"
        "把坏片『测成』清晰——这是本次分析中最容易踩的坑。"
    )
    lines.append("")
    lines.append(
        "**与参考方法的口径一致性**：帧间位移用 `cv2.phaseCorrelate`（FFT 相位相关）、"
        "红外粗糙度 ρ 用 `(Σ|∂x/∂y 梯度|) / Σ像素`——**两条都按参考方法原样实现**，"
        "只把位移的单位从 px 换成「占画幅宽比例」以便两组不同分辨率可比；"
        "分位口径也照参考方法：位移/条纹/粗糙度取 **P95**，清晰度/信噪比取 **P5**。"
    )
    lines.append("")
    for pair in ("ir", "vis", "osd"):
        rows = [r for r in comparison if r["pair"] == pair]
        if not rows:
            continue
        lines.append(f"### 2.{1 + ['ir', 'vis', 'osd'].index(pair)} {PAIR_TITLE[pair]}")
        lines.append("")
        if pair == "osd":
            lines.append(
                "> ⚠️ 本组为**操作员桌面录屏**，画面里混有地面站 UI 文字、按钮与终端窗口，"
                "纹理/清晰度类指标被 UI 严重污染，**仅作参考，不作为结论依据**。"
            )
            lines.append("")
        table_rows = []
        for row in rows:
            table_rows.append(
                {
                    "label": row["label_cn"],
                    "stat": row["stat"],
                    "good": fmt(row["good_median"]),
                    "bad": fmt(row["bad_median"]),
                    "ratio": (
                        "0/0（两组中位数均为 0）"
                        if float(row["good_median"]) == 0 and float(row["bad_median"]) == 0
                        else "∞（有一组中位数为 0）"
                        if float(row["good_median"]) == 0 or float(row["bad_median"]) == 0
                        else f'{row["ratio"]:.2f}×'
                    ),
                    "side": "高" if row["separation"] == "bad_high" else "低",
                    "clip_disc": f'{row["clip_discrimination"]:.3f}',
                    "clip_p": fmt_p(row.get("clip_p_exact", float("nan"))),
                    "cluster_ci": fmt_ci(row.get("window_auc_cluster_ci_low", float("nan")),
                                         row.get("window_auc_cluster_ci_high", float("nan"))),
                    "win_disc": f'{row["window_discrimination"]:.3f}',
                    "verdict": classify(row),
                }
            )
        lines.append(
            table(
                table_rows,
                [
                    ("label", "指标（序列级分位数）"),
                    ("stat", "取分位"),
                    ("good", "好组中位数"),
                    ("bad", "坏组中位数"),
                    ("ratio", "区分倍数"),
                    ("side", "坏组在"),
                    ("clip_disc", "判别力(视频级)"),
                    ("clip_p", "精确置换p(视频级)"),
                    ("win_disc", "判别力(1秒窗口)"),
                    ("cluster_ci", "窗口级 cluster CI"),
                    ("verdict", "判定"),
                ],
            )
        )
        lines.append("")
        n_good = rows[0]["n_good_clips"]
        n_bad = rows[0]["n_bad_clips"]
        n_gw = rows[0]["n_good_windows"]
        n_bw = rows[0]["n_bad_windows"]
        lines.append(
            f"*样本量：视频级 好 {n_good} 条 / 坏 {n_bad} 条；窗口级 好 {n_gw:,} 个 / 坏 {n_bw:,} 个（1 秒窗口）。*"
        )
        lines.append("")
        descriptive = [r for r in rows if classify(r) == "✅ 有效（描述性）"]
        if descriptive:
            names = "、".join(f"「{r['label_cn']}」" for r in descriptive)
            lines.append(
                f"> 表中标「✅ 有效（描述性）」的 {names} **只作描述性统计、不进红线候选集**"
                f"（没有「哪个方向算差」的物理含义，随场景/昼夜变化，不是画质缺陷）；"
                f"判定为「✅ 有效」的其余行才是候选集，红线见第三节。"
            )
            lines.append("")
    lines.append(
        "> **AUC 读法**：0.5 = 两组完全重叠（该指标没有区分度）；0.7 = 有明显区分；0.9+ = 近乎可分。"
        "本报告统一用 **AUC = P(好组 > 坏组)** 的原始值，另给 **判别力 = max(AUC, 1-AUC)**（恒 ≥ 0.5，越大越有区分度）"
        "与 **「坏组在 高/低」** 的方向列。"
    )
    lines.append(
        "> **判定口径（重要）**：**门槛用窗口级判别力 ≥ 0.70**（保证统计功效），"
        "显著性要求**窗口级 cluster bootstrap CI 不跨 0.5**（按视频重采样、保留该视频的全部窗口）"
        "**或**视频级精确置换检验 **p < 0.05**。"
        "**这个「或」是并列证据之一**：窗口级 CI 跨 0.5 但视频级 p<0.05 的指标仍可判有效"
        "（如参考组 `死黑像素比`：CI [0.000, 0.504]、p=0.0143）——它只用于参考组、**不进红线**。"
        "**视频级才是「接不接」这件事的单位**，但坏组只有 2 条时它的检验力有硬上限："
        "红外 6 好 + 2 坏只有 C(8,2)=28 种标签分配，**双侧**置换 p 的最小可能值就是 2/28 = 0.071，"
        "**红外侧永远不可能拿到 p<0.05**，所以红外的显著性证据由 cluster CI 承担（见表内 CI 列）。"
    )
    lines.append(
        "> 为什么不用窗口级的普通 Mann-Whitney p：同一条视频内的 1 秒窗口高度相关，把它们当独立样本是**伪重复**，"
        "会把置信度吹大。cluster bootstrap CI 跨 0.5，说明**窗口级的分离不可靠**（是伪重复吹出来的）；"
        "此时能不能用，只能看视频级 p。"
    )
    lines.append(
        "> **精确置换检验是双侧的**：按 |AUC − 0.5| 判定极端程度，所以「坏组偏高」型指标"
        "（如块效应）在完全可分时 p = 0.0714，**不会**因为 AUC=0 而显示成 1.0。"
    )
    lines.append("")

    # ---------------- 2.x 跟丢切片对比 ----------------
    if tail:
        lines.append("### 2.4 跟丢切片对比（坏片故障窗 vs 好片全局）")
        lines.append("")
        lines.append(
            "坏视频往往前几百帧稳、只在最后一段崩掉；整段平均会把恶劣帧稀释成正常值。"
            "下面把坏片的**故障切片**取出来，直接对比好片的**全局**统计量。"
        )
        lines.append("")
        lines.append(
            "* **tail 口径（headline）**：每条坏片取**最后 5 秒**。选窗时**不看任何指标**，"
            "因此没有选择偏差，可直接引用。*"
        )
        lines.append(
            "* **worst 口径**：每条坏片取晃动最大的 5 秒窗。它是**按位移指标挑出来的**，"
            "所以位移类指标在此口径下有**构造性上偏**，只能作参考。"
        )
        lines.append(
            "* **`max` 类指标（帧间最大位移）在本表的倍数被放大、不可直接对外引用**："
            f"本表「好组全局」对 `max` 取的是**整段视频的最大值**"
            f"（好组片长 IR {min(int(c['frames']) for c in good_ir)}~{max(int(c['frames']) for c in good_ir):,} 帧、"
            f"VIS {min(int(c['frames']) for c in good_vis)}~{max(int(c['frames']) for c in good_vis):,} 帧，"
            "哪怕只有一帧相位相关跳变就会被拉高），"
            "而「坏组尾窗」只有 125 帧 ⇒ 两者不是同口径。**2.1/2.2 表的视频级数字用的是"
            "「各 1 秒窗口最大值的中位数」**，那才是稳定口径；本表的 `max` 行只说明**方向**"
            "（坏片尾窗的最大位移也远小于好片的典型值）。"
        )
        lines.append("")
        for pair in ("ir", "vis", "osd"):
            rows = [r for r in tail if r.get("pair") == pair]
            if not rows:
                continue
            rows.sort(key=lambda r: -float(r.get("fold_tail") or 0))
            lines.append(f"**{PAIR_TITLE[pair]}**")
            lines.append("")
            table_rows = []
            for row in rows[:10]:
                table_rows.append({
                    "label": row["label_cn"],
                    "stat": row["stat"],
                    "good": fmt(float(row["good_global_median"])),
                    "tail": fmt(float(row["bad_tail_median"])),
                    "ratio_tail": f'{float(row["fold_tail"]):.2f}×',
                    "worst": fmt(float(row["bad_worst_median"])),
                    "ratio_worst": f'{float(row["fold_worst"]):.2f}×',
                    "verdict": row["verdict_tail"],
                })
            lines.append(
                table(
                    table_rows,
                    [
                        ("label", "指标"),
                        ("stat", "取分位"),
                        ("good", "好组全局中位数"),
                        ("tail", "坏组尾窗中位数"),
                        ("ratio_tail", "差距倍数(尾窗)"),
                        ("worst", "坏组最差窗中位数"),
                        ("ratio_worst", "差距倍数(最差窗)"),
                        ("verdict", "判定(尾窗)"),
                    ],
                )
            )
            lines.append("")
        lines.append(
            "> 判定规则沿用参考方法：两组中位数的**差距倍数**（方向对齐后取 max(r, 1/r)）"
            "**≥2×** 判为极具说服力并保留；**1.0×~1.2×** 判为无区分度予以剔除。"
        )
        lines.append("")
        lines.append(
            "> **两点必须一起讲，否则会被误读**："
            "① 本小节的判定只看**差距倍数**（不看方向），因此这里的「保留」**不代表能当红线**——"
            "**噪声 σ、梯度能量比 ρ** 虽然在本节被保留，但它们的**方向**与画质直觉相反"
            "（糊图高频更少 → 实测噪声更小、ρ 更低，2.5 已列），**依然不能作准入判据**；"
            "（2.6 的尺度偏置对这两项是**保守方向**，不改变该结论）；"
            "② 坏片的故障窗表现为**「静止 + 无纹理」**（位移比好片小两个数量级、清晰度差 65×），"
            "**不是「晃」**——这与第一节「坏片不是晃，而是糊」的结论是一致的："
            "挂飞时飞机多半是机头朝地/悬停，画面又糊又没细节，跟丢的是**目标**，不是被晃丢的。"
        )
        lines.append("")
        tail_rows = []
        for pair in ("ir", "vis"):
            for row in comparison:
                if row["pair"] != pair or classify(row) != "✅ 有效" or row["balanced_acc"] < 0.90:
                    continue
                tail_row = next((t for t in tail if t["pair"] == pair and t["metric"] == row["metric"]), None)
                if tail_row is None:
                    continue
                tail_median = float(tail_row["bad_tail_median"])
                threshold = float(row["threshold"])
                crossed = tail_median > threshold if row["bad_side"] == "high" else tail_median < threshold
                tail_rows.append(
                    {
                        "pair": PAIR_TITLE[pair],
                        "label": f'{row["label_cn"]} `{row["metric"]}`',
                        "tail": fmt(tail_median),
                        "thr": fmt(threshold),
                        "worst": fmt(float(tail_row["bad_worst_median"])),
                        "good": fmt(float(tail_row["good_global_median"])),
                        "cross": "**是（仍然越线）**" if crossed else "否",
                    }
                )
        if tail_rows:
            lines.append("**核心红线在坏片故障窗里还成不成立？**（这是必被问的一个问题，单独回答）")
            lines.append("")
            lines.append(
                table(
                    tail_rows,
                    [
                        ("pair", "模态"),
                        ("label", "核心红线"),
                        ("tail", "坏片尾窗中位数"),
                        ("worst", "坏片最差窗中位数"),
                        ("thr", "红线阈值"),
                        ("good", "好片全局中位数"),
                        ("cross", "尾窗是否仍越线"),
                    ],
                )
            )
            lines.append("")
            lines.append(
                "> **结论：核心红线在故障窗里照样越线**，"
                "说明它判的是「这一段视频已经不可用」，**不是「整条片子的平均值」**——"
                "所以接入前体检与在线实时告警都能用同一套阈值。"
            )
            lines.append("")

    # ---------------- 2.5 方向相反 / 剔除清单 ----------------
    lines.append(
        "### 2.5 被剔除的指标（重要结论：不是所有『能分开』的指标都能用）"
        "（**本表只统计源视频两组（红外/可见光）；桌面录屏参考组的标记见 2.3，不计入本表**）"
    )
    lines.append("")
    lines.append(
        "一个指标能把两组分开，不代表它能当准入红线：**只有坏组落在指标的『更差』一侧时，红线才有物理意义**。"
        "下表是本批实测里**能分开、但方向与画质直觉相反**的组合——"
        "如果照搬它们设阈值，会把好片误判为坏片。"
    )
    lines.append("")
    reverse_rows = []
    for pair in ("ir", "vis"):
        for row in comparison:
            if row["pair"] != pair or classify(row) != "⚠️ 方向相反·不可作判据":
                continue
            reverse_rows.append(
                {
                    "pair": "红外" if pair == "ir" else "可见光",
                    "label": row["label_cn"],
                    "good": fmt(row["good_median"]),
                    "bad": fmt(row["bad_median"]),
                    "disc": f'{row["window_discrimination"]:.3f}',
                    "why": REVERSE_REASON_PAIR.get(
                        (pair, row["metric"]), REVERSE_REASON.get(row["metric"], "坏组落在指标的『更好』一侧")
                    ),
                }
            )
    if reverse_rows:
        lines.append(
            table(
                reverse_rows,
                [
                    ("pair", "模态"),
                    ("label", "指标"),
                    ("good", "好组中位数"),
                    ("bad", "坏组中位数"),
                    ("disc", "判别力(窗口级)"),
                    ("why", "为什么不能作判据"),
                ],
            )
        )
        lines.append("")
    none_rows = []
    for pair in ("ir", "vis"):
        for row in comparison:
            if row["pair"] != pair or classify(row) != "❌ 无区分度（剔除）":
                continue
            none_rows.append(
                {
                    "pair": "红外" if pair == "ir" else "可见光",
                    "label": row["label_cn"],
                    "good": fmt(row["good_median"]),
                    "bad": fmt(row["bad_median"]),
                    "disc": f'{row["window_discrimination"]:.3f}',
                }
            )
    if none_rows:
        lines.append("**完全无区分度（判别力 ≈ 0.5，两组分布重叠）→ 直接剔除：**")
        lines.append("")
        lines.append(
            table(
                none_rows,
                [
                    ("pair", "模态"),
                    ("label", "指标"),
                    ("good", "好组中位数"),
                    ("bad", "坏组中位数"),
                    ("disc", "判别力(窗口级)"),
                ],
            )
        )
        lines.append("")
    red_metrics = []
    for row in comparison:
        if row["pair"] in ("ir", "vis") and classify(row) == "✅ 有效" and row["metric"] not in red_metrics:
            red_metrics.append(row["metric"])
    core_metrics = [
        metric
        for metric in red_metrics
        if min(r["balanced_acc"] for r in comparison if r["metric"] == metric and r["pair"] in ("ir", "vis")
               and classify(r) == "✅ 有效") >= 0.90
    ]
    weak_metrics = [metric for metric in red_metrics if metric not in core_metrics]
    core_rows = sum(
        1
        for row in comparison
        if row["pair"] in ("ir", "vis") and classify(row) == "✅ 有效" and row["balanced_acc"] >= 0.90
    )
    weak_rows = sum(
        1
        for row in comparison
        if row["pair"] in ("ir", "vis") and classify(row) == "✅ 有效" and row["balanced_acc"] < 0.90
    )
    lines.append(
        f"> **给领导的说法**：本批一共挖出 **{core_rows} 条核心红线 + {weak_rows} 条弱红线"
        f"（合计 {core_rows + weak_rows} 条阈值，见第三节表）**，"
        f"对应 **{len(core_metrics)} 个有效指标（核心画质红线）**："
        + "、".join(METRIC_CN[metric] for metric in core_metrics)
        + (f"；另有 {len(weak_metrics)} 个可见光弱红线指标（{'、'.join(METRIC_CN[m] for m in weak_metrics)}），"
           "只告警、不单独触发拒绝" if weak_metrics else "")
        + "。"
        "『噪声大、信噪比低、红外条纹脏、画面晃』这些直觉指标，"
        "在本批数据上要么没区分度、要么方向相反，**上报时要明确说清楚，避免被反问**。"
    )
    lines.append("")

    # ---------------- 2.6 尺度归一化的偏置 ----------------
    lines.append("### 2.6 口径自查：归一化画幅带来的尺度偏置（已量化）")
    lines.append("")
    lines.append(
        "两组原生分辨率不同（好组红外原生 640×512 不缩放；坏组红外裁黑边后 1280×1030 → 640；"
        "好组可见光 1920×1080 → 640；坏组 `130CT-02可见光_OSDT.mp4` 裁黑边后只有 1280×1030 → 640），"
        "而**缩放到宽 640 会整体抬高依赖像素尺度的指标**（下采样把高频折进画面）。"
        "**这个抬升系数已在 9/25 条片上逐条实测**（好 IR 2/6、好 VIS 3/12、坏组 4/4，"
        "其余按同内容框宽度外推；`manu/video_quality/vq_bias.py` → `report/resample_bias.txt`，"
        "每条片取 120 帧，同一批帧分别按原生分辨率与 640 归一化各算一次）："
    )
    lines.append("")
    bias_lines = [
        "| 片单（组 / 模态 / 裁剪后内容框宽） | 噪声 sigma | 图像信噪比 | 梯度能量比 rho | 拉普拉斯清晰度 |",
        "| --- | --- | --- | --- | --- |",
    ]
    bias = load_bias(report_dir)
    bias_clips = [c for c in clips if c["file"] in bias]
    bias_clips.sort(key=lambda c: (c["group"] != "good", c["modality"], c["file"]))
    for clip_row in bias_clips:
        ratios = bias[clip_row["file"]]
        crop_width = clip_row["crop"].split("x")[0]
        bias_lines.append(
            f'| {clip_row["group"]} / {clip_row["modality"]} / {clip_row["file"][:26]}（{crop_width} px 宽） | '
            f'{ratios.get("noise_sigma", float("nan")):.2f}× | {ratios.get("snr_db", float("nan")):.3f}× | '
            f'{ratios.get("grad_energy_ratio", float("nan")):.2f}× | {ratios.get("lap_var", float("nan")):.2f}× |'
        )
    lines.extend(bias_lines)
    lines.append("")
    ir_gap = corrected_gap("ir", "lap_var", comparison, clips, bias)
    vis_gap = corrected_gap("vis", "lap_var", comparison, clips, bias)
    ir_noise_gap = corrected_gap("ir", "noise_sigma", comparison, clips, bias)
    vis_lap = next((r for r in comparison if r["pair"] == "vis" and r["metric"] == "lap_var"), None)
    lines.append(
        "**这张表的三个读法（每条都有数）：**"
    )
    lines.append("")
    lines.append(
        f"① **红外侧：偏置是保守的，表里的差距被低估。** "
        f"坏组红外被 640 归一化抬高 {bias.get('120BT-04-红外02.mp4', {}).get('lap_var', float('nan')):.2f}×~"
        f"{bias.get('130CT-02红外_IRT.mp4', {}).get('lap_var', float('nan')):.2f}×，"
        f"好组红外原生 640 根本不缩放（1.00×）。"
        f"把抬升折算回原生尺度后，红外拉普拉斯清晰度的差距是 "
        f"**{ir_gap[0]:.1f}×**（表里写的 {next(r for r in comparison if r['pair'] == 'ir' and r['metric'] == 'lap_var')['ratio']:.2f}×）"
        f"——**真实差距比表中更大，结论方向不会因偏置反转**。"
        if ir_gap
        else "① 红外侧偏置数据缺失。"
    )
    lines.append("")
    lines.append(
        f"② **可见光侧：偏置方向相反，表里的倍数要打折。** "
        f"好组可见光是 1920 px 宽 → 640（拉普拉斯抬高 6.97~7.21×），"
        f"而坏组 `130CT-02可见光_OSDT.mp4` 的内容框只有 1280 px 宽（只抬高 2.08×）——"
        f"**两组被抬高的幅度不一样**（这条片的内容框是在它开头的红外时段检测出来的，裁剪尺度与好组不同）。"
        f"折算回与好组相同的原生尺度后，可见光清晰度的差距约 **{vis_gap[0]:.0f}×**"
        f"（表里写的 {vis_lap['ratio']:.1f}×）。"
        f"**差距依然远大于 1、结论不变，但不能拿 51× 这个数字对外说。**"
        f"（折算口径：好组侧取 3 条实测好片系数的**均值**，坏组侧**逐片去偏后取中位数**；"
        f"好组侧改用中位数口径的结果是 29.13×，不改变结论）"
        if vis_gap and vis_lap
        else "② 可见光侧偏置数据缺失。"
    )
    lines.append("")
    lines.append(
        "**② 的一条必要说明（`130CT-02可见光_OSDT.mp4` 的系数来自它的红外段）**："
        f"`vq_bias.py` 每条片只取**前 120 帧**（25 fps ≈ 前 4.8 s）。实测该片"
        f"{modality['first_vis_phrase']}（首个彩色采样点在第 {modality['first_vis_seconds']:.0f} s，"
        "故前 120 帧必然是纯灰度＝红外），且它这 120 帧的原生值与纯红外片 `130CT-02红外_IRT` **逐项重合**"
        "（梯度能量比 0.00792 vs 0.00792、噪声 sigma 0.04207 vs 0.04209、清晰度 18.97 vs 19.06）"
        f"⇒ **它的 2.08× 抬升系数其实是红外内容的系数**，而被折算的生产统计量是全片"
        f"（{modality['vis_pct']:.1f}% 可见光）。"
        "**方向是保守的**：若改按好组可见光内容的系数（6.97~7.21×）折算，差距会**更大**（≈45×），"
        "所以上面那个 **29× 是偏保守的下界**，结论方向不变。"
    )
    lines.append("")
    lines.append(
        f"③ **噪声 σ / ρ / 信噪比被剔除，理由是「方向」而不是「偏置」。** "
        f"实测显示把抬升折算回原生尺度后，红外噪声 σ 的差距是 **{ir_noise_gap[0]:.1f}× 且方向仍是「坏组更小」**"
        f"（好组 0.3353 vs 坏组 0.138）——"
        f"**方向反转是真实的，不是归一化造成的假象**，所以这三个指标确实不能作判据。"
        if ir_noise_gap
        else "③ 噪声 σ 折算数据缺失。"
    )
    lines.append("")

    # ---------------- 3. 红线阈值 ----------------
    lines.append("## 三、建议的工程准入红线")
    lines.append("")
    lines.append(
        "阈值取法：取**两组内侧尾部的中点**——"
        "**坏组偏高型**用 (好组 P90 + 坏组 P10)/2，**坏组偏低型**用 (好组 P10 + 坏组 P90)/2。"
        "零重叠判据同理分方向：坏组偏高型看「好组最大值 < 坏组最小值」，坏组偏低型看「好组最小值 > 坏组最大值」。"
    )
    lines.append("")
    high_example = next((r for r in comparison if r["pair"] == "ir" and r["metric"] == "blockiness"), None)
    low_example = next((r for r in comparison if r["pair"] == "ir" and r["metric"] == "entropy"), None)
    if high_example and low_example:
        lines.append(
            f"**当场可复算的两个例子**：块效应（坏组偏高）= (好组 P90 {high_example['good_p90']:.4g} + "
            f"坏组 P10 {high_example['bad_p10']:.4g}) / 2 = **{high_example['threshold']:.4g}**；"
            f"灰度熵（坏组偏低）= (好组 P10 {low_example['good_p10']:.4g} + "
            f"坏组 P90 {low_example['bad_p90']:.4g}) / 2 = **{low_example['threshold']:.4g}**。"
        )
        lines.append("")
    # a metric can separate the two groups perfectly and still be useless as a red line, so the
    # candidate set has to be spelled out in prose: the descriptive-only verdicts (average
    # brightness has no "worse" direction) and the window level gate are both part of the rule.
    descriptive_rows = [
        r for r in comparison if r["pair"] in ("ir", "vis") and classify(r) == "✅ 有效（描述性）"
    ]
    descriptive_note = ""
    if descriptive_rows:
        names = "、".join(
            f"红外「{r['label_cn']}」" if r["pair"] == "ir" else f"可见光「{r['label_cn']}」"
            for r in descriptive_rows
        )
        best_bal = max(r.get("balanced_acc", float("nan")) for r in descriptive_rows)
        descriptive_note = (
            f"**{names}只作描述性统计、不进候选集**——它没有「哪个方向算差」的物理含义"
            f"（随场景/昼夜变化），不是画质缺陷；即使阈值处平衡准确率有 {best_bal:.3f} 也不做红线，"
            f"否则会按场景/昼夜误拒好片（2.1/2.2 表已标「✅ 有效（描述性）」）。"
        )

    # only the rows that were dropped *by the window level gate* - a metric that is excluded
    # because its direction is reversed is a different story and must not be mixed in here.
    motion_rows = sorted(
        (
            r
            for r in comparison
            if r["pair"] == "vis"
            and r.get("balanced_acc", 0.0) >= 0.90
            and classify(r) == "⚠️ 弱"
        ),
        key=lambda r: r.get("window_discrimination", 0.0),
    )
    motion_note = ""
    if motion_rows:
        # two different "video level" numbers exist for the same row - the discrimination
        # (2.1/2.2 table column) and the balanced accuracy at the threshold - so both are named
        # explicitly here to keep the two tables comparable at a glance.
        details = "、".join(
            f"「{r['label_cn']}」{r['balanced_acc']:.3f} / {r['window_discrimination']:.3f}"
            for r in motion_rows
        )
        clip_discs = " / ".join(f"{r['clip_discrimination']:.3f}" for r in motion_rows)
        motion_note = (
            f"**注意**：可见光的 {details}（依次为**阈值处平衡准确率 / 窗口级判别力**）——"
            f"它们的**视频级判别力**分别是 {clip_discs}、看着能分开，"
            f"但窗口级判别力都没到 0.70、**没进候选集**，所以都不进红线——"
            f"级别是「先过候选集、再分级」，不是只看平衡准确率。"
            f"（红外那几个位移口径没进红线是**另一个原因**：方向与画质直觉相反，见 2.1 表。）"
        )

    lines.append(
        "红线分两级。**候选集**＝2.1/2.2 表里判定为「✅ 有效」的指标"
        "（**不含「✅ 有效（描述性）」**；窗口级判别力 ≥ 0.70、显著性成立、**方向正确**、两组中位数拉开 > 1.2×）；"
        "**在候选集内**再按阈值处平衡准确率分级："
        "**核心红线**（平衡准确率 ≥ 0.90）用于**拒绝接入**；"
        "**弱红线**（平衡准确率 < 0.90，本批只有可见光的 2 条）"
        "**只做告警**，命中不单独触发拒绝——因为它们在本批样本上确实会误报（见 3.1）。"
        + descriptive_note
        + motion_note
    )
    lines.append("")
    red_rows = []
    for pair in ("ir", "vis"):
        for row in comparison:
            if row["pair"] != pair or classify(row) != "✅ 有效":
                continue
            side = row["bad_side"]
            if side == "high":
                good_edge, bad_edge = row["good_max"], row["bad_min"]
            else:
                good_edge, bad_edge = row["good_min"], row["bad_max"]
            red_rows.append(
                {
                    "pair": PAIR_TITLE[pair],
                    "label": f'{row["label_cn"]} `{row["metric"]}`',
                    "tier": "核心" if row["balanced_acc"] >= 0.90 else "弱",
                    "stat": row["stat"],
                    "side": "高" if side == "high" else "低",
                    "threshold": fmt(row["threshold"]),
                    "clean": "是（两组零重叠）" if row["separated"] else "否（有少量重叠）",
                    "good_edge": fmt(good_edge),
                    "bad_edge": fmt(bad_edge),
                    "bal": f'{row["balanced_acc"]:.3f}',
                    "loco": f'{row["loco_balanced_acc_min"]:.3f}',
                    "auc": f'{row["window_discrimination"]:.3f}',
                }
            )
    if red_rows:
        lines.append(
            table(
                red_rows,
                [
                    ("pair", "模态"),
                    ("tier", "级别"),
                    ("label", "指标"),
                    ("stat", "取分位"),
                    ("side", "坏组在"),
                    ("threshold", "建议红线阈值"),
                    ("good_edge", "好组实测极值"),
                    ("bad_edge", "坏组实测极值"),
                    ("clean", "好/坏零重叠"),
                    ("bal", "阈值处平衡准确率"),
                    ("loco", "留一法平衡准确率"),
                    ("auc", "判别力(窗口级)"),
                ],
            )
        )
        lines.append("")
        loco_core = [
            (row["label_cn"], float(row["loco_balanced_acc_min"]))
            for row in comparison
            if row["pair"] in ("ir", "vis") and classify(row) == "✅ 有效" and row["balanced_acc"] >= 0.90
        ]
        weakest = min(loco_core, key=lambda item: item[1]) if loco_core else None
        ir_win = next((r for r in comparison if r["pair"] == "ir"), {})
        vis_win = next((r for r in comparison if r["pair"] == "vis"), {})
        lines.append(
            "**阈值来自哪组样本、样本量多少**：所有阈值**只用源视频**（3 条桌面录屏不参与任何阈值标定）——"
            f"**红外阈值来自好组 {len(good_ir)} 条 + 坏组 {len(bad_ir)} 条**"
            f"（1 秒窗口 好 {ir_win.get('n_good_windows', 0):,} / 坏 {ir_win.get('n_bad_windows', 0):,}），"
            f"**可见光阈值来自好组 {len(good_vis)} 条 + 坏组 {len(bad_vis)} 条**"
            f"（1 秒窗口 好 {vis_win.get('n_good_windows', 0):,} / 坏 {vis_win.get('n_bad_windows', 0):,}）。"
            "阈值取的是**视频级数字**（每条片 = 各 1 秒窗口分位的中位数）的 P90/P10，"
            "所以**每条片只贡献一个数、与片长无关**（343 帧的好片与 5 万帧的坏片可比）。"
        )
        lines.append(
            "**「留一法平衡准确率」这一列要怎么看**：它把**每条片轮流拿掉再重算阈值与平衡准确率**，"
            "取其中最差的一次——也就是**这套阈值的外推下界**。"
            + (f"本批最脆的是 **{weakest[0]}（{weakest[1]:.3f}）**，" if weakest else "")
            + "**红外侧的三条核心红线留一后只剩 0.667~0.750**，"
            "原因是坏组只有 2 条：拿掉 1 条就只剩 1 条可判，阈值自然失准。"
            "**这正是「上线前必须扩样重新标定」的直接证据**，不是可以忽略的细节。"
        )
        lines.append("")
        lines.append(
            "> 表中「好组实测极值 / 坏组实测极值」两列是**方向感知的组内极值**（用于判断两组是否零重叠），"
            "**不是**用来算阈值的 P10/P90（阈值算法见上面的两个例子）。"
        )
        lines.append("")
        lines.append(
            "**用法（分级）**：**该模态的全部核心红线同时越线 → 拒绝接入**；"
            "**命中 1 条及以上核心红线（但没全部越线）→ 人工复核**（含只越 1~2 条、或越了 3~4 条的情况）；"
            "**仅命中弱红线 → 通过，但必须记录并抽检**（弱红线的阈值处平衡准确率只有 0.708、**TPR 仅 0.50**——"
            "2 条坏 VIS 里每条弱红线都只抓到 1 条（`120BT-04-可见光.mp4` 的帧间亮度跳变 0.001587 < 0.001680、"
            "行条纹指数 32.99 < 33.53，均不触发）；**本批能拒绝坏片完全靠核心红线全部越线**，"
            "弱红线只作提示、不作判据；本批 12 条好 VIS 里 2 条命中弱红线，属正常误报率）；"
            "一条不越 → 通过。"
            "分级规则与它在**本批 25 条样本上的回溯结果**见 3.1——"
            "**任何单一阈值都存在误报，必须按条数与幅度分级使用**。"
        )
        lines.append("")

    # ---------------- 3b. 逐条视频红线命中 ----------------
    lines.append("### 3.1 用第三节的红线回溯判定本批 25 条视频")
    lines.append("")
    thresholds: dict[tuple[str, str], tuple[float, str, str, str]] = {}
    for pair in ("ir", "vis"):
        for row in comparison:
            if row["pair"] != pair or classify(row) != "✅ 有效":
                continue
            tier = "核心" if row["balanced_acc"] >= 0.90 else "弱"
            thresholds[(pair, row["metric"])] = (row["threshold"], row["bad_side"], tier, row["label_cn"])
    hit_rows = []
    for clip in clips:
        pair = "ir" if clip["modality"] == "ir" else ("vis" if clip["modality"] == "vis" else "osd")
        core_hits, weak_hits = [], []
        for (thr_pair, metric), (threshold, side, tier, label_cn) in thresholds.items():
            if thr_pair != pair:
                continue
            value = float(clip[metric])
            hit = value > threshold if side == "high" else value < threshold
            if hit:
                breach = (value - threshold) / threshold if side == "high" else (threshold - value) / threshold
                text = f"{label_cn}={value:.4g}(越线{breach:+.1%})"
                (core_hits if tier == "核心" else weak_hits).append(text)
        n_core = len(core_hits)
        n_weak = len(weak_hits)
        core_total = sum(1 for key in thresholds if key[0] == pair and thresholds[key][2] == "核心")
        if n_core == 0 and n_weak == 0:
            action = "通过"
        elif n_core == 0:
            action = "通过（弱红线告警）"
        elif n_core == core_total:
            action = "建议拒绝接入"
        else:
            action = "建议人工复核"
        hit_rows.append(
            {
                "group": "好" if clip["group"] == "good" else "坏",
                "kind": "源视频" if clip["kind"] == "source" else "桌面录屏",
                "mod": clip["modality"],
                "file": clip["file"],
                "n_core": n_core,
                "n_weak": n_weak,
                "hits": "；".join(core_hits + weak_hits) if (core_hits or weak_hits) else "—",
                "action": action,
            }
        )
    hit_rows.sort(key=lambda r: (r["group"] != "好", -r["n_core"], -r["n_weak"]))
    lines.append(
        table(
            hit_rows,
            [
                ("group", "组"),
                ("kind", "类型"),
                ("mod", "模态"),
                ("file", "文件"),
                ("n_core", "命中核心红线"),
                ("n_weak", "命中弱红线"),
                ("hits", "命中的具体指标"),
                ("action", "建议动作"),
            ],
        )
    )
    lines.append("")
    tally: dict[tuple[str, str], int] = {}
    for row in hit_rows:
        if row["kind"] != "源视频":
            continue
        tally[(row["group"], row["action"])] = tally.get((row["group"], row["action"]), 0) + 1

    def count_action(group: str, action_prefix: str) -> int:
        return sum(n for (grp, act), n in tally.items() if grp == group and act.startswith(action_prefix))

    good_reject = count_action("好", "建议拒绝")
    good_review = count_action("好", "建议人工复核")
    good_pass = count_action("好", "通过")
    weak_alerts = count_action("好", "通过（弱红线告警）")
    bad_reject = count_action("坏", "建议拒绝")
    bad_review = count_action("坏", "建议人工复核")
    bad_pass = count_action("坏", "通过")
    lines.append(
        "**回溯判定汇总（只用源视频，桌面录屏不参与）：**"
        f"好组 18 条 → **拒绝 {good_reject}、人工复核 {good_review}、通过 {good_pass}**"
        f"（其中 {weak_alerts} 条有弱红线告警）；"
        f"坏组 4 条 → **拒绝 {bad_reject}、人工复核 {bad_review}、通过 {bad_pass}**。"
        f"即**自动拒绝在本批样本上误拒好片 {good_reject}/18、命中坏片 {bad_reject}/4**"
        f"（自动放行 {bad_pass} 条坏片；{bad_review} 条坏片降级到人工复核，不会被自动放行）。"
    )
    lines.append("")
    lines.append(
        "> **必须一起说明的边界情况**：好组的 `VIDEO00007_19700101_012038IR.h264` 命中了 3/5 条红外核心红线"
        "（拉普拉斯清晰度 37.49、边缘密度 0.01203，都低于坏组的最大值 49.44 / 0.01209），"
        "而坏组的 `130CT-02红外_IRT.mp4` 恰好命中全部 5 条。"
        "**这两片在「清晰度 / 边缘密度」上是真的重叠的**——"
        "该好片大概率是拍到了大面积均匀天空/地面（**场景本身没纹理**），而不是画质差。"
        "这说明**无参考全图指标无法区分「画质差」与「场景没内容」**，"
        "所以规则必须分级、并且必须保留「人工复核」这一档。"
    )
    lines.append("")
    lines.append(
        "> 该表是**用第三节阈值对本批样本做的回溯判定**，用于验证阈值可用性；"
        "上线时应把阈值固化到接入前的自动体检里。桌面录屏组不参与判定（画面被 UI 污染）；"
        "**阈值与分级规则都是在只有 4 条坏片的样本上定的，上线前必须用更多样本重新标定**。"
    )
    lines.append("")

    # ---------------- 4. 落地建议 ----------------
    lines.append("## 四、落地建议")
    lines.append("")
    lines.append("1. **接入前体检**：对每个新到的视频片段跑 `manu/video_quality/vq_metrics.py`，按第三节阈值做**分级判定**（分红外/可见光两套阈值）。")
    lines.append("2. **三级处置**（与 3.1 的回溯结果完全对应）：")
    lines.append(f"   - **该模态的全部核心红线都越线 → 拒绝接入**：本批实测命中坏片 {bad_reject}/4、误拒好片 {good_reject}/18；"
                 "这类视频画质已不可用，继续投入跟踪算力是浪费。")
    lines.append(f"   - **命中 1 条及以上核心红线（但没全部越线）→ 人工复核**：本批落在这里的好片 {good_review} 条、坏片 {bad_review} 条都**不自动放行**，"
                 "因为好片与坏片在边界区是真的重叠的（见 3.1 的边界情况说明）。")
    lines.append(f"   - **仅命中弱红线 → 通过，但必须记录并抽检**：本批 {weak_alerts} 条好 VIS 命中弱红线"
                 f"（误报率 {weak_alerts}/{len(good_vis)}；弱红线是可见光专用，分母是好 VIS 总数），"
                 f"属正常波动，不阻断接入（**弱红线在坏片上 TPR 仅 0.50，只作提示、不作判据**，见第三节用法与 3.1）。")
    lines.append("   - **一条不越 → 通过**。")
    lines.append("   - 注意：拒绝的依据是「画质已不可用」，**不等于「算法一定能跟丢」**——因果关系要靠本节第 5 条的对照实验来证。")
    lines.append("3. **上线前必须先补两件事**：")
    lines.append("   - **扩样重新标定阈值**：现在只有 4 条坏片，红外侧三条核心红线的**留一法平衡准确率只有 0.667~0.750**"
                 "（见第三节），**必须补样本重标**；")
    lines.append("   - **固定裁剪与缩放策略**：坏片 `130CT-02可见光_OSDT.mp4` 裁剪后内容框 1280 px 宽、好组可见光 1920 px 宽，"
                 "两者被 640 归一化抬高的幅度差 3 倍（见 2.6），**接入前必须统一，否则阈值不可迁移**。")
    lines.append("4. **补齐硬件/链路**：黑边（pillarbox）是采集/推流链路的问题，不是算法问题，建议在源头侧统一裁剪；本报告已把黑边裁掉后再评估，说明**裁边即可挽回一部分指标**。")
    lines.append("5. **下一步要做的实验（本报告未覆盖）**：")
    lines.append("   - 用当前交付模型（`trial_0080` / RC2）在两组视频上**实测 F1 / 跟踪丢失率**，把视频质量指标与算法实际失效直接挂钩；")
    lines.append("   - 补 **GT 框** 后计算局部信杂比 SCR 与归一化位移（位移/目标尺寸），这是小目标跟踪最直接的可分辨性指标；")
    lines.append("   - 对越线视频做**后处理增强**（更激进的阈值/多帧融合/重检测）是否能救回，作为算法侧的对照实验。")
    lines.append("")

    # ---------------- 5. 局限 ----------------
    lines.append("## 五、本结论的局限（必须一起汇报）")
    lines.append("")
    lines.append(f"1. **样本量小**：坏组源视频只有 {len(bad_sources)} 条（红外 {len(bad_ir)} + 可见光 {len(bad_vis)}，"
                 f"另有 {len(bad_osd)} 条桌面录屏不算源视频），视频级统计量置信区间宽，结论主要依赖 1 秒窗口级样本；"
                 "**第三节的阈值与 3.1 的分级规则都是在这 4 条坏片上标定的，上线前必须扩样重标**；")
    lines.append("2. **没有 GT 框，故未算真值局部信杂比 SCR**：已实测 `find /mnt/data/xxxu -maxdepth 3` "
                 "在这两组视频目录里**不存在任何 .txt / .json / .yaml / .xml 标注或初始框文件**，"
                 "因此参考方法里的「用目标框算局部信杂比 SCR = |μt-μb|/σb」**无法按原样执行**。"
                 "本次改用 **DoG 小目标响应 ÷ 噪声 σ**（`dog_snr_max` / `dog_snr_top8`，即无框 SCR 代理）作为替代口径，"
                 "它对画面中最显著的小目标区域取信噪比，但不依赖人工框。"
                 "**归一化位移（位移/目标尺寸）同样因缺框未算**——这两项对小目标跟踪比全图指标更直接，"
                 "**补齐标注后必须重算**；");
    lines.append("3. **『噪声』指标的口径限制**：Immerkær 类估计在可见光上会把纹理当噪声（好组 VIS 的噪声估计值明显高于 IR），"
                 "因此该指标**只在同模态内部横向比较**才有意义，不能跨模态比较；")
    lines.append("4. **模态判定按像素饱和度而非文件名**：实测 `130CT-02可见光_OSDT.mp4` 名字写「可见光」，"
                 f"实测却是红外/可见光来回切换的混合录制（{modality['vis_pct']:.1f}% 可见光）。"
                 "本报告按**全片时长多数表决**把它划入可见光组；"
                 f"代价是这条片的逐帧指标里混有约 {modality['ir_pct']:.1f}% 的红外帧，"
                 "若改成只取可见光时段，可见光组的红线会更保守（需要重跑指标提取才能验证）。")
    lines.append(f"5. **可见光一侧尤其薄弱**：**可见光只有 {len(bad_vis)} 条坏样本**，"
                 "因此可见光一侧的结论只能算『假设级』，需要补样本才能定稿；"
                 "红外一侧好样本更多（{1} 条），但坏样本同样只有 {0} 条，**两侧都需扩样**。".format(len(bad_ir), len(good_ir)))
    lines.append("6. **相关性而非因果**：本报告给出的是「哪些画质指标能预测视频不可用」的统计判据，"
                 "不等于「该指标就是跟丢的原因」——因果需要第四节第 5 条的对照实验来证。")
    lines.append("7. **全图无参考指标分不清「画质差」与「场景没内容」**：好组的 `VIDEO00007_19700101_012038IR.h264` "
                 "在拉普拉斯清晰度与边缘密度上落进了坏组区间（3.1 的边界情况），"
                 "原因是它大概率拍到了大面积均匀天空/地面。"
                 "**解法是补 GT 框算目标局部指标**（局限 2）或**把指标改成局部 ROI 统计**，本批样本做不到；"
                 "因此当前红线只适合做**首轮筛查**，不能直接做自动拒绝的唯一依据。")
    lines.append("8. **可见光的归一化尺度并不统一**：坏组 `130CT-02可见光_OSDT.mp4` 裁剪后内容框 1280 px 宽，"
                 "好组可见光是 1920 px 宽，两者被缩放到宽 640 时**抬升系数差 3 倍**"
                 "（拉普拉斯 2.08× vs 6.97~7.21×，见 2.6）。"
                 "**可见光清晰度的表观 51.4× 里有约 1.7× 来自这个尺度差**，折算回原生尺度约 29×（结论方向不变，但倍数要打折）；"
                 "**接入前必须统一裁剪/缩放策略，否则可见光阈值不可迁移**。"
                 "另需注意：该片偏置实测只取了开头 120 帧（≈4.8 s），实测落在它的**红外时段**"
                 "（与纯红外片 `130CT-02红外_IRT` 的原生值逐项重合），故 2.08× 是红外内容系数；"
                 "若按好组可见光内容系数折算差距会更大（≈45×），**29× 是偏保守的下界**。")
    lines.append("9. **坏组源文件本身有损坏**：实测 `120BT-04-红外02.mp4` 的**文件尾被截 5,205 字节**"
                 "（最后一个 mdat 声明 676,711 B、实际只有 671,506 B），解码报 "
                 "`Invalid NAL unit size (4343 > 4249)`。"
                 "**模态判定不受影响**（采样覆盖全程、全判红外），但**逐帧指标是在受损码流上算的**"
                 "（尾部丢 2 帧）——属**源数据质量问题、不是算法问题**，上线前应要求重新导出一份。"
                 "另：**短片的模态采样点很少**（2~18 个），引用其模态占比时必须带样本数"
                 "（已落盘在各逐片 json 的 `modality_samples` / `modality_coloured`）。")
    lines.append("")

    # ---------------- 6. 图表 ----------------
    lines.append("## 六、图表")
    lines.append("")
    lines.append("![红外模态指标区分度](chart_metrics_ir.png)")
    lines.append("")
    lines.append("![可见光模态指标区分度](chart_metrics_vis.png)")
    lines.append("")
    lines.append("![全部指标 AUC 排序](chart_auc_ir.png)")
    lines.append("")
    lines.append("![可见光模态 AUC 排序](chart_auc_vis.png)")
    lines.append("")
    lines.append("![红外逐秒质量指标时间线](chart_timeseries_ir.png)")
    lines.append("")
    lines.append("![可见光逐秒质量指标时间线](chart_timeseries_vis.png)")
    lines.append("")
    lines.append("![挂飞录屏证据：实为操作员桌面录屏](evidence_挂飞录屏实为桌面录屏_t300s.png)")
    lines.append("")
    lines.append("> 图中横轴为组别（好/坏），纵轴为该指标在序列级分位数上的取值；"
                 "标题里的 AUC 视频级/窗口级分别是保守口径与统计功效口径。")
    lines.append("")
    lines.append("> 另有 `chart_metrics_osd.png` / `chart_auc_osd.png` / `chart_timeseries_osd.png` "
                 "对应桌面录屏组，**因画面被地面站 UI 污染、不作为结论依据，故不列入正文**。")
    lines.append("")

    # ---------------- 7. 复现 ----------------
    lines.append("## 七、复现命令（可直接复制）")
    lines.append("")
    lines.append("```bash")
    lines.append("# 执行机：huangzhe@192.168.99.40:32222    conda env: uav")
    lines.append("cd /tmp/pycharm_project_10ae9e2e")
    lines.append("source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav")
    lines.append("# 1) 逐帧指标提取（25 条片，4 worker，线程钉 1）")
    lines.append("bash manu/video_quality/vq_launch_server.sh 4")
    lines.append("# 1b) 模态重分类 + 采样占比落盘（报告里「81.0% 可见光」等数字的来源，缺这步会 KeyError）")
    lines.append("python manu/video_quality/vq_reclassify.py --metrics-dir manu/video_quality/out")
    lines.append("# 2) 好/坏对比 + 阈值挖掘 + 出图")
    lines.append("python manu/video_quality/vq_compare.py \\")
    lines.append("  --metrics-dir manu/video_quality/out --report-dir manu/video_quality/report")
    lines.append("# 3) 统计严谨性：精确置换检验 + clip/cluster bootstrap CI")
    lines.append("python manu/video_quality/vq_stats.py \\")
    lines.append("  --metrics-dir manu/video_quality/out --out manu/video_quality/report")
    lines.append("# 4) 跟丢切片对比（坏片故障窗 vs 好片全局）")
    lines.append("python manu/video_quality/vq_tail.py \\")
    lines.append("  --metrics-dir manu/video_quality/out --out manu/video_quality/report --slice-seconds 5.0")
    lines.append("# 5) 归一化尺度偏置实测（2.6/4.3 节的抬升系数，产物 report/resample_bias.txt）")
    lines.append("python manu/video_quality/vq_bias.py \\")
    for clip in BIAS_CLIPS:
        lines.append(f'  --clip "{clip}" \\')
    lines.append("  --samples 120 --out manu/video_quality/report/resample_bias.txt")
    lines.append("# 6) 本报告")
    lines.append("python manu/video_quality/vq_report.py \\")
    lines.append("  --metrics-dir manu/video_quality/out \\")
    lines.append("  --report-dir manu/video_quality/report --out manu/video_quality/report/REPORT_视频质量指标.md")
    lines.append("# 7) 独立复算验收（AUC 用秩和法重算，不 import 上面两个脚本）")
    lines.append("python manu/video_quality/vq_verify.py \\")
    lines.append("  --metrics-dir manu/video_quality/out --report-dir manu/video_quality/report \\")
    lines.append("  --memory manu/memory/video_quality_metrics_20261119.md")
    lines.append("# 8) 机械验收（27 项检查，完成判据）")
    lines.append("python manu/video_quality/vq_acceptance.py \\")
    lines.append("  --metrics-dir manu/video_quality/out --report-dir manu/video_quality/report \\")
    lines.append("  --memory manu/memory/video_quality_metrics_20261119.md")
    lines.append("")
    lines.append("```")
    lines.append("")
    lines.append("产物：`manu/video_quality/report/`（`comparison.csv` / `stats.csv` / `tail_compare.csv` / "
                 "`tail_slices.csv` / `metrics_table.csv` / `windows.csv` / `comparison.json` / `*.png` / 本报告）。")
    lines.append("")

    # ---------------- 附录 ----------------
    lines.append("## 附录：逐条视频明细")
    lines.append("")
    appendix = []
    for clip in clips:
        appendix.append(
            {
                "group": "好" if clip["group"] == "good" else "坏",
                "kind": "源视频" if clip["kind"] == "source" else "挂飞录屏",
                "mod": clip["modality"],
                "file": clip["file"],
                "res": f'{clip["width"]}x{clip["height"]}',
                "fps": clip["fps"],
                "frames": clip["frames"],
                "crop": clip["crop"],
                "shift": fmt(float(clip["shift_frac"])),
                "rot": fmt(float(clip["rot_deg"])),
                "lap": fmt(float(clip["lap_var"])),
                "noise": fmt(float(clip["noise_sigma"])),
                "contrast": fmt(float(clip["global_contrast"])),
                "dog": fmt(float(clip["dog_snr_max"])),
            }
        )
    lines.append(
        table(
            appendix,
            [
                ("group", "组"),
                ("kind", "类型"),
                ("mod", "模态"),
                ("file", "文件"),
                ("res", "分辨率"),
                ("fps", "fps"),
                ("frames", "帧数"),
                ("crop", "裁剪后画幅"),
                ("shift", "位移P95(占宽)"),
                ("rot", "旋转P95(度)"),
                ("lap", "清晰度P5"),
                ("noise", "噪声P95"),
                ("contrast", "对比度P5"),
                ("dog", "目标SNR P5"),
            ],
        )
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] report written -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Chinese leadership report")
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--metrics-dir", type=str, default="",
                        help="Per-clip metric dir (default: <report-dir>/../out)")
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else report_dir.parent / "out"
    render(report_dir, Path(args.out), metrics_dir)


if __name__ == "__main__":
    main()
