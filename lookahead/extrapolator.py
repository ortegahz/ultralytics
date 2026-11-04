"""快速前向推演外挂（Look-ahead Extrapolator）。

**为什么做这个**：板端流水线里检测框滞后 ``Δ=8`` 帧（相机 25 Hz ⇒ 320 ms）才到，
转台拿到的坐标其实是 ``t−Δ`` 帧的旧位置。本模块在**成熟跟踪管线的输出末端**外挂
一层**无状态**的前向推演，把上报坐标推到物理帧 ``t``，共四档：

* ``off``：原样透传（开关关闭 ⇒ 与原脚本逐位一致，这是硬验收项）；
* ``stale``：老逻辑基线（硬吃 Δ 帧滞后），尺子 2 的对照组；
* ``kinematic``：纯运动学外推 ``p(t−Δ) + Δ·v(t−Δ)``，零图像开销；
* ``replay``：从 ``t−Δ`` 的锚点出发，用局部质心快核逐帧重推到 ``t``。

**核心约束**：外挂**不得改动任何历史估计**（成熟跟踪资产零回归），所以本模块
只读输入流、只产出新的上报流，**不写回任何状态**；速度来源与位置起点都由调用方
显式指定（``velocity_source`` / ``position_source``）。

时间轴约定（**最容易搞错**）：所有下标都是 **record index**（记录下标），不是视频
绝对帧号；``Δ`` 也以 record 为单位。物理帧 ``t`` 的上报，其**源帧**是 ``f = t − Δ``。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker
from manu.tracking.lookahead.polarity import PolarityTracker
from manu.tracking.lookahead.xcorr import match as _xcorr_match
from manu.tracking.lookahead.xcorr import make_template as _xcorr_make_template
from manu.tracking.lookahead.xcorr import chain_step as _xcorr_chain_step_raw

__all__ = [
    "LookaheadConfig",
    "FrameSource",
    "build_report_stream",
    "estimate_velocity",
    "servo_metrics",
]


@dataclass(frozen=True)
class LookaheadConfig:
    """前向推演外挂的配置。默认值 = 板端那一套（Δ=8 @25Hz = 320ms）。"""

    delta: int = 8
    mode: str = "off"                 # off | stale | kinematic | replay
    velocity_source: str = "kf"       # kf | diff1 | diffN
    velocity_window: int = 3          # diffN 的拟合窗（record 数）
    position_source: str = "output"   # output | kf
    win_size: int = 32                # 快核 ROI 边长（局部质心）
    th_k: float = 3.0                 # 快核底噪倍数
    min_energy: float = 1.0
    min_contrast: float = 5.0
    fast_mode: str = "centroid"       # 快核提取模式：centroid | peak_centroid
    bg_window: int = 64               # 快核底噪估计窗边长（必须 ≥ win_size）
    peak_window: int = 9              # peak_centroid 模式下极大点邻域边长
    replay_anchor: str = "kernel"     # kernel | delta | pred | kf（上报坐标怎么由快核结果合成）
    replay_kf_r: float = 2.0          # kf 档位的观测噪声标准差（px）：质心观测的可信度
    replay_kf_q_pos: float = 1.0      # kf 档位的过程噪声（位置）
    replay_kf_q_vel: float = 4.0      # kf 档位的过程噪声（速度）
    require_valid: bool = True        # replay 中途丢目标 ⇒ 弃权
    replay_fallback: str = "kinematic"  # kinematic | stale | abstain（replay 弃权时的兜底）
    replay_step_gate_px: float = 0.0   # 逐帧位移门（px/步）：重推单步跳太远 = 跟丢了，0 = 不设门
    replay_max_hold: int = 0           # 连续观测缺失时允许 hold 几步（0 = 严格，任一步失效即作废）
    replay_anchor_check_px: float = 0.0  # delta 档位：源帧快核位置与管线锚点的最大容差（0 = 不校验）
    replay_pred_radius_px: float = 12.0  # pred 档位：ROI 中心允许偏离匀速预测的最大半径（0 = 纯预测，不回看）
    # ↓ 2026-11-03 新增：selfpred ROI 档位。delta 档位的 ROI 中心 = 链上一帧的质心，
    #    目标在 ROI 里会**越走越偏**（机动期直接失锁）。selfpred 把 ROI 中心前推到
    #    「链上一帧质心 + 链自身速度」，让目标始终待在窗中心 ⇒ 质心偏置更小、
    #    机动期更不容易掉出 ROI。链速度取已接受步的中位数（实测链位移几乎无偏，
    #    标度比 0.92~0.97、方向余弦 0.992~0.999），比 KF 速度可靠。
    replay_roi_mode: str = "chain"     # chain（默认，等价于历史行为）| selfpred
    replay_roi_selfpred_window: int = 3  # selfpred：取最近 N 步已接受步的中位数作速度
    replay_roi_selfpred_max_px: float = 0.0  # selfpred：ROI 前推量的上限（px，0 = 不限）
    # ↓ 2026-11-03 新增：模板匹配（xcorr）位移精修。默认 0 = 关闭。
    # 机理：质心链的位移误差 = 「质心偏置在链两端的**变化**」，偏置随目标姿态/亮度
    # 漂移 ⇒ 8 步链的位移尾部很重（实测 p90 4.37 px / p95 8.13 px）。改用**同一份
    # 模板**在源帧/目标帧之间配准 ⇒ 两端用同一个偏置，偏置**精确抵消**，只剩真实
    # 位移（实测位移 p90 4.37→3.17 px、p95 8.13→6.34 px，同批 24503 帧 hit@5px
    # 91.36%→94.02%）。模板 = 源帧锚点处按底噪阈值二值化的权重图（去掉背景）。
    replay_xcorr_radius: float = 0.0     # 目标帧搜索半径（px）；0 = 关闭该档位
    replay_xcorr_template: int = 24      # 模板边长（px）
    replay_xcorr_th_k: float = 2.5       # 模板二值化的底噪倍数
    replay_xcorr_min_peak: float = 0.25  # 相关峰值的最低要求（低于则放弃精修）
    replay_xcorr_bg: int = 96            # 底噪估计窗边长（px）
    replay_xcorr_core_frac: float = 0.0  # 模板核心化：只保留最亮的前景像素比例（0 = 关）
    # 相关峰**唯一性比**门：次峰/最佳峰 > 阈值 ⇒ 配准峰不唯一（大概率锁到了背景结构）
    # ⇒ 放弃精修、回退到锚点。0 = 关（= 历史行为，逐位不变）。
    replay_xcorr_uniq_gate: float = 0.0
    # ↓ 2026-11-04 新增：**不自信就不上报**（把误报换成沉默）。
    # 机理：模板配准的**相关峰**是这次重推唯一的自洽性证据。峰值低 ⇒ 要么目标
    # 机动太大（模板跟不上，但快核跟得住，位置其实是对的），要么快核跟丢了
    # （位置是错的）。这两种情况靠峰值本身分不开，但**链位移**可以：跟丢时链
    # 从锚点跑出去很远；机动时链几乎没跑动。所以弃权条件是
    # 「峰值 < replay_abstain_peak_below **且** 链位移 > replay_abstain_disp_px」。
    # 峰值门为 0 ⇒ 整个机制关闭（= 历史行为，逐位不变）。
    replay_abstain_peak_below: float = 0.0
    replay_abstain_disp_px: float = 0.0
    # ↓ 2026-11-04 新增：源帧**无检测**时「再往回退一点重推」。
    # 机理：Δ 帧前的检测器会偶尔漏检（8 序列实测 6.14% 的 GT 帧源帧没有任何输出）
    # ⇒ 外挂拿不到锚点 ⇒ 该帧必然无输出（必然脱靶）。退到**最近一次有检测**的帧
    # 当锚点，多走几步把链重推回来，把「无坐标帧」变成「有坐标帧」。
    # 0 = 不回退（= 历史行为，逐位不变）。
    replay_rollback_max_frames: int = 0
    # 回退链的**峰值置信门**：回退链的模板是从更早的帧切出来的，链更长、位移更大，
    # 目标可能早就飞走了 ⇒ 模板配准会锁到背景结构上 ⇒ 产出的是「幽灵坐标」。
    # 回退链要求相关峰值 ≥ 此值，否则该帧不上报（0 = 不设门）。
    replay_rollback_min_peak: float = 0.0
    # ↓ 2026-11-03 新增：xcorr 的**用法**。direct = 一次跨 Δ 帧的大匹配（历史行为）；
    #    chain = 逐步配准（每步用「上一帧的模板」配准「当前帧」，模板逐步刷新）。
    #    chain 的机理：direct 模式两端相隔 Δ 帧，目标姿态/亮度在这 Δ 帧里变了 ⇒
    #    同一个模板在目标帧里不再精确对应 ⇒ 残差。chain 每步只隔 1 帧，模板与
    #    目标几乎不变 ⇒ 逐步配准的误差只剩亚像素定位噪声；而且被跟踪的**物理点**
    #    在链上保持不变 ⇒ 质心偏置不再逐帧累积（链的误差 = 锚点误差 + 配准漂移）。
    replay_xcorr_mode: str = "direct"    # direct | chain | chain_direct | stride
    replay_xcorr_chain_radius: float = 0.0  # chain 每步的搜索半径（px）；0 = 复用 replay_xcorr_radius
    # ↓ 2026-11-04 新增：**分段配准**（stride 档位）的段长（帧）。0 = 关。
    # 机理：direct 的一次跨 Δ 帧大匹配里，模板与目标的外观已漂移 Δ 帧 ⇒ 相关峰不准
    # （离线大比试实测位移 p95 ~48.9 px、hit@5px 91.2%）。分段后每段只隔 stride 帧、
    # 模板每段刷新 ⇒ 峰很锐（实测 p95 ~2.75 px、hit@5px 99.2%）。
    replay_xcorr_stride: int = 0

    # ↓ 2026-11-03 新增：目标**极性**（亮/暗）自适应。
    # 机理：红外目标的极性不是恒定的——实测 014453 对比度 −12.91、015000 −22.56
    # （都是暗点），其余 6 条是 +12.46~+76.81（亮点）。快核与 xcorr 模板都写死了
    # 「比背景亮」⇒ 暗目标上权重恒为 0、模板为空、质心退化成窗心 ⇒ 配准峰锁到
    # 画面里最亮的云边（实测 014453 上 replay 偏到 70~100 px 外）。
    # bright = 历史行为（**默认值**）；dark = 强制暗假设；auto = 在锚点帧现场判。
    replay_polarity: str = "bright"
    # auto 模式判暗的**置信倍数**（暗向能量 > 倍数 × 亮向能量才判暗）。实测 10 是最优：
    # 亮目标序列上误判的链条被挡回亮假设（001810 的 68 条误判 → 2 条），暗目标 014453 上
    # 99.9% 的链条保留（FAR 0.3608 → 0.0021）。只在 replay_polarity="auto" 下生效。
    replay_polarity_ratio: float = 10.0
    replay_consistency_gate_px: float = 0.0  # 一致性门：重推结果与"锚点+Δ·v"偏离超过它 ⇒ 判跑飞，改用运动学预测（0 = 不设门）
    replay_step_rel_tol_px: float = 0.0   # 首步离群门：重推**第一步**位移与参考速度之差超过它 ⇒ 该步不入链（0 = 不设门，**默认关**：实测净亏 1.74pp F1）
    replay_hold_use_ref: bool = True    # hold 时按参考速度推进（False = 按最后一次实测步长推进）
    replay_outlier_tol_px: float = 0.0  # 批量稳健位移：单步位移偏离步长中位数超过它 ⇒ 换成中位数（0 = 不处理）
    replay_hold_window: int = 3         # hold 推进速度的参考窗：取最近 N 步已接受步（0 = 全窗口）
    replay_hold_outlier_tol_px: float = 10.0  # 末步与参考中位数差超过它 ⇒ 末步判为离群，改用中位数
    replay_hold_robust: bool = False    # hold 用稳健参考推进（False = 末步口径，**8 序列实测 F1 最优**）
    replay_source_anchor_fallback: bool = True  # 源帧快核失锁 ⇒ 用管线锚点当位移原点（不再整段作废）
    # ↓ 2026-11-03 新增：针对 hit@5px 的两把药（见 README「hit@5px 缺口归因」）
    # ① 观测层 shrinkage：把灰度质心往「锚点+匀速预测」收缩，抑制 2~4 px 的系统性
    #    质心偏置（机身高光/拖尾把质心拽偏）。1.0 = 不收缩（关闭）。
    replay_obs_shrink_alpha: float = 1.0
    # ② 低速冻结：|参考速度| < 阈值 ⇒ 位移**锁死为 0**（上报管线锚点）。
    #    治「悬停目标被重推链拽走」：真值位移只有 0~1 px/帧时，重推位移只会是噪声。
    #    0 = 不冻结（关闭）。
    replay_freeze_speed_px: float = 0.0
    # ③ 位移层 shrinkage：把**最终位移**（而非链内观测）向「参考位移 = v·Δ」收缩。
    #    与①观测层收缩的区别：①会把链内的 ROI 中心也拽向预测 ⇒ 机动目标跟不上
    #    （实测 001324 F1 −10.8 pp）；③只改最后一步的位移合成，链内跟踪不动。
    #    α=1.0 = 不收缩（关闭）。
    replay_disp_shrink_alpha: float = 1.0
    # ④ 位移融合（自适应）：实测位移与参考位移（v·Δ）**独立**、误差不相关 ——
    #    实测（视觉）更准但有 walk-off 尾部，参考（卡尔曼速度）无尾部但中位误差 3~10 px。
    #    按两者差值 d 给实测赋权 w = 1/(1+(d/σ)²)：一致时几乎纯实测，矛盾时退向参考。
    #    σ = 0 = 不融合（关闭）。
    replay_disp_merge_sigma_px: float = 0.0
    replay_mode: str = "perframe"        # perframe | rolling（replay 档位的两种实现方式）
    rolling_recenter: bool = True        # rolling：锚点到达时把快核 ROI 中心重锚到校正后的绝对位置
    rolling_hold_mode: str = "last_step"  # rolling：链观测失效时的推进依据 last_step | prev_report
    rolling_max_step_px: float = 12.0   # rolling：单步位移门（px/步）：超过 ⇒ 快核抓住杂波，该步作废走 hold
    rolling_max_offset_px: float = 40.0  # rolling：锚点残差上限（px）：超过 ⇒ 认为链已跟丢，不采纳该锚点
    rolling_reacquire_px: float = 30.0   # rolling：链位置相对"锚点+Δ·v"的允许偏离（px）：超出 ⇒ 投影回边界并清零步长
    rolling_vel_alpha: float = 0.5      # rolling：链平滑速度的指数滑动平均系数（0<α≤1，越大越跟手）
    refine_gate_px: float = 24.0          # refine：快核单步修正的最大位移（px），超门 ⇒ 只用运动学预测
    output_max_step_px: float = 0.0        # 输出级限幅（伺服 slew limiter）：每帧输出位移上限（px），0=不限幅
    refine_mode: str = "obs"            # refine：修正量取快核观测 obs（obs）还是 obs−预测（delta 同义）


class FrameSource(Protocol):
    """按**管线时间轴**（record index）取灰度整帧；取不到返回 None。"""

    def get(self, frame_index: int) -> np.ndarray | None: ...


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _detection_position(det: dict) -> np.ndarray:
    """取检测的绝对像素位置 ``(x, y)``（x = 列，y = 行，与 GT 标签口径一致）。"""
    return np.asarray(det.get("pos"), dtype=np.float64).reshape(2).copy()


def _lock_polarity(tracker, frame, xy) -> int:
    """在锚点帧判目标极性并锁定；返回锁定后的极性（``+1`` 亮 / ``−1`` 暗）。

    tracker 不是 :class:`~manu.tracking.lookahead.polarity.PolarityTracker`
    （即 ``--replay-polarity bright``，**默认值**）时直接返回 ``+1``
    ⇒ **历史行为逐位不变**。
    """
    if tracker is None or not hasattr(tracker, "lock_from"):
        return 1
    if frame is None or getattr(frame, "ndim", 0) != 2:
        return int(getattr(tracker, "polarity", 1))
    return int(tracker.lock_from(frame, xy))


def _record_polarity(result: dict, tracker, polarity: int) -> None:
    """记录一条链的极性判定结果（诊断计数，不影响任何数值行为）。"""
    if tracker is None or not hasattr(tracker, "last_ratio"):
        return
    ratio = float(getattr(tracker, "last_ratio", 0.0))
    result["polarity_chains"] = int(result.get("polarity_chains", 0)) + 1
    if polarity < 0:
        result["polarity_dark"] = int(result.get("polarity_dark", 0)) + 1
    for threshold in (3.0, 10.0, 30.0):
        if ratio >= threshold:
            key = f"polarity_ratio_ge{int(threshold)}"
            result[key] = int(result.get(key, 0)) + 1


def _xcorr_refine(frame_source, src_frame, dst_frame, anchor, chain_pos, cfg, result, polarity=1):
    """用模板匹配把重推位置精修成「锚点 + 真实位移」；不可靠时返回 None。

    详见 :mod:`manu.tracking.lookahead.xcorr` 的机理说明。默认 ``replay_xcorr_radius
    == 0`` ⇒ 直接返回 None ⇒ **历史行为逐位不变**。
    """
    if frame_source is None or float(cfg.replay_xcorr_radius) <= 0.0:
        return None
    src_img = frame_source.get(src_frame)
    dst_img = frame_source.get(dst_frame)
    if src_img is None or dst_img is None:
        return None
    if getattr(src_img, "ndim", 0) != 2 or getattr(dst_img, "ndim", 0) != 2:
        return None
    template, origin = _xcorr_make_template(
        src_img,
        anchor,
        tsize=int(cfg.replay_xcorr_template),
        bg_size=int(cfg.replay_xcorr_bg),
        th_k=float(cfg.replay_xcorr_th_k),
        core_frac=float(cfg.replay_xcorr_core_frac),
        polarity=int(polarity),
    )
    if template is None:
        result["xcorr_no_template"] = int(result.get("xcorr_no_template", 0)) + 1
        return None
    diagnostics: dict = {}
    displacement, peak = _xcorr_match(
        dst_img,
        template,
        anchor,
        chain_pos,
        radius=int(round(float(cfg.replay_xcorr_radius))),
        tsize=int(cfg.replay_xcorr_template),
        origin=origin,
        polarity=int(polarity),
        diagnostics=diagnostics,
    )
    if displacement is None:
        result["xcorr_no_template"] = int(result.get("xcorr_no_template", 0)) + 1
        return None
    result["xcorr_peak_sum"] = float(result.get("xcorr_peak_sum", 0.0)) + peak
    # 这次大匹配的峰值：调用方用它判断「这次重推有没有自洽证据」。
    result["xcorr_peak_last"] = float(peak)
    if peak < float(cfg.replay_xcorr_min_peak):
        result["xcorr_rejected"] = int(result.get("xcorr_rejected", 0)) + 1
        return None
    # ---- 唯一性比门：最佳峰不唯一 ⇒ 大概率配到了背景结构（walk-off）⇒ 二次意见 ----
    # 机理（24 序列 8 条、51205 帧离线实测）：好帧的最佳相关峰**压倒性独大**
    # （唯一性比 p50 = 0.000），walk-off 帧的峰**有歧义**（p50 = 0.593）⇒ 这个量
    # 是目前唯一能把 walk-off 帧与好帧分开的**无状态**信号。
    # 但「一看到歧义就回退锚点」实测是**净负**（8 序列 τ=0.3：F1 95.06→87.03、
    # hit@5px 84.96→67.20）—— 锚点比 8 帧前的真实位置更差。
    # 所以改成**二次意见**：歧义时以**锚点**（已知可靠的位置）为搜索中心**重配一次**，
    # 只有当重配结果**也唯一**（uniqueness ≤ 门限）且峰值达标时才采纳；
    # 否则**保持原结果不动** ⇒ 最坏情况是「什么都没改」⇒ 该机制**不可能**让指标变差。
    uniq_gate = float(cfg.replay_xcorr_uniq_gate)
    if uniq_gate > 0.0:
        uniq = float(diagnostics.get("uniqueness", 0.0))
        result["xcorr_uniq_sum"] = float(result.get("xcorr_uniq_sum", 0.0)) + uniq
        result["xcorr_uniq_n"] = int(result.get("xcorr_uniq_n", 0)) + 1
        if uniq > uniq_gate:
            result["xcorr_ambiguous"] = int(result.get("xcorr_ambiguous", 0)) + 1
            diag2: dict = {}
            disp2, peak2 = _xcorr_match(
                dst_img,
                template,
                anchor,
                anchor,
                radius=int(round(float(cfg.replay_xcorr_radius))),
                tsize=int(cfg.replay_xcorr_template),
                origin=origin,
                polarity=int(polarity),
                diagnostics=diag2,
            )
            uniq2 = float(diag2.get("uniqueness", 0.0))
            result["xcorr_uniq2_sum"] = float(result.get("xcorr_uniq2_sum", 0.0)) + uniq2
            # 采纳条件（三条都要满足，缺一不可）：
            # ① 重配结果**唯一**（uniq2 ≤ 门限）⇒ 它锁的是真目标，不是背景结构；
            # ② 重配**比原配更自信**（peak2 > peak）⇒ 原配是糊的、重配是清晰的；
            #    缺这一条时，高速目标会被「更靠近锚点但错误」的结构骗走
            #    （实测 24 序列 τ=0.5：002438 FP +69、000322 +42、001415 +27）。
            # ③ 重配的峰值仍要达标（min_peak）。
            if (
                disp2 is not None
                and peak2 >= float(cfg.replay_xcorr_min_peak)
                and uniq2 <= uniq_gate
                and peak2 > peak
            ):
                result["xcorr_rematch_applied"] = int(result.get("xcorr_rematch_applied", 0)) + 1
                return np.asarray(anchor, dtype=np.float64).reshape(2) + disp2
            result["xcorr_rematch_rejected"] = int(result.get("xcorr_rematch_rejected", 0)) + 1
    matched = anchor + displacement
    limit = float(cfg.replay_xcorr_radius) + float(cfg.replay_xcorr_template)
    if float(np.linalg.norm(matched - chain_pos)) > limit:
        result["xcorr_rejected"] = int(result.get("xcorr_rejected", 0)) + 1
        return None
    result["xcorr_applied"] = int(result.get("xcorr_applied", 0)) + 1
    return matched


def _xcorr_stride_refine(frame_source, src_frame, dst_frame, anchor, chain_pos, cfg, result, polarity=1):
    """分段配准：把 ``[src_frame, dst_frame]`` 每 ``stride`` 帧切一段，每段用「段源帧模板」配准。

    机理：``direct`` 的一次跨 Δ 帧大匹配里，模板与目标的外观已漂移 Δ 帧 ⇒ 相关峰
    不准（离线大比试实测位移 p95 ~48.9 px、hit@5px 91.2%）。分段后每段只隔
    ``stride`` 帧、模板每段刷新 ⇒ 峰很锐（实测 p95 ~2.75 px、hit@5px 99.2%）。

    第 k 段 = ``(src_frame + k*stride → src_frame + (k+1)*stride)``（末段截断到
    ``dst_frame``）：模板从**该段的源帧**、以当前 ``position`` 为中心切出，在**该段
    的目标帧**里配准 ⇒ 量出的是该段的真实位移，累加即整段位移。

    段失败（模板为空 / 匹配不上 / 峰值不达标）⇒ **hold**：沿用上一次成功段的位移
    向量往前推；**第一段就失败**（还没有可 hold 的位移）⇒ 整个精修作废，返回
    ``None``，调用方保持原位置、走既有兜底。

    返回精修后的**绝对坐标**（起点就是 ``anchor``）；不可靠时返回 ``None``。
    """
    if frame_source is None or float(cfg.replay_xcorr_radius) <= 0.0:
        return None
    src_frame = int(src_frame)
    dst_frame = int(dst_frame)
    if dst_frame <= src_frame:
        return None
    stride = max(1, int(cfg.replay_xcorr_stride))
    radius = int(
        round(
            float(
                cfg.replay_xcorr_chain_radius
                if float(cfg.replay_xcorr_chain_radius) > 0.0
                else cfg.replay_xcorr_radius
            )
        )
    )
    min_peak = float(cfg.replay_xcorr_min_peak)
    position = np.asarray(anchor, dtype=np.float64).reshape(2).copy()
    last_step = None
    segment_peak = 1.0
    for start in range(src_frame, dst_frame, stride):
        end = min(start + stride, dst_frame)
        src_img = frame_source.get(start)
        dst_img = frame_source.get(end)
        template = origin = None
        if (
            src_img is not None
            and dst_img is not None
            and getattr(src_img, "ndim", 0) == 2
            and getattr(dst_img, "ndim", 0) == 2
        ):
            template, origin = _xcorr_make_template(
                src_img,
                position,
                tsize=int(cfg.replay_xcorr_template),
                bg_size=int(cfg.replay_xcorr_bg),
                th_k=float(cfg.replay_xcorr_th_k),
                core_frac=float(cfg.replay_xcorr_core_frac),
                polarity=int(polarity),
            )
        displacement = None
        peak = 0.0
        if template is not None:
            diagnostics: dict = {}
            displacement, peak = _xcorr_match(
                dst_img,
                template,
                position,
                position,
                radius=radius,
                tsize=int(cfg.replay_xcorr_template),
                origin=origin,
                polarity=int(polarity),
                diagnostics=diagnostics,
            )
            if displacement is not None:
                # 分段配准的置信度取**最弱的一段**（任一段糊 ⇒ 整体不可信），
                # 这样「峰值低就不上报」的弃权门在 stride 档位照样生效。
                segment_peak = min(segment_peak, float(peak))
                result["xcorr_peak_last"] = float(segment_peak)
        if displacement is None or float(peak) < min_peak:
            result["xcorr_stride_hold"] = int(result.get("xcorr_stride_hold", 0)) + 1
            if last_step is None:
                # 第一段就失败 ⇒ 没有可 hold 的位移 ⇒ 整个精修作废。
                return None
            position = position + last_step
            continue
        position = position + displacement
        last_step = displacement
    limit = float(cfg.replay_xcorr_radius) + float(cfg.replay_xcorr_template)
    if float(np.linalg.norm(position - np.asarray(chain_pos, dtype=np.float64).reshape(2))) > limit:
        result["xcorr_rejected"] = int(result.get("xcorr_rejected", 0)) + 1
        return None
    result["xcorr_applied"] = int(result.get("xcorr_applied", 0)) + 1
    return position


def _xcorr_chain_step(
    frame_source, prev_frame, dst_frame, prev_pos, search_center, cfg, result, polarity=1
):
    """逐步配准的一步（包装 :func:`manu.tracking.lookahead.xcorr.chain_step`）。

    返回新位置；配准不可靠时返回 ``None``，调用方回落到快核质心观测。
    """
    if frame_source is None:
        return None
    radius = float(cfg.replay_xcorr_chain_radius)
    if radius <= 0.0:
        radius = float(cfg.replay_xcorr_radius)
    if radius <= 0.0:
        return None
    src_img = frame_source.get(prev_frame)
    dst_img = frame_source.get(dst_frame)
    if src_img is None or dst_img is None:
        return None
    if getattr(src_img, "ndim", 0) != 2 or getattr(dst_img, "ndim", 0) != 2:
        return None
    matched, peak = _xcorr_chain_step_raw(
        dst_img,
        src_img,
        prev_pos,
        search_center,
        radius=int(round(radius)),
        tsize=int(cfg.replay_xcorr_template),
        bg_size=int(cfg.replay_xcorr_bg),
        th_k=float(cfg.replay_xcorr_th_k),
        min_peak=float(cfg.replay_xcorr_min_peak),
        core_frac=float(cfg.replay_xcorr_core_frac),
        polarity=int(polarity),
    )
    if matched is None:
        result["xcorr_chain_rejected"] = int(result.get("xcorr_chain_rejected", 0)) + 1
        return None
    result["xcorr_chain_applied"] = int(result.get("xcorr_chain_applied", 0)) + 1
    result["xcorr_chain_peak_sum"] = float(result.get("xcorr_chain_peak_sum", 0.0)) + peak
    return matched


def _find_detection(dets_by_frame: list[list[dict]], frame: int, track_id: int) -> dict | None:
    """在某帧里按 ``track_id`` 找检测；没有返回 ``None``。"""
    if frame < 0 or frame >= len(dets_by_frame):
        return None
    for det in dets_by_frame[frame]:
        if int(det.get("track_id", -1)) == int(track_id):
            return det
    return None


def _estimate_velocity(
    dets_by_frame: list[list[dict]],
    frame: int,
    track_id: int,
    *,
    source: str,
    window: int,
    kf_states: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] | None,
) -> tuple[np.ndarray, bool]:
    """返回 ``(速度, 是否真的拿到了速度)``。

    ``available=False`` 表示"没有数据"（与"速度恰好为 0"是两回事）——
    ``stats["missing_velocity"]`` 靠这个标志计数，静止目标不该被算成缺速度。
    """
    zero = np.zeros(2, dtype=np.float64)
    normalized = str(source).strip().lower()

    if normalized == "kf":
        if not kf_states:
            return zero, False
        per_frame = kf_states.get(int(frame))
        if not per_frame:
            return zero, False
        entry = per_frame.get(int(track_id))
        if entry is None:
            return zero, False
        return np.asarray(entry[1], dtype=np.float64).reshape(2).copy(), True

    if normalized == "diff1":
        current = _find_detection(dets_by_frame, frame, track_id)
        previous = _find_detection(dets_by_frame, frame - 1, track_id)
        if current is None or previous is None:
            return zero, False
        return _detection_position(current) - _detection_position(previous), True

    if normalized == "diffn":
        # 只取"最近 window 个**出现过**的"位置：中间漏检的帧不该把拟合窗撑破。
        points: list[tuple[int, np.ndarray]] = []
        last = min(int(frame), len(dets_by_frame) - 1)
        for index in range(last, -1, -1):
            det = _find_detection(dets_by_frame, index, track_id)
            if det is None:
                continue
            points.append((index, _detection_position(det)))
            if len(points) >= max(int(window), 1):
                break
        if len(points) < 2:
            return zero, False
        xs = np.asarray([point[0] for point in points], dtype=np.float64)
        ys = np.asarray([point[1] for point in points], dtype=np.float64)
        # 最小二乘直线拟合 y = a·x + b，取斜率 a（像素/record）。
        design = np.stack([xs, np.ones_like(xs)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, ys, rcond=None)
        return np.asarray(coefficients[0], dtype=np.float64).copy(), True

    raise ValueError(f"unknown velocity_source: {source!r}")


def _base_position(det: dict, frame: int, cfg: LookaheadConfig, kf_states) -> np.ndarray:
    """按 ``cfg.position_source`` 取上报坐标的**起点**位置。

    ``position_source="kf"`` 但该 track 在该帧没有 KF 状态时退回检测流自己的位置：
    宁可粗一点，也不能把这条 track 从上报流里弄丢（丢一条 = 转台那一帧没坐标）。
    """
    if str(cfg.position_source).strip().lower() == "kf" and kf_states:
        per_frame = kf_states.get(int(frame))
        if per_frame:
            entry = per_frame.get(int(det.get("track_id", -1)))
            if entry is not None:
                return np.asarray(entry[0], dtype=np.float64).reshape(2).copy()
    return _detection_position(det)


# 档位 B′ 的卡尔曼常量：与成熟管线里的 ``PointKalmanTrack`` 保持同一套模型
# （匀速模型 x=[x,y,vx,vy]，单位是「record」）。
_KF_F = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_KF_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_KF_I4 = np.eye(4, dtype=np.float64)


def _initial_kf_state(
    det: dict,
    frame: int,
    cfg: LookaheadConfig,
    kf_states,
) -> tuple[np.ndarray, np.ndarray]:
    """取重推起点的卡尔曼后验 ``(状态 x[4], 协方差 P[4,4])``。

    优先用成熟管线在源帧上的真实 KF 后验（位置 + 速度 + 协方差）；
    取不到时退回「位置=检测位置、速度=0、P=管线初值」。
    """
    state = np.zeros(4, dtype=np.float64)
    covariance = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float64)
    found = False
    if kf_states:
        per_frame = kf_states.get(int(frame))
        if per_frame:
            entry = per_frame.get(int(det.get("track_id", -1)))
            if entry is not None:
                state[0:2] = np.asarray(entry[0], dtype=np.float64).reshape(2)
                state[2:4] = np.asarray(entry[1], dtype=np.float64).reshape(2)
                if len(entry) > 2:
                    covariance = np.asarray(entry[2], dtype=np.float64).reshape(4, 4).copy()
                found = True
    if not found:
        state[0:2] = _detection_position(det)
    return state, covariance

def _kinematic_detections(
    dets_by_frame: list[list[dict]],
    source_dets: list[dict],
    frame: int,
    cfg: LookaheadConfig,
    kf_states,
    stats: dict,
) -> list[dict]:
    """kinematic 臂：``位置 = 起点 + Δ·v``，其余字段与源帧一致。"""
    delta = int(cfg.delta)
    out: list[dict] = []
    for det in source_dets:
        track_id = int(det.get("track_id", -1))
        base = _base_position(det, frame, cfg, kf_states)
        velocity, available = _estimate_velocity(
            dets_by_frame,
            frame,
            track_id,
            source=cfg.velocity_source,
            window=int(cfg.velocity_window),
            kf_states=kf_states,
        )
        if not available:
            # 没有速度就按 0 外推：宁可少推一点，也不要凭空编一段运动。
            stats["missing_velocity"] += 1
        position = base + delta * velocity
        new_det = dict(det)
        new_det["pos"] = (float(position[0]), float(position[1]))
        out.append(new_det)
    return out


def _replay_detections(
    source_dets: list[dict],
    frame: int,
    target_frame: int,
    cfg: LookaheadConfig,
    kf_states,
    tracker: LocalPatchTracker | None,
    frame_source: FrameSource | None,
    dets_by_frame: list[list[dict]] | None = None,
) -> dict:
    """从源帧锚点逐帧重推到物理帧 ``target_frame``。

    重推区间是 ``frame+1 .. target_frame``（**含**目标帧），共 Δ 步；每步用上一位置的
    输出作为下一帧的输入中心。

    **观测缺失时的处理**（``cfg.replay_max_hold``）：

    * ``0``（严格）：任一步快核判无效 ⇒ 整段重推作废，走 ``cfg.replay_fallback``；
    * ``N > 0``（hold-over）：允许**连续 N 步**观测缺失，这几步按**最后一次实测的
      单步位移向量**继续外推（不是按噪声速度外推）。这条对应用户文档里的
      "自持周期（Hold-over Duration）"——转台最怕跟到一半丢了乱飘。

    **上报坐标怎么合成**（``cfg.replay_anchor``）：

    * ``"kernel"``（默认，原始口径）：上报坐标 = 快核在 ``t`` 帧的**绝对**位置。
    * ``"delta"``：上报坐标 = **管线锚点 A** + （快核在 ``t`` 的位置 − 快核在
      源帧 ``f`` 的位置）。写成真值 + 误差：``A = p(f) + e(f)``、``K(f) = p(f) + b(f)``、
      ``K(t) = p(t) + b(t)``，则上报坐标 ``= p(t) + e(f) + (b(t) − b(f))``——
      **消掉了绝对偏置 b(t)**，只留下锚点自身的误差 e(f) 与偏置的**变化量**。
    * ``"pred"``：与 ``"delta"`` 的**融合公式完全相同**，唯一区别是重推链的 ROI
      中心钉在"锚点 + 匀速预测"上（可在 ``replay_pred_radius_px`` 内朝实测位置回看），
      而不是跟着快核自己的质心走。质心偏置 b 因此**不累积**，跑飞被结构性消除。
    * ``"kf"``：见 :func:`_replay_kf_detections`（逐帧递推卡尔曼状态）。

    Returns
    -------
    dict
        ``entries`` / ``failed`` / ``steps`` / ``reason`` / ``max_step_disp`` /
        ``anchor_calls`` / ``held`` / ``best_effort``（失败时带出"尽力而为"的
        ``(位置, 最后实测位移, 剩余步数)``，供 ``hold`` / ``holdv`` 兜底使用）。
    """
    anchor_mode = str(cfg.replay_anchor).strip().lower()
    if anchor_mode not in ("kernel", "delta", "pred", "kf"):
        raise ValueError(f"unknown replay_anchor: {cfg.replay_anchor!r}")
    roi_mode = str(cfg.replay_roi_mode).strip().lower()
    if roi_mode not in ("chain", "selfpred"):
        raise ValueError(f"unknown replay_roi_mode: {cfg.replay_roi_mode!r}")
    xcorr_mode = str(cfg.replay_xcorr_mode).strip().lower()
    if xcorr_mode not in ("direct", "chain", "chain_direct", "stride"):
        raise ValueError(f"unknown replay_xcorr_mode: {cfg.replay_xcorr_mode!r}")
    # chain 档位：链内每一步都做模板配准（而不是质心观测）。
    chain_grounded = xcorr_mode in ("chain", "chain_direct") and anchor_mode in ("delta", "pred")
    # 只有 xcorr 开启时 chain 档位才有意义（否则链退回纯质心，等价于历史行为）。
    if not float(cfg.replay_xcorr_radius) > 0.0 and not float(cfg.replay_xcorr_chain_radius) > 0.0:
        chain_grounded = False
    # pred 档位自带 ROI 中心逻辑（锚点 + 匀速预测），selfpred 只作用于 chain/delta 链路。
    selfpred_mode = roi_mode == "selfpred" and anchor_mode != "pred"
    if anchor_mode == "kf":
        return _replay_kf_detections(
            source_dets, frame, target_frame, cfg, kf_states, tracker, frame_source
        )

    max_hold = int(cfg.replay_max_hold)
    if max_hold <= 0 and not cfg.require_valid:
        max_hold = 10**6
    result: dict = {
        "entries": [],
        "failed": False,
        "steps": 0,
        "reason": "",
        "max_step_disp": 0.0,
        "anchor_calls": 0,
        "held": 0,
        "step_rejected": 0,
        "outlier_replaced": 0,
        "source_anchor_fallback": 0,
        "frozen": 0,
        "disp_shrunk": 0,
        "disp_merged": 0,
        "disp_merge_weight_sum": 0.0,
        "consistency_rejected": 0,
        "best_effort": None,
    }
    for det in source_dets:
        anchor = _base_position(det, frame, cfg, kf_states)
        kernel_start = anchor
        polarity = int(getattr(tracker, "polarity", 1))
        polarity_locked = False
        # 参考速度（管线卡尔曼/差分速度）：源帧快核就失败时，兜底仍要用它做匀速外推。
        fallback_velocity = np.zeros(2, dtype=np.float64)
        fallback_velocity_available = False
        if anchor_mode in ("delta", "pred"):
            fallback_velocity, fallback_velocity_available = _estimate_velocity(
                dets_by_frame if dets_by_frame is not None else [],
                frame,
                int(det.get("track_id", -1)),
                source=cfg.velocity_source,
                window=int(cfg.velocity_window),
                kf_states=kf_states,
            )
        if anchor_mode in ("delta", "pred"):
            # 先在**源帧**跑一次快核取 K(f)：位移的两端都必须有快核观测。
            source_image = None if frame_source is None else frame_source.get(frame)
            if source_image is None or getattr(source_image, "ndim", 0) != 2:
                result.update(failed=True, reason=f"missing_frame:{frame}")
                return result
            # 目标极性：在**锚点帧**（位置已知 = 检测框中心，误差 ~2 px）现场判一次，
            # 并**锁定整条链**——ROI 一旦漂走，局部极性判断会跟着翻，锁定可防级联跑飞。
            polarity = _lock_polarity(tracker, source_image, anchor)
            polarity_locked = True
            _record_polarity(result, tracker, polarity)
            anchor_result = tracker.step(source_image, anchor)
            result["anchor_calls"] += 1
            if not anchor_result.valid:
                # 源帧快核失锁（实测 001324 有 846 帧是**源帧就失败**，不是链中失败）。
                # 旧行为：整段作废走兜底 ⇒ 这 846 帧退化成"Δ 帧前的旧位置"。
                # 新行为（默认）：**把管线锚点当作位移原点**继续重推——
                # 位移 K(t) − anchor 仍然是 8 帧的真实位移，只是多带一份快核绝对偏置。
                # 整段作废只在显式关闭时才走。
                if not cfg.replay_source_anchor_fallback:
                    # 兜底**不要**"原样搬运源帧坐标"
                    # （那是 Δ 帧前的旧位置，实测 001324 的 846 个兜底帧 CLE p50 ≈ 12.5 px），
                    # 改用"锚点 + Δ·v"匀速外推（实测同批帧 CLE p50 ≈ 7.5 px）。
                    result.update(failed=True, reason=f"lost_target:{frame}")
                    result["best_effort"] = {
                        "position": (float(anchor[0]), float(anchor[1])),
                        "step_vector": (
                            float(fallback_velocity[0]),
                            float(fallback_velocity[1]),
                        ),
                        "remaining_steps": int(target_frame - frame),
                    }
                    return result
                kernel_start = anchor.copy()
                result["source_anchor_fallback"] = int(result.get("source_anchor_fallback", 0)) + 1
            kernel_start = np.asarray(anchor_result.xy, dtype=np.float64).reshape(2).copy()
            # 源帧一致性校验：管线在源帧的位置是准的（误差 ~2 px），快核在源帧若与它
            # 差得远，说明**快核在源帧就锁错了目标**（实测抓到 180 px 外的城市灯光）。
            # 此时 K(f) 不可信 ⇒ 位移基准 K(t) − K(f) 也不可信 ⇒ 整段作废走兜底。
            if float(cfg.replay_anchor_check_px) > 0.0:
                anchor_gap = float(np.linalg.norm(kernel_start - anchor))
                if anchor_gap > float(cfg.replay_anchor_check_px):
                    result.update(failed=True, reason=f"anchor_mismatch:{frame}")
                    return result
        # ---- xcorr chain 档位：把被跟踪的**物理点**锚在「管线锚点」上 ----
        # 机理：delta 档位上报的是「锚点 + (链内位置 − 源帧快核质心)」，链内位置
        # 是在**快核质心**的坐标系里累加的 ⇒ 源帧质心相对锚点的偏置被整段继承下来。
        # chain 档位改成「锚点 + 逐帧真实位移」⇒ 被跟踪的物理点全程是锚点那个点，
        # 误差只剩「锚点自身误差 + 配准漂移」，质心偏置不再进入。
        chain_origin = kernel_start
        if chain_grounded:
            chain_origin = anchor.copy()
        position = chain_origin

        def _fuse_to_anchor(pos):
            """链内绝对位置 -> 上报位置。chain 档位直接用链内位置（链以锚点为原点）。"""
            arr = np.asarray(pos, dtype=np.float64).reshape(2)
            if chain_grounded:
                return arr.copy()
            return anchor + (arr - kernel_start)
        # ---- pred 档位：ROI 中心跟**运动学预测**走，不跟快核自己的质心走 ----
        # 机理：centroid 模式的快核输出是"ROI 内加权质心"，本身带偏置 b。若把上一步
        # 的质心当作下一步的 ROI 中心，偏置会**自我放大**（拖尾把 ROI 中心拽偏 →
        # 目标被切掉 → 质心更偏），实测 8 步能跑飞 190 px（win48）/ 77 px（win32）。
        # pred 档位把 ROI 中心钉在"锚点 + 匀速预测"上，快核只提供**局部**观测，
        # 偏置 b 不再累积；允许在 radius 内朝实测位置回看，兼顾目标机动。
        velocity = np.zeros(2, dtype=np.float64)
        velocity_available = False
        if anchor_mode == "pred":
            velocity, velocity_available = _estimate_velocity(
                dets_by_frame if dets_by_frame is not None else [],
                frame,
                int(det.get("track_id", -1)),
                source=cfg.velocity_source,
                window=int(cfg.velocity_window),
                kf_states=kf_states,
            )
        pred_radius = float(cfg.replay_pred_radius_px)
        last_step_vector = np.zeros(2, dtype=np.float64)
        failed_frame = frame
        # ---- 相对位移门：把"离群单步"从 hold 放大器里摘掉 ----
        # 机理：hold 是按"最后一次实测步长"往前盲推的。若那一步本身是离群跳变
        # （实测抓到的 21 px 假跳变），hold 8 步就把它放大成 130 px 的幻影位移，
        # 实测贡献了重推臂 CLE 尾巴的大头。参考速度取管线卡尔曼速度（先验）
        # 再用已接受步的中位数修正（稳健），离群步直接作废、按参考速度 hold。
        rel_tol = float(cfg.replay_step_rel_tol_px)
        ref_velocity = np.zeros(2, dtype=np.float64)
        ref_available = False
        accepted_steps: list[np.ndarray] = []
        step_records: list[tuple[bool, np.ndarray]] = []
        if rel_tol > 0.0 or cfg.replay_hold_use_ref or chain_grounded:
            ref_velocity, ref_available = _estimate_velocity(
                dets_by_frame if dets_by_frame is not None else [],
                frame,
                int(det.get("track_id", -1)),
                source=cfg.velocity_source,
                window=int(cfg.velocity_window),
                kf_states=kf_states,
            )
        """最后一次**实测**的单步位移向量（hold-over 用它，不靠噪声速度）。"""
        last_valid_frame = frame
        consecutive_hold = 0
        failed = False
        reason = ""
        for step_frame in range(frame + 1, target_frame + 1):
            image = None if frame_source is None else frame_source.get(step_frame)
            if image is None or getattr(image, "ndim", 0) != 2:
                failed = True
                failed_frame = step_frame
                reason = f"missing_frame:{step_frame}"
                if anchor_mode in ("delta", "pred"):
                    result["best_effort"] = {
                        "position": (
                            float(_fuse_to_anchor(position)[0]),
                            float(_fuse_to_anchor(position)[1]),
                        ),
                        "step_vector": (
                            float(fallback_velocity[0]),
                            float(fallback_velocity[1]),
                        ),
                        "remaining_steps": int(max(0, target_frame - failed_frame + 1)),
                    }
                break
            if anchor_mode == "pred":
                pure = anchor + velocity * (step_frame - frame)
                if not velocity_available:
                    roi_center = position
                elif pred_radius > 0.0:
                    # 朝实测位置回看，但**偏离纯预测不超过 radius**（防跑飞的上界）。
                    follow = position + velocity
                    offset = follow - pure
                    norm = float(np.linalg.norm(offset))
                    if norm > pred_radius:
                        offset = offset * (pred_radius / norm)
                    roi_center = pure + offset
                else:
                    roi_center = pure
            elif selfpred_mode and accepted_steps:
                # ---- selfpred ROI：把窗心前推到「链上一帧质心 + 链自身速度」----
                # 机理：chain 档位的 ROI 中心 = 链上一帧的质心，目标在窗里会越走越偏
                # （机动期直接被切掉）。用**链自身**的位移中位数前推一帧，目标就始终
                # 待在窗中心 ⇒ 质心偏置更小、失锁更少。速度取已接受步的中位数
                # （实测链位移几乎无偏：标度比 0.92~0.97、方向余弦 0.992~0.999）。
                roi_window = int(cfg.replay_roi_selfpred_window)
                recent = accepted_steps[-roi_window:] if roi_window > 0 else accepted_steps
                v_self = np.median(np.asarray(recent, dtype=np.float64), axis=0)
                cap = float(cfg.replay_roi_selfpred_max_px)
                norm_v = float(np.linalg.norm(v_self))
                if cap > 0.0 and norm_v > cap:
                    v_self = v_self * (cap / norm_v)
                roi_center = position + v_self
            else:
                roi_center = position
            if not polarity_locked:
                # 非 delta/pred 档位没有独立的锚点帧 ⇒ 在链的**第一步**判极性。
                polarity = _lock_polarity(tracker, image, roi_center)
                polarity_locked = True
            # selfpred 下 ROI 中心已经前移过，位移必须仍相对**链上一帧质心**度量，
            # 否则 accepted_steps 里存的会是"残差"而不是真实逐帧位移（hold 兜底会失真）。
            disp_ref = position if selfpred_mode else roi_center
            chain_xy = None
            if chain_grounded:
                # 逐步配准：用「上一帧 chain 位置处的模板」在当前帧里配准。
                # 模板每步刷新 ⇒ 模板与目标几乎没变（只隔 1 帧）⇒ 只剩亚像素定位噪声；
                # 被跟踪的物理点全程不变 ⇒ 质心偏置不再逐帧累积。
                chain_xy = _xcorr_chain_step(
                    frame_source,
                    step_frame - 1,
                    step_frame,
                    position,
                    roi_center,
                    cfg,
                    result,
                    polarity=polarity,
                )
            observation = tracker.step(image, roi_center)
            result["steps"] += 1
            if chain_xy is not None:
                observation_valid = True
                updated = np.asarray(chain_xy, dtype=np.float64).reshape(2).copy()
            elif chain_grounded:
                # chain 档位：**不**用质心观测兜底。
                # 机理：质心观测是「ROI 内加权质心」的**绝对**位置，它相对锚点带有
                # 一个系统偏置 b；而 chain 档位的链是以「锚点」为原点的物理点链，
                # 把质心绝对位置直接塞进链里 ⇒ 坐标系不一致、偏置被重新引入。
                # 正确的"无偏兜底"就是 **hold**：按已接受步的中位数（= 链自身速度）
                # 推进，纯位移、不引入偏置。等价于把质心的偏置 b 从这一步里减掉。
                observation_valid = False
                updated = np.asarray(observation.xy, dtype=np.float64).reshape(2).copy()
            else:
                observation_valid = bool(observation.valid)
                updated = np.asarray(observation.xy, dtype=np.float64).reshape(2).copy()
            if not observation_valid:
                consecutive_hold += 1
                if consecutive_hold > max_hold:
                    failed = True
                    failed_frame = step_frame
                    reason = f"lost_target:{step_frame}"
                    break
                # 观测缺失：按参考速度（稳健）或最后一次实测步长继续外推（hold-over）。
                # 参考速度优先取"已接受步的中位数"——单步离群跳变（实测 21 px 假跳变）
                # 若被当成步长盲推 8 步，会被放大成 130 px 的幻影位移。
                if chain_grounded:
                    # chain 档位：hold 用**链自身速度**（已接受步的中位数），
                    # 还没有已接受步时用管线 KF 参考速度。不用 last_step_vector——
                    # 那一步可能是离群跳变，盲推 8 步会放大成幻影位移。
                    if accepted_steps:
                        hold_window = int(cfg.replay_hold_window)
                        recent = (
                            accepted_steps[-hold_window:] if hold_window > 0 else accepted_steps
                        )
                        advance = np.median(np.asarray(recent, dtype=np.float64), axis=0)
                    elif ref_available:
                        advance = ref_velocity
                    else:
                        advance = last_step_vector
                elif not cfg.replay_hold_use_ref:
                    # 旧口径：按最后一次实测步长盲推（实测会把单步离群跳变放大成幻影位移）。
                    advance = last_step_vector
                elif len(accepted_steps) >= 3 and cfg.replay_hold_robust:
                    # 稳健参考 = 最近 N 步已接受步的中位数；末步若与它差得远（离群）就用
                    # 中位数，否则**优先用末步**（目标机动时末步更贴近当前运动，实测 001324）。
                    window = int(cfg.replay_hold_window)
                    recent = accepted_steps[-window:] if window > 0 else accepted_steps
                    median_step = np.median(np.asarray(recent, dtype=np.float64), axis=0)
                    last_accepted = accepted_steps[-1]
                    if float(np.linalg.norm(last_accepted - median_step)) > float(
                        cfg.replay_hold_outlier_tol_px
                    ):
                        advance = median_step
                    else:
                        advance = last_accepted
                elif accepted_steps:
                    advance = accepted_steps[-1]
                elif ref_available:
                    advance = ref_velocity
                else:
                    advance = last_step_vector
                position = position + advance
                step_records.append((False, np.zeros(2, dtype=np.float64)))
                result["held"] += 1
                continue
            consecutive_hold = 0
            # ---- 观测层 shrinkage：把灰度质心往「锚点 + 匀速预测」收缩 ----
            # 机理：快核的输出是 ROI 内的**灰度加权质心**，机身高光/拖尾会让它相对
            # 目标几何中心有 2~4 px 的系统性偏置；F1 的 inbox 判据容得下这点偏置，
            # 但伺服"击杀精度"（hit@5px）容不下 —— 偏置一跨过 5 px 门槛就整帧丢命中。
            # 收缩公式 x = α·x_centroid + (1-α)·x_pred：质心只提供轻微修正力，
            # 绝对偏置被按比例压掉，同时保留质心对机动的跟手能力。
            shrink_alpha = float(cfg.replay_obs_shrink_alpha)
            if shrink_alpha < 1.0 and fallback_velocity_available:
                pred_center = anchor + fallback_velocity * float(step_frame - frame)
                updated = shrink_alpha * updated + (1.0 - shrink_alpha) * pred_center
            # 逐帧位移门：单步跳几十 px 只可能是快核抓住了旁边的杂波亮点——
            # 那种观测**作废**（当无效处理），但**不弃权整帧**：按最后一次实测的
            # 单步位移 hold 一步继续往下走。旧语义（超门 ⇒ 整帧弃权走兜底）已被实测
            # 证伪（三个配置全部负收益：目标本身可以合法地每步走 20 px 以上）。
            step_vector = updated - disp_ref
            step_disp = float(np.linalg.norm(step_vector))
            if step_disp > result["max_step_disp"]:
                result["max_step_disp"] = step_disp
            # ⚠️ 必须是**入链之前**判"是不是第一步"：旧写法把 append 放在判定前面，
            # 于是 `not accepted_steps` 恒为 False ⇒ 首步离群门从来没真正触发过
            # （8 序列实测 `replay_step_rejected` 恒为 0）。
            is_first_step = len(accepted_steps) == 0
            if rel_tol > 0.0 and is_first_step and ref_available:
                # **首步**离群门：只校验重推的第一步。实测取证（VIDEO00029 frame 427）：
                # 源帧快核正常 → 第一步跳出 21 px 假跳变（目标真实只走 2.5 px/帧）→
                # 后面 7 步 hold 把这个假步长盲推 8 倍，放大成 130 px 幻影位移。
                # 只卡第一步是因为：第二步以后没有可靠的参考（快核步长本身的噪声就有
                # 2~10 px，任何阈值都会误杀真实机动，实测 12 px 门在 001324 上净亏）。
                if float(np.linalg.norm(step_vector - ref_velocity)) > rel_tol:
                    consecutive_hold += 1
                    result["held"] += 1
                    result["step_rejected"] += 1
                    if consecutive_hold > max_hold:
                        failed = True
                        failed_frame = step_frame
                        reason = f"step_jump:{step_frame}"
                        break
                    position = position + ref_velocity
                    step_records.append((False, np.zeros(2, dtype=np.float64)))
                    continue
                # 接受这一步。用已接受步的**中位数**修正参考速度（中位数抗单步离群）。
                accepted_steps.append(step_vector.copy())
                step_records.append((True, step_vector.copy()))
                if len(accepted_steps) >= 3:
                    ref_velocity = np.median(np.asarray(accepted_steps, dtype=np.float64), axis=0)
                ref_available = True
            else:
                accepted_steps.append(step_vector.copy())
                step_records.append((True, step_vector.copy()))
                if len(accepted_steps) >= 3:
                    ref_velocity = np.median(np.asarray(accepted_steps, dtype=np.float64), axis=0)
            if float(cfg.replay_step_gate_px) > 0.0 and step_disp > float(cfg.replay_step_gate_px):
                consecutive_hold += 1
                result["held"] += 1
                result["step_rejected"] += 1
                if consecutive_hold > max_hold:
                    failed = True
                    failed_frame = step_frame
                    reason = f"step_jump:{step_frame}"
                    break
                position = position + last_step_vector
                continue
            position = updated
            last_step_vector = step_vector
            last_valid_frame = step_frame
        if failed:
            # 失败：把"尽力而为"的最后位置带出去，供 hold / holdv 兜底使用。
            # 步长必须用**稳健参考**（已接受步的中位数 / 卡尔曼速度），不能用"最后一次
            # 实测步长"——那一步可能正是离群假跳变，holdv 会把它乘上剩余步数放大。
            if not cfg.replay_hold_use_ref:
                fallback_vector = last_step_vector
            elif len(accepted_steps) >= 3 and cfg.replay_hold_robust:
                window = int(cfg.replay_hold_window)
                recent = accepted_steps[-window:] if window > 0 else accepted_steps
                median_step = np.median(np.asarray(recent, dtype=np.float64), axis=0)
                last_accepted = accepted_steps[-1]
                if float(np.linalg.norm(last_accepted - median_step)) > float(
                    cfg.replay_hold_outlier_tol_px
                ):
                    fallback_vector = median_step
                else:
                    fallback_vector = last_accepted
            elif accepted_steps:
                fallback_vector = accepted_steps[-1]
            elif ref_available:
                fallback_vector = ref_velocity
            else:
                fallback_vector = last_step_vector
            result["failed"] = True
            result["reason"] = reason
            # 兜底必须与主路径**同一套融合口径**：delta/pred 档位上报的是
            # "锚点 + 位移"，兜底若直接报快核绝对位置，就会与相邻帧产生偏置跳变。
            if anchor_mode in ("delta", "pred") and not chain_grounded:
                fused_position = anchor + (position - kernel_start)
            else:
                fused_position = position
            # 链失败时也试一次模板匹配：链失锁（机动期）不等于目标丢了，
            # 源帧模板在目标帧里往往还能配上 ⇒ 把"弃权帧"救回来。
            if float(cfg.replay_xcorr_radius) > 0.0:
                refined = _xcorr_refine(
                    frame_source,
                    frame,
                    target_frame,
                    anchor,
                    fused_position,
                    cfg,
                    result,
                    polarity=polarity,
                )
                if refined is not None:
                    fused_position = refined
                    result["xcorr_rescued"] = int(result.get("xcorr_rescued", 0)) + 1
            result["best_effort"] = {
                "position": (float(fused_position[0]), float(fused_position[1])),
                "step_vector": (float(fallback_vector[0]), float(fallback_vector[1])),
                # 失败发生在第 failed_frame 步、且该步**没有推进位置** ⇒ 位置仍停在
                # failed_frame − 1 帧 ⇒ 还要补 (target − failed_frame + 1) 步。
                # （少算一步 ⇒ 001324 的 846 个兜底帧每帧欠推一步，实测净亏 0.5pp F1。）
                "remaining_steps": int(max(0, target_frame - failed_frame + 1)),
            }
            return result
        # ---- 批量稳健位移：把离群单步换成"步长中位数"再求和 ----
        # 为什么要批量而不是逐门：重推的 8 步是一次性算完的（环形缓冲里帧都在），
        # 所以可以**事后**看全 8 步再决定哪几步离群——逐门只能拿"上一帧步长"当参考，
        # 而上一帧恰恰可能就是那个 21 px 的假跳变（实测取证：step1 假跳变 21 px，
        # 后面 7 步 hold 把幻影位移放大到 130 px）。
        outlier_tol = float(cfg.replay_outlier_tol_px)
        if outlier_tol > 0.0 and step_records:
            valid_steps = [d for ok, d in step_records if ok]
            if len(valid_steps) >= 2:
                median_step = np.median(np.asarray(valid_steps, dtype=np.float64), axis=0)
                total = np.zeros(2, dtype=np.float64)
                replaced = 0
                for ok, d in step_records:
                    if ok and float(np.linalg.norm(d - median_step)) <= outlier_tol:
                        total += d
                    else:
                        total += median_step
                        replaced += 1
                position = chain_origin + total
                result["outlier_replaced"] = replaced
        if anchor_mode in ("delta", "pred") and not chain_grounded:
            # 位移融合：管线锚点 + 快核估计的位移（消掉快核的绝对质心偏置）。
            displacement = position - kernel_start
            # ---- 位移层 shrinkage：把实测位移向「参考位移 v·Δ」收缩 ----
            # 动机（来自实测的两难）：①观测层收缩治得了悬停目标（015000 +0.68 pp）
            # 却毁掉机动目标（001324 −10.8 pp），因为它把**链内 ROI 中心**也拽向预测；
            # ③只收缩**最终位移**，链内跟踪照旧 ⇒ 两类目标都不吃亏：
            #   · 悬停：参考位移 ≈ 0 ⇒ 位移被拉回 ≈ 0 ⇒ 等价于「上报锚点」；
            #   · 机动：参考位移 ≈ 真实位移 ⇒ 实测位移只提供修正力。
            disp_alpha = float(cfg.replay_disp_shrink_alpha)
            if disp_alpha < 1.0 and fallback_velocity_available:
                ref_displacement = fallback_velocity * float(int(cfg.delta))
                displacement = (
                    disp_alpha * displacement + (1.0 - disp_alpha) * ref_displacement
                )
                result["disp_shrunk"] = int(result.get("disp_shrunk", 0)) + 1
            # ---- 位移自适应融合：实测 vs 参考，按「两者差值」赋权 ----
            # 两个估计器**独立**：实测来自局部快核的重推链（精度高，但会被杂波拽走
            # ⇒ walk-off 帧整体偏 100+ px）；参考来自管线的卡尔曼速度（不会有 walk-off，
            # 但速度本身有 3~10 px 的中位误差）。差值 d 的大小就是「至少有一个错」的
            # 程度：d 小 ⇒ 两者一致 ⇒ 取更精确的实测；d 大 ⇒ 矛盾 ⇒ 实测可疑（杂波）
            # ⇒ 权重退向参考，把 walk-off 帧拉回锚点附近。
            merge_sigma = float(cfg.replay_disp_merge_sigma_px)
            if merge_sigma > 0.0 and fallback_velocity_available:
                ref_displacement = fallback_velocity * float(int(cfg.delta))
                gap = float(np.linalg.norm(displacement - ref_displacement))
                weight = 1.0 / (1.0 + (gap / merge_sigma) ** 2)
                displacement = weight * displacement + (1.0 - weight) * ref_displacement
                result["disp_merged"] = int(result.get("disp_merged", 0)) + 1
                result["disp_merge_weight_sum"] = float(
                    result.get("disp_merge_weight_sum", 0.0)
                ) + weight
            position = anchor + displacement
        # ---- 低速冻结：目标几乎不动 ⇒ 重推位移**锁死为 0** ----
        # 机理：红外小目标常常**悬停**（真值帧间位移 0~1 px/帧，见 015000）。此时
        # 重推链测到的位移只会是噪声/杂波（实测单步假跳变 13~21 px），8 步累积后
        # 整体偏掉 100+ px ⇒ 帧帧脱靶。速度低时直接上报管线锚点（管线在同一帧的
        # 位置误差实测 ~2 px），是**免费**的稳态答案。
        freeze_px = float(cfg.replay_freeze_speed_px)
        if (
            freeze_px > 0.0
            and fallback_velocity_available
            and float(np.linalg.norm(fallback_velocity)) < freeze_px
        ):
            position = anchor.copy()
            result["frozen"] = int(result.get("frozen", 0)) + 1
        # ---- 一致性门：重推结果与运动学预测偏离过大 ⇒ 判"跑飞"，改用运动学预测 ----
        # 机理：重推链的 ROI 偶尔会抓住旁边的杂波亮点（实测 014053 frame 427 报到
        # 180 px 外，下一帧又回来了 ⇒ **孤立单帧飞跳**，转台看到的就是一次抽搐）。
        # 判据用"锚点 + Δ·v"这个**独立**参考：重推是视觉推的，运动学是管线速度推的，
        # 两者独立；它们对不上 ⇒ 至少有一个错了，而跑飞的那个一定是重推。
        # 门限取"运动学预测自身的误差量级"以上（实测 014053 的运动学 p95 ≈ 67 px），
        # 于是正常帧（重推误差 p50 ≈ 2 px）几乎不会被误杀，跑飞的 180~400 px 一定被拦。
        consistency_gate = float(cfg.replay_consistency_gate_px)
        if consistency_gate > 0.0 and fallback_velocity_available:
            kinematic = anchor + float(int(cfg.delta)) * fallback_velocity
            if float(np.linalg.norm(position - kinematic)) > consistency_gate:
                position = kinematic
                result["consistency_rejected"] += 1
        # ---- 模板匹配位移精修（xcorr 档位）----
        # 把「锚点 + 质心链位移」换成「锚点 + 相关位移」：同一份模板跨两端配准，
        # 质心偏置在两端**精确抵消**（位移 p90 4.37→3.17 px、p95 8.13→6.34 px）。
        # 搜索中心 = 链给的位置（链负责"大概在哪"，模板负责"精确在哪"）。
        if (
            anchor_mode in ("delta", "pred")
            and float(cfg.replay_xcorr_radius) > 0.0
            and xcorr_mode in ("direct", "chain_direct", "stride")
        ):
            if xcorr_mode == "stride":
                # 分段配准：把跨 Δ 帧的大匹配换成「每 stride 帧配一次、模板每段刷新」。
                # 质心链照跑（chain_grounded 仍为 False）⇒ 链负责给出搜索中心与兜底。
                refined = _xcorr_stride_refine(
                    frame_source,
                    frame,
                    target_frame,
                    anchor,
                    position,
                    cfg,
                    result,
                    polarity=polarity,
                )
            else:
                refined = _xcorr_refine(
                    frame_source,
                    frame,
                    target_frame,
                    anchor,
                    position,
                    cfg,
                    result,
                    polarity=polarity,
                )
            if refined is not None:
                position = refined
            # ---- 「不自信就不上报」：宁可沉默，也不要给转台一个错坐标 ----
            # 模板配准的峰值低 ⇒ 这次重推缺自洽证据；但峰值低有两种成因，用
            # **链位移**区分：跟丢时链从锚点跑出去很远（位置是错的），机动时链
            # 几乎没跑动（位置其实是对的）。只有「峰值低 **且** 链跑得远」才弃权。
            peak_gate = float(cfg.replay_abstain_peak_below)
            if peak_gate > 0.0:
                peak = float(result.get("xcorr_peak_last", 1.0))
                displacement = float(np.linalg.norm(position - anchor))
                if peak < peak_gate and displacement > float(cfg.replay_abstain_disp_px):
                    result["xcorr_abstained"] = int(result.get("xcorr_abstained", 0)) + 1
                    continue
        new_det = dict(det)
        new_det["pos"] = (float(position[0]), float(position[1]))
        result["entries"].append(new_det)
    return result


def _replay_kf_detections(
    source_dets: list[dict],
    frame: int,
    target_frame: int,
    cfg: LookaheadConfig,
    kf_states,
    tracker: LocalPatchTracker | None,
    frame_source: FrameSource | None,
) -> dict:
    """档位 B-KF：**逐帧递推卡尔曼状态** + 局部质心观测（文档档位 B 原文口径）。

    每一步做三件事：

    1. **预测**：``x <- F*x``（位置 += 速度）、``P <- F*P*F^T + Q``；
    2. **观测**：以**预测位置**为 ROI 中心跑局部质心快核 => 观测 ``z``；
    3. **更新**（观测有效时）：标准卡尔曼增益 ``K = P*H^T*S^-1``、``x <- x + K*(z - H*x)``。

    与"只递推质心"（anchor="kernel"）的本质差别：

    * 只递推质心时，ROI 中心 = 上一步的质心 => 质心一旦有偏（ROI 装不下目标、
      截断了长条的一端），下一步的 ROI 就跟着偏，误差**自我放大**；
    * 递推卡尔曼状态时，ROI 中心 = **运动学预测** => ROI 始终落在目标附近，
      质心偏置被卡尔曼增益限制住（观测噪声 ``R`` 越大，增益越小），
      而且速度分量会被观测**反向修正**，不再是把初始噪声乘 Δ。
    """
    max_hold = int(cfg.replay_max_hold)
    if max_hold <= 0 and not cfg.require_valid:
        max_hold = 10**6
    result: dict = {
        "entries": [],
        "failed": False,
        "steps": 0,
        "reason": "",
        "max_step_disp": 0.0,
        "anchor_calls": 0,
        "held": 0,
        "step_rejected": 0,
        "best_effort": None,
    }
    q = np.diag(
        [
            float(cfg.replay_kf_q_pos),
            float(cfg.replay_kf_q_pos),
            float(cfg.replay_kf_q_vel),
            float(cfg.replay_kf_q_vel),
        ]
    ).astype(np.float64)
    r = float(cfg.replay_kf_r) ** 2
    measurement_noise = np.diag([r, r]).astype(np.float64)

    for det in source_dets:
        state, covariance = _initial_kf_state(det, frame, cfg, kf_states)
        # 目标极性：kf 档位没有独立的锚点帧 ⇒ 在**源帧**、用初始状态的位置判一次。
        _lock_polarity(
            tracker,
            None if frame_source is None else frame_source.get(frame),
            state[0:2],
        )
        last_valid_frame = frame
        consecutive_hold = 0
        failed = False
        reason = ""
        for step_frame in range(frame + 1, target_frame + 1):
            image = None if frame_source is None else frame_source.get(step_frame)
            if image is None or getattr(image, "ndim", 0) != 2:
                failed = True
                failed_frame = step_frame
                reason = f"missing_frame:{step_frame}"
                break
            # 1) 预测：位置 += 速度。
            state = _KF_F @ state
            covariance = _KF_F @ covariance @ _KF_F.T + q
            # 2) 观测：ROI 中心取**预测位置**（不是上一步的质心）。
            observation = tracker.step(image, state[0:2])
            result["steps"] += 1
            if not observation.valid:
                consecutive_hold += 1
                if consecutive_hold > max_hold:
                    failed = True
                    failed_frame = step_frame
                    reason = f"lost_target:{step_frame}"
                    break
                result["held"] += 1
                continue
            consecutive_hold = 0
            # 3) 更新：标准卡尔曼增益。
            measured = np.asarray(observation.xy, dtype=np.float64).reshape(2)
            innovation = measured - (_KF_H @ state)
            innovation_covariance = _KF_H @ covariance @ _KF_H.T + measurement_noise
            gain = covariance @ _KF_H.T @ np.linalg.inv(innovation_covariance)
            state = state + gain @ innovation
            covariance = (_KF_I4 - gain @ _KF_H) @ covariance
            step_disp = float(np.linalg.norm(measured - state[0:2]))
            if step_disp > result["max_step_disp"]:
                result["max_step_disp"] = step_disp
            if float(cfg.replay_step_gate_px) > 0.0 and step_disp > float(cfg.replay_step_gate_px):
                # 观测作废（快核跳到杂波）⇒ 这一步只预测不更新，继续往下走。
                consecutive_hold += 1
                result["held"] += 1
                result["step_rejected"] += 1
                if consecutive_hold > max_hold:
                    failed = True
                    failed_frame = step_frame
                    reason = f"step_jump:{step_frame}"
                    break
                continue
            last_valid_frame = step_frame
        if failed:
            result["failed"] = True
            result["reason"] = reason
            result["best_effort"] = {
                "position": (float(state[0]), float(state[1])),
                "step_vector": (float(state[2]), float(state[3])),
                "remaining_steps": int(target_frame - last_valid_frame),
            }
            return result
        new_det = dict(det)
        new_det["pos"] = (float(state[0]), float(state[1]))
        result["entries"].append(new_det)
    return result


# ---------------------------------------------------------------------------
# rolling 档位：一条链跨物理帧持续推进
# ---------------------------------------------------------------------------


@dataclass
class _RollingChain:
    """一条 track 的滚动链状态（**跨物理帧持久**，rolling 档位专用）。

    字段与任务书 3.1 的状态表一一对应：

    * ``ring``：长度 ``Δ+1`` 的环形缓冲，``ring[frame % (Δ+1)]`` 存该物理帧的链位置
      ``chain_pos[frame]``；该帧链不可用（连续失效超限 / 还没起步）时存 ``None``。
      长度取 ``Δ+1`` 是因为物理帧 ``t`` 收到的是第 ``j = t−Δ`` 帧的锚点，
      而 ``j`` 与 ``t`` 在 ``mod (Δ+1)`` 下不同 ⇒ 槽位不会被覆盖。
    * ``roi``：下一步喂给快核的 ROI 中心。
    * ``pos``：链的当前位置估计（观测失效时按步长续推，链因此始终能重新捕获目标）。
    * ``last_step``：最后一次**实测**的单步位移向量（hold-over 用）。
    * ``offset``：最近一次锚点残差 ``A_j − chain_pos[j]``（上报与重锚都用它）。
    * ``report_prev``：上一帧的上报坐标（``holdc`` 兜底用）。
    * ``report_step``：上一次上报的帧间位移（``rolling_hold_mode="prev_report"`` 用）。
    * ``hold``：连续观测失效步数（与 ``replay_max_hold`` 比较）。
    * ``anchor``：最近一次采纳的锚点，已按 ``Δ·v`` 前推到**当前物理帧**，
      用作重捕获边界（链位置不得离它太远）。
    """

    ring: list
    roi: np.ndarray | None = None
    pos: np.ndarray | None = None
    last_step: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    anchor: np.ndarray | None = None
    report_prev: np.ndarray | None = None
    report_step: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    hold: int = 0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    """链的**平滑速度**（已接受步长的指数滑动平均）。

    为什么要它：快核的观测窗口是 ``win_size`` 方形；目标单帧位移一旦接近窗口
    半径，目标会被窗口截断 ⇒ 阈值化后的**质心系统性偏向旧位置** ⇒ 实测步长
    偏小 ⇒ 链在快速目标上**逐帧累积滞后**（实测末段 18 px/帧时滞后 2~6 帧）。
    用 ``vel`` 把 ROI 中心**预测**到 ``pos + vel``，目标就重新落回窗口中央，
    质心不再有偏，测得的单步位移才是真的。
    """


def _rolling_fallback(
    det: dict,
    source_frame: int,
    cfg: LookaheadConfig,
    kf_states,
    dets_by_frame: list[list[dict]],
    chain: _RollingChain | None,
    stats: dict,
) -> np.ndarray | None:
    """rolling 档位的兜底坐标；返回 ``None`` 表示"该帧不输出"。

    * ``holdc``：**上一帧上报 + 参考速度**（连续性最好，rolling 的首选）；
    * ``holdv`` / ``kinematic``：管线锚点 + ``Δ·v``（匀速外推，链从未起步时也走这条）；
    * ``hold``：链最后一个位置（链从没起步时退回匀速外推）；
    * ``stale``：原样搬运锚点；
    * ``abstain``：不输出（宁可让转台知道没坐标，也不给半真半假的）。
    """
    fallback = str(cfg.replay_fallback).strip().lower()
    if fallback == "abstain":
        return None
    track_id = int(det.get("track_id", -1))
    anchor = _base_position(det, source_frame, cfg, kf_states)
    velocity, available = _estimate_velocity(
        dets_by_frame,
        source_frame,
        track_id,
        source=cfg.velocity_source,
        window=int(cfg.velocity_window),
        kf_states=kf_states,
    )
    if not available:
        stats["missing_velocity"] += 1
    if fallback == "stale":
        return anchor
    if fallback == "holdc":
        if chain is not None and chain.report_prev is not None:
            return chain.report_prev + velocity
        # 链从未起步 ⇒ 没有上一帧上报可续 ⇒ 退化成"锚点 + 匀速外推"（holdv）。
        return anchor + int(cfg.delta) * velocity
    if fallback == "hold":
        if chain is not None and chain.pos is not None:
            return chain.pos
        return anchor + int(cfg.delta) * velocity
    return anchor + int(cfg.delta) * velocity


def _build_rolling_report_stream(
    dets_by_frame: list[list[dict]],
    num_frames: int,
    cfg: LookaheadConfig,
    kf_states,
    tracker: LocalPatchTracker | None,
    frame_source: FrameSource | None,
) -> tuple[list[list[dict]], dict]:
    """rolling 档位：**一条链跨物理帧持续推进，每物理帧只走一步**。

    与 per-frame 重推的本质区别：per-frame 每帧都从源帧 ``f = t−Δ`` 重新起链、
    重跑 Δ 步 ⇒ 相邻帧的链起点不同、质心偏置 ``b`` 的实现路径不同 ⇒ 上报坐标在
    帧间"随机跳"（实测 ``max_jitter`` 366 px）。rolling 全程只有**一条**链：
    起点固定、每帧只走一步 ⇒ 帧间位移只含目标的真实运动 ⇒ 抖动掉一个数量级；
    而且**不需要**回到源帧重跑快核做锚点校验 ⇒ 源帧失锁的整段作废
    （实测 001324 有 846 帧）结构性消失。

    信念更新（**不重复计入**的关键）：链的位置 ``pos`` 就是上报坐标。
    锚点 ``A_j``（第 ``j = t−Δ`` 帧的管线位置）到达时：

        residual = A_j − belief(j)          # belief(j) = 链对第 j 帧位置的**当前**信念
        把 residual 加到 pos 与环形缓冲里**每一个**有效槽位

    环形缓冲里存的是"当前信念"而不是"原始测量"：每次采纳残差时把缓冲里所有槽位
    一起平移，于是 ``belief(j)`` 天然包含了此前所有修正 ⇒ residual 是**增量**修正，
    不会把上一轮修正重复计入（早期版本只把 offset 加在 ROI 上、却把 offset 又加进
    上报坐标，导致修正被计入两次）。

    三道防 runaway 的门（缺任何一道都会指数发散）：

    1. **单步位移门** ``rolling_max_step_px``：一步跳太远 = 快核抓住杂波 ⇒ 该步作废，
       按 hold 续推。这切断"位置飞了 → 步长巨大 → 续推又放大"的正反馈。
    2. **锚点残差门** ``rolling_max_offset_px``：残差过大 = 链已经跟丢 ⇒ 不采纳该锚点
       （而不是把巨大的残差注入链里）。
    3. **重捕获边界** ``rolling_reacquire_px``：链位置离"锚点 + Δ·v"太远 ⇒ 投影回
       边界并清零步长，链在下一个锚点周期内被拉回。

    兜底优先级：连续失效 ≤ ``replay_max_hold`` ⇒ 按步长续推；超限 ⇒
    ``cfg.replay_fallback``（``holdc`` = 上一帧上报 + v，连续性最好）；链从未起步
    ⇒ 锚点 + 匀速外推。

    上报窗口与其它臂一致：物理帧 ``t < Δ`` 没有源帧 ⇒ 不输出（``no_source_frames``）。
    """
    delta = int(cfg.delta)
    max_hold = int(cfg.replay_max_hold)
    recenter = bool(cfg.rolling_recenter)
    hold_mode = str(cfg.rolling_hold_mode).strip().lower()
    if hold_mode not in ("last_step", "prev_report", "vel"):
        raise ValueError(f"unknown rolling_hold_mode: {cfg.rolling_hold_mode!r}")
    vel_alpha = float(cfg.rolling_vel_alpha)
    step_gate = float(cfg.rolling_max_step_px)
    if step_gate <= 0.0:
        step_gate = float("inf")
    max_offset = float(cfg.rolling_max_offset_px)
    reacquire = float(cfg.rolling_reacquire_px)

    ring_size = delta + 1
    report: list[list[dict]] = [[] for _ in range(num_frames)]
    stats: dict = {
        "mode": "replay",
        "delta": delta,
        "applied": 0,
        "abstained": 0,
        "fallback_used": 0,
        "missing_velocity": 0,
        "replay_steps": 0,
        "frames": num_frames,
        "no_source_frames": 0,
        "replay_failed": 0,
        "replay_abstained": 0,
        "replay_step_jump": 0,
        "replay_max_step_disp": 0.0,
        "replay_anchor_steps": 0,
        "replay_held_steps": 0,
        "replay_step_rejected": 0,
        "replay_outlier_replaced": 0,
        "replay_consistency_rejected": 0,
        "rolling_dead_steps": 0,
        "rolling_step_rejected": 0,
        "rolling_offset_skipped": 0,
        "rolling_reacquired": 0,
    }
    chains: dict[int, _RollingChain] = {}

    for target_frame in range(num_frames):
        # ---- (0) 新 track 起步：链的起点 = 管线在该帧的锚点 ----
        records = dets_by_frame[target_frame] if target_frame < len(dets_by_frame) else []
        for det in records:
            track_id = int(det.get("track_id", -1))
            if track_id in chains:
                continue
            anchor = np.asarray(
                _base_position(det, target_frame, cfg, kf_states), dtype=np.float64
            ).reshape(2)
            chain = _RollingChain(ring=[None] * ring_size)
            chain.pos = anchor.copy()
            chain.roi = anchor.copy()
            chain.anchor = anchor.copy()
            chain.ring[target_frame % ring_size] = anchor.copy()
            chains[track_id] = chain

        # ---- (1) 推进链：每条链每物理帧只走一步 ----
        image = None if frame_source is None else frame_source.get(target_frame)
        observable = image is not None and getattr(image, "ndim", 0) == 2
        for chain in chains.values():
            accepted = False
            # 链已死（pos 为 None）⇒ 不再喂快核：roi 是 None 会让快核崩，
            # 而且死链的观测没有意义（链必须由锚点重捕获，见 (2)）。
            if observable and tracker is not None and chain.pos is not None:
                # 预测式 ROI 中心：把窗口中心放到"上一帧位置 + 平滑速度"上，
                # 快目标就不会被窗口截断 ⇒ 质心无偏 ⇒ 测得的单步位移是真的。
                roi_center = chain.pos + chain.vel
                observation = tracker.step(image, roi_center)
                stats["replay_steps"] += 1
                if observation.valid:
                    updated = np.asarray(observation.xy, dtype=np.float64).reshape(2).copy()
                    step = updated - chain.pos
                    displacement = float(np.linalg.norm(step))
                    if displacement > stats["replay_max_step_disp"]:
                        stats["replay_max_step_disp"] = displacement
                    if displacement <= step_gate:
                        chain.last_step = step
                        chain.vel = (1.0 - vel_alpha) * chain.vel + vel_alpha * step
                        chain.pos = updated
                        chain.hold = 0
                        accepted = True
                    else:
                        # 离群步 = 快核抓住了杂波（城市灯光等）⇒ 该步作废，走 hold
                        stats["rolling_step_rejected"] += 1
            if not accepted:
                chain.hold += 1
                if hold_mode == "last_step":
                    advance = chain.last_step
                elif hold_mode == "prev_report":
                    advance = chain.report_step
                else:
                    advance = chain.vel
                if chain.hold <= max_hold:
                    if chain.pos is not None:
                        chain.pos = chain.pos + advance
                    stats["replay_held_steps"] += 1
                else:
                    stats["rolling_dead_steps"] += 1
                    chain.pos = None
            # ---- 重捕获边界：链位置不得离"锚点 + Δ·v"太远 ----
            if chain.pos is not None and chain.anchor is not None and reacquire > 0.0:
                gap = chain.pos - chain.anchor
                distance = float(np.linalg.norm(gap))
                if distance > reacquire:
                    chain.pos = chain.anchor + gap * (reacquire / distance)
                    # 步长只**限幅**不置零：置零会让快目标在随后的 hold 里彻底停住
                    # （实测 46 次重捕获把链的动量反复抹掉 ⇒ 逐帧累积成 2~6 帧滞后）。
                    speed = float(np.linalg.norm(chain.vel))
                    if speed > step_gate:
                        chain.vel = chain.vel * (step_gate / speed)
                    speed = float(np.linalg.norm(chain.last_step))
                    if speed > step_gate:
                        chain.last_step = chain.last_step * (step_gate / speed)
                    chain.hold = 0
                    stats["rolling_reacquired"] += 1
            chain.roi = chain.pos
            chain.ring[target_frame % ring_size] = (
                None if chain.pos is None else chain.pos.copy()
            )

        # ---- (2) 接收滞后 Δ 帧送达的锚点，做**增量**信念修正 ----
        source_frame = target_frame - delta
        if 0 <= source_frame < len(dets_by_frame):
            for det in dets_by_frame[source_frame]:
                track_id = int(det.get("track_id", -1))
                chain = chains.get(track_id)
                if chain is None:
                    continue
                anchor = np.asarray(
                    _base_position(det, source_frame, cfg, kf_states), dtype=np.float64
                ).reshape(2)
                # 锚点前推到当前物理帧，作为重捕获边界的中心
                velocity, available = _estimate_velocity(
                    dets_by_frame,
                    source_frame,
                    track_id,
                    source=cfg.velocity_source,
                    window=int(cfg.velocity_window),
                    kf_states=kf_states,
                )
                lead = velocity * float(delta) if available else np.zeros(2, dtype=np.float64)

                belief = chain.ring[source_frame % ring_size]
                if belief is None:
                    # 链在源帧没有信念（刚起步 / 那几帧死了）⇒ 本轮做不了残差修正。
                    # 只有链**也死了**才需要重捕获；链还活着时它有自己的观测，
                    # 不能用运动学外推把它覆盖掉（否则会抹掉刚拿到的快核观测）。
                    if chain.pos is None:
                        chain.pos = anchor + lead
                        chain.roi = chain.pos
                        chain.hold = 0
                        chain.last_step = (
                            velocity if available else np.zeros(2, dtype=np.float64)
                        )
                        chain.vel = chain.last_step
                        chain.ring[source_frame % ring_size] = anchor.copy()
                        stats["rolling_reacquired"] += 1
                        stats["replay_anchor_steps"] += 1
                    chain.anchor = anchor + lead
                    if recenter:
                        chain.roi = chain.pos
                    continue

                residual = anchor - belief
                distance = float(np.linalg.norm(residual))
                if max_offset > 0.0 and distance > max_offset:
                    # 链已跟丢（残差过大）⇒ 不采纳，避免把巨大残差注入链里。
                    # **同时不能更新 chain.anchor**：锚点既然被判定为不可信，
                    # 就不能再拿它当"重捕获边界"，否则边界会把链拖向一个已被
                    # 否决的位置（残差门与边界门自相矛盾）。
                    stats["rolling_offset_skipped"] += 1
                    if chain.pos is None:
                        # 链已死且锚点也救不回来 ⇒ 只能以锚点 + Δv 重新起步
                        chain.pos = anchor + lead
                        chain.roi = chain.pos
                        chain.hold = 0
                        chain.last_step = (
                            velocity if available else np.zeros(2, dtype=np.float64)
                        )
                        chain.vel = chain.last_step
                        stats["rolling_reacquired"] += 1
                    continue
                # 采纳：把整条链的信念（当前位置 + 历史槽位）一起平移 residual
                for slot in range(ring_size):
                    value = chain.ring[slot]
                    if value is not None:
                        chain.ring[slot] = value + residual
                if chain.pos is None:
                    # 链中途死了，但源帧有信念 ⇒ 从锚点沿速度重推 Δ 步复活
                    chain.pos = anchor + lead
                    chain.hold = 0
                    chain.last_step = (
                        velocity if available else np.zeros(2, dtype=np.float64)
                    )
                    chain.vel = chain.last_step
                    stats["rolling_reacquired"] += 1
                else:
                    chain.pos = chain.pos + residual
                chain.roi = chain.pos
                stats["replay_anchor_steps"] += 1
                chain.anchor = anchor + lead
                if recenter:
                    chain.roi = chain.pos

        # ---- (3) 上报 ----
        if source_frame < 0 or source_frame >= len(dets_by_frame):
            stats["no_source_frames"] += 1
            continue
        source_dets = dets_by_frame[source_frame]
        if not source_dets:
            continue
        entries: list[dict] = []
        used_fallback = False
        abstained = 0
        for det in source_dets:
            track_id = int(det.get("track_id", -1))
            chain = chains.get(track_id)
            position = None
            if chain is not None and chain.pos is not None:
                position = chain.pos
            if position is None:
                position = _rolling_fallback(
                    det, source_frame, cfg, kf_states, dets_by_frame, chain, stats
                )
                if position is None:
                    abstained += 1
                    continue
                used_fallback = True
            new_det = dict(det)
            new_det["pos"] = (float(position[0]), float(position[1]))
            entries.append(new_det)
            if chain is not None:
                previous = chain.report_prev
                chain.report_prev = np.asarray(position, dtype=np.float64).reshape(2).copy()
                chain.report_step = (
                    chain.report_prev - previous
                    if previous is not None
                    else np.zeros(2, dtype=np.float64)
                )
        report[target_frame] = entries
        if used_fallback:
            stats["fallback_used"] += 1
            stats["replay_failed"] += 1
        if abstained:
            stats["replay_abstained"] += 1

    _finalize_stats(report, stats, num_frames)
    _apply_output_slew_limit(report, cfg, stats)
    return report, stats


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def estimate_velocity(
    dets_by_frame: list[list[dict]],
    frame: int,
    track_id: int,
    *,
    source: str = "diff1",
    window: int = 3,
    kf_states: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] | None = None,
) -> np.ndarray:
    """估计某条 track 在 `frame` 时刻的速度（单位：像素/record）。

    - ``source="kf"``：取 ``kf_states[frame][track_id]`` 的速度分量；
      该 track 在该帧没有 KF 状态时返回 ``np.zeros(2)``。
    - ``source="diff1"``：``pos(frame) - pos(frame-1)``（同一 track_id）；
      上一帧没有该 track 时返回 ``np.zeros(2)``。
    - ``source="diffN"``：对该 track 最近 ``window`` 个**出现过的**输出位置做
      最小二乘直线拟合，取斜率（像素/record）；不足 2 个点时返回 ``np.zeros(2)``。
    返回 dtype float64、shape (2,)。
    """
    velocity, _ = _estimate_velocity(
        dets_by_frame,
        frame,
        track_id,
        source=source,
        window=window,
        kf_states=kf_states,
    )
    return velocity


def _build_refine_report_stream(
    dets_by_frame: list[list[dict]],
    num_frames: int,
    cfg: LookaheadConfig,
    kf_states,
    tracker: LocalPatchTracker | None,
    frame_source: FrameSource | None,
) -> tuple[list[list[dict]], dict]:
    """refine 档位：**绝对位置永远锚在管线上**，快核只做**一次**局部修正。

    上报公式::

        report(t) = P(f) + Δ·v̂(f) + δ(t),      f = t − Δ
        δ(t) = 快核在**帧 t** 上以 K(t) = P(f) + Δ·v̂(f) 为中心的一次局部修正

    为什么是这个形状（三个设计约束同时满足）：

    1. **零漂移**——``P(f)`` 是管线自己的输出，**每一帧都重新取**，
       链/积分器里没有任何会累积的量 ⇒ 不存在"累积漂移"这一失效模式
       （per-frame 重推把误差累积在 8 步重推链里，rolling 把它累积在链上）。
    2. **低抖动**——δ(t) 是一次**局部**修正，量级是"预测误差"（几 px）而不是
       "重推路径差"（几十~几百 px）。per-frame 档位的 366 px 抖动就是这么来的：
       相邻物理帧的源帧差 1 ⇒ 重推路径不同 ⇒ 质心偏置不同 ⇒ 上报坐标跳变；
       这里路径每帧都一样（都从 P(f) 出发），跳变只剩 δ 的抖动。
    3. **不拒帧**——δ 拿不到就用 δ=0（= 运动学预测），**永远有输出**，
       不会像 per-frame 那样整段作废转兜底（兜底与重推是**两个不同的估计器**，
       交替出现正是抖动的第二大来源）。

    局限：预测误差大时（例如目标强加速，Δ=8 的匀速外推偏 40 px），
    快核的 ROI 会把目标截断 ⇒ δ 有偏（向预测方向偏），修正不完整但**不发散**。
    """
    delta = int(cfg.delta)
    gate = float(cfg.refine_gate_px)
    if gate <= 0.0:
        gate = float("inf")

    report: list[list[dict]] = [[] for _ in range(num_frames)]
    stats: dict = {
        "mode": "replay",
        "delta": delta,
        "applied": 0,
        "abstained": 0,
        "fallback_used": 0,
        "missing_velocity": 0,
        "replay_steps": 0,
        "frames": num_frames,
        "no_source_frames": 0,
        "replay_failed": 0,
        "replay_abstained": 0,
        "replay_step_jump": 0,
        "replay_max_step_disp": 0.0,
        "replay_anchor_steps": 0,
        "replay_held_steps": 0,
        "replay_step_rejected": 0,
        "replay_outlier_replaced": 0,
        "replay_consistency_rejected": 0,
        "refine_applied": 0,
        "refine_rejected": 0,
        "refine_no_obs": 0,
    }

    for target_frame in range(num_frames):
        source_frame = target_frame - delta
        if source_frame < 0 or source_frame >= len(dets_by_frame):
            stats["no_source_frames"] += 1
            continue
        source_dets = dets_by_frame[source_frame]
        if not source_dets:
            continue

        image = None if frame_source is None else frame_source.get(target_frame)
        observable = image is not None and getattr(image, "ndim", 0) == 2

        entries: list[dict] = []
        for det in source_dets:
            track_id = int(det.get("track_id", -1))
            anchor = np.asarray(
                _base_position(det, source_frame, cfg, kf_states), dtype=np.float64
            ).reshape(2)
            velocity, available = _estimate_velocity(
                dets_by_frame,
                source_frame,
                track_id,
                source=cfg.velocity_source,
                window=int(cfg.velocity_window),
                kf_states=kf_states,
            )
            if not available:
                stats["missing_velocity"] += 1
            predicted = anchor + float(delta) * velocity

            position = predicted
            if observable and tracker is not None:
                observation = tracker.step(image, predicted)
                stats["replay_steps"] += 1
                if observation.valid:
                    observed = np.asarray(observation.xy, dtype=np.float64).reshape(2)
                    correction = observed - predicted
                    distance = float(np.linalg.norm(correction))
                    if distance > stats["replay_max_step_disp"]:
                        stats["replay_max_step_disp"] = distance
                    if distance <= gate:
                        position = observed
                        stats["refine_applied"] += 1
                    else:
                        stats["refine_rejected"] += 1
                else:
                    stats["refine_no_obs"] += 1

            new_det = dict(det)
            new_det["pos"] = (float(position[0]), float(position[1]))
            entries.append(new_det)

        report[target_frame] = entries
        if entries:
            stats["applied"] += 1
            stats["replay_anchor_steps"] += len(entries)
        else:
            stats["abstained"] += 1

    _finalize_stats(report, stats, num_frames)
    _apply_output_slew_limit(report, cfg, stats)
    return report, stats


def build_report_stream(
    dets_by_frame: list[list[dict]],
    num_frames: int,
    cfg: LookaheadConfig,
    *,
    kf_states: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] | None = None,
    frame_source: FrameSource | None = None,
) -> tuple[list[list[dict]], dict]:
    """把原管线的输出流 D(f) 变成上报流 report(t)。

    - ``mode="off"``：``report[t] == D(t)``（**逐位**相同的拷贝，不做任何数值运算）。
    - ``mode="stale"``：``report[t] = D(t-Δ)``（位置/分数/track_id 原样搬运）。
    - ``mode="kinematic"``：``report[t]`` 的位置 = 源帧位置 + ``Δ * v``，
      其余字段（score / track_id / infilled）与源帧一致。
    - ``mode="replay"``：从源帧锚点出发，用局部质心快核逐帧重推到 t；
      重推成功用重推位置，失败按 ``cfg.replay_fallback`` 兜底
      （kinematic → 同 kinematic 臂；stale → 原样搬运；abstain → 该帧不输出）。

    位置起点：``cfg.position_source="output"`` 用 D(f) 的位置；``"kf"`` 用 KF 位置。
    报告的物理帧 t 对应源帧 f = t - Δ；``t < Δ`` 的帧没有源帧 ⇒ 该帧不输出。

    返回 ``(report_by_frame, stats)``：
    ``report_by_frame`` 长度 == ``num_frames``，元素是检测 dict 列表；
    ``stats`` 至少含 ``mode``/``delta``/``applied``/``abstained``/``fallback_used``/
    ``missing_velocity``/``replay_steps`` 六个键。

    补充口径（本模块的实现约定，契约未展开的部分）：
    * ``applied`` = 产出了至少一个上报点的物理帧数；``abstained`` = 没有产出上报点的
      物理帧数（两者互补，合计 == ``num_frames``）；
    * ``fallback_used`` = 触发 replay 兜底的帧数；``replay_abstained`` = 其中
      ``replay_fallback="abstain"``（该帧直接不输出）的帧数；
    * ``missing_velocity`` = 速度估计**没有数据**的 (帧, track) 次数（静止目标不算）；
    * ``replay_steps`` = 实际调用快核的重推步数；
    * ``no_source_frames`` = ``t < Δ``（没有源帧）的帧数。
    """
    num_frames = int(num_frames)
    delta = int(cfg.delta)
    mode = str(cfg.mode).strip().lower()
    fallback = str(cfg.replay_fallback).strip().lower()
    if mode not in ("off", "stale", "kinematic", "replay"):
        raise ValueError(f"unknown lookahead mode: {cfg.mode!r}")
    if fallback not in ("kinematic", "stale", "abstain", "hold", "holdv", "holdc"):
        raise ValueError(f"unknown replay_fallback: {cfg.replay_fallback!r}")

    report: list[list[dict]] = [[] for _ in range(num_frames)]
    stats: dict = {
        "mode": mode,
        "delta": delta,
        "applied": 0,
        "abstained": 0,
        "fallback_used": 0,
        "missing_velocity": 0,
        "replay_steps": 0,
        "frames": num_frames,
        "no_source_frames": 0,
        "replay_failed": 0,
        "replay_abstained": 0,
        "replay_step_jump": 0,
        "replay_max_step_disp": 0.0,
        "replay_anchor_steps": 0,
        "replay_held_steps": 0,
        "replay_step_rejected": 0,
        "replay_outlier_replaced": 0,
        "replay_consistency_rejected": 0,
        "replay_rollback_used": 0,
        "replay_rollback_extra_steps": 0,
        "replay_rollback_abstained": 0,
    }

    if mode == "off":
        # 逐位透传：只拷 dict 本身，**不做任何数值运算**（硬验收项：与原脚本逐位一致）。
        for index in range(num_frames):
            source_dets = dets_by_frame[index] if index < len(dets_by_frame) else []
            report[index] = [dict(det) for det in source_dets]
        _finalize_stats(report, stats, num_frames)
        return report, stats

    tracker = None
    if mode == "replay":
        tracker_kwargs = dict(
            win_size=int(cfg.win_size),
            th_k=float(cfg.th_k),
            min_energy=float(cfg.min_energy),
            min_contrast=float(cfg.min_contrast),
            mode=str(cfg.fast_mode),
            peak_window=int(cfg.peak_window),
            bg_window=int(cfg.bg_window),
        )
        polarity_mode = str(cfg.replay_polarity).strip().lower()
        if polarity_mode == "bright":
            # 默认路径：仍是原来的 LocalPatchTracker，**一行未改** ⇒ 逐位一致。
            tracker = LocalPatchTracker(**tracker_kwargs)
        else:
            tracker = PolarityTracker(
                polarity=polarity_mode,
                dark_ratio=float(cfg.replay_polarity_ratio),
                **tracker_kwargs,
            )
        # replay 档位的两种实现：perframe = 每物理帧从源帧重新起链重推（默认，保持现状
        # 可复现）；rolling = 一条链跨物理帧持续推进、每帧只走一步（抖动更低，且源帧
        # 失锁不再整段作废）。两条路径**互不影响**，这里的分派只决定走哪一条。
        replay_mode = str(cfg.replay_mode).strip().lower()
        if replay_mode not in ("perframe", "rolling", "refine"):
            raise ValueError(f"unknown replay_mode: {cfg.replay_mode!r}")
        if replay_mode == "refine":
            return _build_refine_report_stream(
                dets_by_frame,
                num_frames,
                cfg,
                kf_states,
                tracker,
                frame_source,
            )
        if replay_mode == "rolling":
            return _build_rolling_report_stream(
                dets_by_frame, num_frames, cfg, kf_states, tracker, frame_source
            )

    # 连续性兜底（holdc）用：每条 track 最近一次**实际上报**的位置。
    # 这是**输出级**的小状态（不回写管线任何估计），只用来保证上报轨迹连续。
    last_reported_pos: dict[int, np.ndarray] = {}

    def _remember(entries: list[dict]) -> None:
        for det in entries:
            tid = int(det.get("track_id", -1))
            pos = det.get("pos")
            if pos is None:
                continue
            last_reported_pos[tid] = np.asarray(pos, dtype=np.float64).reshape(2).copy()

    for target_frame in range(num_frames):
        source_frame = target_frame - delta
        if source_frame < 0 or source_frame >= len(dets_by_frame):
            stats["no_source_frames"] += 1
            continue
        source_dets = dets_by_frame[source_frame]
        rolled_back = False
        if not source_dets and mode == "replay" and int(cfg.replay_rollback_max_frames) > 0:
            # 源帧检测器漏检 ⇒ 退到**最近一次有检测**的帧当锚点，多走几步重推。
            back = int(cfg.replay_rollback_max_frames)
            for step_back in range(1, back + 1):
                candidate = source_frame - step_back
                if candidate < 0:
                    break
                if dets_by_frame[candidate]:
                    source_frame = candidate
                    source_dets = dets_by_frame[candidate]
                    stats["replay_rollback_used"] += 1
                    stats["replay_rollback_extra_steps"] += step_back
                    rolled_back = True
                    break
        if not source_dets:
            continue

        if mode == "stale":
            report[target_frame] = [dict(det) for det in source_dets]
            continue

        if mode == "kinematic":
            report[target_frame] = _kinematic_detections(
                dets_by_frame, source_dets, source_frame, cfg, kf_states, stats
            )
            continue

        # 回退后源帧比 t−Δ 更早 ⇒ 重推步数变多 ⇒ 运动学参考位移必须按**真实步数**
        # 算（否则 hold / 一致性门拿 Δ 帧的参考去比一个更长的链）。
        chain_cfg = cfg
        if source_frame != target_frame - delta:
            chain_cfg = replace(cfg, delta=target_frame - source_frame)
        outcome = _replay_detections(
            source_dets,
            source_frame,
            target_frame,
            chain_cfg,
            kf_states,
            tracker,
            frame_source,
            dets_by_frame,
        )
        stats["replay_steps"] += int(outcome["steps"])
        stats["replay_anchor_steps"] += int(outcome["anchor_calls"])
        stats["replay_polarity_dark"] = int(stats.get("replay_polarity_dark", 0)) + int(
            outcome.get("polarity_dark", 0)
        )
        stats["replay_polarity_chains"] = int(stats.get("replay_polarity_chains", 0)) + int(
            outcome.get("polarity_chains", 0)
        )
        for threshold in (3, 10, 30):
            key = f"replay_polarity_ratio_ge{threshold}"
            stats[key] = int(stats.get(key, 0)) + int(outcome.get(f"polarity_ratio_ge{threshold}", 0))
        for key in (
            "xcorr_ambiguous",
            "xcorr_uniq_n",
            "xcorr_rematch_applied",
            "xcorr_rematch_rejected",
            "xcorr_abstained",
            "xcorr_stride_hold",
        ):
            stats[f"replay_{key}"] = int(stats.get(f"replay_{key}", 0)) + int(outcome.get(key, 0))
        for key in ("xcorr_uniq_sum", "xcorr_uniq2_sum"):
            stats[f"replay_{key}"] = float(stats.get(f"replay_{key}", 0.0)) + float(
                outcome.get(key, 0.0)
            )
        stats["replay_held_steps"] += int(outcome["held"])
        stats["replay_step_rejected"] += int(outcome.get("step_rejected", 0))
        stats["replay_outlier_replaced"] += int(outcome.get("outlier_replaced", 0))
        stats["replay_consistency_rejected"] = int(
            stats.get("replay_consistency_rejected", 0)
        ) + int(outcome.get("consistency_rejected", 0))
        stats["replay_max_step_disp"] = max(
            float(stats["replay_max_step_disp"]), float(outcome["max_step_disp"])
        )
        # 回退链的峰值置信门：模板来自更早的帧，链更长 ⇒ 目标可能已经飞走，
        # 配准会锁到背景结构上（"幽灵坐标"）。峰值不达标 ⇒ 该帧不上报。
        if rolled_back and float(cfg.replay_rollback_min_peak) > 0.0:
            peak = float(outcome.get("xcorr_peak_last", 0.0))
            if peak < float(cfg.replay_rollback_min_peak):
                stats["replay_rollback_abstained"] += 1
                continue
        if outcome["failed"]:
            stats["replay_failed"] += 1
            reason = str(outcome["reason"])
            if reason.startswith("step_jump"):
                stats["replay_step_jump"] += 1
            if fallback == "abstain":
                # 兜底口径："abstain ⇒ 该帧不输出"（宁可让转台知道没坐标，也不给半真半假的）。
                stats["replay_abstained"] += 1
                continue
            stats["fallback_used"] += 1
            if fallback == "stale":
                report[target_frame] = [dict(det) for det in source_dets]
            elif fallback in ("hold", "holdv", "holdc"):
                # **尽力而为**兜底：用重推途中最后一个有效位置（而不是从头作废）。
                best = outcome.get("best_effort")
                if best is None:
                    report[target_frame] = [dict(det) for det in source_dets]
                elif fallback == "holdc":
                    # **连续性兜底**：不从"重推链的失败点"重新外推，而是从**上一帧的
                    # 实际上报位置**沿参考速度推进一步。
                    # 机理：holdv 用"失败点 + 剩余步数 × 步长"，剩余步数最多 8、单步
                    # 位移实测可达 22 px ⇒ 单帧就能跳出 ~176 px 的幻影位移，转台必然
                    # 抽搐（001324 实测 max_jitter 366 px）。改成"上一帧上报 + v"后，
                    # 单帧跳变被钉在 |v|（几 px）量级。
                    held_entries = []
                    for det in source_dets:
                        tid = int(det.get("track_id", -1))
                        base = last_reported_pos.get(tid)
                        if base is None:
                            position = np.asarray(best["position"], dtype=np.float64)
                        else:
                            velocity, _ = _estimate_velocity(
                                dets_by_frame,
                                source_frame,
                                tid,
                                source=cfg.velocity_source,
                                window=int(cfg.velocity_window),
                                kf_states=kf_states,
                            )
                            position = base + velocity
                        new_det = dict(det)
                        new_det["pos"] = (float(position[0]), float(position[1]))
                        held_entries.append(new_det)
                    report[target_frame] = held_entries
                else:
                    position = np.asarray(best["position"], dtype=np.float64)
                    if fallback == "holdv":
                        # 再按最后实测的单步位移补上剩余步数（短程外推，位移是**实测**的）。
                        position = position + int(best["remaining_steps"]) * np.asarray(
                            best["step_vector"], dtype=np.float64
                        )
                    held_entries = []
                    for det in source_dets:
                        new_det = dict(det)
                        new_det["pos"] = (float(position[0]), float(position[1]))
                        held_entries.append(new_det)
                    report[target_frame] = held_entries
            else:
                report[target_frame] = _kinematic_detections(
                    dets_by_frame, source_dets, source_frame, chain_cfg, kf_states, stats
                )
            _remember(report[target_frame])
            continue
        report[target_frame] = outcome["entries"]
        _remember(report[target_frame])

    _finalize_stats(report, stats, num_frames)
    # 限幅**只作用于 replay 臂**：stale / kinematic 是对照基线（老逻辑硬吃滞后、
    # 纯前向外推），给它们加限幅等于粉饰基线，会低估外挂的真实收益。
    if mode == "replay":
        _apply_output_slew_limit(report, cfg, stats)
    return report, stats


def _apply_output_slew_limit(
    report: list[list[dict]],
    cfg: LookaheadConfig,
    stats: dict,
) -> None:
    """输出级限幅（伺服 slew limiter）：把每帧的输出位移钉在 ``output_max_step_px`` 以内。

    机理：转台看到的是**帧间位移**。重推链的 ROI 偶尔会抓住旁边的杂波 ⇒ 某帧跳
    190 px、下一帧又回来了 ⇒ **孤立单帧飞跳** ⇒ 二阶差分 331 px ⇒ 伺服抽搐
    （实测 014053 frame 427）。本限幅**不看运动学参考**、只约束输出轨迹本身，
    因此不会像「一致性门」那样把好帧误杀成运动学预测（实测一致性门在 000703 上
    F1 98.0% → 79.4%，在 014053 上 95.1% → 68.8%）。

    规则：

    * 参考位置 = 同一 track **最近一次**的实际上报位置（中间即使缺帧也保留参考：
      伺服在目标丢失期间是"保持"而不是"瞬移"，重新捕获时也应按速率限幅追过去，
      否则跨缺口的重新捕获会原样放行，直接把二阶差分顶到 100 px 以上）；
    * 本帧输出 = 参考位置 + clip(原始输出 − 参考位置, max_step_px)；
    * 代价是真实快速机动也会被限住 ⇒ 引入滞后（实测 8 序列 F1 仅 −0.10 pp）。

    ``output_max_step_px = 0``（默认）= 不限幅，既有行为完全可复现。
    """
    limit = float(cfg.output_max_step_px)
    if limit <= 0.0:
        return
    last_pos: dict[int, np.ndarray] = {}
    limited_frames = 0
    limited_steps = 0
    for index in range(len(report)):
        entries = report[index]
        if not entries:
            continue
        touched = False
        limited: list[dict] = []
        for det in entries:
            tid = int(det.get("track_id", -1))
            position = np.asarray(det["pos"], dtype=np.float64).reshape(2).copy()
            base = last_pos.get(tid)
            if base is not None:
                vector = position - base
                norm = float(np.linalg.norm(vector))
                if norm > limit:
                    position = base + vector * (limit / norm)
                    touched = True
                    limited_steps += 1
            new_det = dict(det)
            new_det["pos"] = (float(position[0]), float(position[1]))
            limited.append(new_det)
            last_pos[tid] = position
        report[index] = limited
        if touched:
            limited_frames += 1
    stats["output_step_limited"] = int(stats.get("output_step_limited", 0)) + limited_frames
    stats["output_step_clamped"] = int(stats.get("output_step_clamped", 0)) + limited_steps


def _finalize_stats(report: list[list[dict]], stats: dict, num_frames: int) -> None:
    """补齐 ``applied``/``abstained`` 这对互补计数。"""
    applied = sum(1 for dets in report if dets)
    stats["applied"] = int(applied)
    stats["abstained"] = int(num_frames - applied)


def servo_metrics(
    report_by_frame: list[list[dict]],
    truth: dict[int, np.ndarray],
    num_frames: int,
    *,
    hit_tol_px: float = 5.0,
    return_samples: bool = False,
) -> dict:
    """尺子 2 的物理指标（**当前帧口径**）。

    ``truth``：record 下标 → (N,4) 的 ``(cx, cy, w, h)`` 数组（原生像素）。
    返回字典至少含：
    ``frames_with_report``/``coverage``/``gt_frames``/``inbox_hits``/``hit_rate_inbox``/
    ``hit_rate_5px``/``cle_p50``/``cle_p90``/``cle_p95``/``cle_p99``/``cle_max``/``cle_mean``/
    ``max_jitter``/``p99_jitter``/``fragmentation``/``lost_frames``。

    口径定义（**照抄，不要自己发明**）：
    - ``inbox_hits``：某 record 下标上存在**至少一个**上报点落在某个 GT 框内
      （与业务域 ``inbox`` 判据一致：``|px-cx|<=w/2 and |py-cy|<=h/2``）。
    - ``hit_rate_inbox`` = ``inbox_hits / gt_frames``，``gt_frames`` = 有 GT 的 record 数。
    - ``hit_rate_5px``：存在某个上报点到**最近 GT 中心**的欧氏距离 ≤ ``hit_tol_px``。
    - CLE 系列：每个「有 GT 且有上报」的 record，取**离最近 GT 中心最近的那个上报点**，
      算它到该 GT 中心的欧氏距离；再统计 p50/p90/p95/p99/max/mean（无样本时给 0.0）。
    - ``max_jitter``/``p99_jitter``：同一 track_id 的上报序列的二阶差分
      ``| (p_t - p_{t-1}) - (p_{t-1} - p_{t-2}) |`` 的 max / p99（跨 track 汇总）。
    - ``fragmentation``：某 track_id 的上报序列里「有 → 无 → 有」的次数之和。
    - ``lost_frames``：有 GT 但没有任何上报的 record 数。

    补充口径（本模块的实现约定，契约未展开的部分）：
    * ``hit_rate_5px`` 的分母与 ``hit_rate_inbox`` 同口径（``gt_frames``）；
    * 分位数用 ``numpy.percentile`` 的线性插值；``coverage`` = 有上报的帧数 /
      ``num_frames``；抖动按 track 分组、组内按帧序跳过缺检帧后再取二阶差分。
    """
    num_frames = int(num_frames)
    frames_with_report = 0
    gt_frames = 0
    inbox_hits = 0
    hits_within_tol = 0
    lost_frames = 0
    cle_values: list[float] = []
    track_points: dict[int, dict[int, np.ndarray]] = {}

    for frame_index in range(num_frames):
        detections = report_by_frame[frame_index] if frame_index < len(report_by_frame) else []
        if detections:
            frames_with_report += 1

        boxes = truth.get(frame_index)
        box_array = None
        if boxes is not None:
            candidate = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            if candidate.size:
                box_array = candidate
        if box_array is None:
            continue
        gt_frames += 1
        if not detections:
            lost_frames += 1
            continue

        centres = box_array[:, :2]
        sizes = box_array[:, 2:4]
        points = np.asarray([_detection_position(det) for det in detections], dtype=np.float64)
        distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        nearest = float(distances.min())
        cle_values.append(nearest)
        if nearest <= float(hit_tol_px):
            hits_within_tol += 1
        inside_x = np.abs(points[:, None, 0] - centres[None, :, 0]) <= sizes[None, :, 0] / 2.0
        inside_y = np.abs(points[:, None, 1] - centres[None, :, 1]) <= sizes[None, :, 1] / 2.0
        if bool((inside_x & inside_y).any()):
            inbox_hits += 1

        for det in detections:
            track_id = int(det.get("track_id", -1))
            per_frame = track_points.setdefault(track_id, {})
            per_frame.setdefault(frame_index, _detection_position(det))

    jitter_values: list[float] = []
    fragmentation = 0
    for per_frame in track_points.values():
        ordered = [per_frame[index] for index in sorted(per_frame)]
        for index in range(2, len(ordered)):
            first = ordered[index] - ordered[index - 1]
            second = ordered[index - 1] - ordered[index - 2]
            jitter_values.append(float(np.linalg.norm(second - first)))
        present = [index in per_frame for index in range(num_frames)]
        runs = 0
        previous = False
        for flag in present:
            if flag and not previous:
                runs += 1
            previous = flag
        fragmentation += max(0, runs - 1)

    if cle_values:
        cle_array = np.asarray(cle_values, dtype=np.float64)
        cle_p50 = float(np.percentile(cle_array, 50))
        cle_p90 = float(np.percentile(cle_array, 90))
        cle_p95 = float(np.percentile(cle_array, 95))
        cle_p99 = float(np.percentile(cle_array, 99))
        cle_max = float(cle_array.max())
        cle_mean = float(cle_array.mean())
    else:
        cle_p50 = cle_p90 = cle_p95 = cle_p99 = cle_max = cle_mean = 0.0

    if jitter_values:
        jitter_array = np.asarray(jitter_values, dtype=np.float64)
        max_jitter = float(jitter_array.max())
        p99_jitter = float(np.percentile(jitter_array, 99))
    else:
        max_jitter = 0.0
        p99_jitter = 0.0

    result = {
        "frames_with_report": int(frames_with_report),
        "coverage": float(frames_with_report / num_frames) if num_frames else 0.0,
        "gt_frames": int(gt_frames),
        "inbox_hits": int(inbox_hits),
        "hit_rate_inbox": float(inbox_hits / gt_frames) if gt_frames else 0.0,
        "hit_rate_5px": float(hits_within_tol / gt_frames) if gt_frames else 0.0,
        "cle_p50": cle_p50,
        "cle_p90": cle_p90,
        "cle_p95": cle_p95,
        "cle_p99": cle_p99,
        "cle_max": cle_max,
        "cle_mean": cle_mean,
        "max_jitter": max_jitter,
        "p99_jitter": p99_jitter,
        "fragmentation": int(fragmentation),
        "lost_frames": int(lost_frames),
        "hits_5px": int(hits_within_tol),
        "cle_samples": int(len(cle_values)),
        "jitter_samples": int(len(jitter_values)),
        "tracks": int(len(track_points)),
        "hit_tol_px": float(hit_tol_px),
    }
    if return_samples:
        # 逐帧 CLE 样本，供「跨序列池化分位数」用。
        # CLE 是「本帧上报点到最近 GT 中心」的距离，没有跨帧差分，跨序列拼接是安全的；
        # 抖动不行（二阶差分会在序列边界造出假抖动），所以抖动仍取逐序列最差值。
        result["cle_values"] = cle_values
    return result
