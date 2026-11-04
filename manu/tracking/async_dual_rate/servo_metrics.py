"""时序对齐的"伺服有效性"度量层。

**为什么不能用常规检测的单帧 mAP/F1 脚本**：常规检测评的是"第 t 帧的图
送进去，拿第 t 帧的框算 IoU"。而异步检跟分离下真正要回答的是：

    在物理时刻 t，伺服拿到的坐标 (x̂_t, ŷ_t) 与**第 t 帧**的物理真值
    差多少？此时后台检测器只能喂 **Det_{t−Δ}**（Δ 帧以前的旧结果）。

所以本模块的全部指标都按**物理时刻 t** 与 **GT_t** 对账，而不是拿慢流
结果去算 IoU。四项核心指标（对应《异步双频检跟分离系统》文档 §四 与
验收契约）：

1. **CLE**（中心位置误差）：``‖(x̂_t, ŷ_t) − (x*_t, y*_t)``，
   统计 p50 / p95 / p99 / max。合格线：p50 ≤ 1.0 px，p95 ≤ 3.0 px。
2. **Hit Rate @T**：``CLE_t ≤ T`` 的帧数 / 目标存在的总帧数（T 默认 5 px）。
   同时给出冻结口径的 **Inbox 命中率**（点落在 GT 框内），以便与纯检测
   的 Recall 同口径对话。
3. **脱靶与断裂**：脱靶帧数、断裂次数（锁定→脱靶→被滞后检测器拉回）、
   自持周期（最长连续锁定帧数）。
4. **抖动**：两个互补口径——
   * **Step Jump**（规格判据 1）：一阶差分 ``‖p_t − p_{t−1}‖``，阈值 **2 px**，
     并**归因到异步校正帧**（smooth / hard_reinit / replay）：回答"检测框异步到达
     校正那一帧，转台会不会被瞬移一下"；
   * **Max Jitter**（二阶差分）：``Δv_t = ‖(p_t − p_{t−1}) − (p_{t−1} − p_{t−2})‖``，
     回答"画面抽不抽"。**全局 max 会被状态机切换与故意重定位吃掉**，必须按
     来源分桶后看 ``steady`` 桶的稳健分位数。

比率一律给 **Wilson 置信区间**（项目铁律：比率要用 Wilson CI，不许裸
百分比）；抖动用 **median** 而不是 mean（同铁律）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ServoMetrics",
    "servo_metrics",
    "wilson_interval",
    "percentiles",
    "mcnemar_exact",
    "STEP_JUMP_THRESHOLD_PX",
]

# 规格《异步双频检跟分离系统》§四 判据 1 的阈值：
# "观察每帧输出坐标的一阶差分（速度矢量），在检测框异步到达校正时，
#  是否会出现大于 2 像素的瞬态跳跃"。
STEP_JUMP_THRESHOLD_PX = 2.0


ATTRIBUTION_BUCKETS = (
    "transition",
    "replay",
    "replay_abstain",
    "hard_reinit",
    "smooth",
    "steady",
)


_CAUSE_PRIORITY = ("transition", "replay", "replay_abstain", "hard_reinit", "smooth")


def _dominant_cause(recent_causes: list[str]) -> str:
    """一组帧的归因里取"最有解释力"的那一个（优先级见 ``_CAUSE_PRIORITY``）。

    全部是 steady 才返回 steady。用于二阶差分：它跨 3 帧，只要其中任一帧
    发生了校正/切换，这次加速度就不该算到"稳态抖动"头上。
    """
    for cause in _CAUSE_PRIORITY:
        if cause in recent_causes:
            return cause
    return "steady"


def _attribution(events: str, fusion_kind: str) -> str:
    """把一帧归到"抖动来源"桶里。

    桶的含义（**必须互斥且穷尽**，否则会把"故意的大跳变"混进稳态统计里）：

    * ``transition``      —— 状态机切换帧（冷启动 / 重捕）；
    * ``replay``         —— 臂 3「退回重推演」生效帧（**故意**的重定位）；
    * ``replay_abstain`` —— 重推演被弃权（拿不到历史 / 目标已丢），未施加修正；
    * ``hard_reinit``    —— 硬拉回帧（**故意**的跳变）；
    * ``smooth``         —— 残差平滑修正帧（α·e 注入，**这正是平滑该压住的**）；
    * ``steady``         —— 无任何校正、无状态切换的纯前向跟踪帧。

    ⚠️ 漏掉 ``replay`` 会把臂 3 最大的那批有意跳变算进 ``steady``，
    污染"稳态抖动"这个**平滑滤波该负责**的统计量。
    """
    if "->" in events:
        return "transition"
    if fusion_kind == "replay":
        return "replay"
    if fusion_kind == "replay_abstain":
        return "replay_abstain"
    if fusion_kind == "hard_reinit":
        return "hard_reinit"
    if fusion_kind == "smooth":
        return "smooth"
    return "steady"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间（小样本/极端比例下比正态近似稳）。"""
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _log_binomial_tail(smaller: int, trials: int) -> float:
    """``P(X <= smaller)``，其中 ``X ~ Binomial(trials, 0.5)``。

    用对数空间（lgamma + logsumexp）求和，避免 ``2 ** trials`` 在 trials>1023 时
    溢出成 OverflowError。极小概率下 exp 会下溢成 0.0——那正是"极显著"的
    正确表示，不是错误。
    """
    if smaller < 0:
        return 0.0
    if smaller >= trials:
        return 1.0
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        - trials * math.log(2.0)
        for k in range(smaller + 1)
    ]
    peak = max(logs)
    total = peak + math.log(sum(math.exp(value - peak) for value in logs))
    return math.exp(total)


def mcnemar_exact(b: int, c: int) -> dict:
    """配对二分类比较的 **McNemar 精确检验**（双侧）。

    为什么必须有它：三条臂跑在**同一批帧、同一份 GT** 上，每帧的"命中/未命中"
    是**配对二元数据**。用"两条独立样本的 Wilson CI 是否重叠"来判断差异，
    是**用错了检验**（而且 Wilson CI 的独立性假设在这里也不成立——相邻帧的
    命中事件强自相关，CI 偏窄）。

    Parameters
    ----------
    b:
        A 命中、B 未命中的帧数。
    c:
        B 命中、A 未命中的帧数。

    Returns
    -------
    dict
        ``discordant``（不一致帧数）、``z``（符号检验统计量）、``p_value``
        （精确二项双侧 p 值）、``significant_5pct``。
    """
    discordant = int(b) + int(c)
    if discordant == 0:
        return {
            "b": int(b),
            "c": int(c),
            "discordant": 0,
            "z": 0.0,
            "p_value": 1.0,
            "significant_5pct": False,
            "note": "两臂在所有配对帧上一致，无可检验的分歧。",
        }
    # 精确二项：H0 下 b ~ Binomial(discordant, 0.5)。
    # ⚠️ 不能写成 sum(comb(n,k)) / 2**n —— discordant 上千时 2**n 直接溢出
    # （OverflowError: Numerical result out of range）。改用**对数空间**求和。
    smaller = min(int(b), int(c))
    tail = _log_binomial_tail(smaller, discordant)
    p_value = min(1.0, 2.0 * tail)
    z = (int(b) - int(c)) / math.sqrt(discordant)
    return {
        "b": int(b),
        "c": int(c),
        "discordant": discordant,
        "z": float(z),
        "p_value": float(p_value),
        "significant_5pct": bool(p_value < 0.05),
        "note": (
            "McNemar 精确二项检验（双侧）；z 为符号检验统计量 "
            "(b−c)/sqrt(b+c)。配对帧集 = 两臂都有输出且目标存在的帧。"
        ),
    }


def percentiles(values, points=(50, 95, 99)) -> dict:
    """中位数/分位数统计（median 优先，不给裸均值当结论）。

    ⚠️ ``mean`` 仅供**诊断**（看分布是否被长尾拉偏），**不作为对外结论**——
    铁律：抖动/对比度统计用中位数，不用均值。
    """
    array = np.asarray(list(values), dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return {"n": 0, **{f"p{point}": float("nan") for point in points}, "max": float("nan")}
    out = {"n": int(array.size)}
    for point in points:
        out[f"p{point}"] = float(np.percentile(array, point))
    out["max"] = float(array.max())
    out["mean"] = float(array.mean())
    return out


@dataclass
class ServoMetrics:
    """一个臂的时序对齐度量（全部按物理时刻 t 与 GT_t 对账）。"""

    cle: dict
    hit_rate: dict
    inbox_hit_rate: dict
    jitter: dict
    fragmentation: dict
    states: dict
    fusion: dict
    detector_silence: dict

    def to_dict(self) -> dict:
        return {
            "cle": self.cle,
            "hit_rate": self.hit_rate,
            "inbox_hit_rate": self.inbox_hit_rate,
            "jitter": self.jitter,
            "fragmentation": self.fragmentation,
            "states": self.states,
            "fusion": self.fusion,
            "detector_silence": self.detector_silence,
        }


def _gt_centre(boxes) -> np.ndarray:
    """GT 框中心（多框时取第一个——本序列每帧只有一个框）。"""
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[0, :2]


def servo_metrics(
    records: list[dict],
    ground_truth: dict[int, np.ndarray],
    tolerance: float = 5.0,
    lock_radius: float = 16.0,
) -> ServoMetrics:
    """按物理时刻 t 计算时序对齐度量。

    Parameters
    ----------
    records:
        逐帧证据记录（``run_lockstep`` 产出），必须含 ``frame_index`` /
        ``x`` / ``y`` / ``state`` / ``fusion_kind`` / ``det_source``。
    ground_truth:
        ``frame_index -> (N,4)`` 的 GT 框（cx, cy, w, h，像素）。
    tolerance:
        Hit Rate 的距离门限 T（默认 5 px）。
    lock_radius:
        "锁定"的距离门限（默认 16 px = 32×32 窗口半宽），用于脱靶/自持周期。
    """
    positions: dict[int, np.ndarray] = {}
    for record in records:
        x = record.get("x", float("nan"))
        y = record.get("y", float("nan"))
        if not (math.isnan(x) or math.isnan(y)):
            positions[int(record["frame_index"])] = np.asarray([x, y], dtype=np.float64)

    ordered_frames = sorted(positions)

    # ---- 1. CLE / 命中率（只在目标存在的帧上评）----
    cle_values: list[float] = []
    hits = 0
    inbox_hits = 0
    evaluated = 0
    missing_output = 0
    for frame_index in sorted(ground_truth):
        evaluated += 1
        position = positions.get(int(frame_index))
        if position is None:
            missing_output += 1
            continue
        centre = _gt_centre(ground_truth[frame_index])
        distance = float(np.linalg.norm(position - centre))
        cle_values.append(distance)
        if distance <= tolerance:
            hits += 1
        box = np.asarray(ground_truth[frame_index], dtype=np.float64).reshape(-1, 4)[0]
        if (
            abs(position[0] - box[0]) <= box[2] / 2
            and abs(position[1] - box[1]) <= box[3] / 2
        ):
            inbox_hits += 1

    cle = percentiles(cle_values, points=(50, 95, 99))
    cle["frames_with_output"] = len(cle_values)
    cle["frames_target_present"] = evaluated
    cle["frames_no_output"] = missing_output

    low, high = wilson_interval(hits, evaluated)
    hit_rate = {
        "tolerance_px": tolerance,
        "hits": hits,
        "evaluated": evaluated,
        "rate": (hits / evaluated) if evaluated else 0.0,
        "wilson95": [low, high],
    }
    low, high = wilson_interval(inbox_hits, evaluated)
    inbox_hit_rate = {
        "hits": inbox_hits,
        "evaluated": evaluated,
        "rate": (inbox_hits / evaluated) if evaluated else 0.0,
        "wilson95": [low, high],
        "note": "冻结 inbox 判据：点落在 GT 框内",
    }

    # ---- 2. 抖动：一阶差分（位移）与二阶差分（加速度）----
    #
    # **必须拆分归因**：max(Δv) 这种全局最大值几乎总是被"状态机切换"
    # （SEARCH→TRACK 的冷启动跳变、LOST→TRACK 的重捕跳变）吃掉，跟残差融合
    # 没关系。不拆分就会得出"平滑滤波没用"的错误结论。拆分口径：
    #
    # * ``steady``：无状态切换、且不是硬拉回帧——这才是**平滑滤波**该负责的
    #   稳态抖动，也是验收契约"转台抽搐"真正关心的部分；
    # * ``hard_reinit``：硬拉回帧——**故意**的跳变（把目标拉回来），单独算；
    # * ``transition``：状态机切换帧——冷启动/重捕，本来就会跳，单独算。
    record_by_frame = {int(record["frame_index"]): record for record in records}
    steps: list[float] = []
    accels: list[float] = []
    accels_steady: list[float] = []
    jumps: dict[str, int] = {bucket: 0 for bucket in ATTRIBUTION_BUCKETS}
    jump_threshold = 5.0
    previous = None
    previous_previous = None
    recent_causes: list[str] = []
    for frame_index in ordered_frames:
        current = positions[frame_index]
        record = record_by_frame.get(frame_index, {})
        events = str(record.get("events", ""))
        kind = str(record.get("fusion_kind", "none"))
        cause = _attribution(events, kind)
        recent_causes.append(cause)
        if len(recent_causes) > 3:
            recent_causes.pop(0)
        if previous is not None:
            velocity = current - previous
            steps.append(float(np.linalg.norm(velocity)))
            if previous_previous is not None:
                acceleration = float(np.linalg.norm(velocity - (previous - previous_previous)))
                accels.append(acceleration)
                # ⚠️ 二阶差分跨 **3 帧**（t−2, t−1, t）：一次校正跳变的"反冲"
                # 会落到**下一帧**的加速度上。若只看当前帧的归因，这个反冲会被
                # 记成 "steady"，污染"平滑滤波该负责的稳态抖动"。
                # ⇒ 只要这 3 帧里**有任何一帧**是校正/切换，就不计入 steady。
                accel_cause = _dominant_cause(recent_causes)
                if accel_cause == "steady":
                    accels_steady.append(acceleration)
                if acceleration > jump_threshold:
                    jumps[accel_cause] = jumps.get(accel_cause, 0) + 1
            previous_previous = previous
        else:
            previous_previous = None
        previous = current

    # ---- 规格判据 1：一阶差分 Step Jump（阈值 2 px，定位到异步校正帧）----
    #
    # 规格原文："观察每帧输出坐标的一阶差分（速度矢量），**在检测框异步到达校正时**，
    # 是否会出现大于 2 像素的瞬态跳跃"。这与上面的二阶差分 MaxJitter **互补**：
    # MaxJitter 看"抽"（加速度），Step Jump 看"瞬移"（一阶位移），
    # 且判据要落在**异步校正帧**上——只有那几帧的跳变才是伺服真正会被打到的。
    #
    # 与二阶差分口径的两点区别（都是刻意为之）：
    #   1. **只统计相邻帧**：帧号不连续时 ‖p_t − p_{t−1}‖ 跨了多帧，不能当"每帧位移"；
    #   2. **归因粒度更细**：把 smooth（平滑修正帧）与 hard_reinit（硬拉回帧）分开——
    #      前者是残差滤波该压住的，后者是**故意**的跳变（把目标拉回视野）。
    step_by_cause: dict[str, list[float]] = {bucket: [] for bucket in ATTRIBUTION_BUCKETS}
    previous_frame_index: int | None = None
    for frame_index in ordered_frames:
        current = positions[frame_index]
        record = record_by_frame.get(frame_index, {})
        events = str(record.get("events", ""))
        kind = str(record.get("fusion_kind", "none"))
        step_cause = _attribution(events, kind)
        if previous is not None and previous_frame_index is not None:
            if int(frame_index) == previous_frame_index + 1:
                step_by_cause[step_cause].append(float(np.linalg.norm(current - previous)))
        previous = current
        previous_frame_index = int(frame_index)

    def _merge(*causes: str) -> list[float]:
        merged: list[float] = []
        for cause in causes:
            merged.extend(step_by_cause[cause])
        return merged

    correction_steps = _merge("smooth", "hard_reinit", "replay")
    over_threshold = {
        cause: int(sum(1 for value in values if value > STEP_JUMP_THRESHOLD_PX))
        for cause, values in step_by_cause.items()
    }
    over_threshold["correction_any"] = int(
        sum(1 for value in correction_steps if value > STEP_JUMP_THRESHOLD_PX)
    )
    over_threshold["total_frames"] = int(sum(len(v) for v in step_by_cause.values()))
    step_jump = {
        "threshold_px": STEP_JUMP_THRESHOLD_PX,
        "definition": "一阶差分 ‖p_t − p_{t−1}‖（像素/帧），仅统计**相邻帧**",
        "at_correction_smooth": percentiles(step_by_cause["smooth"], points=(50, 95, 99)),
        "at_correction_hard_reinit": percentiles(
            step_by_cause["hard_reinit"], points=(50, 95, 99)
        ),
        "at_correction_replay": percentiles(
            step_by_cause["replay"], points=(50, 95, 99)
        ),
        "at_correction_any": percentiles(correction_steps, points=(50, 95, 99)),
        "at_steady": percentiles(step_by_cause["steady"], points=(50, 95, 99)),
        "at_transition": percentiles(step_by_cause["transition"], points=(50, 95, 99)),
        "over_threshold_counts": over_threshold,
        "criterion_2px_pass_p95": bool(
            correction_steps
            and float(np.percentile(correction_steps, 95)) <= STEP_JUMP_THRESHOLD_PX
        ),
        "criterion_2px_pass_max": bool(
            correction_steps
            and float(np.max(correction_steps)) <= STEP_JUMP_THRESHOLD_PX
        ),
        "note": (
            "规格 §四 判据 1：检测框异步到达校正那一帧的一阶位移是否 > 2 px。"
            "hard_reinit / replay 的跳变是**故意**的（把目标拉回 32×32 视野），"
            "与 smooth 分开报；'校正帧' = smooth ∪ hard_reinit ∪ replay。"
        ),
    }

    jitter = {
        "first_difference": percentiles(steps, points=(50, 95, 99)),
        "acceleration": percentiles(accels, points=(50, 95, 99)),
        "acceleration_steady": percentiles(accels_steady, points=(50, 95, 99)),
        "jump_threshold_px": jump_threshold,
        "jump_attribution": jumps,
        "step_jump": step_jump,
        "note": (
            "二阶差分 = ‖(p_t−p_{t−1}) − (p_{t−1}−p_{t−2})‖（转台抽搐判据）。"
            "acceleration_steady 排除状态机切换与硬拉回帧——那两类跳变是"
            "设计内的，不是平滑滤波该负责的。"
        ),
    }

    # ---- 3. 脱靶 / 断裂 / 自持周期 ----
    locked_flags: list[bool] = []
    locked_frames: list[int] = []
    no_output_frames = 0
    for frame_index in sorted(ground_truth):
        position = positions.get(int(frame_index))
        if position is None:
            locked_flags.append(False)
            no_output_frames += 1
            continue
        centre = _gt_centre(ground_truth[frame_index])
        locked_flags.append(bool(np.linalg.norm(position - centre) <= lock_radius))
        locked_frames.append(int(frame_index))

    lock_streaks: list[int] = []
    lost_streaks: list[int] = []
    current_lock = 0
    current_lost = 0
    transitions = 0
    was_locked = None
    for flag in locked_flags:
        if flag:
            if current_lost:
                lost_streaks.append(current_lost)
                current_lost = 0
            current_lock += 1
        else:
            if current_lock:
                lock_streaks.append(current_lock)
                current_lock = 0
            current_lost += 1
        if was_locked is not None and was_locked and not flag:
            transitions += 1
        was_locked = flag
    if current_lock:
        lock_streaks.append(current_lock)
    if current_lost:
        lost_streaks.append(current_lost)

    fragmentation = {
        "lock_radius_px": lock_radius,
        "target_frames": len(locked_flags),
        "lost_frames": int(sum(1 for flag in locked_flags if not flag)),
        "lost_frame_rate": (
            (sum(1 for flag in locked_flags if not flag) / len(locked_flags))
            if locked_flags
            else 0.0
        ),
        "no_output_frames": no_output_frames,
        "no_output_rate": (no_output_frames / len(locked_flags)) if locked_flags else 0.0,
        "lock_streaks": len(lock_streaks),
        "lock_streak_max": max(lock_streaks) if lock_streaks else 0,
        "lock_streak_median": float(np.median(lock_streaks)) if lock_streaks else float("nan"),
        "lost_streaks": len(lost_streaks),
        "lost_streak_max": max(lost_streaks) if lost_streaks else 0,
        "locked_to_lost_transitions": transitions,
        "fragmentations": sum(1 for length in lost_streaks if length),
        "note": (
            "断裂次数 = 锁定→脱靶→（被滞后检测器拉回）的次数。"
            "脱靶率 lost_frame_rate = 目标存在但**未锁定**（无输出 或 输出离 GT > lock_radius）"
            "的帧占比；no_output_rate 是其中**完全没输出**的那部分。"
        ),
    }

    # ---- 4. 状态机 / 融合分布 ----
    states: dict[str, int] = {}
    fusion_kinds: dict[str, int] = {}
    for record in records:
        state = str(record.get("state", ""))
        states[state] = states.get(state, 0) + 1
        kind = str(record.get("fusion_kind", "none"))
        fusion_kinds[kind] = fusion_kinds.get(kind, 0) + 1
    hard_reinits = fusion_kinds.get("hard_reinit", 0)

    # ---- 5. 检测器静默段（自持周期的压力场景）----
    arrival_frames = sorted(
        {
            int(record["det_source"])
            for record in records
            if int(record.get("det_source", -1)) >= 0
        }
    )
    silence: list[int] = []
    for previous_frame, next_frame in zip(arrival_frames, arrival_frames[1:]):
        silence.append(int(next_frame - previous_frame))
    detector_silence = {
        "detector_source_frames": len(arrival_frames),
        "silence_gap_median": float(np.median(silence)) if silence else float("nan"),
        "silence_gap_max": int(max(silence)) if silence else 0,
        "hard_reinits": hard_reinits,
    }

    return ServoMetrics(
        cle=cle,
        hit_rate=hit_rate,
        inbox_hit_rate=inbox_hit_rate,
        jitter=jitter,
        fragmentation=fragmentation,
        states=states,
        fusion=fusion_kinds,
        detector_silence=detector_silence,
    )
