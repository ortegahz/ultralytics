#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3 维红外物理退化仿真（v2）。

为什么必须做「物理退化」而不是只用现成坏片
--------------------------------------------
坏片样本只有 4 条 ⇒ 无法把「画质掉到什么程度」和「跟踪跟不住」建立因果关系。
这里的做法是：**对清晰的真实红外窗口逐档施加单轴物理退化**，得到一条
「退化后的实测画质 X → 5 帧跟踪跟住率 Y」的衰减曲线，再从曲线上反解
**物理崩溃拐点**（膝点）。整个过程在**内存里**完成，不写任何临时视频文件。

三个退化族（任务书口径，**JPEG 族已剔除**）
---------------------------------------------
===== ===================== ==================================================
族    物理病因              档位
===== ===================== ==================================================
blur  散焦 / 过度降噪涂抹   GaussianBlur sigma ∈ {0.6, 1.0, 1.5, 2.2, 3.0, 4.0}（6 档）
motion 俯仰抖动 / 机动拖影   线性核 filter2D，核长 L ∈ {3, 5, 7, 9, 11}（5 档）
contrast 热交叉 / 未做 NUC   动态范围向全帧均值线性收缩 alpha ∈ {0.80, 0.65, 0.50, 0.35, 0.25, 0.15}（6 档）
===== ===================== ==================================================

上面这张表是**基线轮（baseline）**的档位，已随交付产物冻结。深轮（deep）只做一件事：
**把退化上限抬上去**（报告 §12.13 的处方）——基线轮在净化后的「好片段」池上，跟住率
只从 ~98.7% 掉到 ~91% ⇒ 衰减曲线没有真正坍塌、膝点只是「弯曲点」而不是崩溃点，
3 条核心约束线里有 2 条作废。深轮把上限抬到 sigma 8 / 核长 15 / alpha 0.05。

🔴 **深轮档位是基线档位的严格超集**（低档位完全共用）⇒ 两轮的衰减曲线在同一批
低档位上逐点可比；抬高上限**旨在**让曲线真正塌下来（是否真的塌下来，由规格书 §二 A1
判据**实测**判定，**不**在此处预断结论），不是换了一条轴。

🔴 **为什么没有 JPEG 族**（任务书 §四.2）：v1 与 v2 实测均已证实，JPEG 压缩在红外上会
**平滑微弱热噪声**，导致 ``surround_blockiness`` **不升反降（-16.8%）**，违背单调性假设；
且 DCT 振铃会让 ``target_lap_var`` 不降反升 ⇒ JPEG 轴上**不存在可用的膝点**。
⇒ 重标只保留**散焦模糊 / 运动拖影 / 对比度衰减**三项真实红外退化。

🔴 **为什么没有加性白噪声轴**：任务书同款实测结论——红外图像加性噪声指标与退化方向**相反**
（噪声越大反而越"好"），同样违背单调性假设 ⇒ **本流水线从未引入、也不引入**该轴。

运动模糊的角度按**固定种子**逐窗口随机生成（``theta ∈ [0, 180°)``），
且**窗口内 5 帧方向一致**（模拟一次连续机动，而不是逐帧乱抖）。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

BLUR_SIGMAS = (0.6, 1.0, 1.5, 2.2, 3.0, 4.0)
MOTION_LENGTHS = (3, 5, 7, 9, 11)
CONTRAST_ALPHAS = (0.80, 0.65, 0.50, 0.35, 0.25, 0.15)

#: **深轮（deep）档位**：在基线档位之上**继续抬高退化上限**（报告 §12.13 的处方）。
#: 基线轮在净化后的「好片段」池上，跟住率只从 ~98.7% 掉到 ~91% ⇒ 衰减曲线不坍塌、
#: 膝点只是「弯曲点」，2/3 条核心约束线作废 ⇒ 必须把轴的**上限**抬上去，
#: 而不是放松守卫门限（放松门限 = 拿噪声当拐点）。
DEEP_BLUR_SIGMAS = (0.6, 1.0, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
DEEP_MOTION_LENGTHS = (3, 5, 7, 9, 11, 13, 15)
DEEP_CONTRAST_ALPHAS = (0.80, 0.65, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05)

#: 档位剖面 → (blur sigma 档, motion 核长档, contrast alpha 档)
#: （``baseline`` 是已交付产物的冻结口径，**不得改动**）
CONDITION_PROFILES: dict[str, tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...]]] = {
    "baseline": (BLUR_SIGMAS, MOTION_LENGTHS, CONTRAST_ALPHAS),
    "deep": (DEEP_BLUR_SIGMAS, DEEP_MOTION_LENGTHS, DEEP_CONTRAST_ALPHAS),
}

#: 仅三族真实红外退化（**JPEG 族已按任务书 §四.2 剔除**）
FAMILIES = ("blur", "motion", "contrast")


@dataclass(frozen=True)
class Condition:
    """一个退化条件（``family == "clean"`` 表示不退化）。"""

    family: str
    level: float
    label: str

    @property
    def is_clean(self) -> bool:
        return self.family == "clean"


def _check_levels(family: str, levels) -> tuple[float, ...]:
    """校验单族档位：非空、无重复、单调方向正确、取值范围合法。"""
    values = tuple(float(value) for value in levels)
    if not values:
        raise ValueError("退化族 %s 的档位集合为空" % family)
    if len(set(values)) != len(values):
        raise ValueError("退化族 %s 的档位存在重复：%s" % (family, values))
    if family == "contrast":
        if any(not (0.0 < value <= 1.0) for value in values):
            raise ValueError("退化族 %s 的 alpha 必须落在 (0, 1]：%s" % (family, values))
        if any(b >= a for a, b in zip(values, values[1:])):
            raise ValueError("退化族 %s 的 alpha 必须严格递减：%s" % (family, values))
        return values
    if any(value <= 0.0 for value in values):
        raise ValueError("退化族 %s 的档位必须 > 0：%s" % (family, values))
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("退化族 %s 的档位必须严格递增：%s" % (family, values))
    return values


def build_conditions(profile: str = "baseline", *, blur_sigmas=None, motion_lengths=None,
                      contrast_alphas=None) -> list[Condition]:
    """全部退化条件（clean 放在最前）。

    ``profile`` 选整套档位（见 ``CONDITION_PROFILES``）；三个关键字参数可**逐族覆盖**
    （``None`` = 沿用 profile 自带档位）。深轮用 ``profile="deep"`` 抬高退化上限，
    两轮共用同一批低档位 ⇒ 衰减曲线可比。
    """
    if profile not in CONDITION_PROFILES:
        raise ValueError("未知档位剖面：%s（可选：%s）"
                         % (profile, "、".join(CONDITION_PROFILES)))
    base_blur, base_motion, base_contrast = CONDITION_PROFILES[profile]
    blur_values = _check_levels("blur", base_blur if blur_sigmas is None else blur_sigmas)
    motion_values = _check_levels("motion", base_motion if motion_lengths is None else motion_lengths)
    contrast_values = _check_levels("contrast",
                                    base_contrast if contrast_alphas is None else contrast_alphas)

    conditions = [Condition("clean", 0.0, "clean")]
    conditions += [Condition("blur", s, "blur_sigma%.1f" % s) for s in blur_values]
    conditions += [Condition("motion", float(L), "motion_len%d" % int(round(L)))
                   for L in motion_values]
    conditions += [Condition("contrast", a, "contrast_a%.2f" % a) for a in contrast_values]
    return conditions


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """线性运动模糊核（长度 ``length`` px、方向 ``angle_deg``，行归一化）。"""
    length = max(3, int(length) | 1)          # 强制奇数
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    theta = math.radians(float(angle_deg))
    dx, dy = math.cos(theta), math.sin(theta)
    for index in range(length):
        offset = index - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = float(kernel.sum())
    if total <= 0.0:
        kernel[int(center), int(center)] = 1.0
        total = 1.0
    return kernel / total


def motion_angle_for(window_key: str, seed: int = 42) -> float:
    """逐窗口**确定性**的运动角度（``theta ∈ [0, 180)``），窗口内 5 帧共用。"""
    import hashlib

    digest = hashlib.md5(("%d|%s" % (int(seed), str(window_key))).encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / float(1 << 32)
    return float(value * 180.0)


def apply_degradation(frame: np.ndarray, family: str, level: float,
                      angle_deg: float = 0.0) -> np.ndarray:
    """对**单帧**施加单轴退化（不修改输入）。"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if family == "clean":
        return gray.copy()
    if family == "blur":
        if float(level) <= 0.0:
            return gray.copy()          # sigma=0 不是合法的 GaussianBlur 参数 ⇒ 视为不退化
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=float(level), sigmaY=float(level))
    if family == "motion":
        kernel = motion_kernel(int(round(float(level))), angle_deg)
        return cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    if family == "contrast":
        alpha = float(level)
        mu = float(gray.mean())
        out = alpha * gray.astype(np.float32) + (1.0 - alpha) * mu
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError("未知退化族：%s" % family)


def apply_window_degradation(frames: list[np.ndarray], condition: Condition,
                             angle_deg: float = 0.0) -> list[np.ndarray]:
    """对窗口内**所有帧**施加同一退化（运动模糊时 5 帧方向一致）。"""
    return [apply_degradation(frame, condition.family, condition.level, angle_deg)
            for frame in frames]


def degradation_strength(family: str, level: float) -> float:
    """把档位映射成「退化强度」标量（越大＝越糊/越灰/越压），供单调性测试使用。"""
    if family == "blur":
        return float(level)
    if family == "motion":
        return float(level)
    if family == "contrast":
        return 1.0 - float(level)          # alpha 越小＝退化越强
    raise ValueError("未知退化族：%s" % family)
