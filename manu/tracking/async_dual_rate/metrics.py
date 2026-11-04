"""冻结口径适配层：把 ``ms_cache`` 回放出来的检测结果 / 标注转成评测可用的
分数与位置序列。

**关键设计：匹配口径不重写，直接 import 冻结实现。**

项目铁律要求「同源受控 A/B」与「口径可比」。若我自己重写一遍 GT 匹配，
一旦阈值/判据有一丝偏差，所有指标都不可与冻结基线并列。因此这里直接
``from manu.evaluation.eval_multiscale_cache_th import count_matches, metrics,
load_ground_truth, build_frames``——口径**按构造**与
``runs/.../track3_blind.json``（F1 0.897694）完全一致，而不是"照抄一遍"。

冻结口径（来自 ``track3_blind.json`` 的 ``config``）：

* ``match_mode = "inbox"``：预测中心落在 GT 框内即命中（``tol`` 在 inbox 模式下不参与）；
* ``count_matches``：**按分数降序**贪心，每个检测吸收**最近的、尚未被吸收的** GT，
  一个 GT 最多被一个检测吸收；
* ``fn`` 拆成 ``fn_absent``（附近根本没有检测）与 ``fn_compete``
  （有检测够格但被更高分的那个抢走）——后者是"合并半径过大"的信号。

本模块只做**适配**：把逐帧检测集合（``score_predictions``）与逐帧单一位置序列
（``score_positions``）折算成冻结口径下的分数，**不提供任何时序指标**。

规格三项指标（Step Jump / 归因 / 脱靶）实现于 ``servo_metrics.py``，本模块不提供。
"""

from __future__ import annotations

import numpy as np

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: F401  (口径必须同源)
    build_frames,
    count_matches,
    load_ground_truth,
    metrics as frozen_metrics,
)

__all__ = [
    "build_frames",
    "count_matches",
    "load_ground_truth",
    "frozen_metrics",
    "score_predictions",
    "score_positions",
]


def score_predictions(
    ground_truth: dict[int, np.ndarray],
    predictions: dict[int, tuple],
    tol: float = 8.0,
    frames_subset=None,
) -> dict:
    """给"逐帧检测集合"打分，判据与冻结实现逐字一致。

    Parameters
    ----------
    ground_truth:
        ``frame_index -> (N,4)`` 的 GT 框（cx, cy, w, h，像素）。
    predictions:
        ``frame_index -> (scores, points)``，分数降序（与冻结
        ``load_predictions`` 的输出格式一致）。
    frames_subset:
        只在这些帧上打分。**真值与预测同时受限**——否则没有预测的帧会被
        计成 FN，把"漏帧"和"漏检"混在一起。

    Returns
    -------
    dict
        含 tp/fp/fn/fn_absent/fn_compete/det/gt/recall/precision/f1/far_per_frame。
    """
    if frames_subset is not None:
        keep = {int(frame) for frame in frames_subset}
        ground_truth = {k: v for k, v in ground_truth.items() if int(k) in keep}
        predictions = {k: v for k, v in predictions.items() if int(k) in keep}
    frames = build_frames(ground_truth, predictions)
    counts = count_matches(frames, 0.0, tol, "inbox")
    return frozen_metrics(counts, len(frames))


def score_positions(
    ground_truth: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    tol: float = 8.0,
    frames_subset=None,
) -> dict:
    """给"每帧至多一个位置"的系统打分，判据与冻结实现逐字一致。

    Parameters
    ----------
    ground_truth:
        ``frame_index -> (N,4)`` 的 GT 框（cx, cy, w, h，像素）。
    positions:
        ``frame_index -> (2,)`` 的系统输出位置。
    frames_subset:
        只在这些帧上打分（真值与预测同时受限）。

    Returns
    -------
    dict
        含 tp/fp/fn/fn_absent/fn_compete/det/gt/recall/precision/f1/far_per_frame。
    """
    predictions = {
        frame_index: (
            np.ones(1, dtype=np.float32),  # 分数：只有 1 个检测时排序无意义
            np.asarray(xy, dtype=np.float32).reshape(1, 2),
        )
        for frame_index, xy in positions.items()
    }
    return score_predictions(ground_truth, predictions, tol=tol, frames_subset=frames_subset)
