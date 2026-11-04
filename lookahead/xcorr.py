"""模板匹配（xcorr）位移精修：让重推链的位移不再带「质心偏置的变化」。

机理
----
质心链的位移 = ``K(t) − K(f)``，其中 ``K`` 是快核的灰度质心。质心相对目标**几何
中心**有一个偏置 ``b``，它随目标姿态/亮度/拖尾**漂移** ⇒ 位移误差 = ``b(t) − b(f)``，
尾部很重（实测 8 序列：位移误差 p90 4.37 px / p95 8.13 px）。

改用**同一份模板**（源帧锚点处按底噪阈值二值化的权重图）在目标帧里配准：
两端共用同一个偏置 ⇒ 偏置**精确抵消**，只剩真实位移。实测同一批 24503 帧：
位移 p90 4.37→3.17 px、p95 8.13→6.34 px，上报位置 hit@5px 91.36%→94.02%。

本模块只做纯函数，不依赖外挂在仓库里的其它部件，便于单测。
"""
from __future__ import annotations

import numpy as np


def window(img: np.ndarray, cx: float, cy: float, size: int):
    """以 ``(cx, cy)`` 为中心取 ``size`` 边长的窗；越界时**整体平移**回画面内。

    平移（而不是裁剪）与快核 ``LocalPatchTracker._window`` 同一约定：裁剪会让质心
    偏置（分母与分子的坐标系不一致），平移只改覆盖范围、坐标仍是绝对的。
    """
    height, width = img.shape
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        return None, (0, 0)
    return img[y0:y1, x0:x1], (x0, y0)


def make_template(
    img: np.ndarray,
    xy: np.ndarray,
    tsize: int = 24,
    bg_size: int = 96,
    th_k: float = 2.5,
    core_frac: float = 0.0,
    polarity: int = 1,
):
    """源帧锚点处的模板：按底噪阈值二值化的权重图（背景被压掉，只留目标本体）。

    ``polarity`` = ``+1`` 目标比背景**亮**（门限 ``bg + th_k*sigma``，权重
    ``max(patch - 门限, 0)``）；``-1`` 目标比背景**暗**（门限 ``bg - th_k*sigma``，
    权重 ``max(门限 - patch, 0)``）。权重图恒为**非负**，"越大 = 越极端 = 越像
    目标本体"，所以 ``core_frac`` 的语义与极性无关。

    暗目标必须走 ``-1``：用亮假设时权重**恒为 0**（目标比背景暗 ⇒ 没有比
    ``bg + th_k*sigma`` 更亮的像素）⇒ 模板直接为空 ⇒ xcorr 档位静默失效。

    返回 ``(weights, origin)``：``origin`` 是模板**左上角在源帧的绝对坐标**。
    报位置时必须用它：模板是从 ``round(cx)-tsize//2`` 切出来的，它的**离散中心**
    对应源帧 ``round(cx)-0.5``，而 matchTemplate 的峰给出的是「模板左上角在目标帧
    的落点」。少了这一项修正，每步会系统性少半个像素 ⇒ 链式配准 8 步累计 -4 px。
    """
    patch, (ox, oy) = window(img, float(xy[0]), float(xy[1]), tsize)
    if patch is None:
        return None, None
    bg_size = max(int(bg_size), int(tsize))
    bg_patch, _ = window(img, float(xy[0]), float(xy[1]), bg_size)
    if bg_patch is None:
        return None, None
    stride = max(1, bg_size // 32)
    sample = bg_patch[::stride, ::stride].astype(np.float32)
    background = float(np.median(sample))
    mad = float(np.median(np.abs(sample - background)))
    sigma = max(1.4826 * mad, 1e-3)
    patch_f = patch.astype(np.float32)
    if int(polarity) < 0:
        weights = np.clip((background - float(th_k) * sigma) - patch_f, 0.0, None)
    else:
        weights = np.clip(patch_f - (background + float(th_k) * sigma), 0.0, None)
    # ---- 核心化：只保留最亮的那一小撮前景像素，把「运动拖尾」剔掉 ----
    # 机理：红外小目标是「本体 + 一条运动拖尾」。拖尾比本体暗、但比背景亮，所以
    # 单按底噪阈值二值化时，**整条拖尾都会留在模板里**。目标越快拖尾越长，而拖尾
    # 在源帧/目标帧的**长度不同**（加速时变长）⇒ 相关峰被拖尾拽偏 ⇒ 实测表现为
    # 「位移方向与真实运动反向」（001324 上 4.3% 的帧偏 30 px）。
    # 只保留权值最高的那一小撮前景像素 ⇒ 模板只剩本体，拖尾被剔除。
    core_frac = float(core_frac)
    if core_frac > 0.0:
        positive = weights[weights > 0.0]
        if positive.size:
            cutoff = float(np.quantile(positive, max(0.0, 1.0 - min(1.0, core_frac))))
            if cutoff > 0.0:
                weights = np.where(weights >= cutoff, weights, 0.0)
    if float(weights.sum()) <= 1e-6:
        return None, None
    return weights, (float(ox), float(oy))


def match(
    dst_img: np.ndarray,
    template: np.ndarray,
    anchor: np.ndarray,
    center: np.ndarray,
    radius: int = 48,
    tsize: int = 24,
    origin=None,
    polarity: int = 1,
    diagnostics: dict | None = None,
):
    """在目标帧里把模板配准过去，返回 ``(位移, 峰值)``；失败返回 ``(None, 0.0)``。

    ``origin`` = 模板左上角在**源帧**的绝对坐标（来自 :func:`make_template`）。
    上报位置 = ``anchor + 模板左上角的位移``——纯位移口径，与切窗的取整约定无关。

    ``polarity`` = ``-1``（暗目标）时模板取**反**再配准。机理：归一化相关
    ``TM_CCOEFF_NORMED`` 在真匹配处的取值符号取决于目标的极性——暗目标的权重图
    在目标处为正、目标帧灰度在那里为低 ⇒ 零均值化后两者**反号** ⇒ 真匹配处的
    相关值是**负的**。若仍取极大，就会锁到画面里最亮的云边（实测 014453 上
    replay 偏到 70~100 px 外）。把模板整体取反，真匹配处的相关值即变为正极大。

    ``diagnostics`` 非 None 时，额外填入**唯一性比** ``uniqueness`` =
    「排除与最佳峰重叠的邻域后的次峰」÷「最佳峰」。机理：walk-off 帧里配准峰锁到了
    背景结构（云边 / 拖尾），那个位置往往**不是**全图唯一的峰 ⇒ 次峰接近最佳峰 ⇒
    比值大；好帧里最佳峰压倒性独大 ⇒ 比值≈0。实测（24 序列 8 条、51205 帧）：
    TP 帧唯一性比 p50 = 0.000，FP 帧 p50 = 0.593 ⇒ 这是目前唯一能把 walk-off 帧与
    好帧分开的**无状态**信号。
    """
    import cv2  # 局部导入：关掉 xcorr 档位时不需要 OpenCV

    if int(polarity) < 0:
        template = -template
    size = 2 * int(radius) + int(tsize)
    search, (ox, oy) = window(dst_img, float(center[0]), float(center[1]), size)
    if search is None:
        return None, 0.0
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(search.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_loc = cv2.minMaxLoc(res)
    px, py = max_loc
    dx = dy = 0.0
    if 0 < px < res.shape[1] - 1:
        a = float(res[py, px - 1])
        b = float(res[py, px])
        c = float(res[py, px + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = 0.5 * (a - c) / den
    if 0 < py < res.shape[0] - 1:
        a = float(res[py - 1, px])
        b = float(res[py, px])
        c = float(res[py + 1, px])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = 0.5 * (a - c) / den
    if diagnostics is not None:
        mask = np.ones_like(res, dtype=bool)
        half = max(1, int(tsize))
        mask[
            max(0, py - half) : min(res.shape[0], py + half + 1),
            max(0, px - half) : min(res.shape[1], px + half + 1),
        ] = False
        second = float(res[mask].max()) if mask.any() else 0.0
        diagnostics["peak"] = float(max_value)
        diagnostics["second"] = second
        diagnostics["uniqueness"] = second / float(max_value) if max_value > 1e-6 else float("nan")
    if origin is None:
        origin = (
            float(np.round(float(anchor[0]))) - int(tsize) // 2,
            float(np.round(float(anchor[1]))) - int(tsize) // 2,
        )
    matched_x = float(anchor[0]) + (ox + px + dx - origin[0])
    matched_y = float(anchor[1]) + (oy + py + dy - origin[1])
    displacement = np.asarray(
        [matched_x - float(anchor[0]), matched_y - float(anchor[1])], dtype=np.float64
    )
    return displacement, float(max_value)


def chain_step(
    dst_img: np.ndarray,
    src_img: np.ndarray,
    prev_pos: np.ndarray,
    search_center: np.ndarray,
    radius: int = 12,
    tsize: int = 24,
    bg_size: int = 96,
    th_k: float = 2.5,
    min_peak: float = 0.25,
    core_frac: float = 0.0,
    polarity: int = 1,
):
    """逐步配准的一步：用「上一帧 prev_pos 处的模板」在当前帧里配准。

    与 :func:`match` 的一次跨多帧大匹配的本质差别：

    * 大匹配的两端相隔 Delta 帧 => 目标在这 Delta 帧里换了姿态/亮度 => 同一个模板
      在目标帧里不再精确对应 => 残差（外观漂移误差）；
    * 本函数两端只隔 1 帧 => 模板与目标几乎没变 => 残差只剩亚像素定位噪声；
      而且被跟踪的物理点在链上保持不变 => 质心偏置不再逐帧累积。

    返回 ``(新位置, 峰值)``；不可靠时返回 ``(None, 0.0)``。
    """
    import cv2  # 局部导入

    template, origin = make_template(
        src_img,
        prev_pos,
        tsize=tsize,
        bg_size=bg_size,
        th_k=th_k,
        core_frac=core_frac,
        polarity=polarity,
    )
    if template is None:
        return None, 0.0
    displacement, peak = match(
        dst_img,
        template,
        prev_pos,
        search_center,
        radius=int(radius),
        tsize=int(tsize),
        origin=origin,
        polarity=polarity,
    )
    if displacement is None:
        return None, 0.0
    if peak < float(min_peak):
        return None, float(peak)
    matched = np.asarray(prev_pos, dtype=np.float64).reshape(2) + displacement
    limit = float(radius) + float(tsize)
    if float(np.linalg.norm(matched - np.asarray(search_center, dtype=np.float64).reshape(2))) > limit:
        return None, float(peak)
    return matched, float(peak)
