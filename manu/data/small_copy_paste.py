"""微小目标 Copy-Paste 增强（高性能模型方案 §5.2，2026-11-13）。

**这一版的两个关键设计（都是实测逼出来的）**
  ① **贴「带上下文的小块」，不是贴裸目标**
     只抠目标框本身会把一个**冷目标**贴到**亮云层**上 ⇒ 变成"看不见的幽灵正样本"
     ⇒ 模型学到「云里随便一块冷斑就是目标」⇒ **T2 虚警 / T3 反而变差**（与目标相反）。
     ⇒ 本实现连同**上下文边距**一起抠（默认 3px），整块羽化贴过去
     ⇒ 贴出来的是「像真目标的局部场景」，保持局部对比度自洽。
  ② **按背景亮度匹配落点**
     落点区域的背景亮度必须与源块上下文亮度同量级（默认 0.5×~2.0×）
     ⇒ 不把冷目标贴到亮云、不把亮目标贴到暗地 ⇒ 贴出来的样本**物理上说得通**。

**为什么官方 CopyPaste 不能用**
  官方实现（`ultralytics/data/augment.py:1890`）第一行就是
  `if len(labels["instances"].segments) == 0 or self.p == 0: return labels`
  ⇒ **检测任务（无分割多边形）下是空操作**；本仓库也显式 `cfg.copy_paste = 0.0`。

**坐标口径**
  插在 **mosaic + 仿射之后、Format 之前**：
  此时 `labels["img"]` = uint8 (H, W, 3) 640×640，
  `labels["instances"].bboxes` = (N, 4) **像素级 xyxy**（normalized=False），
  `labels["cls"]` = (N, 1) float32。

**为什么这件事值得做（实测，2026-11-13 18:20）**
  * UAV 训练标签里 nano 桶（≤4px / ≤16px²）只有 **0.03%**、micro 桶（≤8px / ≤64px²）**6.30%**。
  * 更要命的是**增强管线会系统性地删掉小目标**：`RandomPerspective.box_candidates`
    （`ultralytics/data/augment.py:1369`）要求变换后 **w2 > 2 且 h2 > 2（像素）**、
    且面积比 > 0.1；而 mosaic 会把每个子图缩到 ~0.5×
    ⇒ 实测：**UAV 150 个样本在增强后剩下的框里，max_side ≤ 8px 的占比是 0.000**
    （而标签里 micro 桶有 6.3%）⇒ **模型训练时几乎见不到真正的纳米级正样本**。
  * 本增强插在**过滤之后** ⇒ 贴进去的 ≤12px 目标**不会被删**
    ⇒ 它可能是模型整个训练期里**唯一的纳米级正样本来源**。

**默认关闭**：`prob=0` ⇒ 原样返回 ⇒ 与旧行为**逐位一致**。

**🟢🟢 降尺度粘贴（2026-11-13，为 NanoBench 的 `nano4` 桶补 ≤4px 正样本）**
  训练集里 9~12px 的源目标**不少**，但 4px 桶的正样本**几乎没有**
  ⇒ 只贴原尺寸目标对 `nano4`（4px 桶严格定位精度）没用。
  ⇒ 新增 `scale_min / scale_max`：把**抠出来的块整体缩小**到 `scale × 原尺寸`
  ⇒ 贴出来的是「同样场景里更远的小目标」⇒ 目标 9~12px → 4.5~6px（scale=0.5）
  ⇒ 局部对比度自洽（上下文与目标一起变小，没有残留环）。
  ⚠️ **缩放顺序**：**先在原分辨率把块抠出来，再对这块做面积平均降采样**
  ⇒ 绝不「先缩放原图大区域再抠」⇒ 块边界不引入插值污染。
  ⚠️ `scale_min == scale_max == 1.0`（默认）⇒ 缩放分支**整体跳过** ⇒ 与旧行为**逐位一致**。
"""
from __future__ import annotations

import math

import numpy as np

try:  # 官方 Instances（像素级 xyxy 容器）
    from ultralytics.utils.instance import Instances
except Exception:  # noqa: BLE001 —— 允许在没有 ultralytics 的环境里 import 本模块做单测
    Instances = None  # type: ignore[assignment]


class SmallObjectCopyPaste:
    """把图里**真实存在的微小目标**连同局部上下文抠出来，羽化后贴到同图的杂波背景位置。

    参数
    ----
    prob : float
        每个样本的触发概率（0 ⇒ 完全关闭 ⇒ 逐位等价旧行为）。
    max_src_px : float
        源目标的最大边长（像素，640 尺度）；大于该值的框不作为源。
    ctx_px : float
        抠图时在目标框外扩的**上下文边距**（像素）⇒ 贴出去的是"局部场景"而不是裸目标。
    feather_px : float
        贴图边缘羽化宽度（像素）：线性淡出，避免硬边。
    max_per_img : int
        每张图最多粘贴几个目标。
    min_gap_px : float
        粘贴位置与已有框的最小间距（像素），避免叠在别的目标上。
    match_bg : bool
        落点背景亮度是否要与源上下文同量级（默认开 ⇒ 贴出来的样本物理上说得通）。
    bg_lo / bg_hi : float
        背景亮度允许的相对区间（对源上下文中位数的倍数）。
    p_flip : float
        贴图左右翻转的概率（增加姿态多样性）。
    scale_min / scale_max : float
        **降尺度粘贴**：把源目标（连同上下文）整体缩小到 `scale × 原尺寸`，
        `scale` 在 [scale_min, scale_max] 里逐目标随机取。
        两个都 = 1.0（默认）⇒ 完全不缩放 ⇒ 与旧行为**逐位一致**；
        缩放后目标 < 2px ⇒ 该源目标被跳过（贴不出有意义的目标）。
    """

    def __init__(
        self,
        prob: float = 0.0,
        max_src_px: float = 12.0,
        ctx_px: float = 3.0,
        feather_px: float = 2.0,
        max_per_img: int = 3,
        min_gap_px: float = 2.0,
        match_bg: bool = True,
        bg_lo: float = 0.5,
        bg_hi: float = 2.0,
        p_flip: float = 0.5,
        max_tries: int = 12,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
    ) -> None:
        self.prob = float(prob)
        self.max_src_px = float(max_src_px)
        self.ctx_px = float(ctx_px)
        self.feather_px = float(feather_px)
        self.max_per_img = int(max_per_img)
        self.min_gap_px = float(min_gap_px)
        self.match_bg = bool(match_bg)
        self.bg_lo = float(bg_lo)
        self.bg_hi = float(bg_hi)
        self.p_flip = float(p_flip)
        self.max_tries = int(max_tries)
        # 🟢🟢 降尺度粘贴：min > max ⇒ 交换；取值夹到 [0, 1]（>1 是放大，没有意义）
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if self.scale_min > self.scale_max:
            self.scale_min, self.scale_max = self.scale_max, self.scale_min
        self.scale_min = min(max(self.scale_min, 0.0), 1.0)
        self.scale_max = min(max(self.scale_max, 0.0), 1.0)
        # 下界 < 1 ⇒ 允许缩小；两个都 = 1.0（默认）⇒ 缩放分支整体跳过 ⇒ 逐位一致
        self.scaling = self.scale_min < 1.0
        # 运行期统计（便于训练日志核对"到底贴了多少个"）
        self.n_calls = 0
        self.n_pasted = 0
        self.n_images_pasted = 0
        self.n_no_src = 0
        self.n_no_place = 0

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _feather_alpha(h: int, w: int, feather: float) -> np.ndarray:
        """贴图的羽化 alpha（中心 1、四边线性淡出到 0）。"""
        a = np.ones((h, w), dtype=np.float32)
        f = int(min(feather, max(1, min(h, w) // 2)))
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            a[:f, :] *= ramp[:, None]
            a[-f:, :] *= ramp[::-1][:, None]
            a[:, :f] *= ramp[None, :]
            a[:, -f:] *= ramp[::-1][None, :]
        return a

    @staticmethod
    def _boxes_too_close(a: tuple[float, float, float, float],
                         b: tuple[float, float, float, float], gap: float) -> bool:
        """两个轴对齐框是否"太近"（外扩 gap 后相交）。"""
        return not (a[2] + gap <= b[0] or b[2] + gap <= a[0]
                    or a[3] + gap <= b[1] or b[3] + gap <= a[1])

    @staticmethod
    def _ring_median(patch: np.ndarray, bx1: int, by1: int, bx2: int, by2: int) -> float:
        """贴块里「目标框以外」那圈上下文的中位亮度（用于背景亮度匹配）。"""
        h, w = patch.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        mask[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)] = False
        if not mask.any():
            return float(np.median(patch))
        return float(np.median(patch[mask]))

    @staticmethod
    def _round_half_up(x: float) -> int:
        """确定性四舍五入（不是 numpy 的 banker's rounding）⇒ 同输入永远同输出。"""
        return int(math.floor(x + 0.5))

    @staticmethod
    def _area_resize(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """面积平均降采样（确定性、可复现；host 上没有 cv2 ⇒ 自己实现 numpy 版）。

        输出像素 (y, x) = 源区域 [y*h/out_h, (y+1)*h/out_h) × [x*w/out_w, ...) 的均值
        ⇒ 插值只发生在**已经抠出来的那块内部** ⇒ 块边界不会被邻域污染。
        """
        src_h, src_w = arr.shape[:2]
        if out_h == src_h and out_w == src_w:
            return arr.astype(np.float64)
        rows = []
        for y in range(out_h):
            y0 = int(np.floor(y * src_h / out_h))
            y1 = min(src_h, max(y0 + 1, int(np.ceil((y + 1) * src_h / out_h))))
            rows.append(arr[y0:y1].mean(axis=0))
        out = np.stack(rows, axis=0)
        cols = []
        for x in range(out_w):
            x0 = int(np.floor(x * src_w / out_w))
            x1 = min(src_w, max(x0 + 1, int(np.ceil((x + 1) * src_w / out_w))))
            cols.append(out[:, x0:x1].mean(axis=1))
        return np.stack(cols, axis=1)

    def _downscale(self, patch: np.ndarray, tx1: int, ty1: int,
                   tx2: int, ty2: int):
        """把贴块（目标 + 上下文）**整体**降尺度。

        顺序（关键）：
          ① **先在原分辨率把块抠出来**（传进来的 patch 就是原分辨率的抠块）
             ⇒ 绝不「先缩放原图大区域再抠」⇒ 块边界不引入插值污染；
          ② 再对这块做**面积平均降采样** ⇒ 目标与上下文**一起**变小
             ⇒ 贴出来的是「同样场景里更远的小目标」⇒ 局部对比度自洽、无残留环。

        返回 (新贴块, 新目标框 xyxy, 新宽, 新高, 新羽化半径)；
        返回 None ⇒ 缩放后目标 < 2px（贴不出有意义的目标）⇒ 调用方跳过该源目标。
        """
        ph, pw = patch.shape[:2]
        bw, bh = tx2 - tx1, ty2 - ty1
        s = float(np.random.uniform(self.scale_min, self.scale_max))
        nw = max(1, self._round_half_up(bw * s))
        nh = max(1, self._round_half_up(bh * s))
        if nw < 2 or nh < 2:
            return None                      # 缩放后 < 2px ⇒ 无法构成有意义目标
        if nw > self.max_src_px or nh > self.max_src_px:
            return None                      # 缩放后仍须满足 max_src_px 约束
        new_pw = max(1, self._round_half_up(pw * s))
        new_ph = max(1, self._round_half_up(ph * s))
        small = self._area_resize(patch, new_ph, new_pw)
        # 新目标框：与原目标同心、尺寸严格 = (nw, nh) ⇒ 整数像素、确定性可复现
        cx = (tx1 + tx2) * 0.5 * (new_pw / float(pw))
        cy = (ty1 + ty2) * 0.5 * (new_ph / float(ph))
        ntx1 = max(0, min(new_pw - nw, self._round_half_up(cx - nw * 0.5)))
        nty1 = max(0, min(new_ph - nh, self._round_half_up(cy - nh * 0.5)))
        ntx2, nty2 = ntx1 + nw, nty1 + nh
        # 目标四周至少留 1px 上下文（否则羽化会把目标本身淡掉）⇒ 边缘复制补齐
        pad_l = max(0, 1 - ntx1)
        pad_r = max(0, 1 - (new_pw - ntx2))
        pad_t = max(0, 1 - nty1)
        pad_b = max(0, 1 - (new_ph - nty2))
        if pad_l or pad_r or pad_t or pad_b:
            small = np.pad(small, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
            ntx1 += pad_l
            ntx2 += pad_l
            nty1 += pad_t
            nty2 += pad_t
            new_pw += pad_l + pad_r
            new_ph += pad_t + pad_b
        # 目标内容单独缩放后覆盖 ⇒ 贴出的像素与标签框**严格对齐**（不差 1px）
        small[nty1:nty2, ntx1:ntx2] = self._area_resize(patch[ty1:ty2, tx1:tx2], nh, nw)
        # 羽化带随场景一起变小（否则羽化会把小目标本身吃掉）
        feath = max(0.0, self.feather_px * s)
        return small, ntx1, nty1, ntx2, nty2, new_pw, new_ph, feath

    # ------------------------------------------------------------------ 主流程
    def __call__(self, labels: dict) -> dict:
        """对单个样本做微小目标 Copy-Paste（原地修改 labels 并返回）。"""
        if self.prob <= 0.0:
            return labels
        img = labels.get("img")
        inst = labels.get("instances")
        if img is None or inst is None or getattr(img, "size", 0) == 0:
            return labels
        boxes = np.asarray(inst.bboxes, dtype=np.float32)
        cls = labels.get("cls")
        if boxes is None or len(boxes) == 0 or cls is None:
            return labels
        self.n_calls += 1
        if np.random.rand() >= self.prob:
            return labels

        h, w = img.shape[:2]
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        tiny_idx = np.nonzero((ws >= 1) & (hs >= 1)
                              & (ws <= self.max_src_px) & (hs <= self.max_src_px))[0]
        if len(tiny_idx) == 0:
            self.n_no_src += 1
            return labels
        np.random.shuffle(tiny_idx)

        alpha_cache: dict[tuple[int, int, float], np.ndarray] = {}
        new_boxes: list[list[float]] = []
        new_cls: list[list[float]] = []
        existing = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes]
        n_pasted = 0
        ctx = int(self.ctx_px)

        for i in tiny_idx:
            if n_pasted >= self.max_per_img:
                break
            bx1, by1, bx2, by2 = (float(boxes[i][0]), float(boxes[i][1]),
                                  float(boxes[i][2]), float(boxes[i][3]))
            # ---- 抠块（目标框 + 上下文边距，裁到图内）--------------------------
            # 🔴🔴 2026-11-16 修（真根因）：原来 `min(w, ceil(bx2)) + ctx` 会让裁剪框
            #   **超出图像右/下边界 ctx 像素** ⇒ patch 实际宽度 < pw
            #   ⇒ 落点 ROI img[dy:dy+ph, dx:dx+pw] 被截断（宽度差正好 = ctx）
            #   ⇒ 与羽化 alpha (ph,pw) 广播失败 ⇒ 整个 DataLoader worker 崩
            #   （ValueError: (15,10,3) vs (15,13,1)，差 3px = 默认 ctx_px）
            #   ⇒ 四条边一律**钳位到图内**，pw/ph 与 patch 实际尺寸严格一致。
            cx1 = max(0, int(np.floor(bx1)) - ctx)
            cy1 = max(0, int(np.floor(by1)) - ctx)
            cx2 = min(w, int(np.ceil(bx2)) + ctx)
            cy2 = min(h, int(np.ceil(by2)) + ctx)
            pw, ph = cx2 - cx1, cy2 - cy1
            if pw < 1 or ph < 1:
                continue
            patch = img[cy1:cy2, cx1:cx2]
            if patch.size == 0:
                continue
            # 目标在贴块内的位置
            tx1 = int(round(bx1)) - cx1
            ty1 = int(round(by1)) - cy1
            tx2 = int(round(bx2)) - cx1
            ty2 = int(round(by2)) - cy1
            tx1 = max(0, min(pw - 1, tx1))
            ty1 = max(0, min(ph - 1, ty1))
            tx2 = max(tx1 + 1, min(pw, tx2))
            ty2 = max(ty1 + 1, min(ph, ty2))
            if self.p_flip > 0 and np.random.rand() < self.p_flip:
                patch = patch[:, ::-1]
                tx1, tx2 = pw - tx2, pw - tx1
            # 背景亮度取自**原分辨率**的贴块（降尺度前）⇒ 描述的是「源场景的背景」，
            # 而不是降采样把目标能量摊开后产生的人造晕圈（否则落点亮度永远配不上）
            src_bg = self._ring_median(patch, tx1, ty1, tx2, ty2)
            # 🟢🟢 降尺度粘贴（2026-11-13）：把抠出来的块**整体**缩小到 scale × 原尺寸
            #   ⚠️ scale_min == scale_max == 1.0（默认）⇒ 本段整体跳过 ⇒ 与旧行为逐位一致
            feath = self.feather_px
            if self.scaling:
                _ds = self._downscale(patch, tx1, ty1, tx2, ty2)
                if _ds is None:
                    continue                      # 缩放后 < 2px ⇒ 跳过该源目标
                patch, tx1, ty1, tx2, ty2, pw, ph, feath = _ds

            # ---- 选落点 --------------------------------------------------------
            placed = False
            for _ in range(self.max_tries):
                # 🟢🟢 2026-11-16 修（真 bug，已打挂夜间训练轴 T 通道）：
                #   贴块尺寸超过目标图（pw>w 或 ph>h）时，`max(1, w-pw+1)` 把落点钉到 0
                #   ⇒ 切片 img[dy:dy+ph, dx:dx+pw] 被**截断** ⇒ 与羽化 alpha 广播失败
                #   （ValueError: operands could not be broadcast (15,10,3) (15,13,1)）
                #   ⇒ 贴块放不下就直接放弃这次落点（不再截断）。
                if pw > w or ph > h:
                    break
                dx = int(np.random.randint(0, w - pw + 1))
                dy = int(np.random.randint(0, h - ph + 1))
                # 🟢🟢 2026-11-16 修（真 bug 2）：贴块经降尺度 + 边缘补齐后可能**比落点窗口大**
                #   ⇒ img[dy:dy+ph, dx:dx+pw] 被截断 ⇒ 与羽化 alpha 广播失败
                #   ⇒ 这里直接校验 ROI 尺寸，不一致就换落点（而不是崩掉整个 DataLoader）。
                if img[dy:dy + ph, dx:dx + pw].shape[:2] != (ph, pw):
                    continue
                cand = (float(dx + tx1), float(dy + ty1), float(dx + tx2), float(dy + ty2))
                if any(self._boxes_too_close(cand, e, self.min_gap_px) for e in existing):
                    continue
                if self.match_bg:
                    dst_ring = img[dy:dy + ph, dx:dx + pw]
                    if dst_ring.size:
                        dst_bg = self._ring_median(dst_ring, tx1, ty1, tx2, ty2)
                        if not (self.bg_lo * src_bg <= dst_bg <= self.bg_hi * src_bg):
                            continue
                key = (ph, pw, feath)
                alpha = alpha_cache.get(key)
                if alpha is None:
                    alpha = self._feather_alpha(ph, pw, feath)
                    alpha_cache[key] = alpha
                dst = img[dy:dy + ph, dx:dx + pw].astype(np.float32)
                a3 = alpha[:, :, None]
                img[dy:dy + ph, dx:dx + pw] = np.clip(
                    dst * (1.0 - a3) + patch.astype(np.float32) * a3, 0, 255).astype(np.uint8)
                new_boxes.append([cand[0], cand[1], cand[2], cand[3]])
                new_cls.append([float(np.asarray(cls[i]).reshape(-1)[0])])
                existing.append(cand)
                placed = True
                break
            if placed:
                n_pasted += 1
            else:
                self.n_no_place += 1

        if n_pasted == 0:
            return labels

        # ---- 更新标签（像素级 xyxy ⇒ 后续 Format 会再归一化）------------------
        all_boxes = np.concatenate(
            [boxes, np.asarray(new_boxes, dtype=np.float32)], axis=0)
        all_cls = np.concatenate(
            [np.asarray(cls, dtype=np.float32),
             np.asarray(new_cls, dtype=np.float32)], axis=0)
        if Instances is not None:
            labels["instances"] = Instances(
                all_boxes,
                segments=getattr(inst, "segments", None),
                keypoints=getattr(inst, "keypoints", None),
                bbox_format="xyxy",
                normalized=False,
            )
        else:  # pragma: no cover —— 仅在没有 ultralytics 的环境里
            labels["instances"].bboxes = all_boxes
        labels["cls"] = all_cls

        self.n_pasted += n_pasted
        self.n_images_pasted += 1
        return labels

    def stats(self) -> dict:
        """运行统计（训练日志里打印）。"""
        return {
            "calls": self.n_calls,
            "images_pasted": self.n_images_pasted,
            "objects_pasted": self.n_pasted,
            "no_tiny_source": self.n_no_src,
            "no_free_place": self.n_no_place,
            "avg_pasted_per_pasted_image": (
                self.n_pasted / self.n_images_pasted if self.n_images_pasted else 0.0),
        }
