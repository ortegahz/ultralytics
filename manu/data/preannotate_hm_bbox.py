#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pre-annotate one sequence with HM center candidates and YOLO26 BBox sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from manu.data.build_full_median_dataset import FastGMCEstimator
from manu.evaluation.heatmap_evaluate import extract_peaks
from manu.models.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-annotate a sequence: HM center + YOLO26 BBox size")
    parser.add_argument("--frames-dir", required=True, help="Directory of restored native frames (640x512)")
    parser.add_argument("--output-root", required=True, help="YOLO dataset root to create")
    parser.add_argument("--seq", default="", help="Sequence name, defaults to the frames directory name")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="YOLO split folder to write")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--bbox-weights", default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hm-conf", type=float, default=0.06, help="HM peak harvest threshold")
    parser.add_argument("--hm-top-k", type=int, default=50)
    parser.add_argument("--bbox-conf", type=float, default=0.06, help="YOLO26 BBox confidence threshold")
    parser.add_argument("--bbox-iou", type=float, default=0.70)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--coarse-min", type=float, default=6.0, help="Minimum HM coarse box size in native px")
    parser.add_argument("--coarse-max", type=float, default=40.0, help="Maximum HM coarse box size in native px")
    parser.add_argument("--include-bbox-only", action="store_true", help="Also emit YOLO boxes with no HM point as class 2")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy images into the dataset")
    parser.add_argument("--limit", type=int, default=0, help="Optional frame limit for a quick smoke test")
    return parser.parse_args()


def resolve_weights(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        path = PROJECT_ROOT / value
    if not path.exists():
        raise FileNotFoundError(value)
    return path


def load_hm_model(weights: Path, device: torch.device, stride_fallback: int = 2, use_p0_highway: bool = True,
                  no_p2: bool = False, head_level: str = "p3", act: str = "silu",
                  head_conv: int | None = None, late_upsample: bool | None = None, head_src: str = "",
                  p2_out_scale: float = 1.0, offset_init: str = "keep", offset_init_weights: str = "",
                  head_init: str = "reset", hf_skip: bool | None = None,
                  hf_skip_src: str | None = None, hf_skip_channels: int | None = None):
    """offset_init（2026-11-09 新增，针对「offset 头塌缩成常数 0.5」的定向修复）：
        keep    —— 保持默认（用 init ckpt 的 offset 权重）
        reset   —— **只**把 offset 分支留成随机初始化（其余权重照常继承）
                    ⇒ 用于排除「塌缩是初始化带来的」，让高 offset 权重从随机头开始学
        inherit —— 从 offset_init_weights 指定的其他 ckpt **只**继承 offset 分支（形状兼容才继承）
                    ⇒ 用于「借用 cs_t0043 已收敛的 offset 几何先验」
        head_init（**头**的初始化策略，仅在 act 变化（silu→relu）时生效）：
            reset —— 现有行为：头完全随机初始化（实测：20 ep 里只有 5 ep 有效，head 学不透 ⇒
                     种子臂的 T1 只到 0.87~0.89）
            abs   —— ★新：把**继承来的 silu 头权重取绝对值**后再喂 relu
                     （relu(x)=max(0,x) 会让负权重通道死亡；取绝对值可保住「正侧幅度」，
                      等价于把 silu 的双侧响应压成单侧但保留量级）⇒ 相当于给新头一个
                      **已训练过的起点**，而不是随机猜。
    """
    ckpt = torch.load(weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = int(ckpt.get("stride", stride_fallback))
    # ---- 结构线（2026-11-05）：ckpt 自描述 ----
    # 显式传入的值优先（调度器按 search space 给），否则用 ckpt 里的记录。
    _ck_no_p2 = bool(ckpt.get("no_p2", False))
    _ck_head_level = str(ckpt.get("head_level", "p3"))
    _ck_act = str(ckpt.get("act", "silu"))
    _ck_head_src = str(ckpt.get("head_src", "") or "")
    _ck_p2s = float(ckpt.get("p2_out_scale", 1.0) or 1.0)
    # ★ head_conv / late_upsample 也要自描述（2026-11-06 修静默丢权重 bug）：
    #   以前只能靠调用方显式传 ⇒ 漏传就重建出**另一张图**（hc 回落 64、late 回落 False）
    #   ⇒ 头权重形状对不上又被静默跳过 ⇒ 随机头推理 ⇒ 热图全 0 ⇒ 被误判成"死模型"。
    _ck_head_conv = ckpt.get("head_conv", None)
    _ck_head_conv = int(_ck_head_conv) if _ck_head_conv is not None else None
    _ck_late = bool(ckpt.get("late_upsample", False))
    # ---- 跨层高频残差注入（hf_skip）自描述 ----
    # ⚠️ 用 None 当"未指定"哨兵（与 late_upsample 同理）：调用方显式传 0/1 才是"要这个结构"；
    #   不传 ⇒ 用 ckpt 里记录的 ⇒ 缓存/导出/续训都能重建出**同一张图**。
    _ck_hf_skip = bool(ckpt.get("hf_skip", False))
    _ck_hf_skip_src = str(ckpt.get("hf_skip_src", "b1") or "b1")
    _ck_hf_skip_channels = int(ckpt.get("hf_skip_channels", 1) or 1)
    # 显式传入优先（调度器按搜索空间给），否则用 ckpt 里的记录
    if not head_src:
        head_src = _ck_head_src
    if no_p2 is False and _ck_no_p2:
        no_p2 = True
    if head_level == "p3" and _ck_head_level != "p3":
        head_level = _ck_head_level
    if act == "silu" and _ck_act != "silu":
        act = _ck_act
    # ★★ 2026-11-06 18:25：**act 变了（silu→relu）⇒ 重置「头」的权重**（保持随机初始化）。
    #   实测证据（20 epoch 完整轨迹的最终 hm）：
    #     · p3/hc64/**relu**/late（**继承 silu 权重**）⇒ hm 卡在 **0.68~0.80**（差 4~5 倍）
    #     · p2f/hc32/**relu**/late（头形状变了 ⇒ **头重新初始化**）⇒ hm **0.20~0.44**
    #     · 老轮 p3/hc64/**silu**/late（继承 silu 权重）⇒ hm **0.1643**（基准）
    #   ⇒ **relu 本身没问题；问题是「用 silu 训好的头去喂 relu」⇒ 负权重通道在 relu 下大量死亡**
    #   ⇒ 头必须重来（随机初始化），骨干/颈仍可继承（实证：骨干继承 relu 下照常收敛）。
    _act_changed = str(act).lower() != str(_ck_act).lower()
    _head_init = str(head_init or "reset").lower()
    # ★★ 2026-11-10（路线二：P1/P2 + 通道瘦身 + relu）：
    #   head_init 新增 "merge_abs" —— **形状不一致**的头张量走 merge_channels（保住已学滤波器）
    #   之后**再取绝对值**；形状一致的仍按 "abs" 处理。
    #   动机：merge 出来的权重是 **silu 训的**（含负权重）⇒ 直接喂 relu 会让一半通道死亡
    #   （这正是 2026-11-06 记录的现象：silu 头喂 relu ⇒ hm 卡在 0.68~0.80）。
    #   取绝对值保住量级，让新头有一个"活着"的起点；随后必须**充分重训**（50~100 ep）。
    #   注意 "abs" 只处理形状一致的张量，"merge_abs" 才覆盖形状不一致（换宽度/换 head_src）的情形。
    # ★★ 2026-11-10：新增 "merge" —— 换结构时按通道合并继承，**但不取绝对值**
    #   （abs 会抹掉滤波器的符号结构 ⇒ 合并后激活量级爆炸 ⇒ 实测首 epoch loss 8.5e4 vs 健康臂 ~20
    #   ⇒ 三个模式 reset / merge / merge_abs 用短程对照实验决定，见 memory §11x）
    _abs_head_mode = _head_init in ("abs", "merge_abs")
    _merge_head_mode = _head_init in ("merge", "merge_abs")
    _reset_head = _act_changed and not _abs_head_mode
    # ★ offset 分支的定向初始化（2026-11-09）：offset_init ∈ {keep, reset, inherit}
    _offset_mode = str(offset_init or "keep").lower()
    _offset_reset = _offset_mode == "reset"
    # ★ head_conv / late_upsample：None = 调用方**未指定** ⇒ 用 ckpt 自描述；显式传值 = 调用方要这个结构。
    # ⚠️ late_upsample 用 None 当"未指定"哨兵（不能用 False）：False 分不清"没传"和"显式要 False"，
    #   而搜索 worker（optuna_mix_ratio_search.py:903）**确实**显式传搜索空间的 False ⇒ 若让 ckpt
    #   一律优先，init ckpt 带 late=True 的 trial 会被静默改成 True 结构 ⇒ 训出来的结构 ≠ 登记的结构。
    if head_conv is None:
        head_conv = _ck_head_conv
    if late_upsample is None:
        late_upsample = _ck_late
    else:
        late_upsample = bool(late_upsample)
    p2_out_scale = float(p2_out_scale) if p2_out_scale else _ck_p2s
    if hf_skip is None:
        hf_skip = _ck_hf_skip
    else:
        hf_skip = bool(hf_skip)
    if hf_skip_src is None:
        hf_skip_src = _ck_hf_skip_src
    else:
        hf_skip_src = str(hf_skip_src or "b1")
    if hf_skip_channels is None:
        hf_skip_channels = _ck_hf_skip_channels
    else:
        hf_skip_channels = int(hf_skip_channels or 1)
    # "换结构"判定：重建结构 ≠ ckpt 自描述 ⇒ 调用方要的是别的结构（训练侧换结构初始化合法）
    _override = (
        (head_conv, bool(late_upsample), head_src, no_p2, head_level, act, round(p2_out_scale, 6))
        != (_ck_head_conv, _ck_late, _ck_head_src, _ck_no_p2, _ck_head_level, _ck_act, round(_ck_p2s, 6))
    )
    p0_kwargs = ckpt.get(
        "p0_kwargs",
        {
            "use_spatial_gate": True,
            "stem_type": "standard_dw",
            "downsample_mode": "pixel_unshuffle",
            "gate_input_mode": "diff_only",
            "gate_mid_channels": 16,
            "gate_depth": 2,
            "fusion_mode": "scalar_gate",
        },
    )
    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=use_p0_highway,
        p0_highway_kwargs=p0_kwargs,
        no_p2=no_p2,
        head_level=head_level,
        act=act,
        head_conv=head_conv,
        late_upsample=late_upsample,
        head_src=head_src,
        p2_out_scale=p2_out_scale,
        hf_skip=hf_skip,
        hf_skip_src=hf_skip_src,
        hf_skip_channels=hf_skip_channels,
    )
    own_state = model.state_dict()
    matched = 0
    merged_count = 0
    reset_head = 0
    reset_offset = 0
    abs_head = 0
    dropped_p0 = 0
    # ★ 分开统计两种"没加载上"：missing = 模型里**没有这个 key**（历史键，多半无害）；
    #   mismatch = 模型里**有**但形状不一致 ⇒ 重建的结构不是 ckpt 的结构 ⇒ 致命。
    mismatch: list[tuple[str, tuple, tuple]] = []
    missing = 0
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_key not in own_state:
            if clean_key.startswith("p0_highway"):
                dropped_p0 += 1
            else:
                missing += 1
            continue
        if own_state[clean_key].shape == value.shape:
            if _reset_head and (clean_key.startswith("head.") or clean_key.startswith("head_in.")):
                if _abs_head_mode:
                    # ★ head_init=abs：继承来的 silu 头权重**取绝对值**后再喂 relu
                    #   （负权重通道在 relu 下会死；取绝对值保住量级 ⇒ 头有一个已训练的起点）
                    if value.is_floating_point() and value.dim() >= 2:
                        own_state[clean_key].copy_(value.abs())
                        matched += 1
                        abs_head += 1
                    else:
                        # BN 的 running_mean/var 等非权重张量：直接继承
                        own_state[clean_key].copy_(value)
                        matched += 1
                        abs_head += 1
                    continue
                # ★ act 变了 ⇒ 头保持随机初始化（否则 relu 下大量神经元死亡 ⇒ 学不动）
                #   ⚠️ 但 head_init=merge / merge_abs 模式下**形状一致的头张量原样继承**
                #   （合并=恒等拷贝），避免无谓地把已训好的头丢掉。
                if _merge_head_mode:
                    own_state[clean_key].copy_(value)
                    matched += 1
                    merged_count += 1
                    continue
                reset_head += 1
                continue
            if _offset_reset and ".offset." in clean_key:
                # ★ offset_init=reset ⇒ 只重置 offset 分支（其余照常继承）
                reset_offset += 1
                continue
            own_state[clean_key].copy_(value)
            matched += 1
        elif clean_key.startswith("p0_highway"):
            dropped_p0 += 1
        elif _override:
            # ★★ 2026-11-06：训练侧**换结构**（head_conv 64→32 / head_src p1|p3→p2_fused）
            #   ⇒ 头的 conv 形状变了 ⇒ 原本整块丢弃 ⇒ 新头从零学 ⇒ 20 ep 学不动（死模型）。
            #   ⇒ 这里按**通道合并**继承：输出通道等距挑选、输入通道分组取均值
            #   ⇒ 新头至少拿到「训练过的滤波器」⇒ 首 epoch loss 从 127 降到 ~10 量级。
            _merged = _merge_channels(value, own_state[clean_key].shape)
            if _merged is not None:
                # ★★ head_init=merge_abs：形状不一致（换宽度/换 head_src）时，合并结果取绝对值
                #   ⇒ silu 训出的负权重不会在 relu 下整批死亡（路线二 relu 线必需）。
                if (_head_init == "merge_abs" and _act_changed
                        and (clean_key.startswith("head.") or clean_key.startswith("head_in."))
                        and _merged.is_floating_point() and _merged.dim() >= 2):
                    _merged = _merged.abs()
                own_state[clean_key].copy_(_merged)
                matched += 1
                merged_count += 1
            else:
                mismatch.append((clean_key, tuple(value.shape), tuple(own_state[clean_key].shape)))
        else:
            mismatch.append((clean_key, tuple(value.shape), tuple(own_state[clean_key].shape)))

    # STRUCT GATE：结构对不上就崩，绝不静默用随机头（宁可响亮地崩，也不要"热图全 0 = 死模型"的假结论）。
    # 例外：调用方**显式换了结构**（训练侧从旧 ckpt 起训新结构是设计内的）⇒ 只警告不崩。
    if mismatch:
        sample = "; ".join(f"{k} ckpt={a} model={b}" for k, a, b in mismatch[:5])
        if not _override:
            raise RuntimeError(
                f"[STRUCT GATE] {len(mismatch)} 个非 P0 张量形状不匹配 ⇒ 拒绝加载（否则随机头 ⇒ 热图全 0 ⇒ "
                f"被误判成死模型）| ckpt={weights} | head_conv ckpt={_ck_head_conv} "
                f"used={head_conv if head_conv is not None else 64} | late_upsample ckpt={_ck_late} "
                f"used={bool(late_upsample)} | 示例: {sample}"
            )
        print(f"[STRUCT GATE] 警告：{len(mismatch)} 个非 P0 张量形状不匹配（调用方显式换了结构，按设计跳过）"
              f" | ckpt={weights} | 示例: {sample}", flush=True)

    # ABLATION GATE: use_p0_highway=False is only meaningful if the model really has no P0 stream AND
    # every p0_highway tensor was actually dropped. Silently "succeeding" with a P0 model would make
    # the whole ablation a no-op that still reports lower numbers, so fail loudly instead.
    # Same gate as inference/cache_trial0474_inferences.py --ablate-p0.
    if not use_p0_highway:
        if model.p0_highway is not None:
            raise RuntimeError("[ABLATE-P0 GATE] model.p0_highway is not None -- ablation did not take effect")
        if dropped_p0 == 0:
            raise RuntimeError("[ABLATE-P0 GATE] 0 p0_highway.* tensors were dropped -- nothing was ablated")
        print(f"[ABLATE-P0 GATE] PASS: p0_highway=None, {dropped_p0} p0 tensors dropped")

    # ★★ offset_init=inherit：从另一个 ckpt **只**继承 offset 分支（形状兼容才继承）
    #   动机（2026-11-09）：P3/hc64/relu/late 臂的 offset 头塌缩成常数 ≈0.5
    #   （实测输出 std 仅 0.066、与真值回归斜率仅 0.129）⇒ 亚像素信息被压扁，
    #   解码后 σ=2.32px = 8px 格距的量化下限 ⇒ T1 掉 2.4 pp。
    #   cs_t0043 的 offset 头是收敛完备的（σ≈1.1px）⇒ 可作为「几何先验」来源。
    inherited_offset = 0
    if _offset_mode == "inherit":
        if not offset_init_weights:
            raise ValueError("[OFFSET-INIT] offset_init=inherit 需要 --offset-init-weights")
        _oc = torch.load(resolve_weights(offset_init_weights), map_location="cpu")
        _os = _oc["model"] if "model" in _oc else _oc.get("state_dict", _oc)
        if hasattr(_os, "state_dict"):
            _os = _os.state_dict()
        for _k, _v in _os.items():
            _ck = _k.replace("module.", "").replace("model.model.", "").replace("model.", "")
            if ".offset." not in _ck or _ck not in own_state:
                continue
            if own_state[_ck].shape != _v.shape:
                print(f"[OFFSET-INIT] 形状不兼容，跳过 {tuple(_v.shape)} vs {tuple(own_state[_ck].shape)}: {_ck}")
                continue
            own_state[_ck].copy_(_v)
            inherited_offset += 1
        print(f"[OFFSET-INIT] inherit: 从 {offset_init_weights} 继承 {inherited_offset} 个 offset 张量")

    model.to(device).eval()
    print(f"[INFO] HM weights loaded: {matched} tensors from {weights} | use_p0_highway={use_p0_highway} "
          f"| no_p2={no_p2} head_level={head_level} act={act} "
          f"| head_conv={head_conv if head_conv is not None else 64} late_upsample={bool(late_upsample)} "
          f"| missing_keys={missing} | stride={model.stride}"
          + (f" | ★通道合并继承={merged_count} 个张量" if merged_count else "")
          + (f" | ★act变化⇒重置头={reset_head} 个张量" if reset_head else "")
          + (f" | ★head_init={_head_init}" + (f"(abs继承{abs_head}个)" if abs_head else ""))
          + (f" | ★offset_init={_offset_mode}"
             + (f" 重置{reset_offset}个" if reset_offset else "")
             + (f" 继承{inherited_offset}个" if inherited_offset else "")))
    # no_p2 时输出 stride 由 head 吃哪一级决定（P3=>8 / P2=>4），与 ckpt 里的 stride 无关
    return model, model.stride


def _pick_channels(idx_src: int, idx_dst: int) -> list[int]:
    """在 src 的 idx_src 个通道里，等距挑出 idx_dst 个（覆盖全部范围，不只取头部）。"""
    if idx_src == idx_dst:
        return list(range(idx_src))
    if idx_dst >= idx_src:
        return list(range(idx_src)) + [idx_src - 1] * (idx_dst - idx_src)
    step = idx_src / idx_dst
    return [min(idx_src - 1, int(round(i * step + step / 2))) for i in range(idx_dst)]


def merge_channels(src: torch.Tensor, dst_shape: torch.Size):
    """把「训练好的」src 张量按通道合并/裁剪成 dst_shape ⇒ **换宽度的新头也能继承知识**。

    为什么需要（2026-11-06 实测）
    --------------------------
    `head_conv` 从 64 砍到 32（或 `head_src` 换到 p2_fused）⇒ 头的 conv 权重**形状变了**
    ⇒ cs_t0043 的头权重被整块丢弃 ⇒ 新头**从零学** ⇒ 20 epoch 学不动
    （实测首 epoch loss 127、最终 1.07、T1 F1 0.25 ⇒ 死模型）。
    ⇒ 这里把源张量按通道**合并**（输入通道取组均值）/ **重采样**（输出通道等距挑选），
    至少让新头拿到「训练过的滤波器」而不是纯随机。

    规则（保守 ⇒ 只在通道维不同时才动手，空间维不同一律放弃）：
    * 1-D（BN weight/bias/running_mean/var）：等距索引取 ⇒ 与 conv 的输出通道选择**一致**；
    * N-D conv 权重 [O, I, kh, kw]：输出通道等距挑选；输入通道**分组取均值**（保留响应幅度）；
    * 其它：None（调用方按形状不匹配处理）。

    ⚠️ 这是**近似迁移**，不是数学等价的重参数化 ⇒ 迁移后必须**重训**（本项目就是 20 ep 全参微调）。
    """
    if src.shape == dst_shape:
        return src
    if src.dim() != len(dst_shape):
        return None
    if src.dim() == 1:
        if int(src.shape[0]) < int(dst_shape[0]):
            return None
        idx = _pick_channels(int(src.shape[0]), int(dst_shape[0]))
        return src[torch.tensor(idx, dtype=torch.long)]
    for d in range(2, len(dst_shape)):
        if int(src.shape[d]) != int(dst_shape[d]):
            return None
    o_src, i_src = int(src.shape[0]), int(src.shape[1])
    o_dst, i_dst = int(dst_shape[0]), int(dst_shape[1])
    if o_dst > o_src or i_dst > i_src:
        return None
    out_idx = _pick_channels(o_src, o_dst)
    src_o = src[torch.tensor(out_idx, dtype=torch.long)]
    if i_dst == i_src:
        return src_o
    groups = i_src / i_dst                      # 输入通道分组取均值（保留响应幅度）
    parts = []
    for j in range(i_dst):
        lo = int(round(j * groups))
        hi = max(lo + 1, int(round((j + 1) * groups)))
        parts.append(src_o[:, lo:hi].mean(dim=1, keepdim=True))
    return torch.cat(parts, dim=1)


def _merge_channels(src: torch.Tensor, dst_shape: torch.Size):
    """``merge_channels`` 的容错包装（异常 ⇒ None）。"""
    try:
        return merge_channels(src, dst_shape)
    except Exception:  # pragma: no cover
        return None


def letterbox_params(width: int, height: int, size: int) -> tuple[float, int, int, int, int]:
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    return scale, left, top, new_w, new_h


def letterbox_gray(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)


def to_native(point: np.ndarray, scale: float, left: int, top: int) -> np.ndarray:
    return np.array([(point[0] - left) / scale, (point[1] - top) / scale], dtype=np.float32)


def build_hm_features(gray_frames: list[np.ndarray], imgsz: int, window: int, stride: int, gmc: FastGMCEstimator) -> np.ndarray:
    features = np.zeros((len(gray_frames), 3, imgsz, imgsz), dtype=np.uint8)
    for index, current in enumerate(tqdm(gray_frames, desc="HM features (GMC+median)", unit="frame", dynamic_ncols=True)):
        lag_index = max(0, index - stride)
        lag_frame = gray_frames[lag_index]
        motion = cv2.absdiff(current, gmc.warp(lag_frame, gmc.compute_affine(lag_frame, current)))
        history = []
        for step in range(1, window + 1):
            history_index = max(0, index - step * stride)
            history_frame = gray_frames[history_index]
            history.append(gmc.warp(history_frame, gmc.compute_affine(history_frame, current)))
        background = np.median(np.stack(history, axis=0), axis=0).astype(np.float32)
        residual = np.clip(current.astype(np.float32) - background, 0, 255).astype(np.uint8)
        merged = np.stack([residual, motion, current], axis=2)
        features[index] = letterbox_gray(merged, imgsz).transpose(2, 0, 1)
    return features


def build_bbox_inputs(gray_frames: list[np.ndarray]) -> list[np.ndarray]:
    inputs = []
    for index, current in enumerate(gray_frames):
        previous = gray_frames[max(0, index - 1)]
        previous2 = gray_frames[max(0, index - 2)]
        inputs.append(np.stack([current, cv2.absdiff(current, previous), cv2.absdiff(current, previous2)], axis=2))
    return inputs


def run_hm(model, features: np.ndarray, device: torch.device, batch: int, conf: float, top_k: int, stride: int):
    all_peaks, all_heatmaps = [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(features), batch), desc="HM inference", unit="batch", dynamic_ncols=True):
            chunk = torch.from_numpy(features[start : start + batch]).to(device).float() / 255.0
            prediction = model(chunk)
            all_peaks.extend(
                extract_peaks(prediction["heatmap"], prediction["offset"], stride=stride, conf_thresh=conf, top_k=top_k)
            )
            all_heatmaps.extend(prediction["heatmap"][:, 0].cpu().numpy())
    return all_peaks, all_heatmaps


def run_bbox(model, inputs: list[np.ndarray], device: str, conf: float, iou: float, batch: int, imgsz: int):
    results = []
    for start in tqdm(range(0, len(inputs), batch), desc="YOLO26 BBox inference", unit="batch", dynamic_ncols=True):
        chunk = inputs[start : start + batch]
        predictions = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        for prediction in predictions:
            if prediction.boxes is not None and len(prediction.boxes) > 0:
                boxes = prediction.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = prediction.boxes.conf.cpu().numpy().astype(np.float32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)
            results.append((boxes, confs))
    return results


def estimate_coarse_size(
    heatmap: np.ndarray,
    peak_lb: np.ndarray,
    conf: float,
    scale: float,
    stride: int,
    coarse_min: float,
    coarse_max: float,
) -> tuple[float, float]:
    height, width = heatmap.shape[:2]
    radius = 20
    cx = int(round(peak_lb[0] / stride))
    cy = int(round(peak_lb[1] / stride))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    window = heatmap[y0:y1, x0:x1]
    if window.size == 0:
        return coarse_min, coarse_min
    threshold = max(conf, 0.5 * float(window.max()))
    ys, xs = np.nonzero(window >= threshold)
    if len(xs) < 2:
        return coarse_min, coarse_min
    width_native = (xs.max() - xs.min() + 1) * stride / scale
    height_native = (ys.max() - ys.min() + 1) * stride / scale
    return (
        float(np.clip(width_native, coarse_min, coarse_max)),
        float(np.clip(height_native, coarse_min, coarse_max)),
    )


def merge_frame_detections(
    points_lb: np.ndarray,
    scores_lb: np.ndarray,
    heatmap: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    native_width: int,
    native_height: int,
    scale: float,
    left: int,
    top: int,
    stride: int,
    hm_conf: float,
    coarse_min: float,
    coarse_max: float,
    include_bbox_only: bool,
) -> tuple[list[str], dict, dict[int, int]]:
    lines = []
    frame_record = {"detections": []}
    class_counts = {0: 0, 1: 0, 2: 0}
    used_boxes = set()
    for point_lb, score in zip(points_lb, scores_lb):
        point = to_native(point_lb, scale, left, top)
        candidates = [
            (float(confs[b]), b)
            for b in range(len(boxes))
            if boxes[b][0] <= point[0] <= boxes[b][2] and boxes[b][1] <= point[1] <= boxes[b][3]
        ]
        if candidates:
            best_conf, best_box = max(candidates, key=lambda item: item[0])
            x1, y1, x2, y2 = boxes[best_box]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            width, height = x2 - x1, y2 - y1
            class_id = 0
            used_boxes.add(best_box)
            source = "hm_bbox"
            source_conf = best_conf
        else:
            width, height = estimate_coarse_size(heatmap, point_lb, hm_conf, scale, stride, coarse_min, coarse_max)
            cx, cy = point[0], point[1]
            class_id = 1
            source = "hm_coarse"
            source_conf = float(score)
        lines.append(f"{class_id} {cx / native_width:.6f} {cy / native_height:.6f} {width / native_width:.6f} {height / native_height:.6f}")
        class_counts[class_id] += 1
        frame_record["detections"].append({"source": source, "class": class_id, "score": round(source_conf, 4), "center": [round(float(cx), 2), round(float(cy), 2)], "size": [round(float(width), 2), round(float(height), 2)]})

    if include_bbox_only:
        for b in range(len(boxes)):
            if b in used_boxes:
                continue
            x1, y1, x2, y2 = boxes[b]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            width, height = x2 - x1, y2 - y1
            lines.append(f"2 {cx / native_width:.6f} {cy / native_height:.6f} {width / native_width:.6f} {height / native_height:.6f}")
            class_counts[2] += 1
            frame_record["detections"].append({"source": "bbox_only", "class": 2, "score": round(float(confs[b]), 4), "center": [round(float(cx), 2), round(float(cy), 2)], "size": [round(float(width), 2), round(float(height), 2)]})
    return lines, frame_record, class_counts


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_root = Path(args.output_root)
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)
    seq_name = args.seq or frames_dir.name
    frame_paths = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    if args.limit:
        frame_paths = frame_paths[: args.limit]
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    images_dir = output_root / "images" / args.split
    labels_dir = output_root / "labels" / args.split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    gray_frames = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in frame_paths]
    if any(frame is None for frame in gray_frames):
        raise RuntimeError("At least one frame cannot be decoded")
    native_height, native_width = gray_frames[0].shape[:2]
    scale, left, top, _, _ = letterbox_params(native_width, native_height, args.imgsz)
    print(f"[INFO] Sequence: {seq_name} | frames: {len(frame_paths)} | native: {native_width}x{native_height} | letterbox scale={scale}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    hm_model, stride = load_hm_model(resolve_weights(args.hm_weights), device)
    bbox_model = YOLO(str(resolve_weights(args.bbox_weights)))

    started = time.perf_counter()
    features = build_hm_features(gray_frames, args.imgsz, args.median_window, args.temporal_stride, FastGMCEstimator(downscale=2))
    hm_peaks, hm_heatmaps = run_hm(hm_model, features, device, 16, args.hm_conf, args.hm_top_k, stride)
    bbox_inputs = build_bbox_inputs(gray_frames)
    bbox_results = run_bbox(bbox_model, bbox_inputs, args.device, args.bbox_conf, args.bbox_iou, 16, args.imgsz)
    print(f"[INFO] Inference finished in {time.perf_counter() - started:.1f}s")

    provenance = []
    class_counts = {0: 0, 1: 0, 2: 0}
    for index, frame_path in enumerate(tqdm(frame_paths, desc="Merging labels", unit="frame", dynamic_ncols=True)):
        peaks = hm_peaks[index]
        heatmap = hm_heatmaps[index]
        boxes, confs = bbox_results[index]

        if len(peaks["points"]) > 0:
            order = np.argsort(-peaks["scores"])
            points_lb = peaks["points"][order]
            scores_lb = peaks["scores"][order]
        else:
            points_lb = np.zeros((0, 2), dtype=np.float32)
            scores_lb = np.zeros((0,), dtype=np.float32)

        lines, frame_record, counts = merge_frame_detections(
            points_lb,
            scores_lb,
            heatmap,
            boxes,
            confs,
            native_width,
            native_height,
            scale,
            left,
            top,
            stride,
            args.hm_conf,
            args.coarse_min,
            args.coarse_max,
            args.include_bbox_only,
        )
        frame_record["frame"] = frame_path.name
        for class_id, value in counts.items():
            class_counts[class_id] += value

        target_stem = f"{seq_name}__{index:06d}"
        (labels_dir / f"{target_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not args.no_copy:
            shutil.copy(frame_path, images_dir / f"{target_stem}{frame_path.suffix}")
        frame_record["label"] = f"{target_stem}.txt"
        provenance.append(frame_record)

    names = {0: "uav_bbox", 1: "uav_hm_coarse"}
    if args.include_bbox_only:
        names[2] = "uav_bbox_only"
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        f"path: {output_root.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "\n".join(f"  {key}: {value}" for key, value in names.items())
        + "\n",
        encoding="utf-8",
    )
    (output_root / f"provenance_{seq_name}.json").write_text(
        json.dumps({"sequence": seq_name, "frames": provenance}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[SUCCESS] images -> {images_dir}")
    print(f"[SUCCESS] labels -> {labels_dir}")
    print(f"[SUCCESS] data.yaml -> {data_yaml}")
    print(f"[INFO] class 0 (hm_bbox): {class_counts[0]} | class 1 (hm_coarse): {class_counts[1]} | class 2 (bbox_only): {class_counts[2]}")


if __name__ == "__main__":
    main()
