#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optuna 超参搜索：**dataloader 混合比例** 为主变量（口径 1 = Batch 内采样概率）。

背景与目标
----------
路线 B 首轮试跑（5 epoch）已把 Ep3 固化为 RC1，但其源域大盘付出 **−0.38pp** 的代价
（Track1 F1 0.905736 → 0.901918）。本搜索要回答的问题只有一个：
**把 Anti-UAV 的采样比重从自然的 1:3.09 调到严格的 1:4（甚至更偏），
能不能把源域那笔损失救回来，同时不牺牲难例轨 001324 的收益？**

"真正的 1:4" 的定义（唯一口径 = 口径 1：Batch 内采样概率）
------------------------------------------------------------
| 口径 | 定义 | 1:4 是否成立 |
|:--|:--|:--|
| 口径1 | **Batch 内采样概率** P(LQ)=0.20 / P(UAV)=0.80 | ✅ **本搜索采用** |
| 口径2 | 数据集总帧数（两边各遍历一次） | ❌ 物理不成立：13,918×4 = 55,672 > 43,008，实际极限 1:3.09 |
| 口径3 | GT 目标框数 | 1:3.36（12,120 : 40,761） |
| 口径4 | 局部切片张数 | 路线 A 之后才成立 |

实现方式：**WeightedRandomSampler**（零数据复制、零磁盘开销），
通过**只在官方 `build_dataloader` 调用期间替换索引采样器**来实现，
不重写 DataLoader（§十四 教训：自写 DataLoader 曾致样本腰斩 + 凭空多 576 处假警）。

判定标准：三级漏斗（见 `manu/training/mix_ratio_judgement.py`）
-------------------------------------------------------------
* **一级** 安全红线（相对冻结基线）：T2 F1 ≥ 0.985 且 ΔFP_T2 ≤ +10 且 ΔF1_T1 ≥ −0.005
  → 触碰任一条 ⇒ **剪枝 Pruned**（不参与 TPE 采样）
* **三级** 效用分作为 Optuna 标量 objective：
  `U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)`
  （**RC1/Ep3 的效用分 = 0.027579**，是"更好"的及格线）
* **二级** 帕累托占优用于最终选型（`select_best`），与 objective 同尺度，不参与搜索排序

工程特性
--------
* **4 卡并行**，每个 trial **单 GPU**、**3 epoch**；门禁在同卡串行跑
* **可随时停止 / 随时继续**：Optuna SQLite study 持久化 + 断点续训状态 + 陈旧 RUNNING trial 自愈
* **后台常驻**：`setsid nohup` 启动，与 SSH 会话完全解耦（宿主机断电不影响）
* **状态可查**：`manu/training/status_mix_ratio_search.py` 随时读库/读日志/读 GPU
* **有截止时间**：`--max-hours` 到点自动收尾（保证第二天早上有结果）

用法（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，
根目录 `/tmp/pycharm_project_10ae9e2e`）：

```bash
# 启动（后台常驻，宿主机断电无影响）
bash manu/training/launch_mix_ratio_search.sh

# 看状态（任何时候、任何机器都能查）
python manu/training/status_mix_ratio_search.py

# 停止（可恢复）
bash manu/training/stop_mix_ratio_search.sh
```
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from torch.cuda.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import ConcatDataset

from manu.data.preannotate_hm_bbox import letterbox_gray, load_hm_model, resolve_weights  # noqa: E402
from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets  # noqa: E402
from manu.training.mix_ratio_judgement import (  # noqa: E402
    FROZEN_BASELINE,
    RC1_EPOCH3,
    evaluate_model_quality,
    hard_gate,
    metrics_from_gate_report,
    utility,
)

# ==============================================================================
# 常量
# ==============================================================================
LQ_EXPECT, UAV_EXPECT = 13918, 43008  # 满额护栏（§十四 腰斩事故）
TOTAL_EXPECT = LQ_EXPECT + UAV_EXPECT

# 路线 B 的分层参数组（3e-6 : 1.5e-5 : 3e-5），搜索只缩放整体倍率
GROUP_LRS = {"G2_p4p5": 3e-6, "G3_neck": 1.5e-5, "G4_head": 3e-5}
FROZEN_BACKBONE = ("b0", "b1", "b2", "b3", "b4")
# 可训练骨干段（周末扩展：可按 backbone_lr_scale / freeze_backbone_epochs 保护）
G2_MODULES = ("b5", "b6", "b7", "b8", "b9", "b10")

# 混合比例候选（mix_spec）：natural = 路线 B 对照（自然 1:3.09），wN = 加权采样 1:N
MIX_SPECS = [
    "natural",
    "w1.0", "w1.5", "w2.0", "w2.5", "w3.0", "w3.5", "w4.0", "w5.0", "w6.0", "w8.0",
]

# ==============================================================================
# 扩展搜索空间（2026-10-30 周末 no-GMC 复现实验）
# ==============================================================================
# 核心假设（用户提出）：
#   「GMC RC2（trial_0080）作为**初始化权重**起步，比冻结基线（474 去 P0）
#     拥有高得多的『弱小目标表征上限』；但必须配合**更小的步长 + 更长的 epoch
#     + 骨干保护**，否则未对齐残差的巨大梯度会灾难性遗忘掉 trial_0080 的浅层特征核。」
#
# ⇒ 在原 11 维搜索空间上**加 4 维**：
#   init_from              : 初始化权重来源（base474 / rc2）
#   epochs                 : 训练轮数（3 太少 ⇒ 扩到 6/10/15）
#   backbone_lr_scale      : G2(b5..b10，骨干可训练段) 的 lr 相对倍率（0=全冻结）
#   freeze_backbone_epochs : 前 N 个 epoch 锁死骨干，第 N+1 epoch 起解冻（渐进式）
#
# ⚠️ **默认关闭**（EXTENDED_SPACE=False ⇒ suggest_params 与旧版逐位相同），
#    只有显式传 --extended-space 才启用 ⇒ **GMC 侧既有 study 的搜索空间不受污染**。
INIT_CHOICES = ["base474", "rc2"]
# ⚠️ **这里必须覆盖对照矩阵用到的每一个 epoch 数**（含 8）。
#    Optuna 的 suggest_categorical 对"固定参数不在 choices 里"会直接抛
#    `ValueError: '8' not in (3, 6, 10, 15)` ⇒ **入队阶段就炸**（实测踩过）。
#    改这个列表时，必须同步改 nogmc_weekend_plan.json。
EPOCH_CHOICES = [3, 6, 8, 10, 15]
BACKBONE_LR_SCALE_CHOICES = [1.0, 0.3, 0.1, 0.0]
FREEZE_BB_EPOCH_CHOICES = [0, 2]
# lr_scale 下探到 0.2（RC2 初始化需要小步长；旧版下限 0.3）
# ⚠️ **上界必须是 4.5 而不是 3.0**（2026-11-06 实测踩到）：
#    plan 的母配方是 lr_scale=3.5（cs_t0043 复现口径），超参邻域还要 4.0
#    ⇒ 上界 3.0 时 Optuna 抛 `ValueError: 4.0 is out of range` ⇒ 整个搜索起不来。
#    （实测日志里先出现的是 `UserWarning: Fixed parameter lr_scale with value 3.5 is
#      out of range for distribution FloatDistribution(high=3.0...)`，紧接着才在
#      enqueue 阶段抛错。）
# ⚠️ FloatDistribution 的 **low/high 不参与 Optuna 的动态值空间检查**
#    （只有 `log` 参与，见 optuna.distributions.check_distribution_compatibility）
#    ⇒ 调 low/high 是安全的；**但 categorical 的 choices 一旦被记录就不能改**。
LR_SCALE_LOW, LR_SCALE_HIGH = 0.2, 4.5
# warmup 上限抬到 1.5 epoch（用户要求 warmup ≥ 1 个完整 epoch）
WARMUP_EPOCHS_LOW, WARMUP_EPOCHS_HIGH = 0.2, 1.5

# 由调度器按 --extended-space 设置（worker 进程不读它）
EXTENDED_SPACE = False

# ==============================================================================
# 结构搜索空间（--struct-space，2026-11-05 新结构线）
# ==============================================================================
# 目标模型 = **YOLO26n 原生拓扑（P3-P5）+ heatmap 头 + all relu**，
# 从 cs_t0043 初始化，把"删 P2 分支 / 删 bbox 回归分支"做成**可搜索的维度**。
#
# ⚠️ 默认关闭（STRUCT_SPACE=False ⇒ suggest_params 与旧版逐位相同），
#    且**必须用新 study-name**（categorical 取值集合一旦被 study 记录就不能改）。
STRUCT_SPACE = False
# no_p2=True  ⇒ 删掉 neck 的 P2 融合级（c19/up3/down1/c22/up_p2/fuse_p2）与 P1 320 头分支
STRUCT_NO_P2_CHOICES = [True, False]   # False = 原结构，放**复现对照臂**用
#   （对照臂 = 同配方 + 原结构 + silu ⇒ 应当复现 cs_t0043 的 T1/T3，用于验证流水线）
# head 吃哪一级：p3 = 标准 YOLO26 的 P3（/8, 80x80，最省搬运）
#                 p2 = 骨干 /4 stem（/4, 160x160，保住极小目标分辨率）
# ⚠️ 必须与 argparse 的 --head-level choices **逐项一致**（2026-11-06 修）：
#    plan 里会出现 head_level="p1"（对照臂）与 "p2_fused"（主力结构），
#    而 Optuna 的 suggest_categorical 对"固定参数不在 choices 里"**直接抛 ValueError**
#    ⇒ `'p1' not in ('p3', 'p2')` ⇒ 整个搜索起不来。
STRUCT_HEAD_LEVEL_CHOICES = ["p1", "p3", "p2", "p2_fused"]
# ★ head_src 是**统一**的结构维度（2026-11-06）：一次搜索同时覆盖
#   p1(现状 320) / p2_raw(骨干/4 160，已证伪) / p2_fused(结构B 160 融合后) / p3(结构A 80)
STRUCT_HEAD_SRC_CHOICES = ["p1", "p2_fused", "p3"]
# ★ neck 宽度维度：收窄 P2 融合输出通道 ⇒ 砍 160x160 搬运量（只动可训练组 ⇒ 起点仍好）
# ⚠️ 0.25 也必须在列（2026-11-06）：p2_out_scale=0.25 的结构在速度表里存在
#    （p2f_hc48_late_sc025 = 22.43 ms 等），plan 一旦引用它，
#    `suggest_categorical` 对"固定参数不在 choices 里"会抛 ValueError ⇒ 搜索起不来。
STRUCT_P2_OUT_SCALE_CHOICES = [1.0, 0.75, 0.5, 0.25]
# act：relu = 全换（用户口径 "all relu"）；silu = 对照
STRUCT_ACT_CHOICES = ["relu", "silu"]
# head_conv：64 = 现状；48 = 实测甜点（p2f_hc48_relu = 20.60 ms）；
#            32/40 = 更窄（用于定位「头宽 ↔ 容量」拐点）。
# ⚠️ **48 必须在列**（2026-11-06 实测踩到）：plan 引用 hc48，
#    而 choices 只有 (64,32) ⇒ `ValueError: '48' not in (64, 32)` ⇒ **整个搜索起不来**。
# ⚠️ 而且 **choices 一旦被某 study 记录就不能再改**（Optuna 的 CategoricalDistribution
#    不支持 dynamic value space）⇒ 改这里必须配一个**新 study-name**。
STRUCT_HEAD_CONV_CHOICES = [64, 48, 40, 32]
# late_upsample：头内推迟上采样
STRUCT_LATE_CHOICES = [False, True]
# unfreeze_b0b4：all-relu 会改变冻结底模 b0..b4 的函数 ⇒ 冻结着就永久退化
#   ⇒ True 必须同时解冻 b0..b4（仅 74,848 参数 = 全网 2.9%）
STRUCT_UNFREEZE_CHOICES = [True, False]

# ==============================================================================
# 聚焦搜索空间（2026-11-02 定点打靶模式，--focus-space）
# ==============================================================================
# 为什么要它：周末 114 个 trial 已把**宏观规律**打透（rc2 起步 / 6~15 ep / lr 1.0~2.9 /
# 骨干保护 / mix 2.0~6.0），但 TPE 仍在 11~15 维全空间里盲搜 ⇒ 大部分算力浪费在
# 3ep、w8.0、lr>3.0、base474+大 lr 这类**已知无望的角落**。
# ⇒ 聚焦模式把每一维**收窄到贴着已知好点**的区间/取值，
#    让同样的算力换成确定性最高的收敛收益（"定点打靶"而非"发散盲搜"）。
#
# ⚠️ 用在**新 study**上（换 study-name）：categorical 取值集合一旦被某 study 记录，
#    换集合会触发 Optuna 的 "dynamic value space" 报错。
FOCUS_SPACE = False
FOCUS_CHOICES = {
    # 初始化权重：rc2 是主力（周末实测 init 效应 +1.6~2.9 pp），
    # 保留 base474 只作**对照**（同参成对比较要用）
    "init_from": ["rc2", "base474"],
    # epoch：3 ep 已被证"半夹生"，从 6 起（甜点 6~8，探顶 15）
    "epochs": [6, 8, 10, 15],
    # 混合比：2.0~8.0（w1.0/w1.5 已被证无望；w6.0/w8.0 是源域高占比方向）
    "mix_spec": ["w2.0", "w3.0", "w6.0", "w8.0", "natural"],
    # 骨干 lr 倍率：0（整段冻结）/0.1/0.3/1.0（不缩放）
    "backbone_lr_scale": [0.0, 0.1, 0.3, 1.0],
    # 渐进解冻：0（不解冻）/2（前 2 ep 锁骨干）
    "freeze_backbone_epochs": [0, 2],
}
# lr 区间：weekend 实测有效带 1.0~2.92；收窄到 [0.8, 3.0]
FOCUS_LR_SCALE = (0.8, 3.0)

# 扩展 init 来源（soup 融合权重等）。
# ⚠️ **只能用在新参数名 `init_from_x` 下**，原因见 suggest_params() 里的长注释：
#    Optuna 的分布兼容性检查是 **study 级**的，换 choices 会让本 study 的
#    **所有**新 trial 报 "CategoricalDistribution does not support dynamic value space"，
#    而 Optuna 4.9 没有 optuna.delete_trial ⇒ 删不掉历史 trial ⇒ 换不了 study。
INIT_X_CHOICES = ["rc2", "base474", "soup_strict3", "soup_high3_nobn"]

# 路线 B 已验证的超参（作为种子与缩放中心）
ROUTE_B_DEFAULTS = {
    "lr_scale": 1.0,
    "warmup_epochs": 0.5,
    "weight_decay": 1e-4,
    "focal_beta": 2.7,
    "offset_weight": 0.55,
    "mosaic": 0.05,
    "scale": 0.20,
    "translate": 0.10,
    "clip_max_norm": 5.0,
    "freeze_bn": False,
    "lr_min": 1e-6,
}

# 门禁阈值（与用户一级红线一致）
GATE_TRACK2_F1_MIN = 0.985
GATE_TRACK2_FP_MAX_INCREMENT = 10
GATE_TRACK1_MIN_DELTA = -0.005
# auto_epoch_gate_check.py 的 --max-track2-f1/--max-track2-fp 语义是**绝对阈值**
# （默认 36 = 基线 FP 26 + 允许增量 10），而我们内部一律用"相对冻结基线的增量"。
# 直接把增量 10 传给它会被当成绝对上限 ⇒ 只要 FP > 10 就误判一票否决。
# 这里换算成绝对上限，保证脚本自带 verdict 与我们的 hard_gate() 判定一致。
GATE_TRACK2_FP_ABS_MAX = FROZEN_BASELINE["track2_fp"] + GATE_TRACK2_FP_MAX_INCREMENT  # = 36
# 同理，--min-track1-f1 是**绝对** F1 下限。我们的红线是"相对冻结基线 ΔF1 ≥ -0.005"，
# 换算成绝对值 = 0.905736 - 0.005 = 0.900736。这样门禁脚本自带的 verdict 与我们的
# hard_gate() 完全等价 ⇒ gate_report.json 里的 pass/veto 字段可直接审计，无需另读代码。
GATE_TRACK1_F1_ABS_MIN = FROZEN_BASELINE["track1_f1"] + GATE_TRACK1_MIN_DELTA  # = 0.900736
UTILITY_ALPHA, UTILITY_BETA, UTILITY_GAMMA = 0.5, 2.5, 0.0005


# ==============================================================================
# 1. 加权采样器：只在官方 build_dataloader 期间替换索引采样器
# ==============================================================================
@contextlib.contextmanager
def weighted_sampler_patch(weights: np.ndarray, num_samples: int, seed: int):
    """在官方 build_dataloader 调用期间，把索引采样器换成 WeightedRandomSampler。

    为什么要这么做（而不是自己写 DataLoader）
    ------------------------------------------
    torch 1.13 的 `DataLoader.__init__` 在 `sampler is None and shuffle=True` 时
    使用**模块级** `RandomSampler(dataset, generator=generator)`。
    我们只在调用窗口内替换这个名字，于是：
      · 数据集仍是官方 `build_yolo_dataset`（图片/标签扫描、缓存、mosaic 全官方）
      · BatchSampler / _RepeatSampler / collate_fn / worker_init_fn / generator / prefetch 全官方
      · **唯一**变化是"抽哪些下标"由 WeightedRandomSampler 决定
    ⇒ 索引语义（batch_idx）与官方完全一致（collate_fn 里 batch_idx += 批内位置），
      不会出现 §十四 的"样本腰斩 + 凭空多假警"。
    """
    import torch.utils.data.dataloader as dl

    original = dl.RandomSampler

    def _factory(dataset, generator=None, **kwargs):  # noqa: ANN001
        g = generator if generator is not None else torch.Generator()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=g,
        )

    dl.RandomSampler = _factory
    try:
        yield
    finally:
        dl.RandomSampler = original


# ==============================================================================
# 2. 参数空间
# ==============================================================================
# ------------------------------------------------------------------------------
# 2.0 搜索空间覆盖（命令行可改；**只能配新 study-name 用**）
# ------------------------------------------------------------------------------
# 为什么需要它
# ------------
# Optuna 的分布兼容性检查是 **study 级**的：
#   `optuna/storages/_rdb/storage.py::_set_trial_param_without_commit` 按
#   **(study_id, param_name)** 查历史分布，再 `check_distribution_compatibility(old, new)`：
#     · CategoricalDistribution 换 choices ⇒ `ValueError: CategoricalDistribution does
#       not support dynamic value space`
#     · FloatDistribution 改范围 ⇒ `ValueError: FloatDistribution does not support
#       dynamic value space`（或 range 不兼容）
#   ⇒ **同一个 study 记录过某个参数名之后，该 study 之后所有新 trial 都会炸**，
#     跟那个 trial 是不是老 trial 无关。
#   而 Optuna 4.9 **没有** `optuna.delete_trial` ⇒ 历史 trial 删不掉 ⇒ 换不了 study。
# ⇒ 唯一合法路径：**换新 study-name**（如 mix_ratio_search_nogmc_t3）
#    + 让空间**可被命令行覆盖**（本节）。
#
# ⚠️ **默认值 = 空 ⇒ 行为逐位不变**：GMC 侧与既有 no-GMC study 的空间完全不受污染。
#
# 本文件的"两套空间常量"与覆盖的关系
# ----------------------------------
#   · focus 套：FOCUS_CHOICES / FOCUS_LR_SCALE        ⇒ 只在 --focus-space 时被读
#   · legacy 套：MIX_SPECS / (LR_SCALE_LOW,LR_SCALE_HIGH) / EPOCH_CHOICES /
#                INIT_CHOICES / INIT_X_CHOICES         ⇒ 非 focus 分支被读
# 本节的覆盖**两边都同步改**（唯一例外是 lr：只覆盖 focus 套的 FOCUS_LR_SCALE，
# 因为 legacy 套的 (LR_SCALE_LOW, LR_SCALE_HIGH) 属于"既有 GMC study 的空间"，不能碰）
# ⇒ 一套 CLI 参数在两种模式下语义一致，不会两处漂移。

SPACE_OVERRIDE_FIELDS = ("lr_scale", "epochs", "mix_spec", "init_from", "init_from_x")


def _split_csv(raw) -> list[str]:
    """逗号分隔解析：去首尾空格、忽略空项。"""
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def parse_int_list(raw, field: str) -> list[int]:
    """把逗号分隔字符串解析成 int 列表（容错 + 中文报错）。"""
    items = _split_csv(raw)
    if not items:
        raise ValueError(f"[SPACE-OVERRIDE] {field} 解析后为空（原始：{raw!r}）")
    out: list[int] = []
    for tok in items:
        try:
            out.append(int(tok))
        except ValueError:
            raise ValueError(
                f"[SPACE-OVERRIDE] {field} 必须是逗号分隔的整数，非法项 {tok!r}（原始：{raw!r}）"
            ) from None
    return out


def parse_str_list(raw, field: str) -> list[str]:
    """把逗号分隔字符串解析成 str 列表（容错 + 中文报错）。"""
    items = _split_csv(raw)
    if not items:
        raise ValueError(f"[SPACE-OVERRIDE] {field} 解析后为空（原始：{raw!r}）")
    return items


def apply_space_overrides(args) -> dict:
    """把命令行覆盖应用到模块级空间常量，并返回**生效后的空间快照**。

    ⚠️ **纯逻辑**：不连数据库、不起训练、不读数据文件 ⇒ 可以被
       `manu/training/check_space_override.py` 直接导入复用（同一份逻辑 ⇒ 不会两处漂移）。

    ⚠️ **调用时机**：必须在 ``optuna.create_study()`` **之前**调用 ——
       study 一旦建好并记录了分布，之后再改空间就会触发上面说的兼容性报错。

    参数
    ----
    args : argparse.Namespace
        需要含 lr_scale_low / lr_scale_high / epoch_choices / mix_choices /
        init_choices / init_x_choices（缺失 ⇒ 视为"不覆盖"）。

    返回
    ----
    dict：空间快照 —— ``focus`` / ``legacy`` 两套取值 + ``effective``（当前模式实际生效的
    那套）+ ``overridden``（哪些维度被覆盖了）+ ``epoch_min_est``（耗时表）。
    """
    global FOCUS_LR_SCALE, EPOCH_CHOICES, MIX_SPECS, INIT_CHOICES, INIT_X_CHOICES

    low = getattr(args, "lr_scale_low", None)
    high = getattr(args, "lr_scale_high", None)
    epochs_raw = str(getattr(args, "epoch_choices", "") or "")
    mix_raw = str(getattr(args, "mix_choices", "") or "")
    init_raw = str(getattr(args, "init_choices", "") or "")
    init_x_raw = str(getattr(args, "init_x_choices", "") or "")

    # ---- lr 区间（只覆盖 focus 套：FOCUS_LR_SCALE；legacy 的 LR_SCALE_* 不动）----
    lo, hi = FOCUS_LR_SCALE
    if low is not None:
        lo = float(low)
    if high is not None:
        hi = float(high)
    if low is not None or high is not None:
        if not lo > 0:
            raise ValueError(f"[SPACE-OVERRIDE] lr 下界必须 > 0（Optuna log 采样要求），当前 {lo}")
        if not hi > lo:
            raise ValueError(f"[SPACE-OVERRIDE] lr 上界必须 > 下界，当前 lo={lo} hi={hi}")
        FOCUS_LR_SCALE = (lo, hi)

    # ---- epochs（focus 套 + legacy 套同步改）----
    if epochs_raw.strip():
        epochs = parse_int_list(epochs_raw, "epochs")
        if any(e <= 0 for e in epochs):
            raise ValueError(f"[SPACE-OVERRIDE] epochs 必须为正整数：{epochs}")
        FOCUS_CHOICES["epochs"] = list(epochs)
        EPOCH_CHOICES = list(epochs)

    # ---- mix_spec（focus 套 + legacy 套同步改）----
    if mix_raw.strip():
        mix = parse_str_list(mix_raw, "mix_spec")
        FOCUS_CHOICES["mix_spec"] = list(mix)
        MIX_SPECS = list(mix)

    # ---- init_from（focus 套 + legacy 套同步改）----
    if init_raw.strip():
        init = parse_str_list(init_raw, "init_from")
        FOCUS_CHOICES["init_from"] = list(init)
        INIT_CHOICES = list(init)

    # ---- init_from_x（两套共用同一个常量）----
    if init_x_raw.strip():
        INIT_X_CHOICES = list(parse_str_list(init_x_raw, "init_from_x"))

    overridden = {
        "lr_scale": low is not None or high is not None,
        "epochs": bool(epochs_raw.strip()),
        "mix_spec": bool(mix_raw.strip()),
        "init_from": bool(init_raw.strip()),
        "init_from_x": bool(init_x_raw.strip()),
    }

    if any(overridden.values()):
        print(
            f"[SPACE-OVERRIDE] lr=({FOCUS_LR_SCALE[0]},{FOCUS_LR_SCALE[1]}) "
            f"epochs={FOCUS_CHOICES['epochs']} mix={FOCUS_CHOICES['mix_spec']} "
            f"init={FOCUS_CHOICES['init_from']} init_x={INIT_X_CHOICES}",
            flush=True,
        )
        if not FOCUS_SPACE:
            print(
                "[SPACE-OVERRIDE] ⚠️ 未启用 --focus-space ⇒ lr 覆盖不生效"
                "（非 focus 分支读的是 LR_SCALE_LOW/LR_SCALE_HIGH，本节不动它们）",
                flush=True,
            )
        missing = [e for e in FOCUS_CHOICES["epochs"] if e not in EPOCH_MIN_EST]
        if missing:
            print(
                f"[SPACE-OVERRIDE] ⚠️ epoch {missing} 不在 EPOCH_MIN_EST 表里 ⇒ 派单前"
                f"「这条臂能不能在 hard_cut 前跑完」的判断会回落到默认值 "
                f"{EPOCH_MIN_EST_DEFAULT} 分钟（可能误判）",
                flush=True,
            )

    focus = {
        "lr_scale": (FOCUS_LR_SCALE[0], FOCUS_LR_SCALE[1]),
        "epochs": list(FOCUS_CHOICES["epochs"]),
        "mix_spec": list(FOCUS_CHOICES["mix_spec"]),
        "init_from": list(FOCUS_CHOICES["init_from"]),
        "init_from_x": list(INIT_X_CHOICES),
        "backbone_lr_scale": list(FOCUS_CHOICES["backbone_lr_scale"]),
        "freeze_backbone_epochs": list(FOCUS_CHOICES["freeze_backbone_epochs"]),
    }
    legacy = {
        "lr_scale": (LR_SCALE_LOW, LR_SCALE_HIGH),
        "epochs": list(EPOCH_CHOICES),
        "mix_spec": list(MIX_SPECS),
        "init_from": list(INIT_CHOICES),
        "init_from_x": list(INIT_X_CHOICES),
        "backbone_lr_scale": list(BACKBONE_LR_SCALE_CHOICES),
        "freeze_backbone_epochs": list(FREEZE_BB_EPOCH_CHOICES),
    }
    return {
        "focus_space": bool(FOCUS_SPACE),
        "extended_space": bool(EXTENDED_SPACE),
        "focus": focus,
        "legacy": legacy,
        "effective": focus if FOCUS_SPACE else legacy,
        "overridden": overridden,
        "epoch_min_est": {int(k): v for k, v in sorted(EPOCH_MIN_EST.items())},
    }


def suggest_params(trial: optuna.Trial) -> dict:
    """搜索空间。

    ⚠️ **EXTENDED_SPACE=False 时与旧版逐位相同**（11 维），保证 GMC 侧既有 study
    的搜索空间不被新维度污染；只有新实验（传 --extended-space）才走 15 维分支。

    ⚠️ **FOCUS_SPACE=True（--focus-space）时**再进一步把每一维收窄到"贴着已知好点"的
    取值/区间（见 FOCUS_CHOICES / FOCUS_LR_SCALE）⇒ 定点打靶而非发散盲搜。
    ⚠️ 聚焦模式**必须用新 study-name**（categorical 取值集合被 study 记录后不可改）。
    """
    params = {
        # ★ 关键参数：dataloader 混合比例（口径 1 = Batch 内采样概率）
        "mix_spec": trial.suggest_categorical(
            "mix_spec", FOCUS_CHOICES["mix_spec"] if FOCUS_SPACE else MIX_SPECS),
        # 其余超参
        "lr_scale": trial.suggest_float(
            "lr_scale",
            *(FOCUS_LR_SCALE if FOCUS_SPACE else (LR_SCALE_LOW, LR_SCALE_HIGH)),
            log=True),
        "warmup_epochs": trial.suggest_float("warmup_epochs", WARMUP_EPOCHS_LOW, WARMUP_EPOCHS_HIGH),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        "focal_beta": trial.suggest_float("focal_beta", 2.2, 2.9, step=0.05),
        "offset_weight": trial.suggest_float("offset_weight", 0.40, 0.70, step=0.05),
        "mosaic": trial.suggest_float("mosaic", 0.0, 0.15, step=0.05),
        "scale": trial.suggest_float("scale", 0.10, 0.30, step=0.05),
        "translate": trial.suggest_float("translate", 0.05, 0.15, step=0.05),
        "clip_max_norm": trial.suggest_float("clip_max_norm", 2.0, 8.0),
        "freeze_bn": trial.suggest_categorical("freeze_bn", [True, False]),
    }
    if EXTENDED_SPACE:
        # ---- 周末扩展 4 维（2026-10-30）----
        params["init_from"] = trial.suggest_categorical(
            "init_from", FOCUS_CHOICES["init_from"] if FOCUS_SPACE else INIT_CHOICES)
        params["epochs"] = trial.suggest_categorical(
            "epochs", FOCUS_CHOICES["epochs"] if FOCUS_SPACE else EPOCH_CHOICES)
        params["backbone_lr_scale"] = trial.suggest_categorical(
            "backbone_lr_scale",
            FOCUS_CHOICES["backbone_lr_scale"] if FOCUS_SPACE else BACKBONE_LR_SCALE_CHOICES
        )
        params["freeze_backbone_epochs"] = trial.suggest_categorical(
            "freeze_backbone_epochs",
            FOCUS_CHOICES["freeze_backbone_epochs"] if FOCUS_SPACE else FREEZE_BB_EPOCH_CHOICES
        )
        # ---- 扩展 init（soup 融合权重等）：走**新参数名** init_from_x ----
        #
        # ⚠️ **为什么不能直接给 init_from 换 choices**（2026-11-02 实测踩过）：
        #    Optuna 的分布兼容性检查是 **study 级**的 ——
        #    `optuna/storages/_rdb/storage.py::_set_trial_param_without_commit` 里
        #    `previous_distribution` 是按 **(study_id, param_name)** 查
        #    `TrialParamModel` 里的**任意一条**记录拿到的（不是同一个 trial！），
        #    然后 `check_distribution_compatibility(old, new)` 对 CategoricalDistribution
        #    直接抛 `ValueError: CategoricalDistribution does not support dynamic value space`。
        #    ⇒ 本 study 一旦记录过 `init_from=(rc2, base474)`，
        #      **之后任何用新 choices 的 trial 都会炸**，跟那个 trial 是不是老 trial 无关。
        #    ⇒ 而 Optuna 4.9 **没有** `optuna.delete_trial`（实测 hasattr=False），
        #      删不掉历史 trial ⇒ 也换不了 study-name。
        # ⇒ **解法：换一个新参数名 `init_from_x`**。这个名字在本 study 里没有历史记录
        #    ⇒ 不触发兼容性检查 ⇒ 可以安全地承载 soup_* 这类新 init。
        #
        # ⚠️ **只在「固定参数里显式带了 init_from_x」时才 suggest**：
        #    · 已入队的 11 条 WAITING 臂没有这个键 ⇒ 不会被 suggest ⇒ 行为逐位不变；
        #    · TPE 臂（study.ask() 出来的）也没有 ⇒ 不会被 suggest ⇒ TPE 采样空间不变。
        #    ⇒ 只有 plan 里显式写了这个键的确定性臂才会走新分支。
        _fx = (trial.system_attrs or {}).get("fixed_params") or {}
        if "init_from_x" in _fx:
            params["init_from_x"] = trial.suggest_categorical("init_from_x", INIT_X_CHOICES)
    if STRUCT_SPACE:
        # ---- 结构线维度（2026-11-05）----
        # ⚠️ 这些是**新参数名**，本 study 无历史记录 ⇒ 不触发 Optuna 的
        #    "CategoricalDistribution does not support dynamic value space"。
        params["no_p2"] = trial.suggest_categorical("no_p2", STRUCT_NO_P2_CHOICES)
        params["head_level"] = trial.suggest_categorical("head_level", STRUCT_HEAD_LEVEL_CHOICES)
        params["act"] = trial.suggest_categorical("act", STRUCT_ACT_CHOICES)
        params["head_conv"] = trial.suggest_categorical("head_conv", STRUCT_HEAD_CONV_CHOICES)
        params["late_upsample"] = trial.suggest_categorical("late_upsample", STRUCT_LATE_CHOICES)
        params["unfreeze_b0b4"] = trial.suggest_categorical("unfreeze_b0b4", STRUCT_UNFREEZE_CHOICES)
        params["head_src"] = trial.suggest_categorical("head_src", STRUCT_HEAD_SRC_CHOICES)
        params["p2_out_scale"] = trial.suggest_categorical("p2_out_scale", STRUCT_P2_OUT_SCALE_CHOICES)
    return params



# ==============================================================================
# 速度闸门（2026-11-06 新目标：NPU int8 <= 20 ms）
# ==============================================================================
# 板卡今天 18:30 断电，而训练窗口有 ~62.5 h（≈68~75 条臂的预算）
# ⇒ **不可能给每条臂都上板测速**。
# 但 **NPU 推理时间只取决于「结构」（图形状/算子数/搬运量），不取决于「权重」**
#   （唯一例外：塌陷模型被 int8 常量折叠 —— 用 null control 排除）。
# ⇒ 于是：今天（板卡在线）把候选**结构**谱系测成表；周末训练出的臂，结构已知 ⇒
#   **速度查表**（表外用离线预测器）⇒ **派单前就能剪掉 >20 ms 的结构**，不浪费训练。
#
# 表文件 schema（manu/pipeline/npu/probe_struct_speed.py 产出）：
#   {"arms": [{"name":..., "spec":{head_src,head_conv,act,ablate_p0},
#              "cycles":{"npu_cycles":..,"rw_mb":..}, "ms":..}, ...],
#    "predictor": {"a":..,"b":..,"c":..}}


def structure_key(head_src: str, head_conv: int, act: str,
                  late_upsample: bool = False, p2_out_scale: float = 1.0) -> tuple:
    """结构→速度表与查询**共用**的 key。

    ⚠️ **两边必须完全一致**（2026-11-06 实测踩到）：之前表用 4 维（含 ablate_p0）、
    查询用 5 维（含 late/p2_out_scale）⇒ **永远查不到 ⇒ 闸门等于没有 ⇒ 超限臂全被放行**。
    ⇒ 现在统一成 5 维，且**不含 ablate_p0**（部署形态恒为 P0 关 ⇒ 表里只取 ablate_p0=True 的项）。
    """
    return (head_src, int(head_conv), act, bool(late_upsample),
            round(float(p2_out_scale), 3))


def load_speed_table(path: str | None):
    """读结构→速度表；返回 (key→ms 映射, 预测器系数)。"""
    if not path:
        return {}, None
    p = Path(path)
    if not p.is_file():
        print(f"[SPEED-GATE] 表不存在（不启用剪枝）：{p}", flush=True)
        return {}, None
    d = json.loads(p.read_text(encoding="utf-8"))
    key2ms = {}
    for a in d.get("arms", []):
        sp = a.get("spec") or {}
        if a.get("ms") is None:
            continue
        # 部署形态恒为 P0 关 ⇒ 只收 ablate_p0=True 的项
        if not bool(sp.get("ablate_p0", True)):
            continue
        key = structure_key(sp.get("head_src", "p1"), int(sp.get("head_conv", 64)),
                            sp.get("act", "relu"), bool(sp.get("late_upsample", False)),
                            float(sp.get("p2_out_scale", 1.0) or 1.0))
        prev = key2ms.get(key)
        if prev is None or float(a["ms"]) < prev:
            key2ms[key] = float(a["ms"])
    pred = d.get("predictor")
    print(f"[SPEED-GATE] 载入 {len(key2ms)} 个结构的速度；预测器={'有' if pred else '无'}（{p}）", flush=True)
    return key2ms, pred


def speed_of_structure(params: dict, key2ms: dict, pred: dict | None):
    """给定 trial 参数 ⇒ 该结构的板卡 ms。返回 (ms 或 None, 来源)。

    key 必须与表里的 key 完全一致：(head_src, head_conv, act, late_upsample, p2_out_scale)。
    ⚠️ 部署形态恒为 **P0 关** ⇒ 表里的 ablate_p0=True 项才对应部署态。
    表外 ⇒ 用预测器（cycles/RW 近似）⇒ 仍能判"大概率超限"，避免放行超限臂。
    """
    key = structure_key(params.get("head_src") or "p1",
                        int(params.get("head_conv", 64) or 64),
                        params.get("act", "relu"),
                        bool(params.get("late_upsample", False)),
                        float(params.get("p2_out_scale", 1.0) or 1.0))
    if key in key2ms:
        return key2ms[key], "表"
    if pred:
        # 表外：用同族的 cycles/RW 近似（取同 head_src/act 的最宽实测项做参考）
        return None, "表外(预测器未标定该结构)"
    return None, "表外"


def parse_mix_spec(spec: str) -> tuple[str, float]:
    """mix_spec → (mode, uav_per_lq)。natural 模式的实际帧比是 43008/13918 ≈ 3.09。"""
    if spec == "natural":
        return "natural", UAV_EXPECT / LQ_EXPECT
    return "weighted", float(spec[1:])


def read_runtime_config(out_root: Path) -> dict:
    """读取 ``<out_root>/runtime_config.json``（**可选**，解析失败一律回落到空字典）。

    用途：**不重启调度器**就能改后续 trial 的训练协议。
    典型场景（2026-10-21）：把后续 trial 从 3 epoch 切到 5 epoch，
    以便复现 RC1 的完整 5-epoch 曲线（Ep3 与 Ep5 都在同一次运行里）。

    键：
      · ``train_epochs`` (int)  后续 trial 训练几个 epoch
      · ``gate_epoch``   (int)  门禁量第几个 epoch（0/缺省 = 最后一个）
      · ``workers``      (int)  dataloader worker 数
      · ``stop_at``      (str)  **墙钟截止**（``"2026-11-03 04:00"`` 或 epoch 秒）——
                          过了这个时刻就**不再派新 trial**，开始收尾。
                          用途：给「主机定时关机」留出**门禁 + 收尾**的时间窗。
      · ``hard_cut``     (str)  **硬截止**（同上格式）—— 不派「**预计跑不完**」的臂
                          （按实测 epoch→耗时表判断），保证在跑的臂都能在 hard_cut 前跑完。
                          ⇒ 这两个键都**可在运行期改**（不重启调度器）。
    """
    path = Path(out_root) / "runtime_config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {k: data[k] for k in ("train_epochs", "gate_epoch", "workers",
                                     "stop_at", "hard_cut", "n_trials") if k in data}
    except Exception as exc:
        print(f"[RUNTIME] runtime_config.json 解析失败（忽略，用启动参数）：{exc}", flush=True)
        return {}


# ==============================================================================
# 每 epoch 的**实测**臂耗时（分钟）
# ==============================================================================
# 来源：周末 study（mix_ratio_search_nogmc_weekend）80 条 COMPLETE 的
#   datetime_start → datetime_complete 中位数（每档 min/max 极差 < 2 min ⇒ 极稳）：
#     ep=3 → 46.9 ｜ ep=6 → 73.8 ｜ ep=8 → 92.2 ｜ ep=10 → 110.3 ｜ ep=15 → 154.8
# 用途：调度器在**派单前**判断「这条臂能不能在 hard_cut 前跑完」
#       ⇒ 主机定时关机前，**最后一条臂也能准时跑完**（不会拖到关机之后）。
#
# ⚠️ **新增 epoch 档位必须同步补这张表**（否则 `.get(ep, EPOCH_MIN_EST_DEFAULT)`
#    回落到 110 分钟 ⇒ 派单前的超时判断失真 ⇒ 臂会被拖到 hard_cut 之后才跑完）。
# ep=20 / ep=25 的推算依据（2026-11-03）：8ep=92、10ep=110、15ep=155（含末尾门禁
#   ~19.5 min）⇒ 每 epoch 训练 ≈ 9.15 min ⇒ 20ep ≈ 202、25ep ≈ 248 ⇒ 取整 205 / 250。
EPOCH_MIN_EST = {3: 47, 6: 74, 8: 92, 10: 110, 15: 155, 20: 205, 25: 250}
EPOCH_MIN_EST_DEFAULT = 110
EPOCH_MIN_EST_MARGIN = 1.10   # 10% 余量（保守估计，宁可早停不可超时）


def parse_wallclock(value) -> float | None:
    """把墙钟时刻解析成 epoch 秒（失败返回 None）。

    支持 ``"2026-11-03 04:00"`` / ``"2026-11-03 04:00:00"`` / epoch 秒（数字或数字串）。
    ⚠️ 用**本机本地时区**解释（与流水线的 `date -d` 一致）。
    """
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return float(time.mktime(time.strptime(txt, fmt)))
        except Exception:
            pass
    try:
        return float(txt)
    except Exception:
        return None


# ==============================================================================
# 3. Worker：训练 + 门禁
# ==============================================================================
def load_reference_frame(path: str | None) -> dict:
    """加载「参照系包」JSON（可选）。

    为什么需要它
    ------------
    判定链路的基线数字（``FROZEN_BASELINE`` / ``RC1_EPOCH3``）与门禁基线
    （``auto_epoch_gate_check.py`` 的 ``BASE``）都是**在 GMC 语义特征上实测**的。
    一旦换特征语义（no-GMC），这些数字全部失效 ⇒ 必须成组替换，
    否则会拿 GMC 基线判 no-GMC 模型（结构性误判）。

    包结构（缺项 = 用模块常量，不影响其余部分）::

        {
          "frozen_baseline": {"track1_f1":..,"track1_fp":..,"track2_f1":..,"track2_fp":..,
                              "track3_f1":..,"track3_fp":..},
          "rc1":             {同上},                      # 新擂主（U 的比较对象）
          "gate":            {"track2_f1_min":..,"track2_fp_max_increment":..,
                              "track1_min_delta":..},
          "gate_base":       {"track1_f1":..,"track1_tp":..,"track1_fp":.., ... }  # 传给门禁脚本
        }
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"[REF-FRAME] 参照系文件不存在：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"[REF-FRAME] 已加载参照系：{p}（键：{sorted(data.keys())}）", flush=True)
    return data


def resolve_reference(ref: dict) -> dict:
    """由参照系包解析出**全部生效参数**。

    不传参照系（``ref == {}``）⇒ 每一项都与模块常量**逐位相等**（旧行为不变）：
    ``gate_track2_fp_abs_max == 36``、``gate_track1_f1_abs_min == 0.900736245785252``。
    """
    frozen_baseline = {**FROZEN_BASELINE, **(ref.get("frozen_baseline") or {})}
    rc1_ref = {**RC1_EPOCH3, **(ref.get("rc1") or {})}
    gate_cfg = {
        "track2_f1_min": GATE_TRACK2_F1_MIN,
        "track2_fp_max_increment": GATE_TRACK2_FP_MAX_INCREMENT,
        "track1_min_delta": GATE_TRACK1_MIN_DELTA,
        **(ref.get("gate") or {}),
    }
    gate_base = ref.get("gate_base") or {}
    # 绝对阈值随参照系换算（与模块常量同语义：FP 绝对上限 = 基线 FP + 允许增量）
    return {
        "frozen_baseline": frozen_baseline,
        "rc1": rc1_ref,
        "gate": gate_cfg,
        "gate_base": gate_base,
        "gate_track2_fp_abs_max": frozen_baseline["track2_fp"] + gate_cfg["track2_fp_max_increment"],
        "gate_track1_f1_abs_min": frozen_baseline["track1_f1"] + gate_cfg["track1_min_delta"],
    }


def run_worker() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--trial-number", type=int, required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--longquan-data", required=True)
    ap.add_argument("--uav-data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gate-script", default="manu/training/auto_epoch_gate_check.py")
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-epochs", default="last", help="last=只门禁最后一个 epoch")
    ap.add_argument("--gate-epoch", type=int, default=0,
                    help="门禁量第几个 epoch 的权重（0=最后一个）。"
                         "5-epoch 训练时可用 3 去量『RC1 同款 Ep3』，"
                         "从而在同一次 5-epoch 运行里既拿到 Ep5 权重又复现 RC1/Ep3")
    ap.add_argument("--max-iters", type=int, default=0, help=">0 时每 epoch 只跑这么多 iteration（冒烟用）")
    ap.add_argument("--skip-gate", action="store_true", help="跳过门禁（冒烟用）")
    # 搜索参数
    ap.add_argument("--mix-spec", required=True)
    ap.add_argument("--lr-scale", type=float, required=True)
    ap.add_argument("--warmup-epochs", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--focal-beta", type=float, required=True)
    ap.add_argument("--offset-weight", type=float, required=True)
    ap.add_argument("--mosaic", type=float, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--translate", type=float, required=True)
    ap.add_argument("--clip-max-norm", type=float, required=True)
    ap.add_argument("--freeze-bn", type=int, required=True)
    # ---- 结构线旗标（2026-11-05）----
    ap.add_argument("--no-p2", type=int, default=0, help="1=删 neck 的 P2 融合级与 P1 320 头分支")
    # ⚠️ choices 必须含 'p2_fused'（2026-11-06 修）：p2_fused 是**主力结构**，
    #    漏了它 ⇒ 周末轮绝大部分臂 argparse 直接拒绝 ⇒ 0.1 min 全崩 ⇒ 整个周末报废。
    ap.add_argument("--head-level", default="p3", choices=["p1", "p3", "p2", "p2_fused"], help="头吃 P3(/8) 还是骨干 P2(/4)")
    ap.add_argument("--act", default="silu", choices=["silu", "relu"], help="激活函数")
    ap.add_argument("--head-conv", type=int, default=0, help=">0 时覆盖 head 的中间通道数")
    ap.add_argument("--late-upsample", type=int, default=0)
    ap.add_argument("--dw-head", type=int, default=0)
    ap.add_argument("--unfreeze-b0b4", type=int, default=0, help="1=解冻底模 b0..b4（all-relu 必须）")
    ap.add_argument("--p2-out-scale", type=float, default=1.0,
                    help="P2 融合级输出通道缩放（1.0/0.75/0.5）⇒ 直接砍 160x160 搬运量")
    # ---- 跨层高频残差注入（hf_skip，2026-11-10 方向一）----
    # 浅层高频能量以 1~2 个窄通道压到**头网格**，加性注入 head 输入（head_in 之后、feat_conv 之前）
    # ⇒ 头张量形状不变 ⇒ 旧 ckpt 的头照常可加载；project 零初始化 ⇒ 起点与原模型逐位一致。
    ap.add_argument("--hf-skip", type=int, default=0,
                    help="1=开启跨层高频残差注入侧枝（默认关 ⇒ 与原模型逐位一致）")
    ap.add_argument("--hf-skip-src", default="b1", choices=["b0", "b1"],
                    help="侧枝取数点：b1=骨干/4 的 160x160（默认，搬运量减半）/ b0=骨干/2 的 320x320")
    ap.add_argument("--hf-skip-channels", type=int, default=1, choices=[1, 2],
                    help="侧枝窄通道数（1 或 2）")
    ap.add_argument("--head-src", default="", choices=["", "p1", "p2_raw", "p2_fused", "p3"],
                    help="头吃哪一级：p1=320(现状) / p2_raw=骨干/4(已证伪) / p2_fused=结构B(160融合后) / p3=结构A(80)")
    ap.add_argument("--head-seed", type=int, default=0,
                    help="★固定**头**的随机初始化种子（0=不固定，沿用全局 RNG）。"
                         "2026-11-09：跨种子 soup 只有在「头同源」时才安全（旧轮 cs_t0043 就是"
                         "act 未变、头被继承 ⇒ 各臂头逐位相同 ⇒ soup 安全；新轮 act 变了 ⇒ 头每种子"
                         "重置 ⇒ 20 个 head 张量最大差 8.0e4 ⇒ soup 相消）。"
                         "⇒ 本参数让同家族各臂的**头初始逐位相同**，于是可以安全地做跨种子 soup。")
    ap.add_argument("--head-init", default="reset", choices=["reset", "abs", "merge", "merge_abs"],
                    help="头初始化策略（仅 act 变化时生效）：reset=随机重置（现有）；"
                         "abs=把继承来的 silu 头权重取绝对值（保住 relu 下的量级 ⇒ 有已训练的起点）；"
                         "merge_abs=★路线二用：**换宽度/换 head_src** 时先按通道合并继承（保住已学滤波器）"
                         "再取绝对值 ⇒ P1→P2 + 通道瘦身的 relu 线不再把头打回原形")
    ap.add_argument("--min-radius", type=int, default=1,
                    help="高斯目标半径下限（格）。0 = 允许单格点目标（微小目标在粗网格上必需）")
    ap.add_argument("--train-on-native-grid", type=int, default=0,
                    help="1 = 监督目标生成在头的**原生网格**上（P3+late 的原生是 80x80/stride 8），"
                         "并把模型输出的 320 复制图按 ::L 抽回原生格后再算 loss。"
                         "★2026-11-09：late_upsample 的 320 输出是原生网格的最近邻复制，"
                         "在 320 网格上监督 ⇒ 模型不可表示（8px 分辨率复现 14px 高斯）⇒ 峰被稀释 ⇒ 漏检。"
                         "原生=320 的结构（P1 头）行为逐位不变。")
    ap.add_argument("--offset-init", default="keep", choices=["keep", "reset", "inherit"],
                    help="offset 分支初始化策略：keep=照常继承；reset=只重置 offset 分支；inherit=只继承 offset 分支")
    ap.add_argument("--offset-init-weights", default="",
                    help="offset_init=inherit 时的来源 ckpt（例：cs_t0043）")
    # ---- 周末扩展（2026-10-30）：骨干保护 / 渐进式解冻 / 初始化权重标签 ----
    ap.add_argument(
        "--backbone-lr-scale", type=float, default=1.0,
        help="G2(b5..b10，可训练骨干段) 的 lr 相对倍率；0 = 该段完全冻结。"
             "默认 1.0 ⇒ 与旧版逐位相同。",
    )
    ap.add_argument(
        "--freeze-backbone-epochs", type=int, default=0,
        help="前 N 个 epoch 把 G2(b5..b10) 锁死（requires_grad=False + BN eval），"
             "第 N+1 epoch 起解冻。0 = 不冻结（与旧版逐位相同）。",
    )
    ap.add_argument(
        "--init-key", type=str, default="",
        help="初始化权重来源标签（base474 / rc2），只写进产物元数据便于审计。",
    )
    ap.add_argument(
        "--sched-epochs", type=int, default=5,
        help="cosine 退火**周期**按几个 epoch 算（默认 5，与 RC1 同口径）。"
             "0 表示跟随 --epochs。训练仍只跑 --epochs 个 epoch，"
             "但 lr 的退火曲线按 --sched-epochs 的全长走 ⇒ trial 的末 epoch 权重"
             "与 RC1/Ep3（路线 B 5 epoch 实验的第 3 个 epoch）**同 lr 轨迹**，可直接比较。",
    )
    ap.add_argument(
        "--reference-frame",
        type=str,
        default="",
        help="参照系包 JSON（可选）。换特征语义（如 no-GMC）时用："
             "里面给出 frozen_baseline / rc1 / gate 阈值 / gate_base，"
             "以免拿 GMC 语义的基线去判新语义模型。**不传 ⇒ 全部用模块常量，行为与旧版逐位一致**。",
    )
    # ---- 方向三：训练动力学靶向搜索（尺寸加权 / 学习率保底）----
    ap.add_argument(
        "--size-weight-small", type=float, default=1.0,
        help="微小目标（nano：最长边 ≤ --size-max-side-px px 或 面积 ≤ --size-thr-px px²）"
             "的 focal 正样本权重。1.0 = 不加权（默认 ⇒ 与旧版逐位一致）。",
    )
    ap.add_argument(
        "--size-weight-large", type=float, default=1.0,
        help="其余目标的 focal 正样本权重。1.0 = 不加权（默认 ⇒ 与旧版逐位一致）。",
    )
    ap.add_argument(
        "--size-thr-px", type=float, default=16.0,
        help="微小目标的面积阈值（px²，物理像素）。与 NanoBench 的 nano 桶口径一致（16.0）。",
    )
    ap.add_argument(
        "--size-max-side-px", type=float, default=4.0,
        help="微小目标的最长边阈值（px，物理像素）。与 NanoBench 的 nano 桶口径一致（4.0）。",
    )
    ap.add_argument(
        "--size-img-w", type=float, default=640.0,
        help="原始图像宽度（px），用于把归一化框换算成物理像素尺寸。",
    )
    ap.add_argument(
        "--size-img-h", type=float, default=512.0,
        help="原始图像高度（px），用于把归一化框换算成物理像素尺寸。",
    )
    ap.add_argument(
        "--lr-floor-ratio", type=float, default=0.0,
        help="学习率保底：相对各参数组峰值 lr 的下限比例（0=关闭 ⇒ 沿用绝对下限 "
             "ROUTE_B_DEFAULTS['lr_min']=1e-6，行为逐位一致）。"
             "例如 0.10 = 退火到底也保留峰值的 10% ⇒ 阻止卷积核退化为低通平滑。",
    )
    args = ap.parse_args()

    # ---- 参照系（不传 = 现有 GMC 语义常量，逐位等价旧行为）----
    ref = load_reference_frame(args.reference_frame)
    R = resolve_reference(ref)
    frozen_baseline = R["frozen_baseline"]
    rc1_ref = R["rc1"]
    gate_cfg = R["gate"]
    gate_base = R["gate_base"]
    gate_track2_fp_abs_max = R["gate_track2_fp_abs_max"]
    gate_track1_f1_abs_min = R["gate_track1_f1_abs_min"]

    trial_name = f"trial_{args.trial_number:04d}"
    save_dir = Path(args.output_root) / "trials" / trial_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mix_mode, uav_per_lq = parse_mix_spec(args.mix_spec)
    print("=" * 100, flush=True)
    print(f"[WORKER] {trial_name} | GPU {args.gpu_id} | PID {os.getpid()}", flush=True)
    print(f"[WORKER] mix_spec={args.mix_spec} (mode={mix_mode}, uav_per_lq={uav_per_lq:.3f})", flush=True)
    print(
        f"[WORKER] lr_scale={args.lr_scale:.3f} warmup={args.warmup_epochs} wd={args.weight_decay} "
        f"beta={args.focal_beta} offset={args.offset_weight} clip={args.clip_max_norm} "
        f"freeze_bn={bool(args.freeze_bn)}",
        flush=True,
    )
    if args.init_key or args.backbone_lr_scale != 1.0 or args.freeze_backbone_epochs:
        print(
            f"[WORKER] ★扩展协议：init={args.init_key or '(默认)'} | epochs={args.epochs} | "
            f"backbone_lr_scale={args.backbone_lr_scale} | "
            f"freeze_backbone_epochs={args.freeze_backbone_epochs}",
            flush=True,
        )
    print("=" * 100, flush=True)

    # ★ 固定头初始化种子（--head-seed）：必须在**建模之前**设 ⇒ 头的随机初始化逐位可复现
    if int(getattr(args, "head_seed", 0) or 0) > 0:
        torch.manual_seed(int(args.head_seed))
        np.random.seed(int(args.head_seed))
        print(f"[HEAD-SEED] 头初始化种子固定为 {args.head_seed} ⇒ 同家族各臂的头逐位相同 ⇒ 可安全 soup", flush=True)

    # ---- 模型：与底模完全同构（去 P0），并带回源 ckpt 的 p0 张量 ----
    # ⚠️ head_conv / late_upsample **必须走检测器构造器**（heatmap_model.py:759 的 `hc`），
    #    事后改模块会留下 BN 的 running_mean 通道数不匹配 ⇒
    #    "running_mean should contain 32 elements not 64"（2026-11-05 实测踩到）。
    model, model_stride = load_hm_model(
        resolve_weights(args.weights), device, stride_fallback=2, use_p0_highway=False,
        no_p2=bool(args.no_p2), head_level=args.head_level, act=args.act,
        head_conv=int(args.head_conv) if args.head_conv else None,
        late_upsample=bool(args.late_upsample),
        head_src=str(args.head_src),
        p2_out_scale=float(args.p2_out_scale),
        hf_skip=bool(getattr(args, "hf_skip", 0)),
        hf_skip_src=str(getattr(args, "hf_skip_src", "b1") or "b1"),
        hf_skip_channels=int(getattr(args, "hf_skip_channels", 1) or 1),
        offset_init=str(getattr(args, "offset_init", "keep") or "keep"),
        offset_init_weights=str(getattr(args, "offset_init_weights", "") or ""),
        head_init=str(getattr(args, "head_init", "reset") or "reset"),
    )
    if args.dw_head:
        print("[HEAD ] dw_head requested (not implemented in this path; ignored)", flush=True)
    print(f"[STRUCT] no_p2={bool(args.no_p2)} head_level={args.head_level} act={args.act} "
          f"stride={model_stride} unfreeze_b0b4={bool(args.unfreeze_b0b4)} "
          f"offset_init={getattr(args, 'offset_init', 'keep')}", flush=True)
    model.to(device)
    feat_h = feat_w = args.imgsz // model_stride
    src_ckpt = torch.load(resolve_weights(args.weights), map_location="cpu")
    src_state = src_ckpt["model"] if "model" in src_ckpt else src_ckpt.get("state_dict", src_ckpt)
    if hasattr(src_state, "state_dict"):
        src_state = src_state.state_dict()
    p0_tensors = {k: v for k, v in src_state.items() if k.startswith("p0_highway")}
    p0_kwargs = src_ckpt.get("p0_kwargs", {})

    # ---- 数据：官方 build_yolo_dataset × 2 → ConcatDataset ----
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.hsv_h = cfg.hsv_s = cfg.hsv_v = 0.0
    cfg.degrees = cfg.shear = cfg.perspective = 0.0
    cfg.translate = args.translate
    cfg.scale = args.scale
    cfg.mosaic = args.mosaic
    cfg.fliplr = 0.5
    cfg.flipud = cfg.mixup = cfg.copy_paste = 0.0

    datasets = []
    for tag, data_yaml in (("longquan", args.longquan_data), ("uav", args.uav_data)):
        data_dict = check_det_dataset(data_yaml)
        cfg.data = data_yaml
        ds = build_yolo_dataset(
            cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32
        )
        print(f"[DATA] {tag:<9} 官方 build_yolo_dataset 样本数 = {len(ds):,}", flush=True)
        datasets.append(ds)
    n_lq, n_uav = len(datasets[0]), len(datasets[1])
    if n_lq != LQ_EXPECT or n_uav != UAV_EXPECT:
        raise RuntimeError(
            f"[DATA-GATE] 样本数异常: 龙泉山 {n_lq} (期望 {LQ_EXPECT}) / "
            f"Anti-UAV {n_uav} (期望 {UAV_EXPECT}) —— 立即中止"
        )
    print(f"[DATA-GATE] PASS 满额 {n_lq:,} + {n_uav:,} = {n_lq + n_uav:,}", flush=True)

    # ---- LQ 样本判定（**不能硬编码目录名**）-------------------------------------
    # 历史 bug（2026-10-27 抓到）：这里曾硬编码 "longquanshan_ft_gmc_median_official"，
    # 换特征语义后（no-GMC 的目录名是 longquanshan_ft_gmc_median_nogmc）**永远匹配不上**
    # ⇒ n_lq_batch 恒为 0 ⇒ 日志"mix LQ/UAV 0/32" + measured_ratio 变成 ~56926 的垃圾值。
    #
    # ⚠️ **采样本身是对的**：WeightedRandomSampler 的权重是用**样本数**（n_lq/n_uav）构造的，
    # 不看路径 ⇒ LQ 样本确实按 P(LQ)=1/(1+r) 进了 batch。
    # 受影响的**只有上报字段**（日志混样列 / measured_ratio / weights 元数据），
    # **不影响**训练、权重、门禁三轨指标与 U 判定。
    #
    # ★ 历史 bug #2（2026-11-10 抓到）：修复时用了 `Path(args.longquan_data).resolve().**parent**`
    #   ⇒ 标记变成 "/mnt/data/siping/datasets/manu" 与 "manu"，
    #   而 **UAV 数据集也在同一个 datasets/manu 下** ⇒ 两个数据集的路径都命中标记
    #   ⇒ n_lq_batch 恒等于 batch_size ⇒ 日志恒为 "mix LQ/UAV 32/0"、
    #     measured_ratio = 0（与历史 bug #1 的 0/32 方向相反，症状不同、根因同源）。
    #   ⇒ **仍不影响采样**（采样只看样本数权重），但上报字段全错 ⇒ 混样口径无法自证。
    #
    # 修法：用 **数据集目录本身**（不是它的父目录）做标记。
    #   no-GMC: "longquanshan_ft_gmc_median_nogmc" ／ UAV: "uav_gmc_median_nogmc" ⇒ 互不包含。
    #   GMC 场景下目录名与旧硬编码逐位等价 ⇒ 旧行为不变。
    _lq_root = Path(args.longquan_data).resolve()
    _LQ_MARKERS = (str(_lq_root), _lq_root.name)

    def _is_lq_file(f) -> bool:
        s = str(f)
        return any(m in s for m in _LQ_MARKERS)

    print(f"[MIX-MARK] LQ 判定标记：{_LQ_MARKERS}", flush=True)

    mixed = ConcatDataset(datasets)
    mixed.collate_fn = YOLODataset.collate_fn  # 官方静态方法（collate_fn 内 batch_idx += 批内位置）

    if mix_mode == "weighted":
        # 口径 1：Batch 内采样概率 P(LQ) = 1/(1+r)，P(UAV) = r/(1+r)
        p_lq = 1.0 / (1.0 + uav_per_lq)
        p_uav = 1.0 - p_lq
        weights = np.concatenate(
            [
                np.full(n_lq, p_lq / n_lq, dtype=np.float64),
                np.full(n_uav, p_uav / n_uav, dtype=np.float64),
            ]
        )
        num_samples = n_lq + n_uav  # 保持每 epoch 总采样量不变 ⇒ 各 trial 算力与耗时可比
        with weighted_sampler_patch(weights, num_samples, seed=args.seed + args.trial_number):
            loader = build_dataloader(mixed, batch=args.batch, workers=args.workers, shuffle=True)
        print(
            f"[MIX] 口径1 加权采样：P(LQ)={p_lq:.4f} P(UAV)={p_uav:.4f} | "
            f"每 epoch 抽样 {num_samples:,} 次（LQ 期望 {num_samples*p_lq:,.0f} / UAV {num_samples*p_uav:,.0f}）",
            flush=True,
        )
    else:
        loader = build_dataloader(mixed, batch=args.batch, workers=args.workers, shuffle=True)
        print(
            f"[MIX] 对照模式：自然遍历（ConcatDataset 全量，帧比 1:{n_uav/n_lq:.3f}）",
            flush=True,
        )

    # ★ 监督网格：默认与旧版逐位一致（320 网格 / stride=model_stride / min_radius=1）；
    #   --train-on-native-grid 1 ⇒ 监督落到头的原生网格（late_upsample: 80x80 / stride 8）
    native_scale = 1
    if bool(getattr(args, "train_on_native_grid", 0)) and bool(args.late_upsample):
        native_scale = int(getattr(getattr(model, "head", None), "late_scale", 1) or 1)
    loss_feat_h = feat_h // native_scale
    loss_feat_w = feat_w // native_scale
    loss_stride = model_stride * native_scale
    if native_scale > 1:
        print(f"[TARGET-GRID] ★监督落到原生网格：feat=({loss_feat_h},{loss_feat_w}) stride={loss_stride} "
              f"min_radius={args.min_radius}（原生=320 的结构行为逐位不变）", flush=True)
    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * args.epochs
    print(f"[DATA] iters/epoch = {iters_per_epoch} | 总优化步数 = {total_steps}", flush=True)

    # ---- 参数组：G1 严格冻结，G2/G3/G4 按路线 B 比例 × lr_scale ----
    buckets: dict[str, list[torch.nn.Parameter]] = {k: [] for k in GROUP_LRS}
    frozen: list[torch.nn.Parameter] = []
    unmatched: list[str] = []
    g2_params: list[torch.nn.Parameter] = []
    unfreeze_b0b4 = bool(args.unfreeze_b0b4)
    for name, param in model.named_parameters():
        if not unfreeze_b0b4 and any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
            param.requires_grad_(False)
            frozen.append(param)
            continue
        if name.startswith("head"):
            key = "G4_head"
        elif name.startswith("b5") or name.startswith("b6") or name.startswith("b7") or name.startswith("b8") \
                or name.startswith("b9") or name.startswith("b10"):
            key = "G2_p4p5"
        elif unfreeze_b0b4 and any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
            # all-relu 改了底模函数 ⇒ 必须一起训练，否则永久退化（仅 74,848 参数）
            key = "G2_p4p5"
        else:
            key = "G3_neck"
        param.requires_grad_(True)
        buckets[key].append(param)
    if unmatched:
        raise RuntimeError(f"[PARAM-GATE] 有参数未归组: {unmatched[:10]}")
    groups = []
    for key, base_lr in GROUP_LRS.items():
        params = buckets[key]
        if not params:
            continue
        lr = base_lr * args.lr_scale
        if key == "G2_p4p5":
            # 骨干段 lr 相对倍率（周末扩展）：1.0 ⇒ 与旧版逐位相同
            lr = lr * args.backbone_lr_scale
            g2_params = params
        n = sum(p.numel() for p in params)
        groups.append({"params": params, "lr": lr, "base_lr": lr, "name": key, "numel": n})
        print(f"[PARAM] {key:<9} lr={lr:.3e} (基准 {base_lr:.1e} × {args.lr_scale:.3f}"
              + (f" × 骨干倍率 {args.backbone_lr_scale}" if key == "G2_p4p5" else "")
              + f")  参数 {n:,}", flush=True)
    print(f"[PARAM] G1_frozen 参数 {sum(p.numel() for p in frozen):,}（严格冻结）", flush=True)

    # ---- 骨干段保护（周末扩展，2026-10-30）--------------------------------
    # 目的：GMC RC2（trial_0080）的骨干特征核是"弱小目标表征上限"的载体，
    # 未对齐残差（满屏地物错位白边）带来的巨大梯度会把它们冲烂 ⇒ 灾难性遗忘。
    #   backbone_lr_scale == 0            ⇒ G2(b5..b10) 整段冻结（含 BN 统计量）
    #   freeze_backbone_epochs = N > 0    ⇒ 前 N 个 epoch 冻结 G2，第 N+1 epoch 起解冻
    g2_frozen_forever = (args.backbone_lr_scale == 0.0)
    g2_frozen_epochs = max(0, int(args.freeze_backbone_epochs))
    if g2_params and (g2_frozen_forever or g2_frozen_epochs > 0):
        for p in g2_params:
            p.requires_grad_(False)
        print(
            f"[PARAM] G2 骨干段保护：backbone_lr_scale={args.backbone_lr_scale} "
            f"freeze_backbone_epochs={g2_frozen_epochs} ⇒ 前 "
            f"{'全程' if g2_frozen_forever else g2_frozen_epochs} 个 epoch 冻结 "
            f"（{sum(p.numel() for p in g2_params):,} 参数，BN 统计量一并冻结）",
            flush=True,
        )

    optimizer = torch.optim.AdamW(groups, lr=groups[0]["lr"], weight_decay=args.weight_decay)
    for pg, g in zip(optimizer.param_groups, groups):
        pg["lr"] = g["base_lr"]
        pg["initial_lr"] = g["base_lr"]

    # 调度：warmup（按单 epoch 步数算）→ cosine → 绝对下限
    #
    # ⚠️ **cosine 周期与训练总步数解耦**：
    #    RC1/Ep3 是"路线 B 跑 5 epoch"里的第 3 个 epoch 权重，它经历的是
    #    **5 epoch 全长的 cosine 曲线**（总步 8895），在 Ep3 结束时 lr 仍处高位
    #    （G2 ≈ 1.24e-6 = 峰值的 41%）。
    #    若搜索按 3 epoch 自己的 cosine（总步 5337），Ep3 结束时 lr 已退到绝对下限
    #    （G2 = 1.00e-6 = 峰值的 3.3%）⇒ **每个 trial 的末 epoch 权重都系统性地
    #    "比 RC1 学得少一点"**，与 mix_spec 的真实效应混在一起，无法归因。
    #    ⇒ 统一用 --sched-epochs（默认 5）作为 cosine 周期，使 natural 档成为
    #    **RC1 的精确复现**（一个强 sanity check），其余档位与 RC1 直接可比。
    sched_epochs = args.sched_epochs if args.sched_epochs > 0 else args.epochs
    sched_total_steps = sched_epochs * iters_per_epoch
    warmup_steps = max(1, int(round(args.warmup_epochs * iters_per_epoch)))
    warmup_steps = min(warmup_steps, max(1, sched_total_steps - 1))

    def make_lambda(base_lr: float):
        floor_ratio = min(1.0, ROUTE_B_DEFAULTS["lr_min"] / base_lr) if base_lr > 0 else 0.0
        # ★ 学习率保底（方向三）：相对**峰值 lr** 的比例下限。
        #   0 = 关闭 ⇒ 与历史逐位一致（沿用绝对下限 lr_min）。
        if args.lr_floor_ratio > 0.0:
            floor_ratio = max(floor_ratio, min(1.0, float(args.lr_floor_ratio)))

        def fn(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, sched_total_steps - warmup_steps)))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos

        return fn

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[make_lambda(g["base_lr"]) for g in optimizer.param_groups]
    )
    print(f"[SCHED] 训练 {args.epochs} ep（{total_steps} 步）| warmup {warmup_steps} 步"
          f"（{args.warmup_epochs} ep）| cosine 周期 {sched_epochs} ep（{sched_total_steps} 步）"
          f" → {ROUTE_B_DEFAULTS['lr_min']:.0e}"
          f"{'  ← 与 RC1 同口径' if sched_epochs != args.epochs else ''}"
          f"{'  | ★LR 保底 = 峰值的 %.1f%%' % (args.lr_floor_ratio * 100) if args.lr_floor_ratio > 0 else ''}",
          flush=True)

    criterion = HeatmapLoss(
        hm_weight=1.0, offset_weight=args.offset_weight, focal_alpha=2.0, focal_beta=args.focal_beta
    )
    scaler = GradScaler(enabled=True)

    csv_path = save_dir / "logs" / "train_log.csv"
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_file.write(
        "epoch,iter,loss_total,loss_hm,loss_offset,"
        + ",".join(f"lr_{g['name']}" for g in groups)
        + ",batch_lq,batch_uav\n"
    )
    csv_file.flush()

    def set_frozen_eval(epoch: int = 0) -> None:
        model.train()
        for name, module in model.named_modules():
            if any(name == g or name.startswith(g + ".") for g in FROZEN_BACKBONE):
                module.eval()
            if args.freeze_bn and name.endswith("bn"):
                module.eval()  # 冻结 BN running stats（用户点名的下一轮方向）
        # 骨干段（G2: b5..b10）在被冻结期间强制 eval ⇒ 保护它的 BN running stats
        # （§十七 教训：BN 漂移毁掉底模）
        g2_is_frozen = g2_frozen_forever or (g2_frozen_epochs > 0 and epoch <= g2_frozen_epochs)
        if g2_is_frozen:
            for name, module in model.named_modules():
                if any(name == g or name.startswith(g + ".") for g in G2_MODULES):
                    module.eval()

    global_step = 0
    vram_peak = 0.0
    t_start = time.time()
    last_weights = None
    for epoch in range(1, args.epochs + 1):
        # ---- 渐进式解冻：第 (freeze_backbone_epochs + 1) 个 epoch 起放开 G2 ----
        # 目的：先让检测头（Head）单独适应"满屏错位地物白边"的假警抑制，
        # 待 Head 稳定后再以微小 lr 放开骨干 ⇒ 杜绝灾难性遗忘。
        if (not g2_frozen_forever) and g2_frozen_epochs > 0 \
                and epoch == g2_frozen_epochs + 1 and g2_params:
            for p in g2_params:
                p.requires_grad_(True)
            print(
                f"[UNFREEZE] epoch {epoch}：解冻 G2 骨干段（{sum(p.numel() for p in g2_params):,} 参数）"
                f" ⇒ 其后按 backbone_lr_scale={args.backbone_lr_scale} 更新",
                flush=True,
            )
        set_frozen_eval(epoch)
        ep_loss = ep_hm = ep_off = 0.0
        nb = 0
        lq_seen = uav_seen = 0
        t_ep = time.time()
        for it, batch in enumerate(loader, start=1):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            b_idx = batch["batch_idx"].to(device, non_blocking=True)
            bs = imgs.shape[0]
            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes, batch_idx=b_idx, batch_size=bs,
                feat_shape=(loss_feat_h, loss_feat_w), stride=loss_stride,
                min_radius=args.min_radius, device=device,
                size_weight_small=args.size_weight_small,
                size_weight_large=args.size_weight_large,
                size_thr_px=args.size_thr_px,
                size_max_side_px=args.size_max_side_px,
                img_w=args.size_img_w,
                img_h=args.size_img_h,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                preds = model(imgs)
                loss_preds = preds
                if native_scale > 1:
                    # ★ 把 320 复制图按 ::L 抽回原生网格（最近邻复制 ⇒ 逐位精确）
                    #   ⇒ loss 直接作用在头的原生输出上（梯度只到原生头）
                    loss_preds = dict(preds)
                    loss_preds["heatmap"] = preds["heatmap"][:, :, ::native_scale, ::native_scale].contiguous()
                    loss_preds["offset"] = preds["offset"][:, :, ::native_scale, ::native_scale].contiguous()
                loss, loss_items = criterion(loss_preds, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], max_norm=args.clip_max_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            ep_loss += loss_items["loss_total"]
            ep_hm += loss_items["loss_hm"]
            ep_off += loss_items["loss_offset"]
            nb += 1
            if device.type == "cuda":
                vram_peak = max(vram_peak, torch.cuda.max_memory_allocated(device) / 2**30)
            im_files = batch.get("im_file", []) or []
            n_lq_batch = sum(1 for f in im_files if _is_lq_file(f))
            lq_seen += n_lq_batch
            uav_seen += int(bs - n_lq_batch)
            if it % args.log_every == 0 or it == 1:
                csv_file.write(
                    f"{epoch},{it},{loss_items['loss_total']:.6f},{loss_items['loss_hm']:.6f},"
                    f"{loss_items['loss_offset']:.6f},"
                    + ",".join(f"{optimizer.param_groups[i]['lr']:.10e}" for i in range(len(groups)))
                    + f",{n_lq_batch},{int(bs - n_lq_batch)}\n"
                )
                csv_file.flush()
                print(
                    f"[Ep{epoch:02d} it{it:5d}/{iters_per_epoch}] loss {loss_items['loss_total']:.4f} "
                    f"(hm {loss_items['loss_hm']:.4f}/off {loss_items['loss_offset']:.4f}) | "
                    f"mix LQ/UAV {n_lq_batch}/{int(bs-n_lq_batch)} | vram {vram_peak:.2f}GB | "
                    f"{time.time()-t_start:.0f}s",
                    flush=True,
                )
            if args.max_iters and it >= args.max_iters:
                print(f"[SMOKE] 达到 max-iters={args.max_iters}，提前结束本 epoch", flush=True)
                break
        csv_file.write(
            f"{epoch},END,{ep_loss/max(nb,1):.6f},{ep_hm/max(nb,1):.6f},{ep_off/max(nb,1):.6f},"
            + ",".join(f"{optimizer.param_groups[i]['lr']:.10e}" for i in range(len(groups)))
            + f",{lq_seen},{uav_seen}\n"
        )
        csv_file.flush()
        ratio = uav_seen / max(lq_seen, 1)
        print(
            f"[EPOCH {epoch}] loss {ep_loss/max(nb,1):.4f} | 实测混样 LQ {lq_seen:,} / UAV {uav_seen:,} "
            f"= 1:{ratio:.3f}（目标 1:{uav_per_lq:.3f}） | {time.time()-t_ep:.1f}s",
            flush=True,
        )
        state = dict(model.state_dict())
        state.update(p0_tensors)
        last_weights = weights_dir / f"epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": state,
                "metrics": {"train_loss": ep_loss / max(nb, 1)},
                "stride": model_stride,
                # ---- 结构线（2026-11-05）：ckpt 自描述 ⇒ 门禁/缓存/导出都能重建同一张图 ----
                "no_p2": bool(args.no_p2),
                "head_level": str(args.head_level),
                "act": str(args.act),
                "head_conv": int(args.head_conv) if args.head_conv else 64,
                "late_upsample": bool(args.late_upsample),
                "head_src": str(args.head_src),
                "p2_out_scale": float(args.p2_out_scale),
                "hf_skip": bool(getattr(args, "hf_skip", 0)),
                "hf_skip_src": str(getattr(args, "hf_skip_src", "b1") or "b1"),
                "hf_skip_channels": int(getattr(args, "hf_skip_channels", 1) or 1),
                "imgsz": args.imgsz,
                "p0_kwargs": p0_kwargs,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "finetune": {
                    "route": "mix_ratio_search",
                    "mix_spec": args.mix_spec,
                    "mix_mode": mix_mode,
                    "uav_per_lq_target": uav_per_lq,
                    "measured_lq": lq_seen,
                    "measured_uav": uav_seen,
                    "measured_ratio": ratio,
                    "groups": {g["name"]: g["base_lr"] for g in groups},
                    "freeze_bn": bool(args.freeze_bn),
                    "sched_epochs": sched_epochs,
                    "sched_total_steps": sched_total_steps,
                    "train_total_steps": total_steps,
                },
            },
            last_weights,
        )
        print(f"[SAVE] {last_weights}", flush=True)
    csv_file.close()

    # ---- 门禁：默认只门禁最后一个 epoch（保证吞吐；**每个 epoch 的权重全部保留**）----
    # ⚠️ `--gate-epoch N` 可改为门禁**第 N 个 epoch** 的权重：
    #    训练 5 epoch 时，Ep3 的权重**就是 RC1/Ep3 的等价物**（同 LR 轨迹），
    #    Ep5 的权重**就是 RC1/Ep5 的等价物** ⇒ 一次运行同时覆盖两个参照系。
    gate_epochs = [args.epochs] if args.gate_epochs == "last" else list(range(1, args.epochs + 1))
    gate_epoch = args.gate_epoch if args.gate_epoch > 0 else args.epochs
    if gate_epoch < 1 or gate_epoch > args.epochs:
        raise RuntimeError(f"[GATE-ERR] --gate-epoch {gate_epoch} 越界（训练了 {args.epochs} epoch）")
    gate_weights = weights_dir / f"epoch{gate_epoch}.pt"
    if not gate_weights.is_file():
        raise RuntimeError(f"[GATE-ERR] 门禁权重不存在：{gate_weights}")
    gate_dir = save_dir / "gate" / f"epoch{gate_epoch}"
    gate_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, args.gate_script,
        "--weights", str(gate_weights),
        "--out-dir", str(gate_dir),
        "--gpus", "1",
        "--device", str(args.gpu_id),
        "--infer-gpu", str(args.gpu_id),
        "--cache-gpus", str(args.gpu_id),
        "--baseline-cache", args.baseline_cache,
        "--min-track2-f1", str(gate_cfg["track2_f1_min"]),
        # 绝对上限语义：= 基线 FP + 增量（等价于我们自己的 ΔFP ≤ +10）
        "--max-track2-fp", str(gate_track2_fp_abs_max),
        # 绝对 F1 下限 = 基线 F1 + Δ 下限（与我们的 ΔF1 红线等价）
        "--min-track1-f1", str(gate_track1_f1_abs_min),
        # ---- 特征根目录随本 trial 的数据集走（不再用门禁脚本里的硬编码常量）----
        "--uav-data", args.uav_data,
        "--ft-root", str(Path(args.longquan_data).parent),
    ]
    # ---- 门禁基线数字随参照系走（no-GMC 等换语义时必须成组替换）----
    for _k, _flag in (
        ("track1_f1", "--base-track1-f1"), ("track1_tp", "--base-track1-tp"), ("track1_fp", "--base-track1-fp"),
        ("track2_f1", "--base-track2-f1"), ("track2_tp", "--base-track2-tp"),
        ("track2_fp", "--base-track2-fp"), ("track2_fn", "--base-track2-fn"),
        ("track3_f1", "--base-track3-f1"), ("track3_tp", "--base-track3-tp"),
        ("track3_fp", "--base-track3-fp"), ("track3_fn", "--base-track3-fn"),
    ):
        if _k in gate_base:
            cmd += [_flag, str(gate_base[_k])]
    print(f"\n[GATE] 三轨门禁（{args.mix_spec} / epoch{gate_epoch}"
          f"{'（末 epoch）' if gate_epoch == args.epochs else f'（训练了 {args.epochs} epoch）'}）…"
          f"\n$ {' '.join(cmd)}\n", flush=True)
    if args.skip_gate:
        print("[SMOKE] --skip-gate，跳过门禁", flush=True)
        cmd = ["true"]
    t_gate = time.time()
    # ⚠️⚠️ **2026-11-02 关键修复：调门禁前必须把训练模型从显存里放掉**。
    #    门禁的龙泉山热图缓存重建（multiscale_heatmap_cache.py）是**独立子进程**，
    #    且跑在**本 trial 自己的那张卡**上（--cache-gpus = 本卡）；而训练用的模型
    #    此时仍驻留在同一张卡上（约 11 GB）⇒ 11 GB + 缓存峰值 ⇒ **CUDA OOM**
    #    （2026-10-30 傍晚实测炸掉 3 条 trial 的门禁：trial_0006 / 0011 / 0068）。
    #    ⇒ 这里显式把模型搬回 CPU 并释放引用 + 清空分配器，把整张卡让给门禁子进程。
    #    （`model` 在权重落盘后不再被使用，此处 del 是安全的。）
    try:
        model.to("cpu")
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            free_gib = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            torch.cuda.empty_cache()
        else:
            free_gib = float("nan")
        print(f"[GATE-MEM] 已释放训练模型显存（free={free_gib:.1f} GiB）", flush=True)
    except Exception as _e:
        print(f"[GATE-MEM] 释放显存失败（忽略，不影响门禁）：{_e}", flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    print(f"[GATE] rc={rc} 用时 {time.time()-t_gate:.1f}s", flush=True)

    result = {
        "trial": args.trial_number,
        "trial_name": trial_name,
        "mix_spec": args.mix_spec,
        "mix_mode": mix_mode,
        "uav_per_lq_target": uav_per_lq,
        "measured_ratio": ratio,
        "measured_lq": lq_seen,
        "measured_uav": uav_seen,
        "gate_rc": rc,
        "gate_dir": str(gate_dir),
        "train_epochs": args.epochs,
        "gate_epoch": gate_epoch,
        "sched_epochs": sched_epochs,
        "size_weight_small": args.size_weight_small,
        "size_weight_large": args.size_weight_large,
        "size_thr_px": args.size_thr_px,
        "size_max_side_px": args.size_max_side_px,
        "lr_floor_ratio": args.lr_floor_ratio,
        "init_key": args.init_key,
        "init_weights": str(resolve_weights(args.weights)),
        "backbone_lr_scale": args.backbone_lr_scale,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "weights": str(gate_weights),
        "last_weights": str(last_weights),
        "train_loss": ep_loss / max(nb, 1),
        "vram_peak_gb": vram_peak,
        "elapsed_s": time.time() - t_start,
    }
    rep_path = gate_dir / "gate_report.json"
    if rep_path.is_file():
        try:
            metrics = metrics_from_gate_report(rep_path)
        except Exception as exc:
            # ⚠️ 门禁**基础设施失败**（最常见：Track1 推理 CUDA OOM ⇒ 报告缺字段）。
            # 绝不能让异常冒出去把整个 worker 打死 —— 那样会丢掉本 trial 的 27 分钟训练
            # 且 trial_result.json 缺失 ⇒ 无任何诊断信息（实测踩过：trial_0003 因此变成
            # CRASHED，根因要从 track1_infer.log 里翻半天）。
            # 这里捕获后写入**结构化诊断**，调度器据此记为 FAIL（而非"触红线被剪枝"）。
            result["metrics"] = None
            result["utility"] = None
            result["gate_verdict"] = "GATE_REPORT_INCOMPLETE"
            result["tier1_ok"] = False
            result["tier1_reason"] = "GATE_REPORT_INCOMPLETE"
            result["gate_error"] = f"{type(exc).__name__}: {exc}"
            result["infrastructure_failure"] = True
            print(f"[GATE-ERR] 门禁报告不完整 ⇒ 记为基础设施失败：{result['gate_error']}", flush=True)
            print(f"[GATE-ERR] 见 {gate_dir}/track1_infer.log 等子日志", flush=True)
            metrics = None
        if metrics is not None:
            score, reason, detail = evaluate_model_quality(
                metrics, frozen_baseline,
                track2_f1_min=gate_cfg["track2_f1_min"],
                track2_fp_max_increment=gate_cfg["track2_fp_max_increment"],
                track1_min_delta=gate_cfg["track1_min_delta"],
                alpha=UTILITY_ALPHA, beta=UTILITY_BETA, gamma=UTILITY_GAMMA,
            )
            ok, greason, gdetail = hard_gate(
                metrics, frozen_baseline,
                track2_f1_min=gate_cfg["track2_f1_min"],
                track2_fp_max_increment=gate_cfg["track2_fp_max_increment"],
                track1_min_delta=gate_cfg["track1_min_delta"],
            )
            result["metrics"] = metrics
            result["utility"] = score if ok else None
            result["gate_verdict"] = reason
            result["gate_detail"] = detail
            result["tier1_ok"] = ok
            result["tier1_reason"] = greason
            u_rc1, _ = utility(metrics, rc1_ref)
            result["utility_vs_rc1"] = u_rc1
            print(f"[JUDGE] {reason} | U(vs 冻结基线)={score if ok else float('-inf')} | U(vs RC1)={u_rc1:+.6f}", flush=True)
    else:
        result["metrics"] = None
        result["utility"] = None
        result["gate_verdict"] = "NO_REPORT"
        result["tier1_ok"] = False
        result["tier1_reason"] = "NO_REPORT"
        result["infrastructure_failure"] = True
        result["gate_error"] = "门禁未产出 gate_report.json（见 gate 目录下的子日志）"

    (save_dir / "trial_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[RESULT] {save_dir / 'trial_result.json'}", flush=True)
    return 0 if result.get("tier1_ok") else 0  # 一级红线失败不视为进程失败（由调度器剪枝）


# ==============================================================================
# 4. 调度器
# ==============================================================================
def read_trial_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_summary_row(csv_path: Path, trial_name: str, params: dict, res: dict | None, status: str,
                      duration_min: float) -> None:
    header = (
        "trial,status,mix_spec,mix_mode,target_ratio,measured_ratio,gate_verdict,"
        "t1_f1,t1_fp,t2_f1,t2_fp,t3_f1,t3_fp,utility,utility_vs_rc1,"
        "lr_scale,warmup_epochs,weight_decay,focal_beta,offset_weight,mosaic,scale,translate,"
        "clip_max_norm,freeze_bn,"
        "init_from,epochs,backbone_lr_scale,freeze_backbone_epochs,duration_min\n"
    )
    if csv_path.exists():
        # ⚠️ 列头变了**绝不能**直接往老文件追加（会整体错位 ⇒ 读盘时是"神秘"的解析错误）。
        #    老文件改名让位，重新开一份；这样新旧两套表各自自洽。
        try:
            first = csv_path.open("r", encoding="utf-8").readline().strip()
        except Exception:
            first = ""
        if first and first != header.strip():
            import time as _t
            bak = csv_path.with_suffix(csv_path.suffix + f".bak_{int(_t.time())}")
            csv_path.rename(bak)
            print(f"[CSV] 列头变化 ⇒ 老表改名让位：{bak}", flush=True)
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header)
    m = (res or {}).get("metrics") or {}

    def g(key, default=""):
        v = m.get(key, default)
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    def _f3(key, default=0.0):
        try:
            return float((res or {}).get(key, default))
        except Exception:
            return default

    def _p(key, default=""):
        v = params.get(key, default)
        if v is None:
            return default
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
        return str(v)

    row = (
        f"{trial_name},{status},{params.get('mix_spec')},{(res or {}).get('mix_mode','')},"
        f"{_f3('uav_per_lq_target'):.3f},{_f3('measured_ratio'):.3f},"
        f"{(res or {}).get('gate_verdict','')},"
        f"{g('track1_f1')},{m.get('track1_fp','')},{g('track2_f1')},{m.get('track2_fp','')},"
        f"{g('track3_f1')},{m.get('track3_fp','')},"
        f"{(res or {}).get('utility', '')},{(res or {}).get('utility_vs_rc1', '')},"
        f"{params.get('lr_scale'):.4f},{params.get('warmup_epochs'):.3f},{params.get('weight_decay'):.8f},"
        f"{params.get('focal_beta'):.2f},{params.get('offset_weight'):.2f},{params.get('mosaic'):.2f},"
        f"{params.get('scale'):.2f},{params.get('translate'):.2f},{params.get('clip_max_norm'):.2f},"
        f"{params.get('freeze_bn')},"
        f"{_p('init_from')},{_p('epochs')},{_p('backbone_lr_scale')},{_p('freeze_backbone_epochs')},"
        f"{duration_min:.2f}\n"
    )
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(row)


def main() -> int:
    if "--worker" in sys.argv:
        return run_worker()

    ap = argparse.ArgumentParser(description="Optuna 混合比例超参搜索（4 卡并行 / 可断点续跑）")
    ap.add_argument("--longquan-data", default="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml")
    ap.add_argument("--uav-data", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    ap.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    ap.add_argument("--baseline-cache", default="runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl")
    ap.add_argument("--gate-script", default="manu/training/auto_epoch_gate_check.py")
    ap.add_argument("--output-root", default="runs/optuna_mix_ratio")
    ap.add_argument("--study-name", default="mix_ratio_search")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--n-trials", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate-epochs", default="last")
    ap.add_argument("--max-hours", type=float, default=20.0, help="到点自动收尾（不再派新 trial）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-seed-trials", action="store_true", help="不注入种子 trial")
    ap.add_argument(
        "--reference-frame",
        type=str,
        default="",
        help="参照系包 JSON（透传给每个 worker）。换特征语义（如 no-GMC）时必传；不传 = 现有 GMC 语义。",
    )
    ap.add_argument(
        "--focus-space",
        action="store_true",
        help="**定点打靶模式**（2026-11-02）：把搜索空间每一维收窄到贴着已知好点的区间/取值"
             "（mix ∈ {w2.0,w3.0,w6.0,w8.0,natural}；lr ∈ [0.8,3.0]；ep ∈ {6,8,10,15}；"
             "骨干倍率 ∈ {0,0.1,0.3,1.0}；锁骨干 ∈ {0,2}）⇒ 算力换成确定性最高的收敛收益。"
             "⚠️ **必须用新 study-name**（categorical 取值集合被 study 记录后不可改）。",
    )
    ap.add_argument(
        "--speed-table",
        default="",
        help="结构→板卡速度表 JSON（probe_struct_speed.py 产出）。启用后：派单前查表，"
             "预测 > --speed-limit-ms 的臂直接剪枝，不浪费训练",
    )
    ap.add_argument("--speed-limit-ms", type=float, default=21.0,
                    help="速度硬约束（板卡 rknn_run 上限，ms）")
    ap.add_argument(
        "--struct-space",
        action="store_true",
        help="启用结构搜索空间（no_p2/head_level/act/head_conv/late_upsample/unfreeze_b0b4）。"
             "⚠️ 必须配合**新 study-name**（categorical 取值集合被 study 记录后不可改）",
    )
    ap.add_argument(
        "--extended-space",
        action="store_true",
        help="启用周末扩展搜索空间（+init_from / +epochs / +backbone_lr_scale / +freeze_backbone_epochs）。"
             "**不传 ⇒ 搜索空间与旧版逐位相同**（GMC 侧既有 study 不受污染）。",
    )
    # ---- 搜索空间覆盖（2026-11-03）：让**新 study** 能用比现有 study 更宽的空间 ----
    # ⚠️ 默认值全为空 ⇒ **行为逐位不变**（GMC 侧与既有 no-GMC study 的空间不受污染）。
    # ⚠️ 覆盖只能在**新 study-name** 上用：Optuna 的分布兼容性检查是 study 级的，
    #    同一 study 里换 choices / 改 Float 范围 ⇒ 之后所有新 trial 报 ValueError。
    ap.add_argument(
        "--lr-scale-low",
        type=float,
        default=None,
        help="覆盖聚焦模式的 lr_scale 下界（默认 = FOCUS_LR_SCALE 的 0.8）。**必须 > 0**"
             "（Optuna log 采样要求）。不传 ⇒ 不覆盖。",
    )
    ap.add_argument(
        "--lr-scale-high",
        type=float,
        default=None,
        help="覆盖聚焦模式的 lr_scale 上界（默认 = FOCUS_LR_SCALE 的 3.0）。不传 ⇒ 不覆盖。",
    )
    ap.add_argument(
        "--epoch-choices",
        type=str,
        default="",
        help="逗号分隔的 epoch 取值，覆盖 FOCUS_CHOICES['epochs']（同时同步 EPOCH_CHOICES）。"
             "例：--epoch-choices 6,8,10,15,20,25。不传 ⇒ 不覆盖。",
    )
    ap.add_argument(
        "--mix-choices",
        type=str,
        default="",
        help="逗号分隔的 mix_spec 取值，覆盖 FOCUS_CHOICES['mix_spec']（同时同步 MIX_SPECS）。"
             "例：--mix-choices natural,w2.0,w2.5,w3.0,w6.0。不传 ⇒ 不覆盖。",
    )
    ap.add_argument(
        "--init-choices",
        type=str,
        default="",
        help="逗号分隔的 init_from 取值，覆盖 FOCUS_CHOICES['init_from']（同时同步 INIT_CHOICES）。"
             "例：--init-choices rc2,base474。不传 ⇒ 不覆盖。",
    )
    ap.add_argument(
        "--init-x-choices",
        type=str,
        default="",
        help="逗号分隔的 init_from_x 取值，覆盖 INIT_X_CHOICES（soup 融合权重等）。"
             "例：--init-x-choices rc2,soup_strict3。不传 ⇒ 不覆盖。",
    )
    ap.add_argument(
        "--init-weights-map",
        type=str,
        default="",
        help='JSON 文件：{"base474": "<权重路径>", "rc2": "<权重路径>"}。'
             "扩展空间下按每个 trial 的 init_from 取对应权重；不传则所有 trial 都用 --weights。",
    )
    ap.add_argument(
        "--enqueue-plan",
        type=str,
        default="",
        help="JSON 文件：确定性对照臂清单（每项是一组**完整**超参）。启动时按顺序入队，"
             "保证指定的臂一定被跑（不依赖 TPE 采样运气）。",
    )
    args = ap.parse_args()

    global EXTENDED_SPACE, FOCUS_SPACE, STRUCT_SPACE
    key2ms, pred = load_speed_table(args.speed_table)
    EXTENDED_SPACE = bool(args.extended_space)
    STRUCT_SPACE = bool(args.struct_space)
    FOCUS_SPACE = bool(args.focus_space)
    if FOCUS_SPACE:
        EXTENDED_SPACE = True   # 聚焦模式隐含扩展空间（要用 init/epochs/骨干 4 维）
        print("[FOCUS] 定点打靶模式已启用：搜索空间收窄到已知好点附近", flush=True)

    # ---- 搜索空间覆盖（**必须在 create_study 之前**）----
    # ⚠️ 顺序是硬要求：study 一旦建好并记录了某个参数名的分布，之后再改空间就会触发
    #    Optuna 的 **study 级**分布兼容性报错（CategoricalDistribution 不支持动态取值空间）
    #    ⇒ 该 study 之后所有新 trial 全炸。先覆盖、再建 study ⇒ 空间从头就是新的。
    space_snapshot = apply_space_overrides(args)

    # ---- 初始化权重映射（扩展空间用）----
    init_map: dict[str, str] = {}
    if args.init_weights_map:
        init_map = json.loads(Path(args.init_weights_map).read_text(encoding="utf-8"))
        print(f"[INIT-MAP] {json.dumps(init_map, ensure_ascii=False)}", flush=True)
        for _k, _v in list(init_map.items()):
            if not isinstance(_v, str):
                # 允许 JSON 里放说明性字段（如 "_comment": [...]）⇒ 非字符串一律跳过
                continue
            _p = Path(_v)
            if not _p.is_absolute():
                _p = REPO_ROOT / _p
            if not _p.is_file():
                raise FileNotFoundError(f"初始化权重不存在：{_k} → {_p}")
            init_map[_k] = str(_p)  # 归一成绝对路径 ⇒ worker 直接可用
        print(f"[INIT-MAP] 归一后 {json.dumps(init_map, ensure_ascii=False)}", flush=True)
    if EXTENDED_SPACE and not init_map:
        raise RuntimeError("[INIT-MAP] 扩展空间必须提供 --init-weights-map（每个 init_from 都要有对应权重）")

    out_root = Path(args.output_root)
    (out_root / "trials").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    # 调度器自写 PID 文件（比 shell 的 $! 可靠：setsid 在某些情况下会 fork，$! 可能是父进程 PID）
    (out_root / "dispatcher.pid").write_text(str(os.getpid()), encoding="utf-8")
    summary_csv = out_root / "optuna_summary.csv"
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not gpu_list:
        raise ValueError(f"没有合法 GPU：{args.gpus}")

    print("=" * 100, flush=True)
    print("🚀 Optuna 混合比例超参搜索", flush=True)
    print(f"   底模        : {args.weights}", flush=True)
    print(f"   数据       : LQ {args.longquan_data}", flush=True)
    print(f"                UAV {args.uav_data}", flush=True)
    print(f"   GPU        : {gpu_list}（{len(gpu_list)} 并行，每 trial 单卡）", flush=True)
    print(f"   Trial/Epoch: {args.n_trials} / {args.epochs}", flush=True)
    _eff = space_snapshot["effective"]
    print(f"   空间       : mix={_eff['mix_spec']} lr=({_eff['lr_scale'][0]},{_eff['lr_scale'][1]}) "
          f"epochs={_eff['epochs']}", flush=True)
    print(f"   截止       : {args.max_hours} 小时后自动收尾", flush=True)
    print(f"   输出       : {out_root.resolve()}", flush=True)
    print("=" * 100 + "\n", flush=True)

    db_path = out_root / "study.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path.resolve()}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=12, multivariate=True),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 陈旧 RUNNING trial 自愈（进程被断电/杀死后残留）
    #
    # ⚠️ 必须用公共 API study.tell(trial_number, state=FAIL)，**不能**直接调
    #    study._storage.set_trial_state_values(t.number, ...)。
    # 原因（两层坑，都是实测复现）：
    #    1. Optuna 的 RDB 存储里 trial_id 是 **1 起**的自增主键，而 Trial.number 是
    #       **0 起**的业务编号，两者差 1；set_trial_state_values() 收的是前者。
    #       传 number 会改错行，对已完成的 trial 直接抛 UpdateFinishedTrialError
    #       ⇒ 断点续跑时 dispatcher 直接崩溃。
    #    2. study.get_trials(deepcopy=False) 返回的是 **FrozenTrial**，不是 Trial，
    #       传给 study.tell(trial_obj) 会抛 "Trial must be a trial object or trial number"。
    #    study.tell(number) 两个坑都避开：内部按业务编号取回正确 trial 并用其 _trial_id。
    healed = 0
    for t in study.get_trials(deepcopy=False):
        if t.state == optuna.trial.TrialState.RUNNING:
            try:
                study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                healed += 1
            except Exception as exc:  # 已被别的进程结算完等情况，忽略
                print(f"[HEAL] trial {t.number} 自愈跳过：{exc}", flush=True)
    if healed:
        print(f"[HEAL] 自愈 {healed} 个陈旧 RUNNING trial（标为 FAIL，不污染 TPE）", flush=True)

    # 种子 trial：对照（natural）+ 1:2 / 1:3 / 1:4 / 1:6
    if not args.no_seed_trials and len(study.trials) == 0 and not args.enqueue_plan:
        seeds = []
        for spec in ("natural", "w2.0", "w3.0", "w4.0", "w6.0"):
            p = dict(ROUTE_B_DEFAULTS)
            p["mix_spec"] = spec
            seeds.append(p)
        for p in seeds:
            study.enqueue_trial(p, skip_if_exists=True)
        print(f"[SEED] 注入 {len(seeds)} 个种子 trial（natural 对照 + w2/w3/w4/w6）", flush=True)

    # ---- 确定性对照臂（--enqueue-plan）------------------------------------
    # 为什么要有它：TPE 是概率性的；用户点名要验的 4 组假设（对照组/A/B/C）
    # **必须保证被跑到**，不能靠"运气"。入队后 study.ask() 会优先取走。
    #
    # ⚠️ **去重护栏**：看门狗会在调度器意外退出后**重启调度器**；若每次启动都把
    #    整份计划重新入队，同一条臂会被重复跑（浪费 GPU，且拿到的是同一个 seed）。
    #    ⇒ 入队前按**参数集 + 种子**指纹比对 study 里已有 trial（任何状态都算），命中就跳过。
    #
    # ⚠️ **为什么指纹里必须带种子**：复现阶段（plan_replication.py）要干的正是
    #    「同一条配方、换一组种子再跑」⇒ 参数集与已有 trial **完全相同**。
    #    若只按参数集去重，复现臂会被**全部跳过** ⇒ 复现阶段等于没跑。
    #    带种子后：Phase-1 臂（seed 100）在看门狗重启时被跳过 ✅；Phase-3 复现臂（seed 900+）正常入队 ✅。
    def _cfg_sig(d: dict, seed=None) -> tuple:
        sig = []
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, bool):
                sig.append((k, bool(v)))
            elif isinstance(v, (int, float)):
                sig.append((k, round(float(v), 6)))
            else:
                sig.append((k, str(v)))
        sig.append(("__seed__", None if seed is None else int(seed)))
        return tuple(sig)

    existing_sigs = set()
    for _t in study.get_trials(deepcopy=False):
        pr = dict(getattr(_t, "params", None) or {})
        if not pr:
            # ⚠️ **WAITING trial 的 `.params` 是空的**（入队参数存在 system_attrs 里，
            #    要等 `study.ask()` 取走、真正 suggest_* 之后才落到 `.params`）。
            #    只看 `.params` 会把"已入队但还没跑"的臂当成不存在 ⇒ 看门狗重启时**重复入队**。
            try:
                pr = dict((getattr(_t, "system_attrs", None) or {}).get("fixed_params") or {})
            except Exception:
                pr = {}
        if not pr:
            continue
        try:
            _seed = (getattr(_t, "user_attrs", None) or {}).get("seed")
        except Exception:
            _seed = None
        try:
            existing_sigs.add(_cfg_sig(pr, _seed))
        except Exception:
            continue
    print(f"[PLAN] study 已有 {len(study.trials)} 个 trial（{len(existing_sigs)} 个不同「参数集+种子」指纹）", flush=True)

    if args.enqueue_plan:
        allowed = {"mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
                   "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn"}
        if EXTENDED_SPACE:
            allowed |= {"init_from", "init_from_x", "epochs", "backbone_lr_scale",
                        "freeze_backbone_epochs"}
        if STRUCT_SPACE:
            allowed |= {"no_p2", "head_level", "act", "head_conv", "late_upsample", "unfreeze_b0b4", "head_src",
                         "p2_out_scale"}
            # ★ epochs 也允许进 plan（2026-11-06）：周末预算只够 ~16 条 20ep 臂
            #   ⇒ 需要"快筛 15ep + 终验 20ep"的分层 arm（EXTENDED_SPACE 关闭时
            #     epochs 本来由 --epochs 给，现在允许 plan 逐臂覆盖）
            allowed |= {"epochs"}
        plan = json.loads(Path(args.enqueue_plan).read_text(encoding="utf-8"))
        n_enq = 0
        n_skip = 0
        for item in plan:
            raw = dict(item.get("params") or {})
            p = {k: v for k, v in raw.items() if k in allowed}
            item_seed = item.get("seed")
            sig = _cfg_sig(p, item_seed)
            if sig in existing_sigs:
                n_skip += 1
                print(f"[PLAN] 跳过（已存在同「参数集+种子」）role={item.get('role')} "
                      f"init={p.get('init_from')} ep={p.get('epochs')} lr={p.get('lr_scale')} "
                      f"seed={item_seed}", flush=True)
                continue
            ua = {"role": str(item.get("role", "")), "enqueued_by": "enqueue_plan"}
            if item_seed is not None:
                ua["seed"] = int(item_seed)
            # ★★ epochs 走 **user_attrs** 通道（2026-11-06 修，第五个致命坑）：
            #    `epochs` 虽然在 `allowed` 里 ⇒ 被写进 fixed_params ⇒ 但
            #    `suggest_params` 只在 EXTENDED_SPACE 下才 suggest 它
            #    ⇒ 结构轮（不开 --extended-space）里 `params` **根本没有 epochs 键**
            #    ⇒ `trial_epochs = int(ua.get("epochs") or params.get("epochs") or default_epochs)`
            #    永远回落到 `--epochs`（默认 15）
            #    ⇒ **plan 里逐臂写的 20ep（母配方口径）从未生效，全部按 15ep 跑了**。
            #    ⇒ 现在把 plan 的 epochs 显式放进 user_attrs：
            #       派单处 `trial_epochs` 本来就优先读 `ua.get("epochs")` ⇒ 逐臂生效。
            #    （同时把 epochs 从 fixed_params 里摘掉，避免"写了却没人读"的歧义。）
            _ep = p.pop("epochs", None)
            if _ep is None:
                _ep = raw.get("epochs")
            if _ep is not None:
                ua["epochs"] = int(_ep)
            study.enqueue_trial(p, skip_if_exists=False, user_attrs=ua)
            existing_sigs.add(sig)
            n_enq += 1
            print(f"[PLAN] 入队 #{n_enq} role={ua['role']} init={p.get('init_from')} "
                  f"ep={p.get('epochs')} lr={p.get('lr_scale')} mix={p.get('mix_spec')} "
                  f"seed={item_seed}", flush=True)
        print(f"[PLAN] 注入 {n_enq} 个确定性对照臂（跳过已存在 {n_skip} 个）", flush=True)

    stop_flag = {"stop": False}

    def _on_signal(signum, _frame):
        print(f"\n[SIGNAL] 收到信号 {signum} ⇒ 停止派新 trial，等待在跑的收尾 …", flush=True)
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    running: list[dict] = []
    available = list(gpu_list)
    deadline = time.time() + args.max_hours * 3600
    done_count = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    registry_path = out_root / "workers.json"

    def _n_trials_limit() -> int:
        """派单上限（**每轮重读 runtime_config** ⇒ 运行期可调高以追加臂）。

        ⚠️ 为什么需要它（2026-11-06）：`--n-trials` 是启动参数、**不可变**
        ⇒ 想在搜索中途补「同配方多种子」臂（让 Model Soup 的交接口径成立），
        就必须有办法放宽上限。
        ⇒ 现在 `runtime_config.json` 的 `n_trials` 键可以覆盖它（缺省 = 启动参数）。
        ⇒ 只放宽上限，**不影响已入队的臂**。
        """
        try:
            v = int(read_runtime_config(out_root).get("n_trials") or args.n_trials)
        except Exception:
            v = int(args.n_trials)
        return max(v, int(args.n_trials))

    def save_registry(workers=None) -> None:
        ws = running if workers is None else workers
        payload = {
            "updated": time.time(),
            "workers": {
                str(info["trial_number"]): {
                    "pid": info["process"].pid,
                    "gpu": info["gpu_id"],
                    "trial_name": info["trial_name"],
                    "started_at": info["start_time"],
                }
                for info in ws
            },
        }
        registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def harvest(workers: list[dict]) -> tuple[list[dict], int]:
        """收割已结束的 worker：写 study.tell / user_attrs / summary CSV。

        ⚠️ **为什么必须抽成函数**（2026-11-02 18:2x 修 bug #12）：
        收割逻辑原来**只写在主循环里** ⇒ 调度器一旦 break（收到 SIGTERM / 到达截止时间），
        主循环已经退出 ⇒ **在跑臂的结果永远不会被收割**：
          · trial 永远停在 RUNNING（backfill 也补不回来 —— 没有 user_attrs 就没法补）
          · summary CSV 缺行
          · 权重还在，但判定器看不到它 ⇒ 一条 92 分钟的训练**白跑**
        ⇒ 现在**收尾路径也调用它** ⇒ 停机不丢结果。

        返回 ``(仍在跑的, 本次收割条数)``。
        """
        still: list[dict] = []
        n_done = 0
        for info in workers:
            ret = info["process"].poll()
            if ret is None:
                still.append(info)
                continue
            try:
                info["log_handle"].close()
            except Exception:
                pass
            available.append(info["gpu_id"])
            save_registry(still)
            res = read_trial_result(out_root / "trials" / info["trial_name"] / "trial_result.json")
            dur = (time.time() - info["start_time"]) / 60.0
            trial = info["trial"]
            if res is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                write_summary_row(summary_csv, info["trial_name"], info["params"], None, "CRASHED", dur)
                print(f"[CRASH] {info['trial_name']} rc={ret}（{dur:.1f} min）", flush=True)
            else:
                if res.get("tier1_ok") and res.get("utility") is not None:
                    score = float(res["utility"])
                    study.tell(trial, score)
                    status = "SUCCESS"
                    for k, v in res.items():
                        try:
                            trial.set_user_attr(k, v)
                        except Exception:
                            pass
                elif res.get("infrastructure_failure"):
                    # 门禁**基础设施失败**（如 Track1 CUDA OOM）⇒ 记 FAIL 而不是"触红线被剪枝"，
                    # 两者语义不同：剪枝会进 TPE 的"已探索"统计，失败则应视为"没测成"。
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    status = "CRASHED"
                    for k, v in res.items():
                        try:
                            trial.set_user_attr(k, v)
                        except Exception:
                            pass
                    print(f"[INFRA-FAIL] {info['trial_name']} 门禁基础设施失败：{res.get('gate_error')}",
                          flush=True)
                else:
                    # 一级红线失败 ⇒ 剪枝（不参与 TPE 采样）
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    status = "PRUNED"
                    for k, v in res.items():
                        try:
                            trial.set_user_attr(k, v)
                        except Exception:
                            pass
                write_summary_row(summary_csv, info["trial_name"], info["params"], res, status, dur)
                m = res.get("metrics") or {}
                print(
                    f"[{status}] {info['trial_name']} mix={info['params']['mix_spec']} "
                    f"实测 1:{res.get('measured_ratio', 0):.2f} | {res.get('gate_verdict')} | "
                    f"U={res.get('utility')} | UvsRC1={res.get('utility_vs_rc1')} | {dur:.1f} min",
                    flush=True,
                )
            n_done += 1
        return still, n_done

    hard_cut_hit = False
    while True:
        now = time.time()
        if stop_flag["stop"]:
            print("[STOP] 不再派新 trial", flush=True)
            break
        # ---- 墙钟硬截止（runtime_config.json 的 stop_at）------------------
        # ⚠️ **优先级高于 --max-hours**：主机定时关机前必须留出「门禁 + 收尾」的时间窗。
        #    这个键**运行期可改**（不重启调度器）⇒ 需要时可以临场放宽/收紧。
        _rc = read_runtime_config(out_root)
        _stop_at = parse_wallclock(_rc.get("stop_at"))
        if _stop_at and now > _stop_at:
            print(f"[DEADLINE] 到达 runtime_config.stop_at={_rc.get('stop_at')} ⇒ 收尾 …", flush=True)
            break
        if now > deadline:
            print(f"[DEADLINE] 到达 {args.max_hours} 小时截止时间，收尾 …", flush=True)
            break
        if done_count >= _n_trials_limit():
            print(f"[DONE] 已完成 {done_count} 个 trial", flush=True)
            break

        # 派新 trial
        while available and done_count + len(running) < _n_trials_limit() and time.time() < deadline \
                and not stop_flag["stop"]:
            gpu_id = available.pop(0)
            trial = study.ask()
            params = suggest_params(trial)

            # ---- 速度闸门：结构预测 > 上限 ⇒ 直接剪枝，不浪费一次训练 ----
            if key2ms:
                _ms, _src = speed_of_structure(params, key2ms, pred)
                if _ms is not None and _ms > args.speed_limit_ms:
                    study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
                    print(f"[SPEED-GATE] 剪枝 trial_{trial.number:04d}："
                          f"结构 {params.get('head_src')}/hc{params.get('head_conv')}/"
                          f"{params.get('act')} ⇒ {_ms:.1f} ms > {args.speed_limit_ms} ms（{_src}）",
                          flush=True)
                    continue

            # ---- 运行时配置（可在**不重启调度器**的前提下改后续 trial 的协议）----
            # 文件：<out_root>/runtime_config.json
            #   {"train_epochs": 5}   → 后续 trial 训练 5 epoch（cosine 周期仍为 5 ⇒ 与 RC1 同 LR 轨迹）
            #   {"gate_epoch": 3}     → 门禁量第 3 个 epoch（默认 = 最后一个）
            #   {"workers": 16}       → dataloader worker 数
            # 解析失败一律回落到启动参数（绝不因配置问题打死调度器）。
            rt = read_runtime_config(out_root)
            default_epochs = int(rt.get("train_epochs") or args.epochs)
            default_gate_epoch = int(rt.get("gate_epoch") or 0)
            default_workers = int(rt.get("workers") or args.workers)

            # ---- 每 trial 覆盖（user_attrs，由 enqueue_control_trials.py 写入）----
            ua = dict(getattr(trial, "user_attrs", {}) or {})
            seed = int(ua.get("seed", args.seed + trial.number))
            workers = int(ua.get("workers", default_workers))
            gate_epoch = int(ua.get("gate_epoch", default_gate_epoch))

            # ---- 扩展维度（周末，2026-10-30）：每 trial 的 epoch / 初始化权重 / 骨干保护 ----
            # 优先级：user_attrs 覆盖 > 搜索参数 > runtime_config > 启动参数
            trial_epochs = int(ua.get("epochs") or params.get("epochs") or default_epochs)
            # ---- 硬截止：不派「预计跑不完」的臂 --------------------------------
            # ⚠️ 为什么需要（2026-11-03 主机 06:30 定时关机）：
            #    只按墙钟停派单 ⇒ 最后派出的 15ep 臂还要再跑 155 min
            #    ⇒ 门禁被推到关机之后 ⇒ 交付物残缺。
            #    ⇒ 用**实测 epoch→耗时表**判断「这条臂能不能在 hard_cut 前跑完」。
            #    跑不完的臂直接记 PRUNED（不进判定池，也不浪费 GPU）。
            #
            # ⚠️ **必须连带跳出外层主循环**（2026-11-02 18:2x 沙箱冒烟实测抓到）：
            #    只 break 内层派单循环的话，外层会**再进来一次** ⇒ 又 ask 一条臂
            #    ⇒ 又 CUT ⇒ 循环下去 ⇒ 把**所有**待派臂逐个 PRUNED；
            #    待派臂用完后 `study.ask()` 还会**不停新建 TPE trial** 再 PRUNED
            #    ⇒ 直到 deadline 为止 ⇒ 凭空造出成百上千条 PRUNED trial（污染 study）。
            #    ⇒ 用 `hard_cut_hit` 标志把外层一起带出。
            _rc2 = read_runtime_config(out_root)
            _hard_cut = parse_wallclock(_rc2.get("hard_cut"))
            if _hard_cut:
                _est_min = float(EPOCH_MIN_EST.get(trial_epochs, EPOCH_MIN_EST_DEFAULT)) * EPOCH_MIN_EST_MARGIN
                if time.time() + _est_min * 60 > _hard_cut:
                    print(f"[CUT] trial_{trial.number:04d} 不派（ep={trial_epochs} 预估 "
                          f"{_est_min:.0f} min ⇒ 会超过 hard_cut={_rc2.get('hard_cut')}）", flush=True)
                    try:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        trial.set_user_attr("skipped_reason",
                                            f"hard_cut {_rc2.get('hard_cut')}: 预计完成时间超限")
                    except Exception:
                        pass
                    available.append(gpu_id)
                    hard_cut_hit = True
                    break
            init_key = str(params.get("init_from_x") or params.get("init_from") or "")
            if init_key:
                if init_key not in init_map:
                    raise RuntimeError(f"[INIT-MAP] 未知 init_from={init_key}（map={list(init_map)}）")
                init_weights = init_map[init_key]
            else:
                init_weights = args.weights
            backbone_lr_scale = float(params.get("backbone_lr_scale", 1.0) or 1.0)
            freeze_bb_epochs = int(params.get("freeze_backbone_epochs", 0) or 0)
            # cosine 周期：3 epoch 仍用 5（与 RC1/Ep3 同 LR 轨迹，保留旧口径）；
            # ≥6 epoch 用自身长度 ⇒ 让 lr 在整段训练里走完一整个 cosine。
            #
            # ⚠️ **只在扩展模式才由调度器显式传 --sched-epochs**：
            #    旧版调度器从不传这个参数 ⇒ worker 用自己的默认值 5。
            #    若在非扩展模式下也显式传值，当 --epochs≠3 时 cosine 周期会从 5 变成 trial_epochs
            #    ⇒ **GMC 老路径的行为回归**。故此处必须保持逐位不变（None = 不传该参数）。
            sched_epochs = (5 if trial_epochs == 3 else trial_epochs) if EXTENDED_SPACE else None

            trial_name = f"trial_{trial.number:04d}"
            log_file = out_root / "logs" / f"{trial_name}.log"
            cmd = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker",
                "--trial-number", str(trial.number),
                "--gpu-id", str(gpu_id),
                "--output-root", str(out_root),
                "--longquan-data", args.longquan_data,
                "--uav-data", args.uav_data,
                "--weights", init_weights,
                "--gate-script", args.gate_script,
                "--baseline-cache", args.baseline_cache,
                "--epochs", str(trial_epochs),
                "--batch", str(args.batch),
                "--imgsz", str(args.imgsz),
                "--workers", str(workers),
                "--gate-epochs", args.gate_epochs,
                "--gate-epoch", str(gate_epoch),
                "--seed", str(seed),
                "--mix-spec", params["mix_spec"],
                "--lr-scale", str(params["lr_scale"]),
                "--warmup-epochs", str(params["warmup_epochs"]),
                "--weight-decay", str(params["weight_decay"]),
                "--focal-beta", str(params["focal_beta"]),
                "--offset-weight", str(params["offset_weight"]),
                "--mosaic", str(params["mosaic"]),
                "--scale", str(params["scale"]),
                "--translate", str(params["translate"]),
                "--clip-max-norm", str(params["clip_max_norm"]),
                "--freeze-bn", "1" if params["freeze_bn"] else "0",
                "--backbone-lr-scale", str(backbone_lr_scale),
                "--freeze-backbone-epochs", str(freeze_bb_epochs),
                "--init-key", init_key,
            ]
            if STRUCT_SPACE:
                cmd += [
                    "--no-p2", "1" if params.get("no_p2") else "0",
                    "--head-level", str(params.get("head_level", "p3")),
                    "--act", str(params.get("act", "silu")),
                    "--head-conv", str(params.get("head_conv", 64)),
                    "--late-upsample", "1" if params.get("late_upsample") else "0",
                    "--unfreeze-b0b4", "1" if params.get("unfreeze_b0b4") else "0",
                    "--head-src", str(params.get("head_src", "")),
                    "--p2-out-scale", str(params.get("p2_out_scale", 1.0)),
                ]
            if sched_epochs is not None:
                # 只在扩展模式传（见上方注释）⇒ 非扩展模式与旧版**逐位一致**
                cmd += ["--sched-epochs", str(sched_epochs)]
            if args.reference_frame:
                cmd += ["--reference-frame", args.reference_frame]
            fh = log_file.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
            running.append({
                "trial": trial, "trial_number": trial.number, "trial_name": trial_name,
                "gpu_id": gpu_id, "params": params, "process": proc, "log_file": log_file,
                "log_handle": fh, "start_time": time.time(), "epochs": trial_epochs,
            })
            save_registry()
            gate_desc = "量末Ep" if gate_epoch in (0, trial_epochs) else f"量Ep{gate_epoch}"
            print(f"[LAUNCH] {trial_name} → GPU {gpu_id} | mix={params['mix_spec']} | "
                  f"lr_scale={params['lr_scale']:.3f} | 协议 {trial_epochs}ep/{gate_desc} "
                  f"| seed={seed} workers={workers} | 日志 {log_file.name}", flush=True)
            if EXTENDED_SPACE:
                print(f"[LAUNCH]   ★init={init_key or '(默认)'} | backbone_lr_scale={backbone_lr_scale} | "
                      f"freeze_backbone_epochs={freeze_bb_epochs} | cosine周期={sched_epochs}ep | "
                      f"role={ua.get('role', '')}", flush=True)

        # ⚠️ 硬截止触发 ⇒ **跳出外层主循环**（否则会不停 ask/CUT，详见 CUT 分支注释）
        if hard_cut_hit:
            print("[CUT] 硬截止已触发 ⇒ 不再派新 trial，进入收尾", flush=True)
            break

        # 收割（已结束的 worker ⇒ study.tell / user_attrs / summary CSV 行）
        running, n_done = harvest(running)
        done_count += n_done
        save_registry()

        # 心跳（供状态查询）
        (out_root / "status.json").write_text(
            json.dumps(
                {
                    "updated": time.time(),
                    "elapsed_h": round((time.time() - (deadline - args.max_hours * 3600)) / 3600, 3),
                    "done": done_count,
                    "running": [
                        {"trial": i["trial_name"], "gpu": i["gpu_id"], "pid": i["process"].pid,
                         "mix": i["params"]["mix_spec"]}
                        for i in running
                    ],
                    "free_gpus": available,
                    "best_value": (study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None),
                    "best_trial": (study.best_trial.number if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        time.sleep(3)

    # 收尾：等在跑的 trial 结束（或杀掉）⇒ **然后收割**（bug #12 的修复点）
    #
    # ⚠️ 等待超时**不能写死 1800 s**：15ep 臂要跑 155 min（从派单起算），
    #    固定 30 min 超时会把**还在正常训练**的臂杀掉 ⇒ 白跑 92~155 min。
    #    ⇒ 按实测耗时表算「这条臂还需要多久」+ 10 min 宽限。
    if running:
        print(f"[WAIT] 等待 {len(running)} 个在跑的 trial 收尾 …", flush=True)
        for info in running:
            est_total = float(EPOCH_MIN_EST.get(info.get("epochs"), EPOCH_MIN_EST_DEFAULT))
            elapsed = (time.time() - info["start_time"]) / 60.0
            remain_s = max(300.0, (est_total - elapsed) * 60.0 + 600.0)
            try:
                info["process"].wait(timeout=remain_s)
            except Exception:
                try:
                    info["process"].kill()
                except Exception:
                    pass
                try:
                    info["process"].wait(timeout=60)
                except Exception:
                    pass
        time.sleep(3)
        # ★ 收割刚跑完的臂 —— **不收割 ⇒ 结果丢失 / trial 卡 RUNNING / CSV 缺行**
        running, n_done = harvest(running)
        done_count += n_done
        print(f"[WAIT] 收尾收割 {n_done} 条；仍在跑 {len(running)} 条", flush=True)
    save_registry()

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print("\n" + "=" * 100, flush=True)
    print(f"🎉 搜索结束：完成 {len(complete)} / 剪枝 {len(pruned)} / 总计 {len(study.trials)}", flush=True)
    if complete:
        best = study.best_trial
        print(f"🏆 最佳 trial #{best.number}  U={study.best_value:+.6f}", flush=True)
        print(f"   参数：{best.params}", flush=True)
        print(f"   效用分及格线：RC1/Ep3 = 0.027579", flush=True)
    print(f"   汇总表：{summary_csv.resolve()}", flush=True)
    print("=" * 100 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
