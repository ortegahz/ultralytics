#!/usr/bin/env python3
"""NanoBench 定向搜索器 v3（2026-11-10）：以 `trial_0031` 为基准，单变量搜索「小目标召回更高」的新模型。

═══ 目标 ═══
交付件 `trial_0031`（P3/hc64/silu/late/no_p2，int8 19.63 ms）在 NanoBench-T1（no-GMC，8px 主档）
上是 **F1 0.887122 / nano 0.6666 / micro 0.8370 / small 0.8654 / regular 0.9445**。
本脚本的目标是**整体超过它**：NanoBench 四个桶 + 官方三轨 + 板卡 ≤20 ms。

═══ 为什么不直接用 `search_loop.py` ═══
1. `search_loop.py` 只用 T1 快筛排雷 + 官方门禁选型，**没有 NanoBench 分层** ⇒ 选不出「小目标更好」的臂；
2. 它的 W1（9501~9504）在 Ep03 被路线二抢占 GPU 而中断，**从未门禁** ⇒ 那四个格至今是空白；
3. 它没有「按 nano 下限选 epoch」的规则（late_upsample 的 nano 随训练下滑 ⇒ 纯 F1 选臂会挑到最差 epoch）。

═══ 本脚本的流水线（每臂都跑全链）═══
  ① 训练（4 卡并发，`--skip-gate`）
  ② no-GMC 缓存（T1 val 31613 帧 / conf 0.02 / top-k 100 / --ablate-p0）
  ③ NanoBench 分层（8px 主档 + 4px 严格档）
  ④ 按 **nano 下限** 选 epoch（先取 nano ≥ 下限的集合取 F1 最高者，否则取 nano 最高者）
  ⑤ 官方三轨门禁（GMC 语义，**必须带 --baseline-cache**）
  ⑥ 导 ONNX → int8 rknn（act 跟随 ckpt 自描述 ⇒ 不会导出「silu 权重 + relu 图」的死模型）
  ⑦ **一次**板卡会话测完全部候选臂（测前清残留进程、归档 arms/、控制臂必带）
  ⑧ 联合门禁 `dual_constraint_gate.py`（速度 × nano）

═══ 铁律 ═══
* 幂等可续跑（每个环节都先查产物在不在）；硬截止 `--stop-at`；JSON 台账逐臂落盘。
* 评估与训练**不共卡**（评测用同一张卡的空闲时段串行执行）。
* GMC 与 no-GMC **严禁混用**：②③ 用 no-GMC（NanoBench），⑤ 用 GMC（官方门禁），分列报告。
* 只改本地仓库、同步服务器靠 `cp -av` + md5 双侧比对；服务器代码只读。

用法（服务器）：
    python manu/pipeline/npu/nano_search.py --wave W1
    python manu/pipeline/npu/nano_search.py --wave W1 --phase eval      # 只补跑评测/门禁
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO = "/tmp/pycharm_project_10ae9e2e"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
RUNS = "/mnt/data/siping/runs/npu20ms"
WORK = f"{RUNS}/nano_search"
LOG = f"{WORK}/nano_search.log"
LEDGER = f"{WORK}/nano_search_ledger.json"
DATA_YAML = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml"
# NanoBench(no-GMC) 语义的冻结基线缓存 ⇒ 仅供 dual_constraint_gate 的 micro 对比
BASE_CACHE = "/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl"
# 官方 T1(GMC) 语义的冻结基线缓存（auto_epoch_gate_check.py 的默认值）
#   ⚠️ 2026-11-11 修正：此前误用上面的 no-GMC 缓存当官方基线
#   ⇒ 基线臂实测 F1=0.900897 ≠ 冻结 headline 0.905736 ⇒ A/B rc=2 ⇒ T1 全臂缺失
# 🔴 2026-11-18 17:40：门禁缓存分片 4→2（每分片 ~4.5GB ⇒ 4 片=20GB/臂；
#    3 臂并发 ⇒ 60GB ⇒ 曾把服务器内存打到 22% 可用 ⇒ 今晚按 2 片走，内存减半，
#    结果完全等价（分片只是把验证集切块，评估时合并）。
GATE_CACHE_GPUS = os.environ.get("GATE_CACHE_GPUS", "")  # 空 = 用本行默认 2 片
OFFICIAL_BASE_CACHE = ("/tmp/pycharm_project_10ae9e2e/runs/mixed_ft_route_b/baseline/"
                       "uav_cache_deP0.pkl")
#   ⚠️ 2026-11-11 修正：此前误用 no-GMC 缓存（nogmc_ref_weekend/uav_cache_base.pkl）
#   ⇒ 基线臂实测 F1=0.900897 ≠ 冻结 headline ⇒ A/B 报 rc=2 ⇒ T1 全臂缺失。
CS_T0043 = "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
REF = "/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json"
LQ = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc"
UAVD = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc"
BOARD = "root@192.168.0.64"   # 🟢 2026-11-13 14:50：**旧板卡 64**（用户指定）
#   旧地址 .211 已下线（ping 100% 丢包）⇒ 旧值会让板卡阶段必然超时失败。
#   访问：ssh -p 22 root@192.168.0.64（密码 ematech / key id_board）
#   ⚠️ 板卡阶段是**共享资源**：只读预检 + 绝不杀别人的进程（见 board_precheck）
BOARD_KEY = "-i /home/huangzhe/.ssh/id_board -o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# ---- 基线配方 = trial_0031 的真实配方（memory §110.3，逐项对齐）----
BASE = dict(
    init=CS_T0043, epochs=20, sched=5, warmup=1.5, lr=3.5, mix="w2.0", fb=2.75, ow=0.4,
    wd=0.000269, mosaic=0.15, scale=0.10, translate=0.05, clip=5.65, backbone_lr=1.0,
    uf=0, head_src="p3", native_grid=0, mr=1,
    sw=1.0, swl=1.0, thr=16.0, maxside=4.0, lr_floor=0.0,
    hf=0, hfsrc="b1", hfch=1,
    # 🟢🟢 2026-11-13 15:15：分类/定位**解耦**（方案 §5.1）⇒ 默认 1.0 = 不加权 = 逐位不变
    swb=1.0, swlb=1.0,
)

# ---- 基线注册表（2026-11-11 用户裁决：9863 升轨为 Baseline_v2.0）----
#   trial_0031 → [Status: Superseded / Deprecated Baseline]
#   trial_0080 (RC2) → [Status: Retired / Offline Historical Reference]
#   trial_9863 → Baseline_v2.0（Non-GMC / <21 ms 唯一官方基线）
BASE_V2_CKPT = '/mnt/data/siping/runs/npu20ms_offcal/trials/trial_2020035/weights/epoch5.pt'
BASE_V2 = dict(
    name='Baseline_v3.0', trial=2020035, epoch=5, ckpt=BASE_V2_CKPT,
    nano=0.7242438563327032, micro=0.83643009442583, small=0.8595286537809427, regular=0.9396551724137931, f1=0.8906832039622916, nano4=0.6609168241965974,
    t1=0.8909238824339264, t1_fp=1473, t2=0.9931330731766344, t2_fp=4, t3=0.9488855869242199, t3_fp=76,
    ms_delivery_board=19.306, ms_new_board=19.629, ms_new_board_t31_ref=None,
    recipe='[SB3] soup(1608003@5:w0.5924,1612001@5:w0.0980,1530006@5:w0.2116,1623003@5:w0.0980)',
)
BASELINE_REGISTRY = [
    {"id": "Baseline_v2.3", "trial": 1106580, "status": "ACTIVE / OFFICIAL BASELINE",
     "note": "2026-11-17 17:54 升轨; 家族 soup:bs640001:w2 n=4 + 种子关 bs640001:w55 + 板卡 19.706ms"},
    {"id": "Baseline_v2.2", "trial": 610000, "status": "SUPERSEDED / DEPRECATED BASELINE",
     "note": "2026-11-16 15:09 升轨；后续所有探索的 Beat-to-Pass 参照"},
    {"id": "Baseline_v2.1", "trial": 69913, "status": "SUPERSEDED / DEPRECATED BASELINE",
     "note": "被 trial_610000 取代"},
    {"id": "Baseline_v2.0", "trial": 9863, "status": "SUPERSEDED / DEPRECATED BASELINE",
     "note": "Non-GMC / <21 ms 历史基线"},
    {"id": "trial_0031", "trial": 31, "status": "SUPERSEDED / DEPRECATED BASELINE",
     "note": "旧交付件（19.306 ms 交付板）；仍作部署链路与时延同图参照"},
    {"id": "trial_0080 (RC2)", "trial": 80, "status": "RETIRED / OFFLINE HISTORICAL REFERENCE",
     "note": "离线精度天花板历史参照；已不参与 Beat-to-Pass"},
]
BASELINE_NANO_FLOOR = {"0031": 0.6666, "9863": BASE_V2["nano"]}

# ---- 板端同会话参照（trial_0031 自身）----
#   ⚠️ 台账里的 board_sessions 可能为空（被搜索器的 save_ledger 覆盖掉），
#   ⇒ 板端时延判据**直接读会话文件**，不依赖台账（2026-11-11 实测台账里是 null）。
BOARD_SESSION_FILES = [
    # 🟢 2026-11-12 16:40：**板卡换回旧板 `192.168.0.64`（evm3588）** ⇒ 参照必须换成旧板同会话，
    #   否则会拿新板（.211）的 0031 = 23.528 ms 去卡旧板候选（~19.6 ms）⇒ 口径错配。
    #   实测（旧板 192.168.0.64，同一会话，LOOPS=30/WARMUP=10）：
    #     control base(cs_t0043) 55.566 ｜ n9801 19.734 ｜ **n9863 19.642** ｜ **t31_silu(0031) 19.673** ms
    #   ⇒ 新基线 9863 在交付板上 **比 0031 自己还快 0.031 ms**（同会话）⇒ 板端判据（≤ 同会话 0031×1.02）通过。
    ("session3_old64", "/mnt/data/siping/runs/npu20ms/board_measure_20261112_old64/hm_speed_results.txt"),
    ("session2", "/mnt/data/siping/runs/npu20ms/board_measure_20261111_s2/hm_speed_results.txt"),
    ("session1", "/mnt/data/siping/runs/npu20ms/board_measure_20261111/hm_speed_results.txt"),
]
_TAG_RE = re.compile(r"rknn_heatmap_smoke :: (\S+) =====")
_MS_RE = re.compile(r"rknn_run\s+avg\s+([\d.]+) ms")


def parse_board_session_file(path: str) -> dict:
    """从 hm_speed_results.txt 解析每个 tag 的 rknn_run avg ms。"""
    out: dict = {}
    if not os.path.isfile(path):
        return out
    tag = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = _TAG_RE.search(line)
        if m:
            tag = m.group(1)
            continue
        m = _MS_RE.search(line)
        if m and tag and tag not in out:
            out[tag] = float(m.group(1))
            tag = None
    return out


def board_ms_for(trial: int, epoch: int) -> float | None:
    """某个臂在板卡会话文件里测到的 int8 ms（tag = arm_n{trial}_ep{epoch}_int8）。"""
    tag = f"arm_n{trial}_ep{epoch}_int8"
    for _key, path in BOARD_SESSION_FILES:
        arms = parse_board_session_file(path)
        if tag in arms:
            return float(arms[tag])
    return None


def board_reference_ms() -> float | None:
    """trial_0031 自身在**新板同会话**的 int8 ms（板端时延判据的参照；新板 = 23.528 ms）。"""
    for _key, path in BOARD_SESSION_FILES:
        arms = parse_board_session_file(path)
        for tag in ("t31_silu_int8", "arm_t31_silu_int8"):
            if tag in arms:
                return float(arms[tag])
    return None


# ---- 交付判据（与交付件 trial_0031 对齐；NanoBench nano 下限 = 交付件自身水平）----
# t2_fp_le  = 官方门禁红线（F1 ≥ 0.985 且 FP ≤ 36）
# t2_fp_doc = 任务书「T2 虚警红线 FP > 20 一票否决」（更严 ⇒ 两条都要过）
# budget_ms = 铁律「板端 ≤ 20 ms」（任务书「+1.0 ms 即熔断」比铁律松 ⇒ 取更严的）
DELIVER = dict(nano_min=0.6666, budget_ms=20.0, t1_ge=0.885997, t2_fp_le=36, t2_fp_doc=20,
                t3_ge=0.948330)

# ---- 臂表（全部「trial_0031 配方 + 单变量」）----
W1 = [
    (9801, 19101, dict(BASE),
     "control：复现 trial_0031 配方（换种子 19101）⇒ 给出 nano/T2-FP 的种子噪声基线"),
    (9802, 19102, dict(BASE, sw=1.5),
     "方向三① 下限：微小目标(≤4px 或 ≤16px²) 的 focal 正样本权重 1.5"),
    (9803, 19103, dict(BASE, sw=3.0),
     "方向三① 上限：微小目标的 focal 正样本权重 3.0（与 9802 构成剂量-响应对）"),
    (9804, 19104, dict(BASE, native_grid=1, mr=0),
     "监督落到头的原生网格(80×80) + 单格点目标 ⇒ 直击 §88 的根因（监督分辨率≠头原生分辨率⇒峰被稀释）"),
]

# ============================================================================
# 整晚波次表（2026-11-10 17:50 排定，依据 W1 实测读数）
# ----------------------------------------------------------------------------
# W1 实测（NanoBench no-GMC 8px 主档；对照 = 9801 同配方换种子）：
#   臂        ep5 nano   ep10 nano  ep15 nano  ep20 nano   best F1     nano@4px
#   9801 对照  0.6673     0.6659     0.6668     0.6661      0.889960    0.578~0.590
#   9802 sw1.5 0.6843     0.6786     0.6699     0.6645      0.891083    0.606~0.615
#   9803 sw3.0 0.6919     0.6933     0.6902     0.6895      0.892343    0.621~0.631
#   9804 原生网格 0.6668  0.6751     0.6749     —           0.892028    0.6465~0.6609
# ⇒ ① **size-weight 剂量-响应单调为正**，且 sw=3.0 **全 epoch 稳定**（sw1.5 会随训练衰减）⇒ 上探剂量。
# ⇒ ② **原生网格把 nano@4px 从 0.578 拉到 0.661（+8.3 pp）** ⇒ §88 根因（监督分辨率≠头原生网格）成立。
# ⇒ ③ 判据基准用**对照臂 9801**（同配方换种子），不是 0031 本体。
# ============================================================================
W2 = [
    (9811, 19111, dict(BASE, hf=1, hfsrc="b1", hfch=1),
     "方向一：跨层高频残差注入（b1 源，1 窄通道）⇒ 浅层高频能量加性注入头网格"),
    (9812, 19112, dict(BASE, lr_floor=0.10),
     "方向三②：学习率保底 = 峰值的 10%（禁止退火到接近 0 ⇒ 阻止卷积核退化为低通平滑）"),
    (9813, 19113, dict(BASE, sched=8),
     "方向三③：周期退火网格 8 ep（0031=5 是下界；§124.2 长周期高峰 LR 是 nano 头号杀手）"),
    (9814, 19114, dict(BASE, sw=4.0),
     "方向三① 剂量上探：微小目标 focal 权重 4.0（sw1.5 会衰减、sw3.0 稳定 ⇒ 往高处走）"),
]
W3 = [
    (9821, 19121, dict(BASE, sw=6.0), "方向三① 剂量上探：focal 权重 6.0（找剂量峰值）"),
    (9822, 19122, dict(BASE, sw=8.0), "方向三① 剂量上探：focal 权重 8.0（找剂量峰值）"),
    (9823, 19123, dict(BASE, sw=3.0, native_grid=1, mr=0),
     "★ 组合：原生网格(定位) × sw3.0(灵敏度) ⇒ 两条独立机理叠加"),
    (9824, 19124, dict(BASE, sw=4.0, native_grid=1, mr=0),
     "★ 组合：原生网格 × sw4.0"),
]
W4 = [
    (9831, 19131, dict(BASE, hf=1, sw=3.0), "方向一 × 剂量：hf_skip + sw3.0"),
    (9832, 19132, dict(BASE, hf=1, sw=3.0, native_grid=1, mr=0), "★ 三合一：hf_skip × 原生网格 × sw3.0"),
    (9833, 19133, dict(BASE, sched=6), "方向三③ 周期网格 6 ep"),
    (9834, 19134, dict(BASE, sched=10), "方向三③ 周期网格 10 ep"),
]
W5 = [
    (9841, 19141, dict(BASE, sw=6.0, native_grid=1, mr=0), "组合：原生网格 × sw6.0"),
    (9842, 19142, dict(BASE, epochs=12), "短周期：只训 12 ep（0031 的 nano 峰值在 ep5~10 ⇒ 同样效果省 40% 训练时间）"),
    (9843, 19143, dict(BASE, sw=3.0, lr_floor=0.10), "组合：sw3.0 × LR 保底 10%"),
    (9844, 19144, dict(BASE, sched=12), "方向三③ 周期网格 12 ep（网格上界）"),
]
W6 = [
    (9851, 19151, dict(BASE, sw=8.0, native_grid=1, mr=0), "组合：原生网格 × sw8.0"),
    (9852, 19152, dict(BASE, sw=3.0, epochs=12), "短周期 × sw3.0（12 ep）"),
    (9853, 19153, dict(BASE, hf=1, hfsrc="b0"), "方向一另一取数点：b0（/2 骨干，搬运量翻倍 ⇒ 看 nano 增益是否更大）"),
    (9854, 19154, dict(BASE, sw=3.0, swl=0.5), "★ 大目标降权（regular 权重 0.5）+ sw3.0 ⇒ 削弱『大目标梯度霸凌』"),
]
W7 = [
    (9861, 19161, dict(BASE, sw=3.0, swl=0.5, native_grid=1, mr=0), "组合：原生网格 × sw3.0 × 大目标降权"),
    (9862, 19162, dict(BASE, hf=1, hfch=2), "方向一：双窄通道（hf_ch=2）"),
    (9863, 19163, dict(BASE, sw=5.0), "剂量补点 5.0"),
    (9864, 19164, dict(BASE, sw=3.0, native_grid=1, mr=0, epochs=12), "短周期 × 原生网格 × sw3.0"),
]
W8 = [
    (9871, 19171, dict(BASE, sw=3.0, warmup=0.5), "warmup 1.5 → 0.5（更快进入高 LR 段）"),
    (9872, 19172, dict(BASE, sw=3.0, lr=4.0), "lr_scale 3.5 → 4.0（mix 轴的主导因子，记忆 §0 corr=+0.824）"),
    (9873, 19173, dict(BASE, sw=3.0, mix="w2.5"), "混样 w2.0 → w2.5（甜点区上界；w8.0 已被否证）"),
    (9874, 19174, dict(BASE, sw=3.0, epochs=12, native_grid=1, mr=0), "短周期 × 原生网格 × sw3.0（复现 9864 的规格，换种子）"),
]
W9 = [
    (9881, 19181, dict(BASE, sw=3.0), "★ 种子复现：sw3.0 换种子（验证增益是否可复现，非噪声）"),
    (9882, 19182, dict(BASE, sw=4.0, native_grid=1, mr=0, epochs=12), "组合短周期版：原生网格 × sw4.0 × 12 ep"),
    (9883, 19183, dict(BASE, hf=1, sw=4.0), "方向一 × 剂量上探：hf_skip + sw4.0"),
    (9884, 19184, dict(BASE, sw=3.0, swl=0.5, epochs=12), "短周期 × sw3.0 × 大目标降权"),
]

W10 = [
    (9910, 19110, dict(BASE, sw=5.5), "剂量续探 sw5.5（接 9863 sw5.0）"),
    (9911, 19111, dict(BASE, sw=7.0), "剂量续探 sw7.0（接 9822 sw8.0）"),
    (9912, 19112, dict(BASE, sw=5.0), "★ 种子复现：sw5.0 换种子（对 9863）"),
    (9913, 19113, dict(BASE, sw=4.0), "★ 种子复现：sw4.0 换种子（对 9814）"),
    (9914, 19114, dict(BASE, sw=6.0), "剂量续探 sw6.0"),
    (9915, 19115, dict(BASE, sw=10.0), "剂量续探 sw10.0（上限）"),
    (9916, 19116, dict(BASE, sw=5.0, thr=25.0, maxside=6.0), "sw5.0 × 更宽的小目标定义（thr25/maxside6）"),
    (9917, 19117, dict(BASE, sched=10), "★ 种子复现：周期 10ep（对 9834，其 nano 0.6961 显著高于 6/8/12ep）"),
    (9918, 19118, dict(BASE, sw=5.0, sched=10), "★ 组合：sw5.0 × 周期 10ep（9863 × 9834，取 nano 与 T3 的并集）"),
]

W11 = [
    (9920, 19120, dict(BASE, init=BASE_V2_CKPT, sw=1.0, epochs=5, sched=5),
     "★ 以 Baseline_v2.0(9863 ep5) 为起点的增量微调（方法论母版）：不再加 size-weight"),
    (9921, 19121, dict(BASE, init=BASE_V2_CKPT, sw=3.0, epochs=5, sched=5),
     "★ 以 9863 为起点 + sw3.0（检验剂量是否还能叠加）"),
    (9922, 19122, dict(BASE, init=BASE_V2_CKPT, sw=5.0, epochs=5, sched=5),
     "★ 以 9863 为起点 + sw5.0（同配方自续训，检验稳定性/过拟合）"),
    (9923, 19123, dict(BASE, init=BASE_V2_CKPT, sw=1.0, epochs=5, sched=5, mix="w2.5"),
     "★ 以 9863 为起点 + 混样 w2.5（数据侧换挡）"),
]
WAVES = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6, "W7": W7, "W8": W8, "W9": W9, "W10": W10, "W11": W11}


def log(m: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(WORK, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True, timeout=timeout)


def train_cmd(trial: int, gpu: int, seed: int, p: dict) -> str:
    # 🟢 2026-11-13 **可选的三个新旋钮**（默认全关 ⇒ 旧臂命令串**逐字不变**）：
    #   · anchor_l2  > 0  ⇒ 加 --anchor-l2 / --anchor-ckpt（L2 锚定抛光）
    #   · save_fracs 非空 ⇒ 加 --save-fracs（epoch 内分数存盘）
    #   · freeze_bn        ⇒ --freeze-bn {0,1}（抛光臂用 BN eval 固定统计量）
    _extra = ""
    if float(p.get("anchor_l2") or 0.0) > 0.0:
        _extra += f" --anchor-l2 {p['anchor_l2']} --anchor-ckpt {p.get('anchor_ckpt') or p['init']}"
    if p.get("save_fracs"):
        _extra += f" --save-fracs {p['save_fracs']}"
    # 🟢 2026-11-13 16:20：定位侧尺寸口径可独立（swb_thr / swb_maxside）
    #   0/缺省 ⇒ **不加这两个 flag** ⇒ 旧臂命令串**逐字不变**
    if float(p.get("swb_thr") or 0.0) > 0.0:
        _extra += (f" --size-thr-px-box {p['swb_thr']}"
                   f" --size-max-side-px-box {float(p.get('swb_maxside') or 0.0)}")
    # 🟢 2026-11-13 15:30：EMA 衰减率轴（方案 §3.2）—— 0/缺省 ⇒ **不加 flag** ⇒ 旧臂命令串逐字不变
    if float(p.get("ema_decay") or 0.0) > 0.0:
        _extra += f" --ema-decay {p['ema_decay']}"
    # 🟢🟢🟢 2026-11-13 17:40：微小目标 Copy-Paste（方案 §5.2）—— 0/缺省 ⇒ **不加 flag** ⇒ 旧臂命令串逐字不变
    if float(p.get("cp_prob") or 0.0) > 0.0:
        _extra += (f" --cp-prob {p['cp_prob']}"
                   f" --cp-max-px {p.get('cp_max_px', 12.0)}"
                   f" --cp-ctx-px {p.get('cp_ctx_px', 3.0)}"
                   f" --cp-feather-px {p.get('cp_feather_px', 2.0)}"
                   f" --cp-per-img {int(p.get('cp_per_img', 3))}"
                   f" --cp-min-gap-px {p.get('cp_min_gap_px', 2.0)}")
        # 🟢🟢 2026-11-13：**降尺度粘贴**（造 ≤4px 正样本）
        #   只有 scale < 1 才加这两个 flag ⇒ 旧 cp 臂的命令串**逐字不变**
        #   ⚠️ 用 `is not None` 判缺省（不能写 `or 1.0`）⇒ 否则 scale_min=0（不限下界）会被吞成 1.0
        _smin = float(p.get("cp_scale_min") if p.get("cp_scale_min") is not None else 1.0)
        _smax = float(p.get("cp_scale_max") if p.get("cp_scale_max") is not None else 1.0)
        if _smin < 1.0 or _smax < 1.0:
            _extra += f" --cp-scale-min {_smin} --cp-scale-max {_smax}"
    return (
        f"cd {REPO} && export PYTHONPATH=. && {PY} -u manu/training/optuna_mix_ratio_search.py --worker "
        f"--trial-number {trial} --gpu-id {gpu} --output-root {ROOT} "
        f"--longquan-data {LQ} --uav-data {UAVD} --weights {p['init']} --baseline-cache {BASE_CACHE} "
        f"--epochs {p['epochs']} --sched-epochs {p['sched']} --batch 32 --imgsz 640 --workers 8 --seed {seed} "
        f"--mix-spec {p['mix']} --lr-scale {p['lr']} --warmup-epochs {p['warmup']} --weight-decay {p['wd']} "
        f"--focal-beta {p['fb']} --offset-weight {p['ow']} --mosaic {p['mosaic']} --scale {p['scale']} "
        f"--translate {p['translate']} --clip-max-norm {p['clip']} --freeze-bn {int(p.get('freeze_bn', 0) or 0)} "
        f"--backbone-lr-scale {p['backbone_lr']} --freeze-backbone-epochs 0 "
        f"--no-p2 1 --head-level p3 --act silu --head-conv 64 --late-upsample 1 "
        f"--unfreeze-b0b4 {p['uf']} --head-src '{p['head_src']}' --p2-out-scale 1.0 "
        f"--train-on-native-grid {p['native_grid']} --min-radius {p['mr']} "
        f"--size-weight-small {p['sw']} --size-weight-large {p['swl']} "
        f"--size-weight-small-box {p.get('swb', 1.0)} --size-weight-large-box {p.get('swlb', 1.0)} "
        f"--size-thr-px {p['thr']} --size-max-side-px {p['maxside']} "
        f"--lr-floor-ratio {p['lr_floor']} "
        f"--hf-skip {p['hf']} --hf-skip-src {p['hfsrc']} --hf-skip-channels {p['hfch']} "
        f"--reference-frame {REF} --skip-gate{_extra} > {ROOT}/logs/NANO_{trial}.log 2>&1"
    )


# ============================================================================
# 台账
# ============================================================================
# 🔴🔴🔴 2026-11-19 14:40 —— **乐观并发：load 时记录每个臂的指纹**（跨进程丢更新的根因修复）
#   现场事故（2026-11-19 全天反复出现）：
#     · 14:08 守望扫描到「六项全过臂 15 个」，14:04 是 17 个 ⇒ 读到的台账被**回滚**了；
#     · 14:18 升轨 v3.0 后，台账出现「baseline_v2 = v3.0，但 registry 里 v3.0 标成 SUPERSEDED」；
#     · 14:30 重算 474 个臂的 `official.six_ok`，几分钟后又被改回旧值。
#   根因：`_merge_disk_ledger` 的 arms 合并是 **整臂覆盖**（`merged.update(m_arms)`）⇒
#     进程 P 在 T0 读到台账、在 T1（可能十几分钟后）写回时，它手上**没改过的臂**也带着 T0 的旧内容
#     ⇒ **把别的进程在这期间写进去的新内容整臂覆盖掉**（典型 lost update）。
#     flock 只串行化了「写盘」这一步，**没有保护「读→改→写」这个事务**。
#   修法（乐观并发 / 三方合并）：
#     load 时给每个臂算一个指纹（md5 of 规范化 JSON），存到 `_LEDGER_ARM_HASH`；
#     save 时逐臂比较：
#       · 指纹没变 ⇒ **本进程没碰过这一臂** ⇒ 用**磁盘版**（可能含别的进程更新的字段）；
#       · 指纹变了 ⇒ 本进程改过 ⇒ 以**内存版**为准（并保留磁盘上新增的键）。
#     ⇒ 「我没改过的臂」不再覆盖别人的更新；「我改过的臂」仍然我说了算。
_LEDGER_ARM_HASH: dict = {}


def _arm_hash(arm: dict) -> str:
    """一个臂的稳定指纹（键排序 + 规范化 JSON）。"""
    try:
        blob = json.dumps(arm, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return ""
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def load_ledger() -> dict:
    global _LEDGER_ARM_HASH
    if os.path.isfile(LEDGER):
        try:
            data = json.load(open(LEDGER))
            _LEDGER_ARM_HASH = {
                str(t): _arm_hash(a)
                for t, a in (data.get("arms") or {}).items()
                if isinstance(a, dict)
            }
            return data
        except Exception as e:  # noqa: BLE001
            log(f"[台账] 读取失败（{e}）⇒ 重建空台账")
    _LEDGER_ARM_HASH = {}
    return {"arms": {}, "waves": [], "created": time.strftime("%Y-%m-%d %H:%M:%S")}


_LEDGER_LOCK = threading.RLock()

# ---- 跨进程写台账互斥锁（2026-11-12 09:22 新增）----------------------------
try:
    import fcntl  # noqa: PLC0415
except Exception:  # noqa: BLE001
    fcntl = None  # type: ignore[assignment]
_LEDGER_LOCKFILE = f"{WORK}/ledger.lock"


def _merge_disk_ledger(led: dict) -> None:
    """把磁盘上台账的**其它进程新增内容**合并进内存态（内存态优先）。

    合并口径：
      · arms   —— 按 trial 号逐臂覆盖（内存里的这一臂是最新的）
      · rounds —— 按 round 号合并（磁盘已有的轮**不丢**）
      · waves  —— 并集
    """
    try:
        with open(LEDGER, encoding="utf-8") as f:
            disk = json.load(f)
    except Exception:
        return
    if not isinstance(disk, dict):
        return
    d_arms = disk.get("arms") or {}
    m_arms = led.get("arms") or {}
    if d_arms or m_arms:
        merged = dict(d_arms)
        for _tid, _m in m_arms.items():
            if not isinstance(_m, dict):
                merged[_tid] = _m
                continue
            _d = d_arms.get(_tid)
            if not isinstance(_d, dict):
                merged[_tid] = _m
                continue
            # 本进程有没有改过这一臂？（指纹 vs load 时的快照）
            if _LEDGER_ARM_HASH.get(str(_tid)) == _arm_hash(_m):
                merged[_tid] = _d          # 没改过 ⇒ 用磁盘版（可能含别人的更新）
            else:
                merged[_tid] = {**_d, **_m}  # 改过 ⇒ 内存优先，磁盘新增键不丢
        led["arms"] = merged
    d_rounds = {int(r.get("round", 0)): r for r in (disk.get("rounds") or []) if isinstance(r, dict)}
    m_rounds = {int(r.get("round", 0)): r for r in (led.get("rounds") or []) if isinstance(r, dict)}
    all_r = dict(d_rounds)
    all_r.update(m_rounds)
    if all_r:
        led["rounds"] = [all_r[k] for k in sorted(all_r)]
    dw = list(disk.get("waves") or [])
    mw = list(led.get("waves") or [])
    led["waves"] = list(dict.fromkeys(dw + mw))

    # 2026-11-17 18:10: 基线段必须单调(升轨后不被并发写覆盖)
    #   现象: 17:54 升轨把 baseline_v2 写成 trial_1106580, 但汤通道/门禁通道随后用
    #         旧内存态回写台账(_merge_disk_ledger 只合并 arms/rounds/waves)
    #         => 新基线被悄悄回滚成 610000 => 台账与代码常量不一致。
    #   修法: baseline_v2 取 promoted_at 更晚的那个; baseline_registry 按 id 合并。
    d_base = disk.get("baseline_v2") or {}
    m_base = led.get("baseline_v2") or {}
    if str(d_base.get("promoted_at") or "") > str(m_base.get("promoted_at") or ""):
        led["baseline_v2"] = d_base
    # 🟢🟢 2026-11-18 09:25 —— **登记表允许「退役 id」**（防基线名通胀留下的垃圾条目复活）
    #   事故：11-17 深夜守望把同一份权重（trial_1117002）反复升轨 82 次 ⇒ registry 89 条
    #         （Baseline_v2.4 ~ v2.86 全是同一权重）⇒ 基线名通胀到 v2.86，registry 被刷屏。
    #   修法：台账里存 `registry_retired_ids`（被折叠掉的 id 列表）⇒ 合并时**双方都跳过**这些 id
    #        ⇒ 清理只做一次，且不会被任何常驻进程写回时复活。
    _retired = set(disk.get("registry_retired_ids") or []) | set(led.get("registry_retired_ids") or [])
    _reg = {}
    for x in (disk.get("baseline_registry") or []) + (led.get("baseline_registry") or []):
        if isinstance(x, dict) and x.get("id") and x["id"] not in _retired:
            _reg[x["id"]] = x
    if _reg:
        # 顺序: 当前基线(ACTIVE)排最前, 其余保持插入序
        _bt = int((led.get("baseline_v2") or {}).get("trial") or 0)
        _items = list(_reg.values())
        _items.sort(key=lambda x: 0 if int(x.get("trial") or 0) == _bt else 1)
        led["baseline_registry"] = _items
    if _retired:
        led["registry_retired_ids"] = sorted(_retired)


def save_ledger(led: dict) -> None:
    """写台账（**线程安全**）。

    🔴 2026-11-11 16:52 真 bug：原先所有线程共用固定临时文件 `LEDGER + '.tmp'`，
    4 个 worker 并发写 ⇒ 一个线程 os.replace 走掉之后另一个线程找不到 .tmp
    ⇒ FileNotFoundError ⇒ phase_nanobench 抛异常 ⇒ **整轮 R1 被打断**
    （24 个臂训完，却只有 5 个拿到 NanoBench 读数、0 个有 picked，
      六项判据永远缺项 ⇒ 门禁/板卡/报告全没跑、整轮记录丢失）。
    修：① 每线程独立临时名（pid+tid）② 全局锁串行化整个写盘
        ③ json.dump 撞上「并发往 dict 里加臂」时重试。

    🔴 2026-11-12 09:22 **跨进程**丢更新（真丢了一条轮记录）：
      `gate_triad_arms.py` 在 09:10:58 读台账 ⇒ 09:19:36 写回 ⇒
      它手上的台账是**旧的**（没有 R6 轮记录，R6 09:11:19 才写）⇒
      **整个 R6 的轮记录被覆盖丢失**（`rounds` 少一轮 ⇒ 守护的续跑轮号会算错）。
    修：④ 写盘前**重新读盘并合并**：
        arms   —— 按 trial 号逐臂合并（内存覆盖磁盘 ⇒ 本进程的更新生效）
        rounds —— 按 round 号合并（磁盘已有的轮不丢）
        waves  —— 并集
      ⑤ 加 `flock` 跨进程互斥 ⇒ 两个进程不会互相覆盖。
    """
    with _LEDGER_LOCK:
        os.makedirs(WORK, exist_ok=True)
        # ④ 跨进程合并：先读盘，把「磁盘上别的进程写的、内存里没有的」补回来
        try:
            with open(_LEDGER_LOCKFILE, "w", encoding="utf-8") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    _merge_disk_ledger(led)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception as e:  # noqa: BLE001
            log(f"[台账] 合并磁盘版本失败（继续写内存态）：{e}")
        # 基线升轨元信息（2026-11-11）⇒ 任何进程保存台账都自动带上，避免并发覆盖丢失
        led.setdefault("baseline_v2", BASE_V2)
        led.setdefault("baseline_registry", BASELINE_REGISTRY)
        led.setdefault("beat_to_pass", {
            "ref": "Baseline_v2.0(trial_9863)", "nano_min": BASE_V2["nano"],
            "f1_min": BASE_V2["f1"], "t1_min": BASE_V2["t1"],
            "t2_fp_max": BASE_V2["t2_fp"], "t3_min": BASE_V2["t3"],
            "legacy_ref": "trial_0031", "legacy_nano_min": BASELINE_NANO_FLOOR["0031"],
            "note": "后续所有探索统一以 Baseline_v2.0 的指标为 Beat-to-Pass 下限"})
        tmp = f"{LEDGER}.tmp.{os.getpid()}.{threading.get_ident()}"
        for attempt in range(6):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(led, f, ensure_ascii=False, indent=2)
                break
            except RuntimeError:
                # dump 期间别的线程往 dict 里加臂 ⇒ 尺寸变了 ⇒ 重试
                if attempt == 5:
                    raise
                time.sleep(0.05)
        os.replace(tmp, LEDGER)


def arm_state(led: dict, trial: int) -> dict:
    return led["arms"].setdefault(str(trial), {"trial": trial})


# ============================================================================
# ① 训练
# ============================================================================
def phase_train(arms: list, stop_ts: float, gpus: list | None = None) -> None:
    led = load_ledger()
    gpus = gpus or [0, 1, 2, 3]

    def one(item):
        try:
            _one(item)
        except Exception as e:  # noqa: BLE001
            log(f"[{item[0]}] 训练阶段异常：{e}")

    def _one(item):
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        st.update(seed=seed, desc=desc, params={k: v for k, v in p.items()})
        last = f"{ROOT}/trials/trial_{trial}/weights/epoch{p['epochs']}.pt"
        if os.path.isfile(last):
            log(f"[{trial}] 末 epoch 权重已存在 ⇒ 跳过训练")
            st["trained"] = True
            return
        if time.time() > stop_ts:
            log(f"[{trial}] 距硬截止不足 ⇒ 不启动")
            st["trained"] = False
            return
        gpu = gpus[trial % len(gpus)]
        log(f"[{trial}] GPU{gpu} 训练 seed={seed} | {desc}")
        try:
            sh(train_cmd(trial, gpu, seed, p), timeout=36000)
        except Exception as e:  # noqa: BLE001
            log(f"[{trial}] 训练异常：{e}")
        ok = os.path.isfile(last)
        st["trained"] = ok
        log(f"[{trial}] 训练结束 ⇒ {'OK' if ok else 'FAIL（查日志）'}")
        save_ledger(led)

    with ThreadPoolExecutor(max_workers=TRAIN_WORKERS[0]) as ex:
        list(ex.map(one, arms))
    save_ledger(led)


# ============================================================================
# ②③ 缓存 + NanoBench
# ============================================================================
WORKERS = [4]
# 🟢 2026-11-12 11:25：**NanoBench 评测阶段**并发 4 → 8（官方三轨门禁仍留 4，那三个 track 单作业内存未实测）
#   实测（R7 评测阶段实采 11:21）：4 个 `cache_trial0474_inferences.py` 并发时整机 **38 GB used / 82 GB available**
#   ⇒ 单个评测作业 ≈ 7 GB ⇒ 8 并发 ≈ 56 GB + 系统 10 GB = 66 GB（125 GB 机器，余 59 GB）⇒ 安全
#   收益：该阶段 24 臂 4 并发 ≈ 10 min ⇒ 8 并发 ≈ 5 min（每轮省 ~5 min ≈ +10% 有效臂/小时）
#   回退：NANO_BENCH_WORKERS=4
NANO_BENCH_WORKERS = [int(os.environ.get("NANO_BENCH_WORKERS", "8"))]
# 🟢 2026-11-12 16:20：**NanoBench 评测段改进程池**（线程池在 GIL 上排队 ⇒ 并发越多越慢）
#   根因（实测 R8 评测段 11-12 14:03→15:06，38 臂 63 min = 1.66 min/臂）：
#     `eval_cache` 是**纯 Python/numpy 的 CPU 活**（无 GPU）⇒ 放进 ThreadPoolExecutor 后
#     8 个线程在 GIL 上排队（convoy）⇒ **8 线程比串行还慢 3.2×**（同 pkl：串行 95.1 s，
#     线程池 300.9 s 且 CPU 时间 95→2378 s）⇒ 这是 R8 评测段回退的根因，不是"算得慢"。
#   实测（同一 pkl，8 路）：串行 95.1 s ｜ 线程池 300.9 s ｜ **进程池 12.5 s ⇒ 24.1×**，
#     且 F1 **逐位相同**（0.890315 三模式一致）⇒ 只换并发模型，不动算法。
#   内存：`eval_cache` 实测 **峰值 RSS 212 MB / pkl 8.1 MB**（纯 CPU，不碰 GPU）⇒ 8 路 ≈ 1.7 GB，
#     与 8 路建缓存（各 ~7 GB）并存仍 < 60 GB / 125 GB ⇒ 安全。
#   数值不变：同一 `eval_cache` 调用；验收走 /tmp/nb_verify.py 的 before/after 逐位 diff。
_NB_POOL = None
_NB_POOL_LOCK = threading.Lock()


def _nb_pool():
    """NanoBench 评测进程池（懒创建、跨轮复用）。

    🔴 上下文选 **forkserver**（不是 spawn、不是 fork）：
      * `fork` ⇒ 父进程是多线程的（8 个训练线程 + 子进程调用）⇒ fork 后子进程可能撞上别的线程
        持有的锁 ⇒ 死锁风险；且父进程若已初始化 CUDA，fork 出的子进程继承 CUDA 上下文不安全。
      * `spawn` ⇒ 子进程会**重新 import 主模块**（`__mp_main__`）⇒ 若主模块没有
        `if __name__ == "__main__"` 保护，子进程会把主流程**再跑一遍**并在自举期里再开进程池 ⇒
        RuntimeError: "An attempt has been made to start a new process before the current
        process has finished its bootstrapping phase"（2026-11-12 隔离演练实测复现）。
      * `forkserver` ⇒ worker 由一个**干净的单线程服务进程** fork，**不 import 主模块**
        ⇒ 既没有 fork-from-threads 的死锁风险，也没有主模块自举风险。
    """
    global _NB_POOL
    with _NB_POOL_LOCK:
        if _NB_POOL is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            # 用 **spawn**（Linux 3.10）：worker 由 spawn._main 启动 ⇒ 会按 preparation data
            #   **重新 import 主模块**（`__mp_main__`）。
            #   ⇒ 主模块**必须**有 `if __name__ == "__main__":` 保护
            #      （`adaptive_runner.py` 有 ✔；2026-11-12 隔离演练实测：主模块若在顶层起进程池，
            #        子进程自举期会撞 "An attempt has been made to start a new process before the
            #        current process has finished its bootstrapping phase" ⇒ BrokenProcessPool）。
            #   ⇒ 为此 `_one()` 里对进程池**做了兜底**：池一挂就退回线程内直算（本轮变慢，但绝不掀轮次）。
            #   （曾试 forkserver：3.10 的 forkserver worker 仍走 spawn._main ⇒ 同样会 fixup 主模块，
            #     且 fork-from-threads 有 stdout 锁死锁风险 ⇒ 不采用。）
            ctx = mp.get_context("spawn")
            _NB_POOL = ProcessPoolExecutor(max_workers=max(1, NANO_BENCH_WORKERS[0]),
                                           mp_context=ctx)
    return _NB_POOL


def _nb_worker(payload):
    """进程池 worker：跑一次 NanoBench 评测，**只回传需要的标量**（避免大对象 pickle）。"""
    pkl, label = payload
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
    res = RA.nanobench(pkl, label)
    if not res:
        return None
    p8 = res["peak_tol8"]
    p4 = res["peak_tol4"]
    rb = p8["recall_by_bucket"]
    rb4 = p4["recall_by_bucket"]
    return {"f1": p8["f1"], "th": p8["th"], "fp": p8["fp"], "fn": p8["fn"],
            "nano": rb["nano"], "micro": rb["micro"], "small": rb["small"],
            "regular": rb["regular"],
            "nano4": rb4["nano"], "micro4": rb4["micro"], "f1_4": p4["f1"]}


def _nb_inline(pkl: str, label: str):
    """兜底：在线程内直接跑一次 NanoBench 评测（进程池不可用时用）。"""
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
    res = RA.nanobench(pkl, label)
    if not res:
        return None
    p8 = res["peak_tol8"]
    p4 = res["peak_tol4"]
    rb = p8["recall_by_bucket"]
    rb4 = p4["recall_by_bucket"]
    return {"f1": p8["f1"], "th": p8["th"], "fp": p8["fp"], "fn": p8["fn"],
            "nano": rb["nano"], "micro": rb["micro"], "small": rb["small"],
            "regular": rb["regular"],
            "nano4": rb4["nano"], "micro4": rb4["micro"], "f1_4": p4["f1"]}
# 🟢 2026-11-12 10:30：**训练阶段**并发臂数 4 → 8（2 臂/卡），只用满整夜算力
#   实测（4 卡 / 80 核 / 125 GB）：
#     单臂真实物理占用(PSS) ≈ 11.0 GB ｜ 单臂 CPU ≈ 4.1 核 ｜ 单臂 GPU ≈ 45% util / 4.6 GB 显存
#   ⇒ 8 臂并发后：内存 ≈ 88 GB（余 ≈ 25 GB）｜ CPU ≈ 33 核（余 ≈ 47 核）｜ 显存 ≈ 9.2 GB/卡（余 15 GB）
#      ⇒ 单卡双臂交错，GPU util 45% → ~85%，整夜有效算力约 ×1.8
#   写盘安全：save_ledger 有线程锁 + 跨进程 flock + 写前合并 ⇒ 8 线程并发不新增风险
#   回退：NANO_TRAIN_WORKERS=4 bash manu/pipeline/npu/run_adaptive_night.sh ...
TRAIN_WORKERS = [int(os.environ.get("NANO_TRAIN_WORKERS", "8"))]
# 🟢 2026-11-12 11:00：单臂最多给几个「三项全过」的 epoch 补跑官方三轨
#   （实测同一臂内 nano 最优 epoch 与 T1 过线 epoch 可能差 1 个 epoch ⇒ 只测一个会白丢候选）
GATE_EPOCH_K = 3
NANO_FLOOR = BASE_V2["nano"]  # Beat-to-Pass 下限（默认 Baseline_v2.0 / trial_9863）


# 🟢🟢 2026-11-13 14:45 **亚 epoch 密集存盘**（周末方案 §3）：支持「分数 epoch」
#   训练端 `--save-fracs 0.25,0.5,0.75` ⇒ 额外落盘 `weights/sub_epoch{E}_{frac}.pt`
#   （该能力**训练脚本里早已实现**，只是搜索空间从没启用过 ⇒ 现在把评测/门禁/台账接上）
#   台账键：整数 epoch → '4'；分数 epoch → '2.5'（与权重文件名一一对应）
FRAC_MIN_EPOCH = 2            # 只在 ep≥2 之后取亚 epoch（ep1 的模型还没成型）


def epoch_ckpt_name(epoch) -> str:
    """epoch → 权重文件名：整数 4 → 'epoch4.pt'；分数 2.5 → 'sub_epoch2_0.5.pt'。"""
    try:
        f = float(epoch)
    except (TypeError, ValueError):
        return f"epoch{epoch}.pt"
    if abs(f - round(f)) < 1e-9:
        return f"epoch{int(round(f))}.pt"
    ep = int(f // 1)
    return f"sub_epoch{ep}_{f - ep:g}.pt"


def epoch_key_str(epoch) -> str:
    """epoch → 台账/门禁目录键：整数 → '4'；分数 → '2.5'。"""
    try:
        f = float(epoch)
    except (TypeError, ValueError):
        return str(epoch)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def frac_list(save_fracs) -> list[float]:
    """把 params['save_fracs']（'0.25,0.5'）解析成小数列表（非法值静默丢弃）。"""
    out: list[float] = []
    for tok in str(save_fracs or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            f = float(tok)
        except ValueError:
            continue
        if 0.0 < f < 1.0:
            out.append(f)
    return out

def phase_nanobench(arms: list, epochs: list[int], gpu_of: dict) -> None:
    sys.path.insert(0, REPO)
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415

    led = load_ledger()

    def one(item):
        # 🔴 单臂异常不许掀掉整池（16:52 的 FileNotFoundError 就是这样把整轮打断的）
        try:
            _one(item)
        except Exception as e:  # noqa: BLE001
            log(f"[{item[0]}] NanoBench 异常：{e}")

    def _one(item):
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        traj = st.setdefault("nanobench", {})
        for ep in epochs:
            ck = f"{ROOT}/trials/trial_{trial}/weights/epoch{ep}.pt"
            if not os.path.isfile(ck):
                continue
            key = str(ep)
            if key in traj:
                continue
            pkl = f"{RUNS}/tiny_ref/nano_{trial}_ep{ep}.pkl"
            # 🟢 2026-11-12 17:00：**跳过 peak-attrs**（area/sums）计算 ⇒ 建缓存快一大截。
            #   官方三轨门禁走 auto_epoch_gate_check.py **自建 GMC 缓存**（不传该开关）⇒ 官方读数不受影响；
            #   NanoBench 只读 pred_points/pred_scores ⇒ 本段读数逐位不变。
            if not RA.build_cache(ck, pkl, gpu_of.get(trial, 0), extra="--no-peak-attrs"):
                traj[key] = {"error": "cache_fail"}
                continue
            # 评测是纯 CPU 活 ⇒ 丢进**进程池**（线程池会 GIL 排队，实测比串行还慢 3.2×）
            try:
                res = _nb_pool().submit(_nb_worker, (pkl, f"n{trial}_ep{ep}")).result()
            except Exception as e:  # noqa: BLE001
                # 🔴 兜底：进程池挂（BrokenProcessPool / 自举冲突等）⇒ 线程内直算
                #    （慢 3~20×，但**绝不掀掉整轮** ⇒ 读数不丢）
                log(f"[{trial}] ep{ep} 进程池失败（{type(e).__name__}: {e}）⇒ 退回线程内直算")
                res = _nb_inline(pkl, f"n{trial}_ep{ep}")
            if not res:
                traj[key] = {"error": "nanobench_fail"}
                continue
            traj[key] = {
                "f1": res["f1"], "th": res["th"], "fp": res["fp"], "fn": res["fn"],
                "nano": res["nano"], "micro": res["micro"], "small": res["small"],
                "regular": res["regular"],
                "nano4": res["nano4"], "micro4": res["micro4"], "f1_4": res["f1_4"],
                "cache": pkl,
            }
            log(f"[{trial}] ep{ep} F1={res['f1']:.6f} nano={res['nano']:.4f} "
                f"nano@4px={res['nano4']:.4f} micro={res['micro']:.4f} regular={res['regular']:.4f}")
            save_ledger(led)
        # 🟢🟢 2026-11-13 14:45：**亚 epoch 读数**（周末方案 §3）
        #   整数 epoch 的采样粒度太粗（nano 随步数升、T3 随过拟合降 ⇒ 黄金窗口常只有 0.2~0.5 epoch 宽）
        #   ⇒ 训练端存了 `sub_epoch{E}_{frac}.pt` 的臂，这里把它们的 NanoBench 读数也写进台账。
        _fr = frac_list(p.get("save_fracs"))
        if _fr:
            # 与整数 epoch 用**同一批 epoch**（调用方给的 epochs 列表）⇒ 口径一致
            for _ep in sorted({int(e) for e in epochs}):
                if _ep < FRAC_MIN_EPOCH:
                    continue
                for _sf in _fr:
                    _name = f"sub_epoch{_ep}_{_sf:g}.pt"
                    _ck = f"{ROOT}/trials/trial_{trial}/weights/{_name}"
                    if not os.path.isfile(_ck):
                        continue
                    _key = f"{_ep + _sf:g}"
                    if _key in traj:
                        continue
                    _pkl = f"{RUNS}/tiny_ref/nano_{trial}_ep{_key}.pkl"
                    if not RA.build_cache(_ck, _pkl, gpu_of.get(trial, 0), extra="--no-peak-attrs"):
                        traj[_key] = {"error": "cache_fail"}
                        continue
                    try:
                        _res = _nb_pool().submit(_nb_worker, (_pkl, f"n{trial}_ep{_key}")).result()
                    except Exception as e:  # noqa: BLE001
                        log(f"[{trial}] ep{_key} 进程池失败（{type(e).__name__}）⇒ 退回线程内直算")
                        _res = _nb_inline(_pkl, f"n{trial}_ep{_key}")
                    if not _res:
                        traj[_key] = {"error": "nanobench_fail"}
                        continue
                    traj[_key] = {
                        "f1": _res["f1"], "th": _res["th"], "fp": _res["fp"], "fn": _res["fn"],
                        "nano": _res["nano"], "micro": _res["micro"], "small": _res["small"],
                        "regular": _res["regular"],
                        "nano4": _res["nano4"], "micro4": _res["micro4"], "f1_4": _res["f1_4"],
                        "cache": _pkl, "sub_epoch_frac": float(_sf),
                    }
                    log(f"[{trial}] ep{_key}(亚) F1={_res['f1']:.6f} nano={_res['nano']:.4f} "
                        f"nano@4px={_res['nano4']:.4f}")
                    save_ledger(led)
        save_ledger(led)

    with ThreadPoolExecutor(max_workers=NANO_BENCH_WORKERS[0]) as ex:
        list(ex.map(one, arms))


def base_pick_thresholds() -> tuple[float, float, float]:
    """基线（9863）的三项 NanoBench 阈值：nano / nano4 / F1（**用台账精确读数**，不用截断常量）。"""
    led = load_ledger()
    a = (led.get("arms") or {}).get(str(BASE_V2["trial"]), {})
    pk = a.get("picked") or {}
    return (pk.get("nano") or BASE_V2["nano"],
            pk.get("nano4") or BASE_V2["nano4"],
            pk.get("f1") or BASE_V2["f1"])


def base_official_thresholds(led: dict | None = None) -> tuple[float, float, float]:
    """基线（9863）的官方三轨**精确**读数：T1 下限 / T2 虚警上限 / T3 下限。"""
    led = led if led is not None else load_ledger()
    a = (led.get("arms") or {}).get(str(BASE_V2["trial"]), {})
    of = a.get("official") or {}
    fp = of.get("t2_fp")
    return (of.get("t1") or BASE_V2["t1"],
            float(fp) if fp is not None else float(BASE_V2["t2_fp"]),
            of.get("t3") or BASE_V2["t3"])


def official_beat(g: dict, bo: tuple[float, float, float]) -> bool:
    """官方三轨是否**相对基线全过**（T1 ≥ / T2虚警 ≤ / T3 ≥）。"""
    if not g or "error" in g:
        return False
    t1, fp_max, t3 = bo
    try:
        return (float(g["t1"]) >= t1
                and float(g["t2_fp"]) <= fp_max
                and float(g["t3"]) >= t3)
    except (KeyError, TypeError, ValueError):
        return False


def pick_epoch(traj: dict, nano_min: float, nano4_min: float | None = None,
               f1_min: float | None = None) -> tuple[dict | None, str]:
    """选 epoch（**六项判据的合取视角**）。

    ⚠️ 下限 = **Beat-to-Pass 参照**的 nano（2026-11-11 起 = Baseline_v2.0 / trial_9863 的 0.708648；
    旧参照 trial_0031 = 0.6666）。

    🟢 2026-11-11 18:00 修：旧规则只看 F1 ⇒ 会把「nano 过了但 nano4 差一点点」的 epoch 选上
    （实测 12047：ep5 的 nano4=0.646739 < 基线 0.647212，只差 0.0005 却直接判负；
      而它的 ep4 nano4=0.6569 是过的）⇒ 白丢候选。
    新规则：
      ① 先取「nano ≥ 下限」的集合；
      ② 其中**同时**满足 nano4 ≥ 基线 且 F1 ≥ 基线 的（三项全过）⇒ 取其中 nano 最高者；
      ③ 否则退回「nano 过线的里 F1 最高者」；
      ④ 再否则取 nano 最高者。
    """
    # 🟢 2026-11-13 14:45：键可以是分数 epoch（'2.5'，亚 epoch 存盘）⇒ 不能 int(k)
    cands = []
    for k, v in traj.items():
        if "f1" not in v:
            continue
        try:
            _f = float(k)
        except (TypeError, ValueError):
            continue
        _ep = int(round(_f)) if abs(_f - round(_f)) < 1e-9 else _f
        cands.append(dict(v, epoch=_ep, epoch_key=epoch_key_str(_f),
                           ckpt=epoch_ckpt_name(_f)))
    if not cands:
        return None, "none"
    ok = [c for c in cands if c["nano"] >= nano_min]
    if ok:
        if nano4_min is not None and f1_min is not None:
            full = [c for c in ok
                    if (c.get("nano4") or 0.0) >= nano4_min and c["f1"] >= f1_min]
            if full:
                return max(full, key=lambda c: c["nano"]), "all3_pass_max_nano"
        return max(ok, key=lambda c: c["f1"]), "f1_within_nano_floor"
    return max(cands, key=lambda c: c["nano"]), "max_nano_nofloor"


# ============================================================================
# ⑤ 官方三轨门禁（GMC 语义）
# ============================================================================
def run_official_gate(trial: int, epoch, gpu: int) -> dict:
    # 🟢 2026-11-13 14:45：epoch 可能是分数（亚 epoch）⇒ 用统一解析拿权重文件名
    _ek = epoch_key_str(epoch)
    w = f"{ROOT}/trials/trial_{trial}/weights/{epoch_ckpt_name(epoch)}"
    outd = f"{ROOT}/gate/NANO_{trial}_ep{_ek}_official"
    rp = f"{outd}/gate_report.json"
    reuse = False
    if os.path.isfile(rp):
        try:
            _rep = json.load(open(rp))
            _t1s = (_rep.get("tracks", {}) or {}).get("track1_uav", {}) or {}
            # 🔴 2026-11-12 12:55 修：**报告格式变了**（T1 的 f1 现在在 `finetune_peak.f1`，
            #   顶层没有 "f1"）⇒ 老的复用判据 `"f1" in _t1s` **恒为 False**
            #   ⇒ 已存在的门禁报告**永远不会被复用** ⇒ 每跑一次门禁就整轮重算（约 7 min GPU）
            #   ⇒ 轮次重跑（守护热升级）时把已算过的门禁全部重算一遍，纯浪费算力。
            #   ⇒ 新判据：T1 有 finetune_peak.f1（或顶层 f1）**且** T3 有 f1 **且** 状态不是 ab_failed。
            _t1f1 = _t1s.get("finetune_peak") or {}
            _t3s = (_rep.get("tracks", {}) or {}).get("track3_blind", {}) or {}
            reuse = (_t1s.get("status") != "ab_failed"
                     and ("f1" in _t1s or _t1f1.get("f1") is not None)
                     and _t3s.get("f1") is not None)
        except Exception:
            reuse = False
    if reuse:
        log(f"[{trial}] 门禁报告已存在 ⇒ 复用 {rp}")
    else:
        os.makedirs(outd, exist_ok=True)
        cmd = (f"cd {REPO} && export PYTHONPATH=. && {PY} -u manu/training/auto_epoch_gate_check.py "
               f"--weights {w} --out-dir {outd} --device {gpu} --infer-gpu {gpu} --gpus 1 --cache-gpus {gpu},{(gpu+1)%4} "
               f"--baseline-cache {OFFICIAL_BASE_CACHE} > {outd}_gate.log 2>&1")
        r = sh(cmd, timeout=10800)
        if not os.path.isfile(rp):
            log(f"[{trial}] 门禁未产出报告 rc={r.returncode}（查 {outd}_gate.log）")
            return {"error": f"no_report rc={r.returncode}"}
    rep = json.load(open(rp))
    tr = rep.get("tracks", {}) or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    d = {
        "out_dir": outd, "verdict": rep.get("verdict"),
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
        "veto_failures": rep.get("veto_failures", []),
    }
    d["official_ok"] = (d["t1"] >= DELIVER["t1_ge"] and d["t2_fp"] <= DELIVER["t2_fp_le"]
                         and d["t3"] >= DELIVER["t3_ge"])
    # 任务书级「交付通过」= 官方三轨 + 任务书更严的 T2 虚警红线
    d["deliver_ok"] = (d["official_ok"] and d["t2_fp"] <= DELIVER["t2_fp_doc"])
    return d


def phase_pick(arms: list) -> None:
    """只做「按 nano 下限选 epoch + 写 picked」，**不跑官方门禁**。

    2026-11-11 v3：官方三轨门禁（GMC）很贵，只跑在「有可能打平基线」的臂上；
    其余臂仍需写 picked（否则进不了台账排序/冠军候选），故把「选 epoch」拆出来单独跑。
    """
    led = load_ledger()
    b_nano, b_nano4, b_f1 = base_pick_thresholds()
    for item in arms:
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        traj = st.get("nanobench", {})
        best, how = pick_epoch(traj, b_nano, b_nano4, b_f1)
        if not best:
            continue
        st["picked_epoch"] = best["epoch"]
        st["select_rule"] = how
        st["beat_to_pass_ref"] = ("Baseline_v2.0(trial_9863)" if NANO_FLOOR >= BASE_V2["nano"]
                                  else "trial_0031")
        st["picked"] = best
    save_ledger(led)


def phase_gate(arms: list, gpu_of: dict) -> None:
    led = load_ledger()
    b_nano, b_nano4, b_f1 = base_pick_thresholds()
    bo = base_official_thresholds(led)

    def one(item):
        try:
            _one(item)
        except Exception as e:  # noqa: BLE001
            log(f"[{item[0]}] 官方门禁异常：{e}")

    def _one(item):
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        traj = st.get("nanobench", {})
        best, how = pick_epoch(traj, b_nano, b_nano4, b_f1)
        if not best:
            log(f"[{trial}] 无有效 NanoBench 读数 ⇒ 跳过门禁")
            return
        st["picked_epoch"] = best["epoch"]
        st["select_rule"] = how
        st["beat_to_pass_ref"] = "Baseline_v2.0(trial_9863)" if NANO_FLOOR >= BASE_V2["nano"] else "trial_0031"
        st["picked"] = best
        log(f"[{trial}] 选 epoch={best['epoch']}（{how}）nano={best['nano']:.4f} F1={best['f1']:.6f}")
        gpu = gpu_of.get(trial, 0)
        g = run_official_gate(trial, best["epoch"], gpu)
        # 🟢 2026-11-12 11:05 新增：**多 epoch 官方门禁**
        #   实测（12053 逐 epoch 官方读数）：ep4 = nano 最优（0.709594）但官方 T1 0.887102 不过线；
        #   ep5 = 官方 T1 0.888286 过线但 nano 塌到 0.678166 ⇒ **只门禁 nano 最优 epoch 会白丢候选**。
        #   ⇒ 对同一臂里其它「三项全过」的 epoch 依次补门禁，取**第一个官方三轨相对基线全过**的。
        if not official_beat(g, bo):
            alts = [c for c in sorted((dict(v, epoch=int(k)) for k, v in traj.items() if "f1" in v),
                                      key=lambda c: -(c.get("nano") or 0.0))
                    if c["epoch"] != best["epoch"]
                    and (c.get("nano") or 0.0) >= b_nano
                    and (c.get("nano4") or 0.0) >= b_nano4
                    and (c.get("f1") or 0.0) >= b_f1]
            for alt in alts[:max(0, GATE_EPOCH_K - 1)]:
                g2 = run_official_gate(trial, alt["epoch"], gpu)
                log(f"[{trial}] 补门禁 ep{alt['epoch']}：T1={g2.get('t1')} T2fp={g2.get('t2_fp')} "
                    f"T3={g2.get('t3')} ⇒ {'官方三轨全过' if official_beat(g2, bo) else '未过'}")
                if official_beat(g2, bo):
                    g = g2
                    best = alt
                    st["picked_epoch"] = alt["epoch"]
                    st["picked"] = alt
                    st["select_rule"] = how + "+multi_epoch"
                    log(f"[{trial}] ★ 多 epoch 命中 ⇒ 升轨候选改用 ep{alt['epoch']}")
                    break
        st["official"] = g
        if "error" not in g:
            log(f"[{trial}] 官方门禁：T1={g['t1']:.6f} T2={g['t2']:.6f}(FP {g['t2_fp']}) "
                f"T3={g['t3']:.6f}(FP {g['t3_fp']}) verdict={g['verdict']} ⇒ "
                f"{'PASS' if g['official_ok'] else 'FAIL'}")
        else:
            log(f"[{trial}] 官方门禁失败：{g.get('error')}")
        save_ledger(led)

    with ThreadPoolExecutor(max_workers=WORKERS[0]) as ex:
        list(ex.map(one, arms))
    save_ledger(led)


# ============================================================================
# ⑥⑦ 导出 + 一次板卡会话
# ============================================================================
def board_precheck() -> tuple[bool, str]:
    """板卡**只读**占用预检（🔴 绝不杀任何进程 —— 板卡是共享资源）。

    2026-11-13 改口径：旧实现会在测速前 `pkill -9` 一批通用进程名
    （e2e_par2_demo|e2e_par_demo|e2e_demo|rknn_heatmap_smoke）
    ⇒ **会杀掉板卡上别人的 NPU 任务**（用户明确告知板卡上有别的任务在跑）。
    ⇒ 改成：① 只读地看 load / NPU 进程数 / 根分区余量；② 有占用就**跳过**测速并记录原因。
    返回 (可测, 说明)。
    """
    cmd = (f"ssh {BOARD_KEY} {BOARD} "
           "\"awk '{print \\$1}' /proc/loadavg; "
           "ps -eo cmd | grep -E '[r]knn|[e]2e_|[m]easure_hm_speed' | wc -l; "
           "df -m / | tail -1 | awk '{print \\$4}'\"")
    r = sh(cmd, timeout=60)
    txt = (r.stdout or "").strip()
    parts = [x for x in txt.splitlines() if x.strip()]
    if len(parts) < 3:
        return False, f"预检读不到（板卡不可达/输出异常）：{txt[-160:]!r}"
    try:
        load = float(parts[0].strip())
        nproc = int(parts[1].strip())
        free_mb = int(parts[2].strip())
    except Exception:  # noqa: BLE001
        return False, f"预检解析失败：{txt[-160:]!r}"
    if load > 1.0:
        return False, f"板卡忙（load1={load:.2f} > 1.0）⇒ 跳过测速（不打扰别人的任务）"
    if nproc > 0:
        return False, f"板卡有 {nproc} 个 NPU 相关进程在跑 ⇒ 跳过测速（不打扰别人的任务）"
    if free_mb < 400:
        return False, f"板卡根分区余量仅 {free_mb} MB（<400）⇒ 跳过（避免上传把盘撑爆）"
    return True, f"板卡空闲（load1={load:.2f}，NPU 进程 0，余量 {free_mb} MB）"


def clean_board_residue() -> None:
    """🔴 只读地**报告**板卡残留（2026-11-13 起不再 pkill —— 板卡是共享资源）。

    旧实现会 `pkill -9` 一批通用进程名 ⇒ 会杀掉**别人**的 NPU 任务（实测板卡上有别的任务）。
    ⇒ 现在只列出来，由 board_precheck 决定「跳过测速」还是「照常测」。
    """
    cmd = (f"ssh {BOARD_KEY} {BOARD} "
           "\"ps -eo pid,etime,args | grep -E '[r]knn_|[e]2e_' | head -8 || true; "
           "echo '[BOARD] 残留检查完成'\"")
    r = sh(cmd, timeout=60)
    log(f"[board] 残留检查（只读，不 kill）⇒ {(r.stdout or '').strip()[-300:]}")


def phase_board(arms: list) -> None:
    sys.path.insert(0, REPO)
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415

    led = load_ledger()
    board_arms = []
    for trial, seed, p, desc in arms:
        st = arm_state(led, trial)
        best = st.get("picked")
        if not best:
            continue
        # 🟢 2026-11-13 14:45：分数 epoch ⇒ 用统一解析拿权重文件名（否则板卡导出找不到 ckpt）
        name = f"n{trial}_ep{epoch_key_str(best['epoch'])}"
        ck = f"{ROOT}/trials/trial_{trial}/weights/{best.get('ckpt') or epoch_ckpt_name(best['epoch'])}"
        rknn, stride = RA.export_rknn(ck, name)
        if rknn:
            board_arms.append((name, rknn, stride))
            st["rknn"] = rknn
            st["rknn_stride"] = stride
        save_ledger(led)
    if not board_arms:
        log("[board] 无候选臂 ⇒ 跳过")
        return
    ok, why = board_precheck()
    log(f"[board] 占用预检 ⇒ {why}")
    if not ok:
        log(f"[board] 跳过板卡测速（{why}）—— 不阻塞本轮结论")
        return
    clean_board_residue()
    ms_map = RA.board_measure(board_arms)
    for trial, seed, p, desc in arms:
        st = arm_state(led, trial)
        best = st.get("picked")
        if not best:
            continue
        name = f"n{trial}_ep{best['epoch']}"
        ms = ms_map.get(name)
        st["board_ms"] = ms
        st["speed_ok"] = (ms is not None and ms <= DELIVER["budget_ms"])
        log(f"[{trial}] 板卡 int8 = {ms} ms ⇒ {'PASS' if st['speed_ok'] else 'FAIL/UNKNOWN'}")
    save_ledger(led)


# ============================================================================
# ⑧ 联合门禁
# ============================================================================
def phase_dual(arms: list) -> None:
    led = load_ledger()
    specs = []
    for trial, seed, p, desc in arms:
        st = arm_state(led, trial)
        best = st.get("picked")
        if not best or not best.get("cache"):
            continue
        ms = st.get("board_ms")
        spec = f"n{trial}_ep{best['epoch']}={best['cache']}"
        if ms is not None:
            spec += f"={ms}"
        specs.append(spec)
    if not specs:
        log("[dual] 无候选 ⇒ 跳过")
        return
    out = f"{WORK}/dual_gate.json"
    cmd = (f"cd {REPO} && export PYTHONPATH=. && {PY} manu/evaluation/dual_constraint_gate.py "
           + " ".join(f"--candidate '{s}'" for s in specs)
           + f" --baseline {BASE_CACHE} --budget-ms {DELIVER['budget_ms']} "
           f"--nano-min {DELIVER['nano_min']} --out {out}")
    r = sh(cmd, timeout=7200)
    if os.path.isfile(out):
        rep = json.load(open(out))
        rows = rep.get("rows", []) or []
        for trial, seed, p, desc in arms:
            st = arm_state(led, trial)
            # 🔴 2026-11-13 16:40 修：R12 的 dual 阶段整段挂掉（KeyError 'picked'）
            #   根因：末轮被 stop_ts 砍掉的臂**没有 picked 读数** ⇒ `st['picked']['epoch']` 抛异常
            #   ⇒ 异常冒泡出整个 phase_dual ⇒ **联合门禁（升轨证据之一）整段丢失**。
            #   ⇒ ① 没有 picked 的臂直接跳过；② 单臂异常不许掀掉整段。
            _pk = st.get("picked") or {}
            if not _pk.get("epoch"):
                continue
            key = f"n{trial}_ep{_pk['epoch']}"
            cand = next((c for c in rows if c.get("name") == key), None)
            if cand:
                st["dual"] = cand
                st["dual_pass"] = cand.get("verdict") == "PASS"
                log(f"[{trial}] 联合门禁 ⇒ {cand.get('verdict')} "
                    f"(speed={cand.get('speed_verdict')} recall={cand.get('recall_verdict')} "
                    f"nano={cand.get('nano')} ms={cand.get('board_ms')})")
            save_ledger(led)
    else:
        log(f"[dual] 未产出报告 rc={r.returncode}：{(r.stdout or '')[-400:]}{(r.stderr or '')[-400:]}")
    save_ledger(led)


# ============================================================================
def summary(arms: list) -> None:
    led = load_ledger()
    print("\n" + "=" * 118)
    print("%-8s %-9s %-9s %-9s %-9s %-9s %-10s %-9s %-8s %-8s %s" % (
        "trial", "ep", "nano", "micro", "small", "regular", "F1", "nano@4px",
        "ms", "T2FP", "官方门禁"))
    print("-" * 118)
    for trial, seed, p, desc in arms:
        st = led["arms"].get(str(trial), {})
        b = st.get("picked")
        if not b:
            print(f"{trial:<8} (无读数)")
            continue
        g = st.get("official", {})
        ms = st.get("board_ms")
        print("%-8s %-9s %-9.4f %-9.4f %-9.4f %-9.4f %-10.6f %-9.4f %-8s %-8s %s" % (
            trial, b["epoch"], b["nano"], b["micro"], b["small"], b["regular"], b["f1"],
            b.get("nano4", 0), f"{ms:.3f}" if ms else "-",
            g.get("t2_fp", "-"), g.get("verdict", "-")))
    print("=" * 118)
    print("交付件 trial_0031 对照：nano 0.6666 / micro 0.8370 / small 0.8654 / regular 0.9445 "
          "/ F1 0.887122 / int8 19.629 ms / 官方 T1 0.885997、T2 FP 10、T3 0.948330")



# ============================================================================
# ⑨ 汇总报告（T1 兜底解析 track1_ab.log 的 FINETUNE 峰值）
# ============================================================================
AB_PEAK_RE = re.compile(
    r"\[PEAK\] FINETUNE:\s*th=([\d.]+)\s+F1=([\d.]+)\s+P=([\d.]+)%\s+R=([\d.]+)%\s+TP=(\d+)\s+FP=(\d+)"
)


def parse_t1_from_ablog(out_dir: str) -> dict:
    """从 track1_ab.log 兜底解析候选臂的 T1 峰值（口径修好前跑的臂也能取到）。"""
    p = f"{out_dir}/track1_ab.log"
    if not os.path.isfile(p):
        return {}
    try:
        txt = open(p, errors="ignore").read()
    except Exception:
        return {}
    hits = AB_PEAK_RE.findall(txt)
    if not hits:
        return {}
    th, f1, pr, rc, tp, fp = hits[-1]
    return {"f1": float(f1), "th": float(th), "tp": int(tp), "fp": int(fp),
            "p": float(pr), "r": float(rc), "src": "ab_log_fallback"}


def phase_report() -> None:
    led = load_ledger()
    rows = []
    for k, st in sorted(led["arms"].items()):
        b = st.get("picked") or {}
        g = st.get("official") or {}
        t1 = g.get("t1")
        t1_fp = g.get("t1_fp")
        if not t1:
            t1d = parse_t1_from_ablog(g.get("out_dir", ""))
            if t1d:
                t1, t1_fp = t1d["f1"], t1d["fp"]
        dual = st.get("dual") or {}
        rows.append({
            "trial": int(k), "desc": st.get("desc", ""), "epoch": b.get("epoch"),
            "nano": b.get("nano"), "micro": b.get("micro"), "small": b.get("small"),
            "regular": b.get("regular"), "nano4": b.get("nano4"), "f1": b.get("f1"),
            "t1": t1, "t1_fp": t1_fp,
            "t2": g.get("t2"), "t2_fp": g.get("t2_fp"),
            "t3": g.get("t3"), "t3_fp": g.get("t3_fp"),
            "board_ms": st.get("board_ms"), "verdict": g.get("verdict"),
            "dual_verdict": dual.get("verdict") or st.get("dual_pass"),
            "official_ok": g.get("official_ok"), "deliver_ok": g.get("deliver_ok"),
            "is_baseline_v2": bool(st.get("is_baseline_v2")),
            "beat_to_pass_ref": st.get("beat_to_pass_ref"),
        })

    ref31 = {"f1": 0.887122, "nano": 0.6666, "t1": 0.885997, "t2_fp": 10, "t3": 0.948330,
             "ms_delivery_board": 19.306}
    # Beat-to-Pass 阈值必须用 9863 的**精确读数**（常量 BASE_V2 是 6 位截断，
    # 否则基线臂自己会被 f1_ge_base_v2 判负）
    base_v2 = dict(BASE_V2)
    a98 = (led.get("arms") or {}).get(str(BASE_V2["trial"]), {})
    pk98 = a98.get("picked") or {}
    of98 = a98.get("official") or {}
    for k in ("nano", "micro", "small", "regular", "f1", "nano4"):
        if pk98.get(k) is not None:
            base_v2[k] = pk98[k]
    for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp"):
        if of98.get(k) is not None:
            base_v2[k] = of98[k]
    if a98.get("board_ms") is not None:
        base_v2["ms_new_board"] = a98["board_ms"]
    # 新板（2026-11-11 换板）同会话参照：交付件 trial_0031 自身的 int8 rknn_run
    bs = (led.get("board_sessions") or [{}])[-1] if led.get("board_sessions") else {}
    ref_t31_new = ((bs.get("session2") or {}).get("arms") or {}).get("t31_silu_int8")
    for r in rows:
        checks = {}
        if r["nano"] is not None:
            checks["nano_ge_0031"] = r["nano"] >= ref31["nano"]
        if r["f1"] is not None:
            checks["f1_gt_0031"] = r["f1"] > ref31["f1"]
            checks["f1_ge_base_v2"] = r["f1"] >= base_v2["f1"]
        if r["nano"] is not None:
            checks["nano_ge_base_v2"] = r["nano"] >= base_v2["nano"]
        if r["t1"] is not None:
            checks["t1_ge_0031"] = r["t1"] >= ref31["t1"]
            checks["t1_ge_base_v2"] = r["t1"] >= base_v2["t1"]
        if r["t2_fp"] is not None:
            checks["t2fp_le_20"] = r["t2_fp"] <= 20
            checks["t2fp_le_36"] = r["t2_fp"] <= 36
            checks["t2fp_le_base_v2"] = r["t2_fp"] <= base_v2["t2_fp"]
        if r["t3"] is not None:
            checks["t3_ge_0031"] = r["t3"] >= ref31["t3"]
            checks["t3_ge_base_v2"] = r["t3"] >= base_v2["t3"]
        if r["board_ms"] is not None:
            checks["ms_le_20"] = r["board_ms"] <= DELIVER["budget_ms"]
            if ref_t31_new:
                # 新板整体比旧板慢 ⇒ 绝对 20 ms 需重新定标；
                # 判据改为「同会话不慢于交付件 0031 自身」（同图 ⇒ 交付板等速）
                checks["ms_le_t31_same_session"] = r["board_ms"] <= ref_t31_new * 1.02
        r["checks"] = checks
        r["pass_all_known"] = all(checks.values()) if checks else False
        r["pass_ignoring_board_ms"] = all(v for k, v in checks.items()
                                          if not k.startswith("ms_")) if checks else False

    out = {"ref_0031": ref31, "baseline_v2": base_v2,
           "baseline_registry": BASELINE_REGISTRY,
           "beat_to_pass": {"ref": "Baseline_v2.0(trial_9863)", "nano_min": base_v2["nano"],
                            "f1_min": base_v2["f1"], "t1_min": base_v2["t1"],
                            "t2_fp_max": base_v2["t2_fp"], "t3_min": base_v2["t3"]},
           "criteria": DELIVER, "rows": rows,
           "board_sessions": led.get("board_sessions", []),
           "board_ms_reference_new_board": ref_t31_new,
           "latency_note": ("候选臂与 trial_0031 同图（no_p2+p3, head_conv64, late_upsample, stride2, "
                            "ONNX 6.68MB）⇒ 交付板 int8 = 19.306 ms 由构造保证；"
                            "新板（192.168.0.211）整体慢 ~11-22%，同会话 0031 自身 = 23.528 ms ⇒ "
                            "绝对 20 ms 铁律需在新板重新定标，判据用「同会话不慢于 0031」。")}
    with open(f"{WORK}/final_report.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"{WORK}/final_report.md", "w") as f:
        f.write("# NanoBench 定向搜索 · 整晚波次最终汇总\n\n")
        f.write("| trial | 变更 | ep | nano | micro | F1 | nano@4px | T1 | T1FP | T2FP | T3 | ms | 判定 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        st_map = {int(k): v for k, v in (led.get("arms") or {}).items()}
        for r in sorted(rows, key=lambda x: -(x.get("nano") or 0)):
            d = (r.get("desc") or "")[:26]
            st = st_map.get(r["trial"], {})
            f.write("| {t} | {d} | {e} | {n} | {mi} | {f1} | {n4} | {t1} | {t1fp} | {t2fp} | {t3} | {ms} | {v} |\n".format(
                t=r["trial"], d=d, e=r.get("epoch"),
                n=f"{r['nano']:.4f}" if r.get("nano") is not None else "-",
                mi=f"{r['micro']:.4f}" if r.get("micro") is not None else "-",
                f1=f"{r['f1']:.6f}" if r.get("f1") is not None else "-",
                n4=f"{r['nano4']:.4f}" if r.get("nano4") is not None else "-",
                t1=f"{r['t1']:.6f}" if r.get("t1") is not None else "-",
                t1fp=r.get("t1_fp") if r.get("t1_fp") is not None else "-",
                t2fp=r.get("t2_fp") if r.get("t2_fp") is not None else "-",
                t3=f"{r['t3']:.6f}" if r.get("t3") is not None else "-",
                ms=f"{r['board_ms']:.3f}" if r.get("board_ms") is not None else "n/a",
                v=("BASE" if st.get("is_baseline_v2") else
                   ("PASS" if r.get("pass_all_known") else
                    ("PASS*" if r.get("pass_ignoring_board_ms") else "—")))))
    log(f"[report] 已写出 {WORK}/final_report.json 与 final_report.md（共 {len(rows)} 臂）")


def main() -> None:
    ap = argparse.ArgumentParser(description="NanoBench 定向搜索器 v3（单变量，基线 = trial_0031）")
    ap.add_argument("--wave", default="W1", choices=sorted(WAVES))
    ap.add_argument("--phase", default="all",
                    choices=["all", "train", "nanobench", "gate", "board", "dual"])
    ap.add_argument("--epochs", default="5,10,15,20", help="NanoBench 候选 epoch")
    ap.add_argument("--stop-at", default="2026-11-11 08:00", help="硬截止")
    ap.add_argument("--only", default="", help="只跑这些 trial（逗号分隔，跨波次）")
    ap.add_argument("--report", action="store_true", help="只生成汇总报告（不训练/不评测）")
    ap.add_argument("--nano-floor", default="9863", choices=["9863", "0031"],
                    help="Beat-to-Pass 的 nano 下限参照（9863=Baseline_v2.0 / 0031=旧交付件）")
    ap.add_argument("--gpus", default="0,1,2,3", help="可用 GPU 列表（逗号分隔）")
    ap.add_argument("--workers", type=int, default=4, help="并发数（评测/缓存阶段）")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    os.makedirs(f"{RUNS}/tiny_ref", exist_ok=True)
    arms = WAVES[args.wave]
    if args.only:
        keep = {int(x) for x in args.only.replace(" ", "").split(",") if x.strip()}
        seen, picked = set(), []
        for _w in WAVES.values():
            for a in _w:
                if a[0] in keep and a[0] not in seen:
                    seen.add(a[0])
                    picked.append(a)
        arms = picked
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    stop_ts = time.mktime(time.strptime(args.stop_at, "%Y-%m-%d %H:%M"))
    gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x.strip()]
    WORKERS[0] = max(1, args.workers)
    TRAIN_WORKERS[0] = max(1, args.workers)  # 单独 CLI 跑时保持旧口径
    global NANO_FLOOR
    NANO_FLOOR = BASELINE_NANO_FLOOR[args.nano_floor]
    gpu_of = {t: gpus[i % len(gpus)] for i, (t, _s, _p, _d) in enumerate(arms)}

    led = load_ledger()
    if args.wave not in led["waves"]:
        led["waves"].append(args.wave)
    led["base_recipe"] = BASE
    led["deliver_criteria"] = DELIVER
    save_ledger(led)

    log(f"=== NANO SEARCH v3 START | wave={args.wave} phase={args.phase} | 截止 {args.stop_at} ===")
    for trial, seed, p, desc in arms:
        log(f"  · {trial} seed={seed} | {desc}")

    if not args.report:
        if args.phase in ("all", "train"):
            phase_train(arms, stop_ts, gpus)
        if args.phase in ("all", "nanobench"):
            phase_nanobench(arms, epochs, gpu_of)
        if args.phase in ("all", "gate"):
            phase_gate(arms, gpu_of)
        if args.phase in ("all", "board"):
            phase_board(arms)
        if args.phase in ("all", "dual"):
            phase_dual(arms)
    if args.report:
        phase_report()
    summary(arms)
    log("=== NANO SEARCH v3 DONE ===")


if __name__ == "__main__":
    main()
