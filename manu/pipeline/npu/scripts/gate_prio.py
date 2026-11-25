#!/usr/bin/env python3
"""门禁优先级选取器 v2（2026-11-18 11:25）：按「家族能过几项」排队，而不是按 bench 边际。

两个坑的修法：
  ① **默认按 bench margin 排序** ⇒ H1 主攻臂（bench 边际仅 +0.08σ）会被高边际**死路**臂挤掉
     ⇒ 改用「官方三轨通过数」作为主排序。
  ② **通道之间抢同一个臂** ⇒ 每个通道用 `mkdir` 原子占号（claim），已被别的通道占的臂跳过。

打分：
  主键 = 同**成员集**的已门禁兄弟臂里「三轨通过数」的最大值（<2 ⇒ 整个成员集判死，不再浪费门禁机时）
  次键 = 同**成员集 + 权重桶**的通过数最大值
  末键 = bench 边际（越大越好）、sid（旧优先）

🔴🔴🔴 2026-11-23 10:45 新增 **T1 血统插队 / 死路降权**（最高优先级键 `t1lin`）
────────────────────────────────────────────────────────────────────────
【实测依据（2026-11-23 10:20 台账快照，13,793 臂）】
  · 全库「六项全过 ∧ 至少一轴严格优」= **0 臂** ⇒ 没有任何可升轨模型；
  · 已门禁 125 臂的 FAM5+供体家族（成员集 = 1608003/1612001/1530006/1543006/1603002/1623003）
    **dT1 ∈ [−22.23, −3.68]e-5，无一触到 0** ⇒ **该成员集已被证伪**，
    但它 1,259 个未门禁兄弟臂仍占着 strict 档队首 ⇒ **门禁机时 8 倍浪费**；
  · 桥接家族（成员集 = 1100275@1 + 1102004@1 + 1612001@5）已有 2 个已门禁兄弟臂
    **dT1 = +2.81e-5 / +2.35e-5（严格为正）** ⇒ 这是**全库唯一被证明能顶开 T1 的血统**，
    但它的未门禁兄弟臂（212xxxx 段）**三轴不全过 ⇒ 落在 near 档末尾 ⇒ 永远打不到**。

【判据】
  `t1lin = 1`  成员集的已门禁兄弟臂里 **max(dT1) > 0** ⇒ 该血统已被证明能顶开 T1 ⇒ **插到队首**
  `t1lin = -1` 成员集的已门禁兄弟臂 **≥ T1LIN_DEAD_MIN_N(5) 个且 max(dT1) < 0**
              ⇒ 该血统已被证伪 ⇒ **降到队尾**
  `t1lin = 0`  证据不足 ⇒ 保持原排序

【为什么必须做】铁律二十八「否定性判决先证伪搜索空间」在**门禁队列**上的直接应用：
  门禁是 8 倍瓶颈（≈9 min/臂 vs 合成 70 s/臂）⇒ 队列排序错一次 = 损失数小时机时。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu")
from trackb_core import build_gate_map  # noqa: E402  台账 official 段恒为 null ⇒ 必须读磁盘

# 🔴🔴🔴 2026-11-23 10:30 —— **分辨率地板过滤器（maxΔ）**
#   实测（`weight_space.py` 展平口径，台账 13765 臂）：
#     · maxΔ < 0.012            ⇒ nano/nano4 读数与基线**逐位相同** ⇒ 纯烧 GPU
#     · 0.012 ≤ maxΔ ≤ 0.030     ⇒ nano / T1 / T3 **最大增益 = 0**（单调损失区）
#     · 0.03 ≤ maxΔ ≲ 0.16      ⇒ 「活带」（唯一可能动读数的区间）
#   现场（2026-11-23 09:50 台账快照）：待打「bench 三轴 ≥ 基线」的 64 臂里
#     · **55 臂 maxΔ < 0.012** ⇒ 读数按构造与基线同格 ⇒ **门禁机时纯浪费**（≈8 小时）
#     · 只有 **9 臂**在活带（2010414/1910123/1890032/1940136/1940148/1920125/1930147/1940125/1850026）
#   ⇒ 本过滤器把「maxΔ < 阈值」的臂**直接排出队列**，把门禁机时全部让给活带与扫网新臂。
#   ⚠️ 只对**汤臂**生效（训练臂没有 soup_members ⇒ 无展平向量 ⇒ 一律放行）。
#   ⚠️ 可用环境变量 `GATE_MIN_MAXDELTA=0` 关闭（回退开关）；实测值 0.030。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/scripts")
try:
    import weight_space as _WS
except Exception:  # noqa: BLE001
    _WS = None

GATE_MIN_MAXDELTA = float(os.environ.get("GATE_MIN_MAXDELTA", "0.030"))
_FLOOR_STATE = {"fl": None, "base": None, "skip": [], "n": 0}

W = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = os.path.join(W, "nano_search_ledger.json")
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
CLAIM = os.path.join(W, ".gate_claim")
# 🔴🔴 2026-11-23 12:5x —— CLAIM_TTL 45min → 100min
#   坑：单轮 8 臂 × ~9.5min ≈ 76min > 原 45min ⇒ 同一臂在 TTL 过期后会被两条通道同时抢
#   （`night_soup_batch.py:233-235` 只在门禁目录已存在时跳过 ⇒ 存在创建竞态窗口）
#   ⇒ TTL 必须 > 单轮时长；配合 `timeout 3600→5400`（见 gate_loop_d1*.sh / t1close_gate.sh）
CLAIM_TTL = 100 * 60

# 🔴 2026-11-19 09:50 **门禁入队口径放宽**（周末算力不许空转）
#   现场证据：基线 nano .7242439 是全盘最高 ⇒ 「nano/nano4/f1 三轴同时 ≥ 基线」**恒空**
#   ⇒ 三个门禁通道每轮都「无候选」⇒ 实测 4 张卡 GPU util 0%、显存 11MiB ⇒ 90 小时空转。
#   ⇒ 改成「近线容差」入队 + 家族加权 + 近线距离排序（严格臂永远排前面）。
TOL_NANO = 1.5e-3      # nano 允许落后基线 1.5e-3
TOL_NANO4 = 3.0e-3     # nano4 允许落后基线 3.0e-3
TOL_F1 = 4.0e-4        # F1 允许落后基线 4.0e-4
STALE_GATE_SEC = 5400  # 有门禁目录但没报告且 mtime 新于 90min ⇒ 正在打，不动它

# ============================================================================
# 🔴🔴🔴 2026-11-20 R1修复 —— **训练臂（无 soup_members）不再被门禁队列结构性排除**
# ============================================================================
#
# 【实测证据】2026-11-20 14:39 服务器台账快照（$W/nano_search_ledger.json，12,096 臂）：
#   ① **5,169 个臂没有 `soup_members`**（= 全部训练臂/抛光臂）⇒ 老代码
#      `if not pk or not mem: continue`（老 :145-146）把它们**整体挡在官方门禁之外**
#      ⇒ 训练臂的 **T1 / T2_fp / T3 永远量不出来** ⇒ 六项判据里的三轨对训练臂**结构性缺失**
#      ⇒ 训练臂**永远进不了升轨链**（门禁是六项读数的唯一来源）。
#   ② 老代码把门禁目录**写死**成 `NANO_{sid}_ep5_official`，而训练臂的门禁目录是
#      `NANO_{sid}_ep{picked_epoch}_official`（实测：12131→ep4、1102033→ep1、1300001→ep1、999821→ep1）
#      ⇒ 即使把训练臂放进队列，也会被误判成「有目录但没报告」而**反复重排队**。
#   ③ 训练臂的**三轴分布**（1,881 个有台账读数的训练臂，基线 = Baseline_v3.1 / trial 1990055）：
#      d_nano ≥ −1.5e-3 有 **101** 个；d_nano4 ≥ −3.0e-3 有 **254** 个；d_f1 ≥ −4.0e-4 有 **614** 个；
#      但**三轴同时**满足容差的只有 **2** 个 ⇒ **F1 是真正的卡点**（训练臂普遍 nano/nano4 达标、
#      总 F1 差 1e-2 量级）⇒ 三轴容差不放宽的话训练臂**几乎不可能**进队列。
#      （容差 ×2 → 8 个、×4 → 70 个、×8 → 291 个 ⇒ 见 `TRAIN_EXPLORE_MULT`。）
#
# 【改法】
#   ① 候选循环：**汤臂（`soup_members` 非空）逻辑一字不改**；训练臂走独立分支：
#      · 门禁 epoch = `picked_epoch`（或 `picked.epoch`）—— **不再写死 5**；
#      · 必须**存在对应 ckpt**（`trials/trial_{sid}/weights/{ckpt_name}.pt`）才允许入队
#        （没有权重 ⇒ 门禁脚本必然失败 ⇒ 白占机时）；
#      · 家族键用「配方指纹」（抛光用 `warm_key` 配方段、训练臂用 `init`）⇒ 训练臂之间也能
#        共享 `mem_best` 的「已证能过 ≥2 轨」信号（老代码对训练臂一律 `alive=None` ⇒ tier 恒 1）。
#   ② **优先级严格低于汤臂**：候选元组新增 `soup_rank`（汤臂 1 / 训练臂 0）并作为**最前**分档键
#      ⇒ **任何汤臂都排在任何训练臂之前**。分档（`TRAIN_TIER`）：
#        `TRAIN_TIER=1`（**默认**）⇒ 档位 0 strict 汤 / 1 strict 训练 / 2 near 汤 / 3 near 训练
#        `TRAIN_TIER=0` ⇒ 档位 0 strict 汤 / 1 near 汤 / 2 strict 训练 / 3 near 训练（字面最保守）
#      为什么默认 1：实测 near 段汤臂有 **3,143** 个（每臂门禁 ~740 s）⇒ 排在其后等于**永不门禁**；
#      而 near 段汤臂**结构上不可能升轨**（三轴 < 基线 ⇒ 七项门禁必败）⇒ 让位**不损失任何可升轨票**，
#      却能让训练臂**第一次**拿到 T1/T2/T3 读数（下一轮训练必需的反馈信号）。
#      ⇒ **成熟前沿（strict 汤臂）永远排在训练臂之前**，「不挤掉成熟前沿」这条严格满足。
#   ③ 训练臂的**探索档**（`TRAIN_EXPLORE_MULT`）：三轴容差放宽 N 倍，仅作**末位补位**
#      （前三档用尽才轮到）⇒ 绝不挤占汤臂机时。
#
# 【影响面】只影响「没有 soup_members 的臂」的**入队资格**与**排序位置**；
#   **汤臂的候选集合、排序键、相对顺序完全不变**（验证脚本已做 before/after 逐位比对）。
# 【回退】本文件**未被 git 跟踪** ⇒ 回退用交付说明里的备份副本覆盖；
#   或把 `TRAIN_TIER` 设为 0 且 `TRAIN_EXPLORE_MULT` 设为 1（收窄到「仅 strict 训练臂」）。
TRAIN_TIER = 1             # training-arm tier: 1 = after strict soup / before near soup; 0 = after all soup
TRAIN_EXPLORE_MULT = 4     # training-arm explore tier: three-axis tolerance multiplier for the training-arm explore tier (1 = off)

# 🔴 2026-11-23 12:20 —— 汤臂 near 档的 f1 硬门槛开关（见下方 near_ok 注释）；0 = 回到老口径
GATE_NEAR_STRICT = int(os.environ.get("GATE_NEAR_STRICT", "1"))


def _epoch_key(ep) -> str:
    """epoch -> gate dir key (same convention as `nano_search.epoch_key_str`: 4 -> '4', 2.5 -> '2.5')."""
    try:
        f = float(ep)
    except (TypeError, ValueError):
        return str(ep)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def _ckpt_name(ep) -> str:
    """epoch -> weight file name (same convention as `nano_search.epoch_ckpt_name`)."""
    try:
        f = float(ep)
    except (TypeError, ValueError):
        return f"epoch{ep}.pt"
    if abs(f - round(f)) < 1e-9:
        return f"epoch{int(round(f))}.pt"
    i = int(f // 1)
    return f"sub_epoch{i}_{f - i:g}.pt"


def _arm_gate_epoch(st: dict, pk: dict):
    """Gate epoch of a training arm = the epoch the trainer picked (`picked_epoch` / `picked.epoch`)."""
    v = st.get("picked_epoch")
    if v is not None:
        return v
    v = pk.get("epoch")
    if v is not None:
        return v
    return None


def _ckpt_path(sid: int, ep) -> str:
    """Training-arm weight path (`ROOT/trials/trial_{sid}/weights/{ckpt_name}`)."""
    return os.path.join(ROOT, "trials", f"trial_{int(sid)}", "weights", _ckpt_name(ep))


def _train_family_key(st: dict) -> tuple:
    """Family key of a training arm (same convention as `family_key.arm_family_key`:
    polish -> recipe segment, training -> init).

    Returns ("train", <key>) => can NEVER collide with a soup mkey (tuple of ints).
    """
    p = st.get("params") or {}
    wk = str(p.get("warm_key") or "")
    if wk and "|" in wk:
        cfg = re.sub(r"\|rep\d+$", "", wk.split("|", 1)[-1])
        return ("train", "polish:" + cfg)
    return ("train", "base:" + str(p.get("init") or ""))



def crit_pass(of, base):
    t1 = of.get("t1")
    t3 = of.get("t3")
    t2 = of.get("t2_fp")
    p = 0
    if t1 is not None and t1 >= base["t1"]:
        p += 1
    if t3 is not None and t3 >= base["t3"]:
        p += 1
    if t2 is not None and t2 <= base["t2_fp"]:
        p += 1
    return p


def margin(pk, base):
    return ((pk.get("nano", 0) - base["nano"]) / 0.006
            + (pk.get("nano4", 0) - base["nano4"]) / 0.006
            + (pk.get("f1", 0) - base["f1"]) / 0.0008)


def _has_pos_axis(st, of, base) -> bool:
    """该臂是否**有任何一轴**严格优于基线（T2 虚警越少越好 ⇒ 反向比较）。

    用于判「一个成员集是不是全轴都打平的死路」：
    只有全轴打平才算死路；只要有一轴严格更好，就仍可能是候选（例：T3 上探臂）。
    """
    pk = st.get("picked") or {}
    for k in ("nano", "nano4", "f1"):
        v = pk.get(k)
        if v is None:
            continue
        if float(v) > float(base[k]) + 1e-12:
            return True
    for k in ("t1", "t2", "t3"):
        v = of.get(k)
        if v is None:
            continue
        if float(v) > float(base[k]) + 1e-12:
            return True
    v = of.get("t2_fp")
    if v is not None and int(v) < int(base.get("t2_fp") or 0):
        return True
    return False


# 🔴🔴 2026-11-23 11:40 —— **各轴量化台阶（实测，勿改）**
#   读数不是连续量，而是「整数 TP/FP/FN 的有理函数」⇒ 每轴有一个最小台阶：
#     nano  ≈ 2.363e-4  nano4 ≈ 2.363e-4
#     f1    ≈ 1.90e-5   t1   ≈ 1.85e-5（T1 集约 43.5 万样本 ⇒ 1 个 TP = 4.6e-6~1.9e-5）
#     t3    ≈ 5.20e-5   t2_fp = 1 个虚警（整数）
#   ⇒ 「离六项全过还差几个台阶」是一个**可比的整数/半整数**，比比绝对差值有用得多。
# 🔴🔴 2026-11-23 12:05 —— **修 _Q_NANO 写反的 bug**（原表 nano=4.726e-4 / nano4=2.363e-4）
#   实测 2287 个已门禁臂的 Δ 分布：|Δnano| 的**最小非零台阶 = 2.363e-4**（近失族 114 臂命中），
#   |Δnano4| 的最小台阶也是 2.363e-4（nano4 是 nano 的子集 ⇒ 一次检出同时影响两轴，
#   故两轴台阶相同；原表把 nano 写成 4.726e-4 = **大了 2 倍** ⇒ 缺口被低估、排序失真）。
_Q_NANO, _Q_NANO4, _Q_F1, _Q_T1, _Q_T3 = 2.363e-4, 2.363e-4, 1.90e-5, 1.85e-5, 5.20e-5

# 🔴🔴 2026-11-23 12:05 —— **fam_gap 语义修正用的两个常量**
#   _GAP_DEAD  = 「该族已证伪：六项全过但**零头寸**（没有任何一轴严格优于基线）」
#                ⇒ 它离「可升轨」是**无穷远**，绝不能记成 gap=0（否则霸占队首）。
#   _GAP_NEW   = 新血统（无任何已门禁兄弟臂）时，在自身 bench 缺口上加的「门禁轴未知」代价。
_GAP_DEAD = 99.0
_GAP_NEW = 1.0


def _strict3(pk, base) -> bool:
    """bench 三轴（nano/nano4/f1）是否**同时 ≥ 基线** ⇒ 该臂才可能补齐六项。"""
    pk = pk or {}
    for k in ("nano", "nano4", "f1"):
        v = pk.get(k)
        if v is None or float(v) < float(base[k]) - 1e-12:
            return False
    return True


def _gap6(pk, of, base) -> float:
    """离「六项全过」还差多少个量化台阶（欠 t2_fp 记 1 个台阶；无读数的轴不计）。"""
    g = 0.0
    for k, q in (("nano", _Q_NANO), ("nano4", _Q_NANO4), ("f1", _Q_F1)):
        v = (pk or {}).get(k)
        if v is not None and float(v) < float(base[k]) - 1e-12:
            g += (float(base[k]) - float(v)) / q
    for k, q in (("t1", _Q_T1), ("t3", _Q_T3)):
        v = (of or {}).get(k)
        if v is not None and float(v) < float(base[k]) - 1e-12:
            g += (float(base[k]) - float(v)) / q
    v = (of or {}).get("t2_fp")
    if v is not None and int(v) > int(base.get("t2_fp") or 0):
        g += 1.0
    return g


def main():
    chan = sys.argv[1] if len(sys.argv) > 1 else "x"
    # 🔴 --dry：只打印不建 claim（claim 会挡住真正的门禁通道 ⇒ 调试绝不能带 claim 跑）
    DRY = "--dry" in sys.argv
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    led = json.load(open(LEDGER))
    base = led["baseline_v2"]
    arms = led.get("arms") or {}

    # 🟢🟢🟢 2026-11-19 15:55 —— **T3 上探臂插队（唯一有理论根据的「严格更优」方向）**
    #   实测（bw10 族 24 臂）：**T3 随 1530006@5 的权重单调升**
    #     w3=.2016 → .9488335 ； w3=.2116 → **.9488856（= 基线，当前测到的最大值）**
    #     ⇒ w3 > .2116 ⇒ **T3 严格优于基线**（T3 是六项里唯一还能动的轴）
    #   而 nano/nano4/F1 在该族**恒等于基线**（.7242439/.6609168/.8906832）、
    #        T1 是平台（顶格 .8909239 = 基线值）⇒ **只有 T3 能拿到严格更优**。
    #   现场（15:55）：**66 个 w3>.2116 的臂已产出，但一个都没进门禁**
    #        ⇒ 门禁机时全花在「只能打平」的臂上（排序键退化成 sid 升序 ⇒ 先打老臂）。
    #   修法：把 **w3 > 基线 w3** 的臂提到队首（t3win=1），其次按 |Δw3| 升序
    #        （离基线越近 ⇒ T1 越可能留在顶格 ⇒ 六项全过概率越高）。
    BASE_W3 = 0.2116
    _barm = arms.get(str(base.get("trial"))) or {}
    _bmem = _barm.get("soup_members") or []
    _bw = _barm.get("soup_weights") or []
    for _i, _m in enumerate(_bmem):
        try:
            if int(_m[0]) == 1530006 and int(_m[1]) == 5:
                BASE_W3 = float(_bw[_i])
                break
        except Exception:  # noqa: BLE001
            pass

    # 🟢🟢 2026-11-19 17:15 —— **同成员集限定**：只有「基线同成员集」的臂才谈得上「调权重求更优」
    #   （别的家族的臂即使 w3 大也没意义：它们本来就 T1/T3 差，是死路臂）
    BASE_KEY = (tuple(sorted((int(m[0]), int(m[1])) for m in _bmem)) if _bmem else None)
    # 基线权重指纹（成员集 + 权重全同 ⇒ 与基线是**同一个模型** ⇒ 门禁结论必然相同 ⇒ 零增量）
    BASE_WT = tuple(round(float(x), 9) for x in (_bw or []))

    mem_best = {}
    bucket_best = {}
    bucket_n = {}
    # 🔴🔴🔴 2026-11-23 10:45 —— T1 血统证据表（成员集 → max dT1 / 已门禁兄弟臂数）
    lin_t1 = {}      # mkey -> max(dT1)  over 已门禁兄弟臂
    lin_n = {}       # mkey -> 已门禁兄弟臂个数
    lin_pos = {}     # mkey -> 1 若该成员集里**存在**任何一轴严格优于基线的已门禁臂
    # 🔴🔴 2026-11-23 11:40 新增两张表（**修两处误判的关键证据**）：
    #   lin_strict[mkey] = 该成员集里存在「bench 三轴全 ≥ 基线」的已门禁兄弟臂
    #        ⇒ 该血统**有能力**补齐六项（缺的是 T1/T3）⇒ 不是死路
    #   lin_gap[mkey]    = 该成员集已门禁兄弟臂里「离六项全过的最小台阶数」
    #        ⇒ 用来把**离升轨最近的族**排到队首（实测：近失族 1.7~3.0 台阶，
    #          而桥接族 nano4 欠 5 个台阶 ⇒ 桥接族永远不可能 6/6）
    lin_strict = {}
    lin_gap = {}
    # 🔴🔴 2026-11-23 12:05 新增 lin_tie[mkey]：
    #   = 该成员集里存在「六项全过却**零头寸**」（无任何一轴严格优于基线）的已门禁兄弟臂。
    #   ⇒ 该族**永远不可能产出可升轨臂**（读数与基线同格）⇒ fam_gap 记 _GAP_DEAD。
    #   ⚠️ 旧实现把这类兄弟臂算成 gap=0，而**基线臂 1990055 本身就在基线族里**（t2_fp=3 ⇒ gap=0）
    #      ⇒ 基线族 1179 个纯打平臂全部 fam_gap=0 ⇒ 霸占 stage 2 前排，
    #      把 132 个「只欠 T1 一轴」的近失族臂挤到 rk 1400+ ⇒ 门禁机时空转。
    lin_tie = {}
    T1LIN_DEAD_MIN_N = 5   # 证伪所需的最少样本量（<5 ⇒ 证据不足，不降权）
    gmap = build_gate_map()          # ★ 台账 official 段恒为 null ⇒ 官方三轨只能读磁盘门禁报告
    for sid, st in arms.items():
        of = st.get("official") or {}
        mem = st.get("soup_members")
        if not mem:
            continue
        if of.get("t1") is None or of.get("t3") is None:
            g = gmap.get(int(sid))
            if g:
                of = {"t1": g["t1"], "t3": g["t3"], "t2_fp": g["t2_fp"]}
        if of.get("t1") is None or of.get("t3") is None:
            continue
        mkey = tuple(sorted((int(m[0]), int(m[1])) for m in mem))
        p = crit_pass(of, base)
        mem_best[mkey] = max(mem_best.get(mkey, 0), p)
        _d1 = float(of["t1"]) - float(base["t1"])
        if _d1 > lin_t1.get(mkey, -1e9):
            lin_t1[mkey] = _d1
        lin_n[mkey] = lin_n.get(mkey, 0) + 1
        if not lin_pos.get(mkey) and _has_pos_axis(st, of, base):
            lin_pos[mkey] = 1
        _pk0 = st.get("picked") or {}
        if _strict3(_pk0, base):
            lin_strict[mkey] = 1
        _g0 = _gap6(_pk0, of, base)
        # 🔴🔴 2026-11-23 12:05 —— **纯打平兄弟臂不作为「近失距离」证据**
        #   判据：没有任何一轴严格优于基线（_has_pos_axis=False）⇒ 零头寸 ⇒ 永不可升轨。
        if _has_pos_axis(st, of, base):
            if _g0 < lin_gap.get(mkey, 1e9):
                lin_gap[mkey] = _g0
        else:
            lin_tie[mkey] = 1
        fam = str(st.get("soup_family") or "")
        bucket_best[fam] = max(bucket_best.get(fam, 0), p)
        bucket_n[fam] = bucket_n.get(fam, 0) + 1

    # R1 FIX 2026-11-20: training arms also feed `mem_best`.
    #   Without this, a training arm's `alive` is always None => tier is always 1
    #   => the "this member set is proven dead (<2 tracks)" signal is lost for training arms.
    #   (Soup-arm logic above is untouched.)
    for sid, st in arms.items():
        if st.get("soup_members"):
            continue
        of = st.get("official") or {}
        if of.get("t1") is None or of.get("t3") is None:
            g = gmap.get(int(sid))
            if g:
                of = {"t1": g["t1"], "t3": g["t3"], "t2_fp": g["t2_fp"]}
        if of.get("t1") is None or of.get("t3") is None:
            continue
        mkey = _train_family_key(st)
        p = crit_pass(of, base)
        mem_best[mkey] = max(mem_best.get(mkey, 0), p)
        _d1 = float(of["t1"]) - float(base["t1"])
        if _d1 > lin_t1.get(mkey, -1e9):
            lin_t1[mkey] = _d1
        lin_n[mkey] = lin_n.get(mkey, 0) + 1
        if not lin_pos.get(mkey) and _has_pos_axis(st, of, base):
            lin_pos[mkey] = 1
        _pk1 = st.get("picked") or {}
        if _strict3(_pk1, base):
            lin_strict[mkey] = 1
        _g1 = _gap6(_pk1, of, base)
        if _has_pos_axis(st, of, base):
            if _g1 < lin_gap.get(mkey, 1e9):
                lin_gap[mkey] = _g1
        else:
            lin_tie[mkey] = 1

    # 🔴 2026-11-18 17:40：**Track-B 家族兄弟插队**
    #   升轨工具 promote_trackb 要求「同家族 Track-B 合格臂 ≥ 2」才有复现票。
    #   ⇒ 若某家族已有 1 个五轴全过（T2_fp<=5）的臂，它**同家族的所有兄弟臂**都必须优先门禁，
    #     否则票数永远 =1 ⇒ 永远升不了轨（今晚 1605008 就是卡在这里）。
    TRACKB_FAMS = set()
    for _sid, _st in arms.items():
        _of = _st.get("official") or {}
        if _of.get("t1") is None or _of.get("t3") is None:
            _g = gmap.get(int(_sid))
            if _g:
                _of = {"t1": _g["t1"], "t3": _g["t3"], "t2_fp": _g["t2_fp"]}
        if _of.get("t1") is None or _of.get("t3") is None:
            continue
        _pk = _st.get("picked") or {}
        if not (_pk.get("nano", 0) >= base["nano"] and _pk.get("nano4", 0) >= base["nano4"]
                and _pk.get("f1", 0) >= base["f1"]):
            continue
        if _of["t1"] >= base["t1"] and _of["t3"] >= base["t3"] and (_of.get("t2_fp") or 99) <= 5:
            TRACKB_FAMS.add(str(_st.get("soup_family") or ""))

    os.makedirs(CLAIM, exist_ok=True)
    now = time.time()
    cands = []
    for sid, st in arms.items():
        of = st.get("official") or {}
        if of.get("t1") is not None and of.get("t3") is not None:
            continue
        pk = st.get("picked") or {}
        mem = st.get("soup_members")
        w = st.get("soup_weights")
        is_soup = bool(mem)
        if not pk:
            continue
        # R1 FIX 2026-11-20: training arms (no soup_members) are admitted too.
        #   gate epoch = the epoch the trainer picked (NOT hard-coded 5), and the
        #   matching ckpt must exist, otherwise the gate script cannot run at all.
        gate_ep = "5"
        if not is_soup:
            gate_ep = _arm_gate_epoch(st, pk)
            if gate_ep is None:
                continue
            if not os.path.isfile(_ckpt_path(int(sid), gate_ep)):
                continue
        strict_ok = (pk.get("nano", 0) >= base["nano"] and pk.get("nano4", 0) >= base["nano4"]
                     and pk.get("f1", 0) >= base["f1"])
        # 🔴🔴🔴 2026-11-23 12:20 —— **near 档收紧**（实测能省掉 ~50% 的门禁机时）
        #   权威判据（`promote_candidate.py` L206-234，已复核）：nano / nano4 / f1 三轴
        #   都是**硬门槛**（`c["f1"] = pk["f1"] >= base["f1"]`）⇒ **f1 < 基线 ⇒ 六项必败**。
        #   现场：215xxxx「口袋」臂（gen_pocket_recipes.py）的 6 个锚点 dF1 = −1~−4 量子
        #   ⇒ **全部注定失败**，却占掉了 4 条门禁通道里的一整条（实测在打 2150007/2150011~16/2150020）。
        #   ⇒ 汤臂的 near 档改成：**f1 必须 ≥ 基线（逐位）**，nano/nano4 保留原容差
        #      （bench 复测有 ±1~2 量子的抖动 ⇒ 留后路）。
        #   ⇒ 想回退：GATE_NEAR_STRICT=0
        if is_soup and GATE_NEAR_STRICT:
            near_ok = (pk.get("nano", 0) >= base["nano"] - TOL_NANO
                       and pk.get("nano4", 0) >= base["nano4"] - TOL_NANO4
                       and pk.get("f1", 0) >= base["f1"])
        else:
            near_ok = (pk.get("nano", 0) >= base["nano"] - TOL_NANO
                       and pk.get("nano4", 0) >= base["nano4"] - TOL_NANO4
                       and pk.get("f1", 0) >= base["f1"] - TOL_F1)
        if not (strict_ok or near_ok):
            # R1 FIX 2026-11-20: training-arm EXPLORE tier (wider tolerance).
            #   Last-resort backfill only => never steals machine time from soup arms.
            if is_soup or TRAIN_EXPLORE_MULT <= 1:
                continue
            near_ok = (pk.get("nano", 0) >= base["nano"] - TOL_NANO * TRAIN_EXPLORE_MULT
                       and pk.get("nano4", 0) >= base["nano4"] - TOL_NANO4 * TRAIN_EXPLORE_MULT
                       and pk.get("f1", 0) >= base["f1"] - TOL_F1 * TRAIN_EXPLORE_MULT)
            if not (strict_ok or near_ok):
                continue
        # 🔴🔴 2026-11-23 10:30 —— **分辨率地板过滤**（见文件头注释）
        #   展平后 maxΔ < 0.030 ⇒ nano/T1/T3 **最大增益 = 0** ⇒ 排队也是白排。
        if is_soup and _WS is not None and GATE_MIN_MAXDELTA > 0:
            if _FLOOR_STATE["fl"] is None:
                _FLOOR_STATE["fl"] = _WS.Flattener(arms)
                _FLOOR_STATE["base"] = _FLOOR_STATE["fl"].flatten(int(base.get("trial")), 1.0)
            _bv = _FLOOR_STATE["base"]
            if _bv:
                _vec = _FLOOR_STATE["fl"].flatten(int(sid), 1.0)
                if _vec is not None:
                    _mx, _l1 = _WS.max_delta(_vec, _bv)
                    if _mx < GATE_MIN_MAXDELTA:
                        _FLOOR_STATE["skip"].append(int(sid))
                        continue
        # 🔴 近线距离（越小越接近六项全过）⇒ 排序主键之一
        near_dist = (max(0.0, base["nano"] - (pk.get("nano") or 0)) / 1e-3
                     + max(0.0, base["nano4"] - (pk.get("nano4") or 0)) / 1e-3
                     + max(0.0, base["f1"] - (pk.get("f1") or 0)) / 1e-4)
        gdir = os.path.join(ROOT, "gate", f"NANO_{int(sid)}_ep{gate_ep}_official")
        if os.path.isdir(gdir):
            # 🔴 有门禁目录却**没有 gate_report.json** ⇒ 结论不可信（infra 失败/被打断）
            #   ⇒ 放回队列重打；但「正在打」的目录 mtime 很新 ⇒ 不能动它。
            if os.path.isfile(os.path.join(gdir, "gate_report.json")):
                continue
            try:
                if (now - os.path.getmtime(gdir)) < STALE_GATE_SEC:
                    continue
            except OSError:
                continue
        mkey = (tuple(sorted((int(m[0]), int(m[1])) for m in mem)) if is_soup
                else _train_family_key(st))
        alive = mem_best.get(mkey)
        fam = str(st.get("soup_family") or "")
        # 排序口径：① 已证「能过 ≥2 轨」的成员集优先 ② 同成员集里**没被门禁过的桶**优先（探索优先）
        #        ③ 桶内已知通过数 ④ bench 边际 ⑤ 旧的先来
        # \U0001f525 2026-11-18 12:35\uff1a**\u5168\u65b0\u6210\u5458\u96c6\uff08alive is None\uff09\u4e0e\u300c\u5df2\u8bc1\u22652\u8f68\u300d\u540c\u7ea7**\uff08tier=1\uff09\u3002
        #   \u539f\u56e0\uff1a\u65b0\u6ce2\uff08h4 \u7684 epoch \u53d8\u4f53\u3001d5c \u7684 T3 \u5bcc\u5b50\u81c2\uff09\u662f\u6709\u4e25\u683c\u5b9a\u5411\u8bbe\u8ba1\u7684\uff0c
        #   \u4f46\u5b83\u4eec\u7684\u6210\u5458\u96c6\u6ca1\u6709\u5386\u53f2\u8bc1\u636e \u21d2 \u88ab\u6392\u5230\u300c\u5df2\u8bc1\u6b7b\u8def\u300d\u4e4b\u540e \u21d2 \u767d\u767d\u6d6a\u8d39\u95e8\u7981\u673a\u65f6\u3002
        #   \u5df2\u8bc1\u6b7b\u8def\uff08alive < 2\uff09\u4ecd\u4fdd\u6301 tier=-1\uff0c\u4fdd\u8bc1\u4e0d\u53cd\u5411\u6d6a\u8d39\u3002
        tier = 1 if (alive is None or alive >= 2) else -1
        # 🔴 2026-11-19 10:05 **「三轴全过」臂必须插到最前**
        #   六项判据里 nano/nano4/f1 是 bench 三轴（合成后立刻就有），
        #   T1/T2/T3 必须靠门禁（慢）⇒ **三轴全过的臂是唯一可能补齐全六项的候选**，
        #   必须优先占门禁机时；否则会被「F1 单轴超基线但 T1 差」的臂挤掉（实测 1608000/1612000 被挤掉）。
        strict_bench = 1 if strict_ok else 0
        # 🟢🟢 2026-11-23 09:00 —— **「严格三轴 且 非基线克隆」臂插到队首**
        #   实测（2026-11-23 08:48 全盘审计，13765 臂 / 16264 个 (臂,epoch)）：
        #     · 已有门禁报告的 2250 个 ⇒ 六项全过 57 个，**全部是基线逐位克隆** ⇒ 零增量；
        #     · 无门禁报告的 14014 个 ⇒ 严格三轴全过 68 个，其中 55 个是基线克隆，
        #       **只剩 13 个真候选**（1850026/1850032/1890032/1910071/1910123/1920125/
        #        1930087/1930093/1930147/1940125/1940136/1940148/2010414）；
        #     · 而老排序键 `t3win` 压倒一切 ⇒ 队首永远是「基线同成员集的 T3 上探臂」
        #       （2010193/2010194/…）⇒ **真候选排在几百名之后 ⇒ 永远打不到**。
        #   修法：新增**最高优先级**键 novel_bench = 严格三轴全过 且 与基线不是同一个模型
        #         （成员集+权重全同 ⇒ 克隆；bench 读数与基线逐位相同 ⇒ 也是克隆）。
        _is_clone_model = (BASE_KEY is not None and mkey == BASE_KEY
                           and tuple(round(float(x), 9) for x in (w or [])) == BASE_WT)
        # 🔴 2026-11-23 09:30 **修正：克隆判定只能用「模型同一性」，不能用「bench 读数相同」**
        #   实测（promote_candidate.py:383-396 的权威升轨口径）：
        #     六项用 **`>=`**（nano/nano4/f1/t1/t3 ≥ 基线、t2_fp ≤ max(基线,5)），
        #     **外加「至少一轴严格优于基线」**（`strictly_better`）⇒ 才是可升轨。
        #   ⇒ **bench 三轴与基线逐位相同、但权重不同 ⇒ 是不同模型 ⇒ 是合法候选**
        #     （它只要 T1/T3 有一轴严格优于基线就能升轨）——
        #     证据：`1910071`（fam=sb3:1608003+1530006+1623003:bw30）nano/nano4/f1 与基线**逐位相同**，
        #     但 **T3 = 0.9489376 > 基线 0.9488856（严格更优）** ⇒ 只差 T1 一轴 3.2e-5。
        #   ⇒ 老判定（`_is_clone_bench`）会把这一整族 **490 个 strict3 臂**全部误判成克隆而排到队尾
        #     ⇒ 真候选打不到。**改为只认「成员集 + 权重全同」才是模型同一性。**
        novel_bench = 1 if (strict_ok and not _is_clone_model) else 0
        # ★ 2026-11-18 16:40：Track-B 赢家优先（F1 严格超基线 ⇒ 唯一可能直接升轨的形态）
        f1win = 1 if float(pk.get("f1") or 0) > float(base["f1"]) + 1e-9 else 0
        famboost = 1 if fam in TRACKB_FAMS else 0
        # 🟢 2026-11-19 15:55：T3 上探标记（见文件头注释）
        t3win, dw3 = 0, 9.0
        # 🟢🟢 2026-11-19 17:15：**w3 ≥ 基线 w3（且同成员集）⇒ 优先门禁**
        #   理由（实测 6 个块 D 臂后修正）：**w3 微推（.2121~.2151）T3 并没有涨**
        #   （T3 台阶量化在 1 个 TP ≈ 5.2e-5 ⇒ 小 Δw3 常常不动）
        #   ⇒ 真正还有戏的是两件事：
        #     ① **w3 > 基线**（T3 上探，等台阶被跨过）；
        #     ② **w3 = 基线**（=.2116）的那张**细网格** —— 里面藏着
        #        「**T1 顶格 ∧ t2_fp=3**」的交集（2020036 只差 T1 1.86e-5；而 t2_fp<4
        #        已于 16:40 补进「严格更优」轴清单 ⇒ 拿到就是可升轨）。
        #   ⇒ 统一规则：**同成员集 且 w3 ≥ 基线 w3** ⇒ t3win=1，其次按 |Δw3| 升序。
        if w and BASE_KEY and mkey == BASE_KEY:
            for _i, _m in enumerate(mem):
                try:
                    if int(_m[0]) == 1530006 and int(_m[1]) == 5:
                        _w3 = float(w[_i])
                        if _w3 >= BASE_W3 - 1e-9:
                            t3win, dw3 = 1, abs(_w3 - BASE_W3)
                        break
                except Exception:  # noqa: BLE001
                    pass
        # 🟢🔴 2026-11-19 17:52 **回滚：撤销 needletip 插队**
        #   针尖族 {1102033@1,1740004@5,1530006@5} 已被**硬证据判死**（见 gen_w13_recipes.py 簇 T 注释）：
        #     ① w1102033 < .975 ⇒ nano < 基线；② w1102033 ≥ .9758 ⇒ 官方 T1/T3 崩
        #     线性拟合 T1(w)=.8919898−.00356w ⇒ 要 T1 ≥ 基线需 w ≤ .318 ⇒ 该成员集**无解**
        #   ⇒ 187 个针尖臂插队抢门禁机时（每臂 ~740s）纯属浪费 ⇒ 撤销。
        #   保留：BASE_KEY 同成员集限定 + w3 ≥ 基线 w3 的 T3 上探规则（v2，实测有效）。
        # 排序元组索引：0=tier 1=strict_bench 2=f1win 3=famboost 4=t3win 5=dw3
        #                 6=-bucket_n 7=bucket_best 8=-near_dist 9=margin 10=-sid 11=sid
        # R1 FIX 2026-11-20：12=soup_rank（汤臂 1 / 训练臂 0）13=stage（分档，见下）
        #   ⇒ **soup_rank 放在最前** ⇒ 任何汤臂都排在任何训练臂之前（成熟前沿不被挤掉）
        # 🔴🔴🔴 2026-11-23 10:45 —— **T1 血统插队 / 死路降权**
        #   🔴🔴 **关键**：`stage` 分组是在**排序之前**拼进 `_b0.._b3` 的 ⇒ 档位压倒一切。
        #   ⇒ 光加排序键**没用**（实测：加键后队列原样不动，因为桥接臂在 near 档、
        #      证伪家族在 strict 档，near 永远压不过 strict）⇒ **必须让 t1lin 直接改档位**。
        _lt = lin_t1.get(mkey)
        # 🔴🔴 2026-11-23 11:45 —— **修正误判 A**：t1lin=1（血统能顶开 T1）**不等于**该臂能六项全过。
        #   实测：桥接家族（1100275+1102004+1612001…）的兄弟臂 dT1 = +2.81e-5（确实能顶 T1），
        #   但它 **nano4 必负**（如 2120019 nano4=0.6604442 < 基线 0.6609168）
        #   ⇒ **永远不可能六项全过** ⇒ 把它插到队首 = 门禁机时（8 倍瓶颈）纯浪费。
        #   ⇒ t1lin=1 现在**额外要求**「候选臂本身 bench 三轴全过」或「同成员集有 strict3 兄弟臂」。
        if _lt is not None and _lt > 0 and (strict_ok or lin_strict.get(mkey)):
            t1lin = 1          # 该成员集已被证明能顶开 T1 **且**有能力补齐六项 ⇒ 提到第 0 档
        # 🔴🔴 2026-11-23 11:45 —— **修正误判 B**：死路判据不能只看「兄弟臂 dT1 全 < 0」。
        #   实测：近失族（211xxxx 等）132 个兄弟臂**只欠 T1 3~9 个台阶**、其余轴严格优于基线
        #   ⇒ 它们是**最有价值**的候选，不是死路。真正的死路是「连 bench 三轴都过不了」。
        elif (lin_n.get(mkey, 0) >= T1LIN_DEAD_MIN_N
              and not lin_pos.get(mkey)
              and not lin_strict.get(mkey)):
            # 🔴 2026-11-23 11:15 **降权判据升级为「全轴」**：
            #   只看 max(dT1) 会误埋「T1 打平但 T3 严格更优」的候选族
            #   （证据：1910071 的 T3=0.9489376 > 基线 0.9488856，它所属成员集
            #     n=98 且 max(dT1)=0 ⇒ 按旧判据会被降权 ⇒ 真候选被埋）。
            #   ⇒ 只有「已门禁 ≥5 臂 且 **没有任何一轴**严格优于基线」才算真死路。
            t1lin = -1         # 该成员集已被证伪（全轴都打平）⇒ 压到最后一档
        else:
            t1lin = 0
        if t1lin == 1:
            stage = 0         # ★ T1 正血统 ⇒ 无条件进第 0 档（凌驾 strict/near 之分）
        elif t1lin == -1:
            stage = 3         # ✖ 已证伪血统 ⇒ 压到最后一档
        elif is_soup:
            soup_rank = 1
            stage = 0 if strict_ok else (2 if TRAIN_TIER else 1)
        else:
            soup_rank = 0
            stage = (1 if TRAIN_TIER else 2) if strict_ok else 3
        if t1lin != 0:
            # t1lin != 0 时档位已定 ⇒ soup_rank 不再影响分组，但仍保留在元组里供排序参考
            soup_rank = 1 if is_soup else 0
        # 🔴🔴 2026-11-23 11:45 新增索引 16 = fam_gap（该成员集离六项全过的最小台阶数）
        #   证据表 lin_gap 来自**已门禁**兄弟臂 ⇒ 新合成的微扫臂也能继承它所在族的「近失距离」
        #   ⇒ 排队时「离升轨最近的族」永远在前（实测近失族 1.7~3.0 台阶 vs 桥接族 ≥5 台阶）
        # 🔴🔴 2026-11-23 12:05 —— **fam_gap 三态语义**（修 Heisenberg 复核的缺陷 1 + 缺陷 2）
        #   ① 族内有「有头寸」的已门禁兄弟臂 ⇒ 用族的最优台阶（并取自身 bench 缺口的较大者）
        #   ② 族内只有「零头寸纯打平」兄弟臂 ⇒ 该族已证伪 ⇒ _GAP_DEAD（永不可升轨，排末位）
        #   ③ 全新血统（无任何已门禁兄弟臂）⇒ **退化为候选自身的 bench 缺口 + _GAP_NEW**
        #      （旧实现写死 9.9 ⇒ 214xxxx 腿②「掺 T1 供体」这批唯一有望翻正 T1 的配方被排到 rk 2000+）
        _own_gap = _gap6(pk, of, base)
        _gap_fam = lin_gap.get(mkey)
        if _gap_fam is not None:
            _fgap = max(_gap_fam, _own_gap)
        elif lin_tie.get(mkey):
            _fgap = _GAP_DEAD
        else:
            _fgap = _own_gap + _GAP_NEW
        cands.append((tier, strict_bench, f1win, famboost, t3win, dw3,
                      -bucket_n.get(fam, 0), bucket_best.get(fam, 0),
                      -near_dist, margin(pk, base), -int(sid), int(sid),
                      soup_rank, stage, novel_bench, t1lin, _fgap))
    # 🔴🔴 2026-11-19 10:25 **两段式门禁队列（只把「还可能升轨」的臂送门禁）**
    #   升轨判据（promote_candidate.py，`res["ok"] = all(c.values())`）= **七项全过**
    #   = nano / nano4 / F1 / T1 / T2_fp / T3 / 板卡ms
    #   ⇒ **nano 或 nano4 低于基线的臂永远不可能升轨** ⇒ 送它去门禁 = 纯浪费机时。
    #   现场证据：`.prio_sids` 头部 24 臂全是「**老基线时代**的三轴全过」臂 ——
    #     1611000 nano .7237713 < 基线 .7242439、nano4 .6585539 < .6602079（只有 F1 高）
    #     1597002 nano .7240076（−2.4e-4）、nano4 .6578450（−2.4e-3）
    #     ⇒ 全是**死路臂**，却因为 refresh_prio 读**历史日志**里的「三轴全过=True」标记
    #        （那个标记是打老基线 .6666 时代的！）而一直霸占门禁机时。
    #   修法：
    #     第一段 = strict_ok（nano/nano4/F1 **同时 ≥ 当前基线** ⇒ 唯一可能补齐七项的候选）
    #             ⇒ 全部送门禁；
    #     第二段 = near_ok（近线臂）⇒ 只在第一段用完时补位，保持卡不空转；
    #     `.prio_sids` 不再硬性前置（它已过时），只作为同分时的记忆。
    #   排序元组索引：0=tier 1=strict_bench 2=f1win 3=famboost 4=-bucket_n
    #                 5=bucket_best 6=-near_dist 7=margin 8=-sid 9=sid
    # 🟢🟢 2026-11-19 15:55：**t3win 排最前**（T3 上探 = 唯一能严格更优的方向）
    #   其余次序不变：tier → |Δw3| → f1win → famboost → margin → sid
    # R1 FIX 2026-11-20：**按 stage 四档排序**（stage 已在候选构造时定好）
    #   stage 0 = strict 汤臂（成熟前沿，优先级最高，键与老代码**逐位一致**）
    #   stage 1 = strict 训练臂（新增；有升轨可能，但**永远低于所有 strict 汤臂**）
    #   stage 2 = near 汤臂（键与老代码**逐位一致**）
    #   stage 3 = near / 探索档训练臂（新增；末位补位）
    #   ⇒ 汤臂的**相对顺序**与老代码完全一致（stage 0 在 stage 2 之前，内部排序键不变）。
    # 🟢 2026-11-23 09:00：novel_bench（索引 14）置于**最前** ⇒ 真候选优先于基线克隆臂
    # 🔴 2026-11-23 10:45：**t1lin（索引 15）置于最前** ⇒ T1 正血统插队 / 证伪血统降权
    # 🔴🔴 2026-11-23 11:45：**fam_gap（索引 16）紧跟 t1lin** ⇒ 同档位里「离六项全过最近的族」在前。
    def _k_strict(c):
        return (-c[15], c[16], -c[14], -c[4], -c[0], c[5], -c[2], -c[3], -c[9], c[11])

    def _k_near(c):
        return (-c[15], c[16], -c[14], -c[4], -c[0], -c[2], -c[9], c[11])

    _b0 = sorted([c for c in cands if c[13] == 0], key=_k_strict)
    _b1 = sorted([c for c in cands if c[13] == 1], key=_k_strict)
    _b2 = sorted([c for c in cands if c[13] == 2], key=_k_near)
    _b3 = sorted([c for c in cands if c[13] == 3], key=_k_near)
    _strict = _b0 + _b1
    _near = _b2 + _b3
    # 🔴🔴 2026-11-19 18:40 **修复**：17:50 回滚 needletip 时**误删了本行** ⇒ 两段排序结果从未写回 cands
    #   ⇒ 门禁队列退化成「台账字典顺序」⇒ 队首全是随机老臂（1111031/1547001/…，三轴差 1e-3 的死臂）
    #   ⇒ 整晚门禁机时（~740s/臂，瓶颈）全部空转。已定位并补回。
    cands = _strict + _near
    out = []
    for _t, _sb, _fw, _fb, _tw, _dw, _nb, _bp, _nd, _m, _s, sid, _sr, _sg, _nv, _tl, _fg in cands:
        if not DRY:
            cf = os.path.join(CLAIM, str(sid))
            try:
                fd = os.open(cf, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{chan} {now}\n".encode())
                os.close(fd)
            except FileExistsError:
                try:
                    age = now - os.path.getmtime(cf)
                    if age < CLAIM_TTL:
                        continue
                except OSError:
                    pass
        out.append(sid)
        if len(out) >= cap:
            break
    if _FLOOR_STATE["skip"]:
        sys.stderr.write("[gate_prio] 分辨率地板过滤：剔除 %d 臂（展平 maxΔ < %.3f ⇒ 读数必与基线同格）\n"
                         % (len(_FLOOR_STATE["skip"]), GATE_MIN_MAXDELTA))
    print(",".join(str(x) for x in out))


if __name__ == "__main__":
    main()
