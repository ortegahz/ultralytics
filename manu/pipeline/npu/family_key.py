
# -*- coding: utf-8 -*-
"""配方家族键（**唯一权威口径**）—— 升轨「同配方族 n≥4」复现关的基础。

🟢🟢 2026-11-16 11:05 修两个**家族键口径 bug**（都会让复现计数失真）：

  ① **数值未归一化** ⇒ `thr=12`（int）与 `thr=12.0`（float）被算成**两个家族**。
     实测：冠军配方 BASE_A2（sw4.0×thr12×ms3×lrf0.1×swb1.0）共 52 个臂，
     其中 14963 记的是 int 版 `12`，R22 的种子臂记的是 float 版 `12.0`
     ⇒ 两边互不相认 ⇒ 该家族永远 n=1 ⇒ **n≥4 永远凑不齐**（升轨卡死）。
     修法：数值参数统一 `%.6g` 归一化（12 与 12.0 同键）。

  ② **合成汤没带成员数** ⇒ `bs12679:eq` 把 k=2/3/4/5/7 的汤全算成**同一个家族**
     ⇒ 「同盆地 + 同成员数 + 等权 = 同一个方法」（memory §17006 的原意）被放宽成
       「同盆地 + 任意成员数 + 等权」⇒ 69905(k=7)/69911(k=5)/69913(k=5) 被凑成 n=3。
     修法：家族键 = `soup:<盆地>:<tag><成员数>`（成员数 = len(soup_members)）。

  ⚠️ 判据本身**不变**：同家族 n≥4 六项全过 + 板卡 T4 ≤ 19.989 ms。
     只是「家族」的物理含义回到 memory 写的那一条（更严，但可达成）。
"""
from __future__ import annotations

import collections
import os
import re

# 决定训练动力学的配方键（不含 seed / init 之外的噪声项）
BASE_KEYS = ("sw", "swl", "thr", "maxside", "lr", "lr_floor", "mix", "fb", "ow", "sched",
             "backbone_lr", "ema_decay", "swb", "swb_thr", "cp_prob", "cp_max_px")

_CACHE: dict = {}


def norm_value(v) -> str:
    """数值归一化：12 / 12.0 / 12.0000001 → 同一键；None → 'None'；字符串原样。"""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return "%.6g" % float(v)
    return str(v)


def _soup_family_key(arm: dict) -> str:
    """合成汤家族键 = 盆地 + 家族标签 + **成员数**（同盆地同成员数等权 = 同一个方法）。"""
    sf = str(arm.get("soup_family") or "")
    mem = arm.get("soup_members") or []
    k = len(mem)
    # 🟢🟢 2026-11-18 09:55 —— **三路汤（sb3:）家族键**：
    #   sb3 的 `soup_family` 已经自带「成员集合 + 增益子权重档」⇒ 直接用标签当家族键；
    #   否则下面的兜底分支会把**权重精确值**也拼进键里 ⇒ 每份配方自成一族 ⇒ n 永远 = 1。
    if sf.startswith("sb3:"):
        return "soup3:" + sf
    m = re.match(r"^(bs\d+):([A-Za-z]+)(\d*)$", sf)
    if m:
        return "soup:%s:%s%d" % (m.group(1), m.group(2), k)
    ms = "+".join("%s@%s" % (x[0], x[1]) for x in mem)
    w = arm.get("soup_weights") or []
    return "soup:%s|%s@%s" % (sf, ms, ",".join("%.3f" % float(x) for x in w))


def polish_family_key(arm: dict) -> str:
    """**抛光家族键 = 抛光配方指纹（**不含锚点**）**。

    🔴🔴🔴 2026-11-20 R1修复 —— **把锚点从家族键里拿掉**
    --------------------------------------------------
    【实测证据】2026-11-20 14:39 服务器台账快照（12,096 臂）：
      · 抛光臂（`polish` 标记或 `params.warm_key`）共 **785** 个，**全部**带
        `source_anchor` / `warm_anchor`（= 老键里那个「锚点」分量）。
      · 按**老键**（`polish:<锚点>|<配方段>`）分组 ⇒ **161** 个家族 ⇒
        其中 **54 个臂独自成族（族内 n = 1）** ⇒ 这 54 个臂的
        `auto_promote_watch`「同族 n≥4」复现关**结构性不可达**
        ⇒ 抛光臂即使自己六项全过也**永远升不了轨**。
      · 按**新键**（纯配方段）分组 ⇒ **34** 个家族（33 个配方段 + 1 个既带
        `soup_members` 又带 `warm_key` 的臂，它走汤臂分支 —— **老键同样如此，非本次引入**）。
      · **老家族 ⊆ 新家族**（老键 = 锚点+配方段，新键 = 配方段）⇒
        老键下 n≥4 的家族在新键下 n **只增不减** ⇒ 复现关**只会更容易过**，不会更难。
      · 语义可识别性（新键）：**33** 个配方段中 **28 个含 ≥2 个不同锚点**，
        最大组 `lr0.2|l20.001|fb1|ep1` 含 **23 个不同锚点**
        ⇒ 「同一抛光配方作用在 ≥2 个不同锚点」**可被识别为同族**（老键下不可识别）。
      · 误合并检查：**新键相同但配方段不同的臂 = 0** ⇒ 不同配方**不会**被误判成同族。
    【改法】家族键 = `warm_key` 去掉「首个锚点字段」与「末尾 repN 换种子后缀」后的**配方段**；
      没有 `warm_key` 时退回 params 的四个抛光超参（lr / anchor_l2 / freeze_bn / epochs）。
    【为什么不误判同族】配方段里含 lr / λ / freeze_bn / epochs（以及 cp 变体后缀）
      ⇒ **不同配方 ⇒ 不同配方段 ⇒ 不同族**；只有「同一配方作用在不同锚点」才同族
      —— 而这正是要的语义（memory 写的就是「同一套抛光配方作用在 ≥2 个不同锚点」）。
    【影响面】只改**抛光臂**的分组键；**汤臂**（`_soup_family_key`）与
      **普通训练臂**（`base:` 含 init —— 不同底座 = 不同盆地）**完全不变**。
    【回退】`git checkout manu/pipeline/npu/family_key.py`
    """
    arm = arm or {}
    p = arm.get("params") or {}
    wk = str(p.get("warm_key") or "")
    if wk and "|" in wk:
        # 去掉首个锚点字段（`w<anchor>e<ep>`），再去掉 repN 换种子后缀
        cfg = re.sub(r"\|rep\d+$", "", wk.split("|", 1)[-1])
        return "polish:" + cfg
    return "polish:lr=%s|l2=%s|fb=%s|ep=%s" % (norm_value(p.get("lr")),
                                               norm_value(p.get("anchor_l2")),
                                               norm_value(p.get("freeze_bn")),
                                               norm_value(p.get("epochs")))


def arm_family_key(arm: dict) -> str:
    """按**台账 arm** 算家族键。"""
    arm = arm or {}
    p = arm.get("params") or {}
    if arm.get("synth") and (arm.get("soup_members") or []):
        return _soup_family_key(arm)
    wk = p.get("warm_key")
    if wk or arm.get("polish"):
        # 🔴 2026-11-20 R1修复：抛光臂的家族键**不再含锚点**（见 polish_family_key 注释块）
        #   老键 = "polish:<锚点>|<配方段>" ⇒ 不同锚点 ⇒ 不同族 ⇒ n 恒 1 ⇒ n≥4 永远凑不齐。
        #   新键 = "polish:<配方段>" ⇒ 同一配方作用在不同锚点 ⇒ 同族 ⇒ 复现关可达。
        #   （repN 换种子后缀已在 polish_family_key 内去掉 ⇒ rep1/rep2/rep3 算同一个家族。）
        return polish_family_key(arm)
    # 普通臂：配方键必须含 init（底座 ckpt）—— 不同底座 = 不同盆地
    return "base:%s|%s" % (p.get("init"), "|".join(norm_value(p.get(k)) for k in BASE_KEYS))


def family_key(row: dict, led: dict) -> str:
    """按 **audit row**（含 trial）算家族键（与 auto_promote_watch 同口径）。"""
    t = str(row.get("trial"))
    arm = (led.get("arms") or {}).get(t) or {}
    return arm_family_key(arm)


def dedupe_key(row: dict, arm: dict) -> tuple:
    """「独立样本」指纹 —— 同一个模型**只能算一次**复现证据。

    🟢🟢 2026-11-16 10:35 实测踩坑：98901 与 99001 的**成员集合完全相同**
      （[69901@5, 69914@5, 69902@5]）—— 是两个并行扫描实例算的**同一个汤**
      （权重平均完全一样 ⇒ 同一个模型）⇒ 若不判重，eq3 家族会被虚报成 n=2。
    判重规则：
      · 合成汤 ⇒ (成员集合, 权重) 相同 = 同一个模型；
      · 普通/抛光臂 ⇒ 同 **seed** = 确定性同模型；无 seed 时同 trial（含不同 epoch）= 同一次训练。
    """
    arm = arm or {}
    if arm.get("synth") and (arm.get("soup_members") or []):
        mem = tuple(sorted((int(m[0]), int(m[1])) for m in arm["soup_members"]))
        w = tuple(round(float(x), 6) for x in (arm.get("soup_weights") or []))
        return ("soup", mem, w)
    sd = arm.get("seed")
    if sd is not None:
        return ("seed", int(sd))
    return ("trial", int(row.get("trial") or 0))


def six_pass_families(led: dict, rows=None, base=None) -> dict:
    """家族 → 六项全过的臂列表（权威口径：磁盘 gate_report + 台账 NanoBench 读数）。

    带缓存（按台账 mtime+size）⇒ `promote_candidate --list` 逐臂调用也不会重算。
    ⚠️ 同一模型只计一次（见 `dedupe_key`）⇒ 「n≥4 复现」必须真的是 4 个**独立**模型。
    """
    from manu.pipeline.npu import six_criteria_audit as A  # 局部导入 ⇒ 无循环依赖
    led = led if led is not None else {}
    key = None
    if rows is None and base is None:
        try:
            stt = os.stat(A.LEDGER)
            key = (stt.st_mtime_ns, stt.st_size)
        except Exception:  # noqa: BLE001
            key = None
        if key is not None and _CACHE.get("k") == key:
            return _CACHE["v"]
    if base is None:
        base = A.live_base(led)
    if rows is None:
        rows = A.collect()
    fam = collections.defaultdict(list)
    seen = set()
    for r in rows:
        n, _miss = A.judge(r, base)
        if n != 6:
            continue
        arm = (led.get("arms") or {}).get(str(r.get("trial"))) or {}
        dk = dedupe_key(r, arm)
        if dk in seen:
            continue
        seen.add(dk)
        fam[family_key(r, led)].append(r)
    out = dict(fam)
    if key is not None:
        _CACHE["k"] = key
        _CACHE["v"] = out
    return out
