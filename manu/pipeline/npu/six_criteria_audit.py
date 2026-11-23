
"""六项判据全盘复算审计（只读）。

🟢 2026-11-12 15:30 新增（§152.1）：回答「有没有更好的模型」的唯一权威口径。

为什么需要单独一个脚本（而不是信台账）：
  ① 台账 `official` 字段可能是 `{'stale': 'epoch_changed 5->4'}` ⇒ T1/T2/T3 缺失
     ⇒ `core_checks()` 对缺项是**跳过** ⇒ 六项漏斗里该臂「既不算过也不算不过」。
  ② 台账只对 **picked epoch** 存官方读数；但磁盘上 `gate/NANO_<trial>_ep<epoch>_official/`
     里可能躺着**别的 epoch** 的门禁报告 ⇒ 单看台账会漏掉候选。
  ⇒ 本脚本以**磁盘 gate_report.json 为权威**，把台账里所有 (臂, epoch) 的
     NanoBench 读数（nano / nano4 / f1）与官方三轨读数（t1 / t2 虚警 / t3）配对，
     逐个算六项，给出**全盘六项全过清单**与**最接近候选榜**。

口径（与 Baseline_v2.0 = trial_9863 ep5 完全一致，`>=` 判过）：
    nano  ≥ 0.708648 ｜ nano4 ≥ 0.647212 ｜ F1 ≥ 0.889712
    T1    ≥ 0.888086 ｜ T2 虚警 ≤ 5       ｜ T3 ≥ 0.948489
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
GATE_DIR = f"{ROOT}/gate"
LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
# 🟢 2026-11-13 14:45：epoch 可能是**分数**（亚 epoch 存盘，方案 §3）⇒ 允许小数
NAME_RE = re.compile(r"^NANO_(\d+)_ep(\d+(?:\.\d+)?)_official$")

BASE = {
    "nano": 0.7242438563327032,
    "nano4": 0.6609168241965974,
    "f1": 0.8906832039622916,
    "t1": 0.8909238824339264,
    "t3": 0.9488855869242199,
    "t2_fp": 3,
}











CRIT = ["nano", "nano4", "f1", "t1", "t2fp", "t3"]

# 🟢🟢 2026-11-18 18:00 —— **修 T2 轴口径漂移（今晚升轨链的真正阻塞点 · 根因级修复）**
#   现象：本模块 docstring 写明的口径就是「T2 虚警 ≤ 5」（与 `trackb_core.T2_USER`、
#         `board_t4_watch.DELIVER` 的任务书交付红线 T2_fp ≤ 20 / 官方门禁 ≤ 36 一致），
#         但 `judge()` 的实现却是 `t2_fp <= B["t2_fp"]` ⇒ **跟着当前基线漂移**。
#   根因：9863 当年基线 t2_fp=5 ⇒ 两种写法等价；后来基线换成 t2_fp=1 的臂
#         ⇒ 口径悄悄收紧成「T2 虚警必须 ≤1」⇒ 今晚所有真·更优候选（T2_fp=2~4，
#         远低于任务书红线）被判 5/6 ⇒ 既进不了升轨链的六项计数，
#         也过不了 `promote_candidate` 的种子复现关（`soup_family_six_ok` 用的就是本判据）
#         ⇒ **整晚算力跑出来的更优臂永远升不了轨**（与板卡是否空闲无关）。
#   修法：T2 轴 = `t2_fp <= max(基线 t2_fp, 5)` ⇒ 恢复文档口径，且**永不比基线更严**
#         （基线 t2_fp 更高时自动取基线值）。与 `trackb_core.T2_USER = 5` 完全对齐。
#   影响：六项全过臂数 2 → 6（1540030/1590013/1605008/1617002/1616005/1622004），
#         升轨链的「同家族 n≥4 + 板卡 T4 + 种子复现」三关才有可能同时满足。
T2_USER_CAP = 5


def t2_cap(base: dict) -> int:
    """T2 虚警上限：max(基线自身的 t2_fp, 用户口径 5) ⇒ 不比基线更严，也不低于 5。"""
    return max(int((base or {}).get("t2_fp") or 0), T2_USER_CAP)


def parse_report(path: str) -> dict | None:
    try:
        rep = json.load(open(path))
    except Exception:
        return None
    tr = rep.get("tracks") or {}
    t1 = ((tr.get("track1_uav") or {}).get("finetune_peak") or {})
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    if not t1.get("f1") or not t3.get("f1"):
        return None
    return {
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
        "verdict": rep.get("verdict"), "veto": rep.get("veto_failures", []),
    }


def collect(ledger_path: str = LEDGER, gate_dir: str = GATE_DIR) -> list[dict]:
    led = json.load(open(ledger_path))
    arms = led.get("arms") or led
    rows = []
    for d in sorted(glob.glob(f"{gate_dir}/NANO_*_official")):
        m = NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        trial, ep = int(m.group(1)), m.group(2)   # ep 保持字符串（可能是 '2.5'）
        rp = os.path.join(d, "gate_report.json")
        if not os.path.isfile(rp):
            continue
        g = parse_report(rp)
        if not g:
            continue
        st = arms.get(str(trial)) or {}
        nb = (st.get("nanobench") or {}).get(str(ep))
        if not nb:
            continue
        p = st.get("params") or {}
        rows.append({
            "trial": trial, "ep": ep, "seed": st.get("seed"),
            "nano": nb.get("nano"), "nano4": nb.get("nano4"), "f1": nb.get("f1"),
            "t1": g["t1"], "t2_fp": g["t2_fp"], "t3": g["t3"], "veto": g["veto"],
            "sw": p.get("sw"), "swl": p.get("swl"), "lrf": p.get("lr_floor"),
            "mix": p.get("mix"), "thr": p.get("thr"), "ms": p.get("maxside"),
            "init": (p.get("init") or "").split("/")[-1],
            "picked_ep": (st.get("picked") or {}).get("epoch"),
        })
    return rows


_BASE_CACHE: dict = {}


def live_base(led: dict | None = None) -> dict:
    """当前基线的判据口径 = **台账 `baseline_v2`** 的精确读数（升轨后自动跟着新基线走）。

    🟢🟢 2026-11-16 09:55 修根因：升轨后 `BASE` 常量虽会被回填，但**已在跑的守望进程**
      内存里仍是旧常量 ⇒ 判据口径停留在旧基线（浪费算力 + 可能把「不如新基线」的臂
      当成候选）。⇒ 判据统一走 `live_base()`；台账读不到时退回常量。
    """
    if led is None:
        try:
            st = os.stat(LEDGER)
            key = (st.st_mtime_ns, st.st_size)
            if _BASE_CACHE.get("key") == key:
                return dict(_BASE_CACHE["val"])
            led = json.load(open(LEDGER, encoding="utf-8"))
            _BASE_CACHE["key"] = key
        except Exception:  # noqa: BLE001
            return dict(BASE)
    b = led.get("baseline_v2") or {}
    out = dict(BASE)
    # 🟢 台账 baseline_v2 的数值可能是**截断**的（常量回填时就是 6 位小数）
    #   ⇒ 优先取基线臂在台账里的**精确读数**（picked/official），再退回 baseline_v2。
    st = (led.get("arms") or {}).get(str(b.get("trial"))) or {}
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    for k in ("nano", "nano4", "f1"):
        v = pk.get(k)
        if v is None:
            v = b.get(k)
        if v is not None:
            out[k] = float(v)
    for k in ("t1", "t3"):
        v = of.get(k)
        if v is None:
            v = b.get(k)
        if v is not None:
            out[k] = float(v)
    if of.get("t2_fp") is not None:
        out["t2_fp"] = int(of["t2_fp"])
    elif b.get("t2_fp") is not None:
        out["t2_fp"] = int(b["t2_fp"])
    return out


def judge(r: dict, base: dict | None = None) -> tuple[int, list[str]]:
    B = base if base is not None else live_base()
    flags = [
        (r["nano"] is not None and r["nano"] >= B["nano"]),
        (r["nano4"] is not None and r["nano4"] >= B["nano4"]),
        (r["f1"] is not None and r["f1"] >= B["f1"]),
        (r["t1"] >= B["t1"]),
        (r["t2_fp"] <= t2_cap(B)),
        (r["t3"] >= B["t3"]),
    ]
    miss = [n for n, ok in zip(CRIT, flags) if not ok]
    return 6 - len(miss), miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--gate-dir", default=GATE_DIR)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    rows = collect(a.ledger, a.gate_dir)
    if not rows:
        print("（没有可配对的 (臂, epoch) 读数）")
        return
    for r in rows:
        r["npass"], r["miss"] = judge(r)
    rows.sort(key=lambda r: (-r["npass"], -(r["nano"] or 0.0)))

    full = [r for r in rows if r["npass"] == 6]
    print(f"===== 六项判据全盘复算审计（{len(rows)} 个 (臂,epoch) 有完整读数）=====")
    try:
        import manu.pipeline.npu.nano_search as _ns
        _bn = getattr(_ns, "BASE_V2", None) or {}
    except Exception:
        _bn = {}
    # 🟢🟢 2026-11-18 10:10 修**基线标签取不到**：`nano_search.BASE_V2` 导入失败 ⇒ 打印「基线 = (未知)」
    #   ⇒ 晨报头部看不到当前基线是谁。修法：优先读台账 `baseline_v2`（权威），再退回常量。
    if not _bn.get("name"):
        try:
            with open(a.ledger, encoding="utf-8") as _fh:
                _b2 = (json.load(_fh) or {}).get("baseline_v2") or {}
            _bn = {"name": _b2.get("name"), "trial": _b2.get("trial"), "epoch": _b2.get("epoch")}
        except Exception:  # noqa: BLE001
            pass
    _bl = (f"基线 = {_bn.get('name')} / trial_{_bn.get('trial')} ep{_bn.get('epoch')}；判过口径 `>=`"
           if _bn.get("name") else "基线 = (未知)；判过口径 `>=`")
    _btrial = int(_bn.get("trial") or 0)
    print(_bl)
    print(f"六项全过：**{len(full)}** 个")
    for r in full:
        # 🟢🟢 2026-11-18 10:10 修**「基线本体」标记硬编码 9863** ⇒ 现在会把基线臂自己标成「新候选」
        tag = ("  ← 基线本体" if r["trial"] == _btrial else "  🔴 **新候选（可升轨）**")
        print(f"  ✔ trial={r['trial']} ep={r['ep']} seed={r['seed']} "
              f"nano={r['nano']:.6f} nano4={r['nano4']:.6f} f1={r['f1']:.6f} "
              f"t1={r['t1']:.6f} t2fp={r['t2_fp']} t3={r['t3']:.6f}{tag}")
    print()
    print(f"--- 最接近候选榜 TOP {a.top}（按过项数，同项按 nano 排）---")
    _B = live_base()   # 🟢 判据/差距一律跟台账当前基线走（不用可能过期的模块常量 BASE）
    for r in rows[: a.top]:
        gaps = []
        if r["nano"] < _B["nano"]:
            gaps.append(f"nano{r['nano'] - _B['nano']:+.6f}")
        if r["nano4"] < _B["nano4"]:
            gaps.append(f"nano4{r['nano4'] - _B['nano4']:+.6f}")
        if r["f1"] < _B["f1"]:
            gaps.append(f"f1{r['f1'] - _B['f1']:+.6f}")
        if r["t1"] < _B["t1"]:
            gaps.append(f"t1{r['t1'] - _B['t1']:+.6f}")
        if r["t2_fp"] > _B["t2_fp"]:
            gaps.append(f"t2fp+{r['t2_fp'] - _B['t2_fp']}")
        if r["t3"] < _B["t3"]:
            gaps.append(f"t3{r['t3'] - _B['t3']:+.6f}")
        _pe = r.get("picked_ep")
        # 🟢 ep 现在是字符串（可能是 '2.5'）⇒ 与 picked_ep 按数值比较
        star = "★" if (_pe is None or abs(float(r["ep"]) - float(_pe)) > 1e-9) else " "
        print(f" {r['npass']}/6{star} t={r['trial']:<6} ep={r['ep']:<3} "
              f"sw={r['sw']} swl={r['swl']} lrf={r['lrf']} mix={r['mix']} "
              f"init={r['init']} | " + (" ".join(gaps) if gaps else "全过"))
    print("（★ = 该 epoch 不是台账 picked epoch ⇒ 只看台账会漏掉它）")


if __name__ == "__main__":
    main()
