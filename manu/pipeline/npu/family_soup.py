
# -*- coding: utf-8 -*-
"""同配方多种子权重汤（family soup）—— 直接打「T3 缺口 1e-4 < σ_t3=8.6e-4」这个噪声级缺口。

动机（2026-11-13 17:55，台账全量复算）：
  * 6 条近失臂（12679@3 / 96835@5 / 12519@3 / 12694@5 / 12709@4 / 12520@4）**只差 T3**：
    缺口 1.2e-4 ~ 1.8e-3，而 **σ_t3 ≈ 8.6e-4** ⇒ 单臂差 0.2σ 本质是**噪声**，不是配方缺陷。
  * 这 6 条里有 4 条是**完全相同的配方**（sw4.0 × thr12 × ms3 × lrf0.10）只是**换种子**
    ⇒ 教科书场景：**同盆地多种子权重平均（SWA）** ⇒ 期望 T3 上升、nano 保留。
  * ⚠️ `polish_seed_soup.py` 按 `params.init` 分组 ⇒ **只覆盖抛光臂**（有 warm_key），
    而这 4 条**基线近失臂没有 warm_key** ⇒ 一直没人给它们做种子汤 ⇒ 本脚本补这个洞。
  * 反面教训（已在 memory 记录）：**跨配方汤**会把 nano 特化平均掉（9863+12679 ⇒ nano 0.6238）
    ⇒ 本脚本只做**同配方同底座**的汤；混合族只做「同 thr/ms3 家族」的小规模对照。

用法（服务器）：
    python manu/pipeline/npu/family_soup.py --gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
REPO = "/tmp/pycharm_project_10ae9e2e"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))

LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
CACHE_DIR = "/mnt/data/siping/runs/npu20ms/tiny_ref"
SIDE_CAR = "family_soup_synth.json"
SYNTH_BASE = 96000          # 合成 id 空间 [96000, 96099]（避开 fine_swa/seed_soup 97000~97899、cross 97900+）
MARKER = "family_soup"
BASE = {"nano": 0.708648, "nano4": 0.647212, "f1": 0.889712}

# ── 同配方多种子组（全部来自台账实测：配方逐项相同，只有 seed 不同）──
GROUPS = [
    ("sw4x4", [(12679, 3), (96835, 5), (12694, 5), (12709, 4)]),
    ("sw5x2", [(12519, 3), (12520, 4)]),
    ("all6", [(12679, 3), (96835, 5), (12694, 5), (12709, 4), (12519, 3), (12520, 4)]),
    ("sw4x3", [(12679, 3), (96835, 5), (12709, 4)]),
    ("sw4sw5best", [(12679, 3), (12519, 3)]),
    ("sw4x2", [(12679, 3), (96835, 5)]),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def import_helpers():
    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
    from manu.pipeline.npu import nano_search as ns  # noqa: PLC0415
    return average_checkpoints, RA, ns


def synth_dir(sid: int) -> str:
    return os.path.join(ROOT, "trials", f"trial_{int(sid)}")


def sidecar_path(sid: int) -> str:
    return os.path.join(synth_dir(sid), "weights", SIDE_CAR)


def read_sidecar(sid: int):
    p = sidecar_path(sid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def alloc_synth_id(key: str) -> int:
    used = set()
    for sid in range(SYNTH_BASE, SYNTH_BASE + 100):
        sc = read_sidecar(sid)
        if not sc or sc.get("key") != key:
            used.add(sid)
    for sid in range(SYNTH_BASE, SYNTH_BASE + 100):
        if sid not in used:
            return sid
    return SYNTH_BASE


def arm_ckpt(trial: int, epoch: int) -> str:
    return os.path.join(ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{int(epoch)}.pt")


def register_synth_arms(entries: list[dict], ns) -> int:
    """六项全过的族汤登记进台账（⇒ board_t4_watch 会自动补板端 T4）。"""
    try:
        led = ns.load_ledger()
    except Exception as exc:  # noqa: BLE001
        log(f"[登记] 台账读取失败（跳过）：{exc}")
        return 0
    arms = led.get("arms") or {}
    n = 0
    for e in entries:
        if e.get("error") or not e.get("triad"):
            continue
        g = e.get("gate") or {}
        if g.get("error") or g.get("t1") is None or g.get("t3") is None:
            continue
        sid = int(e["synth_id"])
        ck = os.path.join(synth_dir(sid), "weights", "epoch5.pt")
        if not os.path.isfile(ck):
            continue
        src = (arms.get(str(e["members"][0][0])) or {}).get("params") or {}
        p = dict(src)
        p["init"] = ck
        p["synth"] = 1
        _nb = e.get("nanobench") or {}
        nb = {k: (e.get(k) if e.get(k) is not None else _nb.get(k))
              for k in ("nano", "nano4", "f1", "micro", "small", "regular", "f1_4")}
        nb.update({"epoch": 5, "epoch_key": "5", "ckpt": "epoch5.pt", "cache": e.get("cache")})
        arms[str(sid)] = {
            "trial": sid, "synth": True, "source_trial": e["members"][0][0],
            "soup_members": [list(m) for m in e["members"]], "soup_weights": e["weights"],
            "soup_family": (e.get("family")
                           or ("bs%s:eq%d" % (e["members"][0][0], len(e["members"])))),
            "params": p, "seed": None,
            "nanobench": {"5": nb}, "picked": dict(nb), "picked_epoch": 5,
            "official": {k: g.get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")},
            "six_ok": bool(g.get("six_ok")),
            "registered_by": MARKER,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        n += 1
    if n:
        led["arms"] = arms
        try:
            ns.save_ledger(led)
            log(f"[登记] {n} 个族汤候选已进台账（⇒ board_t4_watch 会自动测板端）")
        except Exception as exc:  # noqa: BLE001
            log(f"[登记] 台账写盘失败：{exc}")
            return 0
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--report", default="/mnt/data/siping/runs/npu20ms/nano_search/family_soup.json")
    args = ap.parse_args()

    raw, RA, ns = import_helpers()
    with open(LEDGER, encoding="utf-8") as fh:
        led = json.load(fh)
    base = led.get("baseline_v2") or {}
    bt1 = float(base.get("t1") or 0.0)
    bt2fp = int(base.get("t2_fp") or 5)
    bt3 = float(base.get("t3") or 0.0)
    log(f"[基线] T1={bt1} T2fp={bt2fp} T3={bt3}")

    entries = []
    for tag, members in GROUPS:
        if len(entries) >= args.limit:
            break
        cks = [arm_ckpt(t, e) for t, e in members]
        missing = [c for c in cks if not os.path.isfile(c)]
        if missing:
            log(f"[跳过] {tag}：缺权重 {missing}")
            continue
        key = "fs|" + "|".join(f"{t}@{e}" for t, e in members)
        sid = alloc_synth_id(key)
        sc = read_sidecar(sid)
        if sc and sc.get("scored"):
            log(f"[跳过] 已算过 {key}")
            continue
        out_dir = os.path.join(synth_dir(sid), "weights")
        os.makedirs(out_dir, exist_ok=True)
        out_ck = os.path.join(out_dir, "epoch5.pt")
        weights = [1.0 / len(members)] * len(members)
        try:
            raw(cks, out_ck, weights=weights, epoch=5)
        except Exception as exc:  # noqa: BLE001
            log(f"[合成失败] {key}：{exc}")
            continue
        pkl = os.path.join(CACHE_DIR, f"familysoup_{sid}.pkl")
        if not RA.build_cache(out_ck, pkl, args.gpu, extra="--no-peak-attrs"):
            log(f"[缓存失败] {key}")
            continue
        res = RA.nanobench(pkl, f"familysoup_{sid}")
        if not res:
            log(f"[评测失败] {key}")
            continue
        rb8 = res["peak_tol8"]["recall_by_bucket"]
        rb4 = res["peak_tol4"]["recall_by_bucket"]
        m = {"nano": float(rb8["nano"]), "nano4": float(rb4["nano"]), "f1": float(res["peak_tol8"]["f1"]),
             "micro": float(rb8["micro"]), "small": float(rb8["small"]), "regular": float(rb8["regular"]),
             "f1_4": float(res["peak_tol4"]["f1"]), "cache": pkl}
        triad = bool(m["nano"] >= BASE["nano"] and m["nano4"] >= BASE["nano4"] and m["f1"] >= BASE["f1"])
        log(f"[评测][{tag}] 成员={members} ⇒ nano={m['nano']:.6f} nano4={m['nano4']:.6f} "
            f"F1={m['f1']:.6f} triad={triad}")
        gate = None
        if triad:
            g = ns.run_official_gate(sid, 5, args.gpu)
            if g and "error" not in g:
                gate = {k: g.get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
                gate["six_ok"] = bool(gate.get("t1") is not None and gate.get("t3") is not None
                                     and gate["t1"] >= bt1 and gate["t2_fp"] <= bt2fp and gate["t3"] >= bt3)
            else:
                gate = {"error": (g or {}).get("error", "no_report")}
            log(f"[门禁] synth={sid} T1={gate.get('t1')} fp={gate.get('t2_fp')} T3={gate.get('t3')} "
                f"six_ok={gate.get('six_ok')}")
        with open(sidecar_path(sid), "w", encoding="utf-8") as fh:
            json.dump({"key": key, "scored": True, "members": [list(x) for x in members],
                       "weights": weights, "nanobench": {k: m.get(k) for k in ("nano", "nano4", "f1")},
                       "triad": triad, "gate": gate}, fh, ensure_ascii=False, indent=2)
        entries.append({"family": tag, "members": members, "weights": weights, "synth_id": sid,
                        "nanobench": m, "triad": triad, "gate": gate, "cache": pkl})
    n_reg = register_synth_arms(entries, ns)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({"ts": time.strftime("%F %T"), "entries": entries, "registered": n_reg},
                  fh, ensure_ascii=False, indent=2)
    log(f"[完成] 组合 {len(entries)} 个｜登记 {n_reg} 个｜报告 {args.report}")


if __name__ == "__main__":
    main()