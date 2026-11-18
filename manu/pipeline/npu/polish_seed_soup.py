#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**抛光家族「种子汤」（seed soup）** —— 同配方不同种子的权重平均（零训练成本）。

为什么需要它（2026-11-13 17:20 设计）
------------------------------------
1. **T3 是唯一真正的绑定约束**：台账里 6 个 triad 通过臂中，5 个**只差 T3**
   （亏 1.15e-4 ~ 1.84e-3），而 T3 的种子噪声 σ_t3 ≈ 8.6e-4
   ⇒ 这些「缺口」**在噪声量级以内** ⇒ 单种子结论不可信（§14.3/§16 的 12869 教训）。
2. **抛光臂只有 1~2 个 epoch** ⇒ 臂内细粒度 SWA（`fine_swa_sweep.py`）**对抛光臂无效**
   （`select_arms` 要求磁盘上 ≥2 个 epoch ckpt）⇒ 抛光臂这条线**缺少零训练成本的降噪手段**。
3. 本工具补上这块缺口：**同一配方（同一锚点 + 同一 lr/λ/fb/ep + 同一 cp 设置）、
   只是随机种子不同的兄弟臂** ⇒ 权重平均 = 降方差（σ/√K），且保留配方共有的偏置
   ⇒ 有机会把「T3 差 1e-4」的噪声缺口**平均掉**。

与已证伪的「跨格汤」的区别
--------------------------
- 🔴 已证伪：跨**配方**（不同 sw/thr/ms/lrf 格）权重平均 ⇒ 把 nano 特化平均掉
  （9863+12679 → nano 0.6238 / nano4 0.5810）。
- 🟢 本工具：跨**种子**（同配方同盆地，只换 seed）⇒ 与文献里的「model soup / 种子汤」同构。

产物
----
- 合成臂落盘 `{ROOT}/trials/trial_<synth_id>/weights/epoch5.pt` + 标记文件
  `weights/seed_soup_synth.json` ⇒ **官方门禁能按普通臂读**；
- NanoBench 读数 + triad 判定 ⇒ triad 通过才跑官方三轨；
- 六项全过 ⇒ 登记进台账（`registered_by=polish_seed_soup`）
  ⇒ `six_criteria_audit` / `board_t4_watch`（板卡 T4）/ 抛光锚点选择都能看到。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）
    python manu/pipeline/npu/polish_seed_soup.py --gpu 2 --report <out.json>
    python manu/pipeline/npu/polish_seed_soup.py --gpu 2 --watch 1800   # 常驻轮询
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
RUNS = "/mnt/data/siping/runs/npu20ms"
WORK = f"{RUNS}/nano_search"
LEDGER = f"{WORK}/nano_search_ledger.json"
CACHE_DIR = f"{RUNS}/tiny_ref"
REPORT = f"{WORK}/polish_seed_soup.json"

# ---- 六项判据（与 Baseline_v2.0 / trial_9863 ep5 同口径）----
BASE = {
    "nano": 0.708648393194707, "nano4": 0.6472117202268431, "f1": 0.8897117275427806,
    "t1": 0.888086, "t2_fp": 5, "t3": 0.948489,
}

# ---- 合成臂 id 分配（与 fine_swa_sweep 的 [96000,96999] **错开**）----
SYNTH_ID_BASE = 97000
SYNTH_ID_SPAN = 900
SIDE_CAR = "seed_soup_synth.json"
MARKER = "polish_seed_soup"

REP_RE = re.compile(r"\|rep\d+$")
_HELPERS: tuple | None = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def ensure_repo_on_path() -> None:
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def import_helpers() -> tuple:
    """延迟导入 torch 依赖的 helper（import 期不做重活）。"""
    global _HELPERS
    if _HELPERS is None:
        ensure_repo_on_path()
        from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415
        from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
        from manu.pipeline.npu import nano_search as ns  # noqa: PLC0415
        _HELPERS = (average_checkpoints, RA, ns)
    return _HELPERS


def trial_weights_dir(trial: int, weights_root: str = ROOT) -> str:
    return os.path.join(weights_root, "trials", f"trial_{int(trial)}", "weights")


def arm_ckpt(trial: int, epoch: int, weights_root: str = ROOT) -> str:
    return os.path.join(trial_weights_dir(trial, weights_root), f"epoch{int(epoch)}.pt")


def synth_dir(synth_id: int, weights_root: str = ROOT) -> str:
    return os.path.join(weights_root, "trials", f"trial_{int(synth_id)}")


def sidecar_path(synth_id: int, weights_root: str = ROOT) -> str:
    return os.path.join(synth_dir(synth_id, weights_root), "weights", SIDE_CAR)


def read_sidecar(synth_id: int, weights_root: str = ROOT) -> dict | None:
    p = sidecar_path(synth_id, weights_root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def cand_key(family: str, members: list[tuple[int, int]], weights: list[float]) -> str:
    """候选唯一键（族 + 成员(臂,epoch) + 权重）⇒ 幂等。"""
    ms = ",".join(f"{t}@{e}" for t, e in members)
    ws = ",".join(f"{float(w):.6f}" for w in weights)
    return f"{family}|{ms}|{ws}"


def alloc_synth_id(key: str, weights_root: str = ROOT) -> int:
    """确定性 id：``97000 + sha1(key) % 900``；同键复用，撞别的组合则顺延。"""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    for k in range(SYNTH_ID_SPAN):
        sid = SYNTH_ID_BASE + (h + k) % SYNTH_ID_SPAN
        d = synth_dir(sid, weights_root)
        if not os.path.isdir(d):
            return sid
        side = read_sidecar(sid, weights_root)
        if side and side.get("key") == key:
            return sid
    raise RuntimeError("种子汤合成臂 id 空间已耗尽")


# ============================================================================
# 家族（族 = 同一配方，只差随机种子）
# ============================================================================
def collect_families(led: dict, weights_root: str = ROOT) -> list[dict]:
    """把抛光臂按「**同盆地**」分组。

    分组键 = 抛光臂的 ``params.init``（**锚点权重路径**）
    ⇒ 同组臂**必然从同一个锚点权重出发** ⇒ 同一个损失盆地 ⇒ 权重平均安全。

    为什么用 init 而不是配方当分组键
    ------------------------------
    · **同锚点换种子**（rep 臂）⇒ 教科书里的「种子汤」⇒ 降方差、保留配方偏置
      ⇒ 正好打「T3 缺口 1e-4 < σ_t3=8.6e-4」这个噪声级缺口；
    · **同锚点不同抛光配置**（lr0.2 / lr0.15 / cp0.2 …）⇒ 同起点的多条短轨迹 ⇒ 也同盆地；
    · 🔴 **不同锚点即便配方相同也分到不同组**（9863@5 与 12679@3 虽然都属于 sw/lrf 家族，
      但 init 不同）⇒ 避免重蹈「跨配方汤把 nano 特化平均掉」的覆辙（9863+12679 → nano 0.6238）。

    组内成员 = 磁盘上有 ckpt 的臂（rep0 = 原配方臂，repN = 换种子复现臂）。
    """
    arms = led.get("arms") or {}
    fam: dict[str, list[dict]] = {}
    for key, arm in arms.items():
        if not isinstance(arm, dict):
            continue
        p = arm.get("params") or {}
        wk = p.get("warm_key")
        init = p.get("init")
        if not wk or not init:
            continue
        ep = arm.get("picked_epoch")
        if ep is None:
            ep = (arm.get("picked") or {}).get("epoch")
        if ep is None:
            continue
        try:
            trial = int(key)
        except (TypeError, ValueError):
            continue
        ck = arm_ckpt(trial, int(ep), weights_root)
        if not os.path.isfile(ck):
            continue
        nb = (arm.get("nanobench") or {}).get(str(int(ep))) or arm.get("picked") or {}
        cfg = str(wk).split("|", 1)[1] if "|" in str(wk) else str(wk)
        cfg = REP_RE.sub("", cfg)
        fam.setdefault(str(init), []).append({
            "trial": trial, "epoch": int(ep), "ckpt": ck,
            "nano": float(nb.get("nano") or 0.0),
            "nano4": float(nb.get("nano4") or 0.0),
            "f1": float(nb.get("f1") or 0.0),
            "warm_key": str(wk), "cfg": cfg,
        })
    out = []
    for init, mem in fam.items():
        if len(mem) < 2:
            continue
        mem.sort(key=lambda m: (-m["nano"], m["trial"]))
        out.append({"family": init, "members": mem})
    out.sort(key=lambda f: (-len(f["members"]), -f["members"][0]["nano"]))
    return out


def generate_candidates(fam: dict) -> list[dict]:
    """组内候选：

    ① **同配置子组**（种子汤）：同一抛光配置、只差随机种子的成员
       ⇒ 全员等权 / 最佳成员 0.6 / 最佳成员 0.3 / 任意两两 0.5；
    ② **整组等权**（同锚点多配置轨迹的汤）。
    """
    mem = fam["members"]
    cands: list[dict] = []
    seen: set[str] = set()

    def _add(members: list[dict], weights: list[float], tag: str) -> None:
        ms = [(m["trial"], m["epoch"]) for m in members]
        key = cand_key(fam["family"], ms, weights)
        if key in seen:
            return
        seen.add(key)
        cands.append({"family": fam["family"], "members": ms, "weights": weights, "key": key,
                      "paths": [m["ckpt"] for m in members], "tag": tag})

    def _soups(sub: list[dict], tag: str) -> None:
        n = len(sub)
        if n < 2:
            return
        _add(sub, [1.0 / n] * n, tag)
        rest = sub[1:]
        w_rest = 0.4 / len(rest)
        _add([sub[0]] + rest, [0.6] + [w_rest] * len(rest), tag)
        w_rest = 0.7 / len(rest)
        _add([sub[0]] + rest, [0.3] + [w_rest] * len(rest), tag)
        for i in range(n):
            for j in range(i + 1, n):
                _add([sub[i], sub[j]], [0.5, 0.5], tag)

    by_cfg: dict[str, list[dict]] = {}
    for m in mem:
        by_cfg.setdefault(m["cfg"], []).append(m)
    for cfg, sub in by_cfg.items():
        sub.sort(key=lambda m: (-m["nano"], m["trial"]))
        _soups(sub, f"seed:{cfg}")
    _soups(mem, "all")
    return cands


# ============================================================================
# 合成 / 评测 / 门禁
# ============================================================================
def build_synth_ckpt(cand: dict, weights_root: str = ROOT) -> str:
    average_checkpoints, _RA, _ns = import_helpers()
    sid = cand["synth_id"]
    out_dir = os.path.join(synth_dir(sid, weights_root), "weights")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "epoch5.pt")
    average_checkpoints(cand["paths"], out, weights=cand["weights"], epoch=5,
                        note="seed-soup:" + "+".join(f"{t}@{e}" for t, e in cand["members"]))
    side = {
        "marker": MARKER, "key": cand["key"], "family": cand["family"],
        "members": [list(m) for m in cand["members"]],
        "member_paths": cand["paths"], "weights": cand["weights"],
        "synth_id": sid, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, SIDE_CAR), "w", encoding="utf-8") as fh:
        json.dump(side, fh, ensure_ascii=False, indent=2)
    return out


def evaluate_candidate(cand: dict, gpu: int, cache_dir: str = CACHE_DIR,
                       weights_root: str = ROOT) -> dict:
    _average, RA, _ns = import_helpers()
    ck = os.path.join(synth_dir(cand["synth_id"], weights_root), "weights", "epoch5.pt")
    pkl = os.path.join(cache_dir, f"seedsoup_{cand['synth_id']}.pkl")
    os.makedirs(cache_dir, exist_ok=True)
    if not RA.build_cache(ck, pkl, gpu, extra="--no-peak-attrs"):
        raise RuntimeError(f"建缓存失败：{pkl}")
    res = RA.nanobench(pkl, f"seedsoup_{cand['synth_id']}")
    if not res:
        raise RuntimeError(f"NanoBench 失败：{pkl}")
    rb8 = res["peak_tol8"]["recall_by_bucket"]
    rb4 = res["peak_tol4"]["recall_by_bucket"]
    m = {
        "nano": float(rb8["nano"]), "nano4": float(rb4["nano"]),
        "f1": float(res["peak_tol8"]["f1"]), "micro": float(rb8["micro"]),
        "small": float(rb8["small"]), "regular": float(rb8["regular"]),
        "f1_4": float(res["peak_tol4"]["f1"]), "cache": pkl,
    }
    m["triad"] = bool(m["nano"] >= BASE["nano"] and m["nano4"] >= BASE["nano4"]
                       and m["f1"] >= BASE["f1"])
    return m


def gate_candidate(cand: dict, gpu: int) -> dict:
    _average, _RA, ns = import_helpers()
    g = ns.run_official_gate(cand["synth_id"], 5, gpu)
    if not g or "error" in g:
        return {"error": (g or {}).get("error", "no_report")}
    out = {k: g.get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")}
    out["six_ok"] = bool(
        out["t1"] is not None and out["t2_fp"] is not None and out["t3"] is not None
        and out["t1"] >= BASE["t1"] and out["t2_fp"] <= BASE["t2_fp"]
        and out["t3"] >= BASE["t3"] and bool(cand.get("triad")))
    return out


def register_synth_arms(entries: list[dict], ledger_path: str = LEDGER) -> int:
    """把六项全过的种子汤登记进台账（口径与普通臂同构）。"""
    _average, _RA, ns = import_helpers()
    try:
        led = ns.load_ledger()
    except Exception as exc:  # noqa: BLE001
        log(f"[登记] 台账读取失败（跳过）：{exc}")
        return 0
    arms = led.get("arms") or {}
    n = 0
    for e in entries:
        if e.get("error"):
            continue
        sid = e.get("synth_id")
        g = e.get("gate") or {}
        if sid is None or not e.get("triad") or g.get("error"):
            continue
        if g.get("t1") is None or g.get("t3") is None:
            continue
        ck = os.path.join(synth_dir(int(sid), ROOT), "weights", "epoch5.pt")
        if not os.path.isfile(ck):
            continue
        src = (arms.get(str(e["members"][0][0])) or {}).get("params") or {}
        p = dict(src)
        p["init"] = ck
        p["synth"] = 1
        nb = {k: e.get(k) for k in ("nano", "nano4", "f1", "micro", "small", "regular", "f1_4")}
        nb.update({"epoch": 5, "epoch_key": "5", "ckpt": "epoch5.pt", "cache": e.get("cache")})
        arms[str(int(sid))] = {
            "trial": int(sid), "synth": True, "source_trial": e["members"][0][0],
            "soup_members": [list(m) for m in e["members"]], "soup_weights": e["weights"],
            "soup_family": e.get("family"),
            "params": p, "seed": None,
            "nanobench": {"5": nb}, "picked": dict(nb), "picked_epoch": 5,
            "official": {k: g.get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")},
            "registered_by": MARKER,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        n += 1
    if n:
        led["arms"] = arms
        try:
            ns.save_ledger(led)
            log(f"[登记] {n} 个种子汤候选已进台账")
        except Exception as exc:  # noqa: BLE001
            log(f"[登记] 台账写盘失败：{exc}")
            return 0
    return n


def write_report(path: str, report: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ============================================================================
# 主流程
# ============================================================================
def run_once(gpu: int, report_path: str = REPORT, limit: int = 40,
             min_members: int = 2, do_gate: bool = True) -> dict:
    _average, _RA, ns = import_helpers()
    led = ns.load_ledger()
    fams = collect_families(led)
    log(f"[族] 抛光配方族 {len(fams)} 个（成员≥{min_members}）"
        f"｜成员数分布 {[len(f['members']) for f in fams[:12]]}")
    entries: list[dict] = []
    done = 0
    for fam in fams:
        if len(fam["members"]) < min_members:
            continue
        for cand in generate_candidates(fam):
            if done >= limit:
                break
            sid = alloc_synth_id(cand["key"])
            cand["synth_id"] = sid
            side = read_sidecar(sid)
            if side and side.get("key") == cand["key"] and side.get("scored"):
                continue                      # 已算过 ⇒ 幂等跳过
            try:
                build_synth_ckpt(cand)
            except Exception as exc:  # noqa: BLE001
                log(f"[合成] 失败 {cand['key']}：{exc}")
                cand["error"] = str(exc)
                entries.append(cand)
                continue
            try:
                m = evaluate_candidate(cand, gpu)
            except Exception as exc:  # noqa: BLE001
                log(f"[评测] 失败 {cand['key']}：{exc}")
                cand["error"] = str(exc)
                entries.append(cand)
                continue
            cand.update(m)
            log(f"[评测] [{cand.get('tag','')}] 族={fam['family']} 成员={[f'{t}@{e}' for t, e in cand['members']]} "
                f"权重={[round(w,3) for w in cand['weights']]} ⇒ nano={m['nano']:.6f} "
                f"nano4={m['nano4']:.6f} F1={m['f1']:.6f} triad={m['triad']}")
            if m["triad"] and do_gate:
                g = gate_candidate(cand, gpu)
                cand["gate"] = g
                log(f"[门禁] synth={sid} T1={g.get('t1')} fp={g.get('t2_fp')} "
                    f"T3={g.get('t3')} six_ok={g.get('six_ok')}")
            # 标记已算（幂等）
            sc = os.path.join(synth_dir(sid), "weights", SIDE_CAR)
            try:
                with open(sc, encoding="utf-8") as fh:
                    side = json.load(fh)
                side["scored"] = True
                side["nanobench"] = {k: m.get(k) for k in ("nano", "nano4", "f1")}
                side["triad"] = bool(m["triad"])
                side["gate"] = cand.get("gate")
                with open(sc, "w", encoding="utf-8") as fh:
                    json.dump(side, fh, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
            entries.append(cand)
            done += 1
    n_reg = register_synth_arms(entries)
    rep = {"ts": time.strftime("%F %T"), "families": len(fams), "candidates": len(entries),
           "registered": n_reg, "entries": entries}
    write_report(report_path, rep)
    log(f"[完成] 候选 {len(entries)} 个｜登记 {n_reg} 个｜报告 {report_path}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="抛光家族种子汤（同配方换种子权重平均）")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--min-members", type=int, default=2)
    ap.add_argument("--no-gate", action="store_true", help="只跑 NanoBench，不跑官方门禁")
    ap.add_argument("--watch", type=int, default=0,
                    help=">0 ⇒ 常驻轮询（秒），每轮重新扫族（rep 臂落地后自动补汤）")
    args = ap.parse_args()
    if args.watch and args.watch > 0:
        while True:
            try:
                run_once(args.gpu, args.report, args.limit, args.min_members, not args.no_gate)
            except Exception as exc:  # noqa: BLE001
                log(f"[循环] 本轮异常（继续）：{exc}")
            time.sleep(int(args.watch))
    run_once(args.gpu, args.report, args.limit, args.min_members, not args.no_gate)


if __name__ == "__main__":
    main()
