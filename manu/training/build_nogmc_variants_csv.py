#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把**变体**（跨种子 soup / 同 run 相邻 epoch 平均）的门禁结果汇成一张 CSV，
交给判定器的 `--variants-csv` 通道（2026-11-03）。

为什么需要它
============
变体**不是 study 里的 trial** ⇒ 不在 `optuna_summary.csv` 里 ⇒ 判定器按配方分组时
**看不见它们** ⇒ 已实测的事实：`cs_t0008`（T1 0.8959005 / T2 F1 0.99862 FP 42 /
T3 0.868572 / verdict=PASS）是**严格 PASS**（T1/T2 两个 veto 都过），却对判定器不可见；
判定器当时给出的候选（`w8.0·lr2.8819`，n=2）T3 只有 0.845。

⚠️ **变体绝不能混进「同配方种子中位」**（口径铁律：决策单元 = 配置的中位）：
  · soup 是**多粒子权重平均**，不是该配置的第 N 个种子；
  · SWA 是**同 run 相邻 epoch 平均**，也不是种子。
⇒ 变体**另起一组**，硬门禁用**它自己的三轨指标**判，
  多种子证据取 `seed_evidence_n`（soup = 成员数；SWA = 同配方完成 trial 数）。

用法（服务器，conda `uav`，根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/build_nogmc_variants_csv.py \
  --swa-dir /mnt/data/siping/runs/nogmc_focus/swa \
  --swa-dir /mnt/data/siping/runs/nogmc_t3/swa \
  --families-json manu/training/nogmc_t3_families.json \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out /mnt/data/siping/runs/nogmc_t3/variants.csv
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

# 判定器的 15 维配方键（顺序与 judge_handover_candidate.CFG_COLS 一致）
CFG_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
            "init_from", "epochs", "backbone_lr_scale", "freeze_backbone_epochs")

OUT_COLS = ("variant_key", "variant_kind", "variant_source", "seed_evidence_n",
            "seed_evidence_source", "trial", "status", "gate_verdict",
            "t1_f1", "t1_fp", "t2_f1", "t2_fp", "t3_f1", "t3_fp",
            "utility", "utility_vs_rc1", "weights") + CFG_COLS


# ---------------------------------------------------------------- study 读取
def _decode(conn) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for tid, num, stt in conn.execute("SELECT trial_id, number, state FROM trials"):
        out[tid] = {"num": num, "state": stt, "params": {}, "attrs": {}}
    for tid, pn, pv, dj in conn.execute(
            "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params"):
        if tid not in out:
            continue
        v = pv
        try:
            d = json.loads(dj)
            ch = (d.get("attributes") or {}).get("choices")
            if ch:
                v = ch[int(pv)]
        except Exception:
            pass
        out[tid]["params"][pn] = v
    for tid, k, vj in conn.execute("SELECT trial_id, key, value_json FROM trial_user_attributes"):
        if tid not in out:
            continue
        try:
            out[tid]["attrs"][k] = json.loads(vj)
        except Exception:
            out[tid]["attrs"][k] = vj
    return out


def read_study(db_path: str, study_name: str) -> dict[int, dict]:
    """→ {trial_number: {state, params, attrs}}（只含该 study 的 trial）。"""
    db = Path(db_path)
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    ids = [r[0] for r in conn.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))]
    keep = set()
    for sid in ids:
        keep |= {r[0] for r in conn.execute("SELECT trial_id FROM trials WHERE study_id=?", (sid,))}
    tr = _decode(conn)
    conn.close()
    out = {}
    for tid, t in tr.items():
        if keep and tid not in keep:
            continue
        out[t["num"]] = t
    return out


def resolve_member(study_root: str, study_name: str, member: dict, cache: dict) -> dict | None:
    """按 role 或 trial 目录名解析出一个成员 trial（返回 study 条目）。"""
    key = (study_root, study_name)
    if key not in cache:
        cache[key] = read_study(str(Path(study_root) / "study.db"), study_name)
    trials = cache[key]
    if not trials:
        return None
    if member.get("role"):
        for num, t in trials.items():
            if (t["attrs"].get("role") or "") == member["role"]:
                return t
        return None
    if member.get("trial"):
        name = str(member["trial"])
        num = int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else None
        if num is not None and num in trials:
            return trials[num]
    return None


def recipe_of(params: dict) -> dict:
    out = {}
    for c in CFG_COLS:
        v = params.get(c, "")
        if isinstance(v, float):
            out[c] = f"{v:.6f}"
        elif isinstance(v, bool):
            out[c] = str(v)
        else:
            out[c] = str(v)
    return out


def same_recipe_count(trials: dict[int, dict], params: dict) -> int:
    """同配方的 COMPLETE trial 数（= 该配置的复制次数 ⇒ SWA 的多种子证据）。"""
    key = {c: params.get(c) for c in CFG_COLS}
    n = 0
    for t in trials.values():
        if t["state"] != "COMPLETE":
            continue
        if all(t["params"].get(c) == key[c] for c in CFG_COLS):
            n += 1
    return n


# ---------------------------------------------------------------- 门禁报告
def metrics_from_report(rep: dict) -> dict:
    tr = rep.get("tracks") or {}
    t1 = ((tr.get("track1_uav") or {}).get("finetune_peak") or {})
    t2 = (tr.get("track2_golden") or {})
    t3 = (tr.get("track3_blind") or {})
    return {
        "t1_f1": t1.get("f1"), "t1_fp": t1.get("fp"), "t1_tp": t1.get("tp"), "t1_th": t1.get("th"),
        "t2_f1": t2.get("f1"), "t2_fp": t2.get("fp"), "t2_tp": t2.get("tp"),
        "t3_f1": t3.get("f1"), "t3_fp": t3.get("fp"), "t3_tp": t3.get("tp"),
        "verdict": rep.get("verdict", ""),
        "weights": rep.get("weights", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swa-dir", action="append", required=True,
                    help="含 gate/<name>/gate_report.json 的目录（可给多个）")
    ap.add_argument("--families-json", default="", help="家族定义（含成员 ⇒ 配方与种子数）")
    ap.add_argument("--study-map", action="append", default=[],
                    help="tag=root:study名（用于读 <swa-dir>/crossseed_manifest.json 里的旧家族）")
    ap.add_argument("--ref", required=True, help="reference_frame.json（算 U 用）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from manu.training.mix_ratio_judgement import utility  # noqa: E402

    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    base = ref.get("frozen_baseline") or {}
    baseline = {"track1_f1": base.get("track1_f1"), "track1_fp": base.get("track1_fp"),
                "track2_f1": base.get("track2_f1"), "track2_fp": base.get("track2_fp"),
                "track3_f1": base.get("track3_f1"), "track3_fp": base.get("track3_fp")}

    families = {}
    if args.families_json and Path(args.families_json).exists():
        families = (json.loads(Path(args.families_json).read_text(encoding="utf-8")).get("families") or {})

    # tag → (root, study_name)（给旧 manifest 里的家族用）
    study_map = {}
    for spec in args.study_map:
        tag, rest = spec.split("=", 1)
        root, _, study = rest.rpartition(":")
        study_map[tag] = (root, study)

    def manifest_families(swa_dir: str) -> dict:
        """<swa-dir>/crossseed_manifest.json 里的家族（{fam: {study_tag: [成员]}}）
        ⇒ 统一转成 families JSON 的 {fam: {members: [{root, study, role|trial}]}}。"""
        mp = Path(swa_dir) / "crossseed_manifest.json"
        if not mp.exists():
            return {}
        try:
            raw = json.loads(mp.read_text(encoding="utf-8")).get("families") or {}
        except Exception:
            return {}
        out = {}
        for fam, per_study in raw.items():
            members = []
            for tag, names in (per_study or {}).items():
                root, study = study_map.get(tag, ("", ""))
                if not root:
                    continue
                for nm in names:
                    m = {"root": root, "study": study}
                    # 名字像 trial_0008 ⇒ 目录名；否则当 role
                    if str(nm).startswith("trial_"):
                        m["trial"] = nm
                    else:
                        m["role"] = nm
                    members.append(m)
            if members:
                out[fam] = {"members": members,
                            "recipe": "（来自 %s 的 manifest）" % Path(swa_dir).name}
        return out

    cache: dict = {}
    rows = []
    for swa_dir in args.swa_dir:
        gate_root = Path(swa_dir) / "gate"
        if not gate_root.is_dir():
            print("[VARIANTS] 跳过（无 gate 目录）%s" % gate_root)
            continue
        for rep_path in sorted(gate_root.glob("*/gate_report.json")):
            name = rep_path.parent.name
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            m = metrics_from_report(rep)
            if m["t1_f1"] is None or m["t2_f1"] is None or m["t3_f1"] is None:
                print("[VARIANTS] 跳过（三轨指标不全）%s" % name)
                continue

            # ---- 家族 / 配方 / 种子证据 ----
            recipe, seed_n, seed_src, source = {}, 1, "unknown", ""
            if name.startswith("cs_"):
                fam = name[3:]
                fdef = families.get(fam) or manifest_families(str(swa_dir)).get(fam) or {}
                members = fdef.get("members") or []
                used, first = [], None
                for mem in members:
                    t = resolve_member(mem.get("root", ""), mem.get("study", ""), mem, cache)
                    if t is None or t["state"] != "COMPLETE":
                        continue
                    used.append(t)
                    if first is None:
                        first = t
                if first is not None:
                    recipe = recipe_of(first["params"])
                seed_n = max(1, len(used))
                seed_src = "soup成员数（family=%s）" % fam
                source = fam + ": " + ",".join(
                    str((t.get("attrs") or {}).get("role") or t["num"]) for t in used)
                kind = "crossseed_soup"
            else:
                # SWA：sidecar 里记着源 epoch 权重 ⇒ 反查它属于哪个 trial
                side = Path(swa_dir) / (name + ".json")
                srcs = []
                if side.exists():
                    try:
                        srcs = json.loads(side.read_text(encoding="utf-8")).get("sources") or []
                    except Exception:
                        srcs = []
                kind = "swa"
                if srcs:
                    p0 = Path(srcs[0])
                    # .../runs/<study_root>/trials/trial_XXXX/weights/epochN.pt
                    trial_dir = p0.parent.parent
                    tname = trial_dir.name
                    tnum = int(tname.split("_")[-1]) if tname.split("_")[-1].isdigit() else None
                    root = trial_dir.parent.parent
                    for study_name in ("mix_ratio_search_nogmc_focus", "mix_ratio_search_nogmc_weekend",
                                       "mix_ratio_search_nogmc_t3"):
                        trials = cache.setdefault((str(root), study_name), None)
                        if trials is None:
                            trials = read_study(str(Path(root) / "study.db"), study_name)
                            cache[(str(root), study_name)] = trials
                        if trials and tnum in trials:
                            t = trials[tnum]
                            recipe = recipe_of(t["params"])
                            seed_n = max(1, same_recipe_count(trials, t["params"]))
                            seed_src = "同配方完成trial数（study=%s）" % study_name
                            source = "%s/%s" % (root.name, tname)
                            break
                    if not recipe:
                        source = source or tname
                if not source:
                    source = name

            # ---- U ----
            u = u_rc1 = None
            try:
                u, _ = utility({"track1_f1": m["t1_f1"], "track1_fp": m["t1_fp"],
                                "track2_f1": m["t2_f1"], "track2_fp": m["t2_fp"],
                                "track3_f1": m["t3_f1"], "track3_fp": m["t3_fp"]}, baseline)
                rc1 = ref.get("rc1") or {}
                if rc1.get("track1_f1"):
                    u_rc1, _ = utility({"track1_f1": m["t1_f1"], "track1_fp": m["t1_fp"],
                                        "track2_f1": m["t2_f1"], "track2_fp": m["t2_fp"],
                                        "track3_f1": m["t3_f1"], "track3_fp": m["t3_fp"]},
                                       {"track1_f1": rc1["track1_f1"], "track1_fp": rc1.get("track1_fp"),
                                        "track2_f1": rc1["track2_f1"], "track2_fp": rc1.get("track2_fp"),
                                        "track3_f1": rc1["track3_f1"], "track3_fp": rc1.get("track3_fp")})
            except Exception as exc:
                print("[VARIANTS] %s 算 U 失败：%s" % (name, exc))

            row = {
                "variant_key": name, "variant_kind": kind, "variant_source": source,
                "seed_evidence_n": seed_n, "seed_evidence_source": seed_src,
                "trial": name, "status": "SUCCESS", "gate_verdict": m["verdict"],
                "t1_f1": m["t1_f1"], "t1_fp": m["t1_fp"], "t2_f1": m["t2_f1"], "t2_fp": m["t2_fp"],
                "t3_f1": m["t3_f1"], "t3_fp": m["t3_fp"],
                "utility": u, "utility_vs_rc1": u_rc1, "weights": m["weights"],
            }
            row.update(recipe)
            rows.append(row)
            print("[VARIANTS] %-22s %-16s T1=%.6f T2=%.6f(FP %s) T3=%.6f U=%s 种子证据=%s"
                  % (name, kind, m["t1_f1"], m["t2_f1"], m["t2_fp"], m["t3_f1"],
                     ("%.6f" % u) if u is not None else "n/a", seed_n))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(OUT_COLS), extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in OUT_COLS})
    print("[VARIANTS] 写出 %s：%d 行" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
