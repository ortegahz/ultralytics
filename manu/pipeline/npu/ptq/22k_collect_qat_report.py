#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22k_collect_qat_report.py -- assemble the delivery tables of the QAT arms.

Reads, on the server:
    $W/qat/trial_0080/board_eval/<manifest>/<manifest>_threshold_sweep.json
        the board sweep of the four QAT arms (19_board_threshold_sweep.py)
    $W/qat/trial_0080/board_eval_baseline/<manifest>/<manifest>_threshold_sweep.json
        the SAME sweep for the PTQ baseline arms (22j_board_sweep_ptq_baseline.sh)
    $W/qat/trial_0080/pc_sim/<manifest>/<arm>_pcsim.json
        the PC-simulator runs (22h_pc_sim_eval_qat_arms.py)

and prints the three tables the delivery asks for:
  1. the board numbers of every arm at th = 0.30 and th = 0.45
  2. the controlled comparison against the PT float arm (the quantisation loss)
  3. the PC-simulator numbers

Usage (server):
    python manu/pipeline/npu/ptq/22k_collect_qat_report.py --work /mnt/data/siping/ptq_work
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFESTS = ["tier1_sensitive", "001324_sub2000", "golden_pool_sub1200"]
ARMS = ["pt_float", "board_a2_qat_learned", "board_a1_qat_maxclip",
        "board_a0_qat_float", "board_a1b_base_maxclip",
        "board_p0_off_A_normal", "board_p0_off_A_none"]
LABEL = {
    "pt_float": "PT float (base weights, PC)",
    "board_a2_qat_learned": "A2 QAT weights + LSQ-learned clip",
    "board_a1_qat_maxclip": "A1 QAT weights + max-calibrated clip",
    "board_a0_qat_float": "A0 QAT weights, no Q/DQ (fp16)",
    "board_a1b_base_maxclip": "A1b BASE weights + max-calibrated clip",
    "board_p0_off_A_normal": "PTQ int8 (toolkit-quantised, baseline)",
    "board_p0_off_A_none": "fp16 (toolkit, baseline)",
}


def curve_at(curve: list[dict], th: float) -> dict | None:
    for row in curve:
        if abs(float(row["th"]) - th) < 1e-9:
            return row
    return None


def load_sweep(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt_row(manifest: str, arm: str, th: float, c: dict) -> str:
    return (f"{manifest:<20}{LABEL.get(arm, arm):<40}{th:>5.2f}"
            f"{c['f1']:>9.4f}{c['precision']:>9.4f}{c['recall']:>9.4f}"
            f"{c['tp']:>7}{c['fp']:>7}{c['fn']:>7}{c.get('det', 0):>7}")


HDR = (f"{'manifest':<20}{'arm':<40}{'th':>5}{'F1':>9}{'P':>9}{'R':>9}"
       f"{'TP':>7}{'FP':>7}{'FN':>7}{'det':>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/data/siping/ptq_work")
    ap.add_argument("--ths", default="0.30,0.45")
    args = ap.parse_args()
    ths = [float(x) for x in args.ths.split(",") if x.strip()]
    root = Path(args.work) / "qat" / "trial_0080"

    qat = {m: load_sweep(root / "board_eval" / m / f"{m}_threshold_sweep.json")
           for m in MANIFESTS}
    base = {m: load_sweep(root / "board_eval_baseline" / m / f"{m}_threshold_sweep.json")
            for m in MANIFESTS}

    print("#" * 120)
    print("# 1. BOARD -- every arm, th = " + " / ".join(f"{t:.2f}" for t in ths))
    print("#" * 120)
    print(HDR)
    for m in MANIFESTS:
        for src, label in ((qat, "QAT"), (base, "BASE")):
            rep = src.get(m)
            if not rep:
                continue
            for arm in ARMS:
                a = rep.get("arms", {}).get(arm)
                if not a:
                    continue
                for th in ths:
                    c = curve_at(a.get("curve", []), th)
                    if c:
                        print(fmt_row(m, arm, th, c))

    print("")
    print("#" * 120)
    print("# 2. CONTROLLED COMPARISON -- the quantisation loss against the PT float arm")
    print("#    loss = F1(PT float) - F1(arm);  the baseline row is the toolkit's int8 PTQ model")
    print("#" * 120)
    print(f"{'manifest':<20}{'arm':<40}{'th':>5}{'F1':>9}{'loss vs PT float':>18}")
    for m in MANIFESTS:
        rep = qat.get(m) or base.get(m)
        merged = {}
        for src in (base.get(m), qat.get(m)):
            if src:
                merged.update(src.get("arms", {}))
        pt = merged.get("pt_float")
        for arm in ARMS:
            a = merged.get(arm)
            if not a or arm == "pt_float":
                continue
            for th in ths:
                c = curve_at(a.get("curve", []), th)
                p = curve_at(pt.get("curve", []), th) if pt else None
                if not c:
                    continue
                loss = (p["f1"] - c["f1"]) if p else float("nan")
                print(f"{m:<20}{LABEL.get(arm, arm):<40}{th:>5.2f}{c['f1']:>9.4f}{loss:>18.4f}")

    print("")
    print("#" * 120)
    print("# 3. PC SIMULATOR (rknn-toolkit2 simulator, CPU emulation of the NPU)")
    print("#" * 120)
    print(HDR)
    for m in MANIFESTS:
        for f in sorted((root / "pc_sim" / m).glob("*_pcsim.json")):
            d = json.loads(f.read_text())
            for th, c in d.get("curve", {}).items():
                print(f"{m:<20}{d['arm'] + ' (pcsim)':<40}{float(th):>5.2f}"
                      f"{c['f1']:>9.4f}{c['precision']:>9.4f}{c['recall']:>9.4f}"
                      f"{c['tp']:>7}{c['fp']:>7}{c['fn']:>7}{c['det']:>7}")

    print("")
    print("#" * 120)
    print("# 4. BOARD IO ATTRIBUTES -- the out[N] zp/scale the board actually reports")
    print("#" * 120)
    for m in MANIFESTS:
        for src in (base.get(m), qat.get(m)):
            if not src:
                continue
            for arm, a in src.get("arms", {}).items():
                if arm == "pt_float":
                    continue
                br = a.get("board_run") or {}
                ia = br.get("input_attr") or {}
                print(f"{m:<20}{arm}")
                print(f"{'':<20}  {ia.get('raw')}")
                for o in br.get("out_attrs", []) or []:
                    print(f"{'':<20}  {o.get('raw')}")

    print("")
    print("#" * 120)
    print("# 5. MODEL IDENTITY -- the md5 of the model that produced each board row")
    print("#" * 120)
    for m in MANIFESTS:
        for src in (base.get(m), qat.get(m)):
            if not src:
                continue
            for arm, a in src.get("arms", {}).items():
                if arm == "pt_float":
                    continue
                print(f"{m:<20}{arm:<30}md5={a.get('model_md5')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
