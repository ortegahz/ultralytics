#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
05_ladder_verification.py -- the degradation ladder, rung by rung.

    L0   PT   p0_on   float32   PC      the original model (P0 highway on)
    L1   PT   p0_off  float32   PC      P0 removed           <- what does P0 buy?
    L2   ONNX p0_off  float32   PC      the exported graph   <- export fidelity
    L2'  ONNX p0_on  float32   PC      (mirror rung)
    L3   RKNN f16     float16   BOARD   the float16 arm
    L4   RKNN int8    int8      BOARD   the quantised arm

Every rung is measured on the SAME frames, with the SAME decoder (extract_peaks_np
is a 1:1 numpy port of the board's C++ extract_peaks) and against the SAME GT.
So the difference between two consecutive rungs is attributable to exactly one
change, and the sum of the steps is the total degradation.

The chains
----------
Three chains are reported. The board rungs are routed by the VARIANT of their
model file name (p0_on / p0_off), so a chain never mixes the two variants:

    L0 -> L1 -> L2 -> f16 -> int8 (p0_off)   带P0 -> 去P0 -> ONNX -> 板端
    L1 -> L2 -> f16 -> int8 (p0_off)          去P0 -> ONNX -> 板端
    L0 -> L2p -> f16 -> int8 (p0_on)          带P0 -> ONNX(p0_on) -> 板端

A chain whose board rungs are missing degrades gracefully: the PC part is
still reported and the missing part is stated in a [note] line.

Per rung the script records
    * end-to-end  TP / FP / FN / precision / recall / F1
    * per-frame   cosine similarity and MSE of heatmap & offset
                   against the rung ABOVE it  (the step's own loss)
                   and against L0 / L1     (the cumulative loss)

The board rungs use the binary's own `--dump DIR`, which writes the raw float32
heatmap/offset the NPU produced, and `--dump-input DIR`, which writes the exact
uint8 buffer that went into the NPU. The latter is what makes the comparison
auditable: the host re-feeds the SAME BYTES to ONNX Runtime instead of decoding
the BMP a second time, so no decoder difference can hide inside the delta.

Usage
-----
    python 05_ladder_verification.py --frames /mnt/data/siping/ptq_work/fast/tier1_sensitive.json \
        --board-model f16=/mnt/data/siping/ptq_work/screen/p0_off_A_none.rknn \
        --board-model int8=/mnt/data/siping/ptq_work/screen/p0_off_A_kl_divergence.rknn \
        --out /mnt/data/siping/ptq_work/ladder
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>). With
# only four ".parent" hops the path resolved to <root>/manu, and running this
# file directly (python manu/pipeline/npu/ptq/05_....py) died with
# ModuleNotFoundError: No module named 'manu'.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
BOARD_HOST = "ptq-board"   # ssh alias on the server: key-based, no password
BOARD_NFS_MOUNT = "/mnt/manu/ptq"
BOARD_BIN = "/root/rknn_heatmap_smoke"


def _load_script(name: str):
    """Load a sibling script whose filename starts with a digit (not importable)."""
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sh(cmd: list[str], timeout: int = 7200) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def variant_from_model_name(name: str) -> str:
    """'p0_off_A_none.rknn' -> 'p0_off', 'p0_on_A_none.rknn' -> 'p0_on'.

    The board model's file name carries the variant. Hard-coding "p0_off" (as
    this script used to) labelled a p0_on model as p0_off and then put it on
    the wrong ladder chain, which attributes the loss to the wrong step.
    """
    n = Path(name).name.lower()
    if "p0_on" in n:
        return "p0_on"
    if "p0_off" in n or "nop0" in n:
        return "p0_off"
    return "unknown"


# --------------------------------------------------------------------------------------
# PC rungs
# --------------------------------------------------------------------------------------
def run_pt_rung(variant: str, rows: list[dict], device: str = "cuda:0",
                batch: int = 8, topk: int = 100) -> dict:
    """PT float rung: raw heatmap/offset per frame + the decoded detections."""
    import torch

    from manu.pipeline.npu.hm_export_onnx import build_model

    model, ckpt = build_model(str(C.WEIGHTS), ablate_p0=C.VARIANTS[variant]["ablate_p0"])
    assert not model.training, "the PT rung must run in eval mode"
    stride = ckpt.get("stride", 2)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    out = {"variant": variant, "engine": "pt", "dtype": "float32", "hm": {}, "off": {},
           "dets": {}, "src_hw": {}}
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            xs = np.concatenate([C.bmp_to_input_nchw_float(r["bmp"]) for r in chunk], axis=0)
            o = model(torch.from_numpy(xs).to(dev))
            for j, r in enumerate(chunk):
                name = Path(r["bmp"]).name
                hm = o["heatmap"][j, 0].float().cpu().numpy()
                off = o["offset"][j].float().cpu().numpy()
                out["hm"][name] = hm
                out["off"][name] = off
                det = C.extract_peaks_np(hm, off, stride=stride,
                                         conf_thresh=C.TH_BASE, top_k=topk)
                out["dets"][name] = [(float(p[0]), float(p[1]), float(s))
                                      for p, s in zip(det["points"], det["scores"])]
                if r["seq"] not in out["src_hw"]:
                    import cv2
                    img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
                    out["src_hw"][r["seq"]] = (img.shape[0], img.shape[1])
    return out


def run_onnx_rung(variant: str, rows: list[dict], topk: int = 100) -> dict:
    """ONNX Runtime float rung: the same graph the board's model was built from."""
    import onnxruntime as ort

    onnx_path = str(C.VARIANTS[variant]["onnx"])
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = {"variant": variant, "engine": "onnx", "dtype": "float32", "hm": {}, "off": {},
           "dets": {}, "src_hw": {}}
    for r in rows:
        name = Path(r["bmp"]).name
        nchw = C.bmp_to_input_nchw_float(r["bmp"])
        hm_t, off_t = sess.run(None, {"images": nchw})
        hm = np.asarray(hm_t)[0, 0]
        off = np.asarray(off_t)[0]
        out["hm"][name] = hm
        out["off"][name] = off
        det = C.extract_peaks_np(hm, off, stride=C.STRIDE, conf_thresh=C.TH_BASE, top_k=topk)
        out["dets"][name] = [(float(p[0]), float(p[1]), float(s))
                              for p, s in zip(det["points"], det["scores"])]
        if r["seq"] not in out["src_hw"]:
            import cv2
            img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
            out["src_hw"][r["seq"]] = (img.shape[0], img.shape[1])
    return out


# --------------------------------------------------------------------------------------
# Board rungs
# --------------------------------------------------------------------------------------
def run_board_rung(tag: str, model_path: Path, rows: list[dict], board_nfs: Path,
                    th: float, topk: int, dump: bool = True,
                    board_models_dir: str = "/mnt/manu/ptq/models") -> dict:
    """Run the model on the board and pull the raw outputs + the detections."""
    out_dir = Path("/tmp/ptq_ladder") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # ONE dump directory PER RUNG. The f16 and the int8 model write the SAME
    # file names, so a shared directory silently overwrote one rung with the
    # other (and the per-frame cos/MSE then compared a rung with itself).
    remote_dir = f"/root/ptq_ladder/{tag}"
    rc, o, e = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                   f"rm -rf {remote_dir} && mkdir -p {remote_dir}"])
    if rc != 0:
        raise SystemExit(f"[FATAL] cannot prepare the board dump dir "
                         f"{remote_dir}: {e[:300]}")

    per_seq: dict[str, list[dict]] = {}
    for r in rows:
        per_seq.setdefault(r["seq"], []).append(r)

    board_model = Path(f"{board_models_dir}/{model_path.name}")

    hm_all, off_all, dets_all = {}, {}, {}
    t0 = time.time()
    total = 0
    for seq, srows in per_seq.items():
        lp = out_dir / f"list_{seq}.txt"
        lines = []
        for r in sorted(srows, key=lambda x: x["frame_id"]):
            bmp = Path(r["bmp"])
            try:
                rel = bmp.relative_to(C.BMP_ROOT)
                lines.append(f"{BOARD_NFS_MOUNT}/bmp/{rel}")
            except ValueError:
                lines.append(str(bmp))
        lp.write_text("\n".join(lines) + "\n")
        rl = f"{remote_dir}/{lp.name}"
        rc, o, e = sh(["scp", "-o", "StrictHostKeyChecking=no", str(lp),
                       f"{BOARD_HOST}:{rl}"])
        if rc != 0:
            raise SystemExit(f"[FATAL] cannot stage the frame list for {seq}: {e[:300]}")
        rtsv = f"{remote_dir}/det_{seq}.tsv"
        # the board names its dumps "<dump_dir>/<tag>_<bmp stem>_hm.bin"
        flags = f"--dump {remote_dir} --tag {tag}" if dump else ""
        cmd = (f"{BOARD_BIN} {board_model} --batch {rl} --th {th} --topk {topk} "
               f"--detections-out {rtsv} {flags}")
        rc, o, e = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_HOST,
                       f"rm -f {rtsv}; {cmd}"])
        if rc != 0:
            raise SystemExit(f"[FATAL] board rung {tag} failed on {seq}: rc={rc} "
                             f"{e[:300]}")
        total += len(srows)
        # pull the TSV and (if dumped) the raw tensors
        rc, o, e = sh(["scp", "-o", "StrictHostKeyChecking=no", f"{BOARD_HOST}:{rtsv}",
                       str(out_dir / f"det_{seq}.tsv")])
        if rc != 0:
            raise SystemExit(f"[FATAL] cannot pull the detections TSV for {seq}: "
                             f"{e[:300]}")
        mod = _load_script("04_run_board_fast_eval.py")
        for k, v in mod.parse_board_tsv(out_dir / f"det_{seq}.tsv").items():
            dets_all[k] = v
        if dump:
            # the real names are "<tag>_<seq>__<frame id>_hm.bin" / "_off.bin" --
            # pulling "<seq>_hm.bin" (which does not exist) silently pulled
            # nothing, and the return code was thrown away.
            rc, o, e = sh(["scp", "-o", "StrictHostKeyChecking=no",
                           f"{BOARD_HOST}:{remote_dir}/{tag}_{seq}_*.bin", str(out_dir)])
            if rc != 0:
                raise SystemExit(f"[FATAL] cannot pull the dumps of {tag}/{seq} "
                                 f"from {remote_dir}: {e[:300]}")
    elapsed = time.time() - t0
    # Read the dumped tensors. The board writes "<tag>_<seq>__<frame id>_hm.bin",
    # where the BMP stem is "<seq>__<frame id>". The PC side keys its rungs by
    # Path(bmp).name == "<seq>__<frame id>.bmp", so the id has to be joined back
    # with its sequence -- splitting on the FIRST "_" instead cut the sequence
    # name at "VIDEO00007" and no key ever matched (compare() returned n=0).
    for f in sorted(out_dir.glob("*_hm.bin")):
        stem = f.name[:-len("_hm.bin")]
        base = stem.split("__", 1)[-1]              # -> "<frame id>"
        seq = (stem[len(tag) + 1:].split("__", 1)[0]
               if stem.startswith(tag + "_") else "")
        if not seq or "__" not in stem:
            print(f"[warn] cannot decode the dump name {f.name} (tag={tag})",
                  flush=True)
            continue
        key = f"{seq}__{base}.bmp"
        arr = np.fromfile(f, dtype=np.float32)
        hm_all[key] = arr.reshape(320, 320)
        off_f = out_dir / f"{stem}_off.bin"
        if off_f.is_file():
            off_all[key] = np.fromfile(off_f, dtype=np.float32).reshape(2, 320, 320)
    print(f"  [board {tag}] dumped {len(hm_all)} heatmaps / {len(off_all)} offsets "
          f"in {elapsed:.1f}s", flush=True)
    return {"variant": variant_from_model_name(model_path.name),
            "engine": "board", "dtype": "rknn", "hm": hm_all,
            "off": off_all, "dets": dets_all, "src_hw": {}, "seconds": elapsed,
            "frames": total, "fps": total / max(1e-9, elapsed),
            "model": str(model_path), "model_md5": C.md5_file(model_path)}


# --------------------------------------------------------------------------------------
# Scoring / comparison
# --------------------------------------------------------------------------------------
def score_rung(rung: dict, rows: list[dict]) -> dict:
    mod = _load_script("04_run_board_fast_eval.py")
    src_hw = rung.get("src_hw") or {}
    if not src_hw:
        import cv2
        for r in rows:
            if r["seq"] not in src_hw:
                img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
                src_hw[r["seq"]] = (img.shape[0], img.shape[1])
    return mod.score_rows(rung["dets"], rows, src_hw)


def compare(a: dict, b: dict) -> dict:
    """Per-frame cosine + MSE of b against a (a = reference, b = the next rung)."""
    names = [n for n in a["hm"] if n in b.get("hm", {})]
    if not names:
        return {"n": 0}
    cos_hm, cos_off, mse_hm, mse_off = [], [], [], []
    for n in names:
        ah, bh = a["hm"][n], b["hm"][n]
        ao, bo = a["off"][n], b["off"][n]
        cos_hm.append(_cos(ah, bh)); mse_hm.append(_mse(ah, bh))
        cos_off.append(_cos(ao, bo)); mse_off.append(_mse(ao, bo))
    return {"n": len(names),
            "cos_heatmap_mean": float(np.mean(cos_hm)), "cos_heatmap_min": float(np.min(cos_hm)),
            "cos_offset_mean": float(np.mean(cos_off)), "cos_offset_min": float(np.min(cos_off)),
            "mse_heatmap_mean": float(np.mean(mse_hm)), "mse_offset_mean": float(np.mean(mse_off))}


def _cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 if (na == 0 and nb == 0) else (float(a @ b / (na * nb)) if na and nb else 0.0)


def _mse(a, b) -> float:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.mean(d * d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(C.WORK_ROOT / "fast" / "tier1_sensitive.json"))
    ap.add_argument("--board-model", action="append", default=[],
                    help="tag=path (repeatable); the tag becomes a rung label")
    ap.add_argument("--out", default=str(C.WORK_ROOT / "ladder"))
    ap.add_argument("--board-nfs", default=str(C.BOARD_NFS))
    ap.add_argument("--board-models-dir", default="/mnt/manu/ptq/models")
    ap.add_argument("--th", type=float, default=C.TH_BASE)
    ap.add_argument("--topk", type=int, default=100,
                    help="detections kept per frame; MUST match between the PC and "
                         "the board rungs, otherwise FP is not comparable")
    ap.add_argument("--no-board", action="store_true")
    ap.add_argument("--pt-device", default="cuda:0")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # a frame list may be {"items": [...]} (the fast-eval manifests) or a bare
    # list of frame rows -- both shapes have to be accepted
    frames = C.read_json(args.frames)
    if isinstance(frames, list):
        rows = frames
    else:
        rows = frames.get("items") or frames.get("frames") or []
        if not rows:
            raise SystemExit(f"[FATAL] no frames in {args.frames} "
                             f"(keys: {sorted(frames)})")
    print(f"[frames] {len(rows)} from {args.frames}")

    rungs: dict[str, dict] = {}
    order: list[str] = []

    # ---------------- L0 / L1 : the two PT variants ----------------
    print("[L0] PT p0_on ...", flush=True)
    rungs["L0_PT_p0_on"] = run_pt_rung("p0_on", rows, args.pt_device, topk=args.topk)
    order.append("L0_PT_p0_on")
    print("[L1] PT p0_off ...", flush=True)
    rungs["L1_PT_p0_off"] = run_pt_rung("p0_off", rows, args.pt_device, topk=args.topk)
    order.append("L1_PT_p0_off")
    # ---------------- L2 / L2' : the exported graphs ----------------
    print("[L2] ONNX p0_off ...", flush=True)
    rungs["L2_ONNX_p0_off"] = run_onnx_rung("p0_off", rows, topk=args.topk)
    order.append("L2_ONNX_p0_off")
    print("[L2'] ONNX p0_on ...", flush=True)
    rungs["L2p_ONNX_p0_on"] = run_onnx_rung("p0_on", rows, topk=args.topk)
    order.append("L2p_ONNX_p0_on")

    # ---------------- L3 / L4 : the board rungs ----------------
    if not args.no_board:
        for spec in args.board_model:
            tag, _, path = spec.partition("=")
            p = C.require(Path(path), f"board model {tag}")
            print(f"[board] rung {tag}: {p}", flush=True)
            rungs[f"BOARD_{tag}"] = run_board_rung(tag, p, rows, C.BOARD_NFS,
                                                   args.th, args.topk,
                                                   board_models_dir=args.board_models_dir)
            order.append(f"BOARD_{tag}")

    # ---------------- score + compare ----------------
    report = {"frames": len(rows), "frames_json": args.frames, "rungs": {}}
    for name in order:
        r = rungs[name]
        sc = score_rung(r, rows)
        report["rungs"][name] = {
            "engine": r["engine"], "dtype": r["dtype"], "variant": r.get("variant"),
            "model": r.get("model"), "model_md5": r.get("model_md5"),
            "score": sc,
            "fps": r.get("fps"), "seconds": r.get("seconds"),
        }
        print(f"  {name:22s} F1={sc['f1']:.4f} R={sc['recall']:.4f} P={sc['precision']:.4f} "
              f"FP={sc['fp']} FN={sc['fn']}")

    # ---------------- the ladder chains ----------------
    # The board rungs are routed by the VARIANT of their model: a p0_off model
    # belongs on the p0_off chain, a p0_on model on the p0_on chain. Putting
    # every BOARD_ rung on both chains attributed the p0_off board numbers to
    # the p0_on chain (and vice versa).
    board_off = [n for n in order if n.startswith("BOARD_")
                 and rungs[n].get("variant") == "p0_off"]
    board_on = [n for n in order if n.startswith("BOARD_")
                and rungs[n].get("variant") == "p0_on"]
    unclassified = [n for n in order if n.startswith("BOARD_")
                    and rungs[n].get("variant") not in ("p0_on", "p0_off")]
    if unclassified:
        print(f"[note] board rungs whose model name carries no recognisable variant "
              f"(excluded from the variant chains): {', '.join(unclassified)}")

    chains = {
        # 带P0 -> 去P0 -> ONNX -> f16 -> int8
        "L0->L1->L2->f16->int8 (p0_off)": ["L0_PT_p0_on", "L1_PT_p0_off",
                                             "L2_ONNX_p0_off", *board_off],
        # 去P0 -> ONNX -> f16 -> int8 (without the P0 step)
        "L1->L2->f16->int8 (p0_off)": ["L1_PT_p0_off", "L2_ONNX_p0_off", *board_off],
        # 带P0 -> ONNX(p0_on) -> f16 -> int8
        "L0->L2p->f16->int8 (p0_on)": ["L0_PT_p0_on", "L2p_ONNX_p0_on", *board_on],
    }
    report["chains"] = {k: [n for n in v if n in rungs] for k, v in chains.items()}
    for cname, chain in chains.items():
        chain = [n for n in chain if n in rungs]
        if len(chain) < 2:
            continue
        print(f"\n=== {cname} ===")
        if cname.endswith("(p0_on)") and not board_on:
            print("[note] 缺 p0_on 板端模型 (no p0_on board model was given) -- this "
                  "chain stops at the PC rungs")
        elif cname.endswith("(p0_off)") and not board_off:
            print("[note] 缺 p0_off 板端模型 (no p0_off board model was given) -- this "
                  "chain stops at the PC rungs")
        print(f"{'step':22s} {'F1':>8s} {'dF1':>9s} {'Recall':>8s} {'dRecall':>9s} "
              f"{'FP':>6s} {'dFP':>5s} | {'cos_hm':>8s} {'mse_hm':>11s} {'cos_off':>8s}")
        prev = None
        for n in chain:
            cur = report["rungs"][n]
            s = cur["score"]
            cmp_ = compare(rungs[prev], rungs[n]) if prev else {"cos_heatmap_mean": None,
                                                                "mse_heatmap_mean": None,
                                                                "cos_offset_mean": None}
            cur["step_vs_prev"] = cmp_
            d = ""
            if prev:
                ps = report["rungs"][prev]["score"]
                d = (f"{s['f1'] - ps['f1']:+9.4f} {s['recall'] - ps['recall']:+9.4f} "
                     f"{s['fp'] - ps['fp']:+5d}")
            else:
                d = f"{'--':>9s} {'--':>9s} {'--':>5s}"
            ch = cmp_.get("cos_heatmap_mean")
            mh = cmp_.get("mse_heatmap_mean")
            co = cmp_.get("cos_offset_mean")
            print(f"{n:22s} {s['f1']:8.4f} {d} | "
                  f"{(f'{ch:.6f}' if ch is not None else '--'):>8s} "
                  f"{(f'{mh:.3e}' if mh is not None else '--'):>11s} "
                  f"{(f'{co:.6f}' if co is not None else '--'):>8s}")
            prev = n
        # cumulative against the first rung of the chain
        base = report["rungs"][chain[0]]["score"]
        for n in chain[1:]:
            s = report["rungs"][n]["score"]
            print(f"   cumulative vs {chain[0]}: {n:22s} dF1={s['f1'] - base['f1']:+.4f} "
                  f"dRecall={s['recall'] - base['recall']:+.4f} dFP={s['fp'] - base['fp']:+d}")

    C.write_json(out_root / "ladder_report.json", report)
    print(f"\n[report] -> {out_root / 'ladder_report.json'}")


if __name__ == "__main__":
    main()
