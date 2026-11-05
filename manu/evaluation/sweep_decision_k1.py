#!/usr/bin/env python3
"""决策层旋钮 K1：(th_base × dedup_radius) 二维网格 —— no-GMC / T3 / 单帧臂上**从未测过**。

为什么值得做
------------
* T3 的阈值**冻结**在交付参数 JSON（`th_base=0.30`、`dedup_radius=768.0`）⇒ **T3 没有 peak 选择偏差**
  ⇒ 在 T3 上扫阈值属于 out-of-sample 的诚实测量。
* 已有工作只覆盖 **GMC 特征 + 24 序业务池 + 完整 pipeline**（`manu/postprocess/tune_sys_sota_longquan.py`，
  `dedup_radius` 14 档 / `th_base` 11 档），**no-GMC + T3 单序列 + 单帧臂**这一格**从未测过**。
* RC2 在那 4 个锚点上的实测：32/0.22→0.848330、32/0.30→0.873163、**768/0.22→0.898487（最佳）**、
  768/0.30→0.897694 ⇒ 网格在最优附近**很平**（best 仅 +0.08 pp）⇒ 本脚本大概率也复现「平」，
  **但那本身是可入库的证伪结论**（防止团队重复扫）。

**关键实现**：**不重写决策逻辑** —— 直接调用门禁自己用的
`manu/evaluation/system_sota_longquan.py`，只换 `--params-json` 的 `th_base` 与 `--dedup-radius`
⇒ **口径与门禁逐位一致**（同一份交付 JSON、同一套 GT、同一套 `--no-pipeline` 单帧臂）。

**零 GPU**：只读已落盘的 `ms_cache/*.pkl`。

用法（服务器，conda uav，根 /tmp/pycharm_project_10ae9e2e）：
    python manu/evaluation/sweep_decision_k1.py \
        --arms "trial_0049=/mnt/.../trials/trial_0049/gate/epoch20/ms_cache,cs_t0008p2=/mnt/.../swa/gate/cs_t0008p2/ms_cache" \
        --out-csv /mnt/data/siping/runs/nogmc_t3/interp/decision_sweep_k1.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DELIVERY_JSON = REPO / "manu/evaluation/system_sota_longquan_delivery_params.json"
LABELS_ROOT = "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
BLIND = "VIDEO00007_19700101_001324"
GOLDEN = ",".join([
    "VIDEO00009_19700101_001531", "VIDEO00029_19700101_014053", "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052", "VIDEO00006_19700101_000703", "VIDEO00008_19700101_001307",
    "VIDEO00007_19700101_001818" if False else "VIDEO00007_19700101_000818",
])
THS = [0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]
RADII = [64, 128, 256, 512, 768]


def write_params(th: float, dest: Path) -> Path:
    """复制交付参数 JSON，只改 th_base（dedup_radius 走 CLI 覆盖）。"""
    payload = json.loads(DELIVERY_JSON.read_text(encoding="utf-8"))
    payload.setdefault("params", {})["th_base"] = th
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return dest


def run_one(cache_root: str, th: float, radius: float, workdir: Path) -> dict:
    pj = write_params(th, workdir / f"params_th{th:.2f}.json")
    out_json = workdir / f"t3_th{th:.2f}_r{radius:g}.json"
    cmd = [
        sys.executable, "manu/evaluation/system_sota_longquan.py",
        "--cache-root", cache_root,
        "--labels-root", LABELS_ROOT,
        "--params-json", str(pj),
        "--dedup-radius", str(radius),
        "--seq", BLIND,
        "--no-pipeline",
        "--out-json", str(out_json),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, cwd=str(REPO), env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
    except Exception as e:  # noqa: BLE001
        return {"th": th, "radius": radius, "error": repr(e)[:200]}
    try:
        d = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"th": th, "radius": radius, "error": "read " + repr(e)[:120]}
    o = d.get("overall", d)
    return {
        "th": th, "radius": radius,
        "f1": o.get("f1"), "tp": o.get("tp"), "fp": o.get("fp"), "fn": o.get("fn"),
        "fn_absent": o.get("fn_absent"), "fn_compete": o.get("fn_compete"),
    }


def run_golden(cache_root: str, th: float, radius: float, workdir: Path) -> dict:
    """同一参数下跑 T2（金池 7 序列），验证 T2 硬门禁是否仍满足。"""
    pj = write_params(th, workdir / f"params_g_th{th:.2f}.json")
    out_json = workdir / f"t2_th{th:.2f}_r{radius:g}.json"
    cmd = [
        sys.executable, "manu/evaluation/system_sota_longquan.py",
        "--cache-root", cache_root,
        "--labels-root", LABELS_ROOT,
        "--params-json", str(pj),
        "--dedup-radius", str(radius),
        "--seqs", GOLDEN,
        "--no-pipeline",
        "--out-json", str(out_json),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, cwd=str(REPO), env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
    except Exception as e:  # noqa: BLE001
        return {"th": th, "radius": radius, "error": repr(e)[:200]}
    try:
        d = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"th": th, "radius": radius, "error": "read " + repr(e)[:120]}
    o = d.get("overall", d)
    return {"th": th, "radius": radius, "f1": o.get("f1"), "tp": o.get("tp"),
            "fp": o.get("fp"), "fn": o.get("fn")}


def main() -> int:
    ap = argparse.ArgumentParser(description="决策层 K1：th_base × dedup_radius 网格（零 GPU）")
    ap.add_argument("--arms", required=True, help="name=cache_root,name=cache_root")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--golden-check", action="store_true", help="额外跑 T2 金池验证")
    args = ap.parse_args()

    arms = []
    for part in args.arms.split(","):
        if "=" in part:
            n, p = part.split("=", 1)
            arms.append((n.strip(), p.strip()))
    workdir = Path(tempfile.mkdtemp(prefix="dec_k1_"))
    rows = []
    for name, cache in arms:
        if not Path(cache).is_dir():
            print(f"[K1] ⚠️ 缓存不存在，跳过：{name} {cache}")
            continue
        jobs = [(th, r) for th in THS for r in RADII]
        print(f"[K1] {name}：{len(jobs)} 个点（{cache}）", flush=True)
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, cache, th, r, workdir): (th, r) for th, r in jobs}
            for i, fu in enumerate(cf.as_completed(futs)):
                th, r = futs[fu]
                try:
                    res = fu.result()
                except Exception as e:  # noqa: BLE001
                    res = {"th": th, "radius": r, "error": repr(e)[:200]}
                res["arm"] = name
                res["track"] = "T3"
                rows.append(res)
                if (i + 1) % 5 == 0:
                    print(f"   [{i+1}/{len(jobs)}] th={th} r={r} f1={res.get('f1')}", flush=True)
        print(f"[K1] {name} 完成", flush=True)

    if args.golden_check and rows:
        best = {}
        for r in rows:
            if r.get("f1") is None:
                continue
            k = r["arm"]
            if k not in best or r["f1"] > best[k]["f1"]:
                best[k] = r
        for name, r in best.items():
            cache = dict(arms)[name]
            print(f"[K1] T2 金池验证：{name} th={r['th']} r={r['radius']} …", flush=True)
            g = run_golden(cache, r["th"], r["radius"], workdir)
            g["arm"] = name
            g["track"] = "T2"
            rows.append(g)
            print(f"   ⇒ T2 f1={g.get('f1')} fp={g.get('fp')}", flush=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["arm", "track", "th", "radius", "f1", "tp", "fp", "fn", "fn_absent", "fn_compete", "error"]
    with out.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    print(f"[K1] 已写 {out}（{len(rows)} 行）")

    if args.out_md:
        md = ["| arm | 轨 | th_base | dedup_radius | F1 | TP | FP | FN |",
              "|:--|:--|--:|--:|--:|--:|--:|--:|"]
        for r in sorted(rows, key=lambda x: -(x.get("f1") or 0)):
            md.append("| %s | %s | %.2f | %g | %s | %s | %s | %s |" % (
                r.get("arm"), r.get("track"), r.get("th", 0), r.get("radius", 0),
                r.get("f1"), r.get("tp"), r.get("fp"), r.get("fn")))
        Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[K1] 已写 {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
