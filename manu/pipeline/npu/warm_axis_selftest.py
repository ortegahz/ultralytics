#!/usr/bin/env python3
"""warm-start 抛光轴 + 分数存盘轴的**自检**（2026-11-13 新增）。

三个断言（不依赖 GPU、不依赖台账）：
  ① 默认配方的训练命令串与**旧版逐字节相同**（老臂零影响）
  ② 抛光臂的命令串里三个新旋钮都出现且取值正确
  ③ 训练脚本的三个新参数**默认全关**（--help 里默认值可见）

用法：
    python manu/pipeline/npu/warm_axis_selftest.py            # 本地/服务器通用
    python manu/pipeline/npu/warm_axis_selftest.py --ledger   # 额外用台账查抛光臂
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

GOLDEN_OLD_CMD = (
    "cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && "
    "/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/training/optuna_mix_ratio_search.py --worker "
    "--trial-number 1 --gpu-id 0 --output-root /mnt/data/siping/runs/npu20ms_offcal "
    "--longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc "
    "--uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc --weights /INIT "
    "--baseline-cache /mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl "
    "--epochs 5 --sched-epochs 5 --batch 32 --imgsz 640 --workers 8 --seed 123 "
    "--mix-spec w2.0 --lr-scale 3.5 --warmup-epochs 1.5 --weight-decay 0.000269 "
    "--focal-beta 2.75 --offset-weight 0.4 --mosaic 0.15 --scale 0.1 --translate 0.05 "
    "--clip-max-norm 5.65 --freeze-bn 0 --backbone-lr-scale 1.0 --freeze-backbone-epochs 0 "
    "--no-p2 1 --head-level p3 --act silu --head-conv 64 --late-upsample 1 "
    "--unfreeze-b0b4 0 --head-src 'p3' --p2-out-scale 1.0 --train-on-native-grid 0 "
    "--min-radius 1 --size-weight-small 5.0 --size-weight-large 1.0 --size-thr-px 16.0 "
    "--size-max-side-px 4.0 --lr-floor-ratio 0.1 --hf-skip 0 --hf-skip-src b1 "
    "--hf-skip-channels 1 --reference-frame /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json "
    "--skip-gate > /mnt/data/siping/runs/npu20ms_offcal/logs/NANO_1.log 2>&1"
)

BASE_RECIPE = {
    "init": "/INIT", "epochs": 5, "sched": 5, "mix": "w2.0", "lr": 3.5, "warmup": 1.5,
    "wd": 0.000269, "fb": 2.75, "ow": 0.4, "mosaic": 0.15, "scale": 0.1, "translate": 0.05,
    "clip": 5.65, "backbone_lr": 1.0, "uf": 0, "head_src": "p3", "native_grid": 0, "mr": 1,
    "sw": 5.0, "swl": 1.0, "thr": 16.0, "maxside": 4.0, "lr_floor": 0.1,
    "hf": 0, "hfsrc": "b1", "hfch": 1,
}

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="", help="台账路径（可选，用台账查抛光臂）")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    sys.path.insert(0, REPO)
    import nano_search as ns  # noqa: PLC0415

    # ① 默认配方命令串逐字节不变
    cmd0 = ns.train_cmd(1, 0, 123, dict(BASE_RECIPE))
    check("默认命令串与旧版逐字节相同", cmd0 == GOLDEN_OLD_CMD,
          "" if cmd0 == GOLDEN_OLD_CMD else f"\n  new={cmd0}\n  old={GOLDEN_OLD_CMD}")

    # ② 抛光臂：三个新旋钮都在
    warm = dict(BASE_RECIPE, anchor_l2=1e-3, freeze_bn=1, save_fracs="0.5")
    cmd1 = ns.train_cmd(2, 1, 321, warm)
    check("抛光臂含 --anchor-l2", "--anchor-l2 0.001" in cmd1, cmd1[-260:])
    check("抛光臂含 --anchor-ckpt", "--anchor-ckpt /INIT" in cmd1, cmd1[-260:])
    check("抛光臂含 --save-fracs", "--save-fracs 0.5" in cmd1, cmd1[-260:])
    check("抛光臂 --freeze-bn 1", "--freeze-bn 1" in cmd1, cmd1[-260:])
    check("新旋钮都在重定向之前", cmd1.index("--skip-gate") < cmd1.index("--anchor-l2")
          and cmd1.rindex(">") > cmd1.index("--save-fracs"), cmd1[-260:])

    # ③ 训练脚本新参数默认全关
    wpath = os.path.join(REPO, "manu", "training", "optuna_mix_ratio_search.py")
    with open(wpath, encoding="utf-8") as fh:
        wsrc = fh.read()
    check("训练脚本含 --save-fracs", '"--save-fracs"' in wsrc)
    check("训练脚本含 --anchor-ckpt", '"--anchor-ckpt"' in wsrc)
    check("训练脚本含 --anchor-l2", '"--anchor-l2"' in wsrc)
    check("训练脚本默认关 save-fracs", '"--save-fracs", type=str, default=""' in wsrc)
    check("训练脚本默认关 anchor-l2", '"--anchor-l2", type=float, default=0.0' in wsrc)

    # ④ 训练脚本能编译 + --help 正常
    r = subprocess.run([sys.executable, "-m", "py_compile", wpath], capture_output=True, text=True)
    check("训练脚本 py_compile", r.returncode == 0, r.stderr[-300:])
    try:
        import optuna  # noqa: F401, PLC0415
        _have_optuna = True
    except Exception:  # noqa: BLE001
        _have_optuna = False
    r2 = subprocess.run([sys.executable, wpath, "--help"], capture_output=True, text=True)
    if _have_optuna:
        check("训练脚本 --help 正常", r2.returncode == 0, r2.stderr[-300:])
    else:
        print("[SKIP] 训练脚本 --help（本机无 optuna，服务器上有）", flush=True)

    # ⑤ 可选：用台账查抛光臂（需要服务器台账）
    if args.ledger and os.path.isfile(args.ledger):
        import adaptive_runner as ar  # noqa: PLC0415
        led = json.load(open(args.ledger, encoding="utf-8"))
        base = ar.base_ref(led)
        anchors = ar.warm_anchor_candidates(led, base, k=5)
        check("抛光锚非空", len(anchors) > 0, str([(a["trial"], a["epoch"], a["npass"]) for a in anchors]))
        nxt = [999000]
        arms = ar.build_warm_arms(led, base, 99, nxt, max_arms=8, seed_base=900000)
        check("抛光臂非空", len(arms) > 0, f"{len(arms)} 臂")
        for t, s, p, d in arms[:3]:
            check(f"抛光臂 {t} 的 init 是锚 ckpt", str(p.get("init", "")).endswith(".pt"), str(p.get("init")))
            check(f"抛光臂 {t} 的 epoch ≤ 3", int(p.get("epochs", 99)) <= 3, str(p.get("epochs")))
            cmd = ns.train_cmd(t, 0, s, p)
            check(f"抛光臂 {t} 命令串含 anchor-l2 或不含(λ=0)",
                  ("--anchor-l2" in cmd) == (float(p.get("anchor_l2") or 0) > 0), d)

    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}", flush=True)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
