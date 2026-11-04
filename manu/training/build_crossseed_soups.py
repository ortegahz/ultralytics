#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建**跨种子 soup**（同一配方、不同种子的末 epoch 权重平均）—— 2026-11-02。

为什么
----
变体 C（跨**配置** soup）判负：trial_0009 / trial_0025 / trial_0018 的 T1/T3 画像差很多
⇒ 平均只是两端插值。
**跨种子 soup** 是另一回事：**同一配方、同一起点、只换随机种子**
⇒ 三个模型在同一盆地里的不同位置 ⇒ 平均 = 降方差、且通常**同时**提升 T1 与 T3
（文献常报 +0.5~1.0 pp F1）。

它正好打在当前死结上：最近几个配置的 T1 缺口只有 0.00015~0.00123，
而种子噪声是 0.0025 ⇒ 单种子结论不可信 ⇒ **把同配方的几个种子平均**，
既降方差又保留 T3 ⇒ 是当前最可能命中「T3 ≥ 0.87 且 T1 过线」的零训练成本手段。

要平均哪些
----------
三个离双目标最近的配方（每个都有 ≥2 个种子）：
| 家族 | 周末 | 本轮 focus |
|:--|:--|:--|
| trial_0008 配方 | trial_0008 | FOCUS_T0008_REPLAY / _SEED2 / _SEED3 |
| trial_0077 配方 | trial_0077 | FOCUS_T0077_SEED2 / _SEED3 |
| trial_0082 配方 | trial_0082 | FOCUS_T0082_SEED2 / _SEED3 |

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/build_crossseed_soups.py \
  --focus-root /mnt/data/siping/runs/nogmc_focus \
  --weekend-root /mnt/data/siping/runs/nogmc_weekend \
  --out-dir /mnt/data/siping/runs/nogmc_focus/swa
```

家族定义外置（`--families-json`，2026-11-03 新增）
---------------------------------------------------
上面那张表是**硬编码**在 `FAMILIES` 里的（周末给 trial 目录名、focus 给 role）。
新的 no-GMC T3 study 要**跨 3 个 study** 取成员（weekend / focus / t3），
硬编码表达不了 ⇒ 家族定义可以外置成 JSON：

```json
{"families": {
  "t0033": {"recipe": "w3.0·lr3.0·8ep", "nobn": false, "members": [
    {"root": "/mnt/data/siping/runs/nogmc_focus", "study": "mix_ratio_search_nogmc_focus",
     "role": "FOCUS_T0008_LR30"},
    {"root": "/mnt/data/siping/runs/nogmc_t3", "study": "mix_ratio_search_nogmc_t3",
     "role": "INS_T0033_S4"}
  ]}
}}
```

成员两种写法（**周末的是 trial 目录名、focus/t3 的是 role ⇒ 必须能区分**）：
* `{"root":…, "study":…, "role": "FOCUS_T0008_LR30"}` —— 按 `user_attrs["role"]` 找 trial；
* `{"root":…, "study":…, "trial": "trial_0008"}` —— 直接给 trial 目录名（trial_0008 = 编号 8）。

⚠️ **不给 `--families-json` 时行为与旧版逐位不变**（仍走内置 `FAMILIES`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIES = {
    "t0008": {"weekend": ["trial_0008"],
              "focus": ["FOCUS_T0008_REPLAY", "FOCUS_T0008_SEED2", "FOCUS_T0008_SEED3"]},
    "t0077": {"weekend": ["trial_0077"],
              "focus": ["FOCUS_T0077_SEED2", "FOCUS_T0077_SEED3"]},
    "t0082": {"weekend": ["trial_0082"],
              "focus": ["FOCUS_T0082_SEED2", "FOCUS_T0082_SEED3"]},
}
NOBN = {"t0008": False, "t0077": False, "t0082": False}   # 同配方 ⇒ p0 与 BN 都同源 ⇒ 用普通版


def last_epoch_weights(trial_dir: Path):
    wdir = trial_dir / "weights"
    if not wdir.is_dir():
        return None
    fs = sorted(wdir.glob("epoch*.pt"), key=lambda p: int(p.stem.replace("epoch", "")))
    return fs[-1] if fs else None


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 家族定义外置（--families-json）
# ==============================================================================
_STUDY_CACHE: dict = {}


def _load_study(root: Path, study_name: str):
    """按 (root, study_name) 缓存加载 Optuna study；不可用时返回 None。"""
    key = (str(root), study_name)
    if key in _STUDY_CACHE:
        return _STUDY_CACHE[key]
    db = Path(root) / "study.db"
    if not db.is_file():
        print(f"    [WARN] {root}/study.db 不存在 ⇒ 该 study 的成员全部跳过")
        _STUDY_CACHE[key] = None
        return None
    try:
        st = optuna.load_study(storage="sqlite:///" + str(db), study_name=study_name)
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] 加载 study {study_name}（{db}）失败：{exc}")
        _STUDY_CACHE[key] = None
        return None
    _STUDY_CACHE[key] = st
    return st


def _resolve_member(member: dict) -> tuple:
    """解析一条成员定义。

    返回 ``(weight_path | None, info_dict)``；``weight_path is None`` 时
    ``info_dict["skip"]`` 里写着跳过原因。
    """
    root = str(member.get("root") or "")
    study_name = str(member.get("study") or "")
    role = str(member.get("role") or "")
    trial_name = str(member.get("trial") or "")
    info = {"root": root, "study": study_name, "role": role, "trial": trial_name}

    if not root or not study_name:
        info["skip"] = "成员定义缺 root/study"
        return None, info
    if not role and not trial_name:
        info["skip"] = "成员定义既没 role 也没 trial"
        return None, info

    st = _load_study(Path(root), study_name)
    if st is None:
        info["skip"] = f"study {study_name} 不可用（study.db 缺失或加载失败）"
        return None, info
    by_num = {t.number: t for t in st.trials}

    num = None
    if role:
        hits = [t.number for t in st.trials if (t.user_attrs or {}).get("role") == role]
        if not hits:
            info["skip"] = f"role {role} 在 study {study_name} 里还没入队"
            return None, info
        # 同一 role 可能有多条（重复入队）⇒ 优先取「可用」的那条：
        #   COMPLETE 优先；否则取「PRUNED 但产物完整」的（门禁 FAIL 但训练成功）
        done = [n for n in hits if by_num[n].state.name == "COMPLETE"]
        pruned = [n for n in hits if by_num[n].state.name == "PRUNED"
                  and (Path(root) / "trials" / f"trial_{n:04d}" / "trial_result.json").is_file()]
        num = (done or pruned or hits)[-1]
        if len(hits) > 1:
            info["dup_roles"] = hits
    else:
        # trial 目录名（形如 trial_0008）⇒ 取编号
        try:
            num = int(str(trial_name).replace("trial_", ""))
        except ValueError:
            info["skip"] = f"trial 名 {trial_name} 解析不出编号"
            return None, info

    trial = by_num.get(num)
    if trial is None:
        info["skip"] = f"trial_{num:04d} 不在 study {study_name} 里"
        return None, info
    # ⚠️ **必须用 state.name（字符串），不能用 state.value（数值）**：
    #    Optuna 的 TrialState 枚举值是 RUNNING=0 / COMPLETE=1 / PRUNED=2 / FAIL=3 / WAITING=4
    #    ⇒ 存/比 value 会让 `state != "COMPLETE"` 恒为真
    #    ⇒ **所有已完成的臂都被当成"未完成"跳过**（2026-11-03 06:33 实测踩到）。
    state = trial.state.name
    info["trial_number"] = num
    info["state"] = state
    tdir = Path(root) / "trials" / f"trial_{num:04d}"
    p, why = _usable_member(trial, tdir)
    if p is None:
        info["skip"] = why
        return None, info
    if why:
        info["note"] = why
    info["weights"] = str(p)
    return p, info


def _usable_member(trial, tdir: Path):
    """判定一个 trial 能否作为 soup 粒子 ⇒ 返回 (权重路径|None, 说明)。

    ⚠️ **2026-11-03 18:25 修正：不能用 `state == "COMPLETE"` 当唯一判据。**
    本轮调度器把「训练跑完、但门禁判定 FAIL」的臂标成 **PRUNED**
    （Optuna 语义里 PRUNED = 被剪枝/未完成）⇒ 若只收 COMPLETE，
    **今晚跑出来的每一条臂都会被判成"未完成"而跳过**
    ⇒ 判断文档点名的「跨种子 Soup」在今晚**完全落空**
    （实测：w2lr30_15 / w3lr30_15 两个今晚主攻家族可用粒子数 = **0**）。

    ⇒ 改用**产物判据**（与 Optuna state 解耦）：
      1. state 不能是 RUNNING / WAITING / FAIL（正在训练 ⇒ 权重是半成品）
      2. 权重存在，且**最大 epoch 号 == 计划的 epochs**（末 epoch 完整）
      3. `trial_result.json` 存在（门禁真的跑完了、有完整指标）
    ⇒ 手动剪掉的 WAITING 臂（压根没跑、无权重）仍被正确排除；
      被 hard_cut 砍在半路的臂（epoch 数不足）也被正确排除。
    """
    state = trial.state.name
    if state in ("RUNNING", "WAITING", "FAIL"):
        return None, f"状态={state} ⇒ **未完成，跳过**（半成品权重不能进 soup）"
    p = last_epoch_weights(tdir)
    if p is None:
        return None, f"{tdir} 里没有 epoch*.pt 权重"
    if state == "PRUNED":
        plan_ep = trial.params.get("epochs")
        got_ep = int(p.stem.replace("epoch", ""))
        if plan_ep and got_ep != int(plan_ep):
            return None, f"状态=PRUNED 且只跑到 epoch{got_ep}/{int(plan_ep)} ⇒ 半成品，跳过"
        if not (tdir / "trial_result.json").is_file():
            return None, f"{tdir} 没有 trial_result.json ⇒ 门禁未跑完，跳过"
        return p, "PRUNED=训练完成但门禁 FAIL ⇒ 按产物判据收录"
    return p, ""


def run_from_json(args) -> int:
    """按外置 JSON 家族定义构建跨 study 的跨种子 soup。"""
    spec_path = Path(args.families_json)
    if not spec_path.is_file():
        print(f"[FATAL] 找不到家族定义文件 {spec_path}")
        return 2
    try:
        doc = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] 家族定义 JSON 解析失败：{exc}")
        return 2
    fams = doc.get("families") or {}
    if not fams:
        print(f"[FATAL] {spec_path} 里没有 families")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CROSSSEED] 家族定义来自 {spec_path}：{len(fams)} 族 "
          f"（{', '.join(fams)}）")

    details = {}
    made = []
    for fam, fs in fams.items():
        if not isinstance(fs, dict):
            print(f"  · {fam}: 家族定义不是对象 ⇒ 跳过")
            continue
        members = fs.get("members") or []
        nobn = bool(fs.get("nobn", False))
        recipe = str(fs.get("recipe", ""))
        paths, descs = [], []
        for m in members:
            if not isinstance(m, dict):
                print(f"  · {fam}: 成员定义不是对象 ⇒ 跳过")
                continue
            w, info = _resolve_member(m)
            if w is None:
                print(f"  · {fam}: {info.get('skip')}（{info.get('role') or info.get('trial')}）")
                continue
            paths.append(w)
            descs.append(info)
            print(f"  · {fam}: 成员 {info.get('role') or info.get('trial')} "
                  f"(t{info['trial_number']:04d}@{info['study']}) ⇒ {w.name}")

        det = {"recipe": recipe, "nobn": nobn, "members": descs,
               "n_usable": len(paths), "out": None, "md5": None}
        if len(paths) < 2:
            print(f"[CROSSSEED] {fam}: 可用种子只有 {len(paths)} 个 ⇒ 不够平均，跳过")
            details[fam] = det
            continue

        out = out_dir / f"cs_{fam}.pt"
        det["out"] = str(out)
        det["sources"] = [str(p) for p in paths]
        if args.dry_run:
            print(f"[CROSSSEED] {fam}: [DRY-RUN] 平均 {len(paths)} 个种子 → {out.name}（不落盘）")
            det["md5"] = None
            details[fam] = det
            continue

        cmd = [sys.executable, str(Path(args.repo) / "manu/training/fuse_weights.py"),
               "--inputs"] + [str(p) for p in paths] + ["--out", str(out),
                                                      "--name", f"cs_{fam}"]
        if nobn:
            cmd.append("--nobn")
        print(f"[CROSSSEED] {fam}: 平均 {len(paths)} 个种子 → {out.name}")
        for p in paths:
            print(f"     · {p}")
        subprocess.run(cmd, check=True, cwd=args.repo)
        det["md5"] = _md5(out)
        made.append(out.name)
        details[fam] = det

    if not args.dry_run:
        (out_dir / "crossseed_manifest.json").write_text(json.dumps(
            {"families": fams, "made": made,
             "note": "跨种子 soup（同配方不同种子·末 epoch 平均）· 家族定义外置",
             "source_file": str(spec_path),
             "details": details},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print("[CROSSSEED] [DRY-RUN] 解析结果：")
        print(json.dumps({"families": fams, "details": details},
                         ensure_ascii=False, indent=2))
    print(f"[CROSSSEED] 完成：{made or '无'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus-root", default="/mnt/data/siping/runs/nogmc_focus")
    ap.add_argument("--weekend-root", default="/mnt/data/siping/runs/nogmc_weekend")
    ap.add_argument("--out-dir", default="/mnt/data/siping/runs/nogmc_focus/swa")
    ap.add_argument("--study-name", default="mix_ratio_search_nogmc_focus")
    ap.add_argument("--repo", default="/tmp/pycharm_project_10ae9e2e")
    ap.add_argument("--families-json", default="",
                    help="家族定义外置 JSON（可选）。给了就用它，"
                         "**不给则沿用内置 FAMILIES（旧行为逐位不变）**")
    ap.add_argument("--dry-run", action="store_true",
                    help="只解析成员并打印，**不写 .pt**（也不写 manifest）")
    args = ap.parse_args()

    if args.families_json:
        return run_from_json(args)

    focus_root = Path(args.focus_root)
    weekend_root = Path(args.weekend_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 本轮 study：按 role 找 trial 号
    # ⚠️ **只用 state==COMPLETE 的 trial**（2026-11-02 实测踩到）：
    #    **正在训练**的 trial，其 weights/ 目录里已经有 epoch1..epochN 的**半成品**权重
    #    （实测 t0015 在训练中就有 epoch1-3）⇒ 若取"最后一个 epoch"会把**半训练模型**
    #    混进 soup ⇒ 门禁量的是一个从未存在过的模型 ⇒ 结论无效。
    st = optuna.load_study(storage="sqlite:///" + str(focus_root / "study.db"),
                            study_name=args.study_name)
    role2trial = {}
    for t in st.trials:
        r = (t.user_attrs or {}).get("role")
        if r:
            # ⚠️ **必须存 state.name（字符串），不能存 state.value（数值）**：
            #    Optuna 的 TrialState 枚举值是 RUNNING=0 / COMPLETE=1 / PRUNED=2 / FAIL=3 / WAITING=4
            #    ⇒ 存 value 会让下面的 `state != "COMPLETE"` **恒为真**
            #    ⇒ **所有已完成的臂都被当成"未完成"跳过** ⇒ 每族只剩 1 个（weekend）种子
            #    ⇒ 一个 soup 都建不出来（2026-11-03 06:33 实测踩到：构建输出
            #    「状态=1 ⇒ 未完成，跳过」× 8 ⇒ 「完成：无」）。
            role2trial[r] = (t.number, t.state.name)
    by_num = {t.number: t for t in st.trials}
    print(f"[CROSSSEED] 本轮 study 里带 role 的 trial：{len(role2trial)}")

    made = []
    for fam, spec in FAMILIES.items():
        paths = []
        for w in spec["weekend"]:
            p = last_epoch_weights(weekend_root / "trials" / w)
            if p:
                paths.append(p)
        for role in spec["focus"]:
            hit = role2trial.get(role)
            if hit is None:
                print(f"  · {fam}: role {role} 还没入队 ⇒ 跳过")
                continue
            num, state = hit
            p, why = _usable_member(by_num[num], focus_root / "trials" / f"trial_{num:04d}")
            if p is None:
                print(f"  · {fam}: role {role} (t{num:04d}) {why}")
                continue
            if why:
                print(f"  · {fam}: role {role} (t{num:04d}) {why}")
            paths.append(p)
        if len(paths) < 2:
            print(f"[CROSSSEED] {fam}: 可用种子只有 {len(paths)} 个 ⇒ 不够平均，跳过")
            continue
        out = out_dir / f"cs_{fam}.pt"
        cmd = [sys.executable, str(Path(args.repo) / "manu/training/fuse_weights.py"),
               "--inputs"] + [str(p) for p in paths] + ["--out", str(out),
                                                      "--name", f"cs_{fam}"]
        if NOBN.get(fam):
            cmd.append("--nobn")
        print(f"[CROSSSEED] {fam}: 平均 {len(paths)} 个种子 → {out.name}")
        for p in paths:
            print(f"     · {p}")
        subprocess.run(cmd, check=True, cwd=args.repo)
        made.append(out.name)

    (out_dir / "crossseed_manifest.json").write_text(json.dumps(
        {"families": {k: v for k, v in FAMILIES.items()}, "made": made,
         "note": "跨种子 soup（同配方不同种子·末 epoch 平均）"},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[CROSSSEED] 完成：{made or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
