#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**结构线跨种子 soup 的家族定义生成器** —— 2026-11-06。

为什么需要它
------------
`build_struct_soups.py` 的家族定义是**外置 JSON**（`--families-json`），
但本轮的成员是**运行时才知道**的：

* 第一轮（`mix_ratio_search_npu20ms_v2`）跑完 26 条臂 ⇒ 母配方臂的产物才存在；
* 第二轮（`mix_ratio_search_npu20ms_seed`）跑完 6 条种子臂 ⇒ 种子成员才存在。

⇒ 家族定义**不能在开跑前写死**，必须在收割时按「实际入队的臂」现生成。

配对规则（**逐字段相等，不靠 role 名猜**）
----------------------------------------
role 名里带了 `sc25 / sc50 / ep15 / mixw2.5` 等信息，手写正则极易配错
（实测：`p2_out_scale=1.0` 的母臂 role 是 `S_p2_fused_hc32_relu_late_lr3.5_ep20`
——**名字里根本没有 sc100**）。

⇒ 本生成器按 **params 字典逐字段相等**来配对：
「同结构 **且** 同配方（只差种子）」的臂才是同家族成员。
种子在计划里是**独立字段**（`arm["seed"]`），**不在 params 里** ⇒ params 相等
天然蕴含「配方相同、只差种子」，正是 soup 的前提。

家族名
------
`{head_src}_hc{head_conv}_{act}[_late]_sc{tag}`
（`tag`：0.25→025 / 0.5→050 / 1.0→100，与速度表命名习惯一致）

产物（`--out`）
----------------
```json
{"families": {
   "p2_fused_hc32_relu_late_sc025": {
     "recipe": "...",
     "epoch": 20,
     "nobn": false,
     "members": [
       {"role": "S_p2_fused_hc32_relu_late_sc25_lr3.5_ep20"},
       {"role": "SEED_hc32_late1_sc25_s3000", "root": "<R2>", "study": "<S2>"},
       {"role": "SEED_hc32_late1_sc25_s3001", "root": "<R2>", "study": "<S2>"}
     ]
   }
}, "_meta": {...}}
```

用法（服务器，conda `uav`）：
```bash
python manu/training/build_struct_families.py \
    --round1-root /mnt/data/siping/runs/npu20ms \
    --round1-study mix_ratio_search_npu20ms_v2 \
    --round1-plan  /mnt/data/siping/runs/npu20ms/npu20ms_plan.json \
    --round2-root /mnt/data/siping/runs/npu20ms_seed \
    --round2-study mix_ratio_search_npu20ms_seed \
    --round2-plan  /mnt/data/siping/runs/npu20ms_seed/seed_arms.json \
    --out          /mnt/data/siping/runs/npu20ms/struct_families.json
```

退出码：0 = 全部家族都凑够 `--require-n` 个成员；1 = 有家族没凑够（明细打印）；
2 = 致命错误（计划文件缺失 / 解析失败 / 一个家族匹配到多个母臂）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 结构维度（决定「是不是同一个结构」）——与 build_struct_soups.py 的
# `_speed_key` / STRUCT_FIELDS 口径对齐，但**这里只用它做展示与命名**，
# 配对本身用的是**完整 params 相等**。
STRUCT_DIM = ("head_src", "head_conv", "act", "late_upsample", "p2_out_scale")


# ==============================================================================
# 工具
# ==============================================================================

# ★ 比较时**必须剔除**的键：
#   · `epochs` —— 入队时被弹出到 user_attrs（dispatcher 的 epochs 通道），
#     所以 study 里的 fixed_params **没有** epochs，而计划里的臂有 ⇒ 不剔除就配不上；
#   · `seed`  —— 种子本身是区分「同配方不同种子」的字段，但它**不在 params 里**。
_IGNORE_KEYS = ("epochs", "seed")


def _norm_params(params: dict) -> str:
    """把 params 归一成**可比较的稳定字符串**（布尔/数字统一，避免 1 vs 1.0 误判）。"""
    def _n(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:.6g}"
        if v is None:
            return "null"
        return str(v)

    return json.dumps({k: _n(params.get(k)) for k in sorted(params)
                        if k not in _IGNORE_KEYS},
                      ensure_ascii=False, sort_keys=True)


def _sc_tag(v) -> str:
    """p2_out_scale ⇒ 标签（0.25→025 / 0.5→050 / 1.0→100）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "x"
    if abs(f - 1.0) < 1e-9:
        return "100"
    if abs(f - 0.75) < 1e-9:
        return "075"
    if abs(f - 0.5) < 1e-9:
        return "050"
    if abs(f - 0.25) < 1e-9:
        return "025"
    return f"{f:g}".replace(".", "")


def _struct_tag(params: dict) -> str:
    """结构 ⇒ 家族名标签。"""
    src = str(params.get("head_src") or params.get("head_level") or "p1")
    hc = int(params.get("head_conv") or 64)
    act = str(params.get("act") or "silu")
    late = "late" if params.get("late_upsample") else "nolate"
    return f"{src}_hc{hc}_{act}_{late}_sc{_sc_tag(params.get('p2_out_scale', 1.0))}"


def _recipe_text(params: dict) -> str:
    """配方一行摘要（写进 families 便于审计）。"""
    keys = ("mix_spec", "lr_scale", "epochs", "unfreeze_b0b4", "freeze_bn",
            "warmup_epochs", "weight_decay", "focal_beta", "offset_weight",
            "mosaic", "scale", "translate", "clip_max_norm")
    return "·".join(f"{k}={params.get(k)}" for k in keys if k in params)


def _load_plan(path: Path, tag: str) -> list[dict]:
    """读计划文件 ⇒ [{role, seed, params}]（顶层是 list；也兼容 {"arms": [...]}）。"""
    if not path.is_file():
        raise SystemExit(f"[FATAL] {tag} 计划文件不存在：{path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[FATAL] {tag} 计划文件解析失败：{path}：{exc}")
    arms = doc if isinstance(doc, list) else (doc.get("arms") or doc.get("plans") or [])
    out = []
    for a in arms:
        role = str(a.get("role") or "").strip()
        if not role:
            continue
        out.append({"role": role, "seed": a.get("seed"),
                    "params": dict(a.get("params") or {})})
    if not out:
        raise SystemExit(f"[FATAL] {tag} 计划文件里没有可用臂：{path}")
    return out


# ==============================================================================
# 从 study 取「真实存在」的臂（★ 2026-11-06 加）
# ==============================================================================

def load_study_arms(root: str, study_name: str) -> list[dict]:
    """从 optuna study 取所有带 role 的臂 ⇒ [{role, seed, params, trial_number, trial_dir}]。

    **为什么不能只靠计划文件**：
    * 计划文件是**静态**的，不反映「哪条臂真的跑了、跑完没有」；
    * 计划下标 ⇒ trial 编号是「FIFO 入队」推出来的**脆弱假设**
      （一旦有臂被剪掉/被追加，这个映射就错位）；
    * 同一 params 可能对应**多条臂**（补投/复现臂）⇒ 必须靠 study + 磁盘产物来消歧。

    ⇒ study 是「哪些臂真的存在」的**唯一真源**。
    """
    db = Path(root) / "study.db"
    if not db.is_file():
        print(f"[WARN] {root}/study.db 不存在 ⇒ 无法从 study 取臂")
        return []
    try:
        import optuna
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 导入 optuna 失败（{exc}）⇒ 无法从 study 取臂")
        return []
    try:
        st = optuna.load_study(storage="sqlite:///" + str(db), study_name=study_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 加载 study {study_name} 失败：{exc}")
        return []

    out = []
    for t in st.trials:
        role = str((t.user_attrs or {}).get("role") or "").strip()
        if not role:
            continue
        pr = dict(getattr(t, "params", None) or {})
        if not pr:
            # WAITING 臂的 params 还没落到 .params ⇒ 在 system_attrs["fixed_params"]
            pr = dict((getattr(t, "system_attrs", None) or {}).get("fixed_params") or {})
        if not pr:
            continue
        out.append({
            "role": role,
            "seed": (t.user_attrs or {}).get("seed"),
            "params": pr,
            "trial_number": int(t.number),
            "trial_dir": str(Path(root) / "trials" / f"trial_{int(t.number):04d}"),
            "state": t.state.name,
        })
    print(f"[OK] study {study_name}：{len(out)} 条带 role 的臂 <- {db}")
    return out


def _has_products(trial_dir: str, epoch: int) -> bool:
    """这条臂的产物齐不齐（epoch 权重 + 门禁报告）⇒ 能当 soup 成员吗。"""
    if not trial_dir:
        return False
    d = Path(trial_dir)
    if not (d / "weights" / f"epoch{int(epoch)}.pt").is_file():
        return False
    return bool(list(d.glob("gate*/epoch*/gate_report.json")))


# ==============================================================================
# 主流程
# ==============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="结构线跨种子 soup 的家族定义生成器")
    ap.add_argument("--round1-root", default="/mnt/data/siping/runs/npu20ms")
    ap.add_argument("--round1-study", default="mix_ratio_search_npu20ms_v2")
    ap.add_argument("--round1-plan", default="")
    ap.add_argument("--round2-root", default="/mnt/data/siping/runs/npu20ms_seed")
    ap.add_argument("--round2-study", default="mix_ratio_search_npu20ms_seed")
    ap.add_argument("--round2-plan", default="")
    ap.add_argument("--out", default="/mnt/data/siping/runs/npu20ms/struct_families.json")
    ap.add_argument("--require-n", type=int, default=3,
                    help="每个家族至少要有几个成员（3 = 同配方 3 粒子，对齐 cs_t0043 形态）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    r1_plan = Path(args.round1_plan or (Path(args.round1_root) / "npu20ms_plan.json"))
    r2_plan = Path(args.round2_plan or (Path(args.round2_root) / "seed_arms.json"))
    if not r2_plan.is_file() and args.round2_plan:
        raise SystemExit(f"[FATAL] 第二轮计划文件不存在：{r2_plan}")
    if not r2_plan.is_file():
        # 接力脚本会把种子臂计划复制回第一轮目录 ⇒ 兜底
        alt = Path(args.round1_root) / "seed_arms.json"
        if alt.is_file():
            r2_plan = alt
            print(f"[WARN] {Path(args.round2_root) / 'seed_arms.json'} 不存在 ⇒ 兜底用 {alt}")

    arms2 = _load_plan(r2_plan, "第二轮")
    # ★ 第一轮：**优先从 study 取**（唯一真源 = 哪些臂真的存在、跑完没有）。
    #   计划文件只是「当初打算跑什么」，不含补投/复现臂，也映射不到 trial 号。
    arms1 = load_study_arms(args.round1_root, args.round1_study)
    if not arms1:
        arms1 = _load_plan(r1_plan, "第一轮")
        print("[WARN] study 取不到第一轮臂 ⇒ 回落到计划文件（消歧能力下降）")
    print(f"[OK] 第一轮 {len(arms1)} 条臂（来源：{'study' if arms1 else 'plan'}）")
    print(f"[OK] 第二轮 {len(arms2)} 条臂 <- {r2_plan}")

    # ---- 第一轮按 params 归组（母臂候选）----
    by_params: dict[str, list[dict]] = {}
    for a in arms1:
        by_params.setdefault(_norm_params(a["params"]), []).append(a)

    # ---- 第二轮按 params 归组（种子成员）----
    seeds_by_params: dict[str, list[dict]] = {}
    for a in arms2:
        seeds_by_params.setdefault(_norm_params(a["params"]), []).append(a)

    families: dict[str, dict] = {}
    problems: list[str] = []
    n_ok = 0

    for key, seeds in sorted(seeds_by_params.items(), key=lambda kv: kv[1][0]["role"]):
        mothers = by_params.get(key, [])
        tag = _struct_tag(seeds[0]["params"])
        recipe = _recipe_text(seeds[0]["params"])

        # ★★ 2026-11-06：**只留产物齐全的成员**（权重 + 门禁报告）
        #   —— 否则 n 会被「还在训练 / 跑挂了」的臂撑大 ⇒ soup 成员不齐 ⇒ 建不出 soup。
        #   ⚠️ **没有 trial_dir 的臂（study 取不到、只能回落计划文件）不过滤**
        #   —— 那条路径本来就查不了产物 ⇒ 保持旧行为（宁可多算 n，也别把家族整个丢掉）。
        _ep0 = int(seeds[0]["params"].get("epochs") or 0)
        _ready_seeds = [s for s in seeds
                        if not s.get("trial_dir")
                        or _has_products(s.get("trial_dir", ""), _ep0)]
        if not _ready_seeds:
            problems.append(
                f"{tag}：第二轮 {len(seeds)} 条种子臂**没有一条产物齐全**"
                f"（要 epoch{_ep0}.pt + 门禁报告）⇒ 跳过")
            continue
        seeds = _ready_seeds

        # ★★ 2026-11-06：**允许「只有第二轮成员」的家族**。
        #   背景：第一轮 dispatcher 的 `--max-hours 34` 天花板会把队尾的几条臂截掉
        #   （含 sc25 家族的替补母臂 trial_0037）⇒ 有些结构**第一轮没有母臂**。
        #   而「同配方 + 不同种子」的成员集合**本身**就是合法 soup 家族
        #   （家族 = params 逐字段相等的一组臂），母臂只是「n 的来源」，不是必要条件。
        if not mothers:
            if len(seeds) < args.require_n:
                problems.append(
                    f"{tag}：只有第二轮成员 {len(seeds)} 条（< {args.require_n}）"
                    f"（role={[s['role'] for s in seeds]}）⇒ 跳过")
                continue
            mothers = []
        if len(mothers) > 1:
            # ★★ 消歧（2026-11-06）：同 params 可能有多条臂
            #   （补投臂 / 被杀后重投的臂 / 同配方复现臂）
            #   ⇒ 只有**产物齐全**的那条才是真母亲臂（有权重 + 有门禁报告）。
            #   都齐 ⇒ 取 trial 号最大的（最新的）；都不齐 ⇒ 明确跳过
            #   （宁可没有这个家族，也不要一个半成品混进 soup）。
            _ep = int(seeds[0]["params"].get("epochs") or 0)
            _ready = [m for m in mothers if _has_products(m.get("trial_dir", ""), _ep)]
            if not _ready:
                problems.append(
                    f"{tag}：同配方匹配到 {len(mothers)} 条第一轮臂"
                    f"（{[m['role'] for m in mothers]}），但**没有一条产物齐全**"
                    f"（要 epoch{_ep}.pt + 门禁报告）⇒ 跳过")
                continue
            mothers = [max(_ready, key=lambda m: m.get("trial_number") or -1)]

        if mothers:
            mother = mothers[0]
            members = [{"role": mother["role"]}]
            for s in sorted(seeds, key=lambda x: (x["seed"] is None, x["seed"])):
                members.append({"role": s["role"],
                                "root": str(args.round2_root),
                                "study": str(args.round2_study)})
        else:
            # ★ 只有第二轮成员（第一轮母臂被天花板截掉）⇒ 家族成员全是种子臂
            mother = None
            members = [{"role": s["role"],
                        "root": str(args.round2_root),
                        "study": str(args.round2_study)}
                       for s in sorted(seeds, key=lambda x: (x["seed"] is None, x["seed"]))]

        n = len(members)
        if n < args.require_n:
            problems.append(
                f"{tag}：只凑到 {n} 个成员（要求 ≥ {args.require_n}）⇒ 跳过")
            continue

        families[tag] = {
            "recipe": recipe,
            "epoch": int(seeds[0]["params"].get("epochs") or 0),
            "nobn": False,
            "members": members,
            "seeds": [s.get("seed") for s in seeds],
            "structure": {k: seeds[0]["params"].get(k) for k in STRUCT_DIM},
        }
        n_ok += 1
        print(f"[FAMILY] {tag}: n={n} | 母臂={mother['role'] if mother else '(无·纯第二轮)'} "
              f"| 种子={[s['role'] for s in seeds]} | epoch={families[tag]['epoch']}")

    # ---- 自检 ----
    print(f"[SUMMARY] 家族 {n_ok} 个 / 需要 ≥ {args.require_n} 成员")
    if problems:
        print("[PROBLEM] 以下家族没凑齐（不会被建成 soup）：")
        for p in problems:
            print(f"  · {p}")

    doc = {
        "families": families,
        "_meta": {
            "generated_by": "build_struct_families.py",
            "round1_root": str(args.round1_root), "round1_study": str(args.round1_study),
            "round2_root": str(args.round2_root), "round2_study": str(args.round2_study),
            "require_n": int(args.require_n),
            "pairing_rule": "params 逐字段相等（种子在 arm['seed']，不在 params 里）",
            "n_families": n_ok,
            "problems": problems,
        },
    }

    if n_ok == 0:
        print("[FATAL] 一个家族都没凑出来 ⇒ 无法建 soup")
        return 1
    if problems:
        print("[WARN] 有家族没凑齐（其余家族照常建）")

    if args.dry_run:
        print("[DRY-RUN] 不写文件")
        return 0

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 家族定义已写 {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
