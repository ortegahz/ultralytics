#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**结构线专用**的 Model Soup 构建器（同配方多种子 ⇒ 跨种子权重平均）—— 2026-11-06。

为什么不能直接复用 `build_crossseed_soups.py`
--------------------------------------------
1. **本轮 ckpt 带「结构自描述」**（``no_p2 / head_level / act / head_conv /
   late_upsample / head_src / p2_out_scale / stride``）⇒ **只有结构完全相同的臂
   才能平均**：结构不同 ⇒ 张量形状不同 ⇒ 平均出来的模型权重对不上 ⇒ 门禁/导出
   重建出**另一张图**（本轮已经踩过：hc32 被当成 hc64 ⇒ 热图全 0 ⇒ 误判「死模型」）。
   ⇒ 本构建器**逐字段校验**成员的结构自描述，**不一致直接报错并指出是哪几个字段**。
2. **soup ckpt 必须保留结构自描述**：``fuse_weights.py`` 的输出**不抄**这些字段
   （它只抄 ``stride / imgsz / p0_kwargs``）⇒ 直接拿去门禁会重建出错的图。
   ⇒ 这里在平均之后**补齐**（从成员 ckpt 抄一份，见 `STRUCT_FIELDS`）。
3. 家族定义沿用 ``build_crossseed_soups.py --families-json`` 的格式（外置 JSON）。

⚠️ **权重平均逻辑不重写**：直接调用 ``fuse_weights.fuse()``（含 p0 取第一个模型、
``--nobn`` 时 BN 统计量取第一个模型等语义**逐位沿用**）。

家族定义（``--families-json``）
-------------------------------
```json
{"families": {
  "p2f_hc32_late_sc25": {
    "recipe": "w2.0·lr3.5·ep20·seed{2102,2103}",
    "epoch": 20,
    "nobn": false,
    "members": [
      {"role": "S_p2_fused_hc32_relu_late_sc25_lr3.5_ep20_s2102"},
      {"role": "S_p2_fused_hc32_relu_late_sc25_lr3.5_ep20_s2103"},
      {"trial": "trial_0007"}
    ]
  }
}}
```
成员两种写法（与 `build_crossseed_soups.py` 一致）：
* ``{"role": "XXX"}`` —— 按 ``user_attrs["role"]`` 找 trial（需要 optuna + study.db）；
* ``{"trial": "trial_0008"}``（或 ``{"trial": 8}``）—— 直接给 trial 目录名/编号。
成员可另外带 ``{"root": ..., "study": ...}`` 覆盖默认的 out-root / study-name。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/build_struct_soups.py \
  --out-root /mnt/data/siping/runs/npu20ms \
  --study-name mix_ratio_search_npu20ms_v2 \
  --families-json /mnt/data/siping/runs/npu20ms/struct_families.json \
  --out-dir  /mnt/data/siping/runs/npu20ms/soups
```
产物：
* ``<out-dir>/soup_<family>.pt`` —— 与训练产物同格式的 ckpt（``{"model": state_dict, ...}``）
  **且保留结构自描述**；
* ``<out-dir>/soup_candidates.json`` —— 候选清单（家族名 / 成员 trial / 每个成员的
  ckpt 路径与结构字段 / soup ckpt 路径 / soup 的速度）。

退出码：0 = 全部家族建成；1 = 有家族被跳过或结构不一致；2 = 致命错误。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

# ==============================================================================
# 结构自描述字段（与 optuna_mix_ratio_search.py:1243-1251 写进 ckpt 的字段一致）
# ==============================================================================
STRUCT_FIELDS = (
    "no_p2",          # bool
    "head_level",     # str
    "act",            # str
    "head_conv",      # int
    "late_upsample",  # bool
    "head_src",       # str
    "p2_out_scale",   # float
    "stride",         # int
)

# 速度表查表 key 的 5 个维度（与 optuna_mix_ratio_search.structure_key 一致）
SPEED_FIELDS = ("head_src", "head_conv", "act", "late_upsample", "p2_out_scale")

DEFAULT_OUT_ROOT = "/mnt/data/siping/runs/npu20ms"
DEFAULT_STUDY = "mix_ratio_search_npu20ms_v2"


# ==============================================================================
# 基础工具
# ==============================================================================
def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt(path: Path) -> dict:
    """读 ckpt（**惰性 import torch** ⇒ 宿主没 torch 也能 import 本模块）。"""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 老版 torch 没有 weights_only 参数
        return torch.load(path, map_location="cpu")


def save_ckpt(path: Path, ckpt: dict) -> None:
    """写 ckpt（惰性 import torch）。"""
    import torch

    torch.save(ckpt, path)


def _load_fuse_weights():
    """按文件路径加载 ``manu/training/fuse_weights``（不依赖包结构）。"""
    path = Path(__file__).resolve().parent / "fuse_weights.py"
    spec = importlib.util.spec_from_file_location("manu_fuse_weights", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"[FATAL] 无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fuse_states(paths: list[Path], nobn: bool) -> dict:
    """**权重平均** ⇒ 委托给 ``fuse_weights.fuse()``（不重写它的核心逻辑）。"""
    fw = _load_fuse_weights()
    return fw.fuse([str(p) for p in paths], bool(nobn))


# ==============================================================================
# 结构自描述：抽取 / 归一 / 比对
# ==============================================================================
def _norm(field: str, v):
    """把 ckpt 里的原始值归一成可比对的类型（None = 字段缺失）。"""
    if v is None:
        return None
    if field in ("no_p2", "late_upsample"):
        return bool(v)
    if field in ("head_level", "act", "head_src"):
        return str(v)
    if field in ("head_conv", "stride"):
        try:
            return int(v)
        except (TypeError, ValueError):
            return str(v)
    if field == "p2_out_scale":
        try:
            return round(float(v), 6)
        except (TypeError, ValueError):
            return str(v)
    return v


def struct_of(ckpt: dict) -> dict:
    """从 ckpt 抽出结构自描述（只含 ``STRUCT_FIELDS``）。"""
    return {f: _norm(f, (ckpt or {}).get(f)) for f in STRUCT_FIELDS}


def struct_diff(members: list[dict]) -> list[tuple[str, dict]]:
    """比对多个成员的结构自描述 ⇒ 返回 [(字段, {成员标签: 值})]（只含不一致的字段）。"""
    if len(members) < 2:
        return []
    out = []
    for f in STRUCT_FIELDS:
        vals = {m["label"]: m["struct"].get(f) for m in members}
        if len(set(vals.values())) > 1:
            out.append((f, vals))
    return out


def format_struct_diff(diff: list[tuple[str, dict]]) -> str:
    """把结构不一致渲染成可读的多行文本。"""
    lines = []
    for f, vals in diff:
        detail = " | ".join(f"{k}={v!r}" for k, v in vals.items())
        lines.append(f"    · {f}: {detail}")
    return "\n".join(lines)


# ==============================================================================
# 速度表（结构 → 板卡 ms）
# ==============================================================================
def _speed_key(spec: dict) -> tuple:
    """速度表 key（与 optuna_mix_ratio_search.structure_key 逐位一致）。"""
    return (
        str(spec.get("head_src") or "p1"),
        int(spec.get("head_conv") or 64),
        str(spec.get("act") or "relu"),
        bool(spec.get("late_upsample", False)),
        round(float(spec.get("p2_out_scale", 1.0) or 1.0), 3),
    )


def struct_speed_key(struct: dict) -> tuple:
    """由 ckpt 的结构自描述算出速度表 key（缺字段时用与 dispatcher 相同的默认值）。"""
    return _speed_key({
        "head_src": struct.get("head_src"),
        "head_conv": struct.get("head_conv"),
        "act": struct.get("act"),
        "late_upsample": struct.get("late_upsample"),
        "p2_out_scale": struct.get("p2_out_scale"),
    })


def load_speed_table(path: str) -> dict:
    """读速度表 ⇒ {key: [候选 arm, ...]}（按优先级排好序）。

    **同 key 多项时的优先级**：
      1. ``ms_source == "measured"``（板卡实测，**优先用这个**）
      2. 无 ``ms_source``（早期探针的板卡实测）
      3. ``ms_source == "pred"``（预测值）
    同层内取**表中最后出现**的那条 ⇒ 与 `npu20ms_acceptance.load_speed_table`
    的 ``out[key] = ms``（后写覆盖）**逐位一致** ⇒ 同一结构在两处显示同一个 ms。
    """
    p = Path(path)
    if not p.is_file():
        print(f"[WARN] 速度表不存在：{p} ⇒ soup 的速度列留空")
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    buckets: dict[tuple, list[dict]] = {}
    for a in d.get("arms", []):
        if a.get("ms") is None:
            continue
        sp = a.get("spec") or {}
        key = _speed_key(sp)
        src = a.get("ms_source")
        tier = 1 if src == "measured" else (3 if src == "pred" else 2)
        buckets.setdefault(key, []).append({
            "arm": a.get("name"), "ms": float(a["ms"]), "ms_source": src,
            "fatal": a.get("fatal"), "tier": tier,
        })
    out = {}
    for key, items in buckets.items():
        # 同层内取**最后出现**的（与 acceptance 的后写覆盖一致）
        best_tier = min(i["tier"] for i in items)
        same = [i for i in items if i["tier"] == best_tier]
        out[key] = [same[-1]]
    print(f"[OK] 速度表 {len(out)} 个结构 <- {p}")
    return out


def speed_of(struct: dict, table: dict) -> dict:
    """查速度 ⇒ {"ms":…, "arm":…, "ms_source":…, "fatal":…}（表外 ⇒ 全 None）。"""
    key = struct_speed_key(struct)
    items = table.get(key)
    if not items:
        return {"ms": None, "arm": None, "ms_source": None, "fatal": None}
    best = items[0]
    return {"ms": best["ms"], "arm": best["arm"],
            "ms_source": best["ms_source"], "fatal": best["fatal"]}


# ==============================================================================
# 成员解析
# ==============================================================================
def _trial_number(trial) -> int | None:
    """``"trial_0008"`` / ``8`` / ``"0008"`` ⇒ 8。"""
    if isinstance(trial, int):
        return trial
    s = str(trial).strip()
    if not s:
        return None
    s = s.replace("trial_", "").replace("trial", "")
    try:
        return int(s)
    except ValueError:
        return None


def _epoch_files(trial_dir: Path) -> list[Path]:
    wdir = trial_dir / "weights"
    if not wdir.is_dir():
        return []
    return sorted(wdir.glob("epoch*.pt"),
                  key=lambda p: int(p.stem.replace("epoch", "") or 0))


def _gate_reports(trial_dir: Path) -> list[Path]:
    return sorted(trial_dir.glob("gate*/epoch*/gate_report.json"))


def _member_trial_dirs(member: dict, args) -> tuple[list[Path], dict]:
    """成员 ⇒ 候选 trial 目录列表（role 写法可能命中多条）+ 说明。"""
    root = Path(str(member.get("root") or args.out_root))
    study_name = str(member.get("study") or args.study_name)
    role = str(member.get("role") or "")
    trial = member.get("trial")
    info = {"root": str(root), "study": study_name, "role": role,
            "trial": str(trial) if trial is not None else ""}

    if not role and trial is None:
        info["skip"] = "成员定义既没 role 也没 trial"
        return [], info

    if trial is not None:
        num = _trial_number(trial)
        if num is None:
            info["skip"] = f"trial 名 {trial!r} 解析不出编号"
            return [], info
        tdir = root / "trials" / f"trial_{num:04d}"
        if not tdir.is_dir():
            info["skip"] = f"{tdir} 不存在"
            return [], info
        d = dict(info)
        d.update({"trial_number": num, "trial_dir": str(tdir)})
        return [(tdir, d)], info

    # ---- role ⇒ 查 optuna study ----
    try:
        import optuna
    except Exception as exc:  # noqa: BLE001
        info["skip"] = f"导入 optuna 失败（{exc}）⇒ 无法按 role 找 trial；请改用 trial 写法"
        return [], info
    db = root / "study.db"
    if not db.is_file():
        info["skip"] = f"{db} 不存在 ⇒ 无法按 role 找 trial"
        return [], info
    try:
        st = optuna.load_study(storage="sqlite:///" + str(db), study_name=study_name)
    except Exception as exc:  # noqa: BLE001
        info["skip"] = f"加载 study {study_name} 失败：{exc}"
        return [], info
    hits = [t for t in st.trials if (t.user_attrs or {}).get("role") == role]
    if not hits:
        info["skip"] = f"role {role} 在 study {study_name} 里还没入队"
        return [], info
    info["dup_roles"] = [t.number for t in hits]
    # 同一 role 可能重复入队 ⇒ 全部作为候选，后面按「产物是否完整」挑
    out = []
    for t in sorted(hits, key=lambda x: x.number):
        tdir = root / "trials" / f"trial_{t.number:04d}"
        d = dict(info)
        d.update({"trial_number": t.number, "trial_dir": str(tdir),
                  "state": t.state.name})
        out.append((tdir, d))
    return out, info


def resolve_member(member: dict, args, family_epoch: int) -> tuple[Path | None, dict]:
    """解析一条成员 ⇒ (权重路径 | None, 说明)。

    说明里带 ``skip``（跳过原因）。role 命中多条时 ⇒ 挑**产物最完整**的那条
    （有门禁报告 + 有权重 ⇒ 训练与门禁都跑完了）。
    """
    cand_dirs, info = _member_trial_dirs(member, args)
    if not cand_dirs:
        return None, info

    want = int(family_epoch or args.epoch or 0)
    first_skip = None
    for tdir, d in cand_dirs:
        if not tdir.is_dir():
            d["skip"] = f"{tdir} 不存在"
            first_skip = first_skip or d
            continue

        # ---- 权重：指定 epoch ⇒ 必须存在；否则取该 trial 的最大 epoch 号 ----
        if want > 0:
            w = tdir / "weights" / f"epoch{want}.pt"
            if not w.is_file():
                d["skip"] = f"没有 {w.name}（本轮臂的 epoch 可能不同）"
                first_skip = first_skip or d
                continue
        else:
            fs = _epoch_files(tdir)
            if not fs:
                d["skip"] = f"{tdir}/weights 里没有 epoch*.pt"
                first_skip = first_skip or d
                continue
            w = fs[-1]

        # ---- 门禁报告必须存在（确认这条臂不是崩的）----
        grs = _gate_reports(tdir)
        if not grs:
            d["skip"] = f"{tdir} 没有 gate*/epoch*/gate_report.json ⇒ 门禁没跑完（臂可能没训完）"
            first_skip = first_skip or d
            continue
        gr = grs[-1]
        try:
            g = json.loads(gr.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            d["skip"] = f"门禁报告解析失败：{exc}"
            first_skip = first_skip or d
            continue
        d["gate_report"] = str(gr)
        d["gate_verdict"] = g.get("verdict")
        # 基础设施失败 ⇒ 这条臂本身就是崩的（推理 OOM / 缓存重建失败）⇒ 不进 soup
        if g.get("infrastructure_failure"):
            d["skip"] = (f"门禁是基础设施失败（infrastructure_failure=True，"
                         f"verdict={g.get('verdict')!r}）⇒ 臂本身崩了")
            first_skip = first_skip or d
            continue
        # ★ 2026-11-06：逐轨状态检查。
        #   老门禁（已修）会把「评测没跑出来」写成 verdict="PASS" + gates={}，
        #   而 `infrastructure_failure` 字段压根不存在 ⇒ 只查那个字段会漏掉崩掉的臂
        #   ⇒ 崩臂被当成合格 soup 成员 ⇒ 污染整个 soup。
        _tr = g.get("tracks") or {}
        _bad = sorted(k for k, v in _tr.items()
                      if isinstance(v, dict) and v.get("status") not in (None, "ok"))
        if _bad:
            d["skip"] = (f"门禁里这些轨没跑出指标（{_bad}，verdict={g.get('verdict')!r}）"
                         f"⇒ 臂本身是崩的，不进 soup")
            first_skip = first_skip or d
            continue

        d["weights"] = str(w)
        d["epoch"] = int(w.stem.replace("epoch", "") or 0)
        return w, d

    return None, (first_skip or info)


# ==============================================================================
# soup 构建
# ==============================================================================
def build_soup(family: str, paths: list[Path], struct: dict, epoch: int,
               out_path: Path, nobn: bool, ref_ckpt: dict) -> dict:
    """平均 ⇒ 写 ckpt（**保留结构自描述**）。"""
    state = fuse_states(paths, nobn)
    ckpt = {
        "epoch": int(epoch),
        "model": state,
        "metrics": {"soup": True, "n_models": len(paths), "nobn": bool(nobn)},
        "stride": ref_ckpt.get("stride", 2),
        "imgsz": ref_ckpt.get("imgsz", 640),
        "p0_kwargs": ref_ckpt.get("p0_kwargs", {}),
        "soup": {
            "name": out_path.stem,
            "family": family,
            "sources": [str(p) for p in paths],
            "nobn": bool(nobn),
            "averaged_tensors": len(state),
            "struct": dict(struct),
        },
    }
    # ★ 结构自描述补齐（fuse_weights.py 不抄这些字段 ⇒ 不补 ⇒ 门禁重建出错的图）
    for f in STRUCT_FIELDS:
        ckpt[f] = struct.get(f)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(out_path, ckpt)
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description="结构线专用的跨种子 Model Soup 构建器")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help="study 产物根（trials/ 与 study.db 都在它下面）")
    ap.add_argument("--study-name", default=DEFAULT_STUDY)
    ap.add_argument("--families-json", required=True,
                    help="家族定义 JSON（格式同 build_crossseed_soups.py --families-json）")
    ap.add_argument("--epoch", type=int, default=0,
                    help="统一 epoch 号（默认 0 = 每个 trial 取自己的最大 epoch 号）")
    ap.add_argument("--out-dir", default="",
                    help="soup 产物目录（默认 <out-root>/soups）")
    ap.add_argument("--speed-table", default="",
                    help="结构→速度表（默认 manu/pipeline/npu/struct_speed_table.json）")
    ap.add_argument("--on-mismatch", choices=("fail", "skip"), default="fail",
                    help="成员结构不一致时的行为：fail = 立即报错退出（默认）；skip = 跳过该家族")
    ap.add_argument("--dry-run", action="store_true", help="只解析与校验，不写 .pt")
    args = ap.parse_args()

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

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.out_root) / "soups"
    speed_path = args.speed_table
    if not speed_path:
        speed_path = str(Path(__file__).resolve().parents[2]
                         / "manu/pipeline/npu/struct_speed_table.json")
    speed_table = load_speed_table(speed_path)

    print(f"[STRUCT-SOUP] study={args.study_name} | out-root={args.out_root} | "
          f"家族 {len(fams)} 个（{', '.join(fams)}）| out-dir={out_dir}")

    made, skipped, entries = [], {}, []
    for fam, fs in fams.items():
        if not isinstance(fs, dict):
            print(f"  · {fam}: 家族定义不是对象 ⇒ 跳过")
            skipped[fam] = "家族定义不是对象"
            continue
        recipe = str(fs.get("recipe", ""))
        nobn = bool(fs.get("nobn", False))
        family_epoch = int(fs.get("epoch", 0) or 0)
        members = fs.get("members") or []
        if not isinstance(members, list) or len(members) < 2:
            n = len(members) if isinstance(members, list) else 0
            print(f"  · {fam}: 成员数 {n} < 2 ⇒ 不够平均，跳过")
            skipped[fam] = "成员数 < 2"
            continue

        paths, descs, structs = [], [], []
        for m in members:
            if not isinstance(m, dict):
                print(f"  · {fam}: 成员定义不是对象 ⇒ 跳过该成员")
                continue
            w, info = resolve_member(m, args, family_epoch)
            if w is None:
                print(f"  · {fam}: 成员 {info.get('role') or info.get('trial')} ⇒ "
                      f"{info.get('skip')}")
                continue
            try:
                ck = load_ckpt(w)
            except Exception as exc:  # noqa: BLE001
                print(f"  · {fam}: 成员 {w} 的 ckpt 读不出来：{exc}")
                continue
            st = struct_of(ck)
            label = (info.get("role") or info.get("trial")
                     or f"trial_{info.get('trial_number', 0):04d}")
            info["struct"] = st
            paths.append(w)
            descs.append(info)
            structs.append({"label": label, "struct": st})
            print(f"  · {fam}: 成员 {label}（{info.get('trial_dir')}）epoch{info['epoch']} "
                  f"⇒ {w.name} | head_conv={st.get('head_conv')} act={st.get('act')} "
                  f"late={st.get('late_upsample')} head_src={st.get('head_src')}")

        if len(paths) < 2:
            print(f"[STRUCT-SOUP] {fam}: 可用成员只有 {len(paths)} 个 ⇒ 不够平均，跳过")
            skipped[fam] = f"可用成员只有 {len(paths)} 个"
            continue

        # ★★ 结构自描述逐字段一致（核心坑）★★
        diff = struct_diff(structs)
        if diff:
            msg = (f"[FATAL] 家族 {fam} 的成员结构自描述不一致 ⇒ 拒绝平均"
                   f"（结构不同 ⇒ 权重形状不同 ⇒ 平均出来的模型无法重建同一张图）\n"
                   f"{format_struct_diff(diff)}")
            if args.on_mismatch == "fail":
                print(msg)
                return 1
            print(msg.replace("[FATAL]", "[SKIP]", 1))
            skipped[fam] = "结构自描述不一致"
            continue

        struct = structs[0]["struct"]
        # ★ 2026-11-06：结构自描述必须**完整**。
        #   loader 对 head_level **没有默认值**（None 直接 AssertionError）；
        #   head_conv/late_upsample/p2_out_scale 缺了会静默退回默认值
        #   ⇒ 重建出**另一张图**（这正是「hc32 死模型」那类假象的来源）。
        #   ⇒ 成员 ckpt 缺字段 ⇒ 拒绝建 soup（宁可没有，不要一个加载不了的）。
        _missing = [f for f in STRUCT_FIELDS if struct.get(f) is None]
        if _missing:
            msg = (f"[FATAL] 家族 {fam} 的成员 ckpt 缺结构自描述字段 {_missing} ⇒ 拒绝建 soup"
                   f"（loader 会用默认值重建出另一张图 ⇒ 门禁/导出会得到错误的模型）")
            if args.on_mismatch == "fail":
                print(msg)
                return 1
            print(msg.replace("[FATAL]", "[SKIP]", 1))
            skipped[fam] = f"结构自描述缺字段 {_missing}"
            continue
        epochs = {d.get("epoch") for d in descs}
        if len(epochs) > 1:
            print(f"[WARN] {fam}: 成员的 epoch 不一致（{sorted(epochs)}）⇒ "
                  f"平均的是不同训练长度的权重，请确认这是有意的")

        out = out_dir / f"soup_{fam}.pt"
        sp = speed_of(struct, speed_table)
        if args.dry_run:
            print(f"[STRUCT-SOUP] {fam}: [DRY-RUN] 平均 {len(paths)} 个成员 → {out.name}（不落盘）")
        else:
            ref_ckpt = load_ckpt(paths[0])
            build_soup(fam, paths, struct, max(epochs), out, nobn, ref_ckpt)
            print(f"[STRUCT-SOUP] {fam}: 已写出 {out} | md5={_md5(out)} | "
                  f"速度={sp['ms']} ms（{sp['arm']}, {sp['ms_source']}）")
        made.append(out.name)

        entries.append({
            "name": f"soup_{fam}",
            "family": fam,
            "recipe": recipe,
            "nobn": nobn,
            "epoch": max(epochs),
            "out": str(out),
            "md5": _md5(out) if out.is_file() else None,
            "struct": struct,
            "speed_ms": sp["ms"],
            "speed_arm": sp["arm"],
            "speed_ms_source": sp["ms_source"],
            "speed_fatal": sp["fatal"],
            "members": [{
                "label": d.get("role") or d.get("trial") or d.get("trial_dir"),
                "trial_dir": d.get("trial_dir"),
                "trial_number": d.get("trial_number"),
                "role": d.get("role"),
                "weights": d.get("weights"),
                "epoch": d.get("epoch"),
                "gate_report": d.get("gate_report"),
                "gate_verdict": d.get("gate_verdict"),
                "struct": d.get("struct"),
            } for d in descs],
        })

    # ---- 候选清单 ----
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "soup_candidates.json").write_text(json.dumps({
            "note": "结构线跨种子 soup（同配方多种子·末 epoch 权重平均）· 结构自描述逐字段校验",
            "study": args.study_name,
            "out_root": args.out_root,
            "out_dir": str(out_dir),
            "speed_table": speed_path,
            "epoch": int(args.epoch),
            "made": made,
            "skipped": skipped,
            "candidates": entries,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[STRUCT-SOUP] 清单已写 {out_dir / 'soup_candidates.json'}")

    print(f"[STRUCT-SOUP] 完成：建成 {len(made)} 个 ⇒ {made or '无'}")
    if skipped:
        print(f"[STRUCT-SOUP] 跳过 {len(skipped)} 个家族：{skipped}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
