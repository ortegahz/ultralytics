#!/usr/bin/env python3
"""跨家族权重插值 / 分层插值 / TIES / 多锚点混合（2026-11-04 · goal「方向一」）

目标
----
`trial_0049` epoch20（T3 全库第一 0.906392）与 `cs_t0008p2`（T1 合规 0.8959005）
落在**同一条 RC2 谱系**的不同优化轨迹上：

* `cs_t0008p2` 的 5 个成员**全部 rc2-init**（见 `swa/crossseed_manifest.json`）；
* `trial_0049` 是 `soup_strict3-init` 起手的 `w2.0·lr3.5·20ep` 臂，其 init 也是 RC2 谱系的平均。

⇒ 两端同源（实测全局余弦 **0.974**、相对 L2 **0.491**）⇒ 属「同盆地不同点」，
在参数空间扫描应当能连出一条连续路径。

**四种合并模式**（都是零训练、零反向传播）：

============  ==================================================================
模式           作用
============  ==================================================================
``linear``    两端线性插值 ``W = α·A + (1−α)·B``；``α ∉ [0,1]`` 即**外插**。
``ties``      TIES-Merging：修剪 → 符号选举 → 不相交平均（抗跨任务符号冲突）。
``groups``    **分层插值**：不同参数组用不同 α（早期骨干 / 颈与检测 / 热图头）。
              ⇒ T1（过饱和比赛集）与 T3（极暗长尾难例）若由**不同深度**主导，
              分层 α 就能拿到全局 α 拿不到的「兼得」点。
``simplex``   **多锚点混合**（任意多 checkpoint 的加权平均）⇒ 权重空间单纯形搜索，
              兼得点可能不在两端连线上，而在 3~4 个锚点的凸组合里。
              也可用来做**沿训练轨迹的 SWA/EMA**（把同一臂的多个 epoch 等权平均）。
============  ==================================================================

参数分组（本模型实测键名，523 个张量）
----------------------------------------
* ``early``  —— ``b2.*`` ``b4.*`` ``b6.*``   （骨干前 3 个 CSP stage：纹理/高频）
* ``mid``    —— ``b8.*`` ``b10.*`` ``c13.*`` ``c16.*`` ``c19.*`` ``c22.*``（颈 + 检测）
* ``head``   —— ``head.*`` ``fuse_p1.*``     （热图头 / 偏移分支）
* ``p0``     —— ``p0_highway.*``（29 个，**永不插值**：两端来源不同，平均会污染它；
  门禁以 ``use_p0_highway=False`` 加载会 drop 掉它们 ⇒ **门禁数字不受影响**）

关键实现约定（与 `fuse_weights.py` 完全对齐，保证门禁能直接加载）
--------------------------------------------------------------------
* 权重读取：``ckpt["model"]``（是 Module 则取 ``.state_dict()``）；
* 输出格式：``{"epoch","model","metrics","stride","imgsz","p0_kwargs","interp"}``
  （门禁 ``cache_trial0474_inferences.py`` 读 ``ckpt["stride"]`` 与 ``ckpt["p0_kwargs"]``）。

用法
----
    # 诊断两端是否同盆地（CPU，秒级）
    python manu/training/build_weight_interpolation.py --diagnose \\
        --a .../trial_0049/weights/epoch20.pt --b .../cs_t0008p2.pt

    # 线性扫参
    python manu/training/build_weight_interpolation.py --mode linear \\
        --a .../epoch20.pt --b .../cs_t0008p2.pt \\
        --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \\
        --out-dir /mnt/data/siping/runs/nogmc_t3/interp --prefix interp_t49_p2

    # 分层插值（early:mid:head）
    python manu/training/build_weight_interpolation.py --mode groups \\
        --a .../epoch20.pt --b .../cs_t0008p2.pt \\
        --alphas "0.4:0.6:0.8,0.6:0.6:0.8,0.8:0.6:0.4" --out-dir ... --prefix interp_grp

    # 多锚点混合
    python manu/training/build_weight_interpolation.py --mode simplex \\
        --anchors "1=ep14.pt,1=ep15.pt,1=ep16.pt" --out-dir ... --prefix interp_swa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

P0_PREFIX = "p0_highway"
BN_STAT_SUFFIX = ("running_mean", "running_var")

# ---- 参数分组（键名前缀，实测自 cs_t0008p2.pt 的 523 个张量）---------------- #
EARLY_PREFIXES = ("b2.", "b4.", "b6.")
MID_PREFIXES = ("b8.", "b10.", "c13.", "c16.", "c19.", "c22.")
HEAD_PREFIXES = ("head.", "fuse_p1.")


def classify(key: str) -> str:
    if key.startswith(P0_PREFIX):
        return "p0"
    if key.startswith(HEAD_PREFIXES):
        return "head"
    if key.startswith(MID_PREFIXES):
        return "mid"
    if key.startswith(EARLY_PREFIXES):
        return "early"
    return "other"


# --------------------------------------------------------------------------- #
# 权重读取
# --------------------------------------------------------------------------- #
def load_state(path: Path) -> tuple[dict, dict]:
    """读 checkpoint，返回 (state_dict, 原始 ckpt)。"""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return {k: v for k, v in state.items()}, ckpt


def is_p0(key: str) -> bool:
    return key.startswith(P0_PREFIX)


def is_bn_stat(key: str) -> bool:
    return key.endswith(BN_STAT_SUFFIX)


def is_float_tensor(v) -> bool:
    return torch.is_tensor(v) and v.is_floating_point()


# --------------------------------------------------------------------------- #
# 权重空间诊断
# --------------------------------------------------------------------------- #
def diagnose(sa: dict, sb: dict) -> dict:
    """逐张量算余弦与相对距离，聚合出「是否同盆地」的证据。

    判读带（本项目实测标定）：
      * 全局余弦 > 0.95 ⇒ **同盆地**，线性插值有意义（soup 常规情形）；
      * 0.85 ~ 0.95 ⇒ 同源但距离大，插值仍可试，但**要同时试 TIES / 分层 α**；
      * < 0.85 ⇒ 很可能跨盆地，线性插值多半撞壁垒。
    """
    rows = []
    tot_a2 = tot_b2 = tot_d2 = 0.0
    cos_num = cos_den = 0.0
    per_group: dict[str, list] = {}
    for k in sorted(sa):
        a, b = sa[k], sb.get(k)
        if b is None or not is_float_tensor(a) or not is_float_tensor(b):
            continue
        af = a.to(torch.float64).reshape(-1)
        bf = b.to(torch.float64).reshape(-1)
        if af.numel() != bf.numel():
            continue
        na = float(af.norm())
        nb = float(bf.norm())
        nd = float((af - bf).norm())
        dot = float((af * bf).sum())
        cos = dot / (na * nb) if na > 0 and nb > 0 else None
        rel = nd / na if na > 0 else None
        g = classify(k)
        rows.append({"key": k, "group": g, "n": int(af.numel()),
                     "cos": cos, "rel": rel, "norm_a": na, "norm_b": nb})
        tot_a2 += na * na
        tot_b2 += nb * nb
        tot_d2 += nd * nd
        if na > 0 and nb > 0:
            cos_num += dot
            cos_den += na * nb
        per_group.setdefault(g, []).append({"cos": cos, "rel": rel, "n": af.numel()})

    glob_rel = math.sqrt(tot_d2) / max(math.sqrt((tot_a2 + tot_b2) / 2.0), 1e-12)
    glob_cos = cos_num / cos_den if cos_den > 0 else None

    def _agg(items):
        cs = [x["cos"] for x in items if x["cos"] is not None]
        rs = [x["rel"] for x in items if x["rel"] is not None]
        return {
            "n": len(items),
            "mean_cos": (sum(cs) / len(cs)) if cs else None,
            "mean_rel": (sum(rs) / len(rs)) if rs else None,
        }

    worst = sorted([r for r in rows if r["cos"] is not None], key=lambda r: r["cos"])[:12]
    if glob_cos is None:
        verdict = "无法判定（张量全零或键不匹配）"
    elif glob_cos > 0.95:
        verdict = "同盆地（线性/分层插值有意义）"
    elif glob_cos > 0.85:
        verdict = "同源但距离较大 ⇒ 线性可试，**必须同时试 TIES / 分层 α / 多锚点混合**"
    else:
        verdict = "疑似跨盆地 ⇒ 线性插值多半撞壁垒（改 TIES / 排列对齐 / 轨迹内 SWA）"
    return {
        "n_tensors_compared": len(rows),
        "global_cosine": glob_cos,
        "global_rel_l2": glob_rel,
        "per_group": {g: _agg(v) for g, v in sorted(per_group.items())},
        "worst_cos": worst,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# 合并
# --------------------------------------------------------------------------- #
def _blend(states: list[dict], weights: list[float], key: str, group: str,
           alphas: dict | None, nobn: bool, p0_anchor: int) -> object:
    """按分组 α（或全局权重）混合单个张量。"""
    ref = states[0][key]
    if not is_float_tensor(ref):
        return ref.clone() if torch.is_tensor(ref) else ref
    if is_p0(key):
        src = states[p0_anchor][key]
        return src.clone() if torch.is_tensor(src) else src
    if nobn and is_bn_stat(key):
        return ref.clone()
    if alphas is not None:
        a = alphas.get(group, alphas.get("default", 0.5))
        acc = (states[0][key].to(torch.float64) * a
               + states[1][key].to(torch.float64) * (1.0 - a))
        return acc.to(ref.dtype)
    acc = torch.zeros_like(ref, dtype=torch.float64)
    for st, w in zip(states, weights):
        acc += st[key].to(torch.float64) * w
    return acc.to(ref.dtype)


def merge_linear(sa: dict, sb: dict, alpha: float, nobn: bool, p0_anchor: int,
                 alphas: dict | None = None) -> dict:
    out = {}
    for k in sorted(sa):
        g = classify(k)
        out[k] = _blend([sa, sb], [alpha, 1.0 - alpha], k, g, alphas, nobn, p0_anchor)
    return out


def merge_groups(sa: dict, sb: dict, alphas: dict, nobn: bool, p0_anchor: int) -> dict:
    out = {}
    for k in sorted(sa):
        out[k] = _blend([sa, sb], [0.5, 0.5], k, classify(k), alphas, nobn, p0_anchor)
    return out


def merge_simplex(states: list[dict], weights: list[float], nobn: bool,
                  p0_anchor: int) -> dict:
    out = {}
    for k in sorted(states[0]):
        out[k] = _blend(states, weights, k, classify(k), None, nobn, p0_anchor)
    return out


def merge_surgery(sa: dict, sb: dict, from_b_prefixes: list[str], nobn: bool,
                  p0_anchor: int) -> dict:
    """**模块手术**：按键名前缀整块取自某个端点。

    动机（2026-11-04 实测）：trial_0049 与 cs_t0008p2 的 **early/mid 组余弦 0.999**，
    差异几乎全在 ``head.*``（热图头 + 偏移分支 + 其 BN 统计量）⇒
    「哪些 head 子模块带来 T3、哪些带来 T1」是一个**离散**问题，
    连续 α 不一定落在兼得点上，整块换源反而可能。

    语义：``from_b_prefixes`` 里的前缀从 **B** 取，其余全部从 **A** 取。
    """
    out = {}
    for k in sorted(sa):
        g = classify(k)
        if is_p0(k):
            src = (sa, sb)[p0_anchor][k]
            out[k] = src.clone() if torch.is_tensor(src) else src
            continue
        if not is_float_tensor(sa[k]) or not is_float_tensor(sb[k]):
            out[k] = sa[k].clone() if torch.is_tensor(sa[k]) else sa[k]
            continue
        if nobn and is_bn_stat(k):
            out[k] = sa[k].clone()
            continue
        use_b = any(k.startswith(p) for p in from_b_prefixes if p)
        src = sb[k] if use_b else sa[k]
        out[k] = src.clone()
    return out


def _trim_topk(t: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return t
    flat = t.reshape(-1)
    k = max(1, int(round(flat.numel() * density)))
    thresh = torch.kthvalue(flat.abs().cpu().float(), flat.numel() - k + 1).values.item()
    return torch.where(t.abs() >= thresh, t, torch.zeros_like(t))


def merge_ties(sa: dict, sb: dict, density: float, nobn: bool, p0_anchor: int) -> dict:
    """TIES-Merging：修剪 → 符号选举 → 不相交平均。"""
    out = {}
    for k in sorted(sa):
        a, b = sa[k], sb[k]
        if is_p0(k):
            src = (sa, sb)[p0_anchor][k]
            out[k] = src.clone() if torch.is_tensor(src) else src
            continue
        if not is_float_tensor(a) or not is_float_tensor(b):
            out[k] = a.clone() if torch.is_tensor(a) else a
            continue
        if nobn and is_bn_stat(k):
            out[k] = a.clone()
            continue
        at = _trim_topk(a.to(torch.float64), density)
        bt = _trim_topk(b.to(torch.float64), density)
        sign = torch.sign(at + bt)
        ok_a = torch.sign(at) == sign
        ok_b = torch.sign(bt) == sign
        num = torch.where(ok_a, at, torch.zeros_like(at)) + torch.where(ok_b, bt, torch.zeros_like(bt))
        den = ok_a.to(torch.float64) + ok_b.to(torch.float64)
        merged = torch.where(den > 0, num / den.clamp(min=1.0), torch.zeros_like(num))
        out[k] = (merged * sign).to(a.dtype)
    return out


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def save_ckpt(state: dict, ref_ckpt: dict, out: Path, meta: dict) -> str:
    ckpt = {
        "epoch": 0,
        "model": state,
        "metrics": meta.get("metrics", {}),
        "stride": ref_ckpt.get("stride", 2),
        "imgsz": ref_ckpt.get("imgsz", 640),
        "p0_kwargs": ref_ckpt.get("p0_kwargs", {}),
        "interp": meta,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(
        json.dumps({**meta, "out": str(out), "md5": md5, "bytes": out.stat().st_size},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INTERP] 已写出 {out.name}  md5={md5}  {out.stat().st_size:,} B", flush=True)
    return md5


def parse_anchors(spec: str) -> list[tuple[float, Path]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            w, p = part.split("=", 1)
            out.append((float(w), Path(p.strip())))
        else:
            out.append((1.0, Path(part)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="跨家族权重插值 / 分层插值 / TIES / 多锚点混合")
    ap.add_argument("--a", default="", help="端点 A（例：trial_0049 epoch20）")
    ap.add_argument("--b", default="", help="端点 B（例：cs_t0008p2）")
    ap.add_argument("--anchors", default="", help="多锚点：'w=path,w=path'（simplex 模式）")
    ap.add_argument("--alphas", default="", help="linear: '0.1,0.2'｜groups: 'early:mid:head' 三元组")
    ap.add_argument("--mode", default="linear",
                    choices=["linear", "ties", "groups", "simplex", "surgery"])
    ap.add_argument("--ties-density", type=float, default=0.2)
    ap.add_argument("--from-b", default="", help="surgery 模式：这些前缀整块取自 B，其余取自 A")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--prefix", default="interp")
    ap.add_argument("--nobn", action="store_true", help="BN 统计量取第一个锚点")
    ap.add_argument("--p0-anchor", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--diag-out", default="")
    args = ap.parse_args()

    # ---- 读权重 ----
    if args.mode == "simplex":
        anc = parse_anchors(args.anchors)
        if not anc:
            raise SystemExit("[FATAL] simplex 模式需要 --anchors")
        paths = [p for _, p in anc]
        raw_w = [w for w, _ in anc]
        tot = sum(raw_w)
        weights = [w / tot for w in raw_w]
    else:
        if not args.a or not args.b:
            raise SystemExit("[FATAL] 需要 --a 与 --b")
        paths = [Path(args.a), Path(args.b)]
        weights = [0.5, 0.5]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"[FATAL] 权重不存在：{p}")
    loaded = [load_state(p) for p in paths]
    states = [s for s, _ in loaded]
    ref_ckpt = loaded[0][1]
    keys = set(states[0])
    for i, st in enumerate(states[1:], 1):
        if set(st) != keys:
            raise SystemExit(f"[FATAL] 第 {i+1} 个锚点键集合不同："
                             f"{sorted(keys ^ set(st))[:10]}")
    print(f"[INTERP] {len(states)} 个锚点，各 {len(keys)} 个张量，键集合逐位一致 ✅", flush=True)

    # ---- 诊断（只对前两个锚点做）----
    if len(states) >= 2:
        diag = diagnose(states[0], states[1])
        print("[INTERP] 权重空间诊断（锚点0 vs 锚点1）：", flush=True)
        gc = diag["global_cosine"]
        print(f"   全局余弦 = {gc if gc is None else round(gc, 6)}"
              f" | 全局相对 L2 = {round(diag['global_rel_l2'], 6)}", flush=True)
        for g, v in diag["per_group"].items():
            mc = v["mean_cos"]
            print(f"   分组 {g:<6} n={v['n']:<4} 均余弦={mc if mc is None else round(mc, 4)}"
                  f" 均相对={v['mean_rel'] if v['mean_rel'] is None else round(v['mean_rel'], 4)}",
                  flush=True)
        print(f"   判定：{diag['verdict']}", flush=True)
        for r in diag["worst_cos"][:6]:
            print(f"   余弦最低 {r['key']:<52} cos={r['cos']:.4f} rel={r['rel']:.4f}", flush=True)
        if args.diag_out:
            Path(args.diag_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.diag_out).write_text(
                json.dumps({"anchors": [str(p) for p in paths], "diag": diag},
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[INTERP] 诊断已写 {args.diag_out}", flush=True)
    if args.diagnose:
        return 0
    if not args.out_dir:
        raise SystemExit("[FATAL] 落盘需要 --out-dir")
    out_dir = Path(args.out_dir)

    # ---- 生成 ----
    if args.mode == "linear":
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        if not alphas:
            raise SystemExit("[FATAL] linear 模式需要 --alphas")
        for al in alphas:
            state = merge_linear(states[0], states[1], al, args.nobn, args.p0_anchor)
            tag = f"a{al:+.3f}".replace("+", "p").replace("-", "m")
            meta = {"kind": "weight_interpolation", "mode": "linear", "alpha_a": al,
                    "a": str(paths[0]), "b": str(paths[1]), "w_a": al,
                    "w_b": round(1.0 - al, 6), "nobn": bool(args.nobn),
                    "p0_anchor": args.p0_anchor,
                    "metrics": {"interp": True, "mode": "linear", "alpha_a": al}}
            save_ckpt(state, ref_ckpt, out_dir / f"{args.prefix}_{tag}.pt", meta)
    elif args.mode == "groups":
        specs = [x for x in args.alphas.split(",") if x.strip()]
        if not specs:
            raise SystemExit("[FATAL] groups 模式需要 --alphas（early:mid:head 三元组）")
        for sp in specs:
            parts = [float(x) for x in sp.split(":")]
            if len(parts) != 3:
                raise SystemExit(f"[FATAL] 三元组格式错误：{sp}")
            al_e, al_m, al_h = parts
            alphas = {"early": al_e, "mid": al_m, "head": al_h, "other": al_m, "default": al_m}
            state = merge_groups(states[0], states[1], alphas, args.nobn, args.p0_anchor)
            tag = f"g{al_e:.2f}_{al_m:.2f}_{al_h:.2f}".replace("-", "m")
            meta = {"kind": "weight_interpolation", "mode": "groups",
                    "alphas": {"early": al_e, "mid": al_m, "head": al_h},
                    "a": str(paths[0]), "b": str(paths[1]), "nobn": bool(args.nobn),
                    "p0_anchor": args.p0_anchor,
                    "metrics": {"interp": True, "mode": "groups",
                                "alphas": {"early": al_e, "mid": al_m, "head": al_h}}}
            save_ckpt(state, ref_ckpt, out_dir / f"{args.prefix}_{tag}.pt", meta)
    elif args.mode == "simplex":
        state = merge_simplex(states, weights, args.nobn, args.p0_anchor)
        meta = {"kind": "weight_interpolation", "mode": "simplex",
                "anchors": [{"path": str(p), "w": w} for (w, p), w in
                             zip(parse_anchors(args.anchors), weights)],
                "nobn": bool(args.nobn), "p0_anchor": args.p0_anchor,
                "metrics": {"interp": True, "mode": "simplex"}}
        save_ckpt(state, ref_ckpt, out_dir / f"{args.prefix}.pt", meta)
    elif args.mode == "surgery":
        pfx = [x.strip() for x in args.from_b.split(",") if x.strip()]
        state = merge_surgery(states[0], states[1], pfx, args.nobn, args.p0_anchor)
        tag = "b" + "-".join(p.replace(".", "") for p in pfx)[:24] if pfx else "allA"
        meta = {"kind": "weight_interpolation", "mode": "surgery",
                "from_b_prefixes": pfx, "a": str(paths[0]), "b": str(paths[1]),
                "nobn": bool(args.nobn), "p0_anchor": args.p0_anchor,
                "metrics": {"interp": True, "mode": "surgery", "from_b": pfx}}
        save_ckpt(state, ref_ckpt, out_dir / f"{args.prefix}_{tag}.pt", meta)
    else:  # ties
        state = merge_ties(states[0], states[1], args.ties_density, args.nobn, args.p0_anchor)
        meta = {"kind": "weight_interpolation", "mode": "ties",
                "ties_density": args.ties_density, "a": str(paths[0]), "b": str(paths[1]),
                "nobn": bool(args.nobn), "p0_anchor": args.p0_anchor,
                "metrics": {"interp": True, "mode": "ties",
                            "ties_density": args.ties_density}}
        save_ckpt(state, ref_ckpt, out_dir / f"{args.prefix}_ties{args.ties_density:g}.pt", meta)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
