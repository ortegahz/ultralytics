"""一次性重判：把历史板卡 T4 读数按「同会话相对基线」口径重算 pass_t4。

🟢🟢 2026-11-19 15:40 —— 起因（板卡换旧板 .64 + 他任务长期占用）：
   · 绝对读数整体上漂 0.3~0.5 ms；**基线臂 2020035 自己在脏会话里读到 20.067/20.129 ms**
     （远超 19.989 红线）⇒ 绝对口径把**同计算图**的升轨候选全部误杀
     （2020023 / 2000051 / 2010025 / 2000025 被判 FAIL）⇒ 升轨链最后一环被打断。
   · 修法：**同会话差值口径**。板卡一次测速会话会把在板上的所有臂（含基线臂）测一遍
     ⇒ 取同会话基线读数作 ref，diff = 候选 − 同会话基线（消掉板卡整体漂移/被占用）。
        pass = (绝对中位数 ≤ LIMIT_MS) OR (差值中位数 ≤ LIMIT_MS − 基线干净参考 19.629 = 0.36)
   · 语义不变（「不比 trial_0031×1.02 慢」），但不再受换板 / 他任务占用影响。

用法（服务器）：
    python -u manu/pipeline/npu/board_t4_rejudge.py            # 重判并写回
    python -u manu/pipeline/npu/board_t4_rejudge.py --dry      # 只看不写
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
if not os.path.isdir(REPO):
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as ns  # noqa: E402
from manu.pipeline.npu import t4_criterion as T4  # noqa: E402（判据唯一实现 ⇒ 本脚本不再自抄一份）

OUT = "/mnt/data/siping/runs/npu20ms/board_measure_auto"
DONE = os.path.join(OUT, "measured.json")
SAMPLES = os.path.join(OUT, "t4_samples.json")
LIMIT_MS = T4.LIMIT_MS
BASE_REF_DEFAULT = T4.BASE_REF_DEFAULT
DIFF_TOL_MIN = T4.DIFF_TOL_MIN

_ARM_RE = re.compile(r"^arm_n(\d+)_ep(\d+)_int8$")
_SUM_RE = re.compile(r"@@SUMMARY@@ (\S+)\|rknn_run_avg_ms\|([\d.]+)")


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def median(xs: list) -> float:
    s = sorted(float(x) for x in xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def arm_key(name: str) -> str | None:
    """`arm_n2020035_ep5_int8` → `2020035_ep5`"""
    m = _ARM_RE.match(str(name).strip())
    return "%s_ep%s" % (m.group(1), m.group(2)) if m else None


def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return default


def dump_json(path: str, data) -> None:
    tmp = path + ".tmp_rejudge"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def base_ref_ms(led: dict) -> float:
    """基线臂在「干净板卡」上的参考绝对值。

    🔴 与 `board_t4_watch.base_ref_ms()` **同一口径**：只接受 ≤ 红线的读数
      （超红线说明它是「靠相对口径过的漂移读数」⇒ 不是干净参照 ⇒ 否则容差塌到 0.05 ms）。
    """
    b = (led.get("baseline_v2") or {})
    v = float(b.get("ms_clean_ref") or 0)
    if v <= 5.0:
        v = float(b.get("ms_new_board") or 0)
    return v if 5.0 < v <= LIMIT_MS else BASE_REF_DEFAULT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    led = ns.load_ledger()
    arms = led.get("arms") or {}
    b = led.get("baseline_v2") or {}
    bt = int(b.get("trial") or 0)
    be = str(b.get("epoch") or 5)
    base_key = "%s_ep%s" % (bt, be)
    tol = T4.diff_tol(base_ref_ms(led))
    log("基线臂 = %s｜红线 %.3f ms｜干净参考 %.3f ms｜相对容差 %.3f ms"
        % (base_key, LIMIT_MS, base_ref_ms(led), tol))

    # ---- 1. 解析全部会话文件 ----
    pools = load_json(SAMPLES, {}) or {}
    done = load_json(DONE, {}) or {}
    sess_files = sorted(f for f in os.listdir(OUT) if f.startswith("session_") and f.endswith(".txt"))
    abs_new: dict[str, list] = {}
    diff_new: dict[str, list] = {}
    ref_of: dict[str, list] = {}
    n_sess = 0
    for fn in sess_files:
        try:
            txt = open(os.path.join(OUT, fn), "r", encoding="utf-8", errors="ignore").read()
        except Exception:  # noqa: BLE001
            continue
        vals: dict[str, float] = {}
        for m in _SUM_RE.finditer(txt):
            k = arm_key(m.group(1))
            if k:
                vals[k] = float(m.group(2))
        if base_key not in vals:
            continue
        n_sess += 1
        ref = vals[base_key]
        for k, v in vals.items():
            abs_new.setdefault(k, []).append(v)
            diff_new.setdefault(k, []).append(round(v - ref, 4))
            ref_of.setdefault(k, []).append(ref)
    log("会话文件 %d 个，其中有基线臂读数的 %d 个" % (len(sess_files), n_sess))

    # ---- 2. 合并绝对样本池 + 新建差值池 ----
    all_keys = set(pools) | set(done) | set(abs_new)
    abs_keys = {k for k in all_keys if not k.endswith("|d")}
    changes = []
    new_pools = dict(pools)
    for k in abs_keys:
        base = k[:-2] if k.endswith("|d") else k
        pool = list(pools.get(base) or [])
        if not pool:
            pool = list(abs_new.get(base) or [])
        new_pools[base] = pool
        dpool = list(pools.get(base + "|d") or [])
        for dv in diff_new.get(base, []):
            if dv not in dpool:
                dpool.append(dv)
        new_pools[base + "|d"] = dpool

        abs_med = median(pool) if pool else None
        dmed = median(dpool) if dpool else None
        _dec = T4.decide(abs_med, dmed, tol)      # 判据唯一实现
        ok_abs, ok_rel, ok = _dec["ok_abs"], _dec["ok_rel"], _dec["ok"]

        rec = done.get(base) or {}
        old_pass = bool(rec.get("pass_t4"))
        old_ms = rec.get("ms")
        if rec or pool:
            # 🟢 只在会话里出现过（没有 done 记录）的臂也补一条实测记录
            rec.setdefault("trial", int(base.split("_ep")[0]))
            rec.setdefault("ep", base.split("_ep")[-1])
            rec.update({
                "ms": abs_med if abs_med is not None else rec.get("ms"),
                "samples": [round(float(x), 4) for x in pool],
                "n_samples": len(pool),
                "limit_ms": LIMIT_MS,
                "pass_t4": ok,
                "pass_t4_abs": ok_abs,
                "pass_t4_rel": ok_rel,
                "ref_ms": (round(median(ref_of[base]), 4) if base in ref_of else rec.get("ref_ms")),
                "diff_ms": (round(dmed, 4) if dmed is not None else None),
                "diff_samples": [round(float(x), 4) for x in dpool],
                "diff_tol_ms": round(tol, 4),
                "rejudged_at": time.strftime("%F %T"),
            })
            done[base] = rec
        if old_pass != ok:
            changes.append((base, old_pass, ok, old_ms, abs_med, dmed, len(dpool)))

    # ---- 3. 台账回写 ----
    n_led = 0
    for base, _old, ok, _oms, _am, _dm, _nd in changes:
        trial_s, ep_s = base.split("_ep")
        st = arms.get(trial_s)
        if not isinstance(st, dict):
            continue
        rec = done.get(base) or {}
        st["board_ms"] = float(rec.get("ms") or 0)
        st.setdefault("board_ms_epochs", {})[ep_s] = float(rec.get("ms") or 0)
        st["board_pass_t4"] = bool(ok)
        st["board_limit_ms"] = LIMIT_MS
        st["board_measured_at"] = rec.get("rejudged_at") or time.strftime("%F %T")
        if rec.get("diff_ms") is not None:
            st["board_diff_ms"] = float(rec["diff_ms"])
            st["board_diff_tol_ms"] = round(tol, 4)
        arms[trial_s] = st
        n_led += 1

    log("判定变化 %d 个臂（台账回写 %d）" % (len(changes), n_led))
    for base, op, np_, oms, am, dm, nd in sorted(changes, key=lambda x: x[0]):
        log("  %-14s %s ⇒ %s｜绝对 %s → %s｜同会话差值 %s ms（容差 %.3f，样本 %d）"
            % (base, "FAIL" if not op else "PASS", "FAIL" if not np_ else "PASS",
               oms, am, dm, tol, nd))

    if args.dry:
        log("--dry ⇒ 不写盘")
        return
    dump_json(DONE, done)
    dump_json(SAMPLES, new_pools)
    if n_led:
        led["arms"] = arms
        ns.save_ledger(led)
    log("写回完成：measured.json / t4_samples.json / 台账")


if __name__ == "__main__":
    main()
