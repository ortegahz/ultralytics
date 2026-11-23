# -*- coding: utf-8 -*-
"""板卡 T4 时延判据（**纯函数 · 零第三方依赖 · 可离线单测**）。

为什么要有这个模块（2026-11-20 现场事故，memory §266）
--------------------------------------------------
板卡读数**热漂移 ±0.5 ms**（逐小时中位数：11-13 19.68 → 11-19 18 时 20.005 → 11-20 13 时 19.831，
批内 max 21.462），而交付红线 `LIMIT_MS = 19.989` **正落在漂移带中央**
⇒ **同一张计算图**（基线家族 bw10 克隆臂）现在测出 **19.94 ~ 21.46 ms**
⇒ 纯绝对口径把它们**全部误判 FAIL** ⇒ **升轨链最后一环断链**（周末 65 h 算力产出的臂全部升不了轨）。

两条口径（**OR** 关系，语义不变：「不比 trial_0031×1.02 慢」）：

* **绝对口径**：`ms ≤ LIMIT_MS`
* **同会话相对口径**：`ms − 同会话基线臂读数 ≤ LIMIT_MS − 基线干净参考值`
  用**同一次板卡会话**里当前基线臂的读数作参照 ⇒ 消掉板卡整体漂移 / 他任务占用。
  同图臂 diff ≈ 0 ⇒ 稳定 PASS；真更慢的臂 diff > 容差 ⇒ 稳定 FAIL。

🔴 **判据只此一处**（`decide()`）：
之前 `board_t4_watch` / `promote_candidate` / `auto_promote_watch` **各写一遍**，
其中 `sync_measured_to_ledger()` 那次用**绝对口径**把「相对口径」的结果整个刷掉
（2026-11-19 15:27 现场事故：116 个臂的 `board_pass_t4=True` 被覆盖回 False）。
⇒ 现在生产者（`board_t4_watch`）与消费者（`promote_candidate` / `auto_promote_watch`）
**共用本模块的常量与判据**，并由 `manu/pipeline/npu/tests/test_t4_criterion.py` 锁死行为。

用法（服务器 / 本机均可，零依赖）::

    from manu.pipeline.npu import t4_criterion as T4
    d = T4.decide(ms_median=20.05, diff_median=-0.05, tol=T4.diff_tol(19.629))
    d["ok"]        # True（绝对口径 FAIL、相对口径 PASS ⇒ 板卡热漂移下的同图臂不再被误杀）
    d["ok_abs"], d["ok_rel"]
"""

from __future__ import annotations

import re

# ---- 常量（红线沿用交付口径：trial_0031 19.597 ms × 1.02 = 19.989 ms）------
LIMIT_MS = 19.989          # 交付时延红线（绝对口径）
BASE_REF_DEFAULT = 19.629  # Baseline_v3.0 在**干净板卡**上的参考绝对值（容差换算基准）
DIFF_TOL_MIN = 0.05        # 容差下限（防基线读数缺失 / 异常时误判为 PASS）

#: 板卡会话摘要里的臂名（`measure_hm_speed_board.sh` 写出的 `@@SUMMARY@@` 键）
#: ⚠️ epoch 允许 `5` 或 `5_0`（文件名里的 "." 被替换成 "_"，见 `norm_arm`）
ARM_RE = re.compile(r"^arm_n(\d+)_ep(\d+(?:_\d+)?)_int8$")


def norm_arm(key: str) -> str:
    """板卡脚本会把臂名写成 `arm_<name>_int8` ⇒ 归一化后再比对。"""
    k = str(key).strip()
    if k.startswith("arm_"):
        k = k[4:]
    if k.endswith("_int8"):
        k = k[:-5]
    # ⚠️ 文件名侧做了 `.replace(".", "_")`（`n{trial}_ep{ep}`）⇒ 比对侧必须同样归一化，
    #    否则 epoch 存成 "5.0" 时**参照永远取不到**（相对口径静默失效）。
    k = k.replace(".", "_")
    return k


def pick_ms(summary: dict, trial: int, ep) -> float | None:
    """从会话摘要（`{臂名: rknn_run_avg_ms}`）里挑出指定臂的读数。"""
    want = norm_arm(f"n{trial}_ep{ep}")
    for k, v in (summary or {}).items():
        if norm_arm(k) == want:
            return float(v)
    cand = [float(v) for k, v in (summary or {}).items()
            if norm_arm(k).startswith(f"n{trial}_ep")]
    return cand[0] if len(cand) == 1 else None


def session_ref_ms(summary: dict, base_trial: int | None, base_epoch) -> float | None:
    """**同会话**里「当前基线臂」的读数（板卡一次会话会把在板上的臂全测一遍）。

    ⇒ 相对口径的参照。基线臂**不在板上**时返回 None ⇒ 相对口径失效 ⇒ 退回绝对口径。
    """
    if not base_trial:
        return None
    want = norm_arm(f"n{base_trial}_ep{base_epoch}")
    for k, v in (summary or {}).items():
        if norm_arm(k) == want:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def diff_tol(base_ref_ms: float | None) -> float:
    """允许的同会话相对差（候选 − 同会话基线）＝ `LIMIT_MS − 基线干净参考值`。"""
    try:
        ref = float(base_ref_ms)
    except (TypeError, ValueError):
        ref = BASE_REF_DEFAULT
    return max(DIFF_TOL_MIN, LIMIT_MS - ref)


def decide(ms_median: float | None, diff_median: float | None, tol: float) -> dict:
    """T4 判定的**唯一实现**（绝对口径 OR 同会话相对口径）。

    >>> decide(20.05, -0.05, 0.36)["ok"]
    True
    >>> decide(20.05, None, 0.36)["ok"]     # 参照缺失 ⇒ 只剩绝对口径 ⇒ 不误放行
    False
    """
    ok_abs = bool(ms_median is not None and float(ms_median) <= LIMIT_MS)
    ok_rel = bool(diff_median is not None and float(diff_median) <= float(tol))
    # 🔴 **没有绝对读数 ⇒ 不判定**（"只有差值没有读数"不是放行理由）
    ok = bool(ms_median is not None and (ok_abs or ok_rel))
    return {"ok_abs": ok_abs, "ok_rel": ok_rel, "ok": ok}


def abs_ok(ms: float | None) -> bool:
    """**纯绝对口径**判定（零 IO，可在台账回写热路径上调用）。"""
    return bool(ms is not None and float(ms) <= LIMIT_MS)


def stored_pass(rec: dict, key: str = "pass_t4") -> bool:
    """从**实测记录**里取 T4 判定：有 `pass_t4`（= 绝对 OR 同会话相对）就用它，
    缺该字段（老记录）才退回绝对口径。

    🔴 消费者（`auto_promote_watch` / `board_t4_watch` 的台账同步）**必须**走这里：
    自己重算 `ms <= LIMIT_MS` 就等于把「同会话相对口径」的结果整个丢掉
    （2026-11-19 15:27 / 15:30 两次现场事故）。
    """
    if not isinstance(rec, dict):
        return False
    pv = rec.get(key)
    if pv is not None:
        return bool(pv)
    return abs_ok(rec.get("ms"))


def pool_median(xs: list) -> float:
    """样本池中位数（奇数取中位、偶数取中间两数均值）。"""
    s = sorted(float(x) for x in (xs or []))
    n = len(s)
    if n == 0:
        raise ValueError("pool_median: 空样本池")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def merge_session(pools: dict, all_ms: dict, ref_ms: float | None = None,
                  ref_key: str = "") -> list:
    """把一次会话里**所有臂**的读数并进各自样本池 ⇒ 返回被更新的臂键列表。

    🟢🟢 2026-11-19 15:30：**同时把「同会话相对基线的差值」另存一池**（键后缀 `|d`）
      ⇒ 板卡整体漂移（换板 / 他任务占用）被差值抵消 ⇒ 判定不再误杀同图候选。

    🔴🔴 2026-11-20：**差值池必须带参照身份**（键后缀 `|d@{参照臂}`）——
      基线换人后，旧参照的差值与新参照的差值若混在一池，中位数**没有意义**
      （判定数值本身变垃圾）。`ref_key` = 参照臂身份（形如 `1990055_ep5`）。
      ⚠️ 旧格式（`|d`）的池**不再被读取**（历史实测：`ref_ms` 恒 None ⇒ 从未产生过差值池，
      所以没有数据损失）。
    """
    touched: list = []
    for ak, av in (all_ms or {}).items():
        m = ARM_RE.match(str(ak).strip())
        if not m:
            continue
        key = "%s_ep%s" % (m.group(1), m.group(2))
        pools.setdefault(key, []).append(float(av))
        touched.append(key)
        if ref_ms is not None:
            dkey = key + ("|d@" + ref_key if ref_key else "|d")
            pools.setdefault(dkey, []).append(round(float(av) - float(ref_ms), 4))
    return touched
