"""基线守卫：每 60s 校验台账 baseline_v2 == **源码常量** BASE_V2（trial/指标）
   ⇒ 并发写把新基线回滚时立刻补回（升轨的最后一公里不能丢）。

🔴🔴🔴 2026-11-20 **根因修复**（现场事故：守卫把 v3.1 撕成「v3.0 ACTIVE / v3.1 SUPERSEDED」）
   旧版第 3 行 `import sys, time, os, re as _re, ast as _ast` —— **漏了 `import io`**，
   而 `_read_const()` 里用的是 `io.open(NS_SRC, ...)` ⇒ 每轮都抛 NameError ⇒ 被
   `except Exception: return None` **静默吞掉** ⇒ 回退到 `ns.BASE_V2`
   （= **进程启动那一刻** import 进内存的旧常量 = v3.0 / trial_2020035）⇒
   守卫每 60 s 把台账 baseline_v2 / baseline_registry 写成 v3.0，并与 auto_promote_watch
   来回抢（基线身份约 90% 时间是错的）。

   连带损伤（已核实）：
     · `board_t4_watch.base_ref_ms()` 读台账 `baseline_v2.ms_new_board`
       ⇒ 回滚后 T4 **相对口径容差** `diff_tol() = 19.989 − base_ref_ms()` 用错参照；
     · `baseline_registry` 里 v3.0 被标 ACTIVE、v3.1 被标 SUPERSEDED（撕裂态）；
     · 守卫日志每分钟一条「已补回 Baseline_v3.0」。

   本版修法（逐条对应事故）：
     ① **彻底删除 `ns.BASE_V2` 回退** —— 读不到源码常量就**本轮不写台账**并显式报错，
        绝不用可能陈旧的内存常量去覆盖台账（宁可基线身份错 60 s，也不能反向回滚新基线）；
     ② 用 `ast` 显式解析**实际被 import 的那个** `nano_search.py`（`ns.__file__`）
        ⇒ 常量永远与源码同步，不再依赖进程启动时刻；
     ③ **registry 修复独立于 baseline_v2 比对** —— 否则 baseline_v2 恰好正确时，
        撕裂的 registry 不会被修；
     ④ 写回时保留 `promoted_at` / `previous_baseline`
        ⇒ 满足 `_merge_disk_ledger` 的「promoted_at 更晚者胜」合并规则，不破坏并发事务；
     ⑤ sidecar `.baseline_v2_meta.json` 记住「最后一次已验证正确」的快照
        ⇒ 反复回滚/补写不丢 promoted_at；
     ⑥ 新增 `--once` / `--status` 模式 ⇒ 可无副作用自检。

用法：
    python -u baseline_guard.py                       # 常驻，跑到默认截止
    python -u baseline_guard.py "2026-11-23 08:30"    # 常驻，自定义截止
    python -u baseline_guard.py --once                # 只跑一轮（自检）
    python -u baseline_guard.py --status              # 只打印现状，**不写盘**
"""
import argparse
import ast
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.environ.get("NPU_PROJECT_DIR", "/tmp/pycharm_project_10ae9e2e"), _REPO):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import manu.pipeline.npu.nano_search as ns  # noqa: E402

W = "/mnt/data/siping/runs/npu20ms/nano_search"
LOG = os.path.join(W, "baseline_guard.log")
META = os.path.join(W, ".baseline_v2_meta.json")
DEFAULT_END = "2026-11-23 08:30"
# ★ 解析**实际被 import 的** nano_search.py ⇒ 常量永远与源码一致（不再硬编码路径）
NS_SRC = os.path.abspath(getattr(ns, "__file__", "") or "")

_PARSE_ERR = ""


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%F %T"), msg))
    except Exception:
        pass


def _read_consts():
    """实时从**源码**读 BASE_V2 / BASE_V2_CKPT 字面量（不依赖已 import 的模块常量）。

    返回 dict(BASE_V2={...}, BASE_V2_CKPT="...")；读不到返回 None。
    """
    global _PARSE_ERR
    if not NS_SRC or not os.path.isfile(NS_SRC):
        _PARSE_ERR = "源码不存在: %r" % NS_SRC
        return None
    try:
        with open(NS_SRC, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as exc:  # noqa: BLE001
        _PARSE_ERR = "%s: %r" % (type(exc).__name__, exc)
        return None

    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgts = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "BASE_V2" in tgts and isinstance(node.value, ast.Call):
            kv = {}
            for kw in node.value.keywords:
                if kw.arg is None:
                    continue
                try:
                    kv[kw.arg] = ast.literal_eval(kw.value)
                except Exception:  # noqa: BLE001  —— 非常量表达式（如 BASE_V2_CKPT）跳过
                    continue
            out["BASE_V2"] = kv
        elif "BASE_V2_CKPT" in tgts:
            try:
                out["BASE_V2_CKPT"] = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                pass
    if not out.get("BASE_V2") or "trial" not in out["BASE_V2"]:
        _PARSE_ERR = "源码里没找到 BASE_V2 字面量: %s" % NS_SRC
        return None
    return out


def _want():
    """返回**源码真值**（含 ckpt）；读不到返回 None —— 绝不用 ns.BASE_V2 兜底。"""
    consts = _read_consts()
    if not consts:
        return None
    want = dict(consts["BASE_V2"])
    ckpt = consts.get("BASE_V2_CKPT")
    if ckpt:
        want["ckpt"] = ckpt
    return want


def _load_meta():
    try:
        with open(META, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_meta(data):
    try:
        tmp = META + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, META)
    except Exception as exc:  # noqa: BLE001
        log("meta 写入失败(忽略): %r" % exc)


def _build_new(want, cur, meta):
    """构造要写回的 baseline_v2：常量取自源码，promoted_at/previous_baseline 沿用已知正确的快照。"""
    new = dict(want)
    src = None
    if str(meta.get("trial") or "") == str(want.get("trial")):
        src = meta
    elif str(cur.get("trial") or "") == str(want.get("trial")):
        src = cur
    new["promoted_at"] = (src or {}).get("promoted_at") or time.strftime("%F %T")
    prev = (src or {}).get("previous_baseline")
    if not prev:
        prev = {"name": cur.get("name"), "trial": cur.get("trial"),
                "epoch": cur.get("epoch"), "nano": cur.get("nano")}
    new["previous_baseline"] = prev
    return new


def _registry_ok(reg, want):
    reg = reg or []
    if not reg:
        return False
    head = reg[0]
    if str(head.get("trial")) != str(want.get("trial")):
        return False
    if not str(head.get("status") or "").startswith("ACTIVE"):
        return False
    return not any(str(x.get("status") or "").startswith("ACTIVE") for x in reg[1:])


def _repair_registry(reg, want, note):
    """当前基线排首位且 ACTIVE；其余 ACTIVE 条目降级为 SUPERSEDED（保留历史 RETIRED 等状态）。"""
    old = [x for x in (reg or []) if str(x.get("trial")) == str(want.get("trial"))]
    keep_note = old[0].get("note") if old else note
    reg = [x for x in (reg or []) if str(x.get("trial")) != str(want.get("trial"))]
    for x in reg:
        if str(x.get("status") or "").startswith("ACTIVE"):
            x["status"] = "SUPERSEDED / DEPRECATED BASELINE"
    reg.insert(0, {"id": want.get("name"), "trial": want.get("trial"),
                   "status": "ACTIVE / OFFICIAL BASELINE",
                   "note": keep_note or note})
    return reg


def _sync_once():
    """跑一轮：校验 + 必要时修复。返回 (状态字符串, 是否写盘)。"""
    want = _want()
    if want is None:
        return ("❌ 源码常量读取失败 ⇒ 本轮**不写台账**（绝不用陈旧常量回滚）: %s" % _PARSE_ERR), False

    led = ns.load_ledger()
    cur = led.get("baseline_v2") or {}
    meta = _load_meta()
    reg = led.get("baseline_registry") or []

    base_ok = (str(cur.get("trial")) == str(want.get("trial"))
               and abs(float(cur.get("t1") or 0) - float(want.get("t1"))) < 1e-9)
    reg_ok = _registry_ok(reg, want)
    wrote = False

    if not base_ok:
        led["baseline_v2"] = _build_new(want, cur, meta)
        log("基线被并发写回滚 => 已补回 %s(trial=%s)" % (want.get("name"), want.get("trial")))
        wrote = True
    if not reg_ok:
        led["baseline_registry"] = _repair_registry(
            reg, want, "基线守卫自动补写 " + time.strftime("%F %T"))
        log("registry 修复 => %s ACTIVE / 旧项 SUPERSEDED" % want.get("name"))
        wrote = True
    if wrote:
        ns.save_ledger(led)

    final = led.get("baseline_v2") or {}
    _save_meta({"trial": want.get("trial"), "name": want.get("name"),
                "promoted_at": final.get("promoted_at"),
                "previous_baseline": final.get("previous_baseline"),
                "saved_at": time.strftime("%F %T")})
    head = (led.get("baseline_registry") or [{}])[0]
    return ("✅ 基线 %s(trial=%s) 台账一致=%s registry=%s/%s"
            % (want.get("name"), want.get("trial"), base_ok,
               head.get("id"), head.get("status"))), wrote


def _status():
    want = _want()
    led = ns.load_ledger()
    cur = led.get("baseline_v2") or {}
    reg = led.get("baseline_registry") or []
    print("源码常量 : %s" % (json.dumps(want, ensure_ascii=False) if want else "❌ 解析失败: %s" % _PARSE_ERR))
    print("台账基线 : %s(trial=%s) t1=%s promoted_at=%s"
          % (cur.get("name"), cur.get("trial"), cur.get("t1"), cur.get("promoted_at")))
    for x in reg[:4]:
        print("  registry: %-16s trial=%-9s %s" % (x.get("id"), x.get("trial"), x.get("status")))


def main():
    ap = argparse.ArgumentParser(description="基线守卫（台账 baseline_v2 / registry 与源码常量对齐）")
    ap.add_argument("end", nargs="?", default=DEFAULT_END, help="截止时间 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--once", action="store_true", help="只跑一轮")
    ap.add_argument("--status", action="store_true", help="只打印现状，不写盘")
    args = ap.parse_args()

    if args.status:
        _status()
        return
    if args.once:
        msg, wrote = _sync_once()
        print(msg)
        log("单轮自检: %s (写盘=%s)" % (msg, wrote))
        return

    end = args.end
    log("基线守卫启动: END=%s NS_SRC=%s" % (end, NS_SRC))
    fail_streak = 0
    while time.strftime("%Y-%m-%d %H:%M") < end:
        try:
            msg, _ = _sync_once()
            if msg.startswith("✅"):
                fail_streak = 0
            else:
                fail_streak += 1
                log(msg)
                if fail_streak >= 10:
                    log("❌ 连续 10 轮读不到源码常量 ⇒ 守卫退出，需人工检查 %s" % NS_SRC)
                    break
        except Exception as exc:  # noqa: BLE001
            log("守卫异常(忽略): %r" % exc)
        time.sleep(60)
    log("基线守卫到期退出")


if __name__ == "__main__":
    main()

