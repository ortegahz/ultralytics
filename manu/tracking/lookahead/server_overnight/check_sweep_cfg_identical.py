#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""核对两次 24 序 sweep 回显的 **CFG 字符串是否逐 token 相同**。

为什么必须查这个：
  ``arms.json`` 的 ``lookahead`` 元数据里**没有** ``replay_abstain_peak_below`` /
  ``replay_abstain_disp_px`` 这两个交付档旋钮 ⇒ **产物层无法自证 CFG 一致**。
  而「唯一变量 = {模型包, Δ}」的论证要求 CFG 完全相同
  ⇒ 只能回到两次 sweep 的日志回显（``CFG（源自 run_delivery.sh，N 字符）: ...``）做逐字比对。

用法（服务器）：
    python manu/tracking/lookahead/server_overnight/check_sweep_cfg_identical.py \
        --rc2-log runs/lookahead/fi_nogmc24/sweep.log \
        --cs-log  runs/lookahead/nogmc24_pipeline.log \
        --out-json runs/lookahead/nogmc24_cfg_identical.json
    # 加 --wait-for runs/lookahead/fi_nogmc24/cs_t0043/d3/arms.json 可先等出数再核
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

CFG_RE = re.compile(r"CFG（源自 run_delivery\.sh，\s*(\d+)\s*字符）:\s*(.+)")


def extract_cfg(log_path: Path) -> list:
    """从日志里抽出所有 CFG 回显行（可能有多次运行）。"""
    if not log_path.is_file():
        return []
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CFG_RE.search(line)
        if match:
            out.append({"declared_len": int(match.group(1)), "cfg": match.group(2).strip()})
    return out


def tokenize(cfg: str) -> list:
    """把 CFG 拆成 (旗标, 值) 序列，保留顺序。"""
    tokens = cfg.split()
    pairs = []
    for idx in range(0, len(tokens) - 1, 2):
        flag, value = tokens[idx], tokens[idx + 1]
        pairs.append([flag, value])
    if len(tokens) % 2 == 1:
        pairs.append([tokens[-1], "<无值>"])
    return pairs


def compare(rc2: list, cs: list) -> dict:
    """逐 token 比对两份 CFG。"""
    if not rc2 or not cs:
        return {
            "verdict": "MISSING",
            "detail": f"rc2 侧读到 {len(rc2)} 条 CFG 回显，cs 侧读到 {len(cs)} 条 ⇒ 无法比对",
        }
    a, b = rc2[-1], cs[-1]
    ta, tb = tokenize(a["cfg"]), tokenize(b["cfg"])
    diffs = []
    for idx in range(max(len(ta), len(tb))):
        left = ta[idx] if idx < len(ta) else None
        right = tb[idx] if idx < len(tb) else None
        if left != right:
            diffs.append({"index": idx, "rc2": left, "cs": right})
    same_string = a["cfg"] == b["cfg"]
    return {
        "verdict": "IDENTICAL" if (same_string and not diffs) else "DIFFERENT",
        "exact_string_equal": same_string,
        "rc2_declared_len": a["declared_len"],
        "cs_declared_len": b["declared_len"],
        "rc2_actual_len": len(a["cfg"]),
        "cs_actual_len": len(b["cfg"]),
        "rc2_token_count": len(ta),
        "cs_token_count": len(tb),
        "token_diffs": diffs,
        "rc2_cfg": a["cfg"],
        "cs_cfg": b["cfg"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="核对两次 24 序 sweep 的 CFG 是否逐 token 相同")
    parser.add_argument("--rc2-log", default="runs/lookahead/fi_nogmc24/sweep.log")
    parser.add_argument("--cs-log", default="runs/lookahead/nogmc24_pipeline.log")
    parser.add_argument("--out-json", default="runs/lookahead/nogmc24_cfg_identical.json")
    parser.add_argument("--wait-for", default="", help="先等这个文件出现再核（留空＝不等）")
    parser.add_argument("--wait-timeout", type=int, default=14400)
    args = parser.parse_args()

    if args.wait_for:
        target = Path(args.wait_for)
        deadline = time.time() + args.wait_timeout
        while not target.exists() and time.time() < deadline:
            time.sleep(30)
        print(f"[WAIT] {target} {'已就绪' if target.exists() else '等待超时'}")

    rc2 = extract_cfg(Path(args.rc2_log))
    cs = extract_cfg(Path(args.cs_log))
    result = compare(rc2, cs)
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")
    result["rc2_log"] = args.rc2_log
    result["cs_log"] = args.cs_log

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CFG-CHECK] verdict = {result['verdict']}")
    if result.get("token_diffs"):
        print(f"[CFG-CHECK] token 差异 {len(result['token_diffs'])} 处：")
        for diff in result["token_diffs"][:20]:
            print(f"   #{diff['index']}: rc2={diff['rc2']} cs={diff['cs']}")
    print(f"[CFG-CHECK] 写出 {out}")
    if result["verdict"] != "IDENTICAL":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
