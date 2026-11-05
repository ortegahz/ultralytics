#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""证明「一次 sweep 里的全部运行共用同一份插件 CFG」。

`check_sweep_cfg_identical.py` 比的是**两份日志**的 CFG 是否一致；
本脚本证明的是另一件事：**一次 sweep 内**（同一脚本、同一份 `$CFG` 变量）
跑出的全部 (模型包 × Δ) 运行**共用同一份 CFG** —— 这是「唯一变量 = 模型包」的前提。

证据链（缺一不可）：
  ① 日志里 CFG 回显**只出现 1 次**（扫描脚本在模型/Δ 双层循环**之外** echo 一次 `$CFG`）；
  ② 每条 `RUN 模型=X Δ=Y` 都发生在那次回显**之后**；
  ③ 扫描脚本把**同一个 `$CFG` 变量**传给每次运行（无逐次覆盖）;
  ④ 该 CFG 由 `run_delivery.sh` 的 `CFG=` 整块 eval 得到，且脚本自带必需旋钮断言。

    python manu/tracking/lookahead/server_overnight/check_sweep_cfg_shared.py \
        --sweep-log runs/lookahead/fi_nogmc/sweep.log \
        --scanner manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc_sweep.sh \
        --out-json runs/lookahead/fi_nogmc/cfg_shared_8seq.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

CFG_RE = re.compile(r"CFG（源自[^）]*，(\d+) 字符）:\s*(.+)")
RUN_RE = re.compile(r"===== RUN 模型=(\S+) Δ=(\d+) =====")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="一次 sweep 内全部运行共用同一份 CFG 的证明")
    p.add_argument("--sweep-log", default="runs/lookahead/fi_nogmc/sweep.log")
    p.add_argument("--scanner", default="manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc_sweep.sh")
    p.add_argument("--out-json", default="runs/lookahead/fi_nogmc/cfg_shared_8seq.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path, scanner_path = Path(args.sweep_log), Path(args.scanner)
    issues: list[str] = []

    echoes: list[tuple[int, int, str]] = []  # (行号, 声明长度, CFG 串)
    runs: list[tuple[int, str, int]] = []  # (行号, 模型包, Δ)
    if log_path.is_file():
        for lineno, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = CFG_RE.search(line)
            if m:
                echoes.append((lineno, int(m.group(1)), m.group(2).strip()))
            r = RUN_RE.search(line)
            if r:
                runs.append((lineno, r.group(1), int(r.group(2))))
    else:
        issues.append(f"扫描日志不存在：{log_path}")

    if len(echoes) != 1:
        issues.append(f"CFG 回显次数 = {len(echoes)}（应为 1）⇒ 无法证明「全部运行共用同一份 CFG」")
    else:
        echo_line, declared_len, cfg = echoes[0]
        if declared_len != len(cfg):
            issues.append(f"CFG 声明长度 {declared_len} ≠ 实际长度 {len(cfg)} ⇒ CFG 回显被截断")
        for lineno, model, delta in runs:
            if lineno < echo_line:
                issues.append(f"第 {lineno} 行的运行（{model} Δ={delta}）早于第 {echo_line} 行的 CFG 回显")
        if scanner_path.is_file():
            src = scanner_path.read_text(encoding="utf-8", errors="replace")
            if "$CFG" not in src:
                issues.append(f"扫描脚本 {scanner_path} 里找不到 `$CFG` 变量")
            if not re.search(r"--delta\s+\"\$d\"\s+\$CFG", src):
                issues.append("扫描脚本的运行命令行里没有 `--delta \"$d\" $CFG` ⇒ 无法证明复用同一份 CFG")
        else:
            issues.append(f"扫描脚本不存在：{scanner_path}")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sweep_log": str(log_path),
        "scanner": str(scanner_path),
        "cfg_echo_count": len(echoes),
        "cfg_declared_len": echoes[0][1] if len(echoes) == 1 else None,
        "cfg_actual_len": len(echoes[0][2]) if len(echoes) == 1 else None,
        # 与 `check_sweep_cfg_identical.py` 同口径：旗标/值成对计数（奇数尾巴算一对）
        "cfg_token_pair_count": (len(echoes[0][2].split()) + 1) // 2 if len(echoes) == 1 else None,
        "run_count": len(runs),
        "runs": [{"model": m, "delta": d} for _, m, d in runs],
        "models": sorted({m for _, m, _ in runs}),
        "deltas": sorted({d for _, _, d in runs}),
        "verdict": "SHARED" if not issues else "UNPROVEN",
        "issues": issues,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{'OK' if not issues else 'WARN'}] sweep CFG 共享性：{result['verdict']} → {out}")
    for issue in issues:
        print(f"  - {issue}")
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
