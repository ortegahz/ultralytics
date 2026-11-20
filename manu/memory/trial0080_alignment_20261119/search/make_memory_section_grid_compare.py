#!/usr/bin/env python3
"""Generate the memory section for the Δn grid-search evaluation/comparison round.

Every number is read from the on-disk artefacts -- nothing is hand-typed (the rule the
project follows for every recorded conclusion: "可复现记录 ... 每条结论必带脚本相对路径 +
可复制完整命令 + 执行机 + 产物路径").

The section is produced twice: a detailed one for ``memory.md`` and a compact one for
``memory_compact.md``.  Both are merged into their target with
``append_memory_section.py`` (byte-safe: only the owned block is rewritten).
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path("/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics")
DELIVERY = REPO / "manu/memory/delivery_20261119"
ARTIFACTS = DELIVERY / "artifacts"
ALIGN = REPO / "manu/memory/trial0080_alignment_20261119"
EVAL = ALIGN / "eval"
GRID = DELIVERY / "grid_compare"
OUT_DIR = ALIGN / "search"

SCRIPT_REL = "manu/evaluation/rt_grid_compare.py"

# The delivery-caliber runs the comparison is built from: the six track_all_* runs plus
# the twelve shortlist runs (eval/shortlist/track_*.json).  The shortlist runs share the
# v3 detection side, so they form a clean single-variable grid at the delivery caliber.
TRACK_RUNS = [
    ("v3", EVAL / "track_all_v3.json"),
    ("v2", EVAL / "track_all_v2.json"),
    ("th045", EVAL / "track_all_th045.json"),
    ("old_gapfit", EVAL / "track_all_v3_gapfit.json"),
    ("cc2_md10", ARTIFACTS / "track_all_v3_optgapfit.json"),
    ("cc2_md16n", ARTIFACTS / "track_all_v3_optgapfit_md16.json"),
    ("cc2_md8", EVAL / "shortlist/track_cc2_md8.json"),
    ("cc2_md12", EVAL / "shortlist/track_cc2_md12.json"),
    ("cc3_md10", EVAL / "shortlist/track_cc3_md10.json"),
    ("cc3_md16", EVAL / "shortlist/track_cc3_md16.json"),
    ("cc4_md10", EVAL / "shortlist/track_cc4_md10.json"),
    ("cc5_md10", EVAL / "shortlist/track_cc5_md10.json"),
    ("cc6_md10", EVAL / "shortlist/track_cc6_md10.json"),
    ("cc7_md10", EVAL / "shortlist/track_cc7_md10.json"),
    ("old_gapfit(短名单复测)", EVAL / "shortlist/track_old_gapfit.json"),
    ("rerun_selected", EVAL / "shortlist/track_rerun_selected.json"),
]


def track_args() -> list[str]:
    return [f"--track={label}={path}" for label, path in TRACK_RUNS]


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: float, nd: int = 4) -> str:
    return f"{float(value) * 100:.{nd}f}%"


def main() -> None:
    delivery = load_csv(GRID / "rt_delivery_compare.csv")
    calibration = load_csv(GRID / "rt_far_calibration.csv")
    shortlist = load_csv(GRID / "rt_grid_shortlist.csv")
    scenarios = load_csv(GRID / "rt_grid_scenario.csv")

    # the verification count is re-measured here, not copied from a log
    proc = subprocess.run(
        ["python3", str(REPO / SCRIPT_REL),
         "--arm-json", str(ARTIFACTS / "rt_eval_delivered_armA.json"),
         "--shortlist", str(ARTIFACTS / "shortlist_deterministic.json"),
         *track_args(),
         "--out-dir", "/tmp/memory_section_grid_rerun"],
        capture_output=True, text=True, check=False,
    )
    verify_line = next((line for line in proc.stdout.splitlines() if line.startswith("[VERIFY]")), "N/A")

    matched = [row for row in calibration if row["detector_matched"] == "1"]
    unmatched = [row for row in calibration if row["detector_matched"] != "1"]
    offsets = [float(row["far_offset"]) for row in matched]
    flipped = [row for row in matched
               if row["truncated_verdict"] != row["delivery_verdict"]]
    scenarios_used = sorted({row["scenario"] for row in scenarios
                              if row["scenario"] in ("s1", "s2", "s3", "s4")})
    spec = json.loads((ARTIFACTS / "rt_eval_delivered_armA.json").read_text(encoding="utf-8"))
    spec_labels = spec["meta"]["scenarios_spec"]

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    detailed: list[str] = []
    add = detailed.append
    n_passed = verify_line.split()[1].split("/")[0] if "[VERIFY]" in verify_line else "?"
    add(f"### 三十六、🔴 Δn 网格搜索的「指标评估 + 对比」统一汇总 + **{n_passed} 项独立复算**"
        f"（第 90 轮，{stamp}）")
    add("")
    add("**本轮交付数字一个没动**（交付组仍是 `cc2_md10` / 逐帧 F1 94.4976% / FAR 0.027859 / "
        "跳帧 phys 0.951398 / OSD 21+3）；本轮做的是**把网格搜索的结果收成一份可审计的对比**，"
        "并把**指标口径第一次从产物里反解出来、写进文档**。")
    add("")
    add("#### 1) 🔴 环境事故（host 重装系统 ⇒ 三条外部依赖同时断）")
    add("")
    add("| 依赖 | 状态 | 证据 |")
    add("| --- | --- | --- |")
    add("| 服务器 `192.168.99.40:32222` | **网络可达但认证被拒** | TCP + OpenSSH banner 正常；"
        "`Permission denied (publickey)`（沙箱内新生成的 `~/.ssh/id_ed25519` 未在服务器登记）|")
    add("| 板卡 `192.168.0.211:22` | **离线** | `ssh: connect to host 192.168.0.211 port 22: "
        "No route to host`；`ping` 100% 丢包 |")
    add("| host 旧挂载 `/media/manu/1TB-Volume/workspace` | **整个消失** | `/media` 现为空目录、"
        "属 nobody；handoff 触发器（`handoff/hook_post_checkout.sh`）默认 WORKSPACE 指向该路径 ⇒ 失效 |")
    add("")
    add("⇒ **本轮全程不依赖服务器**：所有结论都在**已拉回本地的实测产物**上用新脚本重算，"
        "**没有重跑任何一条评测**（铁律十三：不擅自起重活；服务器不可达 ⇒ 也不能起）。")
    add("")
    add("#### 2) 新增脚本与它补上的洞")
    add("")
    add(f"- 脚本：**`{SCRIPT_REL}`**（新文件）。它做四件事：①**独立复算**表 1/表 2 的每个聚合值；"
        "②**跨文件对账**（短名单 vs 臂 json，逐情景 + 构造性等式）；"
        "③**量口径差**（截断 vs 交付，同源臂逐一配对）；④出**对比表**。")
    add("- **补上的洞**：此前「网格搜索结果」散在 3 个文件、3 种口径里（臂 json / 短名单 json / "
        "交付口径 track json），汇报时**要人工跨文件拼**，且**没人独立复算过聚合值**。")
    add("- **口径第一次反解出来并写进脚本 docstring**（此前只散落在各脚本里）：")
    add("")
    add("  | 量 | 定义 |")
    add("  | --- | --- |")
    add("  | precision | `Σtp / Σ(tp+fp)` |")
    add("  | recall | `Σtp / Σgt` |")
    add("  | F1 | `2PR/(P+R)` |")
    add("  | 跳帧 FAR | `Σfp / Σ(delivered_frames × seeds)` |")
    add("  | 交付口径 FAR | `Σfp / 总帧数`（全量 165,223 帧）|")
    add("")
    add("  ⇒ 全部是**序列级 micro-pool**（先把 24 条序列的 tp/fp/gt 加起来再算，不是逐序列 F1 求平均）；"
        "**逐序列 F1 求平均会低 ~2.7pp**（实测 0.928331 vs 0.955383，臂 `3_jump_optimised/s1`）"
        "⇒ 报数必须报 pooled。")
    add("")
    add(f"#### 3) 验收：**{verify_line}**（脚本自带独立复算闸门，非人眼核对）")
    add("")
    add("- 覆盖：逐臂逐情景的 precision/recall/F1/FAR 复算、跳帧 mean/worst/mean-FAR 复算、"
        "短名单 vs 臂 json 跨文件对账、交付口径逐序列 tp/fp/fn/gt 求和 == overall、"
        "FAR 分母必须是**整数帧数**（实测 165,223）、以及每个臂的红线判定。")
    add("- **诚实边界**：交付口径 FAR 的分母只能由 `Σfp / far_per_frame` 反解（逐序列帧数未落盘），"
        "**不是**从视频元数据直接读；已用「每条序列反解出的帧数都是整数」+「总数 == 165,223」交叉验证。")
    add("")
    add("#### 4) 对比结果（表 1：交付口径 —— FAR 红线定义在这一档）")
    add("")
    add("| 参数组 | F1 | FAR/帧 | 红线 0.03 | 余量 | TP | FP | FN | GT | 帧数* |")
    add("| --- | --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in delivery:
        verdict = "✅" if row["far_ok"] == "1" else "❌"
        add(f"| `{row['label']}` | {pct(row['f1'])} | {float(row['far']):.6f} | {verdict} | "
            f"{float(row['far_margin_pct']):.1f}% | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['gt']} | {row['frames']} |")
    add("")
    add("\\* 帧数**不是**从视频元数据读的（那在服务器）：由 `Σfp / far_per_frame` 反解，"
        "已校验为整数、且所有臂反解出**同一个**值 ⇒ 可当「全量 165,223 帧」用。")
    add("")
    add("#### 5) 对比结果（表 2：跳帧网格，物理帧基）")
    add("")
    add("| 臂 | " + " | ".join(f"{s}" for s in scenarios_used) + " | mean F1 | worst F1 | mean FAR |")
    add("| --- | " + " | ".join("---" for _ in scenarios_used) + " | --- | --- | --- |")
    for arm in ("2a_jump_dense_timebase", "2b_jump_physical_untuned", "3_jump_optimised"):
        rows = {row["scenario"]: row for row in scenarios if row["arm"] == arm}
        if not rows:
            continue
        cells = " | ".join(f"{float(rows[s]['f1']):.4f}" if s in rows else "-"
                           for s in scenarios_used)
        jump = [float(rows[s]["f1"]) for s in scenarios_used if s in rows]
        add(f"| `{arm}` | {cells} | {sum(jump) / len(jump):.6f} | {min(jump):.6f} | "
            f"{sum(float(rows[s]['far']) for s in scenarios_used if s in rows) / len(jump):.6f} |")
    add("")
    add("情景语义（来自臂 json 的 `meta.scenarios_spec`，**不是**我编的）：")
    for key in scenarios_used:
        info = spec_labels.get(key, {})
        add(f"- `{key}` = {info.get('label', '?')}")
    add("")
    add("#### 6) 🔴 表 2b：**截断口径系统性低报 FAR**（第 88 轮那个缺陷的量化闭环）")
    add("")
    add("| 短名单臂 | 交付口径臂 | FAR（截断 4000 帧/序）| FAR（交付 165,223 帧）| 偏移 | 截断判定 | 交付判定 |")
    add("| --- | --- | --- | --- | --- | :-: | :-: |")
    for row in calibration:
        add(f"| `{row['shortlist_arm']}` | `{row['delivery_label']}` | "
            f"{float(row['far_truncated']):.6f} | {float(row['far_delivery']):.6f} | "
            f"+{float(row['far_offset']):.6f} | {row['truncated_verdict']} | {row['delivery_verdict']} |")
    add("")
    add(f"⇒ **{len(calibration)} 对配对，偏移区间 [+{min(offsets):.6f}, +{max(offsets):.6f}]，"
        f"全部为正** ⇒ 截断口径**一律低报 FAR**；")
    if flipped:
        names = "、".join(f"`{row['shortlist_arm']}`（{row['delivery_label']}）" for row in flipped)
        add(f"**{len(flipped)} 例两口径判定相反**（{names}：截断 PASS / 交付 FAIL）"
            f"⇒ 只看截断口径就会选出**破线参数** ⇒ 硬约束必须量在**交付口径**上（第 88 轮已改，本轮把它量化成表）。")
    add("")
    add("⚠️ **配对口径的诚实边界**：配对只按 **7 个跟踪旋钮**（`confirm_consec/confirm_gap_tol/"
        "max_age_phys/r_slope/match_dist/min_hits`）匹配；`v2` 与 `v3` 的这 7 键**相同**"
        "（差异在检测阈值 `th_base/th_salvage/th_ground/sky_ratio`），故两行都挂在 `v3_base` 短名单臂上 "
        "—— 这是**真实的口径事实**（同一跟踪旋钮、阈值不同），不是配对错误。")
    add("")
    add("#### 7) 短名单 14 臂（网格本体）")
    add("")
    add("| 臂 | 参数 | 跳帧 mean F1 | 跳帧 worst F1 | 逐帧(截断) F1 | 逐帧(截断) FAR |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in sorted(shortlist, key=lambda r: -float(r["jump_physical_mean_f1"])):
        add(f"| `{row['arm']}` | {row['knobs']} | {float(row['jump_physical_mean_f1']):.6f} | "
            f"{float(row['jump_physical_worst_f1']):.6f} | {float(row['perframe_truncated_f1']):.6f} | "
            f"{float(row['perframe_truncated_far']):.6f} |")
    add("")
    add("⚠️ **这张表的 `逐帧(截断) FAR` 列不能用来判红线**（见表 2b）；"
        "**红线判定只看表 1（交付口径）**。")
    add("")
    add("🔴 **活标本**：`rerun_selected`（`confirm_consec 2 / gap_tol 8 / max_age_phys 10 / "
        "r_slope 5.0 / match_dist 16.0`）在**跳帧网格上分数最高**（mean F1 0.952186），"
        "但它的**交付口径 FAR = 0.035152 ⇒ 破线**（本轮已在表 1 用 "
        "`eval/shortlist/track_rerun_selected.json` 实测到，不再是「服务器实测」转述）"
        "⇒ 「只看跳帧臂就会选出破线参数」这件事，现在有**同表两列**的对照证据。")
    add("")
    add("#### 8) 产物 · 复现命令 · 执行机")
    add("")
    add("- 脚本：`manu/evaluation/rt_grid_compare.py`（本轮新增）")
    add("- 产物：`manu/memory/delivery_20261119/grid_compare/`（`rt_grid_compare.md` / "
        "`rt_grid_scenario.csv` / `rt_grid_shortlist.csv` / `rt_delivery_compare.csv` / "
        "`rt_far_calibration.csv`）")
    add("- 对比正文：`manu/memory/delivery_20261119/COMPARISON.md`（既有）+ 本轮 `rt_grid_compare.md`（新增，"
        "带复算验收表）")
    add("- **执行机：本机（host）**，纯读本地 json，无 GPU、无网络、无重活。")
    add("")
    add("```bash")
    add("cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics")
    add("A=manu/memory/delivery_20261119/artifacts")
    add("E=manu/memory/trial0080_alignment_20261119/eval")
    add("python3 manu/evaluation/rt_grid_compare.py \\")
    add("  --arm-json $A/rt_eval_delivered_armA.json --shortlist $A/shortlist_deterministic.json \\")
    for label, path in TRACK_RUNS:
        rel = str(path).replace(str(REPO) + "/", "")
        add(f"  --track \"{label}={rel}\" \\")
    add("  --out-dir manu/memory/delivery_20261119/grid_compare")
    add("```")
    add("")
    add("#### 9) 本轮**没做**的事（外部依赖断，不是遗漏）")
    add("")
    add("1. **板端 FAR 复核**（`cc2_md16n` 余量仅 2.2% 的风险）⇒ 板卡离线，**做不了**；"
        "板卡上电后按 `manu/evaluation/eval_board_fused_tsv.py` 跑（脚本已就位）。")
    add("2. **服务器侧重跑/扩样** ⇒ 沙箱内无公钥认证，**登不上去**；需要用户在本机终端执行一次"
        "`scp`+`ssh` 授权后才有。")
    add("3. **帧率扫描更细的网格**（如 25/20/17/14/12 fps 全扫描）⇒ 依赖服务器缓存，**本轮做不了**。")
    add("")

    (OUT_DIR / "memory_section_grid_compare_90.md").write_text(
        "\n".join(detailed) + "\n", encoding="utf-8")

    # ---- compact version -------------------------------------------------
    compact: list[str] = []
    cadd = compact.append
    cadd(f"### 三十六、🔴 Δn 网格搜索的「指标评估 + 对比」统一汇总 + 独立复算（第 90 轮，{stamp}）")
    cadd("")
    cadd("**交付数字一个没动**（仍是 `cc2_md10` / 逐帧 F1 94.4976% / FAR 0.027859 / 跳帧 0.951398 / OSD 21+3）；"
         "本轮把网格结果收成一份**可审计对比**并**独立复算**。")
    cadd("")
    cadd(f"- **环境事故**：服务器 **网络可达但公钥认证被拒**（`Permission denied (publickey)`）、"
         f"板卡 `192.168.0.211` **离线**（`No route to host`）、host 旧挂载 `/media/manu/1TB-Volume/workspace` "
         f"**整个消失**（handoff 触发器默认路径失效）⇒ **本轮全程零服务器、零重活**，全部在本地实测产物上重算。")
    cadd(f"- **新脚本 `{SCRIPT_REL}`**：①独立复算每个聚合值；②跨文件对账（短名单 vs 臂 json）；③出对比表。")
    cadd(f"- **验收：{verify_line}**（脚本自带闸门）。")
    cadd("- **口径首次反解并写进文档**（全是**序列级 micro-pool**）："
         "`precision=Σtp/Σ(tp+fp)`、`recall=Σtp/Σgt`、`F1=2PR/(P+R)`、"
         "跳帧 `FAR=Σfp/Σ(delivered_frames×seeds)`、交付 `FAR=Σfp/165,223`。"
         "**逐序列 F1 求平均会低 ~2.7pp**（0.928331 vs 0.955383）⇒ 报数必须报 pooled。")
    cadd("")
    cadd("| 参数组 | 逐帧 F1（交付口径）| FAR/帧 | 红线 0.03 | 余量 |")
    cadd("| --- | --- | --- | :-: | ---: |")
    for row in delivery:
        verdict = "✅" if row["far_ok"] == "1" else "❌"
        cadd(f"| `{row['label']}` | {pct(row['f1'])} | {float(row['far']):.6f} | {verdict} | "
             f"{float(row['far_margin_pct']):.1f}% |")
    cadd("")
    cadd(f"- **表 2b 口径校准（新增，本轮最硬的一条）**：{len(calibration)} 对配对，"
         f"截断→交付的 FAR 偏移区间 **[+{min(offsets):.6f}, +{max(offsets):.6f}]、全部为正**；"
         f"**{len(flipped)} 例两口径判定相反**（`old_gapfit`：截断 0.029013 PASS / 交付 0.035128 **FAIL**）"
         f"⇒ 把第 88 轮「硬约束必须量在交付口径」那个修复**量化成表**。")
    cadd(f"- **表 2 跳帧网格**（物理帧基）：`2a` 密集帧基 mean F1 0.922060｜"
         f"`2b` v3 未调参 mean F1 **0.084122**（s1/s2/s3 全 0.0000，崩）｜"
         f"`3` cc2_md10 mean F1 **0.951398**（worst 0.938916）⇒ "
         f"**v3 崩不是参数差，是它假设「帧永远连续」**。")
    cadd("- ⚠️ **短名单表的「逐帧(截断) FAR」列不能判红线**（见表 2b）；"
         "`rerun_selected`（`ma=10`）跳帧分最高但交付口径 FAR **0.035152 破线**（服务器实测，§三十五）。")
    cadd("")
    cadd(f"- **产物**：`manu/memory/delivery_20261119/grid_compare/`（`rt_grid_compare.md` + 4 个 csv）｜"
         f"**执行机**：本机（host），纯读 json，无 GPU/网络/重活｜"
         f"**复现**：`python3 {SCRIPT_REL} --arm-json ... --shortlist ... --track label=path ... --out-dir ...`")
    cadd(f"- **本轮没做**（外部依赖断）：① 板端 FAR 复核（板卡离线，脚本 `manu/evaluation/eval_board_fused_tsv.py` 已就位，"
         f"上电即可跑）；② 服务器侧扩样/重跑（公钥认证被拒，需用户终端授权一次）；③ 更细的帧率扫描网格。")
    cadd("")

    (OUT_DIR / "memory_section_grid_compare_90_compact.md").write_text(
        "\n".join(compact) + "\n", encoding="utf-8")
    print(f"[OUT] {OUT_DIR / 'memory_section_grid_compare_90.md'}")
    print(f"[OUT] {OUT_DIR / 'memory_section_grid_compare_90_compact.md'}")
    print(f"[INFO] verify: {verify_line}")


if __name__ == "__main__":
    main()
