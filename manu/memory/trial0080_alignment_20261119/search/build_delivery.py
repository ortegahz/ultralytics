#!/usr/bin/env python3
"""Build the 2026-11-19 delivery package (params file + OSD manifest + comparison table)."""
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo")
RUNS = Path("/mnt/data/siping/runs/trial0080_onnx")
DEL = REPO / "manu/memory/delivery_20261119"
DEL.mkdir(parents=True, exist_ok=True)
(DEL / "artifacts").mkdir(exist_ok=True)

OSD = Path("/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_20261119")
OSD_OLD = Path("/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_gapfit_20261119")
# r88: 候选组 cc2_md16n（修复后搜索器的 argmax）的 OSD 目录 —— **不是交付 OSD**，
#      只是把「切换就绪」这件事做成可核的清单；交付 OSD 仍是上面那个目录，一个字节都没动。
OSD_CAND = Path("/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_md16_20261120")
SL = RUNS / "eval/shortlist"


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def canon_md5(doc):
    return hashlib.md5(json.dumps(doc.get("params", doc), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# 1. the new parameter file
src = REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json"
shutil.copy2(src, DEL / src.name)
params_doc = load(src)

# 2. the key artefacts
copies = {
    RUNS / "search/rt_search_t0080_v3.json": "rt_search_t0080_v3.json",
    RUNS / "search/rt_search_t0080_v3.md": "rt_search_t0080_v3.md",
    RUNS / "search/rt_eval_t0080_v3_opt.json": "rt_eval_t0080_v3_opt.json",
    RUNS / "search/rt_eval_t0080_v3_opt.md": "rt_eval_t0080_v3_opt.md",
    RUNS / "search/opt_params_t0080_v3_flat.json": "opt_params_t0080_v3_flat.json",
    RUNS / "eval/track_all_v3_optgapfit.json": "track_all_v3_optgapfit.json",
    RUNS / "eval/track_all_v3.json": "track_all_v3.json",
    RUNS / "eval/cmp_v3_optgapfit_vs_official_v3.json": "cmp_v3_optgapfit_vs_official_v3.json",
    RUNS / "eval/cmp_v3_optgapfit_vs_rebuilt_v3.json": "cmp_v3_optgapfit_vs_rebuilt_v3.json",
    RUNS / "eval/cmp_v3_rebuilt_vs_official_v3.json": "cmp_v3_rebuilt_vs_official_v3.json",
    RUNS / "search/memory_section_257_18.md": "memory_section_257_18.md",
    # r80: 紧凑档正文此前**不在** copies 里 => 它在包内是手工维护的副本，
    # 而主档却每次从 search/ 重新覆盖 => 两份正文会朝不同方向漂（同一段正文两个版本）。
    # 两份都从 search/ 取，drift 就不可能发生。
    RUNS / "search/memory_section_257_18_compact.md": "memory_section_257_18_compact.md",
    RUNS / "search/final_selection.json": "final_selection.json",
    RUNS / "search/rt_search_t0080_v3_final.json": "rt_search_t0080_v3_final.json",
    RUNS / "search/rt_search_t0080_v3_final.md": "rt_search_t0080_v3_final.md",
    RUNS / "search/rt_search_t0080_v3_seedfix.json": "rt_search_t0080_v3_seedfix.json",
    RUNS / "search/rt_search_t0080_v3_seedfix.md": "rt_search_t0080_v3_seedfix.md",
    REPO / "manu/memory/trial0080_delta_search_spec_20261119.md": "trial0080_delta_search_spec_20261119.md",
    REPO / "manu/memory/trial0080_board_fusion_preset_20261119.md": "trial0080_board_fusion_preset_20261119.md",
    # r79: 三件「被文档当证据引用、却没随包交付」的硬证据 ——
    #   ① self_check 日志 = 任务书要求的 [GATE PASS] 证明（没有它，host 只能凭空信这句话）
    #   ② 冻结单帧闸门 baseline json = 日志里 expected 数字的出处（没有它，日志不可自证）
    #   ③ 搜索空间 json = 「空间逐项覆盖任务书 7 个旋钮」的实证（没有它，这句结论不可核）
    RUNS / "eval/self_check_optgapfit.log": "self_check_optgapfit.log",
    RUNS / "search/space_t0080_v3.json": "space_t0080_v3.json",
    REPO / "manu/evaluation/system_sota_longquan_trial0080_singleframe_gate_baseline.json": "system_sota_longquan_trial0080_singleframe_gate_baseline.json",
    # r81: md 报告的渲染器（纯 stdlib）随包交付 ⇒ 任何人拿到包都能从 json 复渲 md，
    # 而闸门 §6g 用它现渲一遍逐字节比对 ⇒ md 不可能与 json 漂移。
    REPO / "manu/evaluation/render_search_report_md.py": "render_search_report_md.py",
    # r88: ① 搜索器把逐帧硬约束搬到**交付口径**后的首次搜索产物（json / md / 真实运行日志）；
    #      ② 候选组 cc2_md16n（该次搜索的 argmax）的**全量交付口径**权威评测与两份逐序列对账；
    #      ③ 该候选组自己的 [GATE PASS] 证明。
    #   ⚠️ 候选参数档 / opt_params_t0080_v3_r88_flat.json **故意不进包**：
    #      闸门 §6j① 规定包内只允许存在一份 load_params 能静默接受的参数文件（第 85 轮吃过一次亏）。
    RUNS / "search/rt_search_t0080_v3_r88.json": "rt_search_t0080_v3_r88.json",
    RUNS / "search/rt_search_t0080_v3_r88.md": "rt_search_t0080_v3_r88.md",
    RUNS / "search/rt_search_t0080_v3_r88.log": "rt_search_t0080_v3_r88.log",
    RUNS / "eval/track_all_v3_optgapfit_md16.json": "track_all_v3_optgapfit_md16.json",
    RUNS / "eval/cmp_v3_optgapfit_md16_vs_official_v3.json": "cmp_v3_optgapfit_md16_vs_official_v3.json",
    RUNS / "eval/cmp_v3_optgapfit_md16_vs_rebuilt_v3.json": "cmp_v3_optgapfit_md16_vs_rebuilt_v3.json",
    RUNS / "eval/self_check_optgapfit_md16.log": "self_check_optgapfit_md16.log",
    RUNS / "eval/osd_media_check_md16_candidate.json": "osd_media_check_md16_candidate.json",
}
for s, name in copies.items():
    if s.exists():
        shutil.copy2(s, DEL / "artifacts" / name)

# r85: 交付包内**只允许存在一份参数文件**（与交付参数逐键相等的那份）。
#   为什么这是危险品（实测取证，见 spec §10.12）：
#     ① `system_sota_longquan.py::load_params` 读的是 `payload.get("params", payload)`
#        ⇒ 嵌套 `{"params": {...}}` 与顶层扁平两种形态**都被静默接受** —— 喂错文件**不报错**，
#        直接跑出一份「看起来正常」的评测结果；
#     ② 交付包里曾躺着 run2/run3 的 objective-best 扁平参数（match_dist 16.0 / max_age_phys 10
#        ⇒ 全量口径 FAR 0.035152 > 0.03 红线），文件名里的 "final" 读起来像权威版；
#     ③ 板卡施工单 §3 正要把 match_dist 16.0→10.0 / max_age_phys 10→null 改掉 ——
#        误取该文件 = 把**已被拒绝**的参数同步上板。
#   ⇒ 这里主动剔除任何「有效参数 ≠ 交付参数」的参数文件；证据没丢：
#     run2/run3 的 objective-best 全量参数仍在包内 `rt_search_t0080_v3_final.json` /
#     `rt_search_t0080_v3_seedfix.json` 的 `opt_params`（37 键逐键相等），
#     选型过程在 `final_selection.json`。
_del_params = params_doc["params"]


def _effective_params(path):
    """Return the params a consumer would actually get, or None when the file is not a params file.

    Mirrors `system_sota_longquan.py::load_params`: ``payload.get("params", payload)``.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("params", payload)
    return candidate if isinstance(candidate, dict) else None


for _p in sorted((DEL / "artifacts").glob("*.json")):
    _eff = _effective_params(_p)
    if _eff is None or _eff == _del_params:
        continue
    # `th_base` 是所有参数文件的公共键；缺它 ⇒ 这不是一份参数文件（例如搜索空间定义）
    if "th_base" not in _eff:
        continue
    _p.unlink()
    print(f"[CLEAN] 剔除交付包内的非交付参数文件：{_p.name}"
          "（有效参数 ≠ 交付参数 ⇒ load_params 会静默接受它，跑出 FAR 破线的结果）")
# r83: memory 章节级合并工具放在包**根**（HOST_PUSH.sh 以 $HERE/merge_memory_section.py 引用它），
# 不进 artifacts/ —— 否则同一份东西在包里出现「根 + artifacts/」两份，又回到 r81 的老问题。
#   此前包内那份是手工放的（00:44），server 侧 search/ 里跑的却是另一份（22:15 旧版），
#   两份 md5 不同 ⇒ 「同一份东西两个版本」长在**交付链的关键工具**上，且推给 host 的正是包内那份。
#   ⇒ 这里显式刷新成 server 侧那份；闸门 §6i 逐字节兜底，并禁止 artifacts/ 里出现第二份。
# r84: 归一方向此前只写在文档里，实现却是「包内 ← server」——
#   若有人在包内手改（**这正是 r82 的失败模式**：00:44 手工放），下一次 build 会**静默覆盖**掉。
#   现在两端不一致就报错，把「不一致」摆到人眼前，而不是悄悄吃掉。
_mrg_src84 = RUNS / "search/merge_memory_section.py"
_mrg_dst84 = DEL / "merge_memory_section.py"
if _mrg_src84.exists() and _mrg_dst84.exists() and _mrg_src84.read_bytes() != _mrg_dst84.read_bytes():
    raise SystemExit(
        "[FAIL] merge_memory_section.py 两端不一致（server search/ vs 包根）—— 本次会覆盖包内那份，已中止。\n"
        "       若包内那份是你手工改的新版：先把它复制回 search/ 再 build；"
        "若 server 侧那份才是你要的：删掉包根那份再 build。")
shutil.copy2(_mrg_src84, DEL / "merge_memory_section.py")

# 3. OSD manifest
manifest = []
total_bytes = 0
for group in ("pass", "un-pass"):
    for mp4 in sorted((OSD / group).glob("*.mp4")):
        size_bytes = mp4.stat().st_size
        total_bytes += size_bytes
        manifest.append({"group": group, "file": mp4.name, "size_mb": round(size_bytes / 1e6, 1)})
(DEL / "osd_manifest.json").write_text(json.dumps({
    "osd_dir": str(OSD),
    "pass_count": sum(1 for m in manifest if m["group"] == "pass"),
    "un_pass_count": sum(1 for m in manifest if m["group"] == "un-pass"),
    "total": len(manifest),
    "total_bytes": total_bytes,
    "total_mb": round(total_bytes / 1e6, 1),
    "size_unit": "MB = 10^6 bytes (decimal); size_mb / total_mb 同口径, total_bytes 为真实字节数",
    "reference_dir_kept": str(OSD_OLD),
    "reference_dir_pass": len(list((OSD_OLD / "pass").glob("*.mp4"))) if (OSD_OLD / "pass").exists() else 0,
    "reference_dir_un_pass": len(list((OSD_OLD / "un-pass").glob("*.mp4"))) if (OSD_OLD / "un-pass").exists() else 0,
    "files": manifest,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# 3b. 候选组 OSD 清单（r88）：交付 OSD 清单不动，候选另存一份
_cand_manifest = []
_cand_bytes = 0
for _group in ("pass", "un-pass"):
    for _mp4 in sorted((OSD_CAND / _group).glob("*.mp4")):
        _sz = _mp4.stat().st_size
        _cand_bytes += _sz
        _cand_manifest.append({"group": _group, "file": _mp4.name, "size_mb": round(_sz / 1e6, 1)})
if _cand_manifest:
    (DEL / "artifacts" / "osd_manifest_md16_candidate.json").write_text(json.dumps({
        "osd_dir": str(OSD_CAND),
        "role": "CANDIDATE -- not the delivered OSD set. The DELIVERED set is "
                + str(OSD) + " (match_dist=10.0); this one belongs to cc2_md16n "
                "(match_dist=16.0, full-caliber FAR 0.029348, only 2.2% headroom), "
                "kept switch-ready; switching waits for the board-side FAR re-check.",
        "pass_count": sum(1 for m in _cand_manifest if m["group"] == "pass"),
        "un_pass_count": sum(1 for m in _cand_manifest if m["group"] == "un-pass"),
        "total": len(_cand_manifest),
        "total_bytes": _cand_bytes,
        "total_mb": round(_cand_bytes / 1e6, 1),
        "size_unit": "MB = 10^6 bytes (decimal); size_mb / total_mb 同口径, total_bytes 为真实字节数",
        "files": _cand_manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] artifacts/osd_manifest_md16_candidate.json | {len(_cand_manifest)} mp4")
else:
    print(f"[WARN] 候选 OSD 目录为空或不存在，跳过候选清单：{OSD_CAND}")

# 4. comparison table
t_opt = load(RUNS / "eval/track_all_v3_optgapfit.json")
t_v3 = load(RUNS / "eval/track_all_v3.json")
cands = []
for name in ("cc2_md10", "cc2_md8", "cc2_md12", "cc2_md16n", "cc3_md10", "cc4_md10",
             "cc5_md10", "cc6_md10", "cc7_md10", "cc3_md16", "rerun_selected", "old_gapfit"):
    doc = load(SL / f"track_{name}.json")
    if doc:
        cands.append((name, doc["overall"]))
ev = load(RUNS / "search/rt_eval_t0080_v3_opt.json")
scen = ["s1", "s2", "s3", "s4"]
arm3 = ev["arms"]["3_jump_optimised"]
arm2a = ev["arms"]["2a_jump_dense_timebase"]
search = load(RUNS / "search/rt_search_t0080_v3.json")
untuned = search["arms"]["2b_jump_physical_untuned"]

rows = []
o_v3, o_opt = t_v3["overall"], t_opt["overall"]
rows.append(("v3（基准）", o_v3, sum(untuned[s]["f1_delivered_agg"] for s in scen) / 4))
rows.append(("v3 + optgapfit（交付）", o_opt, sum(arm3[s]["f1_delivered_agg"] for s in scen) / 4))
for name, o in cands:
    if name == "old_gapfit":
        rows.append((f"{name}（旧 gapfit，仅对照）", o, None))

lines = []
A = lines.append
A("# trial_0080 + v3 · Δn 重搜 —— 交付包对照表")
A("")
A(f"- 交付参数档：`{src.name}`（**params_md5 `{canon_md5(params_doc)}`**）")
A(f"- 新 OSD 目录：`{OSD}`（pass 21 / un-pass 3 = 24 个 mp4）")
A(f"- 对照 OSD 目录（**保留未动**）：`{OSD_OLD}`")
A("- 基准：v3 = th_base 0.48 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 7 / confirm_gap_tol 0")
A(f"  （params_md5 `{canon_md5(load(REPO / 'manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json'))}`）")
A("")
A("## 1. 逐帧口径（全量 24 序 / 165,223 帧，交付口径）")
A("")
A("| 工作点 | F1 | FAR | Recall | Precision | TP | FP | FN | FAR≤0.03 |")
A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for label, o, _ in rows:
    ok = "✅" if o["far_per_frame"] <= 0.03 else "**❌ 破线**"
    A(f"| {label} | {o['f1'] * 100:.4f}% | {o['far_per_frame']:.6f} | {o['recall'] * 100:.2f}% | "
      f"{o['precision'] * 100:.2f}% | {o['tp']} | {o['fp']} | {o['fn']} | {ok} |")
A("")
A(f"- 交付组 vs v3：**F1 {(o_opt['f1'] - o_v3['f1']) * 100:+.4f}pp**，**FAR {o_opt['far_per_frame'] - o_v3['far_per_frame']:+.6f}**（红线 0.03）")
A(f"- 交付组 vs 旧 gapfit：F1 +{(o_opt['f1'] - dict((n, o) for n, o in cands)['old_gapfit']['f1']) * 100:.4f}pp，"
  f"FAR {o_opt['far_per_frame'] - dict((n, o) for n, o in cands)['old_gapfit']['far_per_frame']:+.6f} ⇒ **两个指标都更好**")
A("")
A("🔴 **两阶段选型（重要）**：搜索器在 **4000 帧截断口径**上最大化跳帧臂，")
A("但**截断口径系统性低报 FAR ≈ +0.005~0.006（绝对值）** ⇒ **红线必须在全量口径上判**。")
A("实测反例：第二次搜索（换一组 cadence 采样）的 objective-best")
A("（confirm_consec 2 / confirm_gap_tol 8 / max_age_phys 10 / r_slope 5.0 / match_dist 16.0）")
A("在截断口径 FAR 0.029013（看着守线），**全量口径 0.0352 ⇒ 破线 ⇒ 被拒**。")
A("详见 `artifacts/final_selection.json`。")
A("")
A("## 2. 跳帧口径（4000 帧仿真，物理帧基 = 板卡真实节奏）")
A("")
A("| 臂 | mean F1 |")
A("| --- | --- |")
A(f"| v3 原参 · 跳帧·物理帧基（未调） | **{sum(untuned[s]['f1_delivered_agg'] for s in scen) / 4:.4f}**（轨迹全判死） |")
A(f"| 交付参数 · 跳帧·物理帧基 | **{sum(arm3[s]['f1_delivered_agg'] for s in scen) / 4:.4f}** |")
A(f"| 交付参数 · 跳帧·密集帧基 | {sum(arm2a[s]['f1_delivered_agg'] for s in scen) / 4:.4f} |")
A(f"| 逐帧天花板 | {ev['arms']['1_perframe_ceiling']['f1']:.4f} |")
A("")
A("## 3. 候选短名单（全量口径实测）")
A("")
A("| 候选 | 逐帧 F1 | 逐帧 FAR |")
A("| --- | --- | --- |")
for name, o in cands:
    A(f"| {name} | {o['f1'] * 100:.4f}% | {o['far_per_frame']:.6f} |")
A("")
A("## 4. 交付参数（叠加在 v3 之上的 7 个 Δn 键）")
A("")
A("```json")
A(json.dumps(params_doc["_provenance"]["applied_on_top_of_v3"], ensure_ascii=False, indent=2))
A("```")
A("")
A("## 5. 目录说明")
A("")
A(f"- `{src.name}` —— 新参数档（**交付物**）")
A("- `osd_manifest.json` —— 新 OSD 目录逐文件清单")
A("- `artifacts/` —— 搜索 json/md、三臂复核 json/md、track json、三份对账 json、memory 章节、搜索规格书、板卡同步预设")
A("")
cmp_path = DEL / "COMPARISON.md"
if cmp_path.exists() and "口径更正" in cmp_path.read_text(encoding="utf-8"):
    # The shipped COMPARISON.md was hand-refined after this generator was written
    # (it carries the caliber-correction notes, the per-scenario jump table and the
    # host-backfill chapter that this template does not have).  Overwriting it would
    # silently regress the delivered document, so keep the refined one.
    print("[SKIP] COMPARISON.md 已是手工精修版（含口径更正），不覆盖")
else:
    cmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] {DEL}")
for p in sorted(DEL.rglob("*")):
    if p.is_file():
        print(f"    {p.relative_to(DEL)}  ({p.stat().st_size} B)")
