#!/usr/bin/env python3
"""End-to-end verification of the 2026-11-19 delta-n delivery."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo")
RUNS = Path("/mnt/data/siping/runs/trial0080_onnx")
OSD = Path("/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_20261119")
OSD_OLD = Path("/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_gapfit_20261119")
PARAMS = REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json"
del_dir_art = REPO / "manu/memory/delivery_20261119"

ok = True
_checks = 0
def check(label, cond, detail=""):
    global ok, _checks
    ok = ok and bool(cond)
    _checks += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

# This script verifies the *server-side* run artefacts (the 165,223-frame evaluation,
# the OSD directory, the re-run predictions).  On the host those paths do not exist,
# so say so plainly instead of dying on a TypeError deep in section 2.
if not (RUNS / "eval/track_all_v3_optgapfit.json").exists():
    print("[SKIP] 本脚本是**服务器侧**核验脚本：依赖 "
          f"{RUNS}/ 的跑批产物（本机没有）。")
    print("       host 侧请用：HOST_PUSH.sh 的内联复核 / HOST_BACKFILL.sh --check。")
    sys.exit(2)

print("=== 1. 交付参数档 ===")
doc = load(PARAMS)
check("参数档存在", doc is not None, str(PARAMS))
canon = hashlib.md5(json.dumps(doc["params"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
check("params_md5 与文件内 provenance 一致",
      canon == doc["_provenance"]["params_md5"], canon)
for k in ("confirm_consec", "confirm_gap_tol", "max_age_phys", "r_slope", "alpha_decay", "match_dist", "min_hits"):
    check(f"显式写入 {k}", k in doc["params"], repr(doc["params"].get(k)))
check("th_base 仍是 0.48", doc["params"]["th_base"] == 0.48)
check("th_ground 仍是 0.26", doc["params"]["th_ground"] == 0.26)
check("sky_ratio 仍是 1.0", doc["params"]["sky_ratio"] == 1.0)

print("\n=== 2. 逐帧口径（交付口径，全量 24 序）===")
t = load(RUNS / "eval/track_all_v3_optgapfit.json")
v3 = load(RUNS / "eval/track_all_v3.json")
o, o3 = t["overall"], v3["overall"]
check("GT 帧数与 v3 一致（165,223 帧）", o["gt"] == o3["gt"] == 133257, str(o["gt"]))
check("FAR <= 0.03 红线", o["far_per_frame"] <= 0.03, f"{o['far_per_frame']:.6f}")
check("F1 >= 88% 交付下限", o["f1"] >= 0.88, f"{o['f1']*100:.4f}%")
check("Recall >= 85%", o["recall"] >= 0.85, f"{o['recall']*100:.2f}%")
print(f"       v3 = F1 {o3['f1']*100:.4f}% / FAR {o3['far_per_frame']:.6f}")
print(f"       opt = F1 {o['f1']*100:.4f}% (Δ{(o['f1']-o3['f1'])*100:+.4f}pp) / FAR {o['far_per_frame']:.6f} (Δ{o['far_per_frame']-o3['far_per_frame']:+.6f})")

print("\n=== 3. 闸门与对账 ===")
for name, path in (("官方 v3", RUNS / "eval/cmp_v3_optgapfit_vs_official_v3.json"),
                   ("重建 v3", RUNS / "eval/cmp_v3_optgapfit_vs_rebuilt_v3.json"),
                   ("可复现性(同参 v3)", RUNS / "eval/cmp_v3_rebuilt_vs_official_v3.json")):
    d = load(path)
    if not d:
        check(f"{name} 对账产物存在", False, str(path)); continue
    rows = d["per_sequence"]
    gt_ok = sum(1 for r in rows if r.get("gt_mine") == r.get("gt_official"))
    check(f"{name}: GT 一致 {gt_ok}/{len(rows)}", gt_ok == len(rows) == 24)

print("\n=== 4. OSD 产物 ===")
n_pass = len(list((OSD / "pass").glob("*.mp4")))
n_un = len(list((OSD / "un-pass").glob("*.mp4")))
check("pass + un-pass = 24", n_pass + n_un == 24, f"pass={n_pass} un-pass={n_un}")
tsv = (OSD / "per_sequence_hyperparams.tsv")
check("per_sequence_hyperparams.tsv 存在", tsv.exists())
if tsv.exists():
    head = tsv.read_text(encoding="utf-8").splitlines()
    gp = head[0] if head else ""
    check("TSV 全局参数列非空", gp.startswith("# global params:") and "th_base=" in gp, gp)
    check("TSV 含 24 行数据", sum(1 for l in head if l and not l.startswith("#") and not l.startswith("sequence")) == 24)
    # The "# global params:" line must record the COMPLETE working point.  It used to be a
    # hard-coded 5-key list that dropped every delta-n knob, so the TSV documented a working
    # point it did not actually use.
    gline = head[0] if head else ""
    for k, v in (("confirm_consec=2", "confirm_consec"), ("confirm_gap_tol=8", "confirm_gap_tol"),
                 ("r_slope=5.0", "r_slope"), ("match_dist=10.0", "match_dist"),
                 ("alpha_decay=1.0", "alpha_decay"), ("max_age_phys=None", "max_age_phys"),
                 ("min_hits=3", "min_hits")):
        check(f"TSV 全局参数记录 {v}", f" {k} " in gline + " ", f"{v}={k.split('=')[1]}")
check("search_json_for_render.json 存在", (OSD / "search_json_for_render.json").exists())
oj = load(OSD / "search_json_for_render.json")
if oj:
    p = oj["best_global"]["params"]
    check("渲染 JSON 的 confirm_consec = 2", p.get("confirm_consec") == 2, str(p.get("confirm_consec")))
    check("渲染 JSON 的 confirm_gap_tol = 8", p.get("confirm_gap_tol") == 8, str(p.get("confirm_gap_tol")))
    check("渲染 JSON 的 r_slope = 5.0", p.get("r_slope") == 5.0, str(p.get("r_slope")))
    check("渲染 JSON 的 max_age_phys = null", "max_age_phys" in p and p["max_age_phys"] is None)

# The mp4 file names carry the per-sequence F1: they must equal the delivered track json
# exactly, otherwise the videos were rendered from a stale prediction set.
import re
_tr = load(RUNS / "eval/track_all_v3_optgapfit.json")["per_sequence"]
_mis = []
_n = 0
for _f in sorted((OSD / "pass").glob("*.mp4")) + sorted((OSD / "un-pass").glob("*.mp4")):
    _m = re.match(r"(VIDEO\d+_\d+_\d+)_F1-([\d.]+)\.mp4$", _f.name)
    if not _m or _m.group(1) not in _tr:
        _mis.append(_f.name)
        continue
    _n += 1
    if abs(_tr[_m.group(1)]["f1"] * 100 - float(_m.group(2))) > 0.01:
        _mis.append(_f.name)
check("OSD 文件名 F1 与交付 track json 逐序列一致", not _mis and _n == 24, f"{_n}/24 一致, 不一致={_mis[:3]}")

# The parameter file, the OSD's search_json_for_render.json and the OSD's TSV header travel to
# different people: they must describe the SAME working point, key for key.
import json as _json
_pf = _json.loads(PARAMS.read_text(encoding="utf-8"))["params"]
_sj = _json.loads((OSD / "search_json_for_render.json").read_text(encoding="utf-8"))["best_global"]["params"]
_tl = [l for l in tsv.read_text(encoding="utf-8").splitlines() if l.startswith("# global params:")][0]
_ts = {}
for _tok in _tl.replace("# global params:", "").split():
    _k, _, _v = _tok.partition("=")
    _ts[_k] = None if _v == "None" else (float(_v) if "." in _v else int(_v))
def _norm(_d):
    return {k: (None if v is None else round(float(v), 9) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in _d.items()}
check("参数档 == OSD search_json_for_render.json（逐键）", _norm(_pf) == _norm(_sj),
      f"{len(_pf)} vs {len(_sj)} 键")
check("参数档 == OSD TSV 全局参数行（逐键）", _norm(_pf) == _norm(_ts), f"{len(_pf)} vs {len(_ts)} 键")

print("\n=== 5. 对照 OSD 目录（必须保留未动）===")
op = len(list((OSD_OLD / "pass").glob("*.mp4")))
ou = len(list((OSD_OLD / "un-pass").glob("*.mp4")))
check("旧 gapfit OSD 仍在 21+3", op + ou == 24, f"pass={op} un-pass={ou}")
check("旧 gapfit OSD 未被覆盖", (OSD_OLD / "per_sequence_hyperparams.tsv").exists())

print("\n=== 6. memory 同步（外科式插入：零删改）===")
def _section_titles(txt):
    """Titles of every `## ` section, used for the loss test."""
    import re

    return [m.group().strip() for m in re.finditer(r"^## .*$", txt, re.M)]


KEY_FACTS = [
    "§255", "§256", "§257", "§258", "§259", "§260",
    "修掉 `--self-check` 闸门陷阱",
    "gapfit 臂",
    "重大发现",
]


targets = ["memory/memory.md", "memory/memory_compact.md",
           "manu/memory/memory.md", "manu/memory/memory_compact.md",
           "memory.md", "memory_compact.md"]
for rel in targets:
    p = REPO / rel
    bak = Path(str(p) + ".bak_20261119_1820")
    txt = p.read_text(encoding="utf-8")
    has = "§257 十八" in txt
    s258 = "## §258" in txt
    s259 = "## §259" in txt
    # NOTE: the `.bak` snapshot predates the concurrent line's own rewrite of §258, so a
    # line-diff against it reports the concurrent line's rewording as "deleted" lines even
    # though nothing was lost.  The correct loss test is per-section: every `## ` section
    # that exists in the snapshot must still exist, and every key fact must still be in
    # place.  Line counts are reported for information only.
    if bak.exists():
        r = subprocess.run(["diff", str(bak), str(p)], capture_output=True, text=True).stdout
        deleted = sum(1 for l in r.splitlines() if l.startswith("<"))
        added = sum(1 for l in r.splitlines() if l.startswith(">"))
        bak_txt = bak.read_text(encoding="utf-8")
        lost = [s for s in _section_titles(bak_txt) if s not in _section_titles(txt)]
        facts_lost = [k for k in KEY_FACTS if k in bak_txt and k not in txt]
    else:
        deleted, added, lost, facts_lost = -1, -1, [], []
    check(f"{rel}: §257 十八已插入", has)
    check(f"{rel}: §258 保留", s258 or rel.endswith("memory_compact.md") and "## §255" not in txt)
    check(f"{rel}: §259 保留", s259 or rel.endswith("memory_compact.md") and "## §255" not in txt)
    check(f"{rel}: 章节零丢失", not lost, f"lost={lost}")
    check(f"{rel}: 关键事实零丢失", not facts_lost, f"lost={facts_lost}")
    check(f"{rel}: 行数变化（信息项）", True, f"deleted={deleted} added={added}")
    check(f"{rel}: 含 arm5 口径澄清（§257 十八之二十）", "arm5" in txt and "逐帧交付口径" in txt)

# The three main copies and the three compact copies must be byte-identical WITHIN
# their own tier, and the main tier must match the canonical section source file.
def _block(rel):
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines(keepends=True)
    s = next(i for i, l in enumerate(lines) if l.startswith("## §257 十八"))
    e = next((i for i, l in enumerate(lines) if i > s and l.startswith("## §")), len(lines))
    return "".join(lines[s:e]).rstrip() + "\n"

main_blocks = {_block(r) for r in ("memory/memory.md", "manu/memory/memory.md", "memory.md")}
comp_blocks = {_block(r) for r in ("memory/memory_compact.md", "manu/memory/memory_compact.md", "memory_compact.md")}
check("3 个主档的 §257 十八 逐字节一致", len(main_blocks) == 1)
check("3 个精简档的 §257 十八 逐字节一致", len(comp_blocks) == 1)
src = Path("/mnt/data/siping/runs/trial0080_onnx/search/memory_section_257_18.md").read_text(encoding="utf-8").rstrip() + "\n"
check("章节源文件 == 主档 §257 十八", src in main_blocks)
check("精简档 ≠ 主档（档位未串）", main_blocks != comp_blocks)

# The delivery judgement must be the FULL caliber, and the truncation caliber must
# be demonstrably different -- otherwise the arm5 numbers get quoted as delivery.
ev = load(RUNS / "search/rt_eval_t0080_v3_opt.json")
if ev:
    a5 = ev["arms"]["5_optimised_perframe"]["perframe"]
    ceil = ev["arms"]["1_perframe_ceiling"]
    check("arm5 跑在截断子集（< 全量 165,223 帧）", ceil["frames"] < 165223, f"{ceil['frames']} 帧 = {ceil['frames']/165223*100:.2f}%")
    check("arm5 FAR(帧加权) != 全量 FAR（口径确有差异）", abs(a5["far_delivered_agg"] - o["far_per_frame"]) > 1e-6,
          f"arm5 {a5['far_delivered_agg']:.6f} vs 全量 {o['far_per_frame']:.6f}")
    check("arm5 样本均值 FAR 口径存在（勿与帧加权混用）", a5["far_delivered"] != a5["far_delivered_agg"],
          f"agg {a5['far_delivered_agg']:.6f} / mean {a5['far_delivered']:.6f}")

print("\n=== 6b. 确定性采样器重测（跳帧臂数字的规范来源）===")
det = load(RUNS / "search/shortlist_deterministic.json")
armA = load(RUNS / "search/rt_eval_delivered_armA.json")
check("重测产物 shortlist_deterministic.json 存在", det is not None)
check("重测产物 rt_eval_delivered_armA.json 存在", armA is not None)
if det and armA:
    res = det["results"]
    check("v3 原参跳帧·物理帧基 = 0.084122（轨迹全判死）",
          abs(res["v3_base"]["jump_physical_mean_f1"] - 0.084122) < 1e-6,
          f"{res['v3_base']['jump_physical_mean_f1']:.6f}")
    check("交付参数跳帧·物理帧基 = 0.951398",
          abs(res["delivered_cc2_md10"]["jump_physical_mean_f1"] - 0.951398) < 1e-6,
          f"{res['delivered_cc2_md10']['jump_physical_mean_f1']:.6f}")
    check("逐帧天花板（v3 基座，截断口径）= 0.960186",
          abs(res["v3_base"]["perframe_truncated_f1_agg"] - 0.960186) < 1e-6,
          f"{res['v3_base']['perframe_truncated_f1_agg']:.6f}")
    a3 = armA["arms"]["3_jump_optimised"]
    a2b = armA["arms"]["2b_jump_physical_untuned"]
    check("产物 A 与产物 B 交叉一致（arm3 == delivered）",
          abs(a3["_jump_mean_f1"] - res["delivered_cc2_md10"]["jump_physical_mean_f1"]) < 1e-9,
          f"{a3['_jump_mean_f1']:.6f} vs {res['delivered_cc2_md10']['jump_physical_mean_f1']:.6f}")
    check("产物 A 与产物 B 交叉一致（arm2b == v3 未调）",
          abs(a2b["_jump_mean_f1"] - res["v3_base"]["jump_physical_mean_f1"]) < 1e-9,
          f"{a2b['_jump_mean_f1']:.6f} vs {res['v3_base']['jump_physical_mean_f1']:.6f}")
    check("守线候选里跳帧最高的是 cc2_md16n（0.951693），交付用 cc2_md10",
          max((k for k in res if k.startswith("cc2_")), key=lambda k: res[k]["jump_physical_mean_f1"]) == "cc2_md16n"
          and res["cc2_md16n"]["jump_physical_mean_f1"] > res["cc2_md10"]["jump_physical_mean_f1"],
          "cc2_md16n 0.951693 > cc2_md10 0.951398（差 0.03pp，换 3 倍 FAR 余量）")
    check("重测脚本随包交付", (del_dir_art / "artifacts/remeasure_shortlist_deterministic.py").exists())

print("\n=== 6c. 板卡施工单与交付参数档一致性 ===")
preset_path = del_dir_art / "artifacts/trial0080_board_fusion_preset_20261119.md"
preset = preset_path.read_text(encoding="utf-8") if preset_path.exists() else ""
check("板卡施工单随包交付", preset_path.exists(), str(preset_path))
if det and preset:
    res6 = det["results"]
    d10 = res6["delivered_cc2_md10"]
    spec_nums = {
        "跳帧·物理帧基": f"{d10['jump_physical_mean_f1']:.6f}",
        "跳帧·密集帧基": f"{d10['jump_dense_mean_f1']:.6f}",
        "v3未调·物理帧基": f"{res6['v3_base']['jump_physical_mean_f1']:.6f}",
    }
    miss_num = [f"{k}={v}" for k, v in spec_nums.items() if v not in preset]
    check("施工单里的跳帧数 == 确定性重测规范值", not miss_num,
          "缺 " + ", ".join(miss_num) if miss_num else " / ".join(spec_nums.values()))
    rows = {}
    for line in preset.splitlines():
        m = re.match(r"^\|\s*`([a-z_]+)`\s*\|", line)
        if m and m.group(1) not in rows:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[m.group(1)] = cells[2] if len(cells) > 2 else ""
    pdoc = load(del_dir_art / "system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json")
    knob_bad = []
    for key in ("confirm_consec", "confirm_gap_tol", "max_age_phys", "r_slope", "alpha_decay", "match_dist", "min_hits"):
        cell = rows.get(key, "").replace("**", "").strip()
        want = pdoc["params"].get(key) if pdoc else None
        if want is None:
            ok_knob = cell.lower().startswith("null")
        else:
            num = re.match(r"^-?\d+(?:\.\d+)?", cell)
            ok_knob = bool(num) and abs(float(num.group(0)) - float(want)) < 1e-9
        if not ok_knob:
            knob_bad.append(f"{key}: 施工单={cell!r} 交付档={want!r}")
    check("施工单的 7 个 Δn 参数 == 交付参数档逐键一致", not knob_bad,
          "; ".join(knob_bad) if knob_bad else
          "confirm_consec=2 confirm_gap_tol=8 max_age_phys=null r_slope=5.0 alpha_decay=1.0 match_dist=10.0 min_hits=3")
    thr_bad = [t for t in ("th_base=0.48", "th_ground=0.26", "sky_ratio=1.0") if t not in preset]
    check("施工单的冻结阈值 == 交付参数档", not thr_bad,
          "; ".join(thr_bad) if thr_bad else "th_base=0.48 / th_ground=0.26 / sky_ratio=1.0")

print("\n=== 6d. 硬证据随包交付（self-check / 冻结 baseline / 搜索空间）===")
sc_path = del_dir_art / "artifacts/self_check_optgapfit.log"
sc_text = sc_path.read_text(encoding="utf-8") if sc_path.exists() else ""
base_path = del_dir_art / "artifacts/system_sota_longquan_trial0080_singleframe_gate_baseline.json"
base_doc = load(base_path)
space_path = del_dir_art / "artifacts/space_t0080_v3.json"
space_doc = load(space_path)
check("self-check 的 [GATE PASS] 证明随包交付", sc_path.exists() and "[GATE PASS]" in sc_text, str(sc_path))
check("冻结单帧闸门 baseline 随包交付", base_doc is not None, str(base_path))
check("搜索空间定义随包交付", space_doc is not None, str(space_path))
if sc_text and base_doc:
    pairs = re.findall(r"^\s*(\w+):\s*expected\s+([-\d.eE+]+)\s*\|\s*actual\s+([-\d.eE+]+)", sc_text, re.M)
    got = {k: (float(e), float(a)) for k, e, a in pairs}
    ne = [f"{k}: {e} != {a}" for k, (e, a) in got.items() if abs(e - a) > 1e-12]
    check("self-check 日志 expected == actual（逐项）", bool(got) and not ne,
          "; ".join(ne) if ne else f"{len(got)} 项全等")
    exp = base_doc["expected"]
    miss = [k for k in exp if k not in got]
    drift = []
    for k, v in exp.items():
        if k in got and abs(got[k][0] - float(v)) > (0.0 if k in ("tp", "fp", "fn", "gt") else 1e-9):
            drift.append(f"{k}: 日志 {got[k][0]} vs baseline {v}")
    check("self-check 日志的 expected == 随包冻结 baseline", not miss and not drift,
          (("缺 " + ",".join(miss) + " ") if miss else "") + ("; ".join(drift) if drift else f"{len(exp)} 项全对"))
if space_doc:
    req_space = {
        "confirm_consec": [2, 3, 4, 5, 6, 7],
        "confirm_gap_tol": [0, 2, 3, 4, 5, 6, 8],
        "max_age_phys": [None, 10, 12, 15, 18],
        "r_slope": [0.0, 2.0, 3.5, 5.0],
        "alpha_decay": [1.0, 0.75, 0.82, 0.88, 0.93],
        "match_dist": [4.0, 6.0, 8.0, 10.0, 12.0, 16.0],
        "min_hits": [2, 3],
    }
    bad = []
    for k, vals in req_space.items():
        if k not in space_doc:
            bad.append(f"{k} 缺失")
        elif not set(vals) <= set(space_doc[k]):
            bad.append(f"{k}: 缺 {sorted(set(vals) - set(space_doc[k]), key=str)}")
    check("搜索空间逐旋钮覆盖任务书要求的值集合", not bad,
          "; ".join(bad) if bad else "7 个旋钮全覆盖")
    check("搜索空间保留 v3 可回退点（confirm_consec=7 / confirm_gap_tol=0）",
          7 in space_doc.get("confirm_consec", []) and 0 in space_doc.get("confirm_gap_tol", []),
          f"confirm_consec={space_doc.get('confirm_consec')} / confirm_gap_tol={space_doc.get('confirm_gap_tol')}")

print("\n=== 6e. 交付清单与 README 登记一致（防清单漂移）===")
readme_path = del_dir_art / "artifacts/README.md"
readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
reg_names = set()
for _line in readme_text.splitlines():
    if _line.startswith("|") and _line.count("|") > 2:
        _first = _line.split("|")[1]
        if "`" in _first:
            reg_names.update(re.findall(r"`([^`]+)`", _first))
art_dir = del_dir_art / "artifacts"
art_files = {p.name for p in art_dir.iterdir() if p.is_file() and ".bak" not in p.name} if art_dir.is_dir() else set()
root_files = {p.name for p in del_dir_art.iterdir() if p.is_file() and ".bak" not in p.name}
# r84: 原先只查 artifacts/ ⇒ 包根 4 个交付物（COMPARISON.md / HOST_BACKFILL.sh /
# osd_manifest.json / 参数档）从未登记，其中 HOST_BACKFILL.sh 恰恰是 README 让 host 执行的那个脚本。
unreg = sorted((art_files | root_files) - reg_names - {"README.md"})
check("交付包里每个交付物都在 README 清单登记（artifacts/ 与包根都算）", not unreg,
      "未登记: " + ", ".join(unreg) if unreg else f"{len(art_files | root_files) - 1} 件全部登记")
ghost = sorted(n for n in reg_names if n not in art_files and n not in root_files)
check("README 清单登记的文件都真实存在", not ghost,
      "登记但不存在: " + ", ".join(ghost) if ghost else f"{len(reg_names)} 条全部对得上")

print("\n=== 6f. 跳帧数字口径一致（作废值不得当现值）===")
det_doc6 = load(RUNS / "search/shortlist_deterministic.json")
canon6 = {}
if det_doc6:
    _r6 = det_doc6["results"]
    canon6 = {
        "交付·物理帧基": f"{_r6['delivered_cc2_md10']['jump_physical_mean_f1']:.4f}",
        "交付·密集帧基": f"{_r6['delivered_cc2_md10']['jump_dense_mean_f1']:.4f}",
        "v3未调·物理帧基": f"{_r6['v3_base']['jump_physical_mean_f1']:.6f}",
    }
check("跳帧规范值可从 shortlist_deterministic.json 现取", bool(canon6),
      " / ".join(f"{k}={v}" for k, v in canon6.items()))
# 「修复前采样器」那一批数字（0.9521 / 0.9374 / …）：只允许出现在明确标注「已作废」的段落里。
_obsolete6 = ["0.9521", "0.9374", "0.951990", "0.952015", "0.952127", "0.937438", "0.9520",
              "0.9458", "0.9378", "0.9289", "0.9163", "0.9047", "0.9558", "0.9539", "0.9414",
              "0.9574", "0.9500", "0.9306", "0.9121", "0.9570",
              # r82: 「差 0.0025pp」这句本身就是误引 —— 0.0025pp 实为 md8↔md10 之差，
              # objective-best 与交付组的真实差是 0.0788pp ⇒ 旧差值一律降级为作废值。
              "0.0025pp"]
_markers6 = ("作废", "早先", "之前写的", "修复前", "旧值", "旧的", "口径更正", "已改", "改成", "不可复现", "废弃")
# r82: 4 份 md 报告 = `--out-md` 自动生成的**某一次跑批原始记录**（跳帧数字不是交付口径），
#   它们含作废跳帧值是**设计如此** ⇒ 不按段落判「作废值」，改判**头部口径声明**在不在；
#   声明缺失 = 读者会把作废值当现值 ⇒ 同样判 FAIL。
_AUTOGEN6 = "由 `--out-md` 从**同名 json** 自动生成"
_CALIBER6 = "跳帧数字不是交付口径"
_audit6 = [del_dir_art / "COMPARISON.md",
           del_dir_art / "artifacts/memory_section_257_18.md",
           del_dir_art / "artifacts/memory_section_257_18_compact.md",
           del_dir_art / "artifacts/trial0080_delta_search_spec_20261119.md",
           del_dir_art / "artifacts/trial0080_board_fusion_preset_20261119.md",
           art_dir / "rt_search_t0080_v3.md",
           art_dir / "rt_search_t0080_v3_final.md",
           art_dir / "rt_search_t0080_v3_seedfix.md",
           art_dir / "rt_eval_t0080_v3_opt.md"]
_bad6 = []
for _p in _audit6:
    if not _p.exists():
        continue
    _txt6 = _p.read_text(encoding="utf-8")
    _head6 = "\n".join(_txt6.splitlines()[:6])
    if _AUTOGEN6 in _head6:
        if _CALIBER6 not in _head6:
            _bad6.append(f"{_p.name}: 自动生成报告缺头部口径声明")
        continue
    for _para in re.split(r"\n\s*\n", _txt6):
        _hits = [t for t in _obsolete6 if re.search(r"(?<![\d.])" + re.escape(t) + r"(?![\d])", _para)]
        if _hits and not any(m in _para for m in _markers6):
            _bad6.append(f"{_p.name}: {','.join(_hits)}")
check("交付文档里作废跳帧值只出现在明确标注「已作废」的段落", not _bad6,
      "; ".join(_bad6[:3]) if _bad6 else f"{len(_audit6)} 份文档全部合规")
# 现值必须是规范值（4 位小数）
_miss6 = []
for _p in _audit6:
    if not _p.exists() or _p.name == "COMPARISON.md":
        continue
    _txt = _p.read_text(encoding="utf-8")
    if _AUTOGEN6 in "\n".join(_txt.splitlines()[:6]):
        continue  # 自动生成报告 = 某次跑批的原始记录，数字不是交付口径（§6f 上方 + §6h③ 已单独判）
    _r6b = det_doc6["results"]
    # 板卡施工单用 6 位小数（0.951398），正文用 4 位（0.9514）——两种精度都算「写的是规范值」
    for _label, _key, _per in (("交付·物理帧基", "delivered_cc2_md10", "jump_physical_mean_f1"),
                               ("交付·密集帧基", "delivered_cc2_md10", "jump_dense_mean_f1")):
        _v = _r6b[_key][_per]
        # 用词边界正则，避免 "0.951398x" 这类子串蒙混过关
        _hit = any(re.search(r"(?<![\d.])" + re.escape(f"{_v:.4f}") + r"(?![\d.a-zA-Z])", _txt)
                   or re.search(r"(?<![\d.])" + re.escape(f"{_v:.6f}") + r"(?![\d.a-zA-Z])", _txt)
                   for _ in (0,))
        if not _hit:
            _miss6.append(f"{_p.name}: 缺 {_label} {f'{_v:.4f}'}/{f'{_v:.6f}'}")
check("交付文档写的是规范跳帧值（0.9514/0.951398、0.9383/0.938340）而非作废值", not _miss6,
      "; ".join(_miss6) if _miss6 else "规范值全部在位")

print("\n=== 6g. 同一份东西只能有一个版本（双份不同源 / 死条目）===")
# r81: search/ 下曾留着两份与包内权威版**不同源**的陈旧副本 ——
#   ① search/verify_all.py（00:15 版，缺 §6c/§6d/§6f）；而 §257 的正文明确指引读者去
#      search/ 跑它 ⇒ 照做会跑到旧闸门（漏掉施工单一致性 / 硬证据随包 / 跳帧口径三类校验）。
#   ② search/memory_md5_manifest.txt（整档 md5 全是旧值，无人引用）⇒ 拿它 md5sum -c 会误报。
_mirror6g = RUNS / "search/verify_all.py"
_pkg_gate6g = art_dir / "verify_all.py"
check("包内闸门 == search/ 闸门（逐字节；同一闸门只有一个版本）",
      _mirror6g.exists() and _pkg_gate6g.exists() and _mirror6g.read_bytes() == _pkg_gate6g.read_bytes(),
      f"search={_mirror6g.stat().st_size if _mirror6g.exists() else 0} B / 包内={_pkg_gate6g.stat().st_size} B")
_stale6g = RUNS / "search/memory_md5_manifest.txt"
check("search/ 下没有陈旧的 memory_md5_manifest.txt 副本", not _stale6g.exists(),
      "已清理（整档 md5 以包内 artifacts/memory_md5_manifest.txt 为准）" if not _stale6g.exists() else f"仍存在 {_stale6g}")
# 生成器的 copies 字典：不允许死条目（指向不存在的源），也不允许包内副本与源不一致。
_gen6g = RUNS / "search/build_delivery.py"
_copies6g = {}
if _gen6g.exists():
    import ast

    _ns6g = {"RUNS": RUNS, "REPO": REPO, "Path": Path}
    for _node6g in ast.walk(ast.parse(_gen6g.read_text(encoding="utf-8"))):
        if isinstance(_node6g, ast.Assign) and getattr(_node6g.targets[0], "id", None) == "copies":
            _copies6g = {
                eval(compile(ast.Expression(_k6g), "<k>", "eval"), _ns6g): eval(
                    compile(ast.Expression(_v6g), "<v>", "eval"), _ns6g)
                for _k6g, _v6g in zip(_node6g.value.keys, _node6g.value.values)
            }
_dead6g = [_n6g for _s6g, _n6g in _copies6g.items() if not _s6g.exists()]
check("生成器 copies 没有死条目（每个源都真实存在）", not _dead6g,
      "; ".join(_dead6g[:4]) if _dead6g else f"{len(_copies6g)} 条全部有源")
_drift6g = [
    _n6g for _s6g, _n6g in _copies6g.items()
    if _s6g.exists() and (art_dir / _n6g).exists()
    and hashlib.md5(_s6g.read_bytes()).hexdigest() != hashlib.md5((art_dir / _n6g).read_bytes()).hexdigest()
]
check("包内生成物 == 其源（逐字节）", not _drift6g,
      "; ".join(_drift6g[:4]) if _drift6g else f"{len(_copies6g)} 条全部一致")
# md 报告：由同名 json 现渲染，必须逐字节一致 ⇒ md 不可能与 json 漂移。
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from manu.evaluation.render_search_report_md import render_report_md  # noqa: E402

_md_bad6g = []
_md_stems6g = ("rt_search_t0080_v3", "rt_search_t0080_v3_final", "rt_search_t0080_v3_seedfix", "rt_eval_t0080_v3_opt",
                # r88: 新增的搜索批次（硬约束在交付口径上的首次搜索）同样纳入「md == 同名 json 现渲染」
                # 这条判据（该判据是**循环外单条** check ⇒ 纳入新批次**不增加闸门项数**）。
                "rt_search_t0080_v3_r88")
for _stem6g in _md_stems6g:
    _j6g = art_dir / f"{_stem6g}.json"
    _m6g = art_dir / f"{_stem6g}.md"
    if not (_j6g.exists() and _m6g.exists()):
        _md_bad6g.append(f"{_stem6g}: 缺文件")
    elif render_report_md(json.loads(_j6g.read_text(encoding="utf-8"))) != _m6g.read_text(encoding="utf-8"):
        _md_bad6g.append(f"{_stem6g}: 与 json 现渲染不一致")
check("md 报告 == 由同名 json 现渲染（md 不可能与 json 漂移）", not _md_bad6g,
      "; ".join(_md_bad6g[:3]) if _md_bad6g else f"{len(_md_stems6g)} 份全部一致")

print("\n=== 6h. 跳帧差值现算 + 生成器无作废字面量（第 82 轮）===")
# r82: 正文里那句「差 0.0025pp」是**误引**（0.0025pp 实为 md8↔md10 之差），
#   objective-best(rerun_selected) 与交付组的真实差 = 0.0788pp。
#   ⇒ 差值不再手写：一律由规范源 shortlist_deterministic.json 现算，正文必须写出现算值。
_r6h = det_doc6["results"]
_diff_best6h = (_r6h["rerun_selected"]["jump_physical_mean_f1"]
                 - _r6h["delivered_cc2_md10"]["jump_physical_mean_f1"]) * 100
_diff_md6h = (_r6h["delivered_cc2_md10"]["jump_physical_mean_f1"]
              - _r6h["cc2_md8"]["jump_physical_mean_f1"]) * 100
_txt_best6h = f"{_diff_best6h:.4f}pp"
_txt_md6h = f"{_diff_md6h:.4f}pp"
_docs6h = [del_dir_art / "artifacts/memory_section_257_18.md",
           del_dir_art / "artifacts/memory_section_257_18_compact.md",
           del_dir_art / "artifacts/trial0080_delta_search_spec_20261119.md"]
_miss6h = []
for _p in _docs6h:
    if not _p.exists():
        _miss6h.append(f"{_p.name}: 缺文件")
        continue
    _t6h = _p.read_text(encoding="utf-8")
    for _v6h in (_txt_best6h, _txt_md6h):
        if not re.search(r"(?<![\d.])" + re.escape(_v6h) + r"(?![\d])", _t6h):
            _miss6h.append(f"{_p.name}: 缺 {_v6h}")
check("正文/规格书写的跳帧差值 == 规范源现算值（objective-best 差 / md8 差）", not _miss6h,
      "; ".join(_miss6h[:3]) if _miss6h else f"{_txt_best6h} / {_txt_md6h} 全部在位")
# final_selection.json 是「交付了什么」的对外声明 ⇒ 它的跳帧 F1 必须等于规范值。
_fs6h = load(art_dir / "final_selection.json")
_can_f16h = round(_r6h["delivered_cc2_md10"]["jump_physical_mean_f1"], 6)
_fs_val6h = (_fs6h or {}).get("delivered", {}).get("jump_arm_f1_4000f")
check("final_selection.json 的交付跳帧 F1 == 规范值（不是旧采样器的值）",
      _fs_val6h is not None and abs(float(_fs_val6h) - _can_f16h) < 5e-7,
      f"{_fs_val6h} vs {_can_f16h}")
# 自动生成报告：头部必须写明「跳帧数字不是交付口径」并指向规范源。
_md_bad6h = []
for _stem6h in _md_stems6g:
    _m6h = art_dir / f"{_stem6h}.md"
    if not _m6h.exists():
        _md_bad6h.append(f"{_stem6h}: 缺文件")
        continue
    _h6h = "\n".join(_m6h.read_text(encoding="utf-8").splitlines()[:6])
    if _CALIBER6 not in _h6h:
        _md_bad6h.append(f"{_stem6h}: 头部缺口径声明")
    elif "shortlist_deterministic.json" not in _h6h or "rt_eval_delivered_armA.json" not in _h6h:
        _md_bad6h.append(f"{_stem6h}: 头部缺规范源指针")
check("4 份自动生成报告的头部写明「跳帧数字不是交付口径」+ 规范源指针", not _md_bad6h,
      "; ".join(_md_bad6h[:3]) if _md_bad6h else f"{len(_md_stems6g)} 份全部在位")
# 生成器：重跑它们不能把作废跳帧值写回磁盘 ⇒ 脚本里不许出现作废字面量。
_gen_lits6h = ("0.952015", "0.952127", "0.937438", "0.951990", "0.0025pp", "0.9521", "0.9374")
_gen_files6h = [RUNS / "search/make_memory_section.py",
                RUNS / "search/build_final_selection.py",
                RUNS / "search/build_delivery.py"]
_lit6h = []
for _p in _gen_files6h:
    if not _p.exists():
        _lit6h.append(f"{_p.name}: 缺文件")
        continue
    _t6h = _p.read_text(encoding="utf-8")
    for _l6h in _gen_lits6h:
        if re.search(r"(?<![\d.])" + re.escape(_l6h) + r"(?![\d])", _t6h):
            _lit6h.append(f"{_p.name}: {_l6h}")
check("三个生成器脚本内没有作废跳帧字面量（重跑不会写回旧值）", not _lit6h,
      "; ".join(_lit6h[:3]) if _lit6h else f"{len(_gen_files6h)} 个脚本全部干净")

print("\n=== 6i. memory 更新工具链（第 83 轮）===")
# r83: 「文档说的 / 服务器跑的 / 包里带的」三方曾对不上 ——
#   ① 规格书 §7 与正文「脚本留存」都点名了 `insert_memory_section.py`（只插不替）⇒ 正文一改就插出**重复块**且不可逆；
#   ② 正文「脚本留存」只列了事故工具 `merge_blocks.py` / `undo_merge_blocks.py`（都未随包交付），漏了真正在用的 `merge_memory_section.py`；
#   ③ 随包 `merge_memory_section.py` 是手工放的 00:44 版，server `search/` 里跑的却是 22:15 旧版（固定名备份 ⇒ 每次覆盖）。
import shutil
import tempfile

_DOCS6i = [del_dir_art / "artifacts/memory_section_257_18.md",
           del_dir_art / "artifacts/memory_section_257_18_compact.md",
           del_dir_art / "artifacts/trial0080_delta_search_spec_20261119.md",
           del_dir_art / "artifacts/memory_sync_incident_20261119.md",
           del_dir_art / "COMPARISON.md",
           art_dir / "README.md",
           # r84: host 真正会读的是这 6 份正本（§257 十八 之外的 §258/§259/§260 也在里面），
           # 只查包内 5 份文档 ⇒ 并发线新写的段落一旦点名旧工具且不带警示，闸门必然漏检。
           REPO / "memory.md", REPO / "memory_compact.md",
           REPO / "memory/memory.md", REPO / "memory/memory_compact.md",
           REPO / "manu/memory/memory.md", REPO / "manu/memory/memory_compact.md"]
_BLOCK_RE6i = re.compile(r"^## 7\. memory 同步说明", re.M)
# r84: 「禁止」「作废」太泛 —— 无关句子里的一个「禁止」就能把
# 「请用 insert_memory_section.py」蒙混过关（反向对照实测）。只认**工具特有**的危害词。
_WARN6i = ("重复",)
_WARN2_6i = ("不可逆", "只插不替")


def _section_slice(text, start_re):
    """Return the '## ' section starting at start_re (up to the next '## ' heading)."""
    m = start_re.search(text)
    if not m:
        return ""
    nxt = re.search(r"^## ", text[m.end():], re.M)
    return text[m.start():m.end() + nxt.start()] if nxt else text[m.start():]


# ① 当前路径必须被点名：规格书 §7 与正文「脚本留存」段都要写明 merge_memory_section.py
_spec7 = _section_slice(
    (del_dir_art / "artifacts/trial0080_delta_search_spec_20261119.md").read_text(encoding="utf-8"),
    _BLOCK_RE6i)
_main_txt6i = (del_dir_art / "artifacts/memory_section_257_18.md").read_text(encoding="utf-8")
_i6i = _main_txt6i.find("**脚本留存**")
_keep6i = _main_txt6i[_i6i:_i6i + 4000] if _i6i >= 0 else ""
_keep6i = _keep6i.split("\n### ")[0]
# 必须是「**被指明为当前路径**」的那一句里点名，而不是全文任意位置出现过这个名字。
# （句级判定：markdown 里相邻 bullet 之间没有空行会被并成一段，段落级判定会互相掩盖。）
_CUR_PATH6i = re.compile(r"(更新路径|更新工具|当前[^。\n]{0,12}可用)[^。\n]{0,24}merge_memory_section\.py")


# r84: 光「同句共现」不够 —— 「当前更新工具 = merge_memory_section.py（其实已废弃，别用）」也会命中。
# 现在按**句**判定，并要求那句**没有**把它否掉。
_NEG6i = ("勿用", "禁用", "作废", "废弃", "别用", "不要用", "停用", "已弃用", "不可用")


def _named_as_current(block):
    """True iff some sentence names merge_memory_section.py AS the current update path/tool
    and that very sentence does not disown it."""
    for _line6i in block.splitlines():
        for _sent6i in re.split(r"(?<=[。；;])", _line6i):
            if _CUR_PATH6i.search(_sent6i) and not any(_n6i in _sent6i for _n6i in _NEG6i):
                return True
    return False


_missing6i = []
if not _named_as_current(_spec7):
    _missing6i.append("规格书 §7 未把 merge_memory_section.py 指明为当前更新路径")
if not _named_as_current(_keep6i):
    _missing6i.append("正文「脚本留存」段未把 merge_memory_section.py 指明为当前更新工具")
check("memory 更新路径（规格书 §7 + 正文「脚本留存」）都把 merge_memory_section.py 指明为当前路径", not _missing6i,
      "; ".join(_missing6i) if _missing6i else "两处都在位")
# ② 提到 insert_memory_section.py 的每一个 **bullet/段落** 都必须带「重复/不可逆/禁止/只插不替/作废」警示
#    （bullet 级切分：相邻 bullet 之间没有空行时，段落级判定会被隔壁 bullet 的警示词掩盖。）
_BULLET6i = re.compile(r"(?m)^(?:[-*+] |\d+\. |#|\|)")


def _blocks6i(text):
    """Split markdown text into bullet/paragraph blocks (a blank line also breaks a block)."""
    _out6i = []
    _cur6i = []
    for _l6i in text.splitlines():
        if not _l6i.strip():
            if _cur6i:
                _out6i.append("\n".join(_cur6i))
                _cur6i = []
            continue
        if _BULLET6i.match(_l6i) and _cur6i:
            _out6i.append("\n".join(_cur6i))
            _cur6i = []
        _cur6i.append(_l6i)
    if _cur6i:
        _out6i.append("\n".join(_cur6i))
    return _out6i


_nowarn6i = []
for _p6i in _DOCS6i:
    if not _p6i.exists():
        continue
    for _blk6i in _blocks6i(_p6i.read_text(encoding="utf-8")):
        if "insert_memory_section.py" not in _blk6i:
            continue
        if "重复" not in _blk6i or not any(_w in _blk6i for _w in _WARN2_6i):
            _nowarn6i.append(f"{_p6i.name}: " + " ".join(_blk6i.split())[:70])
check("凡提到 insert_memory_section.py 的段落都带「重复块」+「不可逆/只插不替」警示", not _nowarn6i,
      "; ".join(_nowarn6i[:2]) if _nowarn6i else f"{len(_DOCS6i)} 份文档全部合规")
# ③ 「脚本留存」清单**分栏**校验（r84）：
#    「随包交付」栏点名的脚本必须**在包内**真实存在 —— host 拿到包就要能跑；
#    「未随包」栏点名的脚本必须**不在**包内 —— 否则清单在骗人。
#    名字按 basename 归一 —— 清单里写 `artifacts/xxx.py` 也要能被抓到（旧的正则会把它整个漏掉 ⇒ 判据空转）。
_keep_blocks6i = _blocks6i(_keep6i)
_ship6i = [b for b in _keep_blocks6i if "随包交付" in b and "未随包" not in b]
_acc6i = [b for b in _keep_blocks6i if "未随包" in b]


def _py_names6i(blocks):
    _s6i = set()
    for _b6i in blocks:
        for _m6i in re.findall(r"`([^`]+\.py)`", _b6i):
            _s6i.add(Path(_m6i).name)
    return _s6i


_ship_names6i = _py_names6i(_ship6i)
_absent6i = [n6i for n6i in sorted(_ship_names6i)
             if not ((del_dir_art / n6i).exists() or (art_dir / n6i).exists())]
check("「随包交付」栏点名的脚本都在包内真实存在（host 拿到就能跑）", not _absent6i,
      "; ".join(_absent6i[:4]) if _absent6i else f"{len(_ship_names6i)} 个脚本都在包内")
_acc_names6i = _py_names6i(_acc6i)
_leak6i = [n6i for n6i in sorted(_acc_names6i)
           if (del_dir_art / n6i).exists() or (art_dir / n6i).exists()]
check("「未随包」栏点名的脚本确实没在包里（清单不骗人）", not _leak6i,
      "; ".join(_leak6i[:4]) if _leak6i else f"{len(_acc_names6i)} 个脚本确认只在 server 侧")
# ④ 随包的合并工具 == server 侧那份（逐字节），且 artifacts/ 里不得出现第二份
_pkg_merge6i = del_dir_art / "merge_memory_section.py"
_srv_merge6i = RUNS / "search/merge_memory_section.py"
_same6i = (_pkg_merge6i.exists() and _srv_merge6i.exists()
           and _pkg_merge6i.read_bytes() == _srv_merge6i.read_bytes())
check("随包 merge_memory_section.py == server 侧那份（逐字节；同一工具只有一个版本）", _same6i,
      f"包内={hashlib.md5(_pkg_merge6i.read_bytes()).hexdigest() if _pkg_merge6i.exists() else 'MISSING'} / "
      f"server={hashlib.md5(_srv_merge6i.read_bytes()).hexdigest() if _srv_merge6i.exists() else 'MISSING'}")
_dup6i = (art_dir / "merge_memory_section.py").exists()
check("artifacts/ 里没有第二份 merge_memory_section.py（同一份东西只有一个版本）", not _dup6i,
      "artifacts/ 无副本" if not _dup6i else "artifacts/ 里也有同名的")
# ⑤ 实测负证据：insert_memory_section.py 对「已有 §257 十八 块」的文件会插出重复块，且 merge 只改第一个
_ins6i = RUNS / "search/insert_memory_section.py"
_mrg6i = RUNS / "search/merge_memory_section.py"
_body6i = RUNS / "search/memory_section_257_18.md"
if _ins6i.exists() and _mrg6i.exists() and _body6i.exists() and (REPO / "memory/memory.md").exists():
    with tempfile.TemporaryDirectory() as _td6i:
        _tgt6i = Path(_td6i) / "memory.md"
        # r85: 这条探针是整条链里**唯一**还会 exec 危险工具 insert_memory_section.py 的地方，
        #   而它的安全性只靠「目标在 tmpdir 里」这一条**调用方约定** ⇒ 把约定变成硬断言：
        #   将来谁把 _tgt6i 指向真实 memory.md，闸门当场拦下，而不是等事故发生。
        try:
            _tgt6i.resolve().relative_to(Path(_td6i).resolve())
        except ValueError:
            raise AssertionError("探针目标必须落在 tmpdir 内（不得指向真实 memory.md）")
        shutil.copy2(REPO / "memory/memory.md", _tgt6i)
        _nb6i = Path(_td6i) / "body_next.md"
        _nb6i.write_text(
            _body6i.read_text(encoding="utf-8").rstrip("\n") + "\n\n### probe\n\n- probe\n",
            encoding="utf-8")
        subprocess.run([sys.executable, str(_ins6i), str(_tgt6i), "§257", str(_nb6i)],
                       capture_output=True, text=True)
        # 只数**标题行**（正文里引用「## §257 十八」的字样不算块）
        _heads6i = lambda pth: sum(1 for l in pth.read_text(encoding="utf-8").splitlines()
                                   if l.startswith("## §257 十八"))
        _dup_blocks6i = _heads6i(_tgt6i)
        subprocess.run([sys.executable, str(_mrg6i), str(_tgt6i), str(_body6i)],
                       capture_output=True, text=True)
        _after_merge6i = _heads6i(_tgt6i)
    check("实测：insert_memory_section.py 会插出重复的 §257 十八 块（所以不能用来更新本段）",
          _dup_blocks6i == 2, f"插入后标题块数={_dup_blocks6i}（应为 2 = 原 1 + 插 1）")
    check("实测：重复块不可逆（merge 只替换第一个块）", _after_merge6i == 2,
          f"merge 后标题块数={_after_merge6i} ⇒ 重复块清不掉")
else:
    check("实测：insert_memory_section.py 会插出重复的 §257 十八 块（所以不能用来更新本段）", False, "缺脚本/正文")
    check("实测：重复块不可逆（merge 只替换第一个块）", False, "缺脚本/正文")
# ⑥ 「host 侧可直接用」必须名副其实（r84）：源码里依赖 server 跑批路径的脚本在 host 上必然跑不通。
#    旧清单把 `verify_all.py` 写进「可在 host 侧直接用」，而它自带 host 侧守卫 ⇒ 照做第一步就 exit 2。
_SRV_RUNS6i = "/mnt/data/siping/runs/trial0080_onnx"


def _needs_server6i(name6i):
    for _c6i in (del_dir_art / name6i, art_dir / name6i):
        if _c6i.exists():
            return _SRV_RUNS6i in _c6i.read_text(encoding="utf-8")
    return False


_host_names6i = _py_names6i([b for b in _ship6i if "host 侧可直接用" in b])
_bad_host6i = [n6i for n6i in sorted(_host_names6i) if _needs_server6i(n6i)]
check("「host 侧可直接用」栏的脚本不依赖 server 跑批路径（否则 host 上必然跑不通）", not _bad_host6i,
      "; ".join(_bad_host6i) if _bad_host6i else f"{len(_host_names6i)} 个脚本标注与源码相符")
_srv_names6i = _py_names6i([b for b in _ship6i if "仅服务器侧可跑" in b])
_bad_srv6i = [n6i for n6i in sorted(_srv_names6i) if not _needs_server6i(n6i)]
check("「仅服务器侧可跑」栏的脚本确实依赖 server 跑批路径（标注与源码相符）", not _bad_srv6i,
      "; ".join(_bad_srv6i) if _bad_srv6i else f"{len(_srv_names6i)} 个脚本标注与源码相符")

print("\n=== 6j. 参数文件唯一性 + 整档覆盖防线（第 85 轮）===")
import shutil
import tempfile

_del_params6j = (doc or {}).get("params", {})


def _stale_param_files6j(root6j, delivery6j):
    """Return params-shaped json files whose effective params differ from the delivery params.

    Mirrors ``system_sota_longquan.py::load_params`` (``payload.get("params", payload)``):
    the nested ``{"params": {...}}`` form and the flat top-level form are **both** accepted
    silently, so handing it the wrong file is never rejected — it just produces a wrong run.
    """
    bad6j = []
    for path6j in sorted(root6j.rglob("*.json")):
        try:
            payload6j = json.loads(path6j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload6j, dict):
            continue
        eff6j = payload6j.get("params", payload6j)
        if not isinstance(eff6j, dict) or "th_base" not in eff6j:
            continue  # not a params file (e.g. the search-space definition)
        if eff6j != delivery6j:
            bad6j.append(path6j.name)
    return bad6j


# ① 交付包内只允许存在一份参数文件 —— 与交付参数逐键相等的那份。
_bad6j = _stale_param_files6j(del_dir_art, _del_params6j)
check("交付包内能被 load_params 静默接受的参数文件，只有交付参数那一份", not _bad6j,
      "; ".join(_bad6j) if _bad6j else "包内无「有效参数 ≠ 交付参数」的参数文件")

# ② 那两份被拒的 objective-best 参数文件必须不在包内。
_gone6j = ("opt_params_t0080_v3_final_flat.json", "opt_params_t0080_v3_seedfix_flat.json")
_still6j = [n6j for n6j in _gone6j if (del_dir_art / "artifacts" / n6j).exists()]
check("被拒的 objective-best 参数文件已从交付包删除（曾装 match_dist 16.0 / max_age_phys 10 ⇒ 全量 FAR 0.035152 破线）",
      not _still6j, "仍在包内: " + ", ".join(_still6j) if _still6j else "两份均已移除")

# ③ 反向对照：判据必须真的会响（否则 ① 是空转）。
with tempfile.TemporaryDirectory() as _td6j:
    _probe6j = Path(_td6j) / "pkg"
    shutil.copytree(del_dir_art, _probe6j, ignore=shutil.ignore_patterns("*.bak*", "__pycache__"))
    _stale6j = {"params": dict(_del_params6j, match_dist=16.0, max_age_phys=10)}
    (_probe6j / "artifacts" / "opt_params_PROBE.json").write_text(json.dumps(_stale6j), encoding="utf-8")
    _probe_bad6j = _stale_param_files6j(_probe6j, _del_params6j)
check("反向对照：往包里塞一份「非交付参数」，判据必须报警（否则判据空转）",
      "opt_params_PROBE.json" in _probe_bad6j, f"探针命中: {_probe_bad6j}")

# ④ 危害不是推测：load_params 真的会静默吃下那组被拒参数。
_stale_src6j = RUNS / "search/opt_params_t0080_v3_final_flat.json"
if _stale_src6j.exists():
    _dont_write6j = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from manu.evaluation.system_sota_longquan import load_params as _lp6j
        _eff6j = _lp6j(str(_stale_src6j))
    finally:
        sys.dont_write_bytecode = _dont_write6j
    check("实测：load_params 静默接受被拒参数（喂旧文件不报错 ⇒ 直接跑出 FAR 破线结果）",
          _eff6j.get("match_dist") == 16.0 and _eff6j.get("max_age_phys") == 10,
          f"match_dist={_eff6j.get('match_dist')} max_age_phys={_eff6j.get('max_age_phys')}")
else:
    check("实测：load_params 静默接受被拒参数（喂旧文件不报错 ⇒ 直接跑出 FAR 破线结果）", False, "缺旧参数文件")

# ⑤ 整树 rsync 不得有能力整档覆盖 memory（host → server 方向）。
_up6j = REPO / "manu/videos/upload_to_server.sh"
_up_txt6j = _up6j.read_text(encoding="utf-8") if _up6j.exists() else ""
_mem6j = ("/memory.md", "/memory_compact.md", "/memory/memory.md", "/memory/memory_compact.md",
          "/manu/memory/memory.md", "/manu/memory/memory_compact.md")
_miss6j = [m6j for m6j in _mem6j if f"--exclude={m6j}" not in _up_txt6j]
check("upload_to_server.sh 的整树 rsync 排除 6 份 memory（否则 host 副本整档覆盖 server 正文）",
      not _miss6j, "未排除: " + ", ".join(_miss6j) if _miss6j else "6 份全部排除")

print("\n=== 7. 交付包 ===")
del_dir = REPO / "manu/memory/delivery_20261119"
check("交付包存在", del_dir.exists())
# r84: 交付包里不得有字节码 —— 它是交付件的「第二份」（且可能是陈旧的），
# 两条回填路虽然都排除了它，但它躺在包里就说明包没被打扫干净。
_bytecode7 = sorted(str(p7.relative_to(del_dir)) for p7 in del_dir.rglob("__pycache__") if p7.is_dir())
_bytecode7 += sorted(str(p7.relative_to(del_dir)) for p7 in del_dir.rglob("*.pyc") if p7.is_file())
check("交付包里没有 __pycache__ / .pyc 字节码（同一份东西只能有一个版本）", not _bytecode7,
      "; ".join(_bytecode7[:4]) if _bytecode7 else "干净")
for f in ("COMPARISON.md", "osd_manifest.json",
          "system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json"):
    check(f"交付包含 {f}", (del_dir / f).exists())
man = load(del_dir / "osd_manifest.json")
if man:
    check("清单 total = 24", man.get("total") == 24, str(man.get("total")))
    check("清单记录对照目录 pass/un-pass", man.get("reference_dir_pass") == 21 and man.get("reference_dir_un_pass") == 3)
    # The manifest ships to the host, where someone re-checks the mp4 sizes: keep the
    # byte count honest and the MB unit stated, or the check silently misses by 10^6.
    osd_dir = Path(man["osd_dir"])
    real_bytes = sum((osd_dir / f["group"] / f["file"]).stat().st_size for f in man["files"])
    check("清单 total_bytes == 磁盘实际总字节",
          man.get("total_bytes") == real_bytes, f"{man.get('total_bytes')} vs {real_bytes}")
    check("清单 total_mb 与 total_bytes 同为十进制 MB 口径",
          man.get("total_mb") == round(real_bytes / 1e6, 1), str(man.get("total_mb")))

print("\n=== 8. 主机回填链路（server -> host）===")
DEL = REPO / "manu/memory/delivery_20261119"
for f in ("HOST_PUSH.sh", "HOST_PUSH_ONCE.sh", "HOST_PUSH_WATCH.sh", "merge_memory_section.py",
          "artifacts/memory_section_257_18.md", "artifacts/memory_section_257_18_compact.md"):
    check(f"交付包含 {f}", (DEL / f).exists())
watch_n = subprocess.run(["bash", "-n", str(DEL / "HOST_PUSH_WATCH.sh")], capture_output=True, text=True)
check("HOST_PUSH_WATCH.sh 语法检查 (bash -n)", watch_n.returncode == 0, watch_n.stderr.strip()[:120])
once_n = subprocess.run(["bash", "-n", str(DEL / "HOST_PUSH_ONCE.sh")], capture_output=True, text=True)
check("HOST_PUSH_ONCE.sh 语法检查 (bash -n)", once_n.returncode == 0, once_n.stderr.strip()[:120])
once_src = (DEL / "HOST_PUSH_ONCE.sh").read_text(encoding="utf-8")
check("零人工路径真的调 HOST_PUSH.sh（否则每次值守等于没推）",
      "HOST_PUSH.sh" in once_src and "exit 3" in once_src and "exit 0" in once_src)
mem_main = (REPO / "memory/memory.md").read_text(encoding="utf-8")
check("记忆含 §257 十八之二十四（确定性采样器重测的口径更正）",
      "### 二十四、🔴 口径更正（第二轮" in mem_main)
check("记忆含 §257 十八之二十五（两条回填路补齐交付包）",
      "### 二十五、🔴 两条回填路补齐「交付包」" in mem_main)
# All six copies must carry the new sub-section -- checking two of them is how a
# half-updated memory archive slips through.
check("记忆含 §257 十八之二十六（交付参数档重跑 track 臂的可复现性闭环）",
      all("### 二十六、🔴 可复现性闭环" in (REPO / rel).read_text(encoding="utf-8")
          for rel in targets),
      f"{len(targets)} 份全查")

# Recompute the reproducibility claim from disk instead of grepping the author's own
# summary file -- the summary file is evidence, not the source of truth.
PREDS_A = RUNS / "eval/preds_v3_optgapfit"
PREDS_B = RUNS / "eval/repro/preds_v3_optgapfit_repro"
PREDS_C = RUNS / "eval/repro3/preds_v3_optgapfit_repro3"


def _md5_map(d):
    return {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(d.iterdir()) if f.is_file()}


ma = mb = mc = {}
if PREDS_A.is_dir() and PREDS_B.is_dir() and PREDS_C.is_dir():
    ma, mb, mc = _md5_map(PREDS_A), _md5_map(PREDS_B), _md5_map(PREDS_C)
    check("预测目录逐位可复现（3 次跑批现场重算 md5，非引用证据文件）",
          ma == mb == mc and len(ma) == 25,
          f"files={len(ma)}/{len(mb)}/{len(mc)} "
          f"差异={[n for n in ma if not (ma[n] == mb.get(n) == mc.get(n))] or '无'}")
    ev_a = load(RUNS / "eval/track_all_v3_optgapfit.json")
    ev_b = load(RUNS / "eval/repro/track_all_v3_optgapfit_repro.json")
    ev_c = load(RUNS / "eval/repro3/track_all_v3_optgapfit_repro3.json")

    def _flat(d, p=""):
        o = {}
        if isinstance(d, dict):
            for k, v in d.items():
                o.update(_flat(v, f"{p}.{k}" if p else k))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                o.update(_flat(v, f"{p}[{i}]"))
        else:
            o[p] = d
        return o

    if ev_a and ev_b and ev_c:
        fa, fb, fc = _flat(ev_a), _flat(ev_b), _flat(ev_c)
        diff_b = sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])
        diff_c = sorted(k for k in set(fa) & set(fc) if fa[k] != fc[k])
        check("评测 JSON 320 键、仅 elapsed_seconds 不同（3 次跑批现场展平比对）",
              len(fa) == len(fb) == len(fc) == 320
              and diff_b == ["elapsed_seconds"] and diff_c == ["elapsed_seconds"],
              f"keys={len(fa)}/{len(fb)}/{len(fc)} 差异={diff_b} {diff_c}")
        check("三次跑批的 overall 与 24 条 per_sequence 逐位一致",
              fa.get("overall") == fb.get("overall") == fc.get("overall")
              and fa.get("per_sequence") == fb.get("per_sequence") == fc.get("per_sequence")
              and len(ev_a["per_sequence"]) == 24)
        check("三次跑批用的确实是交付参数档（config.params 逐键相等）",
              ev_a.get("config", {}).get("params") == doc["params"]
              and ev_b.get("config", {}).get("params") == doc["params"]
              and ev_c.get("config", {}).get("params") == doc["params"])
        check("三次跑批的 elapsed_seconds 各不相同（耗时随磁盘竞争浮动）",
              len({ev_a["elapsed_seconds"], ev_b["elapsed_seconds"], ev_c["elapsed_seconds"]}) == 3,
              f"{ev_a['elapsed_seconds']:.2f} / {ev_b['elapsed_seconds']:.2f} / {ev_c['elapsed_seconds']:.2f} s")
    else:
        check("重跑的评测 JSON 在盘上", False, "缺 track_all_v3_optgapfit_repro[3].json")
else:
    check("重跑的预测目录在盘上（可现场重算 md5）", False, f"缺 {PREDS_B} 或 {PREDS_C}")

# The wall-clock spread is only explained if it is measured, not asserted: run3 carried
# a concurrent iostat sampler.
io_txt = (DEL / "artifacts/repro3_iostat.log").read_text(encoding="utf-8") if (DEL / "artifacts/repro3_iostat.log").exists() else ""
sda = [l.split() for l in io_txt.splitlines() if l.startswith("sda")][1:]
_utils = [float(x[-1]) for x in sda] if sda else []
_rkb = [float(x[2]) for x in sda] if sda else []
check("耗时差异有同期 iostat 留证（sda %util ≈ 100% ⇒ I/O 瓶颈）",
      bool(_utils) and min(_utils) >= 90.0 and max(_utils) <= 100.5,
      f"sda 采样={len(_utils)} %util {min(_utils):.1f}~{max(_utils):.1f} 读 {min(_rkb):.0f}~{max(_rkb):.0f} KB/s")

repro_evidence = (DEL / "artifacts/repro_preds_md5.txt")
_ev_txt = repro_evidence.read_text(encoding="utf-8") if repro_evidence.exists() else ""
_ev_md5 = {name: digest for digest, name in re.findall(r"^([0-9a-f]{32})  (\S+)$", _ev_txt, re.M)}
check("证据文件在交付包里，且它列的 md5 与磁盘现场重算一致",
      repro_evidence.exists() and len(_ev_md5) == 25
      and all(_ev_md5.get(n) == h for n, h in ma.items()),
      f"evidence 条目={len(_ev_md5)}")
check("重跑产物在交付包里（2 份 log + iostat + 2 份 eval json）",
      (DEL / "artifacts/repro_track_arm.log").exists()
      and (DEL / "artifacts/repro3_track_arm.log").exists()
      and (DEL / "artifacts/repro3_iostat.log").exists()
      and (DEL / "artifacts/track_all_v3_optgapfit_repro.json").exists()
      and (DEL / "artifacts/track_all_v3_optgapfit_repro3.json").exists())
push_repro = (DEL / "HOST_PUSH.sh").read_text(encoding="utf-8")
# The push path ships the whole artifacts/ directory instead of a hand-maintained
# whitelist, so "the host receives file X" means "X is in artifacts/ and the push
# copies that directory" -- asserting on the script's file list would only pin the
# mechanism and go stale the moment evidence is added.
push_ships_artifacts = ('find "$HERE/artifacts"' in push_repro
                        and '${PKG_ARTIFACTS[@]}' in push_repro)
check("推送路搬全 6 份可复现性证据",
      push_ships_artifacts
      and all((DEL / f"artifacts/{n}").exists() for n in ("repro_preds_md5.txt",
                                                          "repro_track_arm.log",
                                                          "repro3_track_arm.log",
                                                          "repro3_iostat.log",
                                                          "track_all_v3_optgapfit_repro.json",
                                                          "track_all_v3_optgapfit_repro3.json")))
check("记忆如实记录：本容器无可用守护机制（不谎称守望常驻）",
      "没有可用守护机制" in mem_main and "HOST_PUSH_ONCE.sh" in mem_main)
push = (DEL / "HOST_PUSH.sh").read_text(encoding="utf-8")
check("推送路也搬交付包（参数档 / 对照表 / OSD 清单）",
      "COMPARISON.md" in push and "osd_manifest.json" in push and "delivery_20261119" in push)
backfill = (DEL / "HOST_BACKFILL.sh").read_text(encoding="utf-8")
backfill_n = subprocess.run(["bash", "-n", str(DEL / "HOST_BACKFILL.sh")], capture_output=True, text=True)
check("HOST_BACKFILL.sh 语法检查 (bash -n)", backfill_n.returncode == 0, backfill_n.stderr.strip()[:120])
check("拉取路也搬交付包",
      "拉取交付包" in backfill and "artifacts/" in backfill)
check("HOST_BACKFILL.sh 走章节级合并（不再整文件覆盖）",
      "merge_memory_section.py" in backfill and "${SERVER_ROOT}/${f}" not in backfill)
check("HOST_BACKFILL.sh 探测按三类分流（路由/主机密钥/授权）",
      "Host key verification failed" in backfill and "Permission denied" in backfill)
bash_n = subprocess.run(["bash", "-n", str(DEL / "HOST_PUSH.sh")], capture_output=True, text=True)
check("HOST_PUSH.sh 语法检查 (bash -n)", bash_n.returncode == 0, bash_n.stderr.strip()[:120])
# r84: 两条回填路此前筛选规则不一致 —— 推送路排掉 *.bak*，拉取路只排 __pycache__，
# 于是拉取路会把 12 个 .bak*（含 4 个**旧版闸门脚本**）带到 host。
check("两条回填路的 artifacts 筛选规则一致（都排除 *.bak*）",
      "--exclude '*.bak*'" in backfill and "! -name '*.bak*'" in push,
      "两路都排 *.bak*" if ("--exclude '*.bak*'" in backfill and "! -name '*.bak*'" in push)
      else "筛选规则不一致")
# r84: 原来用 `python -m py_compile` 做语法检查 —— 它会把 .pyc 写进**交付包**（`__pycache__/`），
# 闸门自己跑一次就污染一次包（还留下一份指向旧版 verify_all.py 的陈旧字节码）。
# 改成进程内 compile()：同样验语法，零磁盘副产物。
try:
    compile((DEL / "merge_memory_section.py").read_text(encoding="utf-8"),
            str(DEL / "merge_memory_section.py"), "exec")
    _py_ok, _py_err = True, ""
except SyntaxError as _e6:
    _py_ok, _py_err = False, str(_e6)
check("merge_memory_section.py 语法检查", _py_ok, _py_err[:120])

# The manifest must match the files that are actually on disk right now.
manifest = (DEL / "artifacts/memory_md5_manifest.txt").read_text(encoding="utf-8")
man_files = {}
man_sec = {}
for line in manifest.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        m = re.match(r"^#\s+sec\s+(\S+)\s+=\s+([0-9a-f]{32})", line)
        if m:
            man_sec[m.group(1)] = m.group(2)
        continue
    digest, _, rel = line.partition("  ")
    man_files[rel] = digest


def _sec_md5(rel):
    """md5 of the '## §257 十八' block in one memory copy (the block this line owns)."""
    p = REPO / rel
    if not p.exists():
        return None
    ls = p.read_text(encoding="utf-8").splitlines(keepends=True)
    idx = [i for i, l in enumerate(ls) if l.startswith("## §")]
    for n, s in enumerate(idx):
        if ls[s].startswith("## §257 十八"):
            e = idx[n + 1] if n + 1 < len(idx) else len(ls)
            return hashlib.md5(("".join(ls[s:e]).rstrip("\n") + "\n").encode()).hexdigest()
    return None


# GATE: the block this line owns.  The concurrent line appends §258/§259/§260 to the
# same files, so a whole-file md5 comparison would fail (or hide a real break) the
# moment the other line writes -- the section md5 is immune to that.
body_full_txt = (DEL / "artifacts/memory_section_257_18.md").read_text(encoding="utf-8")
body_cmp_txt = (DEL / "artifacts/memory_section_257_18_compact.md").read_text(encoding="utf-8")
want_full = hashlib.md5(body_full_txt.encode()).hexdigest()
want_cmp = hashlib.md5(body_cmp_txt.encode()).hexdigest()
sec_bad = []
for rel in targets:
    want = want_cmp if "compact" in rel else want_full
    if _sec_md5(rel) != want:
        sec_bad.append(rel)
check("6 份 memory 的 §257 十八 段 md5 == 发布正文 md5（本线闸门，并发线改整档不影响）",
      not sec_bad and len(man_sec) == 6, f"manifest sec={len(man_sec)} drift={sec_bad}")
# INFORMATIONAL: whole-file snapshot.  A difference here just means the concurrent
# line edited the file after this snapshot was taken -- not a delivery break.
drift = [rel for rel, digest in man_files.items()
         if not (REPO / rel).exists() or hashlib.md5((REPO / rel).read_bytes()).hexdigest() != digest]
check("manifest 整档 md5 快照（信息项：并发线会改整档，不判 FAIL）", len(man_files) == 6,
      f"files={len(man_files)} 被并发线改动={drift or '无'}")
body_src = Path("/mnt/data/siping/runs/trial0080_onnx/search/memory_section_257_18.md")
body_cmp = Path("/mnt/data/siping/runs/trial0080_onnx/search/memory_section_257_18_compact.md")
check("manifest 记录了两份正文的 md5",
      hashlib.md5(body_src.read_bytes()).hexdigest() in manifest
      and hashlib.md5(body_cmp.read_bytes()).hexdigest() in manifest)
check("manifest 有可重跑的生成器（regen_manifest.py 随包交付）",
      (DEL / "artifacts/regen_manifest.py").exists()
      and push_ships_artifacts)
check("§257 十八 含「二十二」主机回填小节",
      "### 二十二" in (REPO / "memory/memory.md").read_text(encoding="utf-8"))
check("§257 十八 含「二十三」两路合并 + 守望小节",
      "### 二十三" in (REPO / "memory/memory.md").read_text(encoding="utf-8")
      and "HOST_PUSH_WATCH" in (REPO / "memory/memory.md").read_text(encoding="utf-8"))
watch_log = Path("/mnt/data/siping/runs/trial0080_onnx/search/host_push_watch.log")
check("推送日志存在（每次值守调 HOST_PUSH_ONCE.sh 都会追加）", watch_log.exists(),
      watch_log.read_text(encoding="utf-8").strip().splitlines()[-1][:70] if watch_log.exists() else "")

# The merge tool must INSERT / REPLACE / SKIP, and must never touch bytes outside
# its own section -- that is the whole point (the host owns a concurrent §258).
import shutil
import tempfile
def _mk(text, tag):
    ls = text.splitlines(keepends=True)
    idx = [i for i, l in enumerate(ls) if l.startswith("## §")]
    for n, s0 in enumerate(idx):
        if ls[s0].startswith("## §257 十八"):
            e = idx[n + 1] if n + 1 < len(idx) else len(ls)
            out = "".join(ls[:s0]) + "".join(ls[e:])
            break
    ls2 = out.splitlines(keepends=True)
    i258 = [i for i, l in enumerate(ls2) if l.startswith("## §258")][0]
    ls2.insert(i258 + 1, f"<!-- HOST-ONLY {tag} -->\n")
    return "".join(ls2)

def _outside(text):
    ls = text.splitlines(keepends=True)
    idx = [i for i, l in enumerate(ls) if l.startswith("## §")]
    for n, s0 in enumerate(idx):
        if ls[s0].startswith("## §257 十八"):
            e = idx[n + 1] if n + 1 < len(idx) else len(ls)
            return "".join(ls[:s0]) + "".join(ls[e:])
    return text

main_txt = (REPO / "memory/memory.md").read_text(encoding="utf-8")
tool = DEL / "merge_memory_section.py"
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # (a) host has NO section -> INSERT
    a = td / "a.md"; a.write_text(_mk(main_txt, "A"), encoding="utf-8")
    before = a.read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(tool), str(a), str(body_src)], capture_output=True, text=True)
    after = a.read_text(encoding="utf-8")
    check("merge 工具：host 无该段 -> 插入", "### 二十二" in after and after != before)
    check("merge 工具：host 独有内容存活", "<!-- HOST-ONLY A -->" in after)
    check("merge 工具：块外字节逐字节不变", _outside(after) == _outside(before))
    # (b) host has an OLD section -> REPLACE
    b = td / "b.md"
    ls = main_txt.splitlines(keepends=True)
    idx = [i for i, l in enumerate(ls) if l.startswith("## §")]
    for n, s0 in enumerate(idx):
        if ls[s0].startswith("## §257 十八"):
            e = idx[n + 1] if n + 1 < len(idx) else len(ls)
            b.write_text("".join(ls[:s0]) + "## §257 十八 OLDPLACEHOLDER\n\n旧版本\n" + "".join(ls[e:]), encoding="utf-8")
            break
    subprocess.run([sys.executable, str(tool), str(b), str(body_src)], capture_output=True, text=True)
    after_b = b.read_text(encoding="utf-8")
    check("merge 工具：host 有旧版 -> 原位替换",
          "### 二十二" in after_b and "OLDPLACEHOLDER" not in after_b and "旧版本" not in after_b)
    # (c) host already identical -> SKIP (no write at all)
    c = td / "c.md"; c.write_text(main_txt, encoding="utf-8")
    r = subprocess.run([sys.executable, str(tool), str(c), str(body_src)], capture_output=True, text=True)
    check("merge 工具：已一致 -> no-op", "[SKIP]" in r.stdout and c.read_text(encoding="utf-8") == main_txt)

print("\n=== 9. 闸门项数自证（README 头条不得与实际项数脱节）===")
# r86: README 清单里 verify_all.py 那一行的头条写着「N 项 PASS / 0 FAIL」——
#   第 85 轮加了 5 项却没回头改这行 ⇒ 同一文件里「184」与「184 → 189」自相矛盾，
#   读者查交付包时看到的仍是过期项数（而这行正是「闸门有多少项」的唯一对外声明）。
#   「文档写着的数字」这类漂移必须能被机器发现 ⇒ 末尾现算项数，与 README 头条对账。
#   （本条 check 自己也算一项 ⇒ 期望值 = 已跑项数 + 1）
_readme6k = (del_dir_art / "artifacts/README.md").read_text(encoding="utf-8")
_m6k = re.findall(r"(\d+)\s*项 PASS\s*/\s*0 FAIL", _readme6k)
_expect6k = _checks + 1
check("README 清单里 verify_all.py 行的闸门项数 == 实际跑出的项数",
      len(_m6k) == 1 and int(_m6k[0]) == _expect6k,
      f"README={_m6k[0] if _m6k else '未找到/多处'} 实际={_expect6k}")

print("\n" + ("=" * 60))
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
print("=" * 60)
raise SystemExit(0 if ok else 1)
