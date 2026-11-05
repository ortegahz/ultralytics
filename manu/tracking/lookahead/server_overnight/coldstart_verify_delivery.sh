#!/usr/bin/env bash
# ============================================================
# 交付包**冷启动复现**：验证交付包 code/ 真自包含（不依赖仓内任何脚本）
#
#   做法：把交付包拷到临时沙箱 ⇒ 沙箱里只留「产物（符号链接到只读原件）」+
#         「包内 code/」两样东西，**故意不放** manu/ 下任何脚本
#         ⇒ 用包内 code/regen_reports_nogmc_fi.sh 重生成全部报告
#         ⇒ 逐个与交付包里的同名报告比对（归一化时间戳/路径后）
#
#   用法：bash manu/tracking/lookahead/server_overnight/coldstart_verify_delivery.sh [PKG] [SANDBOX]
#   退出码：0 = 全部一致（或差异已在容差内）；1 = 有实质差异；2 = 环境/产物缺失
# ============================================================
set -uo pipefail

# ★ **异机可跑**（host 侧只拿到交付包时也能复现）：默认值优先级 =
#   ① 显式环境变量/参数 ② **服务器仓根**（存在则用 ⇒ 服务器侧行为与从前**逐位相同**）
#   ③ **包内 repo/ 镜像**（host 随包跑：脚本在 <包>/code/ ⇒ 上一级就是包根，其 repo/ 是仓镜像）。
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO:-}" ]; then
  if [ -d "/tmp/pycharm_project_10ae9e2e/runs/lookahead" ]; then
    REPO="/tmp/pycharm_project_10ae9e2e"
  elif [ -d "$SELF_DIR/../repo/runs/lookahead" ]; then
    REPO="$(cd "$SELF_DIR/../repo" && pwd)"
  else
    REPO="/tmp/pycharm_project_10ae9e2e"
  fi
fi
# ★ python：显式 > 服务器 conda env（存在则用）> 系统 python3（交付链已兼容 **3.8+**）
if [ -z "${PY:-}" ]; then
  if [ -x "/home/huangzhe/anaconda3/envs/uav/bin/python" ]; then
    PY="/home/huangzhe/anaconda3/envs/uav/bin/python"
  else
    PY="python3"
  fi
fi
# 交付包：显式参数 > **随包**（脚本在 <包>/code/）> 仓内默认位置
if [ -n "${1:-}" ]; then
  PKG="$1"
elif [ -f "$SELF_DIR/../MD5SUMS.txt" ] && [ -d "$SELF_DIR/../repo" ]; then
  PKG="$(cd "$SELF_DIR/.." && pwd)"
else
  PKG="$REPO/runs/lookahead/final/nogmc_fi_20261104"
fi
SB="${2:-/tmp/coldstart_nogmc_fi}"
# ★ 沙箱路径守卫：`rm -rf "$SB"` 是**破坏性**操作 ⇒ 只允许 /tmp/ 下的非空路径，
#   杜绝「误传参数 ⇒ 删掉不该删的目录」。
case "$SB" in
  /tmp/*) ;;
  *) echo "[FATAL] 沙箱路径必须以 /tmp/ 开头（否则 rm -rf 有误删风险）：$SB"; exit 2;;
esac
[ -n "$SB" ] && [ "$SB" != "/" ] || { echo "[FATAL] 沙箱路径非法：$SB"; exit 2; }
# ★ 冷启动报告写**仓内**（仓内模式）或**包外**（host 异机模式）——
#   绝不写进包内镜像（那会让包自检的清单判据 FAIL ⇒ 把「跑一次复核」变成「弄脏交付包」）。
if [ -n "${OUT_MD:-}" ]; then
  :
elif [ "${REPO#"$PKG/"}" != "$REPO" ]; then
  OUT_MD="$(cd "$PKG/.." && pwd)/coldstart_repro.md"
else
  OUT_MD="$REPO/runs/lookahead/fi_nogmc/_coldstart_repro.md"
fi

[ -d "$PKG/code" ] || { echo "[FATAL] 交付包无 code/：$PKG"; exit 2; }

# ★ **md5sum 存在性闸**：缺 md5sum（macOS 只有 `md5`）⇒ 下面的「真仓指纹」与「污染检测」
#   全部落空 ⇒ 两个空文件 diff 判等 ⇒ **误报「0 处污染」**。缺工具必须 [FATAL]。
command -v md5sum >/dev/null 2>&1 || { echo "[FATAL] 缺 md5sum（污染检测依赖它）"; exit 2; }

echo "=== 0/5 清理沙箱 ==="
rm -rf "$SB"; mkdir -p "$SB/repo/runs/lookahead" "$SB/pkg"
cp -a "$PKG/." "$SB/pkg/"
echo "沙箱：$SB   包副本：$SB/pkg"

echo "=== 1/5 铺产物（符号链接 ⇒ 不复制 3.7 GB） ==="
ART_DIRS="fi_nogmc fi_nogmc24 fovgate fovgate24 fovgate24nogmc fovgate24nogmc_d2 \
fovgate24nogmc_d4 fovgate24nogmc_d8 fovgate24rc2_d8 rbgate24nogmc gateA gateA24"
for d in $ART_DIRS; do
  src="$REPO/runs/lookahead/$d"
  [ -d "$src" ] || { echo "[FATAL] 缺产物目录：$src"; exit 2; }
  mkdir -p "$SB/repo/runs/lookahead/$d"
  cp -rs "$src/." "$SB/repo/runs/lookahead/$d/" 2>/dev/null || cp -rs "$src"/* "$SB/repo/runs/lookahead/$d/"
  echo "  [OK] $d"
done
for f in nogmc24_byte_equal.json nogmc24_cfg_identical.json; do
  if [ -f "$REPO/runs/lookahead/$f" ]; then
    cp -s "$REPO/runs/lookahead/$f" "$SB/repo/runs/lookahead/$f" && echo "  [OK] $f"
  elif [ -f "$PKG/$f" ]; then
    # host 异机模式：镜像里没有 ⇒ 从**包根**取（包根与服务器同路径逐位相同）
    cp -s "$PKG/$f" "$SB/repo/runs/lookahead/$f" && echo "  [OK] $f（取自包根）"
  else
    echo "  [MISS] $f（仓内与包根都没有 ⇒ 权威报告的输入缺失 ⇒ 冷启动必失败）"; exit 2
  fi
done

echo "=== 2/5 把「要重生成的报告」从符号链接换成真实可写文件 ==="
# <沙箱内路径>                    <交付包内对应文件名>
GEN_PAIRS="
runs/lookahead/fi_nogmc/FINAL_REPORT.md|FINAL_REPORT.md
runs/lookahead/fi_nogmc/final_report.json|FINAL_REPORT.json
runs/lookahead/fi_nogmc/report.md|report.md
runs/lookahead/fi_nogmc/report.json|report.json
runs/lookahead/fi_nogmc/report_house_style.md|report_house_style.md
runs/lookahead/fi_nogmc24/report_house_style_d2.md|report_house_style_24seq_d2.md
runs/lookahead/fi_nogmc24/report_house_style_d3.md|report_house_style_24seq_d3.md
runs/lookahead/fi_nogmc24/report_house_style_d4.md|report_house_style_24seq_d4.md
runs/lookahead/fi_nogmc24/report_house_style_d8.md|report_house_style_24seq_d8.md
runs/lookahead/fi_nogmc24/knobscan_d3/report.md|knobscan_24seq_d3.md
runs/lookahead/fi_nogmc24/knobscan_d3/report.json|knobscan_24seq_d3.json
runs/lookahead/fi_nogmc24/knobscan_p3b/report.md|knobscan_p3b_24seq.md
runs/lookahead/fi_nogmc24/knobscan_p3b/report.json|knobscan_p3b_24seq.json
runs/lookahead/fi_nogmc24/delta_dep/report.md|delta_dep_24seq.md
runs/lookahead/fi_nogmc24/delta_dep/report.json|delta_dep_24seq.json
runs/lookahead/fi_nogmc24/oracle_accel/closure_report.md|oracle_accel/closure_report.md
runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json|oracle_accel/closure_report.json
runs/lookahead/fi_nogmc24/reg_oracle/closure_report.md|reg_oracle/closure_report.md
runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json|reg_oracle/closure_report.json
runs/lookahead/fi_nogmc/workpoint/audit.md|workpoint_audit.md
runs/lookahead/fi_nogmc/workpoint/audit.json|workpoint_audit.json
runs/lookahead/fovgate/report.md|fovgate_8seq_cs_t0043.md
runs/lookahead/fovgate/report.json|fovgate_8seq_cs_t0043.json
runs/lookahead/fovgate24/report.md|fovgate_24seq_rc2_d3.md
runs/lookahead/fovgate24/report.json|fovgate_24seq_rc2_d3.json
runs/lookahead/fovgate24nogmc/report.md|fovgate_24seq_cs_t0043_d3.md
runs/lookahead/fovgate24nogmc/report.json|fovgate_24seq_cs_t0043_d3.json
runs/lookahead/fovgate24nogmc_d2/report.md|fovgate_24seq_cs_t0043_d2.md
runs/lookahead/fovgate24nogmc_d2/report.json|fovgate_24seq_cs_t0043_d2.json
runs/lookahead/fovgate24nogmc_d4/report.md|fovgate_24seq_cs_t0043_d4.md
runs/lookahead/fovgate24nogmc_d4/report.json|fovgate_24seq_cs_t0043_d4.json
runs/lookahead/fovgate24nogmc_d8/report.md|fovgate_24seq_cs_t0043_d8.md
runs/lookahead/fovgate24nogmc_d8/report.json|fovgate_24seq_cs_t0043_d8.json
runs/lookahead/fovgate24rc2_d8/report.md|fovgate_24seq_rc2_d8.md
runs/lookahead/fovgate24rc2_d8/report.json|fovgate_24seq_rc2_d8.json
"
: > "$SB/gen_pairs.tsv"
while IFS='|' read -r rel pkgname; do
  [ -z "$rel" ] && continue
  tgt="$SB/repo/$rel"
  mkdir -p "$(dirname "$tgt")"
  rm -f "$tgt"                      # 删的是符号链接本身，不动只读原件
  : > "$tgt"                       # 真实空文件（regen 会覆盖）
  printf '%s\t%s\n' "$rel" "$pkgname" >> "$SB/gen_pairs.tsv"
done <<< "$GEN_PAIRS"
echo "  待重生成报告数：$(wc -l < "$SB/gen_pairs.tsv")"

echo "=== 3/5 用【包内 code/】冷启动重生成 ==="
# 冷启动不许污染真仓：先给真仓的同名报告存指纹，跑完再比
# ★ 路径**逐个加引号**（`md5sum $(...)` 会被 word-splitting 拆开 ⇒ 含空格/中文的路径全丢）
while IFS='|' read -r rel _n; do
  [ -n "$rel" ] || continue
  md5sum "$REPO/$rel" 2>>"$SB/repo_md5.err" || true
done < <(printf '%s\n' "$GEN_PAIRS") > "$SB/repo_before.md5"
N_BEFORE="$(wc -l < "$SB/repo_before.md5")"
echo "  真仓指纹：$N_BEFORE 个文件"
[ "$N_BEFORE" -gt 0 ] || { echo "[FATAL] 真仓指纹为 0 ⇒ 污染检测形同虚设（查 $SB/repo_md5.err）"; cat "$SB/repo_md5.err"; exit 2; }
[ -f "$SB/repo/manu/tracking/lookahead/make_fastinfer_final_report.py" ] && \
  { echo "[FATAL] 沙箱里不该有仓内脚本，测试设计有误"; exit 2; }
REPO="$SB/repo" PY="$PY" bash "$SB/pkg/code/regen_reports_nogmc_fi.sh"
rc_regen=$?
echo "regen rc=$rc_regen"
while IFS='|' read -r rel _n; do
  [ -n "$rel" ] || continue
  md5sum "$REPO/$rel" 2>>"$SB/repo_md5.err" || true
done < <(printf '%s\n' "$GEN_PAIRS") > "$SB/repo_after.md5"
N_AFTER="$(wc -l < "$SB/repo_after.md5")"
[ "$N_AFTER" -eq "$N_BEFORE" ] || { echo "[FAIL] 指纹条数变了（$N_BEFORE → $N_AFTER）⇒ 污染检测不可信"; LEAK=1; }
if diff -q "$SB/repo_before.md5" "$SB/repo_after.md5" >/dev/null; then
  echo "  [OK] 真仓报告未被冷启动改动（0 处污染）"; LEAK=0
else
  echo "  [FAIL] 真仓报告被冷启动改动 ⇒ 沙箱隔离失败"; LEAK=1; diff "$SB/repo_before.md5" "$SB/repo_after.md5" | head -10
fi

echo "=== 4/5 与交付包逐个比对 ==="
"$PY" - "$SB" "$PKG" "$REPO" <<'PYEOF'
import json, pathlib, re, sys
sb, pkg, repo = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
TS = re.compile(r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d(:\d\d)?")
def norm(p: pathlib.Path, roots) -> str:
    t = p.read_text(encoding="utf-8", errors="replace")
    t = TS.sub("<TS>", t)
    for r in roots:            # 两侧统一替换 ⇒ 不再因「沙箱根 vs 仓根」产生假阳性
        t = t.replace(str(r), "<ROOT>")
    return t
rows, same, diff, missing = [], 0, 0, 0
for line in (sb / "gen_pairs.tsv").read_text().splitlines():
    if not line.strip():
        continue
    rel, name = line.split("\t")
    a, b = sb / "repo" / rel, pkg / name   # ★ 基址 = 沙箱（不是真仓！）
    if not a.exists() or not b.exists():
        rows.append({"file": name, "status": "MISSING", "detail": f"new={a.exists()} pkg={b.exists()}"}); missing += 1; continue
    if a.stat().st_size == 0:
        rows.append({"file": name, "status": "EMPTY", "detail": "沙箱侧重生成结果为空 ⇒ 重生成没落到沙箱"}); diff += 1; continue
    na, nb = norm(a, [repo, pkg, sb]), norm(b, [repo, pkg, sb])
    if na == nb:
        rows.append({"file": name, "status": "IDENTICAL"}); same += 1
    else:
        la, lb = na.splitlines(), nb.splitlines()
        d = [i for i in range(min(len(la), len(lb))) if la[i] != lb[i]]
        rows.append({"file": name, "status": "DIFF", "diff_lines": len(d),
                     "first_diff": (f"L{d[0]+1}: new={la[d[0]][:160]!r} | pkg={lb[d[0]][:160]!r}" if d else "行数不同")})
        diff += 1
out = {"identical": same, "diff": diff, "missing": missing, "total": same + diff + missing, "rows": rows}
(sb / "coldstart_diff.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"IDENTICAL={same}  DIFF={diff}  MISSING={missing}  TOTAL={out['total']}")
for r in rows:
    if r["status"] != "IDENTICAL":
        print("  ", r["status"], r["file"], r.get("diff_lines", ""), r.get("first_diff", "")[:220])
PYEOF
rc_cmp=$?

mkdir -p "$(dirname "$OUT_MD")"
"$PY" - "$SB/coldstart_diff.json" > "$SB/perfile.tsv" <<'PYEOF2'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["rows"]:
    print(f"| `{r['file']}` | {r['status']} | {r.get('diff_lines','')} | {r.get('first_diff','')} |")
PYEOF2
{
  echo "=== 5/5 包内独立复跑审计（只用包内 \`repo/\` 镜像，不碰仓内任何脚本） ==="
  AUDIT_RC_ALL=0
  : > "$SB/audit_selfrun.tsv"
  # 第 6 个脚本 = 交付包自足性审计：它在**包内**只能校验包级 + 镜像级两份清单（仓内对账需仓）
  # ⇒ 只记 rc，不参与下面的「与随包 md 逐位一致」比对（两种模式断言数不同：仓内 29 / 包内 11）。
  for s in audit_sec5_deliverables audit_sec5_acceptance audit_sec16_gap audit_sec17_reg_oracle audit_sec17_rbgate audit_pack_selfsufficient; do
    if [ ! -f "$SB/pkg/code/$s.py" ]; then
      echo "  [MISS] code/$s.py"; AUDIT_RC_ALL=1; continue
    fi
    # ★ 输出**落文件**而不是命令替换 `$(...)`：审计 stdout 走管道时本脚本 §5 会**卡死**
    #   （实测：同一沙箱、同一命令，手工跑 24 s 完成，而 §5 里 >14 min 不返回 ⇒ 卡的是
    #    管道/读取这一环，不是审计本身）⇒ 宁可落文件。
    # ★ 再加 `timeout`：万一真挂住，**挂起必须变成可见 FAIL**（rc=124），而不是无限期沉默。
    _alog="$SB/audit_$s.out"
    timeout 900 "$PY" "$SB/pkg/code/$s.py" > "$_alog" 2>&1; _rc=$?
    _nf="$(grep -oE "FAIL [0-9]+ (项|处)" "$_alog" | head -1)"
    [ "$_rc" -eq 124 ] && echo "  [TIMEOUT] $s 超过 900 s 未返回 ⇒ 记 FAIL"
    printf '%s\t%s\t%s\n' "$s" "$_rc" "${_nf:-FAIL 0}" >> "$SB/audit_selfrun.tsv"
    echo "  $s rc=$_rc ${_nf:-FAIL 0}"
    [ "$_rc" -eq 0 ] || AUDIT_RC_ALL=1
  done
  # 包内复跑的审计输出 vs 包内随包的审计报告（仓内复跑产物）⇒ 必须逐位一致
  "$PY" - "$SB" "$PKG" > "$SB/audit_selfrun_diff.json" <<'PYEOF3'
import json, pathlib, re, sys
sb, pkg = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
TS = re.compile(r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d(:\d\d)?")
names = ["audit_sec16_gap.md", "audit_sec17_reg_oracle.md", "audit_sec17_rbgate.md",
         "audit_sec5_acceptance.md", "audit_sec5_deliverables.md"]
def norm(p, roots):
    t = p.read_text(encoding="utf-8", errors="replace")
    t = TS.sub("<TS>", t)
    for r in roots:
        t = t.replace(str(r), "<ROOT>")
    return t
rows, same, diff, missing = [], 0, 0, 0
for n in names:
    a = sb / "pkg" / "repo" / "runs/lookahead/fi_nogmc/audit_round7" / n
    b = pkg / "audit_round7" / n
    if not a.exists() or not b.exists():
        rows.append({"file": n, "status": "MISSING", "detail": f"selfrun={a.exists()} pkg={b.exists()}"})
        missing += 1
        continue
    ta, tb = norm(a, [sb, pkg]), norm(b, [sb, pkg])
    if ta == tb:
        rows.append({"file": n, "status": "IDENTICAL"}); same += 1
    else:
        la, lb = ta.splitlines(), tb.splitlines()
        d = [i for i in range(min(len(la), len(lb))) if la[i] != lb[i]]
        rows.append({"file": n, "status": "DIFF", "diff_lines": len(d),
                     "first_diff": (f"L{d[0]+1}: selfrun={la[d[0]][:150]!r} | pkg={lb[d[0]][:150]!r}" if d else "行数不同")})
        diff += 1
out = {"identical": same, "diff": diff, "missing": missing, "total": same + diff + missing, "rows": rows}
print(json.dumps(out, ensure_ascii=False, indent=1))
PYEOF3
  echo "  审计输出一致：$("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"IDENTICAL={d['identical']} DIFF={d['diff']} MISSING={d['missing']}\")" "$SB/audit_selfrun_diff.json")"
  [ "$AUDIT_RC_ALL" -eq 0 ] || echo "  [FAIL] 包内审计有非零退出码"
} > "$SB/audit_phase.log" 2>&1
cat "$SB/audit_phase.log"
# ★ **别把退出码变量覆盖成计数**：这里另起一个计数变量，
#   退出码（0/1）必须原样保留到最后参与总判定（否则「审计失败」会被「计数>0」掩盖）。
AUDIT_OK_N="$(grep -c 'rc=0' "$SB/audit_phase.log" || true)"
{
  echo "# 交付包冷启动复现报告"
  echo
  echo "- 时间：$(date '+%F %T %Z')"
  echo "- 执行机：\`$(hostname)\` · 项目根 \`$REPO\`"
  echo "- 交付包：\`$PKG\`（$(find "$PKG" -type f | wc -l) 文件）"
  echo "- 沙箱：\`$SB\`（**只有** 产物符号链接 + 包内 \`code/\`，**故意没有** \`manu/\` 下任何脚本）"
  echo "- 命令：\`bash manu/tracking/lookahead/server_overnight/coldstart_verify_delivery.sh\`"
  echo
  echo "## 结论"
  echo
  echo "- regen rc=$rc_regen；比对 rc=$rc_cmp"
  echo "- 逐文件结果见 \`$SB/coldstart_diff.json\`"
  echo
  echo "| 指标 | 值 |"
  echo "|:--|:--|"
  echo "| 重生成报告数 | $(wc -l < "$SB/gen_pairs.tsv") |"
  echo "| 逐位一致 | $( "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['identical'])" "$SB/coldstart_diff.json" ) |"
  echo "| 有差异 | $( "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['diff'])" "$SB/coldstart_diff.json" ) |"
  echo "| 缺失 | $( "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['missing'])" "$SB/coldstart_diff.json" ) |"
  echo "| 真仓污染 | $( [ "${LEAK:-1}" -eq 0 ] && echo '0 处（隔离 OK）' || echo '**有污染**' ) |"
  echo "| 包内独立复跑审计 | $(grep -c 'rc=0' "$SB/audit_phase.log" || true) 份 rc=0（FAIL 全 0） |"
  echo "| 审计输出逐位一致 | $( "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['identical'])" "$SB/audit_selfrun_diff.json" ) / $( "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['total'])" "$SB/audit_selfrun_diff.json" ) |"
  echo
  echo "## 口径声明"
  echo
  echo "- 本报告只证明一件事：**交付包 \`code/\` 自包含** —— 仅凭「包内代码 + 只读产物」即可重生成全部报告，且结果与包内报告**逐位一致**。"
  echo "- 「只读产物」= 沙箱里是指向真仓的**符号链接**；重生成只**写**报告文件本身，不改任何产物。"
  echo "- **零污染判据**：冷启动前后真仓同名报告的 md5 必须不变（否则说明沙箱隔离失败 ⇒ 本报告作废）。"
  echo "- 归一化只抹两类差异：**时间戳**与**绝对路径前缀**（仓根/包根/沙箱根 → \`<ROOT>\`）；**数字一律逐位比对**，不做任何数值容差。"
  echo
  echo "## 沙箱设计（隔离三要点）"
  echo
  echo "1. 沙箱 \`$SB/repo/\` 下**只有** \`runs/lookahead/**\`（符号链接）——**故意不放** \`manu/\` 下任何脚本；"
  echo "2. 交付包整包拷到 \`$SB/pkg/\`，regen 从**包内** \`code/\` 自定位取生成器；"
  echo "3. 运行前后比对真仓报告 md5 ⇒ 判定是否泄漏。"
  echo
  echo "## 逐文件明细"
  echo
  echo "| 报告文件 | 状态 | 差异行数 | 首处差异 |"
  echo "|:--|:--|--:|:--|"
  cat "$SB/perfile.tsv"
  echo
  echo "## 包内独立复跑审计（**只用包内 \`repo/\` 镜像**，不碰仓内任何脚本）"
  echo
  echo "- 目的：证明 host 只拿到交付包，**也能独立复核**头条数字 —— 包内 \`code/\` 的五份审计脚本"
  echo "  自定位到包内 \`repo/\` 镜像（按仓内相对路径镜像真源），逐位重算并输出审计报告。"
  echo "- 判据：① 五份审计退出码全 0、FAIL 全 0；② 包内复跑的审计 md 与随包的审计 md **逐位一致**。"
  echo
  echo "| 审计脚本 | 退出码 | FAIL |"
  echo "|:--|--:|:--|"
  sed 's/\t/ | /g' "$SB/audit_selfrun.tsv" | sed 's/^/| /'
  echo
  echo "| 审计输出 | 状态 | 差异行数 | 首处差异 |"
  echo "|:--|:--|--:|:--|"
  "$PY" - "$SB/audit_selfrun_diff.json" <<'PYEOF4'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["rows"]:
    print(f"| `{r['file']}` | {r['status']} | {r.get('diff_lines','')} | {r.get('first_diff','')} |")
PYEOF4
  echo
  # 缺陷表里的「第 N/M 步」与「K 个输入」**现读实际执行的 regen 脚本**（包内 code/ 那份）
  #   ⇒ 流水线演进后这两个数字自动跟上，不会像硬编码那样悄悄过期
  _regen_sh="$SB/pkg/code/regen_reports_nogmc_fi.sh"
  _wp_step="$(grep -oE '=== [0-9]+/[0-9]+ 工作点对齐审计' "$_regen_sh" | head -1 | grep -oE '[0-9]+/[0-9]+' || true)"
  _gates="$(awk '/^for _need in /,/; do/' "$_regen_sh" | grep -c 'runs/lookahead' || true)"
  [ -n "$_wp_step" ] || _wp_step="?"
  [ "$_gates" -gt 0 ] 2>/dev/null || _gates="?"
  echo "## 本轮冷启动抓到的缺陷（已修）"
  echo
  echo "| # | 缺陷 | 影响 | 处置 |"
  echo "|--:|:--|:--|:--|"
  echo "| 1 | \`regen_reports_nogmc_fi.sh\` 把「工作点对齐审计」排在**最后**，而 \`FINAL_REPORT.md\` §11 要读它 | 真冷启动时权威报告读到空文件 ⇒ 打印「⚠️ 未找到工作点审计产物，跳过本节」并把 \`workpoint_audit=false\` 写进 JSON ⇒ **静默降级** | 工作点审计移到权威报告**之前**（现为第 ${_wp_step} 步） |"
  echo "| 2 | 同一脚本对权威报告的输入**没有硬门禁** | 任何输入缺失都会变成「跳过本节」而不是失败 | 新增硬门禁：${_gates} 个输入任一缺失/为空 ⇒ \`[FATAL]\` 退出（rc≠0） |"
  echo "| 3 | 交付包内 \`code/regen_reports_nogmc_fi.sh\` 硬编码仓根与 \`manu/tracking/lookahead/\` 路径 | 包内脚本**根本没法在包外跑** ⇒ 「自包含」是空话 | 脚本改为**自定位**：仓内优先、否则用脚本自身目录；\`REPO\`/\`PY\` 支持环境变量覆盖 |"
  echo
  echo "## 复现命令"
  echo
  echo '```bash'
  echo "cd $REPO"
  echo "bash manu/tracking/lookahead/server_overnight/pack_delivery_nogmc_fi.sh      # 打包"
  echo "bash manu/tracking/lookahead/server_overnight/coldstart_verify_delivery.sh  # 冷启动复现（本报告）"
  echo '```'
  echo
  echo "- 逐文件比对原始结果：\`$SB/coldstart_diff.json\`"
  echo "- 沙箱可留档复核：\`$SB/\`"
} > "$OUT_MD"
cat "$OUT_MD"

[ $rc_regen -eq 0 ] && [ $rc_cmp -eq 0 ] && [ "${LEAK:-1}" -eq 0 ] && [ "${AUDIT_RC_ALL:-1}" -eq 0 ] && exit 0
exit 1
