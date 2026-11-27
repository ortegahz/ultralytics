#!/bin/bash
# ============================================================================
# 把服务器侧产出的 MEMORY_SECTION.md 并入 host 权威记忆库 manu/memory/memory.md
#
#   用法: bash iqa_v2/merge_memory.sh <MEMORY_SECTION.md>
#
# 步骤：
#   1) 备份 memory.md / _roll_manifest.json，并**快照 archive/ 卷号清单**（回滚要用）
#   2) 幂等检查：这一章的标题指纹若已在 memory.md 里 ⇒ 判为「已合并」直接退出 0
#   3) 把章节的 `## §NNN` 重编号为**下一个空闲编号**（活区 + archive 一起扫，防撞号）
#      —— 命中 `## §NNN` 就**就地**改号；有 `##` 标题但没编号就**就地**换成带号标题；
#         连 `##` 标题都没有才**前置**一行标题 —— **任何分支都不得整体覆盖正文**
#   4) 插入到 BODY 区末尾（`<!-- ===== MEMORY BODY END ===== -->` **之前**）
#      —— 章节正文**一个字节都不能少**：落盘前**逐行比对**（正文与输入**逐行全等**，
#         不是只比行数）、§ 标题**恰好 1 个**；任一断言不过 ⇒ **硬失败**（exit 1）
#   5) 跑 roll_memory.py（活区超窗就切新卷）→ check_memory_roll.py（零丢失）
#      → validate_nav_table.py（导航逐条解析）—— **两道校验都必须绿**
#   6) 任一失败 ⇒ 打印**可直接照抄的回滚命令**（含新增分卷的删除），不自动回滚，留现场
#
# ⚠️ 退出码：0 = 已合并（或本来就已合并）／1 = 失败（已打印回滚方法）
# ============================================================================
set -uo pipefail

SEC="${1:-}"
if [ -z "$SEC" ] || [ ! -f "$SEC" ]; then
  echo "用法: bash $0 <MEMORY_SECTION.md>"; exit 2
fi
# 🔴 必须在 cd 之前把相对路径解析成绝对路径（cd 之后相对路径会失效 → FileNotFoundError）
SEC="$(cd "$(dirname "$SEC")" && pwd)/$(basename "$SEC")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

M=manu/memory/memory.md
MAN=manu/memory/archive/_roll_manifest.json
ARCH=manu/memory/archive
STAMP=$(date +%Y%m%d_%H%M%S)
BAK_M="$M.bak_$STAMP"
BAK_MAN="$MAN.bak_$STAMP"

# 🔴 回滚命令（可直接照抄）：三处失败路径共用一份，避免抄错
print_rollback() {
  echo "回滚: cp -p '$BAK_M' '$M' && cp -p '$BAK_MAN' '$MAN'"
  echo "      并删除本轮新增分卷: diff <(ls -1 $ARCH) '$ARCH/.volumes_before_$STAMP' | grep '^>' | sed -n 's/^> //p' | xargs -r -I{} rm -f '$ARCH/{}'"
}

cp -p "$M" "$BAK_M" || { echo "[FAIL] 备份 memory.md 失败，终止"; exit 1; }
cp -p "$MAN" "$BAK_MAN" || { echo "[FAIL] 备份 _roll_manifest.json 失败，终止"; exit 1; }
ls -1 "$ARCH" > "$ARCH/.volumes_before_$STAMP" 2>/dev/null || true
echo "已备份: $BAK_M / $BAK_MAN  （archive 卷号快照: $ARCH/.volumes_before_$STAMP）"

python3 - "$SEC" "$BAK_M" <<'PY'
import os, re, shutil, sys

sec_path, bak_path = sys.argv[1], sys.argv[2]
p = 'manu/memory/memory.md'
s = open(p, encoding='utf-8').read()
raw = open(sec_path, encoding='utf-8').read()
sec = raw.strip('\n')
if not sec.strip():
    raise SystemExit('[FAIL] 服务器章节为空（输入文件无内容）')
in_lines = sec.splitlines()
# 🔴 畸形输入闸门：非空行不足 2 行 ⇒ 服务器侧成稿可能被截断 ⇒ 拒绝合并（不报 OK）
nonblank = [ln for ln in in_lines if ln.strip()]
if len(nonblank) < 2:
    raise SystemExit('[FAIL] 服务器章节内容不足（仅 %d 行非空）⇒ 疑似成稿被截断，拒绝合并'
                     % len(nonblank))

# ---- 章节标题指纹（幂等判据：重编号后标题文字不变）----
heading_line = next((ln for ln in in_lines if ln.lstrip().startswith('#')), '')
head = heading_line or next((ln for ln in in_lines if ln.strip()), '')
fingerprint = re.sub(r'^#+\s*§?\s*\d*\s*·?\s*', '', head).strip()

# ---- 活区 + archive 一次性读入（编号扫描 + 幂等指纹都用它）----
texts = [s]
for name in sorted(os.listdir('manu/memory/archive')):
    if not name.endswith('.md'):
        continue
    try:
        texts.append(open(os.path.join('manu/memory/archive', name), encoding='utf-8').read())
    except OSError:
        continue

if fingerprint:
    # 🔴 只与**已有章节标题**比对（不与正文全文比对）——
    #    否则正文里恰好出现同一句话就会被误判成「已合并」⇒ 静默不插入。
    existing = set()
    for t in texts:
        for ln in t.splitlines():
            if re.match(r'^##(?!#)', ln):
                existing.add(re.sub(r'^#+\s*§?\s*\d*\s*·?\s*', '', ln).strip())
    if fingerprint in existing:
        print('[SKIP] 章节标题指纹已在 memory.md 中 ⇒ 判定为「已合并」，不重复插入')
        print('       指纹: %s' % fingerprint[:60])
        raise SystemExit(3)

# ---- 下一个空闲编号：活区 + archive 一起扫（防历史乱序补记撞号）----
used = set()
for t in texts:
    used |= {int(m) for m in re.findall(r'^##\s*§(\d+)', t, re.M)}
nxt = (max(used) + 1) if used else 1

# ---- 结构分析：标题行 + 其余顶层 H2 ----
#   标题行 = 第一个 H2（`## …`）；没有 H2 时退到第一个 H1（`# …`）；再没有就合成标题。
#   🔴 顶层 H2 多于 1 个 ⇒ 一节会被 roll 拆成两节 ⇒ 导航表整组判 FAIL
#      （实测：§319 里塞第二个 H2 ⇒ validate_nav_table 340/355）⇒ 直接拒绝合并。
h2_idxs = [k for k, ln in enumerate(in_lines) if re.match(r'^##(?!#)', ln)]
if len(h2_idxs) > 1:
    raise SystemExit('[FAIL] 输入章节含 %d 个顶层 H2 标题（第 %s 行）⇒ 合并后会被 roll 拆成多节'
                     '（导航表整组判 FAIL），拒绝合并；请把多余的 ## 降级为 ### 后重试'
                     % (len(h2_idxs), ', '.join(str(k + 1) for k in h2_idxs)))
h1_idx = next((k for k, ln in enumerate(in_lines) if re.match(r'^#(?!#)', ln)), None)
title_idx = h2_idxs[0] if h2_idxs else h1_idx

# ---- 重编号 + 标题归位 ----
#   ① 标题行命中 `## §NNN` ⇒ 就地改号（标题文字原样保留）
#   ② 标题行是别的 `##`/`#` ⇒ 就地换成带号标题
#   ③ 没有任何标题 ⇒ 合成标题并**前置**
#   ⚠️ 三条分支都只动**标题那一行**；正文一行不改、不删、不覆盖
#      （历史 bug：new_sec = '## §%d · %s' % (...) 整体覆盖 ⇒ 正文全丢而三道校验全绿）
#   ⚠️ 标题一律**提到本节最前** ⇒ 标题前的引言不会被甩给上一节
#      （若保留内层 `##` 不吸收，一节会有两个 H2 ⇒ 导航表整组判 FAIL）
new_sec, nsub = re.subn(r'^##\s*§\d+', '## §%d' % nxt, sec, count=1, flags=re.M)
nl = new_sec.splitlines()
if title_idx is not None:
    if nsub == 1:
        new_title = nl[title_idx]          # 已就地改号
    else:
        # 标题文字里若还带着旧编号（如 `# §318 · 标题`）⇒ 去掉，免得导航表里出现过期编号
        t = re.sub(r'^§\d+\s*·?\s*', '', re.sub(r'^#+\s*', '', nl[title_idx]).strip()).strip() \
            or '成稿章节（服务器侧产出）'
        new_title = '## §%d · %s' % (nxt, t)
    body_lines = in_lines[:title_idx] + in_lines[title_idx + 1:]
    sep = '\n' if title_idx == 0 else '\n\n'
else:
    new_title = '## §%d · %s' % (nxt, '成稿章节（服务器侧产出）')
    body_lines = list(in_lines)
    sep = '\n\n'
new_sec = new_title + sep + '\n'.join(body_lines)
new_lines = new_sec.splitlines()

# ---- 断言 A：§ 标题**恰好 1 个**（硬失败）----
n_head = len(re.findall(r'^##\s*§\d+', new_sec, re.M))
if n_head != 1:
    raise SystemExit('[FAIL] 章节标题数异常（%d，应为 1）⇒ 疑似同节多标题，终止' % n_head)

# ---- 断言 B：顶层 H2 **恰好 1 个**（输入的多余 H2 已在上游拦掉）（硬失败）----
h2_after = len(re.findall(r'^##(?!#)', new_sec, re.M))
if h2_after != 1:
    raise SystemExit('[FAIL] 顶层标题数异常（%d，应为 1）⇒ 可能产生重复章节，终止' % h2_after)

# ---- 断言 C：正文**逐行全等**（不只比行数）—— 除标题行外一字不差（硬失败）----
#   期望正文 = 输入行去掉标题行；标题提到最前时额外带 1 行分隔空行
expected_body = body_lines if title_idx == 0 else [''] + body_lines
actual_body = new_lines[1:]
if actual_body != expected_body:
    raise SystemExit('[FAIL] 正文逐行比对不一致（期望正文 %d 行 / 落盘正文 %d 行）⇒ 疑似正文丢失，终止'
                     % (len(expected_body), len(actual_body)))

marker = '<!-- ===== MEMORY BODY END ===== -->'
if s.count(marker) != 1:
    raise SystemExit('[FAIL] BODY END 标记不唯一，终止')
i = s.index(marker)
prefix = s[:i].rstrip('\n')
block = '\n\n' + new_sec + '\n\n'
s2 = prefix + block + s[i:]

# ---- 断言 D：既有内容一字不动 + 全库只多出一个 § 标题（硬失败）----
if not s2.startswith(prefix) or not s2.endswith(s[i:]):
    raise SystemExit('[FAIL] 插入后既有内容被破坏（前段/后段比对不一致），终止')
if s2.count(block) != 1:
    raise SystemExit('[FAIL] 插入块定位失败（在成品中出现 %d 次），终止' % s2.count(block))
h_before = len(re.findall(r'^##\s*§\d+', s, re.M))
h_after = len(re.findall(r'^##\s*§\d+', s2, re.M))
if h_after != h_before + 1:
    raise SystemExit('[FAIL] § 标题总数异常（%d → %d，应为 +1），终止' % (h_before, h_after))

# ---- 落盘：先写临时文件再原子替换（关机/断电不会留下半截 memory.md）----
tmp = p + '.tmp_merge'
try:
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(s2)
        f.flush()
        os.fsync(f.fileno())
    shutil.copymode(p, tmp)
    os.replace(tmp, p)
except OSError as e:
    if os.path.exists(tmp):
        os.remove(tmp)
    raise SystemExit('[FAIL] 写盘失败：%s（memory.md 未改动）' % e)

# ---- 断言 E：回读落盘内容，与预期**逐字节**一致（硬失败；不过就自动回滚）----
disk = open(p, encoding='utf-8').read()
if disk != s2:
    shutil.copyfile(bak_path, p)
    shutil.copystat(bak_path, p)
    raise SystemExit('[FAIL] 落盘后回读与预期不一致 ⇒ 已自动从备份回滚 memory.md')

print('[OK] 已插入 §%d（原文 %d 行 → 写入 %d 行；正文 %d 行逐行全等、§ 标题恰好 1 个）'
      % (nxt, len(in_lines), len(new_lines), len(actual_body)))
PY
rc=$?
if [ "$rc" -eq 3 ]; then
  echo "[OK] 记忆库未改动（此前已合并过该章节）"
  exit 0
fi
if [ "$rc" -ne 0 ]; then
  # 🔴 断言不过 ⇒ **硬失败**（exit 1），并打印可直接照抄的回滚命令
  echo "[FAIL] 插入失败（断言未通过 / 写入异常），memory.md 已被脚本自动还原或保持原样"
  print_rollback
  exit 1
fi

echo
echo "=== roll（超窗则切新卷）==="
python3 manu/memory/roll_memory.py > /tmp/_roll_out.txt 2>&1
rc_roll=$?
tail -6 /tmp/_roll_out.txt
if [ "$rc_roll" -ne 0 ]; then
  echo "[FAIL] roll 失败（rc=$rc_roll），原文在: $BAK_M"
  tail -20 /tmp/_roll_out.txt
  exit 1
fi

echo
echo "=== check（零丢失校验）==="
python3 manu/memory/check_memory_roll.py > /tmp/_check_out.txt 2>&1
rc_check=$?
tail -30 /tmp/_check_out.txt
if [ "$rc_check" -ne 0 ]; then
  echo "[FAIL] check_memory_roll 未通过（rc=$rc_check）"
  print_rollback
  exit 1
fi

echo
echo "=== validate_nav_table（导航逐条解析）==="
python3 manu/memory/validate_nav_table.py > /tmp/_nav_out.txt 2>&1
rc_nav=$?
tail -15 /tmp/_nav_out.txt
if [ "$rc_nav" -ne 0 ]; then
  echo "[FAIL] validate_nav_table 未通过（rc=$rc_nav）"
  print_rollback
  exit 1
fi

echo
echo "[OK] 记忆库已合并并通过两道校验（备份保留在 *.bak_$STAMP）"
