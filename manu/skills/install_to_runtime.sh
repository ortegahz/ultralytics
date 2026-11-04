#!/usr/bin/env bash
# 把 manu/skills/ 下的全部 Skill（当前 5 个）安装到「用户级 skill 目录」。
#
# 为什么是"复制"而不是"符号链接"（2026-10-22 实测）：
#   运行时用 `readdir(dir, {withFileTypes:true})` 枚举 skill 目录，
#   再用 `entry.isDirectory()` 过滤。**对符号链接，`isDirectory()` 返回 false**
#   （lstat 语义），所以 symlink 装的 skill 会被整批静默跳过 ⇒ 必须用真实目录。
#
# 运行时源码依据（mcode 0.6.12）：
#   root registry: { id:"user-global", kind:"global", rootPath: <dataDir>/skills }
#   ⇒ ~/.minimax/skills 就是官方用户级 skill 根，且**默认启用**（不需要开 external）。
#   对比：<workspace>/.minimax/skills 属于 external 源，默认关闭。
#
# 单一真值：manu/skills/<name>/SKILL.md 是仓库里的唯一真值；
#           本脚本只负责"复制 + md5 校验"，改完仓库请重跑本脚本。
#
# -- 安装目标（两个根，2026-11-03 实测后补齐）--------------------------------
#   1) minimax : Mavis/minimax 运行时的用户 skill 根  ~/.minimax/skills
#   2) codex   : Codex 本体的用户 skill 根          $CODEX_HOME/skills
#                （默认 ~/.codex/skills）
#   之前只装 minimax 一个根，而 codex 根本不读 ~/.minimax/skills，
#   所以装完 codex 侧照样看不到这些 skill。
#   2026-11-03 用 `codex debug prompt-input hi` 实测的 codex 发现规则：
#     [OK] $CODEX_HOME/skills/<name>/SKILL.md       (r0)
#     [OK] $CODEX_HOME/skills/.system/<name>/SKILL.md (r1)
#     [OK] <cwd>/.codex/skills/<name>/SKILL.md       (r0, 项目级)
#     [OK] <cwd>/.agents/skills/<name>/SKILL.md      (r2, 项目级)
#     [NG] <cwd>/skills/<name>/SKILL.md
#     [NG] 嵌套 git 仓库里的 <sub>/.codex/skills/
#   所以现在两个根都装。
#   注意：agent 沙箱里 ~/.codex 可能是只读 (EROFS)，那种目标会装不上，
#   脚本会跳过并报错，请在普通终端重跑（或加 --minimax-only）。
#
# 用法：
#   bash manu/skills/install_to_runtime.sh                # 两个根都装
#   bash manu/skills/install_to_runtime.sh --minimax-only # 只装 minimax
#   bash manu/skills/install_to_runtime.sh --codex-only   # 只装 codex
#
# 执行机：本地（写 ~/.minimax/skills/ 与 ${CODEX_HOME:-~/.codex}/skills/）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_ROOT="$REPO/manu/skills"
DST_ROOTS=(
  "${MINIMAX_DATA_DIR:-$HOME/.minimax}/skills"
  "${CODEX_HOME:-$HOME/.codex}/skills"
)

case "${1:-}" in
  --minimax-only) DST_ROOTS=("${DST_ROOTS[0]}") ;;
  --codex-only)   DST_ROOTS=("${DST_ROOTS[1]}") ;;
  "")             ;;
  *) echo "用法：bash $0 [--minimax-only|--codex-only]" >&2; exit 2 ;;
esac

echo "源码目录：$SRC_ROOT"
echo "安装目录：${DST_ROOTS[*]}"
echo

fail=0
for DST_ROOT in "${DST_ROOTS[@]}"; do
  mkdir -p "$DST_ROOT" 2>/dev/null || true
  probe="$DST_ROOT/.install_probe.$$"
  if [ ! -d "$DST_ROOT" ] || ! { echo ok > "$probe"; } 2>/dev/null; then
    echo "❌ 目标根不可写（跳过）：$DST_ROOT"
    echo "   （agent 沙箱里 ~/.codex 常被封成只读 EROFS，请在普通终端重跑）"
    fail=1
    continue
  fi
  rm -f "$probe"
  echo "-- $DST_ROOT"
  for d in "$SRC_ROOT"/*/; do
      name="$(basename "$d")"
      [ -f "$d/SKILL.md" ] || continue

      target="$DST_ROOT/$name"
      if [ -L "$target" ]; then
          echo "⚠️  $target 是符号链接（运行时枚举会跳过）=> 改为真实目录"
          rm "$target"
      fi
      if ! mkdir -p "$target" 2>/dev/null; then
          echo "  FAIL $name  无法创建目录 $target"
          fail=1
          continue
      fi
      # 必须复制**整个 skill 目录**（SKILL.md + references/），
      # 只拷 SKILL.md 会把 references/*.md 链接全部拷断（lint 规则 7 会失败）。
      if ! cp -r "$d/." "$target/"; then
          echo "  FAIL $name  复制失败到 $target/"
          fail=1
          continue
      fi

      src_md5="$(md5sum "$d/SKILL.md" | cut -d' ' -f1)"
      dst_md5="$(md5sum "$target/SKILL.md" | cut -d' ' -f1)"
      refs_ok=1
      for r in "$d"/references/*.md; do
          [ -f "$r" ] || continue
          rb="$(basename "$r")"
          if ! cmp -s "$r" "$target/references/$rb"; then
              echo "  FAIL $name  references/$rb 内容不一致"
              refs_ok=0
          fi
      done
      if [ "$src_md5" = "$dst_md5" ] && [ "$refs_ok" = "1" ]; then
          echo "  OK   $name  md5=$src_md5"
      else
          [ "$src_md5" = "$dst_md5" ] || echo "  FAIL $name  md5 不一致：src=$src_md5 dst=$dst_md5"
          fail=1
      fi
  done
  echo
done

echo
if [ "$fail" -ne 0 ]; then
    echo "❌ 存在 md5 不一致的 Skill"
    exit 1
fi

echo "自检（运行时可见的 skill；注意：**需新开 Session 才会出现在 available_skills**）"
for DST_ROOT in "${DST_ROOTS[@]}"; do
  [ -d "$DST_ROOT" ] || continue
  echo "-- $DST_ROOT"
  for f in "$DST_ROOT"/*/SKILL.md; do
      [ -f "$f" ] || continue
      printf '  %-26s %s\n' "$(basename "$(dirname "$f")")" "$(sed -n '2p' "$f")"
  done
done

echo
echo "另：装完可用下面的 lint 自检（每个 Skill 都要 PASS）"
echo "  for d in manu/skills/*/; do node /home/manu/.minimax/.builtin-skills/skill-creator/scripts/lint-skill.js \"\$d\" | tail -1; done"
