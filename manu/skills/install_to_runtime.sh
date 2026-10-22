#!/usr/bin/env bash
# 把 manu/skills/ 下的 4 个 Skill 安装到 Mavis 运行时��用户 skill 目录。
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
# 执行机：本地（写 ~/.minimax/skills/）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_ROOT="$REPO/manu/skills"
DST_ROOT="${MINIMAX_DATA_DIR:-$HOME/.minimax}/skills"

mkdir -p "$DST_ROOT"
echo "源码目录：$SRC_ROOT"
echo "安装目录：$DST_ROOT"
echo

fail=0
for d in "$SRC_ROOT"/*/; do
    name="$(basename "$d")"
    [ -f "$d/SKILL.md" ] || continue

    target="$DST_ROOT/$name"
    if [ -L "$target" ]; then
        echo "⚠️  $target 是符号链接（运行时枚举会跳过）⇒ 改为真实目录"
        rm "$target"
    fi
    mkdir -p "$target"
    cp "$d/SKILL.md" "$target/SKILL.md"

    src_md5="$(md5sum "$d/SKILL.md" | cut -d' ' -f1)"
    dst_md5="$(md5sum "$target/SKILL.md" | cut -d' ' -f1)"
    if [ "$src_md5" = "$dst_md5" ]; then
        echo "  ✅ $name  md5=$src_md5"
    else
        echo "  ❌ $name  md5 不一致：src=$src_md5 dst=$dst_md5"
        fail=1
    fi
done

echo
if [ "$fail" -ne 0 ]; then
    echo "❌ 存在 md5 不一致的 Skill"
    exit 1
fi

echo "自检（运行时可见的 skill；注意：**需新开 Session 才会出现在 available_skills**）"
for f in "$DST_ROOT"/*/SKILL.md; do
    [ -f "$f" ] || continue
    printf '  %-26s %s\n' "$(basename "$(dirname "$f")")" "$(sed -n '2p' "$f")"
done

echo
echo "另：装完可用下面的 lint 自检（每个 Skill 都要 PASS）"
echo "  for d in manu/skills/*/; do node /home/manu/.minimax/.builtin-skills/skill-creator/scripts/lint-skill.js \"\$d\" | tail -1; done"
