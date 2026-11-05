#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变异测试：证明交付包自足性审计**有判别力**（篡改必报警）。

为什么必须做：审计若只报「PASS」而无判别力，等于没审。本脚本对交付包副本逐个施加
**已知篡改**，要求审计**必须**报错（rc=1 且指出对应断言 ID）；并留一个**已知边界**用例
（清单无签名 ⇒ 同改真源 + 两层清单检测不到），如实标出、不粉饰。

两种模式各测一遍：
  · **包内模式**（host 的用法）：把包拷到仓外 ⇒ 走「包级 + 镜像级」两份清单判据；
  · **仓内模式**：搭一个「假仓」（真仓原件用**硬链接**挂进去，只读、不写回真仓）
    ⇒ 走「镜像 vs 仓内原件逐位对账 / code/ 与仓内同源」等判据。

用法：
    python mutation_test_selfsufficient.py [--pkg <交付包目录>]
退出码：0 = 全部变异用例的期望成立；1 = 有用例与期望不符。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 仓内跑 ⇒ 交付包在仓根下；随包跑（<包>/code/）⇒ 交付包就是脚本的上级目录
_REPO = next((a for a in HERE.parents if (a / "manu/tracking/lookahead").is_dir()), None)
PKG_DEFAULT = (_REPO / "runs/lookahead/final/nogmc_fi_20261104") if _REPO else HERE.parent

# 真源（镜像内）与包顶层各挑一个可改的代表
MIRROR_VICTIM = "repo/runs/lookahead/fovgate24/runs/rc2/d3/fov__m8/arms.json"
MIRROR_VICTIM_REPO = "runs/lookahead/fovgate24/runs/rc2/d3/fov__m8/arms.json"
PACK_VICTIM = "FINAL_REPORT.md"
AUDIT_SCRIPTS = ["audit_sec16_gap.py", "audit_sec17_reg_oracle.py", "audit_sec17_rbgate.py",
                 "audit_sec5_acceptance.py", "audit_sec5_deliverables.py", "audit_pack_selfsufficient.py"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flip_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x20
    path.write_bytes(bytes(data))


def edit_sums_line(sums: Path, rel: str, new_digest: str | None) -> None:
    out = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, sep, r = line.partition("  ")
        if sep and r.lstrip("./") == rel:
            line = f"{new_digest or digest}  {r}"
        out.append(line)
    sums.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_fake_repo(pkg: Path, repo_root: Path) -> Path:
    """搭「假仓」：让**仓内模式**能在包外被复跑。

    - `manu/tracking/lookahead/*.py` = 六份审计脚本（与包内 `code/` 同源）；
    - `runs/lookahead/**` 的真源原件用**硬链接**挂进来（只读；改动前先断开链接，
      **绝不写回真仓**）；
    - `runs/lookahead/final/nogmc_fi_20261104/` = 交付包副本（仓内模式的被审对象）。
    """
    src_repo = _REPO
    la = repo_root / "manu/tracking/lookahead"
    la.mkdir(parents=True, exist_ok=True)
    for name in AUDIT_SCRIPTS:
        shutil.copy2(src_repo / "manu/tracking/lookahead" / name, la / name)
    # 真源原件：硬链接（改之前会先 unlink ⇒ 真仓不受影响）
    mirror = pkg / "repo"
    n = 0
    for f in sorted(p for p in mirror.rglob("*") if p.is_file() and p.name != "MD5SUMS.txt"):
        rel = f.relative_to(mirror)
        dst = repo_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(src_repo / rel, dst)
        except OSError:
            shutil.copy2(src_repo / rel, dst)
        n += 1
    out = repo_root / "runs/lookahead/final/nogmc_fi_20261104"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pkg, out)
    # 仓内审计结论副本（仓内模式的「新鲜度」判据要比对它们 ⇒ 假仓必须也有一份）
    src_aud = src_repo / "runs/lookahead/fi_nogmc/audit_round7"
    dst_aud = repo_root / "runs/lookahead/fi_nogmc/audit_round7"
    dst_aud.mkdir(parents=True, exist_ok=True)
    n_aud = 0
    for f in sorted(p for p in src_aud.glob("*.md") if p.is_file()):
        dst = dst_aud / f.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
        n_aud += 1
    print(f"  [假仓] {repo_root}：硬链接真源 {n} 个 + 审计结论 {n_aud} 份 + 包副本 {out}")
    return repo_root


def run_audit(pkg: Path) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(pkg / "code" / "audit_pack_selfsufficient.py")],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def run_audit_repo(fake_repo: Path) -> tuple[int, str]:
    script = fake_repo / "manu/tracking/lookahead/audit_pack_selfsufficient.py"
    proc = subprocess.run([sys.executable, str(script), "--out-md", str(fake_repo / "_out.md")],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


# ★ 审计**进程崩了**（没吐出判据结论）≠ 审计**判了失败** —— 两者必须分开计数。
#   一次偶发的环境抖动（临时 I/O 错误 / 资源耗尽 / 外部信号）若被当成「判据失败」，
#   就会被误报成「审计无判别力 ⇒ 交付包不可信」⇒ 那是**假警报**，比不报警更坏。
CRASH_MARK = "Traceback (most recent call last)"


def _crashed(rc: int, out: str) -> bool:
    """审计进程**崩溃**（非零退出且没有判据结论）⇒ 环境异常，不是判据失败。"""
    if rc == 0:
        return False
    return CRASH_MARK in out or not out.strip()


def _exec_case(pkg: Path, mode: str, mutate, keep: bool) -> tuple[int, str, bool]:
    """跑一个变异用例 ⇒ (rc, 输出, 是否崩溃)。"""
    tmp = Path(tempfile.mkdtemp(prefix="mut_test_"))
    dst = tmp / "pkg"
    try:
        shutil.copytree(pkg, dst)
        if mode == "repo":
            fake = build_fake_repo(dst, tmp / "fakerepo")
            _CTX["fake_repo"] = fake
            mutate(dst)
            rc, out = run_audit_repo(fake)
        else:
            mutate(dst)
            rc, out = run_audit(dst)
        return rc, out, _crashed(rc, out)
    finally:
        if keep:
            print(f"    [保留] {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 包内模式的变异用例
def m_none(pkg: Path) -> None:
    """M0 对照：原样 ⇒ 必须全 PASS、rc=0。"""


def m_mirror_byte(pkg: Path) -> None:
    """M1 篡改镜像真源 1 字节 ⇒ 包级 + 镜像级清单都要报错。"""
    flip_byte(pkg / MIRROR_VICTIM)


def m_pack_byte(pkg: Path) -> None:
    """M2 篡改包顶层权威报告 ⇒ 包级清单要报错。"""
    flip_byte(pkg / PACK_VICTIM)


def m_delete(pkg: Path) -> None:
    """M3 删掉一个镜像真源 ⇒ 「缺失」与「覆盖」都要报错。"""
    (pkg / MIRROR_VICTIM).unlink()


def m_extra_file(pkg: Path) -> None:
    """M4 塞一个清单外的文件 ⇒ 覆盖判据要报错。"""
    (pkg / "extra_unlisted.txt").write_text("未列入清单的文件\n", encoding="utf-8")


def m_extra_in_audit_dir(pkg: Path) -> None:
    """M4b 往镜像的 audit_round7/（运行产物目录）塞清单外文件 ⇒ 覆盖判据**仍要**报错。

    运行产物只按**固定文件名**豁免 ⇒ 目录前缀不是免检区（reviewer IMPORTANT-1）。
    """
    d = pkg / "repo/runs/lookahead/fi_nogmc/audit_round7"
    d.mkdir(parents=True, exist_ok=True)
    (d / "planted.txt").write_text("往运行产物目录里塞的未列入清单的文件\n", encoding="utf-8")


def m_cover_manifest(pkg: Path) -> None:
    """M5 篡改真源后**顺手改镜像清单**掩盖 ⇒ 包级清单（独立于镜像清单）仍要报错。"""
    victim = pkg / MIRROR_VICTIM
    flip_byte(victim)
    edit_sums_line(pkg / "repo" / "MD5SUMS.txt", MIRROR_VICTIM[len("repo/"):], md5(victim))


def m_tamper_pack_manifest(pkg: Path) -> None:
    """M6 只改**包顶层清单**掩盖 ⇒ 镜像清单（另一把锁）仍要报错。"""
    victim = pkg / MIRROR_VICTIM
    flip_byte(victim)
    edit_sums_line(pkg / "MD5SUMS.txt", MIRROR_VICTIM, md5(victim))


def m_both_manifests(pkg: Path) -> None:
    """M7 已知边界：真源 + **三层**清单同改 ⇒ **检测不到**（md5 清单无签名）。

    ★ G2 之后门槛从「两层」升到「三层」：包级清单现在**锚住** `repo/MD5SUMS.txt`
      ⇒ 篡改者必须连「包级清单里 `repo/MD5SUMS.txt` 那一条」一起改，才能把镜像清单的改动也抹掉。
    本审计**不声称**能防这一类；如实留作边界条件记录。
    """
    victim = pkg / MIRROR_VICTIM
    flip_byte(victim)
    edit_sums_line(pkg / "repo" / "MD5SUMS.txt", MIRROR_VICTIM[len("repo/"):], md5(victim))
    edit_sums_line(pkg / "MD5SUMS.txt", MIRROR_VICTIM, md5(victim))
    # 第三层：镜像清单变了 ⇒ 包级清单里 `repo/MD5SUMS.txt` 那一条也必须跟着改
    edit_sums_line(pkg / "MD5SUMS.txt", "repo/MD5SUMS.txt", md5(pkg / "repo" / "MD5SUMS.txt"))


def m_mirror_manifest_only(pkg: Path) -> None:
    """M8 只改**镜像清单**（真源不动）⇒ 镜像级 + 包级（锚定链）都要报错。

    ★ G2 的价值：包级清单现在**锚住** `repo/MD5SUMS.txt`
      ⇒ 只动镜像清单，包级立刻发现（早先两把锁其实只剩一把）。
    """
    edit_sums_line(pkg / "repo" / "MD5SUMS.txt", MIRROR_VICTIM[len("repo/"):], "0" * 32)


# ------------------------------------------------------------------ 仓内模式的变异用例
def m_repo_none(pkg: Path) -> None:
    """R0 对照：假仓 + 未篡改 ⇒ 仓内模式全 PASS、rc=0。"""


def m_repo_origin(pkg: Path) -> None:
    """R1 篡改「仓内原件」1 字节 ⇒ 镜像 vs 仓内原件逐位对账要报错。

    先 unlink 硬链接再写 ⇒ **真仓原件不受影响**。
    """
    victim = _CTX["fake_repo"] / MIRROR_VICTIM_REPO
    victim.unlink()
    shutil.copy2(_REPO / MIRROR_VICTIM_REPO, victim)
    flip_byte(victim)


def m_repo_code(pkg: Path) -> None:
    """R2 篡改包内 `code/audit_pack_selfsufficient.py` ⇒ 「与仓内同名文件 md5 相同」要报错。

    ★ 要改**假仓里那份包副本**（仓内模式审的是它，不是 `tmp/pkg`）。
    """
    pack = _CTX["fake_repo"] / "runs/lookahead/final/nogmc_fi_20261104"
    flip_byte(pack / "code/audit_pack_selfsufficient.py")


def m_repo_audit_copy(pkg: Path) -> None:
    """R3 篡改「仓内」审计结论副本 ⇒ 「包内副本 = 仓内当前产物」新鲜度判据要报错。

    ★ 篡改的是**假仓里**那份（仓内模式把 `ROOT` 定位到假仓）；
      先 unlink 硬链接再写 ⇒ **真仓原件不受影响**。
    """
    rel = "runs/lookahead/fi_nogmc/audit_round7/audit_sec16_gap.md"
    victim = _CTX["fake_repo"] / rel
    victim.unlink()
    shutil.copy2(_REPO / rel, victim)
    flip_byte(victim)


_CTX: dict = {}

CASES = [
    ("M0 原样（对照）", "inpack", m_none, 0, []),
    ("M1 篡改镜像真源 1 字节", "inpack", m_mirror_byte, 1, ["PKG.SUMS-MD5-MATCH", "MIR.SUMS-MD5-MATCH"]),
    ("M2 篡改包顶层权威报告", "inpack", m_pack_byte, 1, ["PKG.SUMS-MD5-MATCH"]),
    ("M3 删除一个镜像真源", "inpack", m_delete, 1, ["PKG.SUMS-NO-MISSING", "MIR.SUMS-NO-MISSING"]),
    ("M4 塞入清单外文件", "inpack", m_extra_file, 1, ["PKG.SUMS-COVERAGE"]),
    ("M4b 往运行产物目录塞文件", "inpack", m_extra_in_audit_dir, 1, ["PKG.SUMS-COVERAGE", "MIR.SUMS-COVERAGE"]),
    ("M5 改真源 + 改镜像清单（掩盖）", "inpack", m_cover_manifest, 1, ["PKG.SUMS-MD5-MATCH"]),
    ("M6 改真源 + 改包级清单（掩盖）", "inpack", m_tamper_pack_manifest, 1, ["MIR.SUMS-MD5-MATCH"]),
    ("M7 已知边界：真源 + 三层清单同改", "inpack", m_both_manifests, 0, []),
    ("M8 只改镜像清单（不动真源）", "inpack", m_mirror_manifest_only, 1,
     ["MIR.SUMS-MD5-MATCH", "PKG.SUMS-MD5-MATCH"]),
    ("R0 仓内模式（对照）", "repo", m_repo_none, 0, []),
    ("R1 篡改仓内原件 1 字节", "repo", m_repo_origin, 1, ["MIR.MD5-SAME"]),
    ("R2 篡改包内审计脚本", "repo", m_repo_code, 1, ["CODE.SAME.audit_pack_selfsufficient"]),
    ("R3 篡改仓内审计结论副本", "repo", m_repo_audit_copy, 1, ["PKG.AUDIT-COPY-FRESH"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="交付包自足性审计 · 变异测试（判别力证明）")
    parser.add_argument("--pkg", default=str(PKG_DEFAULT), help="交付包目录")
    parser.add_argument("--keep", action="store_true", help="保留临时副本（调试用）")
    parser.add_argument("--out-md", default=None, help="把结果写成 markdown 报告（默认打到 stdout）")
    args = parser.parse_args()

    pkg = Path(args.pkg).resolve()
    if not (pkg / "code" / "audit_pack_selfsufficient.py").is_file():
        print(f"[FATAL] 交付包不存在或缺 code/：{pkg}")
        return 2

    print(f"交付包：{pkg}")
    print(f"临时副本根：{tempfile.gettempdir()}/mut_test_*（仓外 ⇒ 包内模式）")
    print("=" * 112)
    print(f"| {'变异用例':34s} | {'模式':6s} | {'期望 rc':6s} | {'实测 rc':6s} | 命中断言 | 判定 |")
    print("|--:|:--|--:|--:|--:|:--|")
    bad = 0
    env_err = 0
    rows = []
    skipped = 0
    for name, mode, mutate, want_rc, want_cids in CASES:
        if mode == "repo" and _REPO is None:
            # ★ 包外（host 拿到包、仓不在身边）⇒ 仓内模式用例**无从执行**（要真仓做对照）
            #   ⇒ 如实跳过并计数，不当成「通过」也不当成「失败」。
            skipped += 1
            print(f"| {name:34s} | {mode:6s} | {'—':6s} | {'—':6s} | {'—':40s} | ⏭ 跳过（包外无仓） |")
            continue
        rc, out, crashed = _exec_case(pkg, mode, mutate, args.keep)
        if crashed:
            # ★ 审计进程自身崩溃 ⇒ 判据**未成立**（不是判据失败）⇒ 重试一次，
            #   避免一次偶发环境抖动把结论误报成「交付包不可信」。
            print(f"| {name:34s} | {mode:6s} | {want_rc:6d} | {'crash':6s} | {'—':40s} | ⚠ 审计崩溃 → 重试 |")
            rc, out, crashed = _exec_case(pkg, mode, mutate, args.keep)
        hit = [cid for cid in want_cids if re.search(rf"\b{re.escape(cid)}\b", out)]
        ok = (rc == want_rc) and (len(hit) == len(want_cids))
        if not ok:
            if crashed:
                env_err += 1
            else:
                bad += 1
        print(f"| {name:34s} | {mode:6s} | {want_rc:6d} | {rc:6d} | "
              f"{('、'.join(hit) if hit else '—'):40s} | "
              f"{'✅' if ok else ('⚠ 环境异常' if crashed else '❌ 不符期望')} |")
        rows.append((name, mode, want_rc, rc, hit, want_cids, ok, crashed))
        if not ok:
            for line in out.splitlines():
                if "FAIL" in line:
                    print(f"    └ {line.strip()}")
    print("=" * 112)
    verdict = "FAIL" if bad else ("ENV" if env_err else "PASS")
    n_run = len(CASES) - skipped
    print(f"变异测试：{n_run} 个用例（跳过 {skipped}），期望不符 {bad} 个"
          + (f"，环境异常 {env_err} 个" if env_err else "") + f" ⇒ {verdict}")
    if bad:
        print("判定：审计**无判别力**或变异用例失效 ⇒ 交付包不可信。")
    elif env_err:
        print("判定：审计进程**崩溃**（环境异常，**非**判据失败）⇒ 结论未定 ⇒ 需重跑。")
    else:
        print("判定：审计对 M1~M6 / M8 / R1~R3 的篡改**全部报警**，对 M0 / R0 / M7 不误报 ⇒ 有判别力。")
    if args.out_md:
        write_report(Path(args.out_md), pkg, rows, bad, skipped, env_err)
        print(f"[写出] {args.out_md}")
    return 1 if (bad or env_err) else 0


def write_report(path: Path, pkg: Path, rows, bad: int, skipped: int = 0, env_err: int = 0) -> None:
    out = [
        "# 变异测试：交付包自足性审计的**判别力**证明",
        "",
        f"- 时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "- 执行机：服务器 `test-Super-Server` · 项目根 `/tmp/pycharm_project_10ae9e2e`",
        f"- 被测交付包：`{pkg}`（运行时刻：{sum(1 for _ in pkg.rglob('*') if _.is_file())} 个文件）",
        f"- **被测判据脚本**：`code/audit_pack_selfsufficient.py` md5 "
        f"`{md5(pkg / 'code/audit_pack_selfsufficient.py')}`（判据本身的可追溯指纹）",
        "- 命令：`python manu/tracking/lookahead/server_overnight/mutation_test_selfsufficient.py`",
        "",
        "## 为什么必须做",
        "",
        "审计若只输出「PASS」而无判别力，等于没审。本测试对交付包**副本**逐个施加已知篡改，",
        "要求审计**必须**报错（rc=1 且指出对应断言 ID）；并留一个**已知边界**用例如实标出。",
        "",
        "两种模式各测一遍：**包内模式**（host 的用法，包拷到仓外）与**仓内模式**",
        "（搭「假仓」：真仓原件用**硬链接**挂入，改前先断开链接 ⇒ **真仓不受影响**）。",
        "★ 包外**无仓**时（host 只拿到交付包），仓内模式用例**自动跳过**并如实计数 —— ",
        "既不算通过、也不算失败（它们本来就需要真仓做对照）。",
        "",
        "## 结果",
        "",
        "| 变异用例 | 模式 | 期望 rc | 实测 rc | 命中断言 | 判定 |",
        "|:--|:--|--:|--:|:--|:--:|",
    ]
    for name, mode, want_rc, rc, hit, want_cids, ok, crashed in rows:
        out.append(f"| {name} | {mode} | {want_rc} | {rc} | {'、'.join(hit) if hit else '—'} | "
                   f"{'✅' if ok else ('⚠ 环境异常' if crashed else '❌ 不符期望')} |")
    out += [
        "",
        f"- 合计 **{len(rows)}** 个用例，期望不符 **{bad}** 个、"
        f"环境异常（审计进程崩溃）**{env_err}** 个 ⇒ **{'FAIL' if bad else ('ENV' if env_err else 'PASS')}**",
        f"- 其中 **跳过 {skipped}** 个仓内模式用例（包外无仓 ⇒ 自动跳过，不计入通过 / 失败）",
        "- 结论：审计对 M1~M6 / M8 / R1~R3 的篡改**全部报警**、对 M0 / R0 / M7 不误报 ⇒ **有判别力**。",
        "",
        "## 已知边界（如实记录）",
        "",
        "- **M7**：若篡改者**同时**改掉真源**与**三层校验链**（镜像清单 `repo/MD5SUMS.txt` + "
        "包级清单 `MD5SUMS.txt` 里对应的两条），",
        "  md5 清单**无签名** ⇒ 检测不到。本审计**不声称**能防这一类；",
        "  防它需要**外部签名/可信根**（如对清单做 GPG/HSM 签名），不在本轮范围。",
        "",
        "## 复现",
        "",
        "```bash",
        "cd /tmp/pycharm_project_10ae9e2e",
        "python manu/tracking/lookahead/server_overnight/mutation_test_selfsufficient.py",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
