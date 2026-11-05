#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立审计：交付包的**自足性**（包外能不能独立复核头条数字）。

背景（夜 12 抓到的真缺口）：
  host 只拉 `runs/lookahead/{fi_nogmc,final}/` ⇒ **24 序真源 `arms.json`、§17 帧级 oracle dump、
  回退门产物、OSD 视频都不在拉取范围内** ⇒ 包里的审计脚本读不到真源，"严格的精度对比报告"
  只能看已生成的 md，**无法独立复核**。

处置：交付包内新增 `repo/` **仓镜像**（按仓内相对路径镜像真源），
  包内 `code/` 的五份审计脚本自定位到本镜像 ⇒ 包外可独立复跑。

本脚本核**静态前提**（动态复跑由 `coldstart_verify_delivery.sh` 的 5/5 阶段负责）：
  1. 包内 `MD5SUMS.txt` 覆盖每个文件且逐位 md5 相同；
  2. 镜像内每个文件与仓内**同相对路径**原件 md5 逐位相同（镜像不是陈旧/被改过的副本）；
  3. 镜像**不缺**任何仓内 `arms.json`（真源完整性）；
  4. 五份审计各自读取的关键输入都在镜像里；
  5. 包内 `code/` 的五份审计脚本与仓内同名文件 md5 相同；
  6. **自定位验证**：把包内审计脚本搬到**仓外**临时目录 import（不执行 `main`），
     断言其根定位指向**包内镜像** ⇒ 证明在包外跑读的是镜像、不是仓。

设计原则（与 audit_sec5_deliverables.py 同一范式）：只用标准库；只读、无副作用（除 --out-md）。

用法：
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_pack_selfsufficient.py
退出码：0 = 无 FAIL；1 = 存在 FAIL。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_SELF = Path(__file__).resolve()
_PKG_CAND = _SELF.parent.parent          # <包>/code/x.py ⇒ <包>
# ★ 包身份**只看脚本自身位置**：脚本在 <包>/code/ 下且 <包> 同时有 MD5SUMS.txt 与 repo/ ⇒ 包内模式。
#   早先靠「向上找仓根」判定 ⇒ ①包被放进任何含 manu/tracking/lookahead 的目录树下时会去审**别的**包；
#   ②包根叫 repo 时会误判。现在这两种摆放下都正确。
IN_PACK = (_SELF.parent.name == "code" and (_PKG_CAND / "MD5SUMS.txt").is_file()
           and (_PKG_CAND / "repo").is_dir())
_REPO_ROOT = next((a for a in _SELF.parents if (a / "manu/tracking/lookahead").is_dir()), None)
if IN_PACK:
    ROOT = _PKG_CAND / "repo"                            # 仓镜像
    PACK = _PKG_CAND                                    # 包根（含 code/ 与 MD5SUMS.txt）
    MIRROR = ROOT
    OUT_MD = PACK.parent / "audit_pack_selfsufficient_inpack.md"   # 写在**包外**：不污染包与镜像
else:
    # 仓内模式：仓根由祖先推导；**仓外**（自定位沙箱）⇒ 兜底到「脚本旁的 repo/ 镜像」
    ROOT = _REPO_ROOT or (_PKG_CAND / "repo")
    PACK = ROOT / "runs/lookahead/final/nogmc_fi_20261104"
    MIRROR = PACK / "repo"
    OUT_MD = ROOT / "runs/lookahead/fi_nogmc/audit_round7/audit_pack_selfsufficient.md"

AUDITS = ["audit_sec16_gap", "audit_sec17_reg_oracle", "audit_sec17_rbgate",
          "audit_sec5_acceptance", "audit_sec5_deliverables"]
# 包内 code/ 要与仓内同名的**全部**审计脚本逐位相同 —— 含本脚本自身
# （早先把本脚本漏掉 ⇒ 包里可以塞一份改过的审计而不被发现）。
CODE_FILES = AUDITS + ["audit_pack_selfsufficient"]

# 五份审计直接读取的文件（相对仓根）—— 镜像必须齐备
AUDIT_INPUTS = [
    "runs/lookahead/fi_nogmc/FINAL_REPORT.md",
    "runs/lookahead/fi_nogmc/report.md",
    "runs/lookahead/fi_nogmc/report_house_style.md",
    "runs/lookahead/fi_nogmc/ruler1_gate_check.json",
    "runs/lookahead/fi_nogmc/workpoint/audit.json",
    "runs/lookahead/fi_nogmc/workpoint/audit.md",
    "runs/lookahead/fi_nogmc24/report_house_style_d2.md",
    "runs/lookahead/fi_nogmc24/report_house_style_d3.md",
    "runs/lookahead/fi_nogmc24/report_house_style_d4.md",
    "runs/lookahead/fi_nogmc24/report_house_style_d8.md",
    "runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json",
    "runs/lookahead/fi_nogmc24/oracle_accel/closure_report.md",
    "runs/lookahead/fi_nogmc24/oracle_accel/oracle_bounds.json",
    "runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json",
    "runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.json",
    "runs/lookahead/fi_nogmc24/gapdiag/gap_report.json",
    "runs/lookahead/gateA/gateA.json",
    "runs/lookahead/gateA24/gateA.json",
    "runs/lookahead/rbgate24nogmc/progress.txt",
    "runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json",
]

# 镜像**自己生成**的文件（仓内没有对应原件）—— 不参与「与仓内逐位对账」
MIRROR_ONLY = {"README.md"}

# --- 新鲜度判据（G1 / G1b）用的常量 -----------------------------------------------
# 交付包根下的权威报告 + 随包审计结论目录 / 仓内审计结论目录
PKG_REPORT = "FINAL_REPORT.md"
PKG_AUDIT_DIR = "audit_round7"
REPO_AUDIT_DIR = "runs/lookahead/fi_nogmc/audit_round7"
# 五份审计的**随包结论**（`_round7_findings` 是历史留痕、故意不刷新 ⇒ 不参与指纹判据）
FINGERPRINT_AUDITS = ["audit_sec16_gap.md", "audit_sec17_reg_oracle.md", "audit_sec17_rbgate.md",
                      "audit_sec5_acceptance.md", "audit_sec5_deliverables.md"]
# 审计报告每次运行都会带新的时间戳 ⇒ 逐位比对前必须抹掉（与冷启动脚本同一套归一化）
TS_RE = re.compile(r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d(:\d\d)?")

_results: list[tuple[str, str, bool, str]] = []


def rec(level: str, cid: str, ok: bool, detail: str) -> None:
    # ★ 判定必须挂在 ok 上：早先所有 CHECK 断言只置 ok=False，而 FAIL 统计是按 level 找 "FAIL"
    #   ⇒ 失败项永远抓不到、rc 恒 0 ⇒ 审计「永远 PASS」、毫无判别力。这里把失败项统一落成 FAIL。
    _results.append(("FAIL" if not ok else level, cid, ok, detail))


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 各项检查
# 包内复跑时，五份审计把**运行产物**写在镜像的 audit_round7/ 下 —— 那不是真源、
# 也不在镜像清单里（清单是打包时生成的）⇒ 覆盖判据要把这类文件排除，否则自伤。
MIRROR_OUT_DIR = "runs/lookahead/fi_nogmc/audit_round7"
# ★ 只认**固定文件名**（五份审计各自的输出）：目录前缀整片放行会给篡改者留免检口子
#   （往 audit_round7/ 里塞任意文件就绕过了覆盖判据）⇒ 豁免必须是白名单，不是路径前缀。
RUNTIME_OUTPUTS = {"audit_sec16_gap.md", "audit_sec17_reg_oracle.md", "audit_sec17_rbgate.md",
                   "audit_sec5_acceptance.md", "audit_sec5_deliverables.md"}


def _is_runtime_output(rel: str) -> bool:
    """**运行产物** —— 不参与「未列入清单」判据，但也不是篡改。

    两类：① 五份审计自己写的报告（镜像 `audit_round7/` 下、**文件名白名单**）；
    ② 跑包内代码时 Python 生成的 `__pycache__/*.pyc`。
    除此之外多出来的文件一律 FAIL —— 那才是篡改/污染。
    """
    r = rel[len("repo/"):] if rel.startswith("repo/") else rel
    parts = r.split("/")
    if "__pycache__" in parts or r.endswith(".pyc"):
        return True
    return parts[:-1] == MIRROR_OUT_DIR.split("/") and parts[-1] in RUNTIME_OUTPUTS


def verify_sums(base: Path, sums: Path, tag: str) -> None:
    """校验 base 下的 MD5SUMS.txt：**清单每条逐位 md5 相同** + **覆盖完整**。

    这是**篡改检测**的本体：清单里任何一条对不上（改了/删了）⇒ FAIL；
    磁盘上多出清单外的文件 ⇒ FAIL。
    """
    rec("CHECK", f"{tag}.SUMS-EXISTS", sums.is_file(), f"校验清单：{sums}")
    if not sums.is_file():
        return
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        # ★ 不用 `str.removeprefix`（Python 3.9+）—— 交付包要在 **host 的 python3** 上跑，
        #   而不少服务器的系统 python3 仍是 **3.8** ⇒ 用等价的切片写法（3.7+ 通用）。
        listed[rel[2:] if rel.startswith("./") else rel] = digest
    # ★ 只排除**本层自己的**清单文件（`base/MD5SUMS.txt`）。
    #   早先排除「所有名字含 MD5SUMS 的文件」⇒ 包级清单漏掉 `repo/MD5SUMS.txt`
    #   ⇒ 镜像清单本身**无锚**（改镜像清单，包级察觉不到）⇒ 两把锁其实只剩一把。
    #   现在包级清单**锚住**镜像清单 ⇒ 单改镜像清单，包级立刻发现。
    sums_rel = str(sums.relative_to(base))
    on_disk = {str(p.relative_to(base)) for p in base.rglob("*")
               if p.is_file() and str(p.relative_to(base)) != sums_rel}
    missing = sorted(rel for rel in listed if rel not in on_disk)
    bad = sorted(rel for rel, digest in listed.items()
                 if rel in on_disk and md5(base / rel) != digest)
    unlisted = sorted(rel for rel in on_disk - set(listed) if not _is_runtime_output(rel))
    rec("CHECK", f"{tag}.SUMS-NO-MISSING", not missing,
        f"清单 {len(listed)} 条中，磁盘上缺失 {len(missing)} 个"
        + (f"：{missing[:5]}" if missing else ""))
    rec("CHECK", f"{tag}.SUMS-MD5-MATCH", not bad,
        f"逐位 md5 复核 {len(listed)} 个文件，不一致 {len(bad)} 个"
        + (f"：{bad[:5]}" if bad else ""))
    rec("CHECK", f"{tag}.SUMS-COVERAGE", not unlisted,
        f"磁盘文件 {len(on_disk)} 个，清单 {len(listed)} 条；未列入 {len(unlisted)} 个"
        + (f"：{unlisted[:5]}" if unlisted else ""))


def check_pyc_outside_code() -> None:
    """包内模式：`.pyc` 只允许出现在 `code/__pycache__/`（跑包内代码的运行产物）。

    交付包里本来一个 `.pyc` 都没有 ⇒ 其它位置出现编译残留 ⇒ FAIL。
    （仓内模式由 `PKG.NO-PYC` 单独盯整包；这里补的是包内这条路径的对应判据。）
    """
    bad = sorted(str(p.relative_to(PACK)) for p in PACK.rglob("*.pyc")
                 if "__pycache__" not in p.parts)
    rec("CHECK", "PKG.NO-PYC-OUTSIDE-CODE", not bad,
        f"包内 .pyc（`code/__pycache__/` 之外）{len(bad)} 个"
        + (f"：{bad[:5]}" if bad else "（无编译残留）"))


def check_pack_checksums() -> None:
    """包内 MD5SUMS.txt 覆盖每个文件且逐位相同。"""
    pyc = sorted(str(p.relative_to(PACK)) for p in PACK.rglob("*.pyc"))
    rec("CHECK", "PKG.NO-PYC", not pyc,
        f"包内 .pyc / __pycache__ {len(pyc)} 个" + (f"：{pyc[:5]}" if pyc else "（已清，无编译残留）"))
    verify_sums(PACK, PACK / "MD5SUMS.txt", "PKG")


def check_mirror_against_repo() -> None:
    """镜像内每个文件与仓内同相对路径原件 md5 逐位相同。"""
    files = sorted(p for p in MIRROR.rglob("*") if p.is_file() and p.name != "MD5SUMS.txt")
    rec("CHECK", "MIR.NONEMPTY", bool(files), f"镜像文件数：{len(files)}")
    same, bad, absent = 0, [], []
    for p in files:
        rel = p.relative_to(MIRROR)
        if str(rel) in MIRROR_ONLY:
            continue
        src = ROOT / rel
        if not src.is_file():
            absent.append(str(rel))
            continue
        if md5(src) == md5(p):
            same += 1
        else:
            bad.append(str(rel))
    rec("CHECK", "MIR.MD5-SAME", not bad,
        f"镜像 {len(files)} 个文件与仓内原件逐位 md5 相同 {same} 个，不一致 {len(bad)} 个"
        + (f"：{bad[:5]}" if bad else ""))
    rec("CHECK", "MIR.SRC-EXISTS", not absent,
        f"镜像中仓内已无对应原件的文件 {len(absent)} 个" + (f"：{absent[:5]}" if absent else ""))


def check_mirror_completeness() -> None:
    """仓内全部 arms.json（排除交付包目录）都在镜像里 —— 真源不缺。"""
    src_all = {str(p.relative_to(ROOT)) for p in (ROOT / "runs/lookahead").rglob("arms.json")
               if PACK not in p.parents}
    mir_all = {str(p.relative_to(MIRROR)) for p in MIRROR.rglob("arms.json")}
    missing = sorted(src_all - mir_all)
    rec("CHECK", "MIR.ARMS-COMPLETE", not missing,
        f"仓内真源 arms.json {len(src_all)} 个，镜像 {len(mir_all)} 个，缺 {len(missing)} 个"
        + (f"：{missing[:5]}" if missing else ""))


def check_audit_inputs() -> None:
    """五份审计直接读取的输入必须在镜像里齐备。"""
    miss = [rel for rel in AUDIT_INPUTS if not (MIRROR / rel).is_file()]
    rec("CHECK", "MIR.AUDIT-INPUTS", not miss,
        f"审计输入 {len(AUDIT_INPUTS)} 项，镜像缺 {len(miss)} 项" + (f"：{miss[:5]}" if miss else ""))
    dumps = sorted((ROOT / "runs/lookahead/fi_nogmc24/reg_oracle").glob("*.jsonl"))
    dm = [d.name for d in dumps if not (MIRROR / "runs/lookahead/fi_nogmc24/reg_oracle" / d.name).is_file()]
    rec("CHECK", "MIR.ORACLE-DUMPS", not dm,
        f"§17 帧级 dump {len(dumps)} 个，镜像缺 {len(dm)} 个" + (f"：{dm[:5]}" if dm else ""))
    rec("CHECK", "MIR.OSD-VIDEOS", (MIRROR / "runs/lookahead/osd/videos").is_dir(),
        "OSD 视频目录存在（§P4 视觉证据）")


def check_code_same_as_repo() -> None:
    """包内 code/ 的六份审计脚本（含本脚本自身）与仓内同名文件 md5 相同。"""
    for name in CODE_FILES:
        a = PACK / "code" / f"{name}.py"
        b = ROOT / "manu/tracking/lookahead" / f"{name}.py"
        ok = a.is_file() and b.is_file() and md5(a) == md5(b)
        rec("CHECK", f"CODE.SAME.{name}", ok,
            f"包内 code/{name}.py 与仓内同名文件 md5 相同" if ok
            else f"包内 code/{name}.py 缺失或与仓内不一致")


def check_self_locating() -> None:
    """把包内审计脚本搬到**仓外**临时目录 import（不执行 main）⇒ 根定位必须指向包内镜像。

    必须在**仓外**验：包在仓内时，向上走会先命中仓根（仓内跑本来就该读仓）；
    host 拉到本地后包不在任何仓里 ⇒ 这时必须落到包内 `repo/` 镜像。
    """
    tmp = Path(tempfile.mkdtemp(prefix="packonly_locate_"))
    try:
        shutil.copytree(PACK / "code", tmp / "code")
        (tmp / "repo").symlink_to(MIRROR, target_is_directory=True)
        for anc in tmp.parents:
            if (anc / "manu" / "tracking" / "lookahead").is_dir():
                rec("FAIL", "LOCATE.SANDBOX", False,
                    f"临时目录的祖先里存在仓根（{anc}）⇒ 自定位测试失真")
                return
        for name in AUDITS:
            script = tmp / "code" / f"{name}.py"
            if not script.is_file():
                rec("FAIL", f"LOCATE.{name}", False, "包内脚本缺失，无法验证自定位")
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"_pack_{name}", script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                got = getattr(mod, "ROOT", None) or getattr(mod, "REPO", None)
            except Exception as exc:  # pragma: no cover - 只在脚本自身崩时触发
                rec("FAIL", f"LOCATE.{name}", False, f"import 失败：{exc!r}")
                continue
            want = (tmp / "repo").resolve()
            ok = got is not None and Path(got).resolve() == want
            # 详情**不打印随机临时路径**（否则本报告每次内容都不同 ⇒ 无法进包对账）
            rec("CHECK", f"LOCATE.{name}", ok,
                f"仓外 import 后根定位 = <SANDBOX>/repo（应为包内镜像），实际={Path(got).name == 'repo'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _norm(text: str) -> str:
    """抹掉时间戳 —— 审计报告每次运行都会带新的时间戳 ⇒ 逐位比对前必须归一化。"""
    return TS_RE.sub("<TS>", text)


def check_audit_fingerprint() -> None:
    """包内审计结论必须与包内权威报告**同代**（结论内嵌的 md5 指纹 = 实际 md5）。

    「只重打包不重跑审计」会把**上一代**的审计结论连同旧指纹一起打包 ⇒ 指纹对不上 ⇒ 报警。
    `_round7_findings` 是历史留痕（故意不刷新）⇒ 不参与本判据。
    """
    report = PACK / PKG_REPORT
    if not report.is_file():
        rec("CHECK", "PKG.AUDIT-FINGERPRINT", False, f"交付包缺权威报告 {PKG_REPORT}")
        return
    want = md5(report)
    pinned, stale, absent = [], [], []
    for name in FINGERPRINT_AUDITS:
        f = PACK / PKG_AUDIT_DIR / name
        if not f.is_file():
            absent.append(name)
            continue
        hits = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if PKG_REPORT not in line:
                continue
            hits += re.findall(r"\b[0-9a-f]{32}\b", line)
        if not hits:
            continue
        if all(h == want for h in hits):
            pinned.append(name)
        else:
            stale.append(f"{name}:{[h[:8] for h in hits]}")
    ok = not stale and not absent and bool(pinned)
    rec("CHECK", "PKG.AUDIT-FINGERPRINT", ok,
        f"随包 {PKG_REPORT} md5={want[:8]}…；审计结论内嵌指纹对齐 {len(pinned)}/{len(FINGERPRINT_AUDITS)} 份"
        + (f"；陈旧 {stale}" if stale else "")
        + (f"；缺失 {absent}" if absent else "")
        + ("" if pinned else "（无任何审计结论锚定指纹 ⇒ 判据空转）"))


def check_audit_copy_fresh() -> None:
    """仓内模式：包内 `audit_round7/` 的审计结论副本必须与仓内**当前**产物一致（新鲜度）。

    只重打包不重跑审计（包内是陈旧结论）、或跑完审计没重打包（包内落后于仓内）⇒ 两边对不上 ⇒ 报警。
    """
    src_dir = ROOT / REPO_AUDIT_DIR
    dst_dir = PACK / PKG_AUDIT_DIR
    if not src_dir.is_dir():
        rec("CHECK", "PKG.AUDIT-COPY-FRESH", False, f"仓内审计结论目录缺失：{REPO_AUDIT_DIR}")
        return
    src = {p.name: p for p in src_dir.iterdir() if p.is_file()}
    dst = {p.name: p for p in dst_dir.iterdir() if p.is_file()} if dst_dir.is_dir() else {}
    only_src = sorted(set(src) - set(dst))
    only_dst = sorted(set(dst) - set(src))
    diff = []
    for name in sorted(set(src) & set(dst)):
        a = _norm(src[name].read_text(encoding="utf-8", errors="replace"))
        b = _norm(dst[name].read_text(encoding="utf-8", errors="replace"))
        if a != b:
            diff.append(name)
    ok = not (only_src or only_dst or diff)
    rec("CHECK", "PKG.AUDIT-COPY-FRESH", ok,
        f"仓内 {len(src)} 份 vs 包内 {len(dst)} 份审计结论（时间戳已归一化）："
        f"仅仓内 {len(only_src)}、仅包内 {len(only_dst)}、内容不一致 {len(diff)}"
        + (f"：{(only_src + only_dst + diff)[:5]}" if (only_src or only_dst or diff) else ""))


# ------------------------------------------------------------------ 报告
def write_report(path: Path) -> None:
    fails = [r for r in _results if r[0] == "FAIL"]
    n_ok = sum(1 for r in _results if r[2])
    out = [
        "# 独立审计：交付包自足性（包外能不能独立复核）",
        "",
        f"- 时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "- 执行机：服务器 `test-Super-Server` · 项目根 `/tmp/pycharm_project_10ae9e2e`",
        f"- 交付包：`{PACK}`（{sum(1 for p in PACK.rglob('*') if p.is_file())} 文件）",
        f"- 仓镜像：`{MIRROR}`（{sum(1 for p in MIRROR.rglob('*') if p.is_file())} 文件）",
        f"- 运行模式：**{'包内模式（仓不在身边 ⇒ 只校验包清单 + 镜像清单）' if IN_PACK else '仓内模式（可与仓内原件逐位对账）'}**",
        ("- 命令：`python code/audit_pack_selfsufficient.py`（包内）"
         if IN_PACK else
         "- 命令：`PYTHONPATH=. python manu/tracking/lookahead/audit_pack_selfsufficient.py`（仓内）"),
        "",
        "## 结论",
        "",
        f"- 共 **{len(_results)} 项断言**：PASS **{n_ok}**，FAIL **{len(fails)}**。",
        ("- 判据（**包内模式**，本模式**只执行**下面四条）：① 包顶层 `MD5SUMS.txt` 覆盖且逐位相同；"
         "② 镜像 `repo/MD5SUMS.txt` 覆盖且逐位相同；③ 除 `code/__pycache__/` 外无 `.pyc` 残留；"
         "④ 随包审计结论与随包 `FINAL_REPORT.md` **同代**（结论内嵌 md5 指纹 = 实际 md5）。"
         if IN_PACK else
         "- 判据：① 包内 `MD5SUMS.txt` 覆盖且逐位相同；② 镜像与仓内原件**逐位 md5 相同**；"
         "③ 镜像**不缺**任何真源 `arms.json`；④ 五份审计的输入齐备；⑤ 包内审计脚本与仓内同源；"
         "⑥ 包外 import 后根定位指向**包内镜像**（不是仓）；"
         "⑦ 随包审计结论与随包 `FINAL_REPORT.md` **同代**；"
         "⑧ 包内 `audit_round7/` 副本 = 仓内**当前**产物（新鲜度）。"),
        ("- ⚠️ **包内模式不执行**「镜像与仓内原件逐位对账 / 镜像不缺 arms.json / 审计输入齐备 / "
         "code/ 与仓内同源 / 仓外 import 自定位」这五条（**仓不在身边，无从对账**）——"
         "需要它们请在仓内运行本脚本。"
         if IN_PACK else
         "- ℹ️ 仓内模式**额外**做「镜像 vs 仓内原件」逐位对账与「包外 import 自定位」验证；"
         "包内模式做不到这两项（仓不在身边），故按上面的「判据」段只报两条清单判据。"),
        "- ⚠️ 失败判据说明：**任一断言失败即计入 FAIL 并返回 rc=1**（失败项的级别由 `ok` 决定，"
        "不再按调用处传入的级别字符串判定）。",
        "",
        "## 断言明细",
        "",
        "| # | 级别 | 断言 ID | 判定 | 详情 |",
        "|--:|:--|:--|:--|:--|",
    ]
    for i, (lvl, cid, ok, detail) in enumerate(_results, start=1):
        out.append(f"| {i} | {lvl} | `{cid}` | {'✅ PASS' if ok else '❌ FAIL'} | {detail} |")
    out += [
        "",
        "## 复现",
        "",
        "```bash",
        *(["cd <交付包目录>", "python code/audit_pack_selfsufficient.py"] if IN_PACK else
          ["cd /tmp/pycharm_project_10ae9e2e",
           "PYTHONPATH=. /home/huangzhe/anaconda3/envs/uav/bin/python "
           "manu/tracking/lookahead/audit_pack_selfsufficient.py"]),
        "```",
        "",
        "（动态复跑：包外独立跑五份审计并与随包审计报告逐位比对 ⇒ "
        "`bash manu/tracking/lookahead/server_overnight/coldstart_verify_delivery.sh` 的 5/5 阶段）",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="独立审计：交付包自足性")
    parser.add_argument("--out-md", default=None, help="本审计报告的输出路径（默认随运行模式）")
    args = parser.parse_args()

    if IN_PACK:
        # 包内模式（host 拿到包、仓不在身边）：没有「仓内那份包」可比 ⇒ 校验
        # ①**包顶层清单**（覆盖包内每个文件，含镜像里的全部真源）②**镜像清单**，
        # 并明确提示「与仓内原件对账」要在仓内跑（否则会误以为包坏了）。
        print(f"[INFO] 检测到包内运行（根 = {ROOT}）⇒ 校验包清单 + 镜像清单自洽性；"
              "与仓内原件的逐位对账请在仓内运行本脚本。")
        verify_sums(PACK, PACK / "MD5SUMS.txt", "PKG")
        verify_sums(MIRROR, MIRROR / "MD5SUMS.txt", "MIR")
        check_pyc_outside_code()
        check_audit_fingerprint()
        rec("CHECK", "PKG.IN-PACK", MIRROR.is_dir() and (MIRROR / "runs/lookahead").is_dir(),
            f"包内运行路径已识别（包根 = {PACK.name}，镜像含 runs/lookahead）")
    elif PACK.is_dir():
        check_pack_checksums()
        verify_sums(MIRROR, MIRROR / "MD5SUMS.txt", "MIR")
        check_mirror_against_repo()
        check_mirror_completeness()
        check_audit_inputs()
        check_code_same_as_repo()
        check_audit_fingerprint()
        check_audit_copy_fresh()
        check_self_locating()
    else:
        print(f"[FATAL] 交付包不存在：{PACK}")
        return 2

    fails = [r for r in _results if r[0] == "FAIL"]
    n_ok = sum(1 for r in _results if r[2])
    print("=" * 104)
    print(f"交付包自足性审计{'（包内模式）' if IN_PACK else ''}：共 {len(_results)} 项断言 —— "
          f"PASS {n_ok} 项，FAIL {len(fails)} 项")
    print("=" * 104)
    for lvl, cid, ok, detail in _results:
        if lvl == "FAIL" or not ok:
            print(f"[{lvl:5s}] {cid:34s} {detail}")
    out_md = Path(args.out_md) if args.out_md else OUT_MD
    write_report(out_md)
    print(f"[写出] {out_md}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
