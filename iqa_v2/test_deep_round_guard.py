#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深轮「**不得覆盖冻结基线产物**」闸门的回归测试（独立审计发现的 4 个洞）。

覆盖的 4 条
------------
1. **FIX-1（BLOCKER）**：闸门曾把基线绑定值按 **CWD** 解析 ⇒ ``cd /tmp`` 之后深轮
   可以直写 ``iqa_v2/REPORT_V2_FINAL.md``（冻结基线报告被毁）⇒ 绑定值必须锚到
   **仓库根**（由脚本位置推导），两侧都取**解析后**的路径；
2. **FIX-2（BLOCKING）**：受保护集合曾只有 ``PROFILE_PATHS`` 的 3 条 ⇒ 清单里登记、
   但不在 ``PROFILE_PATHS`` 里的 ``REPORT_V2_ANTIUAV.md`` / ``MEMORY_SECTION.md``
   会被深轮悄悄覆盖 ⇒ 保护集合必须来自**冻结清单** ``iqa_v2/output/manifest_v2.md5``；
3. **FIX-3（IMPORTANT）**：deep 阶梯曾只要求「包含」基线阶梯 ⇒ 拿**基线轴**冒充
   深轴（meta 却记 ``conditions_profile: deep``）⇒ 必须逐族**严格超集**（⊋）；
4. **FIX-4（IMPORTANT）**：明细底表缺失时 ``run_meta.json`` 曾记
   ``profile_matches_detail: true`` ⇒ 「**没查**」必须记 ``null``。

纪律（本文件自身也守 C3）
------------------------
* 所有 CLI 运行都指向 ``tmp_path`` ⇒ **不写** ``iqa_v2/output/`` 与
  ``iqa_v2/output_deep/``；
* 模块级 fixture 在测试前后比对**冻结基线产物**的 md5 与 ``iqa_v2/output/`` 的
  文件清单 ⇒ 闸门一旦失效（或测试自己手滑），立刻失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "run_eval_v2.py"
MANIFEST = HERE / "output" / "manifest_v2.md5"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from degradation_4d import (BLUR_SIGMAS, CONTRAST_ALPHAS, MOTION_LENGTHS,  # noqa: E402
                            build_conditions)
from run_eval_v2 import (PROFILE_PATHS, _guard_profile_ladder,  # noqa: E402
                         _write_run_meta)


# --------------------------------------------------------------------- 小工具
def _md5(path: Path) -> str | None:
    """文件 md5；文件不存在返回 ``None``。"""
    if not path.is_file():
        return None
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchor(raw: str) -> Path:
    """测试侧的「仓库根锚定」（相对路径按仓库根解释 ⇒ 与 CWD 无关）。"""
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def _manifest_products() -> list[Path]:
    """冻结清单里登记的**产物**（跳过代码文件——代码本轮会改，md5 必然变）。"""
    products: list[Path] = []
    if not MANIFEST.is_file():
        return products
    for raw_line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        entry = parts[1].strip()
        if not entry or entry.endswith(".py"):
            continue
        path = Path(entry)
        products.append(path if path.is_absolute() else REPO / path)
    return products


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """跑一次 ``run_eval_v2.py``（调用方负责把三个产物路径都指到 ``tmp_path``）。"""
    return subprocess.run([sys.executable, str(RUNNER)] + list(args),
                          cwd=str(cwd), capture_output=True, text=True, check=False)


def _cli_args(tmp_path: Path, overrides: dict[str, str] | None = None) -> list[str]:
    """构造一次深轮 CLI 调用：三个产物路径默认全指向 ``tmp_path``，``overrides`` 覆盖之。"""
    values = {"--output-dir": str(tmp_path / "out"),
              "--cache-dir": str(tmp_path / "cache"),
              "--report": str(tmp_path / "report.md")}
    values.update(overrides or {})
    args = ["--phase", "report", "--conditions-profile", "deep"]
    for flag, value in values.items():
        args += [flag, value]
    return args


def _assert_guard_fired(result: subprocess.CompletedProcess, artifact: str) -> None:
    """闸门必须**在写任何东西之前**拦下（argparse 的 usage 错误 ⇒ 返回码 2）。"""
    output = result.stdout + result.stderr
    assert result.returncode == 2, (
        "闸门没拦下来（返回码 %s，应当是 argparse 的 2）：\n%s"
        % (result.returncode, output[-800:]))
    assert "基线" in output, "拒绝信息里必须点明「基线」：\n%s" % output[-800:]
    assert artifact in output, (
        "拒绝信息里必须点名被保护的产物 %s：\n%s" % (artifact, output[-800:]))


@pytest.fixture(scope="module", autouse=True)
def _frozen_baseline_untouched():
    """整轮测试期间**冻结基线产物**必须一字不变（闸门失效 ⇒ 这里立刻炸）。"""
    output_dir = HERE / "output"
    before_listing = {path.name: _md5(path)
                      for path in sorted(output_dir.glob("*")) if path.is_file()}
    before_md5 = {path: _md5(path) for path in _manifest_products()}
    yield
    after_listing = {path.name: _md5(path)
                     for path in sorted(output_dir.glob("*")) if path.is_file()}
    after_md5 = {path: _md5(path) for path in _manifest_products()}
    assert before_listing == after_listing, (
        "iqa_v2/output/ 的文件清单/md5 被改动 ⇒ 有东西写进了冻结目录：%s"
        % sorted(set(before_listing) ^ set(after_listing)))
    assert before_md5 == after_md5, "冻结基线产物的 md5 变了 ⇒ 基线产物被覆盖"


# ------------------------------------------------------------------- 闸门用例
@pytest.mark.parametrize("flag,value,artifact", [
    ("--output-dir", "iqa_v2/output", "iqa_v2/output"),
    ("--cache-dir", "/tmp/iqa_v2_cache", "iqa_v2_cache"),
    ("--report", "iqa_v2/REPORT_V2_FINAL.md", "REPORT_V2_FINAL.md"),
])
def test_guard_rejects_baseline_bound_paths(tmp_path, flag, value, artifact):
    """deep 剖面指向**三条基线绑定路径**时必须硬拒（规格书 C3）。"""
    result = _run_cli(_cli_args(tmp_path, {flag: value}), cwd=REPO)
    _assert_guard_fired(result, artifact)


def test_guard_still_fires_from_foreign_cwd(tmp_path):
    """🔴 **FIX-1 回归**：CWD **不是**仓库根时，闸门**照样**必须拦下。

    旧实现把基线绑定值按 **CWD** 解析 ⇒ ``cd /tmp`` 之后深轮能直写
    ``iqa_v2/REPORT_V2_FINAL.md``（冻结基线报告被毁）⇒ 现在绑定值锚到仓库根，
    且**同一个文件的多种写法**（绝对 / 相对 / 带 ``..`` / 软链）都要命中。
    """
    baseline_report = REPO / "iqa_v2" / "REPORT_V2_FINAL.md"
    baseline_output = REPO / "iqa_v2" / "output"
    link = tmp_path / "link_report.md"
    link.symlink_to(baseline_report)
    cases = [
        (str(baseline_report), "REPORT_V2_FINAL.md"),                  # 绝对写法
        ("iqa_v2/REPORT_V2_FINAL.md", "REPORT_V2_FINAL.md"),           # 相对写法
        ("iqa_v2/../iqa_v2/REPORT_V2_FINAL.md", "REPORT_V2_FINAL.md"),  # 带 ..
        (str(link), "REPORT_V2_FINAL.md"),                             # 软链
        (str(baseline_output), "iqa_v2/output"),                       # 基线产物目录
    ]
    for value, artifact in cases:
        result = _run_cli(_cli_args(tmp_path, {"--report": value}), cwd=tmp_path)
        _assert_guard_fired(result, artifact)

    # 缓存目录同理（基线绑定值里的绝对路径，也必须从**任何** CWD 都拦得住）
    result = _run_cli(_cli_args(tmp_path, {"--cache-dir": "/tmp/iqa_v2_cache"}),
                      cwd=tmp_path)
    _assert_guard_fired(result, "iqa_v2_cache")


@pytest.mark.parametrize("name", ["REPORT_V2_ANTIUAV.md", "MEMORY_SECTION.md"])
def test_guard_rejects_manifest_only_artifacts(tmp_path, name):
    """🔴 **FIX-2 回归**：清单里登记、但**不在** ``PROFILE_PATHS`` 里的产物同样受保护。

    旧实现只比对 ``PROFILE_PATHS`` 的 3 条 ⇒ 这两个基线产物会被深轮直接覆盖。
    """
    artifact = REPO / "iqa_v2" / name
    bound = {_anchor(value) for value in PROFILE_PATHS["baseline"].values()}
    assert artifact.resolve() not in bound, "该产物本来就不在 PROFILE_PATHS 里（测试前提失效）"
    assert artifact.resolve() in {path.resolve() for path in _manifest_products()}, (
        "该产物必须登记在冻结清单里（测试前提失效）")
    result = _run_cli(_cli_args(tmp_path, {"--report": str(artifact)}), cwd=REPO)
    _assert_guard_fired(result, name)


def test_deep_ladder_must_be_strict_superset(tmp_path):
    """🔴 **FIX-3 回归**：deep 阶梯必须是基线阶梯的**严格超集**（⊋），逐族判定。

    只「包含」基线阶梯**不够**：档数与基线相同 ⇒ 退化上限根本没被抬高 ⇒ 跑出来的
    还是基线轴，meta 却会记 ``conditions_profile: deep``（记录与事实相反）。
    """
    baseline_conditions = build_conditions("deep", blur_sigmas=BLUR_SIGMAS,
                                           motion_lengths=MOTION_LENGTHS,
                                           contrast_alphas=CONTRAST_ALPHAS)
    problem = _guard_profile_ladder("deep", baseline_conditions)
    assert problem, "deep 剖面原样用基线阶梯（没加档）必须被拒"
    assert "严格超集" in problem

    extended = build_conditions("deep", blur_sigmas=BLUR_SIGMAS + (5.0,),
                                motion_lengths=MOTION_LENGTHS + (13,),
                                contrast_alphas=CONTRAST_ALPHAS + (0.10,))
    assert _guard_profile_ladder("deep", extended) is None, "三族各多一档必须放行"

    # CLI 层：拿基线轴冒充深轴 ⇒ 必须被拒
    result = _run_cli(_cli_args(tmp_path, {
        "--blur-sigmas": "0.6,1.0,1.5,2.2,3.0,4.0",
        "--motion-lengths": "3,5,7,9,11",
        "--contrast-alphas": "0.8,0.65,0.5,0.35,0.25,0.15"}), cwd=REPO)
    output = result.stdout + result.stderr
    assert result.returncode == 2, "CLI 层必须硬拒基线轴冒充深轴：\n%s" % output[-800:]
    assert "严格超集" in output, "拒绝信息里必须点明「严格超集」：\n%s" % output[-800:]

    # CLI 层：三族各多一档 ⇒ 过闸门、进入阶段执行
    ok_result = _run_cli(_cli_args(tmp_path, {
        "--blur-sigmas": "0.6,1.0,1.5,2.2,3.0,4.0,5.0",
        "--motion-lengths": "3,5,7,9,11,13",
        "--contrast-alphas": "0.8,0.65,0.5,0.35,0.25,0.15,0.10"}), cwd=REPO)
    ok_output = ok_result.stdout + ok_result.stderr
    assert "严格超集" not in ok_output, "三族各多一档必须放行：\n%s" % ok_output[-800:]
    assert (tmp_path / "out" / "run_meta.json").is_file(), (
        "三族各多一档应当过闸门并进入阶段执行：\n%s" % ok_output[-800:])


def test_run_meta_without_detail_csv_has_null_profile_match(tmp_path):
    """🔴 **FIX-4 回归**：明细底表缺失 ⇒ ``profile_matches_detail`` 必须是 ``null``。

    旧实现记 ``matches = not detail_levels`` ⇒ **什么都没验**也记 ``true``。
    """
    meta_dir = tmp_path / "meta"
    cli = argparse.Namespace(conditions_profile="deep", output_dir=str(meta_dir),
                             detail_csv=str(meta_dir / "degrade_detail.csv"))
    _write_run_meta(cli, build_conditions("deep"), status="completed")
    meta = json.loads((meta_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["profile_matches_detail"] is None, (
        "没测到明细轴就不许声称核对通过：%s" % meta)
    assert meta["status"] == "completed"

    # CLI 层同理：明细 CSV 不存在时，写出来的 meta 也不许是 true
    result = _run_cli(_cli_args(tmp_path, {"--detail-csv": str(tmp_path / "missing.csv")}),
                      cwd=REPO)
    assert result.returncode != 0, "明细缺失时阶段应当失败（而不是「核对通过」）"
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["profile_matches_detail"] is None, (
        "没测到明细轴就不许声称核对通过：%s" % meta)
    assert meta["status"] == "in_progress", "阶段未完成时 status 不得声称完成：%s" % meta


# ---------------------------------------------------------------------------
# 第 5~7 条：独立审计（本轮）新发现的 3 个洞
# ---------------------------------------------------------------------------
#: 深轮报告生成器必须读的产物文件（复制到 tmp 用）
_DEEP_PRODUCTS = (
    "degrade_detail_v2.csv", "segment_class_v2.csv", "split_v2.csv",
    "knee_thresholds_v2.csv", "constraint_lines_v2.csv", "regression_v2.csv",
    "blind_real_v2.csv", "correlations_v2.csv", "dataset_manifest.json",
    "run_meta.json",
)


def test_deep_detail_csv_cannot_overwrite_the_frozen_baseline_detail(tmp_path):
    """🔴 **明细底表也是「输出」路径** ⇒ 必须和产物目录一样受保护。

    洞：闸门只查 ``--output-dir`` / ``--cache-dir`` / ``--report`` 三条，
    ``--detail-csv`` 没查 ⇒ ``--detail-csv iqa_v2/output/degrade_detail_v2.csv``
    能把**冻结的基线明细**覆盖掉（那正是 C3 要保的东西之一）。
    """
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--phase", "report",
         "--conditions-profile", "deep",
         "--detail-csv", "iqa_v2/output/degrade_detail_v2.csv"],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode != 0, "deep 剖面竟然允许写基线明细 CSV"
    assert "基线" in (result.stderr + result.stdout), (
        "拒绝信息里必须点明「基线」：%s" % (result.stderr + result.stdout)[-500:])


def test_missing_frozen_manifest_blocks_the_deep_profile(tmp_path, monkeypatch):
    """🔴 **冻结清单缺失 ⇒ 硬失败**，不许静默「降级」成三条路径后照跑。

    降级 = 闸门变弱（``REPORT_V2_ANTIUAV.md`` / ``MEMORY_SECTION.md`` 这些
    清单登记、但不在 ``PROFILE_PATHS`` 里的基线产物就不再受保护），
    却仍然放行 ⇒ 深轮可能悄悄覆盖基线产物而无人察觉 ⇒ C3 证据链断。
    """
    import run_eval_v2

    monkeypatch.setitem(run_eval_v2.PROFILE_PATHS, "baseline",
                        dict(run_eval_v2.PROFILE_PATHS["baseline"],
                             output_dir=str(tmp_path)))
    protected, source = run_eval_v2._protected_artifacts()
    assert not protected and not source, (
        "清单缺失时必须返回空集合 + 空来源（由调用方硬失败），"
        "实得 %d 条 / %r" % (len(protected), source))

    monkeypatch.setattr(sys, "argv",
                        ["run_eval_v2.py", "--phase", "report", "--conditions-profile", "deep"])
    with pytest.raises(SystemExit) as exc:
        run_eval_v2.main()
    assert exc.value.code != 0, "冻结清单缺失时 CLI 竟然放行了"


def test_deep_report_phase_is_written_by_the_deep_generator(tmp_path):
    """🔴 **一个报告路径只能有一个生成器**：deep 剖面的报告阶段必须走
    ``report_deep_v2.py``（剖面感知），**不能**走基线轮的 ``phase_report``。

    洞：``phase_report`` 的正文**与剖面无关**（仍是基线轮口径的「本轮唯一可对外的
    红线是 target_local_contrast」「target_lap_var 本轮没有有效红线」…）⇒
    手册里那条唯一合法命令 ``run_eval_v2.py --conditions-profile deep``
    会把**修好的**深轮报告覆盖成基线口径 ⇒ 判定器 D5（报告正文无基线轮遗留）
    当场变红；而且它还注入一行机器负载快照 ⇒ 报告非字节可复现 ⇒ md5 门禁
    换台机器必然假红。
    """
    deep_dir = tmp_path / "deep"
    deep_dir.mkdir()
    for name in _DEEP_PRODUCTS:
        src = HERE / "output_deep" / name
        if src.is_file():
            shutil.copy2(src, deep_dir / name)

    report = tmp_path / "REPORT_V2_DEEP.md"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--phase", "report", "--conditions-profile", "deep",
         "--output-dir", str(deep_dir),
         "--detail-csv", str(deep_dir / "degrade_detail_v2.csv"),
         "--cache-dir", str(tmp_path / "cache"),
         "--report", str(report)],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, "deep 报告阶段失败：%s" % (result.stderr + result.stdout)[-800:]
    assert "[REPORT-DEEP]" in result.stdout, (
        "deep 剖面的报告阶段必须由深轮生成器写（期望 [REPORT-DEEP]），实得：%s"
        % result.stdout[-500:])
    text = report.read_text(encoding="utf-8")
    # 深轮专属内容在
    assert "最重档跟住率" in text, "报告里没有「最重档跟住率」表"
    assert "0.1005622" in text and "220.6151" in text, "报告里没有本轮红线数值"
    # 基线轮遗留不在（这些是 ``phase_report`` 的口径）
    for leftover in ("本轮唯一可对外的红线", "本轮没有有效红线"):
        assert leftover not in text, (
            "报告里出现基线轮遗留措辞 %r ⇒ 写它的不是深轮生成器" % leftover)
