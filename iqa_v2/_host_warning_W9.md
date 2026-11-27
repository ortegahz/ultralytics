# W9 · host→server 交棒条（**host 18:00 关机，这条 host 通道随后消失**）

> 投递：2026-11-26 17:59 CST（host 侧）
> 本条是 W6/W7/W8 之后的**第四条 host 通道**，讲三件事：① host 关机后的工作面；② 验收仪器已升级并就位；③ 17:49 的自检快照与三条待确认项。
> ⚠️ 与 W6/W7/W8 不重复：那三条讲口径与守卫，这一条讲「host 下线后你独自收尾」。

## 一、host 侧现状：所有 host 专属工作已清空

| 项 | 状态 |
|:--|:--|
| 服务器主会话 | **在跑**（`codex exec` pid 2505229，`setsid+nohup` ⇒ host 断连/关机后**继续存活**） |
| 交棒书 / 任务书 | 已在服务器 `/tmp/pycharm_project_a7d966e6/iqa_v2/`（md5 一致） |
| W6 / W7 / W8 交棒条 | 已在服务器同目录（`_host_warning_W6.md` / `_W7.md` / `_W8.md` / `_W8_tailframe_classification.md`） |
| 验收仪器 | **已升级并装到服务器**：`iqa_v2/audit_final.py`（562 行，md5 `ce07d0b1bfd34dee4617867b7eb31423`） |
| host 专属脚本 | `merge_memory.sh` / `collect_from_server.sh` 只在 host 侧有，**服务器不需要**（记忆库合并以 host 权威副本为准） |
| 主机 `manu/memory/` | **host 侧是权威副本** ⇒ 服务器仍然**不要**改服务器上的 `manu/memory/memory.md` |

⇒ **关机后你就是唯一工作面**：不再有 host 侧的独立复核通道，所有自检由你自己跑。

## 二、验收仪器已硬化（**这是你交付前的自检工具，务必用新版**）

旧版 `audit_final.py`（267 行）靠**文件名片段**猜产物角色（`find_csv(root,"window")`），新产物改叫 `segment_class_v2.csv` 就直接漏检 ⇒ 误判 FAIL。新版（562 行）三处硬化：

1. **按表头内容识别产物**（列名签名）：明细底表 / 切分表 / 分类表 / 膝点表 / 约束线表 / 回归表 / 盲测表。
2. **独立复算**：不采信流水线写出的膝点与约束线，直接从明细底表重算分箱曲线 → 保序回归 → Kneedle → PCHIP，再与产物表逐项比对 ⇒ 能抓「产物写错」与「口径写错」。
3. **退化守卫**：约束线若来自「跟住率从未塌掉」的曲线（y_span≈0）⇒ 判 FAIL（本轮真实踩过的坑：平曲线上的浮点噪声被当成膝点）。

自检命令（服务器）：

```bash
cd /tmp/pycharm_project_a7d966e6
/home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/audit_final.py .
```

## 三、17:49 的自检快照（**在你 17:49 重算约束线的同时跑的，属瞬时态**）

总计 27 条：**PASS 22 / FAIL 3 / WARN 2**（WARN 不算通过）。

| 编号 | 结论 | 证据 | 我的判断 |
|:--|:--|:--|:--|
| R1 | ❌ FAIL | 找不到含 `condition_family` 的窗口明细 CSV | 产物表识别到 blind/constraint/knee/regression/segment/split/windows，**但没有明细底表**。17:39 拉回来的那份叫 `degrade_detail.full.csv`(10.9MB) / `.smoke.csv`，**服务器 `output/` 下现在没有这两个文件** ⇒ 要么被重跑清掉、要么改名 ⇒ 请确认**退化明细底表必须留在 `output/`**（它是独立复算的唯一输入，丢了验收仪器就退化了） |
| R4f | ❌ FAIL | 回归表门槛 ≠ 约束线：`target_local_contrast` 0.10143 vs 0.18917、`target_lap_var` 505.62 vs 541.09、`local_scr` 4.4858 vs 5.8285 | `constraint_lines_v2.csv` mtime **17:49**、`regression_v2.csv` mtime **17:40** ⇒ 疑似你正在重算膝点/约束线、**回归表还是旧口径** ⇒ 交付前请用**同一版**约束线重跑回归 |
| R6c | ❌ FAIL | 免责声明里的门槛没跟上约束线：`target_lap_var` 541.093、`local_scr` 5.82855 未出现在免责声明里 | 同上同因：免责声明是旧约束线写的 |
| R5i | ⚠️ WARN | 约束线表无 `cross_arm_dev_pct` 列 | 跨臂线位差未量化 |
| R9 | ⚠️ WARN | 报告出现 `trial_0080`/RC2 模型名，需人工确认写的是「已替换」还是「待替换」 | **G7 已定：保持 XCorr+KCF、待标注回收后替换**（host 已核查：58 条 `scores.json` 仍是 2026-11-09 14:50 的模型预标注，人工标注未回收）⇒ 报告须明确写「**待业务标注回收后按同口径复测执行**」 |

**结论**：3 个 FAIL 全部落在「**你正在重算、产物时间戳不一致**」的窗口上；请在**所有重算收口后**再跑一次 `audit_final.py`，**要求 FAIL=0**（WARN 需在报告里写清判读依据）。

## 四、host 下线后你收尾的清单（照此执行即可）

1. 跑 `audit_final.py` 直到 **FAIL = 0**；WARN 逐条在报告里给判读依据。
2. 退化明细底表（`condition_family` 那张表）**必须**留在 `output/` 并进 `manifest_v2.md5`。
3. **重生成清单**：`/home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/make_manifest_v2.py`（清单生成于 12:35，之后代码改过 ⇒ 现在 `md5sum -c` 与 pytest 的 `test_output_manifest_md5_all_ok` 必然红）。
4. 三道门禁全绿：`pytest iqa_v2/test_iqa_v2.py -v`、`verify_recalc_v2.py --samples 120`（误差 < 1e-6）、`md5sum -c iqa_v2/output/manifest_v2.md5`。
5. 交付物齐全：`iqa_v2/REPORT_V2_FINAL.md`（详细报告，章节按任务书 §G8 的 ①~⑫）、`iqa_v2/MEMORY_SECTION.md`（可直接并入记忆库的成稿章节，标题写成 `## §318 · …` 占位即可，**host 侧合并脚本会自动重编号为 §319**）、`iqa_v2/output/` 全部 CSV + `manifest_v2.md5`、`iqa_v2/PROGRESS_SERVER.md` 追加一行本轮结论。
6. 报告里的数字**从 CSV 现场生成**，不要手抄。
7. **不要**改服务器上的 `manu/memory/memory.md`（host 侧是权威副本，会分叉）；成稿章节只写 `iqa_v2/MEMORY_SECTION.md`。
8. 收尾时在 `PROGRESS_SERVER.md` 留一行：`时间 / 剩余缺口 / 做了什么 / 产物路径 / 结论数字`，并写明「host 已关机，无 host 侧复核通道，验收由服务器自检完成」。

## 五、host 侧脚本的独立审计结论（**给你知情，不用你动**）

host 侧对 `collect_from_server.sh` / `merge_memory.sh` 做了独立代码审计，查出 **13 处会导致错误结论**的缺陷（假绿、pytest 失败计数被 ANSI 码挡住、`scp` 丢目录、过期 manifest、merge fallback **静默吞掉整章正文**等），已在 host 侧修（`merge_memory.sh` 的 fallback 现在是「前置标题」不是「整体覆盖」，并加了「正文逐行一致 + 标题恰好 1 个」的硬断言）。**这两个脚本只影响 host 回来后的回收与合并动作，不影响你在服务器上的评测与报告**。

🔴 **但请记住一条**：host 的记忆库合并脚本只有在「章节标题含 §」时才会正确重编号 ⇒ 你写的 `MEMORY_SECTION.md` **标题必须带 `## §`**，否则该章节会落不进记忆库索引。
