# 冻结深轮（deep）代码镜像 · 2026-11-27

**这是什么**：深轮（deep round）**交付时**的那一版 `iqa_v2/*.py`，逐字节拷贝。

**为什么需要它**（与 `_baseline_code_20261126/` 同理）：
深轮清单 `iqa_v2/output_deep/manifest_deep.md5` 里登记了代码条目的 md5。
工作区的代码会**继续演进**（例如后续又修了 `acceptance_deep.py` 的 C3 核对口径）
⇒ 拿**当前**代码去核对一份**历史**清单必然假失败 ⇒ 深轮门禁名存实亡。

⇒ 解法：把交付时的代码冻结一份在这里，清单的**代码条目**在这里核对；
清单的**产物条目**（`iqa_v2/output_deep/*`、报告正文）仍然**就地核对**。

由 `iqa_v2/test_iqa_v2.py` 的 `DEEP_CODE_DIR` 引用；
`iqa_v2/acceptance_deep.py` 的 C3 判定（`--baseline-code-dir`）也用它核对基线代码条目。

🔴 **不要删本目录**：删掉之后深轮清单的代码条目就失去了唯一可信的核对依据，
只剩「拿当前代码核对历史清单」⇒ 必然假失败 ⇒ C4 门禁必然红。

🔴 **本目录里的文件一个字都不许改**：它们是证据。改了就等于改了深轮轮的证据链。

逐字节来源：2026-11-27 深轮交付时的 `iqa_v2/*.py` 工作区拷贝。
本轮把同一份代码**同步到服务器**后再在服务器侧生成
`iqa_v2/output_deep/manifest_deep.md5` ⇒ **两侧代码 md5 逐条相同** ⇒
**同一份清单**在 host 与服务器两侧 `md5sum -c` 都是 rc=0（不再另开 server 版清单）。

## 冻结时各文件的 md5


> 2026-11-27 收尾刷新：判定器 `acceptance_deep.py` 修完 12 处**假通过/崩溃**缺陷（C1 恒过、D1/D2/D4/D6 遇不可解析单元格静默过、D7 跳过仍计入核对数、A1 误把对照臂 csrt 算进、C2 遇 None 崩溃等），并新增 `iqa_v2/test_acceptance_deep.py`（21 条永久回归测试）⇒ 本镜像与深轮清单同步刷新到这一版。

> 2026-11-27 收尾刷新（第二次）：报告生成器 `report_deep_v2.py` 修两处**会写进交付正文的假陈述** ——
> ① §11.1 原先把基线清单的失败项**硬写**成「6 条漂移代码 + 1 条 cache 明细 / 第 7 条失败」，
> 而 host 实测是 **7 条漂移代码**（漏了术语统一后一并改动的 `regression_v2.py`）+ 1 条 cache 侧绝对路径
> ⇒ **host OK 17 / FAILED 8**；现改为**现场拿两份清单对账**（基线清单 vs **深轮清单**登记的代码 md5）自动算，
> 两台机器读同一份深轮清单 ⇒ 算出来的数字**两侧逐字相同**，不再手抄；
> ② 原先**没有披露** `compare_rounds.py --strict` 判定的「决定口径变化」（`verdict: NEEDS_ACTION`、
> `notes_required: 3`：决定臂 `xcorr` → `kcf_legacy`、决定族 `motion` → `blur`）⇒ 现已在 §6.2 **逐条披露**。
> ⇒ 本镜像与深轮清单随之同步刷新到这一版（`report_deep_v2.py` md5 `122e3347…`）。

✅ **两侧已重新同步**：刷新新增的 `test_acceptance_deep.py` 已同步到服务器，
⇒ **两侧跑的是同一份代码、同一份清单**，`md5sum -c` 在 host 与服务器两侧**均 rc=0**；
最终口径与数字见 `iqa_v2/HOST_ACCEPTANCE_DEEP_20261127.md` §二 与
`iqa_v2/_gates_deep_final_20261127.json`。

🔴 **本镜像刷新时同步修掉的一处「两侧结论不一致」**：`iqa_v2/test_acceptance_deep.py` 原先
把判定器的门禁 JSON 指向 `iqa_v2/_gates_deep_host_20261127.json`（**只有 host 有**）
⇒ 服务器侧判定器拿不到门禁结果 ⇒ C4 判不通过 ⇒
`test_real_artifacts_keep_the_22_verdicts` 在**服务器侧假失败**。
现已改为优先用**两侧通用**的 `iqa_v2/_gates_deep_final_20261127.json`
（缺失时才回退到 host 专用文件）⇒ 同一份代码在两侧给出同一组 22 条结论。

🔴 **清单条目数随本镜像内容走**：`iqa_v2/make_manifest_v2.py` 用
`repo.glob("iqa_v2/*.py")` 登记代码条目 ⇒ **镜像里有多少个 `.py`，清单里就有多少条代码条目**
（本镜像 **21** 个 ⇒ 清单代码条目 **21** 条；再加 10 条产物 + 4 条报告正文 = **35** 项）。
新增/删除 `iqa_v2/*.py`（例如把 host 专用的 `audit_final_host.py` 收进工作区）**必须**
同步刷新本镜像与清单，否则 `md5sum -c` 与 `test_deep_manifest_md5_all_ok` 会立刻变红。

| `iqa_v2/acceptance_deep.py` | `bfaff96f1b8b334745e43562ff7362bf` |
| `iqa_v2/audit_final.py` | `1634a0c4e529b2c4be2540c54aaac8ec` |
| `iqa_v2/audit_final_host.py` | `968b2e248bca9b6125c5010fe9158639` |
| `iqa_v2/canonical_digest.py` | `b926947154840843acf95608084ffcf2` |
| `iqa_v2/compare_rounds.py` | `0054342ab5947f2fe1155fd05fc2b152` |
| `iqa_v2/dataset_anti_uav.py` | `cc66815f9ac302d3a2aa8603210d4196` |
| `iqa_v2/degradation_4d.py` | `617e12f30812e53b400cc6586d3d9774` |
| `iqa_v2/fix_deep_report_heaviest.py` | `7ea85989bb90431bc1727dc8d63feba8` |
| `iqa_v2/make_manifest_v2.py` | `f7b543abb69dc63af514e275a273519e` |
| `iqa_v2/recheck_curves.py` | `f2e76543150a79a9326fdd79eae6cbf8` |
| `iqa_v2/regression_v2.py` | `d9b4632565aad9e1a594b5c128524441` |
| `iqa_v2/report_deep_v2.py` | `122e3347ca897ca11345daf4166c105f` |
| `iqa_v2/run_eval_v2.py` | `ad5e7ba543d3c1e1f1428ad92c4ad2a7` |
| `iqa_v2/target_roi_metrics.py` | `8622fc0c82342592a40a5e7bfcef8fdd` |
| `iqa_v2/test_acceptance_deep.py` | `74c6228ec81f56d5e3f82b27544d1b95` |
| `iqa_v2/test_canonical_digest.py` | `4b698d4ac95b42fd2ced3e31dbf957d7` |
| `iqa_v2/test_compare_rounds.py` | `2543adf9faddf2535dd918ef63cec489` |
| `iqa_v2/test_deep_round_guard.py` | `fb5c7b8c730d45febcf5fb919a409977` |
| `iqa_v2/test_iqa_v2.py` | `3988e8c8f7ec2d77e91a095c2892481e` |
| `iqa_v2/test_recheck_curves.py` | `0585c344a9176c1043bd37ba36a4263f` |
| `iqa_v2/verify_recalc_v2.py` | `1833cec3b06a22ddf97d65d6c112758e` |
