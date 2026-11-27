# 红外图像质量评估 v2 · **deep 轮 host 侧收尾验收记录**（2026-11-27）

> **性质**：**收尾记录**，不是逐条判定书。
> 逐条判定的**正本**是 `iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md`（22 条 A/B/C/D 判定的逐条展开），
> 本文件只负责**另外四件正本不管的事**：
> ① **最终冻结快照**（各交付面的 md5 —— 正本里的快照指针会随报告重生成而过期，这里给最终值）；
> ② **host 与服务器两侧**的门禁实跑结果（正本只记了单侧快照）；
> ③ **判定器最终结论**（带上最终门禁 JSON 后的 22 条判定）；
> ④ **遗留项的处置**与**本轮明确没做的事**。
>
> **判据来源**：`iqa_v2/DEEP_AXIS_SPEC_20261127.md`（a-priori，写于任何深轮结果落盘之前，md5 `1f9d92de…`）。
> **判定器**：`iqa_v2/acceptance_deep.py`（纯标准库，**只吃 CSV、不采信报告正文**）。
> **执行机**：host 判定（`python3`）；门禁两侧各跑一遍 —— host 与
> `huangzhe@192.168.99.40:32222` 的 `/tmp/pycharm_project_a7d966e6`
> （`/home/huangzhe/anaconda3/envs/uav/bin/python`，`PYTHONPATH=/tmp/pylibs_cvcontrib`）。

---

## 〇、一句话结论

**deep 轮的门禁面已全绿，可交付。**
判定器 22 条判定：**PASS 19 / PARTIAL 1 / FAIL 2**；剩下的 **2 条 FAIL（A1、A4）是本轮如实记录的两条实测结论**，
**不是**门禁缺陷、也不是判定器缺陷 —— 一条说「曲线**没塌**，膝点只能叫『弯折点』」，
一条说「`local_scr` 在真实素材上**区分不开好坏片**」。
**对外可交付两条红线**：`target_local_contrast < 0.1005622` ＋ `target_lap_var < 220.6151`，
`local_scr` **不成立**（不进对外版），且**必须带齐限定段**（见 `REPORT_V2_DEEP_ACCEPTANCE.md` §六）。

---

## 一、最终冻结快照（收尾时刻）

| 交付面 | md5 | 说明 |
|:--|:--|:--|
| `iqa_v2/REPORT_V2_DEEP.md` | `46eff65164b1d0a8421686b99a0876bc` | 深轮报告正文（**由 CSV 现场生成**，重生成逐字节一致） |
| `iqa_v2/report_deep_v2.py` | `122e3347ca897ca11345daf4166c105f` | 报告生成器（**深轮剖面专用**，不碰基线轮生成器） |
| `iqa_v2/output_deep/manifest_deep.md5` | `542f0b864d86e274e057662d5fa4097a` | 深轮产物清单（**35 项** = 21 个代码 + 10 个 `output_deep/` 产物 + 4 份报告正文；自校验 35/35、登记名无重复） |
| `iqa_v2/output_deep/manifest_deep.bulk.canonical.txt` | `1adc546ba3bb4fb26dd5258be2d50743` | 旁挂**顺序无关**规范摘要（明细行序由 `imap_unordered` 决定 ⇒ 不能用裸 md5） |
| `iqa_v2/_deep_code_20261127/` | `report_deep_v2.py` = `122e3347ca897ca11345daf4166c105f` | **冻结深轮代码镜像**（**21 个** `.py` 逐字节拷贝 ＋ `README.md` 记来历与 md5 表；本轮随报告生成器的两处修订重冻一次） |
| `iqa_v2/MEMORY_SECTION_DEEP.md` | `5f0865ac44cb5d39610cabc9f012dcf6` | 记忆库章节成稿（占位 **§320**） |
| `manu/memory/memory.md` | `46dc5387fd179727cfa5eaec51e65cc0` | 记忆库**已并入**（活区 §320，**逐字节等同** `MEMORY_SECTION_DEEP.md`；`check_memory_roll` / `validate_nav_table` **全绿**） |
| `iqa_v2/_gates_deep_final_20261127.json` | `0ad6825bef7c9f017e816edcbee7606d` | **最终**门禁留痕（两侧） |
| `iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md` | `b7ac75e83186b124e5ba62ba961f9e12` | 验收记录**正本**（22 条逐条展开 ＋ 对外口径 ＋ 判定器缺陷审计 §7.3） |
| `iqa_v2/DEEP_AXIS_SPEC_20261127.md` | `1f9d92de857d6d2066fcc94f0647baf0` | a-priori 规格书（**host 与服务器逐字节相同** ⇒ 未被事后改过） |
| `iqa_v2/output/manifest_v2.md5` | `a72bc7fa5c267b853943416fb2670820` | **冻结基线清单**（本轮**逐字未改**） |
| `iqa_v2/REPORT_V2_FINAL.md` | `dbcfbedd9114d13a95538319bc7d8a22` | **冻结基线报告**（本轮**逐字未改**） |
| `iqa_v2/MEMORY_SECTION.md` | `a6bf9a81a5aedfa803153a02e1e505f6` | **冻结基线记忆章节**（§319，本轮**逐字未改**） |

🔴 **本文件自身不在深轮清单里** —— 它引用清单的 md5，清单若引用它的 md5 就成了**循环依赖**。
⇒ 深轮清单登记的是**产物 + 代码 + 报告正文**；**验收记录不入清单**（与基线轮同口径）。

---

## 二、门禁：host 与服务器**两侧**实跑（结论一致）

原始留痕：`iqa_v2/_gates_deep_final_20261127.json`（服务器侧逐项日志：`/tmp/iqa_gates_deep_20261127/`）。

| 门禁 | host | 服务器 | 结论 |
|:--|:--|:--|:--|
| `pytest iqa_v2/ -q` | **152 passed / 0 skipped / 0 failed**（带 `IQA_V2_DATASET_ROOT=<workspace>/data/anti-uav/4th` ⇒ 3 个真实帧用例**真跑**；收尾时新增 `iqa_v2/test_acceptance_deep.py` **21 条**判定器回归测试 ⇒ 131 → 152） | **150 passed / 2 skipped / 0 failed**（2 skipped = 数据可得性分支）—— 📌 服务器侧**已收尾**：深轮清单 **35 项**把 `iqa_v2/test_acceptance_deep.py` 登记为代码条目且服务器 `md5sum -c` rc=0 ⇒ 服务器**物理持有**该文件 ⇒ 同样 collected 152 条 ⇒ 152 − 2 skipped = **150 passed** | ✅ 两侧 rc=0 |
| `verify_recalc_v2.py --samples 120` | ROI 抽样 120 行最大绝对误差 **6.821e-13**；膝点复算 21 个 **0.000e+00**；曲线有效性 162 条**不一致 0 条** | 同口径**同结论** | ✅ 两侧 rc=0 |
| `audit_final.py iqa_v2/output_deep iqa_v2/REPORT_V2_DEEP.md` | **PASS 41 / FAIL 0 / WARN 0** | **PASS 41 / FAIL 0 / WARN 0** | ✅ 两侧 rc=0 |
| `audit_final.py iqa_v2/output iqa_v2/REPORT_V2_FINAL.md`（防基线被改） | **PASS 41 / FAIL 0 / WARN 0** | **PASS 41 / FAIL 0 / WARN 0** | ✅ 两侧 rc=0 |
| `md5sum -c iqa_v2/output_deep/manifest_deep.md5` | rc=0，**OK 35 / FAILED 0** | rc=0，**OK 35 / FAILED 0** | ✅ 两侧均 rc=0 / 35/35 OK（服务器侧已同口径重跑） |
| `canonical_digest.py verify`（明细顺序无关摘要） | rc=0，**ok 2 / failed 0**（digest `md5:82fb447b38160379cde55722e792fe12`） | rc=0，**ok 2 / failed 0** | ✅ 两侧 rc=0 |
| `compare_rounds.py --strict` | rc=0：`DEEP_STRICTER` / `DEEP_NEW` / `DEEP_NONE` ⇒ **无更松线位**（B5 成立）；⚠️ 同一次输出另给 `verdict: NEEDS_ACTION` / `notes_required: 3`（决定臂 `xcorr` → `kcf_legacy`、决定族 `motion` → `blur`）⇒ **已在报告 §6.2 逐条披露** | — | ✅ rc=0 |
| `acceptance_deep.py`（22 条判定，带最终门禁 JSON） | **PASS 19 / PARTIAL 1 / FAIL 2**（判定器为**修完 12 处缺陷**后的版本） | **PASS 19 / PARTIAL 1 / FAIL 2**（**两侧同结论**：判定器回归测试已改为**优先读两侧通用**的 `iqa_v2/_gates_deep_final_20261127.json`，服务器侧不再因「拿不到 host 专用留痕」而假失败） | ✅ 两侧 22 条**逐条一致** |

### 2.1 冻结基线清单（`iqa_v2/output/manifest_v2.md5`）裸跑为什么是红的

| 口径 | 结果 |
|:--|:--|
| **产物条目**（`iqa_v2/output/*.csv` + `*.json` + 3 份基线报告正文） | host **14/14 全 OK**；服务器 **15/15 全 OK**（多一条 cache 侧明细）⇒ **基线产物逐字节未动** |
| **代码条目** | 失败的**全部是代码条目**（深轮**刻意**演进过的代码，共 **7 条**：`audit_final.py` / `degradation_4d.py` / `make_manifest_v2.py` / `recheck_curves.py` / **`regression_v2.py`**（术语统一一并改到）/ `run_eval_v2.py` / `test_iqa_v2.py`）—— 裸 `md5sum -c` 拿**活代码**核对**历史清单**，任何一次后续改码都会变红 ⇒ **设计使然，非数据被改** |
| **host / 服务器两侧的失败数** | **host：rc=1，OK 17 / FAILED 8**（7 条代码漂移 ＋ 1 条 cache 侧绝对路径 `/tmp/iqa_v2_cache/degrade_detail.csv`，host 上无此路径 ⇒ 报「不可读」）；**服务器：rc=1，OK 18 / FAILED 7**（代码已与深轮清单逐条同步 ⇒ 漂移的是**同样这 7 条**；cache 明细在服务器侧可读 ⇒ 该条计入 OK）⇒ 📌 **这 7 条不是手抄的**：报告 §11.1 现场拿两份清单对账自动算，两台机器读同一份深轮清单 ⇒ 数字**两侧逐字相同** |
| **正确核对基准** | 代码条目的唯一可信依据是**冻结代码镜像** `iqa_v2/_baseline_code_20261126/`（基线轮）与 `iqa_v2/_deep_code_20261127/`（深轮轮） |
| **基线清单自身** | md5 `a72bc7fa5c267b853943416fb2670820`，**生成前后逐字未改** ⇒ 冻结证据链完好 |

🔴 **口径决定**：C4 的 `md5sum -c` 门禁 = **深轮清单**（host rc=0、**35/35**；服务器侧同口径 rc=0、**35/35**）；
**冻结基线清单一个字节都不许重生成**（重生成会抹掉「基线代码自 11-26 未变」这条证据）。

---

## 三、判定器最终结论（22 条）

| 系列 | 判定 | 含义 |
|:--|:--|:--|
| **A1** | 🔴 **FAIL** | deep 最重档跟住率最低 **87.9%**（blur/xcorr），无一组合跌破 50% ⇒ 曲线**没塌** ⇒ 膝点只是「**弯折点**」，**不是**「崩溃点」；对外措辞必须继续保留这个限定 |
| **A2** | 🟡 **PARTIAL** | KCF 臂 `target_local_contrast/blur`、`target_local_contrast/contrast`、`target_lap_var/blur` 过守卫，`…/motion` 未过 ⇒ **算法无关性本轮仍不成立**，对外必须继续带「**单臂反解**」限定 |
| **A3** | 🟢 **PASS** | `target_lap_var` 拿到有效红线 **220.61511928541879**（blur/xcorr）—— 深轮**唯一**的实质性收益 |
| **A4** | 🔴 **FAIL** | `local_scr` 盲测区分间隔仅 **2.00 pp** < 50 pp ⇒ **继续不进对外免责声明**（根因是**素材**里真实好片没有目标） |
| **B1~B6** | 🟢 **全部 PASS** | 守卫门槛/三分法/切分**一字未动**；盲测集两轮逐字节一致；池子与基线同池同窗口；线位**每族取跨臂最小**（绝不求平均）；deep 线位**不比基线更松**；档位表与规格书逐位一致 |
| **C1~C3** | 🟢 **全部 PASS** | 报告逐条给出 A1~A4/B1~B5 实测结论；对外版只保留成立的红线且带齐限定段；**基线产物 11/11 逐字节一致**、deep 走独立目录 |
| **C4** | 🟢 **PASS** | 四道门禁均 rc=0（带 `iqa_v2/_gates_deep_final_20261127.json`） |
| **D1~D8** | 🟢 **全部 PASS** | 报告正文 ↔ CSV 逐格对账全绿：逐档跟住率 **84** 读数、曲线分级 **27** 条跨度、约束线 3 条线位、盲测 3 组数字、判据总表 4 个单元格、最重档 15 读数、交叉引用可达性、对外阈值有效位数 |

🔴 **A1 / A4 的 FAIL 是「结论」不是「缺陷」**：判定器判的是「**深轮达成了规格书写的通过线吗**」，
规格书写明的失败分支正是「如实写『本方法在净化底座上造不出崩溃点』」与「`local_scr` 继续不给数值」——
**本轮正是照这两条失败分支写的**。⇒ 判定器与报告**互相印证**，不是互相打架。

---

## 四、与正本 `REPORT_V2_DEEP_ACCEPTANCE.md` 的分工（**不重复**）

| 事项 | 正本 `REPORT_V2_DEEP_ACCEPTANCE.md` | 本文件 |
|:--|:--|:--|
| 22 条判定的**逐条展开**（含每条的口径说明与重算细节） | ✅ 正本 | 只给**结论表** |
| 对外可发布版正文与限定段（逐字可抄） | ✅ 正本 | 指向正本 §六 |
| 报告残留缺陷清单与修法 | ✅ 正本 §七 | 只补**遗留项的最终处置**（下表） |
| **最终冻结快照**（各交付面 md5） | 快照会随报告重生成而**过期** | ✅ **本文件给最终值** |
| **host 与服务器两侧**门禁 | 只记了单侧快照（`_gates_deep_*.json`） | ✅ **本文件给两侧** |
| 记忆库章节（§320）成稿 | — | ✅ `iqa_v2/MEMORY_SECTION_DEEP.md` |

---

## 五、遗留项的最终处置

| # | 遗留 | 处置 |
|:--|:--|:--|
| 1 | 深轮主门禁曾是红的（清单登记的报告 md5 过期 + 代码条目漂移 + 清单里有 host 上不存在的 `audit_final_host.py`） | ✅ **已消除**：`test_deep_manifest_md5_all_ok` **PASSED**。做法 = ① 重建**冻结深轮代码镜像** `iqa_v2/_deep_code_20261127/`（**21 个** `.py` + `README.md`）；② 用 host 版 `make_manifest_v2.py` 重生成清单（**35 项**、自校验 35/35、登记名无重复，代码条目改走**镜像口径**核对）；③ `md5sum -c` **两侧 rc=0 / 35/35 OK** |
| 2 | 报告正文曾不披露 C4 的真实状态（只列深轮清单、没列基线清单裸跑是红的） | ✅ **已消除**：报告新增 **§11.1**「C4 的 `md5sum -c` 口径（必读：这里有两份清单，口径不同）」，两份清单**并列**、各自状态与作用写清 |
| 3 | §10.3 对外限定段未列「Anti-UAV 留出集错误豁免率 4.47% / 0.79%」 | ✅ **已消除**：该登记**当时属实**，但报告此后已重生成 —— 现行正牌 `iqa_v2/REPORT_V2_DEEP.md` 的 **§10.3 ③ 已含 4.47% / 0.79% 与 380 窗口口径**（Anti-UAV 序列级留出集，§7.1）⇒ **对外引用时随 ③ 一起给即可，无需另行手动补** |
| 4 | 交付面出现逐字节相同的副本 | ✅ **已处置**：正牌是 `iqa_v2/REPORT_V2_DEEP.md` 与 `iqa_v2/output_deep/manifest_deep.md5`；副本 `iqa_v2/report_deep_v2_fixed.md` **已移出仓库**（`mv` 到 `/tmp`，保留可追溯性）。🔶 **有意保留** `iqa_v2/output_deep/manifest_deep.server_20261127.md5`（md5 `89e1cc75…`，2250 B）—— 它是**服务器侧 10:19 那一版清单的历史快照**（**与现清单已不同**：现清单 **35 项** / md5 `65fd62a2…`），被 `REPORT_V2_DEEP_ACCEPTANCE.md` §八 引用作留痕，**不是**当前清单的副本。✅ `iqa_v2/audit_final_host.py` 现已**与服务器同源同步到 host** ⇒ 它**在**深轮清单里（35 项 = 21 个 `.py` + 10 个 `output_deep/` 产物 + 4 份报告正文），两侧 `md5sum -c` 都过 |
| 5 | 旁挂规范摘要清单登记的是**绝对路径** ⇒ host 与服务器两侧字节不同 | ✅ **已修**：`make_manifest_v2.py` 改为**仓库内登记相对路径、仓库外仍用绝对路径** ⇒ 旁挂清单**两侧逐字节相同**，两侧 `verify` 都是 rc=0 |

---

## 六、复现命令（两侧各跑一遍）

```bash
# ── host（仓库根）──
export IQA_V2_DATASET_ROOT=<workspace>/data/anti-uav/4th
# 🔴 pytest 必须用带 cv2/numpy/pytest 的解释器（系统 /usr/bin/python3 **没装 pytest**）：
<workspace>/.venv-iqa/bin/python -m pytest iqa_v2/ -q
python3 iqa_v2/verify_recalc_v2.py \
    --detail-csv /tmp/iqa_v2_cache_deep/degrade_detail.csv \
    --knee-csv iqa_v2/output_deep/knee_thresholds_v2.csv \
    --class-csv iqa_v2/output_deep/segment_class_v2.csv \
    --windows-csv iqa_v2/output_deep/windows.csv \
    --manifest-json iqa_v2/output_deep/dataset_manifest.json \
    --dataset-root "$IQA_V2_DATASET_ROOT" --samples 120
md5sum -c iqa_v2/output_deep/manifest_deep.md5
python3 iqa_v2/canonical_digest.py verify iqa_v2/output_deep/manifest_deep.bulk.canonical.txt
python3 iqa_v2/audit_final.py iqa_v2/output_deep iqa_v2/REPORT_V2_DEEP.md
python3 iqa_v2/audit_final.py iqa_v2/output iqa_v2/REPORT_V2_FINAL.md
# 🔴 compare_rounds.py 的 --baseline/--deep 收的是 **constraint_lines_v2.csv 路径**（不是目录）
python3 iqa_v2/compare_rounds.py \
    --baseline iqa_v2/output/constraint_lines_v2.csv \
    --deep iqa_v2/output_deep/constraint_lines_v2.csv --strict
python3 iqa_v2/acceptance_deep.py \
    --deep-dir iqa_v2/output_deep --baseline-dir iqa_v2/output \
    --report iqa_v2/REPORT_V2_DEEP.md \
    --acceptance iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md \
    --gates-json iqa_v2/_gates_deep_final_20261127.json

# ── 服务器（huangzhe@192.168.99.40:32222）──
cd /tmp/pycharm_project_a7d966e6
export PYTHONPATH=/tmp/pylibs_cvcontrib
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
$PY -m pytest iqa_v2/ -q
$PY iqa_v2/verify_recalc_v2.py --detail-csv /tmp/iqa_v2_cache_deep/degrade_detail.csv \
    --knee-csv iqa_v2/output_deep/knee_thresholds_v2.csv \
    --class-csv iqa_v2/output_deep/segment_class_v2.csv \
    --windows-csv iqa_v2/output_deep/windows.csv \
    --manifest-json iqa_v2/output_deep/dataset_manifest.json --samples 120
md5sum -c iqa_v2/output_deep/manifest_deep.md5
$PY iqa_v2/canonical_digest.py verify iqa_v2/output_deep/manifest_deep.bulk.canonical.txt
$PY iqa_v2/audit_final.py iqa_v2/output_deep iqa_v2/REPORT_V2_DEEP.md
```

🔴 **服务器侧 `pyproject.toml` 必须与 host 一致**（`norecursedirs` 排除 `_baseline_code_*` / `_deep_code_*`）：
否则 `--doctest-modules` 会把两个冻结镜像里的**同名模块**抢进 `sys.path`
⇒ `import file mismatch` ⇒ **38 个 collection error**（本轮踩过，见 §五 前置）。

---

## 七、本轮**明确没有做**的事（别误以为做了）

| 项 | 状态 |
|:--|:--|
| 训练 / 重训 / 换跟踪探针 | ❌ **没有**。三条臂（XCorr / KCF / CSRT）实现与基线轮**逐字相同**；正式交付模型（`trial_0080` / RC2）替换**待 58 条业务标注回收后**执行 |
| 重跑深轮评估 / 重生成产物 | ❌ **没有**。深轮产物 `iqa_v2/output_deep/` 是既有结果，本轮只做**核验与封存**（产物 CSV **一个字节都没动**） |
| 数据集下载 | ❌ **没有** |
| 改动基线轮产物 / v1 代码 | ❌ **没有**。`iqa_v2/output/`、`iqa_v2/REPORT_V2_FINAL.md`、`iqa_v2/MEMORY_SECTION.md`、`manu/video_quality/` 全部**未动** |
| 改动 a-priori 规格书 | ❌ **没有**。`iqa_v2/DEEP_AXIS_SPEC_20261127.md` 在 host 与服务器**逐字节相同**（`1f9d92de…`）⇒ 判据未被事后改过 |
| git commit / push | ❌ **没有**（按工作约定，本轮**不提交**） |

---

## 八、对外引用时的三条硬约束（**缺一条就是超发**）

1. **算法无关性本轮不成立**（A2 部分通过）⇒ 两条线是「**单臂反解 ＋ 独立盲测背书**」，**不是**「双算法互证」（两族反解线位相差 **53.99%** / **38.69%**）。
2. **膝点是「弯折点」不是「崩溃点」**（A1 不通过，最重档跟住率仍 **87.9%**）⇒ 对外只保证「低于此画质跟踪性能开始明显劣化」，**不得**宣称「跟踪崩溃」。
3. **红线只在 Anti-UAV 开源红外上标定**⇒ 只证明**方法有效**，**正式上板红线待内部 58 条红外业务 GT 回收后用同一套流水线重标**；
   `local_scr` **本轮不成立**，**不进**对外免责声明（**证据最弱，不建议对外使用**）。

