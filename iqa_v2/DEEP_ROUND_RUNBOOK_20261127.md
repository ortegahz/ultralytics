# 深轮（deep）执行手册 · 2026-11-27

> **本文件是「怎么跑、怎么验」的操作手册，不是判据。**
> 验收判据（A1~A4 / B1~B5 / C1~C4）**全部**以 `iqa_v2/DEEP_AXIS_SPEC_20261127.md` 为准，
> 那份规格书是**任何结果产生之前**写下的防泄露凭证 ⇒ 本手册**一个字都不改它**，
> 只补充「操作性」信息（执行机、命令、门禁清单）。
> 若本手册与规格书冲突，**以规格书为准**。

---

## 一、执行机与环境

| 项 | 值 |
|:--|:--|
| 服务器 | `huangzhe@192.168.99.40:32222` |
| conda 环境 | `uav` |
| 解释器 | `/home/huangzhe/anaconda3/envs/uav/bin/python` |
| 必设环境变量 | `export PYTHONPATH=/tmp/pylibs_cvcontrib`（提供 contrib legacy KCF：`cv2.legacy.TrackerKCF`） |
| 数据集根 | `/mnt/data/siping/datasets/manu/anti-uav`（默认 `--dataset-root`） |
| host 侧（纯校验，无需数据集） | `python` 只需标准库 + pytest；`make_manifest_v2.py` / `canonical_digest.py` / `compare_rounds.py` 都是纯标准库 |

---

## 二、唯一合法的深轮命令

```bash
export PYTHONPATH=/tmp/pylibs_cvcontrib
cd /tmp/pycharm_project_<hash>          # 服务器上的工作副本根目录
python iqa_v2/run_eval_v2.py --conditions-profile deep
```

**路径是「剖面 ⇒ 路径」硬绑定的**，不靠操作者自觉传参（`run_eval_v2.py` 的 `PROFILE_PATHS`）：

| 剖面 | 产物目录 | 明细 cache | 报告正文 |
|:--|:--|:--|:--|
| `baseline`（默认） | `iqa_v2/output` | `/tmp/iqa_v2_cache` | `iqa_v2/REPORT_V2_FINAL.md` |
| **`deep`（本轮）** | **`iqa_v2/output_deep`** | **`/tmp/iqa_v2_cache_deep`** | **`iqa_v2/REPORT_V2_DEEP.md`** |

🔴 深轮**不得**写入基线轮的任何产物。守卫（`main()` 里 `parser.error`）覆盖
**冻结基线清单 `iqa_v2/output/manifest_v2.md5` 登记的全部产物**（15 项仓库内产物
＋ `/tmp/iqa_v2_cache/degrade_detail.csv`），而不是手写三条路径；
且守卫按**脚本自身位置**解析仓库根 ⇒ **换个 cwd 也绕不过去**。

### 🔴 报告正文**只有一个生成器**（`report_deep_v2.py`）

`run_eval_v2.py --phase report` 在 **deep 剖面下会自动改用 `report_deep_v2.py`**
（`phase_report_deep()`）—— 因为基线轮的 `phase_report()` 正文**与剖面无关**
（仍是「本轮唯一红线是 `target_local_contrast`」等**基线轮口径**），
拿它写深轮报告 ⇒ 报告被基线轮遗留覆盖（判定器 **D5** 当场变红），
而且它还注入一行**机器负载快照** ⇒ 报告非字节可复现 ⇒ md5 门禁换台机器必然假红。

⇒ 手册里那条唯一合法命令（`run_eval_v2.py --conditions-profile deep`）**可以**安全跑完整轮；
但如果只想重出报告正文，用 `report_deep_v2.py`（输出与上面那条命令**逐字节相同**）。

### 阶段（`--phase`）

`all`（默认，按序跑完）｜`dataset`｜`eval`｜`classify`｜`knee`｜`constraint`｜`blind`｜`regress`｜`tail`｜`report`

* `all` 一次跑完最省事；单独重跑某个阶段时**必须**带上 `--conditions-profile deep`，
  否则阶段会拿基线轴去读深轮明细 ⇒ 守卫直接 `parser.error`（这是有意的：
  「meta 说 deep、曲线其实来自 baseline 轴」这种记录与事实相反的情况必须报错退出）。
* 阶段开始前写 `run_meta.json`（`status: in_progress`），阶段成功后改写为 `status: completed`。
  `run_meta.json` 同时记「CLI 声明的档位」与「**从明细底表实测反推**的档位」，
  两者不一致 ⇒ `profile_matches_detail: false`；**没有证据可比时**该字段为 `null`（不是 `true`）。

---

## 三、开跑前的两道核对（别跳过）

1. **先跑一次 `--smoke`**，确认池规模真的是 895（规格书 §四「Pass B 池 895（不变）」）：
   ```bash
   python iqa_v2/run_eval_v2.py --conditions-profile deep --smoke
   ```
   明细行数应满足 `2442（clean）+ 895 × 25（退化档位）= 24817`。
   若池规模对不上，**先停下查池逻辑**，不要把错数据跑完。
2. **先把基线轮产物钉一份快照**（事后要证明「基线产物没被覆盖」）：
   ```bash
   md5sum iqa_v2/output/*.csv iqa_v2/output/*.json iqa_v2/*.md > /tmp/baseline_products_before.md5
   ```

---

## 四、交付门禁（C4 全清单，逐条都要跑）

```bash
# 门禁 1：机械门禁（pytest）
python -m pytest iqa_v2/test_iqa_v2.py iqa_v2/test_canonical_digest.py \
    iqa_v2/test_compare_rounds.py iqa_v2/test_recheck_curves.py \
    iqa_v2/test_deep_round_guard.py -q

# 门禁 2：独立复算（不 import 主聚合模块；深轮必须显式指到深轮产物）
python iqa_v2/verify_recalc_v2.py \
    --detail-csv /tmp/iqa_v2_cache_deep/degrade_detail.csv \
    --knee-csv   iqa_v2/output_deep/knee_thresholds_v2.csv \
    --class-csv  iqa_v2/output_deep/segment_class_v2.csv \
    --windows-csv iqa_v2/output_deep/windows.csv \
    --manifest-json iqa_v2/output_deep/dataset_manifest.json \
    --samples 120

# 门禁 3：产物清单（深轮清单 + 明细的顺序无关规范摘要）
#    🔴 第一步：**冻结深轮代码镜像**（清单登记代码条目的 md5 ⇒ 代码会继续演进，
#    不冻结的话「拿活代码核对历史清单」必然假失败）
mkdir -p iqa_v2/_deep_code_20261127 && cp -p iqa_v2/*.py iqa_v2/_deep_code_20261127/
python iqa_v2/make_manifest_v2.py \
    --products-subdir iqa_v2/output_deep \
    --bulk-detail /tmp/iqa_v2_cache_deep/degrade_detail.csv \
    --report iqa_v2/REPORT_V2_DEEP.md \
    --extra-report iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md \
    --extra-report iqa_v2/MEMORY_SECTION_DEEP.md \
    --out iqa_v2/output_deep/manifest_deep.md5
md5sum -c iqa_v2/output_deep/manifest_deep.md5
python iqa_v2/canonical_digest.py verify iqa_v2/output_deep/manifest_deep.bulk.canonical.txt

# 门禁 4：完成度审计
python iqa_v2/audit_final.py . iqa_v2/REPORT_V2_DEEP.md

# 门禁 5：跨轮红线对照（规格 B5「取更严者」）
python iqa_v2/compare_rounds.py \
    --baseline iqa_v2/output/constraint_lines_v2.csv \
    --deep     iqa_v2/output_deep/constraint_lines_v2.csv --strict

# 门禁 6：a-priori 判据判定器（A1~A4 / B1~B6 / C1~C4 / D1~D8，共 22 条）
#    C4 需要「外部门禁结果 JSON」⇒ 把门禁 1~5 的 rc 写进 JSON 再喂给它
python iqa_v2/acceptance_deep.py --gates-json iqa_v2/_gates_deep_host_20261127.json
```

### 4.1 **C4 的 `md5sum -c` 口径（本轮已定，取方案 a）** —— 两份清单，口径不同

| 清单 | 登记什么 | 作用 | 状态 |
|:--|:--|:--|:--|
| `iqa_v2/output_deep/manifest_deep.md5` | **深轮**代码 + 深轮产物 + 深轮报告正文 | **C4 的 `md5sum -c` 门禁查这一份** | host 侧 rc=0、**33/33** |
| `iqa_v2/output/manifest_v2.md5` | **基线轮**冻结清单（2026-11-26 18:47 冻结，自身 md5 `a72bc7fa…`） | **只保留证据链作用**，**不再**当门禁 | 裸 `md5sum -c` rc=1（红的是**代码条目**） |

* 基线清单裸 `md5sum -c` 红的**唯一原因**：它登记的是 **11-26 冻结时**的代码 md5，
  而 deep 轮**刻意演进**了这些代码 ⇒ 裸 `md5sum -c` 是拿**活代码**核对**历史清单**，
  任何一次后续改码都会变红 —— **与「基线产物被改」无关**：
  **产物条目逐字节未动**（规格书 C3 成立）。
* 基线代码的不可变性改由**冻结代码镜像** `iqa_v2/_baseline_code_20261126/` 证明
  （10/10 逐位相符；`test_output_manifest_md5_all_ok` 与 `acceptance_deep.py` 的 C3
  都走这条口径）。
* 🔴 **不要**用「重生成基线清单」把红色刷绿 —— 那会抹掉「基线代码自 11-26 未变」这条证据。

### 关于 `md5sum -c` 的诚实说明（**不要靠「重生成基线清单」糊过去**）

* **基线清单** `iqa_v2/output/manifest_v2.md5` 是**冻结证据链**，一个字节都不许改。
  深轮改了代码 ⇒ 清单里的**代码条目** md5 必然对不上（这是**预期且如实**的）；
  **产物条目全部未变** ⇒ 这才是「基线产物没被覆盖」的证据。
  代码条目的核对基准是冻结镜像 `iqa_v2/_baseline_code_20261126/`（pytest 走这条）。
  ⇒ 若交付材料写「`md5sum -c` 全过」，那必须同时写清**过的是产物条目**。
* **深轮清单** `iqa_v2/output_deep/manifest_deep.md5` 登记的是**深轮**代码 + 深轮产物。
* **大体量明细**（`degrade_detail.csv`）**不进 md5 清单**：明细行由多进程池以
  `imap_unordered` 汇总 ⇒ **同一份输入重跑一次行序就变** ⇒ 对它做裸 md5 必然假失败
  （把「行序抖动」误判成「数据变化」，证明不了任何东西）。
  它的复现性证明改用**顺序无关规范摘要**
  （`iqa_v2/canonical_digest.py`，旁挂清单 `*.bulk.canonical.txt`），由门禁 3 的
  `canonical_digest.py verify` 核对。裸 md5 口径仍可用 `--bulk-mode raw` 复现，
  但那只是历史形状，**重跑必假失败**。

---

## 五、报告落笔要求（C1 / C2）

* 报告必须**逐条**给出 **A1~A4** 与 **B1~B5** 的**实测结论**（通过 / 不通过 / 部分通过）。
* 🔴 **A1 的失败分支**（规格书 §二.1 明写）：若 deep 轴下**最重档跟住率仍 > 90%**，
  必须**如实写「本方法在净化底座上无法造出崩溃点」**，并**继续保留**
  「**弯折点 vs 崩溃点**」的措辞 ⇒ **不许**把「弯曲点」写成「崩溃点」。
  代码里已按此口径写注释（`degradation_4d.py`：「抬高上限**旨在**让曲线塌下来，
  是否成立由 A1 **实测**判定」）—— 报告**不得**反过来把这个注释当既成事实引用。
* 🔴 对外可发布版免责声明**只保留本轮真正成立的红线**（基线轮只有
  `target_local_contrast < 0.1891664` 一条），并带齐限定段。
* 🔴 **B5**：deep 线位比基线**更松**时，必须在报告里写明「**口径变化导致线位移动**」，
  并**取两者更严者**作为对外线。判定由 `compare_rounds.py` 机器给出
  （`verdict: DEEP_LOOSER` ⇒ `final_source: baseline` ⇒ 取基线线位），
  **不要手工口算**。
  （线位方向：`quality_metric < line_value` ⇒ 可豁免 ⇒ **数值越大越松** ⇒ 更严者 = `min`。）

---

## 六、已知风险（开跑前必须知道）

0. ✅ **【已实测更新 2026-11-27】本轮 deep 已经跑完并出结果**（24817 行明细，
   池 895、档位 10/7/8，与规格书 §四 的预期规模逐项吻合）。
   **实测结论：A1 不通过**（最重档跟住率 **87.9%**（XCorr blur）～**99.3%**（CSRT contrast），
   全部远高于 50% ⇒ **曲线没有塌**，膝点只是「弯折点」不是「崩溃点」）；
   **A3 通过**（`target_lap_var` 首次拿到有效红线 **220.61511928541879**，xcorr/blur）；
   `target_local_contrast` 收紧到 **0.10056221586659723**（kcf_legacy/blur）；
   `local_scr` 仍无有效红线。逐条判定见 `iqa_v2/acceptance_deep.py`（机器判定，不采信报告数字）
   与 `iqa_v2/REPORT_V2_DEEP.md` §八。
   ⇒ **本手册第二~四节是「下一轮重跑」用的操作流程**；本轮的实测结论以报告与验收记录为准。
1. 🔴 **A1 不成立**（机理证据来源：`iqa_v2/_deep_axis_sanity_20261127.md`，
   **合成图独立验证，零真实数据集**，只作机理参考、不作结论）：
   deep 档位把 `target_lap_var` 打到 clean 的 0.43%~9.49%（blur/contrast 两族**已经到底**），
   但 **motion 族 L=15 仍塌不下去**（小/中目标剩 20.7%~36.0%）⇒ 预计跟住率仍 > 90%。
   更要紧的是：用**冻结的基线轮产物**做交叉验证发现，基线最重档的 `target_lap_var`
   **早已塌到 clean 的 1.33% / 15.05% / 2.32%**（blur / motion / contrast），
   而 xcorr 跟住率仍停在 **90.84% / 92.74% / 98.88%**
   ⇒ **塌不下去的是跟住率，不是指标轴**；把上限从 σ=4 抬到 σ=8 对**指标轴几乎没有增量**。
   ⇒ **这轮很可能是「白跑」**：跑它的价值在于拿到 A1 的**实测**结论与机理证据，
   而不是拿到新红线。**是否值得花服务器算力，由用户拍板。**
2. `contrast` 族新增的 α=0.10 / 0.05 会**打破 `local_scr` 的逐档单调性**
   （机理定位为 `apply_degradation` 的 uint8 量化，不是 SCR 定义问题）。
   这与基线轮报告 §「`contrast` 族对 `local_scr` 不承载信息」的既有结论**同源**
   ⇒ 该组合在深轮同样会被曲线有效性守卫判作废，**属预期，不是新缺陷**。
3. `recheck_curves.py`（产出 W6/W7 的 A/B/C 曲线有效性证据）原先把档位阶梯写死成
   基线 6/5/6 ⇒ 深轮下会**只覆盖浅半段轴**。现已改为**从明细底表实测反推**档位，
   且膝点表路径可配（深轮要指 `iqa_v2/output_deep/`），否则会拿深轮明细配基线膝点表。
4. `verify_recalc_v2.py` / `audit_final.py` 的档位轴**从明细 CSV 读** ⇒ 自动适配 26 条件；
   但 `audit_final.py` 的**兜底阶梯**必须诚实（明细缺失 ⇒ 显式 FAIL/WARN，
   **不许**静默回落成基线 6/5/6 后「空过」）。

---

## 七、冻结物（一个字都不许动）

| 路径 | 是什么 |
|:--|:--|
| `iqa_v2/output/manifest_v2.md5` | **冻结的基线轮产物清单**（证据链的锚） |
| `iqa_v2/output/*.csv`, `*.json` | 冻结的基线轮产物 |
| `iqa_v2/REPORT_V2_FINAL.md` | 冻结的基线轮报告正文 |
| `iqa_v2/_baseline_code_20261126/` | **冻结的基线轮代码镜像**（基线清单代码条目的唯一核对依据；含 `README.md` 说明来历） |
| `iqa_v2/_deep_code_20261127/` | **冻结的深轮代码镜像**（深轮清单代码条目的唯一核对依据；含 `README.md` 说明来历） |
| `iqa_v2/output_deep/manifest_deep.server_20261127.md5` | **服务器侧生成**的那一版深轮清单（md5 `89e1cc75…`，服务器侧 `md5sum -c` rc=0）—— 与 host 重生成版**内容不同**，是**服务器侧证据**，刻意保留 |
| `iqa_v2/server_out_final/` | 服务器回收的原件（18 个文件 + 产物目录） |
| `iqa_v2/DEEP_AXIS_SPEC_20261127.md` | **a-priori 规格书**（判据来源，不改） |
