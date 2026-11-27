
---

### [assistant #1]
I'll start by reading the handover document and task specification.

---

### [assistant #2]
The handover output was truncated. Let me read the full middle section with the gap details.

---

### [assistant #3]
Now let me check the environment and current code state.

---

### [assistant #4]
Let me spawn parallel explorers while I read the core pipeline code myself.

---

### [assistant #5]
Architecture mapped. Let me check the real video assets while the other agents finish.

---

### [assistant #6]
Let me check server resources (per project iron rule) and inspect the real bad clip frames.

---

### [assistant #7]
All three explorations are done. Key findings: G7 confirmed **annotations NOT recovered** (36 human-marked frames / 0.02%) → keep XCorr+KCF. Let me plan the implementation.

---

### [assistant #8]
Now I have a complete picture. Let me record the plan and start implementing.

---

### [assistant #9]
Starting with **G1** — removing the JPEG degradation axis.

---

### [assistant #10]
Now let me create the new module owning the three-way regression system (G3+G4).

---

### [assistant #11]
Now the main pipeline rewrite. Let me patch `run_eval_v2.py` step by step.

---

### [assistant #12]
Now the knee phase with purified pools, plus the new constraint-line phase.

---

### [assistant #13]
Now adding the regression + blind-test phases to the new module.

---

### [assistant #14]
Now the report generator rewrite — the main deliverable.

---

### [assistant #15]
Let me run a smoke test to validate the new pipeline end-to-end.

---

### [assistant #16]
Let me fix the ROI search so it can't land on frame-edge artifacts (which inflated the bad clips' sharpness).

---

### [assistant #17]
Now updating the test suite (G9) and the independent verifier.

---

### [assistant #18]
Now the test suite updates (G9).

---

### [assistant #19]
Let me record the resource readings and launch the full pipeline in the background.

---

### [assistant #20]
Full pipeline is running. Let me monitor while it progresses.

---

### [assistant #21]
Full run complete. Let me inspect the results.

---

### [assistant #22]
Let me inspect the actual degradation curves to sanity-check the knee extraction.

---

### [assistant #23]
The knee analysis reveals an important finding. Let me verify the pool sizes and refine the constraint-line rule to the two primary arms (CSRT is a control arm, not a red-line source).

---

### [assistant #24]
Let me check the previous round's constraint-line caliber to keep it unchanged as the handover requires.

---

### [assistant #25]
The previous round's caliber is **per-arm lines** (XCorr/KCF separately) with the disclaimer taking the conservative one — my implementation matches. Now let me upgrade the report with the key findings.

---

### [assistant #26]
# HANDOVER · 红外图像质量评估 v2 收口轮（续接用）

## 0. 任务与硬约束

- 工作目录：`/tmp/pycharm_project_a7d966e6`（本机即服务器 `test-Super-Server`，`huangzhe@192.168.99.40:32222` 镜像）。
- **只改 `iqa_v2/`**；`/mnt/data/siping/runs/**` 与数据集目录**只读**。
- 禁止训练 / 下载数据集 / 模型重训；禁止 `pkill -f`（用 `kill <pid>`）；不 git commit/push；**不改 `manu/memory/memory.md`**。
- Python：`/home/huangzhe/anaconda3/envs/uav/bin/python`（3.10.20 / scipy 1.15.3 / cv2 4.10.0）；KCF 需 `PYTHONPATH=/tmp/pylibs_cvcontrib`。
- 运行需 `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`。
- 每完成一个缺口，在 `iqa_v2/PROGRESS_SERVER.md` 追加一行（时间/缺口/做了什么/产物路径/结论数字）。
- 资源铁律读数已记录在 `PROGRESS_SERVER.md`（可用内存 83.3%、load 4.65/80 核、GPU 空闲）。
- 用户要求：中途不要停、不提问、18:00 前独立做完。

## 1. 已完成（必读的两份文件已读：HANDOVER §3 的 G1~G9、GOAL_OBJECTIVE 全文）

### 代码改动（均已落盘）

| 文件 | 改动 |
|:--|:--|
| `iqa_v2/degradation_4d.py` | **G1**：删 `JPEG_QUALITIES` / jpeg 分支 / `FAMILIES=("blur","motion","contrast")`；条件数 24→18；docstring 写明「JPEG 剔除」与「白噪声轴从未存在」的理由 |
| `iqa_v2/run_eval_v2.py` | 两趟法 `run_eval`（Pass A clean → 分类+切分 → Pass B 只对标定好片段做退化）；三臂 tracker 新增 `trace`/`complete`，明细新增 `*_cle_end`/`*_maxcle`/`*_complete` 列；`phase_knee` 改用 `own_good`/`common_good` 曲线集 + 留出集泄露机器校验 + `clean_success_rate` 列；**新增 `phase_constraint`**；`phase_report` 整体重写为 `REPORT_V2_FINAL.md`（12 章）；`main()` 新增 `--phase classify\|constraint\|blind\|regress`，默认报告改为 `iqa_v2/REPORT_V2_FINAL.md` |
| `iqa_v2/regression_v2.py`（**新建**） | G3/G4 唯一 owner：`GOOD_CLE_PX=3.0`、`classify_arm`/`classify_dataset`、`split_sequences`（按 seq_id、种子 20261126、70/30）、`verify_split_disjoint`、`detect_saliency_roi`（DoG 自动 ROI，**搜索内缩 4%/≥16px 避开边缘伪影**）、`evaluate_real_clip`（复用 v1 `manu/video_quality/vq_metrics.detect_content_box` 裁 pillarbox + 最后 5 秒故障窗）、`phase_classify`/`phase_blind`/`phase_regress` |
| `iqa_v2/test_iqa_v2.py` | 删 JPEG 行为用例 → 改「JPEG/白噪声轴必须不存在」；`test_conditions_cover_three_degradation_families`；新增 3px 分类、序列切分零交叉、pillarbox 裁边、底座纯净度、约束线只含三核心指标、回归五集合覆盖等用例 |
| `iqa_v2/verify_recalc_v2.py` | 删 jpeg 分支；`verify_knee` 改为按 `own_good`/`common_good` + 新增 `--class-csv` 复现同一子集 |
| `iqa_v2/make_manifest_v2.py` | 清单纳入 `REPORT_V2_FINAL.md` / `REPORT_V2_ANTIUAV.md` / `MEMORY_SECTION.md` |
| `iqa_v2/REPORT_V2_ANTIUAV.md` | 顶部加「已被 REPORT_V2_FINAL.md 取代」横幅 |
| `iqa_v2/PROGRESS_SERVER.md` | **新建**，含资源铁律读数 + G1 行（**G2~G9 行待补**） |

### 全量流水线已跑完（`nohup` 后台，日志 `/tmp/iqa_v2_logs/full_run.log`，约 6 分钟）

- Pass A 2442 窗口 → 三分法：**好 1275 / 一般 774 / 跟丢 393**；序列切分 **标定 137 / 留出 59（零交叉 ✅）**；Pass B 池 **1259 窗口 × 17 条件**；明细 23845 行 → `/tmp/iqa_v2_cache/degrade_detail.csv`。
- 产物：`output/{windows.csv,dataset_manifest.json,segment_class_v2.csv,split_v2.csv,knee_thresholds_v2.csv,correlations_v2.csv,constraint_lines_v2.csv,regression_v2.csv,blind_real_v2.csv,tail_analysis_v2.csv}` + `REPORT_V2_FINAL.md`（425 行，已生成但**需复核后再定稿**）。

### 关键结论数字（已写入报告）

- 约束线（**只由 XCorr+KCF 两条主臂取 min(每族膝点)，CSRT 仅作对照不取值**）：
  - `target_local_contrast` = **0.101426**（contrast / xcorr；kcf 0.14602；两臂差 30.54%）
  - `target_lap_var` = **505.6238**（contrast / xcorr；kcf 1110.81；两臂差 54.48%）
  - `local_scr` = **4.485792**（blur / kcf_legacy；xcorr 5.82855；两臂差 23.04%）
- 回归（`output/regression_v2.csv`）：
  - `good_holdout` 误杀率：lap_var **0.79%**、local_contrast **4.74%**、local_scr **18.95%**
  - `lost` 拦截率：lap_var 12.47%、local_contrast 24.43%、local_scr 64.12%
  - `real_bad`（2 条真实坏片，248 帧）：lap_var **100%**、local_contrast **100%**、local_scr **99.19%**
  - `real_good`（2 条对照，300 帧）：lap_var **0%**、local_contrast 2.33%、local_scr **98%**
- **本轮最重要的工程事实（已写入报告 §4.1/§12.6）**：净化底座上三族退化**不足以让跟踪崩溃**（最重档跟住率仍 ≥91%，blur XCorr 91.0%），膝点只落在「曲线开始弯折」处（膝点处跟住率 95%~100%），故双算法门禁常判不可靠、跨臂差变大；红线取最严候选，用途靠独立盲测背书。
- G7 核查结论（子代理穷举）：**58 条业务序列标注未回收** —— `manu/point_annotator/` 只有工具无数据；`manu/memory/memory.md` §315 记录的是「三份并行交付」、§316 明写「待回收」；母包 `annotations/` 仅 2 个试跑草稿 JSON，人工标注帧 36 帧 ≈0.02%。⇒ **保持 XCorr+KCF**，报告写「正式交付模型替换待标注回收后执行」，**未启动任何训练/重训**。

## 2. 尚未完成（下一步，按顺序）

1. **复核 `REPORT_V2_FINAL.md`**：重点看 §4 的「47.8% 窗口在干净帧上就跟丢」一句——该数字现由 `lost+mediocre` 算出（774+393=1167/2442=47.8%），**与交棒书旧口径「15.5% 已跟丢」不一致**，需改写为「上一版 15.5% 的窗口在干净帧上就跟丢；本轮三分法把这类再细分后，一般+跟丢合计 47.8%」之类不误导的表述。检查 §5.1 表中 `nan` 单元格是否可接受（当前直接打印 `nan`）。
2. **跑 G9 三道门禁**（必须 100% 通过）：
   - `PYTHONPATH=/tmp/pylibs_cvcontrib python -m pytest iqa_v2/test_iqa_v2.py -v`
   - `PYTHONPATH=/tmp/pylibs_cvcontrib python iqa_v2/verify_recalc_v2.py --samples 120`（误差 <1e-6）
   - `python iqa_v2/make_manifest_v2.py` 后 `md5sum -c iqa_v2/output/manifest_v2.md5`
   - 注意 `make_manifest_v2.py` 必须**最后**跑（它对代码/CSV/报告做 md5）；`MEMORY_SECTION.md` 建好后也要重跑。
3. **写 `iqa_v2/MEMORY_SECTION.md`**（可并入 `manu/memory/memory.md` 的 **§318** 成稿章节）：含 §316 口径声明、§317 收口（口径声明/结论表/适用边界）、§318 可执行流程；必须写明「正式上板红线待 58 条 GT 回收后锁定」「正式交付模型替换待标注回收后执行」。数字与 `REPORT_V2_FINAL.md` 一致。
4. **补 `PROGRESS_SERVER.md` 的 G2~G9 行**（每缺口一行：时间/缺口/做了什么/产物路径/结论数字）。
5. 可选清理：`iqa_v2/output_smoke/`、`iqa_v2/REPORT_SMOKE_V2.md`、`/tmp/iqa_v2_cache_smoke`（本轮冒烟产物，非交付物；删除前确认不影响门禁）。

## 3. 注意事项 / 易踩坑

- **`regression_v2.py` 依赖 `run_eval_v2.py`**（module-level import 反向）；`run_eval_v2` 对 `regression_v2` 用**函数内惰性 import**（既有风格，勿改成 module-level，会循环导入）。
- `phase_knee` 会对「留出集出现非 clean 行」直接 `raise SystemExit`（泄露机器校验）。
- `verify_knee` 依赖新 `segment_class_v2.csv` 复现同一子集，缺文件会失败。
- 报告所有数字必须**现场读 CSV 生成**，禁止手抄（`phase_report` 已如此实现）。
- 不要改 `manu/memory/memory.md`；`manu/video_quality/vq_metrics.py` 只读复用（`detect_content_box`、`BLACK_THRESH=16`）。
- 真实坏片路径：`/mnt/data/xxxu/纵横挂飞/120BT-04-红外02.mp4`、`130CT-02红外_IRT.mp4`（1920×1080，pillarbox 320/320，裁后 1280×1030）；好片对照 `/mnt/data/xxxu/朗基天香20260903/VIDEO00007_*.h264`、`VIDEO00008_*.h264`（640×512，无黑边，**画面里没有无人机目标**，DoG 峰值落在地面景物）。
- 真实片解码会打印 ffmpeg `Invalid NAL unit size` 告警（.h264 裸流），**属正常**，帧数已核对（150/150、123/125）。
- 旧报告 `REPORT_V2_ANTIUAV.md` §3.1/§3.4 留有上一轮数值（819.55 来自 blur 两臂一致；local_scr XCorr 3.20777/KCF 2.69153 来自 contrast/motion；jpeg 候选 local_scr 4.88624、lap_var 4182.89），报告 §10 的「三条原因」已引用它们做对照，**改报告时别删这段**。

---

### [assistant #27]
我先接续上一模型的工作，读取交接文件与当前状态。

---

### [assistant #28]
收到 host 侧 18:05 的新一轮复核（W7_HOST_RECHECK），它推翻了两条当前红线。先独立复算核验。

---

### [assistant #29]
Host 复核已在服务器上复现属实。现在看代码结构，准备落地守卫。

---

### [assistant #30]
现在动代码：给膝点加"曲线有效性守卫"，并重写约束线选取。先看报告生成的相关段落。

---

### [assistant #31]
Now实施核心改动：给膝点反解加"曲线有效性守卫"。

---

### [assistant #32]
资源读数：可用内存 87608/128545 MiB（68%）、load 25.36/80 核（占用来自**另一个** etf_rotation 任务，非本流水线）、GPU 全空闲。本轮只跑轻量单进程阶段。

---

### [assistant #33]
守卫生效：只有 XCorr 的 blur/motion 曲线合格。现在并行推进：派子代理改复算器与测试，我继续改报告层。

---

### [assistant #34]
Now改约束线表：加入"退化曲线参照线"列，说明红线为什么只剩 XCorr。

---

### [assistant #35]
约束线已重算。现在重写报告层（§4/§5/§6/§8.3/§9/§10/§12）。先读现有生成代码。

---

### [assistant #36]
Now插入"实际仿真池"行与曲线有效性分级表（§4.2）。

---

### [assistant #37]
Now加 §4.2「曲线有效性分级」——这是新红线只出自 XCorr 的依据。

---

### [assistant #38]
</tool_call>

---

### [assistant #39]
测试子代理已回：5 条新用例通过，但暴露出 2 个既有失败。先看分类逻辑这个真实缺陷。

---

### [assistant #40]
需要先量化分类口径的实际影响，再决定根因修法。

---

### [assistant #41]
门禁：35 通过，仅剩 manifest 过期（最后统一重生成）。现在通读新报告核查。

---

### [assistant #42]
Now补 PROGRESS_SERVER.md 的 G2~G9 留痕与 W6/W7 复核处理结论。

---

### [assistant #43]
W8/W10 又追加了要求。先看 W10 文件是否已投递。

---

### [assistant #44]
# HANDOVER · 红外图像质量评估 v2 收口轮（第二轮续接用）

## 0. 环境与硬约束

- 工作目录：`/tmp/pycharm_project_a7d966e6`（本机即服务器 `192.168.99.40` 镜像）。**只改 `iqa_v2/`**；`/mnt/data/siping/runs/**`、数据集目录**只读**。
- 禁止训练 / 下载数据集 / 模型重训；禁止 `pkill -f`；不 git commit/push；**不改 `manu/memory/memory.md`**（成稿写 `iqa_v2/MEMORY_SECTION.md`）。
- Python：`/home/huangzhe/anaconda3/envs/uav/bin/python`；KCF 需 `PYTHONPATH=/tmp/pylibs_cvcontrib`；运行需 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`。
- 服务器时间约 **17:59**（18:00 是 host 工作站关机，不是服务器）。资源：可用内存 87608/128545 MiB（68%）、load 25/80 核（占用来自**另一任务** `etf_rotation/qlib_scores.py`，非本流水线）、GPU 空闲。
- 每完成一个缺口要在 `iqa_v2/PROGRESS_SERVER.md` 追加一行（时间/缺口/做了什么/产物路径/结论数字）。
- host 已投递多轮复核：`iqa_v2/_host_warning_W6.md`、`W7`、`W8`、`W9`、`W10`、`W7_HOST_RECHECK.md`（**W8/W9/W10 是我这一轮尚未逐条处理的**）。

## 1. 已完成（全部落盘）

### 代码改动
| 文件 | 改动 |
|:--|:--|
| `iqa_v2/run_eval_v2.py` | **新增曲线有效性守卫**：`CURVE_MIN_SPAN=0.05`；`level_success_profile` / `level_metric_profile`（退化强度递增用 `degradation_strength`）；`extract_knee(rows, metric, tracker, family)` 在「非单调 / span≤0.05pp / 指标非单调」时 `method_used="degenerate"`、**`knee_reliable=0`**、`knee_x_final=nan`、`x_at_y50=nan`；新增列 `curve_y_span` / `curve_monotone` / `metric_monotone` / `knee_x_candidate`（守卫前候选读数，仅参照）；`curve_y50` 改为真·跌破 0.5 插值（从不跌破⇒nan）；`phase_constraint` 四重过滤；报告 §4/§4.2/§5/§6/§7/§8.2/§8.3/§9.2/§10/§12 全面重写（12 章） |
| `iqa_v2/regression_v2.py` | `classify_arm(..., success=True)`：**好片段必须干净帧真跟住**（任务书「全程稳定跟住」）⇒ 底座纯净度 100%；膝点邻域改为膝点×[1/1.5,1.5]（`KNEE_BAND_FACTOR`） |
| `iqa_v2/verify_recalc_v2.py` | 子代理新增**独立实现**的 `verify_curve_validity`（守卫判定双向对账、span 相对误差 <1e-6、单调列一致、degenerate 终值 nan），`--samples 120` **EXIT=0** |
| `iqa_v2/test_iqa_v2.py` | 子代理新增 5 条用例（degenerate 被拒 / 约束线只来自有效曲线 / 单调曲线正控 / 平坦曲线判废 / contrast-local_scr 非单调），**未删改既有用例** |

### 关键结论数字（现场读 CSV）
- **约束线（只三项核心指标）**：`target_local_contrast` **0.189166**（motion/xcorr）、`target_lap_var` **541.0926**（motion/xcorr）、`local_scr` **5.828548**（blur/xcorr）。**三条线全部只由 XCorr 支撑**（KCF/CSRT 曲线未达 A 级）⇒ **算法无关性本轮不成立**。
- **曲线分级（common_good）**：A 级仅 XCorr/blur 7.78pp、XCorr/motion 6.41pp；KCF/blur 3.66pp(B)；其余 C 级。
- **三分法**：好 1275 / 一般 773 / 跟丢 394；切分 标定 137 / 留出 59，**seq_id 交叉 = 0**。
- **回归**：good_holdout(380) 误杀 lap_var **1.32%** / contrast 12.63% / scr 25.79%；real_bad(248) 拦截 100% / 100% / 99.60%；real_good(300) 误杀 **0.00%** / 29.33% / **98.00%** ⇒ 区分间隔 lap_var **100pp** ✅、contrast **70.67pp** ✅、**scr 1.60pp ❌（local_scr 不成立，不得进免责声明）**。
- **免责声明**：§10.1 任务书原文（819.55/2.69）→ §10.2 换本轮值逐字版（541.09 / 5.83 / 0.1892，注明 local_scr 不生效）→ §10.3 **对外可发布版**（只留 `target_lap_var` + `target_local_contrast`）。
- **G7**：58 条业务标注**未回收**（人工标注 36 帧≈0.02%）⇒ 保持 XCorr+KCF，报告写「正式交付模型替换待标注回收后执行」。

### 已产出文件
- `iqa_v2/REPORT_V2_FINAL.md`（497 行，12 章，数字全从 CSV 生成）
- `iqa_v2/MEMORY_SECTION.md`（180 行，含 §316 口径声明 / §317 收口 / §318 可执行流程）
- `iqa_v2/PROGRESS_SERVER.md`：已补 **G1~G8** 行（G9 行待补）
- `iqa_v2/output/`：`windows.csv`、`dataset_manifest.json`、`segment_class_v2.csv`、`split_v2.csv`、`knee_thresholds_v2.csv`、`correlations_v2.csv`、`constraint_lines_v2.csv`、`regression_v2.csv`、`blind_real_v2.csv`、`tail_analysis_v2.csv`
- ⚠️ **`output/manifest_v2.md5` 仍是 12:35 的旧清单**（pytest 只因此 1 条失败：35 passed / 1 failed）

## 2. 待完成（按优先级）

1. **修 `extract_knee` 的提前返回标签**（复算器已报）：`run_eval_v2.py` 中 `if xs.size < 4: return result` 在打 `degenerate` 前返回 ⇒ 10 行 `surround_edge_density`（<4 分箱）守卫触发却 `method_used=""`。修法：`result["method_used"] = "degenerate" if degenerate else ""`（degenerate 判定已在提前返回前算出）→ **重跑 `--phase knee`**（并重跑 constraint/regress/report）。
2. **W8-7**：`blind_real_v2.csv` 的 `n_frames_total` 是哨兵值 `-230584300921369` ⇒ 改成真实总帧数或空值（在 `regression_v2.phase_blind`），重跑 `--phase blind` + regress + report。`__pycache__` 不在清单（manifest 只收 `iqa_v2/*.py` 与 `output/*.csv|json`）✓。
3. **W8-2**：§10 补「**逐族逐臂对照表**」+ 分族归因，**不要**写统一「一律上移/下移」。旧值来源：host W8-2 表（旧：`blur/local_scr` 4.7773/4.7773、`blur/lap_var` 819.55/819.55、`motion/local_scr` 5.6307/2.6915、`motion/contrast` 0.20481/0.20481、`motion/lap_var` 1754.0/2320.9、`contrast/lap_var` 1090.9/1651.9、`contrast/local_scr` KCF 3.2078；**缺** `blur/target_local_contrast`、`contrast/local_scr` 的 XCorr 值、`contrast/target_local_contrast` ⇒ 可查 `iqa_v2/REPORT_V2_ANTIUAV.md` 或标「—」）。新值现场读 `output/knee_thresholds_v2.csv`。
4. **W8-3 / W10-1**：§8.3 与 §12 各补一句「真实好片对照 2 条**无无人机目标**（停车场航拍，DoG 落在车顶/地面）⇒ 不能当目标级误杀率证据；目标级误杀率只有 §8.1 Anti-UAV 留出集」；报告需给「三集合 × 三指标」误杀率表（**已有**区分间隔表，可再确认覆盖）。
5. **W10-2**：§12 明确「**本轮算法无关性不成立**」并写清 KCF 为何无线位：**曲线塌陷/未达 A 级（不是被分类口径清空）** ⇒ 当前 §12 局限 8 已写，需再核一句归因准确。
6. **W10-2（相关性口径）**：37 个 `knee_reliable=1` 中 30 个在相关性表 `usable=0`/`direction_ok=0`（`target_lap_var` 全部 `usable=0`）。**要么**采纳时加 `usable=1` 过滤，**要么**报告逐条说明为何弱相关仍采纳（§9.2 已加解释段：二值标签基线 90% + 阈值型物理关系 ⇒ 秩相关天然压缩；证据链=守卫+膝点+盲测）。**保持两条口径一致**，在 §12 再加一条局限。
7. **W10-4**：`MEMORY_SECTION.md` 标题**必须**以 `## §318 · …` 开头（host 合并脚本靠 `§` 编号入索引）⇒ 我写的文件目前是 `# MEMORY_SECTION · …` + `## §316/§317/§318` 三节，**需把首行改成 `## §318 · …`** 或加一行 `## §318` 标题。
8. **G9 门禁**（全部跑完后再收尾）：
   ```bash
   cd /tmp/pycharm_project_a7d966e6
   export PYTHONPATH=/tmp/pylibs_cvcontrib OMP_NUM_THREADS=4
   python iqa_v2/run_eval_v2.py --phase knee && --phase constraint && --phase regress && --phase report
   python iqa_v2/verify_recalc_v2.py --samples 120          # 期望 EXIT=0
   python -m pytest iqa_v2/test_iqa_v2.py -v                 # 期望 100% 通过
   python iqa_v2/make_manifest_v2.py                        # 最后跑
   md5sum -c iqa_v2/output/manifest_v2.md5                  # 全过
   python iqa_v2/audit_final.py .                           # host 投递的新审计器，要求 FAIL=0
   ```
9. **PROGRESS_SERVER.md 追加两行**：① W6~W10 逐条处理结论（G2~G9 表已补，**G9 行待补**）② W8-8 验收清单 ①~⑦ 逐条回。

## 3. 关键约束提醒

- 报告所有数字**必须现场读 `output/*.csv`**（`phase_report` 已如此实现）⇒ 改报告文字要改 `run_eval_v2.py` 的 `phase_report` 再重生成，**不要手改 `.md`**。
- `regression_v2.py` 与 `run_eval_v2.py` **循环依赖**：regression 在函数内惰性 import run_eval（勿改成模块级）。
- 分类口径改动后**切分仍是 137/59** ⇒ 明细 CSV `/tmp/iqa_v2_cache/degrade_detail.csv`（23845 行）仍有效，**无需重跑 eval**。
- 子代理已完成并可复用：`test_iqa_v2.py`（md5 a1407d0f…）、`verify_recalc_v2.py`（md5 575f50b4…）；其结论均已自行复验。

---

### [assistant #45]
Now I have the full picture. Let me parallelize: delegate the blind-phase CSV fix and an independent audit while I own the main report/code changes.

---

### [assistant #46]
Fixing the `extract_knee` early-return label bug first.

---

### [assistant #47]
Now making the code fixes. First the `extract_knee` early-return label:

---

### [assistant #48]
Now the main report patch. Writing it as a careful scripted edit:

---

### [assistant #49]
Now let me run the knee + constraint phases with the fixed guard.

---

### [assistant #50]
Now restructuring `MEMORY_SECTION.md` for the host merge script (§-indexed single chapter) and folding in the new findings.

---

### [assistant #51]
The audit found real issues — let me examine how it locates the report and recomputes curves.

---

### [assistant #52]
Found the root cause of the audit failures. Let me verify a candidate fix against the existing detail data before committing.

---

### [assistant #53]
Found it: the audit's `binned_curve` has a real bug (`success.mean()` instead of `success[sel].mean()`). Let me verify determinism before changing the pool.

---

### [assistant #54]
The reviewer confirmed my fixes and found more gaps. Now let me check the pipeline's phase flow and the blind-fix agent before changing the pool.

---

### [assistant #55]
Let me check server resources (iron rule) and prepare the pool + audit patches.

---

### [assistant #56]
The blind-fix agent finished. Let me verify its change, then apply the pool fix and launch the full re-run.

---

### [assistant #57]
Launching the full pipeline re-run in the background (pool = 三分法好片段):

---

### [assistant #58]
While the pipeline runs, let me fix the remaining report gaps the reviewer found.

---

### [assistant #59]
Now adding the two tests for the changes I made (pool definition + knee label fix):

---

### [assistant #60]
Adding the clause→test-name evidence table required by W10 §三 to the report generator:

---

### [assistant #61]
Pipeline complete. Let me check the new numbers and regenerate the report with my latest edits.

---

### [assistant #62]
Two FAILs left. Let me look at exactly what the audit requires for each.

---

### [assistant #63]
The audit caught a real defect: the guard uses the level-profile span while the knee comes from a metric-binned curve. Let me analyze the alternatives.

---

### [assistant #64]
# HANDOVER · 红外图像质量评估 v2 收口轮（续接用）

## 0. 环境与硬约束

- 工作目录：`/tmp/pycharm_project_a7d966e6`（本机即服务器 192.168.99.40 镜像）。**只改 `iqa_v2/`**；`/mnt/data/siping/runs/**`、数据集目录**只读**。
- Python：`/home/huangzhe/anaconda3/envs/uav/bin/python`；KCF 需 `PYTHONPATH=/tmp/pylibs_cvcontrib`；运行需 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`。
- 禁止训练/下载/重训；禁止 `pkill -f`（用 `kill <pid>`）；不 git commit/push；**不改 `manu/memory/memory.md`**（成稿写 `iqa_v2/MEMORY_SECTION.md`）。
- 服务器时间约 **18:35**；资源充足（可用内存 106493/128545 MiB、80 核、load 低、GPU 空闲）——已按铁律读过 `free/nproc/uptime/iostat/nvidia-smi/ps -eLf`。
- 每完成一个缺口要在 `iqa_v2/PROGRESS_SERVER.md` 追加一行（时间/缺口/做了什么/产物路径/结论数字）。

## 1. 本轮已完成并落盘

### 代码改动（`iqa_v2/run_eval_v2.py`）
1. `extract_knee` 提前返回标签修复：`result["method_used"] = "degenerate" if degenerate else ""`（method_used 空值 22→7）。
2. §10 新增「**逐族逐臂对照表**」（旧值=冻结常量 `PRIOR_ROUND_KNEES`，新值现场读 CSV）+ 按族方向汇总 + 归因①②③④（底座换血 / 决定族决定臂变了 / 守卫作废 / local_scr 不成立）。
3. §8.3 新增「证据边界」块（真实好片 2 条=**无目标停车场航拍** ⇒ 不能当目标级误杀率证据）。
4. §9.2 改为**逐指标**口径（不再写「真实好片误杀 0%」这种一刀切）。
5. §12 新增 8b（算法无关性不成立 + KCF 无线位归因=y_span 守卫非样本清空）、11（真实好片对照边界）、12（采纳膝点 × 相关性表逐条清单 + 分工说明）。
6. §11 补执行机/解释器绝对路径 + `output/degrade_detail_v2.csv` 路径 + **条款→pytest 用例对照表**（W10 §三）。
7. **仿真池定义改为「三分法好片段 ∩ 标定集」**（原为三臂好片段并集）：`run_eval_v2.py` 中 `pool_keys` 判定改为 `row["class_ds"] == CLASS_GOOD`；eval 末尾把明细镜像到 `output/degrade_detail_v2.csv`。
8. `audit_final.py` 修复**仪器自身 bug**：`ys.append(float(success.mean()))` → `success[sel].mean()`（原来每箱都填全池均值 ⇒ y_span 恒 0 ⇒ R5g/R5h 失真）。

### 其他文件
- `iqa_v2/regression_v2.py`：子代理修复 `n_frames_total` 哨兵（裸流 .h264 用顺序解码真实帧数；mp4 截断修正）。实测：120BT=35848、130CT=44743、VIDEO00007=1604、VIDEO00008=4896（ffprobe 逐帧计数交叉验证一致；120BT 是截断件）。
- `iqa_v2/test_iqa_v2.py`：新增 2 条用例 `test_simulation_pool_is_dataset_level_good_segments`、`test_knee_below_four_bins_still_carries_guard_label`（后者已实跑通过）。
- `iqa_v2/MEMORY_SECTION.md`：首行改为唯一 `## §318 · …`（host 合并脚本要求），三节降级为 `###`；补逐族逐臂表、算法无关性不成立、相关性口径、适用边界新增 2 行、产物路径更新。

### 全量流水线已重跑（`--phase all --workers 24`，nohup 日志 `/tmp/iqa_v2_full_rerun.log`，PID 2559323，已完成）
- 明细 17657 行（2442 clean + 895×17）；池 = **895 窗口**（三分法好片段 ∩ 标定集），池内 clean 起点 **三臂均 100%**。
- **约束线与换池前完全一致**：`target_local_contrast` **0.189166**（motion/xcorr）、`target_lap_var` **541.0926**（motion/xcorr）、`local_scr` **5.828548**（blur/xcorr）；KCF/CSRT 无线位。
- 回归结果不变：good_holdout 误杀 12.63% / 1.32% / 25.79%；real_bad 拦截 100% / 100% / 99.60%；real_good 误杀 29.33% / 0.00% / 98.00%；区分间隔 70.67 / 100.00 / **1.60 pp**。
- 三分法：好 1275 / 一般 773 / 跟丢 394；切分 标定 137 / 留出 59，seq_id 交叉 = 0。

### 门禁现状
- `verify_recalc_v2.py --samples 120` → **EXIT=0**（ROI 1.388e-17、膝点 0.000e+00、曲线守卫 162/162 一致）。
- `pytest -q` → 35 passed / 1 failed（仅 `test_output_manifest_md5_all_ok`，清单过期）。
- `audit_final.py iqa_v2/output iqa_v2/REPORT_V2_FINAL.md` → **PASS 32 / FAIL 2 / WARN 3**（比修前 PASS22/FAIL10 大幅改善）。
  - R2（三臂池内 clean 起点 100%）✅ PASS（895/895）。
  - **FAIL R5g**：`target_lap_var/xcorr/motion 跟住率跨度仅 0.0188`。
  - **FAIL R6c**：免责声明写 `541.09`/`5.83`（2 位小数），约束线是 541.0926320913259 / 5.828548284248887，1e-6 相对容差对不上 ⇒ **把 §10.2/§10.3 的门槛改用 `%.7g`**（541.0926 / 5.828548 / 0.1891664）即可 PASS。
  - WARN R5h（审计独立复算用「全池 895」而产物用 common_good 874 ⇒ 538.12 vs 541.09）、R5i（约束线表无 cross_arm_dev_pct 列）、R9（报告出现 trial_0080，需人工确认写的是「待替换」）。

## 2. 待决核心问题（R5g 的根因，尚未落定）

审计发现**真缺陷**：守卫用的 `curve_y_span` 是**逐档（level）跨度**，而膝点是从**按指标值分箱**的曲线取的 ⇒ 两者不是同一条曲线。
- xcorr/motion/target_lap_var：level_span 6.41pp ✅，但**分箱后 y_span 仅 1.92pp**（曲线几乎没塌）⇒ 膝点 541.09 其实是「曲线最左端分箱边界」，双算法 0.00% 一致属于 W6 所说的**伪一致**。
- xcorr/blur/target_lap_var：分箱 y_span 10.98pp（够深）但 kneedle=2892.85 vs pchip=89.89 ⇒ delta 18.92% > 15% ⇒ 双算法门禁作废。
- 其余有效曲线：xcorr/blur/local_scr（level 7.78 / 分箱 19.79，delta 13.93%，膝点 5.8285）、xcorr/motion/target_local_contrast（6.41 / 16.16，delta 0.04%，膝点 0.18917）。
- 原型（已跑 `/tmp/probe_levelknee.py`）：改用**逐档曲线**（X=该档指标中位、Y=该档跟住率）提膝点，contrast/scr 双算法一致性大幅变好（delta 0.01~0.03%），但 **target_lap_var 仍不合格**（blur delta 44.48%、motion 15.51%）⇒ 无论哪种构造，**target_lap_var 本轮都拿不到可靠膝点**（受 G1「档位不变」约束，无法加码退化档位）。
- 两个待选方案：**X** 保持现状（守卫判据写在错误曲线上，R5g FAIL，报告解释口径差异）vs **Y** 修正守卫（守卫改用分箱曲线跨度 ⇒ target_lap_var **无有效红线**，对外免责声明只剩 `target_local_contrast < 0.189166`；需同步改 `run_eval_v2` 守卫/列语义、`verify_recalc_v2.py` 的 `verify_curve_validity`、`test_iqa_v2.py` 断言、报告与 MEMORY_SECTION 数字）。
- 若选 Y，可选实现：守卫同时要求 level_span>5pp **且** 分箱 y_span>5pp；`curve_y_span` 改为分箱跨度、另加 `level_y_span` 列。

## 3. 剩余必做（收尾清单）

1. 定 R5g 方案（X 或 Y）并落代码；无论哪种都必须在报告里写清**两种 span 口径的区别与取舍**（逐档 = 受控退化轴；分箱 = 受基线离散度稀释）。
2. 修 R6c：免责声明门槛改 `%.7g` 精度。
3. 跑 `--phase report`（报告正文只能由 `phase_report()` 生成，**不要手改 .md**）并复核 §4/§6/§8.3/§9.2/§10/§11/§12 数字与 CSV 一致。
4. 更新 `MEMORY_SECTION.md` 数字（若线位变化）+ `PROGRESS_SERVER.md` 追加留痕（G9 行 + W6~W10 逐条回 + 池定义变更 + audit 仪器 bug 修复 + 「host 已关机，验收由服务器自检完成」）。
5. 最后按顺序跑：`make_manifest_v2.py` → `md5sum -c iqa_v2/output/manifest_v2.md5` → `pytest iqa_v2/test_iqa_v2.py -v`（须 100%） → `verify_recalc_v2.py --samples 120` → `audit_final.py iqa_v2/output iqa_v2/REPORT_V2_FINAL.md`（目标 FAIL=0，WARN 需在报告写判读依据）。
6. 交付物：`iqa_v2/REPORT_V2_FINAL.md`（现 594 行）、`iqa_v2/MEMORY_SECTION.md`、`iqa_v2/output/*`（含新增 `degrade_detail_v2.csv`）、`iqa_v2/PROGRESS_SERVER.md`。

## 4. 关键数字与文件锚点

- 约束线：`output/constraint_lines_v2.csv`（3 行，只有三项核心指标；`kcf_legacy_line` 全 nan、`cross_arm_dev_pct` 全 nan）。
- 膝点表列：`tracker,curve_set,degradation_family,metric,stratum,n_windows,n_bins,curve_y_span,curve_monotone,metric_monotone,knee_x_kneedle,knee_y_kneedle,knee_x_pchip,knee_y_pchip,knee_x_candidate,delta_pct_of_span,knee_reliable,method_used,knee_x_final,knee_y_final,x_at_y50,clean_success_rate`。
- 报告已含：§10 逐族逐臂表（13 行，方向：三族均「同族内部方向不一致」）、§12.12 弱相关清单（37 条可靠膝点中 29 条 usable/direction_ok 弱）、§11 条款→用例对照表。
- host 复核条：`iqa_v2/_host_warning_W6/W7/W8/W9/W10.md`、`W7_HOST_RECHECK.md`（W8-8 验收清单 ①~⑦ 需在 PROGRESS_SERVER 逐条回）。
- 常量：`CURVE_MIN_SPAN=0.05`、`KNEE_AGREE_PCT=15.0`、`GOOD_CLE_PX=3`、`CURVE_SETS=("own_good","common_good")`、`SMALL_TARGET_MAX_DIM=20`。

---

### [assistant #65]
I'll re-orient by reading the handover and task documents, then continue from where the previous run left off.

---

### [assistant #66]
Resources look healthy. Now reading the two mandatory documents in full.

---

### [assistant #67]
Let me read the truncated middle section of the handover (§3, G1–G9).

---

### [assistant #68]
Let me check the current audit state and key CSVs to ground the remaining decisions.

---

### [assistant #69]
Now let me examine the knee/guard code paths that drive R5g.

---

### [assistant #70]
I have the full picture now. Let me run my own data analysis on the R5g question while delegating the host-warning reconciliation in parallel.

---

### [assistant #71]
Agent slots are occupied by the previous run's agents. Let me close them and start my own analysis in parallel.

---

### [assistant #72]
Now the critical R5g analysis on real data. Let me compare the two curve constructions directly.

---

### [assistant #73]
Let me inspect the actual curve shapes to judge which construction is physically meaningful.

---

### [assistant #74]
Let me examine how the product builds curve sets, then recompute on the exact production subset.

---

### [assistant #75]
Now the definitive comparison on the exact production subset (common_good, 874 windows).

---

### [assistant #76]
The binned curve has textbook sigmoid structure for contrast/scr but is flat for lap_var. Let me verify one task-book detail: the book says the curve must be **isotonic + PCHIP-smoothed** before both algorithms.

---

### [assistant #77]
The evidence is decisive. Let me check how the constraint and regression phases are wired before deciding the fix.

---

### [assistant #78]
Decision is taking shape. Let me validate the corrected guard across the full grid before coding.

---

### [assistant #79]
The picture is now complete and robust. Let me look at the remaining code I need to modify.

---

### [assistant #80]
Now I'll implement the fix. Starting with the core guard semantics in `run_eval_v2.py`.

---

### [assistant #81]
Now the regression phase must keep the evidence for metrics without a valid line, clearly labelled as reference-only.

---

### [assistant #82]
Now the independent verifier must match the new guard semantics.

---

### [assistant #83]
Now let me update the verifier's guard recomputation to match the new semantics.

---

### [assistant #84]
Let me re-run the knee phase to validate the guard change end-to-end.

---

### [assistant #85]
Guard fix validated. Now the regression phase:

---

### [assistant #86]
Now the report generator. Let me read the sections that need updating.

---

### [assistant #87]
Now let me apply the report generator updates. Starting with the guard-semantics sections.

---

### [assistant #88]
Let me make the key span figures dynamic rather than hard-coded, so the report can't drift from the CSVs.

---

### [assistant #89]
Now let me generate the report and review it.

---

### [assistant #90]
Now let me update the acceptance audit to match the corrected semantics and add a check for the new no-line metric.

---

### [assistant #91]
Audit is now at FAIL=0. Let me resolve the last WARN by making R9 verify the actual replacement status.

---

### [assistant #92]
Audit fully green (38/38). Now let me check the verifier and test suite.

---

### [assistant #93]
Let me fix the guard's applicability rule and update the tests with regression coverage for the defect I fixed.

---

### [assistant #94]
The synthetic premise needs a better construction. Let me rewrite that test deterministically.

---

### [assistant #95]
Only the stale manifest fails now. Let me run a smoke test to a temp dir to validate all code paths end-to-end.

---

### [assistant #96]
Memory usage jumped. Let me check what's running before launching anything heavy.

---

### [assistant #97]
A concurrent unrelated job (ETF rotation) is using ~60 GB, cutting available memory to 36.6%. Let me run the smoke test in an isolated cache/output dir so it can't clobber real artifacts.

---

### [assistant #98]
Smoke test running. Let me update the memory section while it completes.

---

### [assistant #99]
Smoke test passed end-to-end in isolation. Let me pull the final numbers and update the memory section.

---

### [assistant #100]
Now updating the memory section with the final numbers and the new guard semantics.

---

### [assistant #101]
# HANDOFF · 红外图像质量评估 v2 收口轮（续接用 · 第二版）

## 0. 环境与硬约束（不变）

- 工作目录：`/tmp/pycharm_project_a7d966e6`（= 服务器 192.168.99.40 镜像）；**只改 `iqa_v2/`**。
- Python：`/home/huangzhe/anaconda3/envs/uav/bin/python`；KCF 需 `PYTHONPATH=/tmp/pylibs_cvcontrib`；
  运行需 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`。
- 禁止训练/下载/重训；禁止 `pkill -f`（用 `kill <pid>`）；不 git commit/push；**不改 `manu/memory/memory.md`**。
- `/mnt/data/siping/runs/**`、数据集目录**只读**。
- 每完成一个缺口在 `iqa_v2/PROGRESS_SERVER.md` 追加一行（时间/缺口/做了什么/产物路径/结论数字）。
- **资源铁律**：18:32 读数 `free -m` = 128545 总 / 79997 已用 / **47009 可用（36.6%，刚过 30% 门槛）**，
  load 24.6/80 核，GPU 空闲。**原因：另一个会话的 ETF 轮动 ML 作业
  (`v3_comparison_probe.py`, PID 2572128 及其 3 个 fold worker) 正占 ~60 GB**。
  ⇒ **不要杀它**（不是本任务的）；**不要在它跑的时候重跑 24 worker 的全量 eval**。

## 1. 本轮（我）已完成的工作 —— 核心是**修掉曲线守卫的一处真缺陷**

### 1.1 根因（用真实数据独立复算确认，非猜测）

- 守卫把**逐档跨度**（受控退化轴口径：X=退化档位）写进 `curve_y_span` 列，
  **但膝点实际取自「分箱曲线」**（X=实测指标值、Y=该值下跟住率）⇒ 等于「用 A 曲线健康度给 B 曲线放行」。
- 实测 `target_lap_var`/XCorr/`motion`：逐档跨度 **6.41 pp**（过门槛）但**分箱跨度仅 1.92 pp**
  ⇒ 整条曲线几乎是平的（跟住率 0.94~0.976 抖动）⇒ 膝点 **541.09** 只是噪声局部极大值处的
  **分箱边界**，双算法「0.00% 一致」是**伪一致**。
- 已用两份**独立原型**验证：`/tmp/probe_span.py`、`/tmp/probe_curve.py`、`/tmp/probe_final.py`、
  `/tmp/probe_grid.py`（均在 /tmp，仓库外）。结论对**两种曲线构造**都成立：
  **`target_lap_var` 在全量口径下拿不到可靠膝点**（分箱法：motion 分箱跨度 1.92 pp 过平、
  blur 双算法差 18.92% > 15%；逐档法：blur 差 44.48%、motion 差 15.51%）。
  ⇒ 该指标按任务书 §二.3「降级为物理参照」处理。

### 1.2 已落盘的代码改动

| 文件 | 改动 |
|:--|:--|
| `iqa_v2/run_eval_v2.py` | ① 新增 `binned_y_span(xs, ys)`：分箱曲线（保序回归后）的 Y 量程；② `extract_knee` 守卫改为 **5 道**：跟住率逐档单调 + **分箱跨度 > 5pp** + **逐档跨度 > 5pp** + 指标单调（分箱判据仅在 `xs.size >= 4` 时适用）；③ 膝点表新增列 **`level_y_span`**；④ `phase_constraint` 同步双跨度守卫，无候选时 note 写明原因；⑤ 报告生成器（`phase_report`）大改：§4.2 / §5.1~5.3 / §6 / §8 / §9.2 / §10.2~10.3 / §12.7~12.13 全部按新口径重写（数字一律现场读 CSV） |
| `iqa_v2/regression_v2.py` | `phase_regress`：**无有效红线**的指标改用约束线表的**参照读数**当阈值，`threshold_source = "reference_only(本轮无有效红线,约束线表参照读数)"`（保留判别力证据但明确非红线） |
| `iqa_v2/verify_recalc_v2.py` | 新增 `independent_binned_y_span()` + `N_BINS=12`；`verify_curve_validity` 改为复算**两条跨度**并双向对账（失效判定 / 分箱跨度 / 逐档跨度 / 单调性列） |
| `iqa_v2/audit_final.py` | R5f 跳过无有限线位的行；**新增 R5k**（无有效红线必须显式记 nan + 附原因）；R5g/R5h 独立复算改用**与产物同一批窗口**（用 `segment_class_v2.csv` 独立复现 `common_good` 筛选）；R6c 跳过 nan 线位；**R5i 改为机器判定**「跨臂不可比时报告须声明算法无关性不成立」；R9 改为机器判定「已替换/待替换」；`import math` |
| `iqa_v2/test_iqa_v2.py` | `test_constraint_lines_only_cover_three_core_metrics` 改为接受「nan + 原因」契约；新增 `test_guard_judges_the_curve_the_knee_comes_from`（合成：逐档跨度够深但分箱曲线全平 ⇒ 必须 degenerate）、`test_flat_lap_var_motion_curve_is_not_a_red_line`（产物级回归）、`test_knee_table_carries_both_span_columns` |

### 1.3 当前产物数字（已重跑 knee→constraint→regress→report）

- **约束线表（3 行）**：`target_local_contrast` **0.18916642395250532**（motion/xcorr，**与上一版相同**）、
  `local_scr` **5.828548284248887**（blur/xcorr，**与上一版相同**）、
  `target_lap_var` **nan（本轮无有效红线）**，`n_candidates=0`，note 写明原因。
- **knee_thresholds_v2.csv**：324 行，`knee_reliable=1` 共 **17** 行；`common_good/all` 的核心指标可靠膝点只有
  xcorr/local_scr/blur 5.82855（分箱 19.79pp/逐档 7.78pp，Δ13.93%）、
  xcorr/local_scr/motion 6.77734（21.37/6.41，Δ14.13%）、
  xcorr/target_local_contrast/motion 0.189166（16.16/6.41，Δ0.04%）。
  **`common_good/small` 上 `target_lap_var` 反而有有效膝点**：blur/xcorr **3298.78**、motion/xcorr **7389.37**、
  blur/kcf **652.26**（⇒ 报告 §5.2 / §12.13 给出锐度维度补救路径）。
- **regression_v2.csv**（阈值来源列已标 reference_only）：
  - `target_local_contrast`（0.189166，红线）：好片误杀 12.63% / 一般 33.51% / 跟丢 52.54% / 真实坏片 100% / 真实好片 29.33% ⇒ **区分间隔 70.67 pp**
  - `target_lap_var`（**505.624，参考阈值非红线**）：0.79% / 7.89% / 12.44% / 100% / 0% ⇒ 100 pp（仅作判别力证据）
  - `local_scr`（5.828548，红线但不得单独使用）：25.79% / 46.83% / 69.04% / 99.60% / 98.00% ⇒ **1.60 pp ❌**
- 明细底表 `output/degrade_detail_v2.csv` **17657 行**（2442 clean + 895×17）；仿真池 895 窗口，clean 起点三臂 100%。

### 1.4 门禁当前状态

- `audit_final.py`：**PASS 38 / FAIL 0 / WARN 0** ✅
- `verify_recalc_v2.py --samples 120`：**EXIT=0**；ROI 最大误差 3.553e-15；膝点复算 6 个误差 0.000e+00；
  曲线有效性 162/162 一致；分箱/逐档跨度相对误差均 0.000e+00 ✅
- `pytest iqa_v2/test_iqa_v2.py`：**40 passed / 1 failed**（唯一失败 = `test_output_manifest_md5_all_ok`，
  因为 manifest 尚未重建 —— **必须在所有文件改动完成后最后跑 `make_manifest_v2.py`**）
- `--phase all --smoke`（隔离目录 `/tmp/iqa_v2_smoke_out` + `/tmp/iqa_v2_cache_smoke`，8 workers）：**全阶段通过**
  （EVAL 264 行 / CLASSIFY / KNEE / CONSTRAINT / BLIND 548 行 / REGRESS / TAIL / REPORT）⇒ 代码路径端到端可用。
  真实产物未被污染（`/tmp/iqa_v2_cache/degrade_detail.csv` 与 `output/degrade_detail_v2.csv` md5 均为 `df0728b69c5dc1954decb0f3d018822b`）。

## 2. 关键决策（已定，勿再反复）

1. **采用「分箱曲线为权威口径」**：守卫判在膝点自己的曲线上；逐档跨度降为第二道守卫并单列 `level_y_span`。
   理由：分箱曲线即 P(hold | 指标值)，是**帧级阈值**真正需要的量；逐档曲线只有 5~6 点、对 X 非均匀极敏感。
2. **`target_lap_var` 本轮无有效红线**，按任务书 §二.3 降级为物理参照；上一版的 `541.09` **作废**；
   对外可发布版免责声明**只保留 `target_local_contrast < 0.1891664`**。
3. `local_scr` 有线位但**不得单独进入对外免责**（真实素材区分间隔 1.60 pp）。
4. **不再重跑 24 worker 全量 eval**：eval 段代码未改、明细 CSV 未变（md5 一致），
   且当前有他方作业占 60 GB（可用内存仅 36.6%）⇒ 已在 smoke + 分段重跑中验证。
   **必须在 PROGRESS_SERVER.md 里写清这一取舍与理由**（资源铁律）。

## 3. 剩余待办（按顺序）

1. **改 `iqa_v2/MEMORY_SECTION.md`（手工文件，不是生成物）**——上一次批量替换**因文本不匹配而失败，文件未被改动**。
   需更新的行（`grep -n "541.09" iqa_v2/MEMORY_SECTION.md` 显示 6 处：72 / 91 / 105 / 139 / 168 / 181）：
   - §二.一 守卫：四道 → **五道**，写清**两种跨度口径**与本轮真缺陷修正（6.41pp vs 1.92pp、541.09 伪一致）；
     分级更新为「XCorr 的 blur（分箱 19.79/逐档 7.78）与 motion（21.37/6.41）达 A 级；KCF blur 逐档 3.66pp 不足」。
   - §二.二 约束线表：改成上表（contrast 0.1891664 唯一可用 / scr 5.828548 不得单独使用 / lap_var nan）。
   - 逐族逐臂对照表（行 91 等）：按新 `knee_thresholds_v2.csv` 重写或删掉已作废的 541.09 行。
   - 「原因二」段落（行 105）：改为「三个族全部未过守卫 ⇒ 本轮无有效红线」。
   - 回归表（行 ~120）：lap_var 行改为 0.79% / 7.89% / 12.44% / 100% / 0%（并标「参考阈值·非红线」）；其余两项不变。
   - 免责声明段（行 139）：只留 `target_local_contrast < 0.1891664`，删掉 `target_lap_var < 541.09`。
   - 行 168 相关性表：`target_lap_var` 标「未采纳（无有效红线）」。
   - 行 181 适用边界表：锐度那一行改成「**本轮锐度维度没有对外红线**」。
   - 明细行数 **23845 → 17657**；门禁表补 `audit_final.py PASS 38 / FAIL 0 / WARN 0`；报告行数 594→当前值（约 610）。
2. **审阅 `iqa_v2/REPORT_V2_FINAL.md`**（现约 610 行，§4.2/§5/§6/§8/§10/§12 已重写）：
   重点检查是否还有与新口径矛盾的残留措辞（例如「三条线」「三道守卫」「三条红线」）。
   **报告正文不要手改**——改 `run_eval_v2.py` 的 `phase_report()` 后重跑 `--phase report`。
3. **追加 `iqa_v2/PROGRESS_SERVER.md` 留痕**（时间/缺口/做了什么/产物路径/结论数字），必须包含：
   G1~G9 逐条回、W6~W10 逐条回（含 W8-8 的 ①~⑦ 验收清单）、池定义变更、audit 仪器 bug 修复、
   **本轮真缺陷修正（守卫口径）**、**「host 已关机，验收由服务器自检完成」**、
   **未重跑 24 worker 全量 eval 的理由（他方作业占内存 36.6%，eval 段未改、明细 md5 未变、已用 smoke + 分段重跑验证）**。
4. **最后按顺序跑门禁**（manifest 必须最后）：
   ```
   cd /tmp/pycharm_project_a7d966e6
   export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=/tmp/pylibs_cvcontrib
   /home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/run_eval_v2.py --phase report   # 若报告有改动
   /home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/make_manifest_v2.py
   md5sum -c iqa_v2/output/manifest_v2.md5
   /home/huangzhe/anaconda3/envs/uav/bin/python -m pytest iqa_v2/test_iqa_v2.py -v   # 须 40+ 全过
   /home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/verify_recalc_v2.py --samples 120
   /home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/audit_final.py iqa_v2/output iqa_v2/REPORT_V2_FINAL.md
   ```
5. 最终交付物：`iqa_v2/REPORT_V2_FINAL.md`、`iqa_v2/MEMORY_SECTION.md`、`iqa_v2/output/*.csv` + `manifest_v2.md5`、`iqa_v2/PROGRESS_SERVER.md`。

## 4. 关键文件锚点

- `iqa_v2/run_eval_v2.py`：`binned_y_span`（新增）、`extract_knee`（守卫）、`phase_knee`（`level_y_span` 列）、
  `phase_constraint`、`phase_report`（§4.2 ~ §12.13）。
- `iqa_v2/regression_v2.py`：`phase_regress` 的 reference_only 阈值分支。
- `iqa_v2/verify_recalc_v2.py`：`independent_binned_y_span`、`verify_curve_validity`。
- `iqa_v2/audit_final.py`：R5f / R5g / R5h / R5i / R5k / R6c / R9。
- `iqa_v2/test_iqa_v2.py`：三条新增用例 + 约束线表契约改造。
- host 警告原文（必读，尚未逐条回填 PROGRESS）：`iqa_v2/_host_warning_W6/W7/W8/W9/W10.md`、
  `iqa_v2/W7_HOST_RECHECK.md`、`iqa_v2/_host_warning_W8_tailframe_classification.md`。
- host 权威副本 `manu/memory/memory.md` **禁止修改**。

## 5. 上一版摘要里已作废的结论（勿沿用）

- ❌「`target_lap_var` 541.09 是可用红线」——**已作废**。
- ❌「R5g FAIL 是审计仪器 bug」——**不是**：产品守卫本身语义错位，audit 是对的；已修产品。
- ❌「R5h/R5i 是可忽略 WARN」——已通过修正 audit 口径（R5h 改用 common_good 同批窗口、R5i 机器判定）转为 PASS。
- ❌「KCF 最高跨度 3.66pp」——那是**逐档**口径；分箱口径 KCF 最高 7.78pp，但逐档不过 ⇒ 结论不变（KCF 无有效线位）。

---

### [assistant #102]
我先读交棒书和任务书，确认现状与缺口清单。

---

### [assistant #103]
内存可用 42.8%（55030 MB），超过 30% 门槛。现在读交棒书与任务书。

---

### [assistant #104]
交棒书被截断了，我读完整版 §2 和 §3。

---

### [assistant #105]
交棒书读完。现在核对当前代码/产物状态，确认 G1~G9 各自的完成度。

---

### [assistant #106]
G1 已完成（三族）。现在核对产物与关键数字。

---

### [assistant #107]
产物齐全（G1~G5 已落地）。现在并行：我自己改 MEMORY_SECTION.md，同时派两个子代理做独立核查。

---

### [assistant #108]
子代理已并行启动。现在我自己处理 MEMORY_SECTION.md。

---

### [assistant #109]
需要精确的当前数字，我从 CSV 取真值。

---

### [assistant #110]
报告主体已按新口径重写。我读一下报告 §10 的逐族逐臂对照表，让 MEMORY_SECTION 与报告严格一致。

---

### [assistant #111]
数据齐全。现在改 MEMORY_SECTION.md（守卫口径与真缺陷修正）。

---

### [assistant #112]
MEMORY_SECTION 已改完。核对三分法与切分数字，同时看子代理进展。

---

### [assistant #113]
三分法/切分数字全部核对一致。修正一处表述精度（仿真池只用标定集好片段）。

---

### [assistant #114]
MEMORY_SECTION 完成。现在看 PROGRESS_SERVER.md 的既有格式，准备追加留痕。

---

### [assistant #115]
子代理发现两处报告与代码矛盾（§4 底座口径写成「并集」、csrt 底座数写成 895）。我去修生成器而不是手改报告。

---

### [assistant #116]
四���修复已落地。等审查代理的报告结论，一并重生成报告。

---

### [assistant #117]
审查发现 4 处 BLOCKING + 6 处口径残留。全部在生成器里改（不手改报告）。先定位代码。

---

### [assistant #118]
现在一次性把 10 处修复打进报告生成器。

---

### [assistant #119]
审计仪器 R6b 挡不住「作废阈值残留在对外版」。补一道 R6e 精确校验。

---

### [assistant #120]
现在同步补用例（报告与 CSV 一致性的回归护栏）。先看测试文件风格。

---

### [assistant #121]
代码改动完成。重生成报告（G8 产物）。

---

### [assistant #122]
两处渲染瑕疵（换行丢失、反引号嵌套）修掉。

---

### [assistant #123]
报告重生成完毕（626 行）。现在跑门禁：审计 + pytest + 复算。

---

### [assistant #124]
审计 PASS 39 / FAIL 0 / WARN 0（含新增 R6e）。跑 pytest 与独立复算。

---

### [assistant #125]
三个新用例自身有 bug（列索引/指标范围）。修掉。

---

### [assistant #126]
门禁全绿。更新 MEMORY_SECTION 的行数与审计计数。

---

### [assistant #127]
写 PROGRESS 留痕前，先核对 W8-8 验收清单 ①~⑦ 的当前状态。

---

### [assistant #128]
全部核对完毕。现在写 PROGRESS_SERVER.md 留痕。

---

### [assistant #129]
留痕写完（506 行）。现在最后一步：重建 MD5 清单 → 全量门禁。

---

### [assistant #130]
44 条全过。修正 MEMORY_SECTION 与 PROGRESS 里的用例计数（MEMORY 在清单内，需重建清单）。

---

### [assistant #131]
清单 25/25 OK。最终跑一遍完整门禁确认。

---

### [assistant #132]
全部门禁通过。最后做一遍收尾核查（未越界、交付物完整、章节结构正确）。

---

### [assistant #133]
全部完成。服务器自检收口轮已交付。

## 交付物

| 文件 | 规模 | 说明 |
|:--|:--|:--|
| `iqa_v2/REPORT_V2_FINAL.md` | 12 章 / 626 行 | 主交付物，数字**全部现场读 `output/*.csv`**（零手抄） |
| `iqa_v2/MEMORY_SECTION.md` | §318 / 280 行 | 可并入 `manu/memory/memory.md` 的成稿章节（**未动** host 权威副本） |
| `iqa_v2/output/` | 10 CSV + 1 JSON + `manifest_v2.md5` | 清单 **25/25 OK** |
| `iqa_v2/PROGRESS_SERVER.md` | 506 行 | G1~G9 / W6~W10 / W8-8 ①~⑦ 逐条留痕 |

## 门禁（服务器自检，全绿）

`pytest` **44/44** · `verify_recalc --samples 120` **EXIT=0**（ROI 误差 3.553e-15、膝点复算 0.000e+00、曲线有效性 162/162）· `audit_final` **PASS 39 / FAIL 0 / WARN 0** · `md5sum -c` **25 OK / 0 FAILED**

## 本轮最重要的技术结论

**曲线守卫口径修正（本轮真缺陷）**：守卫原把**逐档跨度**写进 `curve_y_span`，而膝点取自**分箱曲线** ⇒ 等于「用 A 曲线健康度给 B 曲线放行」。实测 `target_lap_var`/XCorr/motion 逐档 6.41 pp 过门槛但**分箱仅 1.92 pp** ⇒ 上一版的 `541.09` 是「双算法 0.00% 一致」的**伪一致**，**本轮作废**。

⇒ **对外可发布版只保留一条线：`target_local_contrast < 0.1891664`**（motion/xcorr）；`target_lap_var` 无有效红线（按任务书 §二.3 降级为物理参照）；`local_scr` 5.828548 有线位但**不得单独使用**（真实素材区分间隔仅 1.60 pp）。

## 两个子代理的独立复核 → 已全部修完

复核查出 **4 个 BLOCKING + 6 个口径残留**，全部改在**生成器**（`run_eval_v2.py`）而非手改报告：§7 把 `global_contrast` 写成「无」（实有 3 条有效膝点）、§5.4 硬编码「A 级」、§8.3 把 `local_scr` 标成「红线」、§4 把底座写成「并集」（实为交集）；另修 §3 切分覆盖缺口披露、§12.9 残留旧池值 1259、§10.2 标「草稿」、§十一 补资源读数。并给审计仪器新增 **R6e**（只截 §10.3 对外版校验，防止作废阈值残留）与 **3 条 pytest 报告-产物一致性用例**。

## 两点需你知悉

- **未重跑 24 worker 全量 eval**：`eval` 段代码未改，明细底表 md5 与 cache 侧**逐位一致**（`df0728b6…`），且已用 `--phase all --smoke` 全阶段验证；但当时另一会话的 ETF 作业占 ~60 GB（可用内存仅 38.0%），按资源铁律未启动。理由与复跑建议写在 `PROGRESS_SERVER.md` §五。
- **host 18:00 已关机**，无独立复核通道；本轮验收由服务器自检 + 两个子代理交叉复核完成，`PROGRESS_SERVER.md` 已写明该事实。


## 工具调用统计

```
{
  "exec_command": 543,
  "spawn_agent": 11,
  "wait_agent": 12,
  "view_image": 3,
  "update_goal": 2,
  "write_stdin": 10,
  "close_agent": 9
}
```
