# 24 序 no-GMC × Δ=3（120 ms）· 夜间服务器自治链路

> 落档时间：2026-11-04 17:50（服务器与 host 同钟）
> 执行机：训练服务器 `huangzhe@192.168.99.40:32222`，项目根 `/tmp/pycharm_project_10ae9e2e`
> 起因：host 18:30 关机 ⇒ 「24 序 no-GMC 特征集 → 缓存 → 24 序官方口径实测」这条链必须**服务器端自治**跑完
> 线程：`01a1a5b5-cab9-7ef0-b566-b4a033b8afdb`（夜间循环 `--until 2026-11-05 08:40`）

---

## §0. 一句话状态

**最终判据 = cs_t0043（no-GMC）在 24 序官方口径、Δ=3、`replay` 臂的 F1 / Recall / FAR / hit@5px / CLE。**
截至本节落档，该数字**尚未产出**；服务器端 driver 已接管全链路，预计 19:00 前后落盘。

---

## §1. 为什么是「host 起构建 + 服务器 driver 收尾」而不是交给夜间 agent

夜间 agent 的沙箱**没有 `/dev/nvidia*`**（`torch.cuda.is_available()==False`，它自己在
`manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_sweep.sh` 头部写明了这点），
⇒ **它建不了 24 序 no-GMC 的 ms_cache**。而它的 P0b 探针
（`probe_p0b_nogmc24_cost.sh`）白纸黑字写着「**绝不执行全量构建**」。

⇒ 结论：**这条链不能等 agent**。host 直接在服务器上起进程 + 写 driver。

| 角色 | 负责 |
|:--|:--|
| host（关机前） | 起特征集构建进程、起 driver、推 4 个新脚本、写本节落档 |
| 服务器 driver | 等构建 → 自检 → 发布 → 建缓存 → 24 序 Δ 扫描 → 重生成报告 |
| 夜间 agent | P1 工作点对齐审计、P3 Δ=3 旋钮/新机理、OSD 三臂视频、**读 driver 产物做归因与落档 §37/§39** |

---

## §2. Step 0 — 24 序 no-GMC 特征集（CPU）

```bash
cd /tmp/pycharm_project_10ae9e2e
setsid nohup /home/huangzhe/anaconda3/envs/uav/bin/python manu/data/build_nogmc_rebuild_dataset.py \
  --ref-dataset /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median \
  --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --output /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc.build \
  --splits train --mode identity --window 21 --stride-step 2 --workers 24 \
  --verify-ch0-sample 200 \
  > runs/lookahead/nogmc24_build.log 2>&1 < /dev/null &
```

| 项 | 值 |
|:--|:--|
| 参照集 | `/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median`（`images/train` **165,223** 张，与官方 FPV8 缓存逐序列帧数全等） |
| 脚本 md5 | `2a35a4b0110a7a95ecd8d817dbf29c89`（= `manu/memory/nogmc_rebuild_20261027.md` §7.5 冻结哈希，三侧一致） |
| 进程 | PID **1683078**，**SID=自身**（`setsid` ⇒ 已脱离 SSH，host 关机不影响） |
| 实测速率 | ≈ **180~208 帧/s**（45 s 写 12,034 张）⇒ 全量 **≈ 14 分钟** |
| 输出目录 | **故意带 `.build` 后缀** —— 脚本 `cv2.imwrite` 直写、**无原子改名** ⇒ 两个进程同写一个目录会写坏 jpg |
| 日志 | `runs/lookahead/nogmc24_build.log` |

---

## §3. Step 1 — 逐字节等价性自检（**8 序结论能外推到 24 序的地基**）

24 序 no-GMC 与 8 序 no-GMC 用**同一份代码、同一个 raw-root**，只是参照集序列集合不同
⇒ 落在同一条序列上的帧，`W≡IDENTITY` 的计算结果必须**完全一样**；同输入数组 ⇒ 同 JPEG 编码 ⇒ **md5 逐字节相同**。

- 新脚本：`manu/tracking/lookahead/server_overnight/check_nogmc24_byte_equal.py`
- 比对对象：新集 `images/train` vs 8 序 no-GMC 的 **`images/val`**（那 8 条 gate 序列恰好全在 val）
- 8 条 gate 序列与帧数：`VIDEO00006_19700101_000703`=1864、`VIDEO00007_19700101_000818`=8263、
  `VIDEO00007_19700101_001324`=11810、`VIDEO00008_19700101_001307`=3767、`VIDEO00009_19700101_001531`=191、
  `VIDEO00029_19700101_014053`=722、`VIDEO00033_19700101_015000`=1141、`VIDEO00034_19700101_015052`=1239
  （合计 **28,997** 帧）
- 产物：`runs/lookahead/nogmc24_byte_equal.json`（`verdict = PASS` 才允许发布）

### 3.1 ✅ 实际结果（服务器 2026-11-04 18:06，**PASS**）

| 序列 | 新 24 序 no-GMC | 8 序 no-GMC | md5 逐字节相同 | 帧数相等 |
|:--|--:|--:|--:|:--|
| VIDEO00006_19700101_000703 | 1864 | 1864 | **1864** | ✅ |
| VIDEO00007_19700101_000818 | 8263 | 8263 | **8263** | ✅ |
| VIDEO00007_19700101_001324 | 11810 | 11810 | **11810** | ✅ |
| VIDEO00008_19700101_001307 | 3767 | 3767 | **3767** | ✅ |
| VIDEO00009_19700101_001531 | 191 | 191 | **191** | ✅ |
| VIDEO00029_19700101_014053 | 722 | 722 | **722** | ✅ |
| VIDEO00033_19700101_015000 | 1141 | 1141 | **1141** | ✅ |
| VIDEO00034_19700101_015052 | 1239 | 1239 | **1239** | ✅ |
| **合计** | **28,997** | **28,997** | **28,997（不一致 0 / 缺失 0/0）** | ✅ |

- 产物：`runs/lookahead/nogmc24_byte_equal.json`（**verdict = PASS**）
- 特征集构建自检：`Ch0 MAE vs reference = 1.67543 灰阶（max 8.14940）`（no-GMC vs GMC 参照集的残差，量级正常）
- 帧数校验：165,223/165,223（`exact count 165223/165223 verified with 100% parity`）
- **已发布**：`mv longquanshan_ir_gmc_median_nogmc.build → longquanshan_ir_gmc_median_nogmc`
  ⇒ 特征根 = `/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train`
- ⇒ **「8 序的 no-GMC 结论可以外推到 24 序」这个前提已被实测证明**（不是推断）。
  这条已作为 **A5 精度对齐项**写进报告（`make_fastinfer_nogmc_report.py` 的 §1 对齐表）。

---

## §4. Step 2 — 24 序 no-GMC 的 ms_cache（**必须 GPU**）

```bash
FEAT=/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train \
W=/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt \
OUT=/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq \
GPUS="1 2 3" NSHARD=3 \
bash manu/tracking/lookahead/server_overnight/rebuild_fpv8_24seq_nogmc24_server.sh
```

| 项 | 值 |
|:--|:--|
| 口径 | **逐字对齐官方 rc2 那份 24 序缓存**（`runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache`，2026-10-22 建、产出冻结基线 F1 94.747% 的那份）：`--scales 160,80 --region-threshold 0.06 --main-threshold 0.22 --fusion-dilate 32 --min-area 4 --chunk-size 64 --batch-size 8 --ablate-p0` |
| 为什么必须 GPU | CPU 慢 13~16×，且 FP32 前后处理顺序不同 ⇒ 数值口径对不上官方基线 |
| **GPU 0 不用** | GPU 0 上有**别人的进程** `honpho` 的 `mutilgpu yolov8| 8class`（已跑 1 天）⇒ 只用 **GPU 1/2/3** |
| batch 不自动降 | 降 batch 会改 FP32 分组顺序 ⇒ 可能翻像素 ⇒ OOM 直接报错，**宁可失败也不要口径漂移** |
| 参考耗时 | 官方那 4 shard 约 15:00~15:10（另有过一次部分重跑）⇒ 量级 **10~30 分钟** |

---

## §5. Step 3 — 24 序官方口径 Δ 扫描（cs_t0043）

```bash
CACHE=/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq \
OUT=/tmp/pycharm_project_10ae9e2e/runs/lookahead/fi_nogmc24 DELTAS="3 2 4 8" \
bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_nogmc24_sweep.sh
```

- 与 `run_fastinfer_24seq_sweep.sh`（rc2 / GMC）**逐字同口径**，唯一差别：`--cache-root` 指向 no-GMC 缓存、产物落 `$OUT/cs_t0043/d<Δ>/`
- CFG 仍从 `run_delivery.sh` 的 `CFG=` 用 awk 取整块再 eval，并断言 7 个关键旋钮在位（防截断）
- 四臂：`original,stale,kinematic,replay`；**rc2 的 Δ=2 已完成，别重跑**
- 单组合实测 **469 s ≈ 7.8 min**（24 序四臂全跑完）⇒ 4 个 Δ ≈ **32 min**

---

## §6. 最终判据与对照基准（**都取 24 序官方口径**）

| 口径 | F1% | FAR/帧 | hit@5px% | CLE p95(池) |
|:--|--:|--:|--:|--:|
| README §5.7 历史交付档 Δ=8（24 序） | 93.1879 | 0.029548 | 81.5612 | — |
| 24 序 rc2 · Δ=2 · `original`（零延时天花板） | **94.7469** | 0.020820 | 85.8957 | 6.844 |
| 24 序 rc2 · Δ=2 · `stale`（**交付基线**） | 92.3116 | 0.039898 ❌ | 71.0777 | 17.292 |
| 24 序 rc2 · Δ=2 · `replay` | 94.4595 | 0.023235 ✅ | 84.6815 | 8.075 |
| **24 序 cs_t0043 · Δ=3 · `replay`（← 目标）** | **待产出** | **待产出** | **待产出** | **待产出** |

对照用的 8 序同 Δ（`replay`）：rc2 `96.2788 / 0.008760 / 87.0680 / 7.1559`；cs_t0043 `96.1297 / 0.007415 / 87.3293 / 6.2009`。

⚠️ **FAR 红线 0.030**。8→24 序 FAR 实测放大倍数落在历史区间 **[×1.92, ×3.01]**
（stale ×1.93、replay ×2.98）⇒ 外推 Δ=3 的 replay 24 序 FAR ≈ **0.0168~0.0264（在线内）**，
stale ≈ **0.0738~0.1156（必然破线）**。**但外推不是判据，24 序实测才是。**

---

## §6.1 ✅ 24 序 rc2（GMC）**实测**（夜间 agent 用 `run_fastinfer_24seq_sweep.sh` 跑出，交付档 CFG）

| Δ | 臂 | F1% | Recall% | FAR/帧 | hit@5px% | CLE p95(池) | 交付三门 |
|:--|:--|--:|--:|--:|--:|--:|:--|
| — | **original（天花板）** | 94.7469 | 92.3419 | 0.020820 | 85.8957 | 6.844 | PASS |
| 2 | stale | 92.3116 | 89.9615 | **0.039898 ❌** | 71.0777 | 17.292 | FAIL |
| 2 | kinematic | 90.4866 | 88.1830 | **0.054242 ❌** | 71.7081 | 17.752 | FAIL |
| 2 | **replay** | 94.4595 | 92.0792 | **0.023235 ✅** | 84.6815 | 8.075 | PASS |
| 3 | stale | 90.0951 | 87.7980 | **0.057286 ❌** | 64.7561 | 23.605 | FAIL |
| 3 | kinematic | 87.4838 | 85.2533 | **0.077810 ❌** | 63.8683 | 24.650 | FAIL |
| 3 | **replay** | **94.2325** | 91.7850 | **0.024361 ✅** | 84.1472 | 8.760 | PASS |
| 4 | stale | 87.5534 | 85.3178 | **0.077229 ❌** | 59.5608 | 30.182 | FAIL |
| 4 | kinematic | 85.7390 | 83.5498 | **0.091488 ❌** | 57.9107 | 31.702 | FAIL |
| 4 | **replay** | 94.0389 | 91.5337 | **0.025311 ✅** | 83.5708 | 9.556 | PASS |

产物：`runs/lookahead/fi_nogmc24/rc2/d{2,3,4}/arms.json`（服务器，2026-11-04 17:30~17:45）
用时：Δ=2 **469 s**、Δ=3 **375 s**、Δ=4 **475 s**（四臂 + 24 序全跑完）

**三条结论（24 序实测，不是 8 序外推）**

1. **FAR 随 Δ 单调上升**：`replay` 0.023235(Δ2) → **0.024361(Δ3)** → 0.025311(Δ4)；
   `stale` 0.039898 → **0.057286** → 0.077229。
   ⇒ **FAR 单调性本身就是交付论据**（延时越大、外挂越容易多报）。
2. **小延时确实提升精度**：`replay` F1 随 Δ 下降 94.4595(Δ2) → **94.2325(Δ3)** → 94.0389(Δ4)；
   相对 README §5 红框「当前交付档 Δ=8」的 24 序锚点（**93.1879 / 0.029548 / 81.5612**），
   **Δ=3 提升 F1 +1.0446pp、FAR −0.005187、hit@5px +2.5860pp**。
3. **老逻辑在 120 ms 下彻底不可交付**：`stale` FAR **0.057286**（红线 0.030，超 91%）、
   `kinematic` **0.077810**（超 159%）⇒ 24 序上**必须**上视觉重推外挂。
   ⚠️ 这正是 README §5.7 说的「**8 序列掩盖了 FAR 问题**」：8 序 Δ=3 的 stale FAR 只有 0.038418。

---

## §7. 交接与拉回（host 18:30 关机 / 明早 09:00 拉回）

| 项 | 值 |
|:--|:--|
| 线程 | `01a1a5b5-cab9-7ef0-b566-b4a033b8afdb` |
| 交接机制 | **adopt → import → resume**（**不是 `thread/fork`**；fork 后 `thread/list` 不可见、`thread_source=NULL`） |
| 服务器 CODEX_HOME | `/mnt/data/.codex`（host 侧只读镜像 `/home/manu/mnt/data/.codex`） |
| 线程在服务器侧 | `state_5.sqlite` 有行：`cwd=/tmp/pycharm_project_10ae9e2e`、`archived=0`、`updated_at` 随夜间轮持续刷新 |
| 夜间循环 | `bash /mnt/data/codex-overnight/server_overnight_loop.sh --session 01a1a5b5-… --until "2026-11-05 08:40" --cwd /tmp/pycharm_project_10ae9e2e`（`setsid` ⇒ **SID=自身、PPID→1**，host 断电不受影响） |
| 明早拉回 | `bash /media/manu/1TB-Volume/workspace/handoff/overnight/morning_pullback.sh` |

⚠️ **本节的一个已知代价**：服务器侧 rollout（1,445 行）**不含 host 17:00 之后的几轮**
（host 侧 2,160 行，含本节的布置）。明早 `--adopt` 会因「host 侧 turn 更多」被拒，
脚本会自动改用 `--force` ⇒ **服务器那份胜出，本节描述的布置在对话记录里会看不到**。
⇒ 所以本节 + 4 个新脚本 + `handoff/overnight/prompt_next.txt`（§0.E）就是**今晚布置的权威载体**。

### 本轮新增 / 修改的文件（双侧 md5 已核对）

| 文件 | md5 | 说明 |
|:--|:--|:--|
| `manu/tracking/lookahead/server_overnight/drive_nogmc24_pipeline.sh` | `9a0d2e7a09ec9889d27e7758b7612321` | 新增：全链路 driver |
| `manu/tracking/lookahead/server_overnight/rebuild_fpv8_24seq_nogmc24_server.sh` | `19a1b6417f5c5e88e3d7ca0324e73f15` | 新增：24 序 no-GMC 的 GPU 分片建缓存（显式 GPU 列表、batch 不自动降） |
| `manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_nogmc24_sweep.sh` | `24840d022f42712b72de60c809b8c7bd` | 新增：24 序 cs_t0043 的 Δ 扫描 |
| `manu/tracking/lookahead/server_overnight/check_nogmc24_byte_equal.py` | `a8c125d36e2acfe56df326bbf05cc743` | 新增：8 条 gate 序列逐字节自检 |
| `handoff/overnight/prompt_next.txt` | `cce642adae6f4d6dab43754648dbfb6e` | 加 §0.E：告诉夜间 agent「24 序链路 host 已接管，别重复建」 |
| `handoff/overnight/morning_pullback.sh` | — | **修缺口**：原来只拉 `runs/lookahead/final` + `osd`，会漏掉 `fi_nogmc24/cs_t0043/*/arms.json` ⇒ 已补 `fi_nogmc` / `fi_nogmc24` / `gateA` / `fpv8_24_nogmc` / `_facts` + 两个关键日志；md5 交叉核对清单加 4 个新脚本 |

---

## §8. 明早收尾清单

1. `bash /media/manu/1TB-Volume/workspace/handoff/overnight/morning_pullback.sh`
2. `grep -c '^## 36\.' manu/memory/lookahead_extrapolator_20261102.md` 与 `grep -c '^## 38\.' manu/memory/memory.md`
   ⇒ **各应为 1**（>1 说明夜间 agent 写了第二份，必须合并；host 版是主）
3. `python3 manu/memory/validate_nav_table.py` ⇒ 必须 **RC=0**
4. `PYTHONPATH=. python -m unittest discover -s manu/tracking/tests -t .` ⇒ 基线 **156 OK**
5. 看 `runs/lookahead/nogmc24_pipeline.log`（driver 全链路进度）与
   `runs/lookahead/nogmc24_byte_equal.json`（逐字节自检 verdict）
6. **24 序 cs_t0043 的数一出来，立刻落 §36 后续小节 + §38 后续小节**（不要动 §36.11~§36.14），
   并把「小延时下精度是否提升」的三个口径答案（相对 rc2 同 Δ / 相对 stale / 相对天花板）补进报告 §0.1

---

## §9. 收尾自动化 + 口径对齐复核（18:18~18:25，host 关机前补齐）

### §9.1 为什么补收尾脚本

driver 的 Step 5 **只**重生成 `make_fastinfer_nogmc_report.py` 那一份报告，
`FINAL_REPORT.md` / house-style 逐序表 / 机器可核对的结果档都不在里面。
而 `drive_nogmc24_pipeline.sh` **正在运行** ⇒ bash 会按字节偏移增量读取脚本文件
⇒ **绝不能编辑它**（会写坏执行流）⇒ 收尾另起独立进程。

| 文件 | md5 | 说明 |
|:--|:--|:--|
| `manu/tracking/lookahead/server_overnight/summarize_nogmc24_results.py` | `fa29a156f507dede02413f927ce510e6` | 新增：从 `arms.json` **自动提取**全部数字（不人工抄写），出 **§0 结论速览（Q1 小延时提精度 / Q2 换模型包净收益 / Q3 交付三门 / Q4 Δ 假设的时延预算依据）** + T1 总体表 / T2 Δ 敏感性 / T3 换模型包对照 / T4 交付三门 / 对齐锚点校验 |
| `manu/tracking/lookahead/server_overnight/finalize_nogmc24_report.sh` | `7311056d14635896290b7cf93bb125e4` | 新增：等 driver 跑完 ⇒ 出三份报告 + house-style 逐序表 + 结果档，并**记录报告脚本 md5**（夜间 agent 若改过，明早可对账） |

服务器启动命令（已执行，PID 1737383，`setsid` ⇒ PPID→1，host 断电不受影响）：

```bash
cd /tmp/pycharm_project_10ae9e2e
setsid nohup bash manu/tracking/lookahead/server_overnight/finalize_nogmc24_report.sh \
  > runs/lookahead/nogmc24_finalize.log 2>&1 < /dev/null &
```

### §9.2 ★「以前的精度对齐是怎么做的」（读 `lookahead_extrapolator_20261102.md` 后的结论）

Fast Infer 线的**精度对齐 = 四件套**，本轮全部照抄执行：

| # | 机制 | 作用 | 本轮执行 |
|:--|:--|:--|:--|
| A1 | **尺子1（历史帧等价性）**：四臂的 `source_frame` F1 必须 **ΔF1 = 0.000000pp** | 证明外挂是**纯后处理**，没动既有跟踪资产；关开关 ⇒ 与原脚本逐位一致 | 由 `lookahead_longquan.py` 内部 `[GATE]` 打印，每次 Δ 扫描都带 |
| A2 | **Gate A（关开关 ⇒ 整份预测 JSON md5 逐字节一致）** | 构造性保证「外挂没偷偷改行为」 | `drive_gate_a_nogmc24.sh`（PID 1708082，等缓存 24 pkl 后自动跑） |
| A3 | **对齐锚点（跨轮次逐位比对）**：本轮 rc2 Δ=8 replay 必须与 README §5.18.1「当前交付档」权威数**逐位相同** | 证明本轮跑的管线 = 历史交付档那一条 | `summarize_nogmc24_results.py` 自动校验：F1 0.931879 / FAR 0.029548 / hit 0.815612 ⇒ **实测全 ✅ 一致** |
| A4 | **两把尺子分工**：对外指标一律取**尺子2 = 当前物理帧口径**（`arms[arm].current_frame.overall`）；臂自身的检测指标不取 `source_frame` | 防止把「源帧天花板」误当成本臂成绩 | 提取器硬编码 `current_frame.overall`；`original` 臂 = 天花板（Δ 无关，F1 恒 94.7469%） |
| A5 | **交付三门 + 基线臂**：基线 = `stale`，`original` 是天花板不是基线；判据 F1≥0.88 / Recall≥0.85 / FAR≤0.030 | 「能不能交付」只看相对 stale 的追回 + 三门 | T4 逐项判定；CLE 一律用**全帧池化** `cle_pooled_*`（`cle_p*` 是逐序最差值，不可混用） |

### §9.3 ★「唯一变量 = {模型包, Δ}」的口径对齐复核（本轮实测，不是推断）

| 变量 | rc2（GMC） | cs_t0043（no-GMC） | 结论 |
|:--|:--|:--|:--|
| GT | `Final_Labels_merged_20261010` | 同 | ✅ |
| 帧根 / 帧缓存 | `frames_ir_jpg` / `frame_cache_lq` | 同 | ✅ |
| 工作点 params | `system_sota_longquan_trial0080_delivery_v3_params.json` | 同 | ✅ |
| 外挂 CFG | 运行时从 `run_delivery.sh` 解析 | 同 | ✅ **`run_delivery.sh` mtime 07:37 < rc2 产物 17:30~17:55 ⇒ 两次解析的是同一份** |
| 缓存构建超参 | `--scales 160,80 --region-threshold 0.06 --main-threshold 0.22 --fusion-dilate 32 --min-area 4 --chunk-size 64 --batch-size 8 --cpu-threads 4 --ablate-p0` | **逐字相同**（diff 两份构建脚本只差 FEAT/W/OUT/GPU 列表 + 去掉 OOM 自动降 batch） | ✅ |
| 是否降过 batch | **否**（`fpv8_24/progress.txt` 只有 4 个 DONE、24/24 pkl；唯一 OOM 记录在 `s6.log`，是早期「每卡 2 shard」的失败实验，不是最终那次） | 否（脚本已去掉自动降 batch，OOM 直接报错退出） | ✅ **FP32 分组口径一致** |
| 建缓存时分片数 | 4 shard × 1/GPU | 3 shard × 1/GPU | ✅ 分片只决定「谁跑哪条序列」，不改单序列数值 |
| **唯一差异** | 特征语义 GMC + 权重 `trial_0080/epoch3.pt` | 特征语义 no-GMC + 权重 `cs_t0043.pt`（**两者都带 `--ablate-p0`**） | ← 这就是要测的量 |

### §9.4 已确认的 rc2 24 序基线（供明早对账，数字取自 `arms.json` 自动提取）

| Δ | 臂 | F1% | FAR/帧 | hit@5px% | CLE p95(池) | 三门 |
|--:|:--|--:|--:|--:|--:|:--:|
| — | original（天花板） | 94.7469 | 0.020820 | 85.8957 | 6.844 | PASS |
| 2 | replay | 94.4595 | 0.023235 | 84.6815 | 8.075 | PASS |
| 3 | stale | 90.0951 | **0.057286** | 64.7561 | 23.605 | FAIL |
| 3 | kinematic | 87.4838 | **0.077810** | 63.8683 | 24.650 | FAIL |
| **3** | **replay** | **94.2325** | **0.024361** | **84.1472** | **8.760** | **PASS** |
| 4 | replay | 94.0389 | 0.025311 | 83.5708 | 9.556 | PASS |
| 8 | replay（历史交付档） | 93.1879 | 0.029548 | 81.5612 | 12.946 | PASS |

⇒ **小延时提精度（24 序实测，rc2）**：Δ=3 vs Δ=8 ⇒ F1 **+1.0446pp**、FAR **−0.005187**、hit@5px **+2.5860pp**。
⇒ FAR 随 Δ 单调上升：replay 0.023235 → 0.024361 → 0.025311；stale 0.039898 → 0.057286 → 0.077229。

### §9.5 ★ 汇总器的「§0 结论速览」与「Q4 时延预算依据」

`summary.md` 的开头是**自动算出来的**结论速览（不是人工抄的）：

| 问题 | 口径 | rc2（当前已有） |
|:--|:--|:--|
| **Q1** 小延时 Δ=3 vs 历史交付档 Δ=8 | replay 臂逐位比 | **ΔF1 +1.0446pp**、**ΔFAR −0.005187**、**Δhit@5px +2.5860pp**、CLE p95 12.946→8.760 |
| **Q2** 换 no-GMC 模型包的净收益 | 同 Δ 跨模型 | 待 cs_t0043 24 序出数 |
| **Q3** Δ=3 能不能交付 | 交付三门 + 相对 stale / 相对天花板 | **PASS**；相对 stale **+4.1374pp**，距天花板仅 **−0.5144pp** |
| **Q4** 为什么 no-GMC 才敢用 Δ=3 | 单帧成本实测 | GMC **183.32 ms/帧**（= 4.58 个 40 ms 帧周期）→ 历史上取 Δ=8；no-GMC **61.78 ms/帧**（= 1.54 个周期，省 **121.54 ms/帧 = 66.3%，加速 2.97×**）→ **Δ=3（120 ms）假设成立** |

> Q4 的单帧成本实测来源：`manu/memory/nogmc_rebuild_20261027.md` §六（成本侧）。
> ⚠️ 「Δ=3 假设成立」是**预算口径**的论证（单帧成本 ≲ 帧周期数），不等于「实测时延已验证」——
> 真机时延要看板卡侧（`manu/memory/gmc_40ms_handles_rk3588.md`），本轮**未实测**，报告里必须标清楚。

### §9.6 明早回传脚本补强（host 侧 `handoff/overnight/morning_pullback.sh`）

| 补什么 | 为什么 |
|:--|:--|
| 产物目录加 `gateA24` | Gate A24 的产物落在 `runs/lookahead/gateA24/`（`drive_gate_a_nogmc24.sh` 的 `OUT`），**原来不在清单里 ⇒ 会整份漏掉** |
| 日志加 `nogmc24_finalize.log` / `nogmc24_finalize.md5` | finalize 的执行记录 + 报告脚本 md5（夜间 agent 若改过报告脚本，明早可对账） |
| md5 交叉核对清单加 6 项 | 2 个新脚本 + 3 个报告脚本（`make_fastinfer_nogmc_report.py` / `make_fastinfer_final_report.py` / `make_report_table.py`）+ 2 个 memory 档 |

### §9.7 「不要编辑正在运行的 bash 脚本」——今晚实测的坑（已守住）

finalize 脚本 18:18 起在服务器上跑（`setsid` ⇒ PPID→1）。
我后来想给它加一行 `--gatea-24` ⇒ **发现这会写坏执行流**：
bash 是**按字节偏移增量读取**脚本文件的，在前面插入一行会让后面所有内容的偏移错位
⇒ 循环结束后 bash 从旧偏移继续读 ⇒ 极可能语法错误 ⇒ 脚本当场死掉。

**处置**：不推送改过的 finalize（保持双侧 md5 一致 `7311056d…`），
改用 `make_fastinfer_final_report.py` 里的 **`--gatea-24` 默认值**达成同样效果
（默认值 `runs/lookahead/gateA24/gateA.json`）⇒ 功能覆盖、零风险。

⇒ 同理，`drive_nogmc24_pipeline.sh` 也在跑 ⇒ **绝不编辑它**（收尾另起进程）。
⇒ 这条是通用的：**服务器上正在跑的 `.sh`，改前先看 `ps` 有没有它**；要么另起进程，要么等它跑完再改。

### §9.8 报告脚本的 Gate A 双版渲染（口径补齐）

原 `make_fastinfer_final_report.py` 的 §5 只渲染 **8 序** Gate A
⇒ 而报告主口径是 **24 序官方口径** ⇒ Gate A 证据与主张口径不匹配（缺口）。
已改为 §5（8 序 gate 集）+ **§5b（24 序官方口径）** 两段都渲染：

| 项 | 值 |
|:--|:--|
| 新参数 | `--gatea-24`（默认 `runs/lookahead/gateA24/gateA.json`，缺失时该段输出「未找到，跳过」） |
| 新 md5 | `make_fastinfer_final_report.py` = `35db033a8eb9b034b79782ef9db99458`（双侧一致） |
| 现状 | §5 已出（8 序 **PASS**，8/8 逐位相同、尺子1 ΔF1 = 0.000000pp）；§5b 等 Gate A24 出数后自动补上 |

### §9.9 ★ 独立口径审计（子代理，只读审计，18:30~18:45）

派了**独立审计子代理**审这套 24 序对照（不重复我的检查，专注找口径漏洞）。
**总判定**：**数据层可交付**（唯一变量确实收敛到 {模型包, Δ}，rc2 参照臂与 README §5.18.1
**四臂 20 个数逐位全中**，尺子1 ΔF1 = 0.000000pp）；**报告层修了 6 条后才能交付**。

**审计逐条核对通过（无需修）**：两把尺子未混用（`current_frame.overall`）｜`original` 未被当基线（基线 = `stale`）｜
CLE 未混用（总体表用 `cle_pooled_*`，`cle_p*` 只进逐序表）｜交付三门阈值与 README §5.7 一致｜
Δ 语义 = record 下标滞后（`stale.source_verbatim` 24/24 全 True）｜
**两份缓存 meta 只差 `features_root` + `hm_weights`**，其余 16 字段全同 ⇒ FAR 分母两边都是 165,223｜
GT 根/帧根/帧缓存/工作点 params（36 键逐字同）/CFG（`run_delivery.sh` md5 `2560de4a…`，mtime 07:37 < rc2 产物 17:30）｜
缓存构建超参逐字相同（含 `--batch-size 8 --ablate-p0`）｜8→24 序外推地基（28,997/28,997 逐字节同）。

**已修（当晚改完并双侧 md5 核对）**：

| # | 严重度 | 问题 | 修法 |
|:--|:--|:--|:--|
| 1 | 🔴 BLOCKING | `summarize_nogmc24_results.py` 交付三门 **fail-open**：指标取不到判 `N/A` ⇒ 总判定仍是 PASS（另两个生成器都是 fail-closed） | 改 `("UNKNOWN", None)`；总判定**只有三项全 PASS 才 PASS** |
| 2 | 🔴 BLOCKING | Q4 无条件断言「Δ=3（120 ms）假设成立」，且缺项目自己要求的免责句 | 标题改「**时延预算依据（预算口径 · 非实测）**」；结论改「单帧预算上**支持**取 Δ=3」；补免责句（真机时延本轮未实测）；cs_t0043 24 序未出数时再补一条提示 |
| 3 | 🟡 IMPORTANT | 合并报告 §0.1 未标「仅 8 序」，且「交付方案建议」排在 24 序权威结论 §0.2 之前 | §0.1 标题加「**仅 8 序 gate 集 · 机理筛选**；24 序权威结论见 §0.2」；交付建议加「⚠️ 基于 8 序机理结论，**排序与上线判据一律以 §0.2 的 24 序实测为准**」 |
| 4 | 🟡 IMPORTANT | `make_fastinfer_final_report.py` 硬编码「165,247 帧」（实为 165,223，差 24 = 序列数） | 改为**由产物反推**（`fp / far_per_frame` = FAR 分母） |
| 5 | 🟡 IMPORTANT | `fi_nogmc24` 下的变体目录（`p3knobs`）会被汇总器当「模型包」读走 | 汇总器加**模型包白名单** `('rc2','cs_t0043','trial_0049')`，不在白名单的目录 `[SKIP]` 并打 stderr |
| 6 | 🟡 IMPORTANT | 「唯一变量」只能靠脚本 diff + 日志自证，产物本身不能自证 | 汇总器新增 **§0.5 产物级自证**：逐 Δ 比对 ①GT 总数 ②逐序列 GT ③序列集合 ④FAR 隐含分母（`fp/far`） |
| 7 | 🟡 NIT | Q1 表头把 Δ=8 统称「历史交付档」 | 改「同模型包 Δ=8」，脚注说明「历史交付档只指 rc2 Δ=8」 |
| 8 | 🟡 NIT | `_original()` 取「最小 Δ 的 original 臂」依赖隐含前提 | 改取 **target Δ（=3）** 的 original，缺失才回退 |
| 9 | 🟡 NIT | 「Δ=3 = 120 ms」其实是**中位数**不是上界 | Q1 加口径脚注：Δ 是 record 下标空间滞后，中位 120 ms（99.74% 转移恰好 3 帧），长尾来自标注稀疏段；**两侧共用同一 record 流 ⇒ 对比结论不受影响** |
| 10 | 🟡 NIT | 报告只有 8 序 Gate A，主口径却是 24 序 | `make_fastinfer_final_report.py` 加 `--gatea-24` + **§5b 24 序 Gate A 段**；`make_fastinfer_nogmc_report.py` 加 `--gate-a-24` + **A3b 行** |

**审计指出的「出数后必须做的核对动作」（非代码修改）**：

1. cs_t0043 的 `overall.gt == 133257` 且逐序 `gt` 与 `rc2/d3` 完全相同 ⇒ 两侧评的是同一批帧
   （→ 已由 §0.5 自动校验）；
2. 两次 sweep 回显的 **CFG 字符串逐字 diff**（`fi_nogmc24/sweep.log` vs `nogmc24_pipeline.log`）
   —— 因为 `lookahead_longquan.py` 写进 `arms.json` 的元数据里**缺** `replay_abstain_peak_below` /
   `replay_abstain_disp_px` 两个交付档旋钮 ⇒ 产物层不能自证 CFG；
3. 报告里把「Δ=3 / 120 ms」写成「**标称 120 ms（中位）**」。

**审计遗留风险（已记录，明早处理）**：

| 风险 | 说明 | 明早动作 |
|:--|:--|:--|
| 服务器独有 11 个脚本未回流 host | `run_fastinfer_24seq_sweep.sh`（**rc2 参照臂脚本**）等只在服务器上 ⇒ 基线不可复现，违反铁律〇 | 回传脚本已整目录 rsync `manu/tracking/lookahead`；已把 `run_fastinfer_24seq_sweep.sh` 加进 **md5 交叉核对清单** |
| 服务器独有的 `run_fastinfer_nogmc24_sweep.sh` | 默认 `CACHE` 指向 **rc2 的 GMC 缓存**、默认 `OUT` 写进 `fi_nogmc24/` ⇒ 误跑会把 GMC 数据伪装成 no-GMC | 汇总器白名单已挡住读入；**该脚本明早拉回后要改默认 CACHE/OUT 或删掉** |
| `rebuild_fpv8_24seq_nogmc24_server.sh` 注释写「官方缓存 2026-10-22 建」 | 实际对齐的是 **2026-11-03 重建版**（4 shard × GPU0~3，batch 恒 8，0 次降 batch） | 口径无害（产物 F1 94.7469 与冻结值逐位一致）；**注释明早改**（⚠️ 该脚本今晚在跑，**不能现在改**，会写坏 bash 执行流） |
| `p3knobs` 变体只改了 1 个旋钮 | `make_fastinfer_final_report.py` §2.7 文字说改了 2 个 | 明早修正 §2.7 表述 |
| `arms.json` 的 `config` 不记 `frames_root`/`frames_cache` | 帧根/帧缓存一致性只能靠脚本 diff + 日志 | 出数后把两次 sweep 的回显 CFG + frames 参数一并存进报告 |
