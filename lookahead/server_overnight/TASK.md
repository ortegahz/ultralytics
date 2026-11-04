# 夜间作业任务书 · Look-ahead 快速外推外挂（第二阶段：精度返工 + 交付口径）

> 你是本会话在服务器上的**接班人**。本文件 = 原会话的全部上下文 + 今晚的作业计划。
> 服务器：`huangzhe@192.168.99.40`，工程根目录 `/tmp/pycharm_project_10ae9e2e`（下文记作 `$ROOT`）。
> 你的产物全部写到 `$ROOT/runs/lookahead/server_out/`（下称 `$OUT`），并**边做边写进度日志** `$OUT/progress.md`，
> 这样早上 9 点用户即使只看你一句话，也能知道进度。

---

## 0. 铁律（违反即作废）

1. **禁止修改** `manu/evaluation/*` 原始评估脚本、`manu/data/build_full_median_dataset.py`。
   外挂只能以**拷贝**形式存在（已在 `manu/tracking/lookahead/`，`lookahead_longquan.py` 是 `manu/evaluation/system_sota_longquan.py` 的逐字拷贝 + CLI）。
2. **禁止 commit / 建分支**（工作区是脏的，`manu/tracking/` 整个目录 untracked，别去动 git）。
3. **开关关闭时必须逐位一致**：任何改动后都要跑 Gate A（见 §3），`--lookahead off --arms original` 的输出 JSON 必须与原脚本输出 **md5 全等**。不一致 = 作废重做。
4. 模型固定 **RC2 / trial_0080 / epoch3**（`runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`），**回放缓存、不重跑推理**。
5. 每完成一个阶段：① 更新 `$OUT/progress.md`；② 在 `$OUT/` 下留产物；③ 不要只在对话里说，要落盘。
6. 长任务（>2 分钟）一律 `nohup`/`setsid` 后台跑 + 轮询日志，别把终端卡死；沙箱后台进程会被回收，**重活必须前台 exec 或 setsid nohup**。
7. 仓库 AGENTS.md 要求多角色委派（planner/coder/tester/reviewer/explorer/worker），**你是主代理，必须复验子代理结论**（重读文件/比对 md5），不能只信自述。
8. 禁止启动训练任务（用户未授权）。只做评估/推理回放/代码改动。

---

## 1. 背景：这套系统在做什么

- 项目：**红外弱小目标检测 + 跟踪**（YOLO/ultralytics 系），目标平台是**转台伺服**：检测器慢（~320ms），但伺服需要**低延迟**的目标坐标。
- 现行架构（`manu/evaluation/system_sota_longquan.py`）：后台全局检测器逐帧出框 → Kalman 跟踪器做时序融合 → 上报坐标。
  **问题**：检测框到达时已滞后 Δ=8 帧（25Hz 下 = 320ms），转台拿到的坐标是"8 帧前"的世界位置 → 物理脱靶。
- 本任务：在**不改动原脚本**的前提下，做一个"快速前向推演外挂"——用局部快核（<1ms）沿 Kalman 速度即时外推，
  把上报坐标从"t−Δ 的真值"变成"当前物理时刻 t 的估计"，从而消除 320ms 物理滞后。
- 用户验收关心两个口径（**必须同时给出**）：
  - **交付口径（同帧 F1，产品）**：原代码 **24 序列带 track F1 = 94.747%**、Recall 92.342%、FAR 0.0208、FP 3440、pass 21/24。
    外挂版同口径 F1 的差值 = "精度损失"。
  - **伺服口径（物理时序）**：原代码实际输出的是 t−Δ 的框，等价于"硬吃 320ms" ⇒ 基准 77.203%；
    外挂若到 94.316% ⇒ **追回 89.1% 的滞后损失**（+17.11pp）。
- 评估口径细节（务必读源码确认，别信我复述）：
  `evaluate_biz()` 在 `manu/evaluation/system_sota_longquan.py:371-394`；`count_matches(..., "inbox")` 在
  `manu/evaluation/eval_multiscale_cache_th.py:172-216`：贪心 score 降序、纯点落框内、一 GT 吸一检测、**inbox 模式下 tol=8.0 被忽略**、FAR 分母=帧数。
  **评估是同帧对齐的（无时间平移）**，所以"外挂版同帧 F1"天然会低于原代码（它引入了物理滞后语义）——这正是需要用"两把尺子"解释的地方。

---

## 2. 已完成的工作（**不要重做**，先读代码确认）

代码目录 `manu/tracking/lookahead/`（全部 untracked，随 rsync 已到服务器）：

| 文件 | 作用 |
|---|---|
| `extrapolator.py` | 外挂核：`LookaheadConfig` + `build_report_stream` + `servo_metrics`（CLE p50/p95、hit@T、jitter、fragmentation、hold-over） |
| `lookahead_longquan.py` | `manu/evaluation/system_sota_longquan.py` 的**逐字拷贝** + `--lookahead/--arms/--delta/...` CLI + 四臂（original/stale/replay/...） |
| `track_fusion.py` | `manu/evaluation/eval_bidirectional_track_fusion.py` 的逐字拷贝（仅加了 `capture_kf_states`） |
| `diag_tail.py` | 分桶归因诊断（按帧段/序列定位尾部失败） |
| `make_report_table.py` | 从产物 JSON 生成 Markdown 对照表 |
| `README.md` | CLI 速查表（全参数） |

单元测试：`manu/tracking/tests/test_lookahead_plugin.py`（18 项；全库 135 项 unittest 全绿）。
记忆文档：`manu/memory/lookahead_extrapolator_20261102.md`（§11~§16 是本阶段新增，含最终对照表/两个 bug/被否路线）。
**本包内副本**：`docs/lookahead_extrapolator_20261102.md`、`docs/rules.md`、`docs/design_spec_user.txt`（用户原始设计书）。

已验证硬门禁 **Gate A**：`--lookahead off --arms original` 与原脚本 9 份 JSON（8 序列 + summary）**md5 与 `cmp` 全等**。

### 2.1 当前 8 序列（gate 缓存）对照表 — 这是**不完整**的中间态

Δ=8（320ms@25Hz），8 序列 gate 缓存：

| 方案 | F1 | FAR/帧 | hit@5px | CLE p50 | CLE p95 | max_jitter | frag |
|---|---|---|---|---|---|---|---|
| 原逻辑 `--lookahead off`（无滞后天花板） | 96.400% | 0.0063 | 88.13% | 1.97 | 13.43 | 65.2 | 4 |
| 臂1 纯检测基线（硬吃 320ms） | 77.203% | 0.1807 | 38.18% | 20.43 | 113.73 | 65.2 | 5 |
| 外挂·最佳（perframe） | 94.316% | 0.0250 | 79.00% | 3.00 | 35.66 | **365.9** | 5 |

当前最佳配置：
```
--lookahead replay --delta 8 --arms original,stale,replay \
--replay-max-hold 8 --replay-fallback holdv --replay-anchor delta \
--fast-win-size 48 --fast-bg-window 96 --no-replay-hold-use-ref
```
（`--replay-step-rel-tol-px` 默认已改为 **0**；首步门修好后净亏 −1.74pp。）

### 2.2 已知的三条"结论性缺陷"（**今晚的主要工作量**）

1. **评估集用错**：8 序列 gate 缓存的特征源是 `ft_gmc_median_official`，**不是**交付用的 `ir_gmc_median` FPV 语义（逐像素差 26~70/255），
   且 8 条全是 pass 序列（偏简单）。⇒ **必须换成 24 序列官方缓存重跑**。
2. **max_jitter 366px ≫ 原逻辑 65px**：转台会抽搐，物理上不可交付。根因是**每帧独立重推**（per-frame replay），
   没有时间连续性 ⇒ 相邻帧的推演结果不连贯。
3. **架构偏差**：用户设计书（`docs/design_spec_user.txt`）描述的是**"滚动链 + 历史缓存"**，
   而当前实现是**每帧独立重推**。→ 必须实现 rolling chain。

---

## 3. 今晚的作业计划（按顺序推进，每个阶段都要落盘）

### Phase 0 · 环境与基线自检（先做，10 分钟）
```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
python -V; nvidia-smi -L
python -c "import numpy,torch;print(numpy.__version__,torch.__version__,torch.cuda.is_available())"
python -m pytest manu/tracking/tests -q            # 期望全绿
```
⚠️ **numpy 版本坑**：`uav` 环境的 numpy 2.2.6 与 torch 1.13.1(编译于 numpy 1.x) 不兼容，报 `Failed to initialize NumPy: _ARRAY_API not found`，
会让 `manu/data/multiscale_heatmap_cache.py` 跑不起来。**服务器上有 GPU，优先在服务器建 numpy<2 的环境**
（例如 `conda create -n lqcache python=3.9 numpy=1.26 -y`），并把结论写进 `$OUT/progress.md`。

### Phase 1 · 24 序列官方缓存（**有权威配方，别猜**）

服务器：`ssh -p 32222 huangzhe@192.168.99.40`，工程 `/tmp/pycharm_project_10ae9e2e`，conda 环境 `uav`（4×RTX 4090D）。
权威交付文档：`runs/optuna_mix_ratio/trials/trial_0080/lq_eval/DELIVERY_README.md` + `DELIVERY_MANIFEST.json`；
完整复现命令清单见 `manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md` §54.10。

**缓存多半已经不在了**（2026-11-02 18:15 经 SSHFS 实测：`trial_0080/` 下只剩 `gate/`、`logs/`、`trial_result.json`、`weights/`，
**没有 `lq_eval/`**）。重建有现成的一条命令（本地与服务器都有这个脚本，md5 一致）：

```bash
cd /tmp/pycharm_project_10ae9e2e
source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav && export PYTHONPATH=.
WAIT_GPU=0 bash manu/evaluation/run_trial0080_longquan_eval.sh \
  > runs/optuna_mix_ratio/trials/trial_0080/lq_eval/pipeline.log 2>&1
```
- 它一次做完 4 段：等 GPU → **在 FPV 语义特征上重建全量 24 序列多尺度热图缓存** → 单帧臂 → 带 track 臂。
- 关键路径：权重 `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`；
  特征 `/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train`（165,223 帧 / 24 序列）；
  GT `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`；
  产物 `runs/optuna_mix_ratio/trials/trial_0080/lq_eval/{ms_cache,*.json,preds_track}`。
- `WAIT_GPU=1`（默认）会等训练让出 GPU；先 `nvidia-smi` 看一眼，闲就用 0，忙就保持 1。

**重建后的自检（必须过）**：单帧臂 `--no-pipeline` 用 `system_sota_longquan_singleframe_params.json`
应得 TP=110788 FP=10435 FN=22469 GT=133257 **F1=87.070%**（冻结基线逐个数字一致）——说明缓存语义/参数口径正确。

**⚠️ 绝对不要用 `trial_0080/gate/epoch3/ms_cache`（8 序列）做评估**：
那 8 条缓存的特征源是 `longquanshan_ft_gmc_median_official`（official 逐 lag 拟合语义），
与交付用的 FPV 仿射连乘语义 `ir_gmc_median` **逐像素差 26~70/255**，混用会让 F1 变成两个分布的加权平均。
服务器有 `manu/memory/trial0080_gate_DO_NOT_USE.md` 专门写了这条禁令。**此前所有 8 序列结论一律作废，必须以 24 序列为准。**

### Phase 2 · 复现 24 序列交付基线 + Gate A
```bash
python manu/evaluation/system_sota_longquan.py \
  --cache-root $CACHE --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  --all-seqs --out-json $OUT/original_24.json --out-preds-dir $OUT/preds_original_24
# Gate A: 外挂关闭必须逐位一致
python manu/tracking/lookahead/lookahead_longquan.py --lookahead off --arms original \
  --cache-root $CACHE --labels-root ... --params-json ... --all-seqs \
  --out-json $OUT/gateA_off.json --out-preds-dir $OUT/preds_gateA_off
cmp/md5 两者 → 必须全等
```

### Phase 3 · 实现 **rolling chain 外挂**（核心算法返工，今晚的主战场）

**为什么必须改**：现在的 `replay` 臂是**每物理帧独立重推**——每一帧都从源帧 f=t−Δ 重新起链、重跑 Δ 步。
后果：(a) 相邻帧的链起点不同、质心偏置 b 的实现路径不同 ⇒ **max_jitter 366px**（原逻辑只有 65px，转台会抽搐）；
(b) 每次都要在源帧重跑一次快核做锚点校验，源帧失锁就整段作废（实测 001324 有 846 帧）。
用户设计书（`docs/design_spec_user.txt` 档位 B）要的是**滚动链 + 深度 Δ 的环形缓冲池**，不是每帧重推。

**已确认可用的接口**（不要重新造）：
- `manu/tracking/async_dual_rate/fast_tracker.py` → `LocalPatchTracker.step(frame, center_xy) -> TrackerResult{xy, valid, energy, snr, peak_xy}`；
  `frame` 是 2-D uint8 灰度原图。
- `lookahead_longquan.py:455` → `SequenceFrameSource(frames_root, frame_indices, delta)`，按**物理帧号**取图
  （`<frames-root>/<sequence>/frame_%06d.jpg`）。CLI 已有 `--frames-root`。
- 外挂核在 `manu/tracking/lookahead/extrapolator.py`：`build_report_stream` / `servo_metrics` / `LookaheadConfig`。

#### 3.1 状态（跨物理帧持久，不能每帧重置）

```
chain_ring        : 长度 Δ+1 的环形缓冲, 存 chain_pos[frame]（快核链在每个物理帧的位置, 无效则 None）
chain_roi         : 快核链的 ROI 中心（下一步的 center_xy 输入）
last_step_vector  : 最后一次**实测**的单步位移向量（hold-over 用）
pending_offset    : 最近的锚点残差 offset = A_j − chain_pos[j]
report_prev       : 上一帧的上报坐标（holdc 兜底用）
```

#### 3.2 每个物理帧 t 的流程（严格按此顺序）

1. **推进链（1 步/帧，只走一步！）**
   - 取 `image_t = frame_source.get(t)`；
   - `res = tracker.step(image_t, chain_roi)`；
   - 若 `res.valid`：`chain_pos[t] = res.xy`；`last_step_vector = res.xy − chain_roi`；
     否则（hold-over）：`chain_pos[t] = chain_pos[t−1] + last_step_vector`（`replay_max_hold` 步内允许，
     超限则进兜底，见 3.3）。
   - `chain_roi ← chain_pos[t]`。
2. **接收新锚点**（滞后检测在物理时刻 t 送达的是**第 j = t−Δ 帧**的记录）
   - `A_j` = 管线锚点（`_base_position(det, j, cfg, kf_states)`，即 KF 修正后的位置）；
   - 若 `chain_pos[j]` 有效 ⇒ `pending_offset = A_j − chain_pos[j]`；
   - **重锚（re-center）**：`chain_roi ← chain_pos[t] + pending_offset`
     （把 ROI 中心拉回"经锚点校正后的绝对位置"，防止链的绝对偏置 b 随时间单调累积；
      校正量通常几像素 ⇒ 不会造成抽搐）；
   - 若 `chain_pos[j]` 无效（链刚起步或丢过）⇒ `pending_offset` 保持上一值不动。
3. **上报**
   - `report(t) = chain_pos[t] + pending_offset`；
   - 若链此刻无效（hold 超限）⇒ 走 3.3 兜底。

#### 3.3 兜底（按优先级）

| 情形 | 兜底 | 备注 |
|---|---|---|
| 链某一步观测无效，但连续无效 ≤ `replay_max_hold` | 用 `last_step_vector` 继续推 | 对应"自持周期"指标 |
| 连续无效 > `replay_max_hold` | `--replay-fallback holdc`：`report_prev + v` | **已实现**，优先用它（连续性最好） |
| 链完全没起步（还没有任何有效观测） | 用管线锚点 + 匀速外推 | 对应 `--replay-fallback holdv` |

**注意**：**不要再在源帧重跑快核做锚点校验**（那是 per-frame 重推才需要的，滚动链不需要）。
源帧失锁那 846 帧的问题会**结构性消失**——链是连续的，从来不需要"跳回源帧重新定位"。
`replay_anchor_check_px` / `replay_source_anchor_fallback` 这些是 per-frame 档位的参数，rolling 档位直接不用。

#### 3.4 Δ 融合公式（为什么它同时消掉漂移和抖动）

设 `A_j = p(j) + e(j)`（管线锚点 = 真值 + 锚点误差），`chain_pos = p + b`（链 = 真值 + 链偏置）：
```
offset      = A_j − chain_pos[j] = e(j) − b(j)
report(t)   = chain_pos[t] + offset = p(t) + e(j) + (b(t) − b(j))
```
- **漂移**：每 Δ 帧被 `e(j)` 重新归零 ⇒ 不累积（与 per-frame delta 同源，但基准更稳）；
- **抖动**：`b(t) − b(j)` 现在是一条**连续链**上平滑的差值（每帧只走一步）⇒ 帧间差分小 ⇒ jitter 掉一个数量级。
  per-frame 重推里这个差值是"每次重新起链的随机实现差"⇒ 366px 抖动就是这么来的。

#### 3.5 建议新增的 CLI（保持 `--lookahead off` 路径**一行都不改**，Gate A 不受影响）

```
--replay-mode {perframe,rolling}     # 默认 perframe（保持现状可复现）；rolling 为新档位
--rolling-recenter {0,1}            # 是否在锚点到达时重锚 ROI 中心（默认 1）
--rolling-hold-mode {last_step,prev_report}   # 链失效时的推进依据
```
外挂核里对应 `LookaheadConfig.replay_mode` / `rolling_recenter` / `rolling_hold_mode`。
**验收**：rolling 档位下 `servo_metrics` 的 `max_jitter` 必须 **≤ 80px**（原逻辑 65.2px 量级）；
CLE p50 应 ≤ per-frame delta 档位（3.00px）。若 jitter 反而变大，说明链的 ROI 中心或重锚逻辑写错了。

#### 3.6 必须补的单元测试（`manu/tracking/tests/test_lookahead_plugin.py`）

1. **合成匀速直线目标**：构造假帧序列（目标按 v 移动的亮斑）+ 假管线锚点 ⇒ rolling 档位
   - 上报坐标误差 **≤ 1px**；
   - 相邻帧上报位移 ≈ v（抖动 ≤ 0.5px）；
   - 与 per-frame 档位对比，rolling 的 max_jitter 显著更小。
2. **锚点延迟生效**：锚点故意晚 Δ 帧到达，验证 `pending_offset` 的时序不串位。
3. **快核连续失效 N 步**：验证 hold-over 用 `last_step_vector` 续推，且不超过 `replay_max_hold` 后切兜底。
4. **Gate A 不回归**：`--lookahead off` 的输出与原脚本逐位一致（已有测试保持绿）。

### Phase 4 · 扫参 + 三臂对照表
- 网格（每个配置 24 序列，服务器 GPU 推理回放，快）：`rolling` × {Δ融合, 绝对式} × {holdc, holdv} × {win 32/48/64}。
- 指标口径：`servo_metrics` 的 CLE p50/p95/max、hit@5px、hit@10px、max_jitter、fragmentation、hold-over 帧数。
- 出表：`python manu/tracking/lookahead/make_report_table.py --out $OUT/comparison_24.md ...`
- **验收目标（用户口径）**：
  1. 同帧 F1：外挂 vs 原代码 94.747%，**损失 ≤ 1.5pp**；
  2. 伺服口径：外挂 ≥ 93%（相对"硬吃 320ms"的 77.203% 追回 ≥ 85%）；
  3. **max_jitter ≤ 80px**（原逻辑 65.2px 量级，转台不抽搐）；
  4. Gate A 逐位一致仍然成立。

### Phase 5 · OSD 三臂视频 + 记忆文档
- OSD 渲染器：`manu/videos/render_sys_sota_longquan.py`；外层脚本 `manu/videos/render_trial0080_longquan_osd.sh`
  （`PREDS_DIR=... SEARCH_JSON=... OUT_DIR=... bash manu/videos/render_trial0080_longquan_osd.sh`）。
- ⚠️ 已知陷阱：OSD 是**按物理帧号**渲染 `per_frame`（key 来自 `record["frame_index"]`），
  离线渲染会**掩盖 320ms 物理滞后**；交付视频必须说明这一点，或额外出一条"按物理真值对齐"的对照视频。
- 交付 OSD 参考目录（服务器上应有）：`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_20261022`（pass 21 / un-pass 3）。
- 把结论追加到 `$ROOT/manu/memory/lookahead_extrapolator_20261102.md`（新开 §17+ 段，注明"服务器夜间作业"），
  并把最终对照表写进 `$OUT/comparison_24.md`。

---

## 4. 早上 9 点用户要看的（你的交付物清单）
1. `$OUT/comparison_24.md` —— **三臂对照表**（24 序列，含原逻辑/臂1纯检测/外挂版 + CLE/jitter/frag）。
2. `$OUT/original_24.json`（复现 94.747% 的证据）、`$OUT/gateA_off.json`（逐位一致证据）。
3. `$OUT/progress.md` —— 一句话结论 + 做过什么 + 遗留问题。
4. 最优外挂的完整命令行（可直接复跑）。
5. 若某阶段没做完：**明确写"卡在哪、为什么、下一步"**，不要糊。

## 5. 常用路径速查
```
$ROOT/manu/evaluation/system_sota_longquan.py            # 原脚本（禁改）
$ROOT/manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json
$ROOT/manu/evaluation/eval_multiscale_cache_th.py        # inbox F1 口径实现
$ROOT/manu/evaluation/verify_delivery_evidence.py        # 交付复验器，期望 PASS 80 / FAIL 0
$ROOT/manu/tracking/lookahead/                          # 外挂（今晚主角）
$ROOT/manu/tracking/async_dual_rate/fast_tracker.py     # LocalPatchTracker（rolling chain 用）
$ROOT/manu/tracking/tests/test_lookahead_plugin.py       # 回归测试
$ROOT/manu/memory/lookahead_extrapolator_20261102.md     # 记忆文档
$ROOT/runs/lookahead/server_out/                        # 你的产物
```
GT 标注：`/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`
官方 OSD 参考：`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_20261022`

## 6. 兜底
若 codex 会话因认证/网络失败，`$OUT/eval.log` 里的**确定性评测脚本**仍会产出 Phase 2 的基线与 Gate A 数字；
请优先保证 `$OUT/` 里有东西。**宁可交付半成品 + 写清卡点，也不要空着手。**
