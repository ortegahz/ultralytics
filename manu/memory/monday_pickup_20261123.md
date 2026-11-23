# 周一（2026-11-23）开箱取数清单 · 周末搜索空间兑现

> 配套：`manu/memory/memory.md` §286（搜索空间设计）、§280~§285（历史）。
> 服务器：`ssh -F /dev/null -p 32222 huangzhe@192.168.99.40`｜工作根 `/tmp/pycharm_project_10ae9e2e`（`export PYTHONPATH=.`）
> conda python：`/home/huangzhe/anaconda3/envs/uav/bin/python`｜运行根 `W=/mnt/data/siping/runs/npu20ms/nano_search`

## 0. 一句话目标
在 **826 臂训练轴（LOT6/PT1 30/L5 28/L5B 42/CP 56/P 600/T 24）** 与 **SB4 汤融合轴（A 255 + B 255，段 2109000+/2109500+）** 的产出里，
找出**六项门禁全过且有真实增量**的候选，并给出可复制的升轨命令。

## 1. 六项严格口径（不要放宽）
基线 **Baseline_v3.1 = trial_1990055**：
`nano 0.7242438563327032`｜`nano4 0.6609168241965974`｜`f1 0.8906832039622916`｜`t1 0.8909238824339264`｜`t3 0.9488855869242199`｜`t2_fp 3`（上限 `max(base,5)=5`）

| 指标 | 判据 |
|---|---|
| nano | **严格 >** 基线 |
| nano4 | **严格 >** 基线 |
| f1 | **严格 >** 基线 |
| T1 | **≥** 基线 |
| T3 | **≥** 基线 |
| t2_fp | **≤ 5** |

台阶量纲：nano/nano4 1 阶 = `2.3629489e-4`（严格格点 1/4232）｜T1 1 阶 = `1.858e-5`｜F1 1 阶当量 ≈ `2.24e-4`｜T3 1 阶 ≈ `1.95e-5`。

## 2. 晨报与审计（已端到端演练过）
```bash
# ① 晨报调度：AT_HHMM=08:55，END=2026-11-23 09:05 ⇒ 08:55 自动出报告
tail -50 $W/morning_report_*.log
# ② 六项审计（只读，列 top）
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/six_criteria_audit.py --top 12
# ③ 可复制粘贴的升轨命令
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_ready_cmd.py --top 6
```
⚠️ 演练结果：当时 65 个「六项全过」候选**全是基线逐位克隆 ⇒ 零改进**（符合 §281）。**若仍如此 ⇒ 说明还没找到真冠军**，继续看 §3。

## 3. 重点看哪两批产出（SB4）
```bash
grep "⇒ nano=" $W/soup_fusion_axis_A.log | tail -40   # 跨家族行（父 14963/15510/1104642）
grep "⇒ nano=" $W/soup_fusion_axis_B.log | tail -60   # 同族互汤行（父 1104200/1104203/1104205）★优先
grep "三轴全过=True" $W/soup_fusion_axis_*.log        # 三轴全过 ⇒ stage-2 会自动送官方门禁
```
- ★ **B 行优先**：台账「同族汤臂 nano 缺口中位 −11 阶 vs 训练臂 −57 阶」的 5 倍压缩效应**只发生在同族内**；
  11042xx 家族 nano/nano4 双过基线但 **T1 从未门禁（未知）** ⇒ 同族汤若三轴全过，就能**第一次读到这批 cp 臂的 T1/T3**。
- 🔴 **A 行基本判死**：实测 `(14963@4,1104200@1)` w=0.10/0.15/0.20/0.25 ⇒ nano −24/−50/−100/**−200** 阶（线性应 +12，否定 36 阶）⇒ 跨家族权重平均对 nano 是破坏性的。
- ⚠️ **B 行有 51 个「自汤」废臂**（parent==child ⇒ 平均后等于子臂本身，如 `fam=bs1104200:w10` 且成员=(1104200@1,1104200@1)）⇒ **识别后忽略**，别当增量。

## 4. 训练轴产出（L5/L5B 是第二条腿）
```bash
grep -a "臂计划\|阶段1-" $W/night_train_axis.log | tail -8
grep -a "\[21" $W/night_train_axis.log | grep -a "ep[0-9] F1=" | tail -60
```
- **L5/L5B 用 cp 增强训练「造 nano」**（源在训练），**SB4 用汤「搬 nano」**（源在权重空间）⇒ 互补。
- L5/L5B 锚点（7 个）：`14963@4`（★离六项全过最近：dT1 +21.3 / dNano −22 / dNano4 +4 / dF1 +16 / dT3 −6 / t2fp 4）·
  `1610001@5`（dNano 只差 −2 阶）· `15510@4`（唯一大 T3 余量 +132.9）· `12703@5` · `12543@3`（T1 +42 但 t2fp=10 超线）· `14523@4` · `14241@2`（台账无 official 读数 ⇒ T1 未验证）。
- 网格：L5 = lr{1.5,3.5} × cp{0.05,0.15}；L5B = lr{0.75,2.5} × cp{0.05,0.10,0.15}（cp 0.10 是插值点，因 cp0.15 实测 nano **+35 阶**）。

## 5. 升轨
```bash
# 正式升轨（会做板端 T4 实测 + 种子复现判定）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_candidate.py \
  --trial <N> --name Baseline_v3.2 [--force] [--require-board]
# 🔴 种子关：对「抛光臂」要求 n_anchor ≥ 2（L5/L5B 同一配方跨 7 锚点 ⇒ 天然满足）
```
🔴 **交付冻结底座不变**：`trial_0080`（`hm_plain.onnx` / `ir_uav_det_rk3588_v1.0.rknn`，14.04~14.84 fps，F1 94.50~95.14%）。
**只有六项真过线的新模型才替换它**，否则不动。

## 6. 板卡与安全边界
- 板卡根分区 843MB/95% ⇒ **`/root/ptq_ladder` 235MB 是交付中间件，别删**。
- 板卡 NPU SIGILL 只能**物理断电**恢复；**`load_watchdog.sh` 不得启动**。
- `nano_watchdog.sh` 在跑（新阈值 12/6 GB）｜`day_supervisor_d1.sh` **没有**训练轴分支 ⇒ 训练轴只由 `heartbeat.sh` 的 `need_train()` 拉起。
- 🔴 **铁律二十**：`pgrep -f` 会匹配**含该模式的任何进程（含自己的 ssh 远端 shell）** ⇒ 停进程一律**按 PID + 排除 `$$`**，禁用 `pkill -f`。
- 🔴 任何「会 pkill 再拉起」的脚本必须加 **`flock -n`**（cron 与守望是两个竞争拉起器 ⇒ 双实例 ⇒ 撞号 ⇒ 台账污染）。

## 7. 台账/注册表/权重路径
- 台账 `$W/nano_search_ledger.json`（12k+ 臂；`arms` 逐臂、`rounds`、`waves`）｜注册表 `$W/arm_recipe_registry.json`
- 权重 `npu20ms_offcal/trials/trial_<sid>/weights/epoch<e>.pt`｜缓存 `npu20ms/tiny_ref/nano_<sid>_ep<e>.pkl`
- 🔴 `save_ledger` 带**跨进程 flock + 读-合并-写事务** ⇒ 多进程并发写安全；但**读-改-写的手工脚本**仍要小心。
