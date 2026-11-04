# 交接档 · 服务器夜间第 3 轮（2026-11-04 02:00 → 05:0x）

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`
> **线程**：`01a199f3-5523-7591-b039-9e07b7a1fb58`（host 18:30 关机，早上接回）
> **专题档**：`manu/memory/lookahead_extrapolator_20261102.md` §29 / §30
> **主档**：`manu/memory/memory.md` §30.9.24
> **铁律**：不 commit / 不 push；不改 `manu/evaluation/*`；外挂全部在 `manu/tracking/lookahead/`

---

## 1. 一句话

交付档**已更新**：冠军 + 新机理「源帧（t−Δ）检测器漏检 ⇒ 往回退找锚点重推」+ 回退链峰值置信门 0.90
（`--replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90`）。
上轮挡住它的两条顾虑本轮被量化清楚并解掉（见 §3）。

## 2. 交付档数字

**8 序（官方 gate 集）**

| 臂 | F1 | FAR/帧 | hit@5px | CLE p50 | CLE p95 | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|
| original（零延迟天花板） | 96.4002% | 0.006277 | 88.1316% | 1.57 | 5.97 | 65.24 |
| stale（老逻辑，真吃 320 ms 滞后） | 77.2034% | 0.180743 | 38.1850% | 6.88 | 57.04 | 65.24 |
| **replay（★交付档）** | **95.4644%** | **0.015105** | **85.2206%** | 1.67 | 9.99 | 60.00 |
| 上一版冠军（rb0） | 95.3038% | 0.015105 | 85.1728% | 1.67 | 9.24 | 60.00 |

**24 序（官方口径）**

| 臂 | F1 | FAR/帧 | hit@5px | CLE p50 | CLE p95 | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|
| original（零延迟天花板） | 94.7469% | 0.020820 | 85.8957% | 1.56 | 6.84 | 76.52 |
| stale（老逻辑） | 77.5225% | 0.155917 | 44.8232% | 5.29 | 56.62 | 76.52 |
| kinematic（**负收益，勿用**） | 75.7641% | 0.169734 | 38.6764% | 6.42 | 61.52 | 188.95 |
| **replay（★交付档）** | **93.1578%** | **0.029838** | **81.5379%** | 1.72 | 13.16 | 60.00 |
| 上一版冠军（rb0） | 93.0865% | 0.029814 | 81.4974% | 1.72 | 13.01 | 60.00 |

**口径**：F1/FAR = `arms.<臂>.current_frame.overall`；hit@5px = `arms.<臂>.servo.overall.hit_rate_5px`；
CLE 池化 = `cle_pooled_p50/95`；`max_jitter` 只能取最差单序列口径。
**尺子 1**（`--lookahead off`，源帧口径）F1：8 序 0.964127 / 24 序 0.947527；
**各臂逐位相同**（8 序三臂 original/stale/replay；24 序四臂 original/stale/kinematic/replay）。
**尺子 2**（当前帧口径，即交付 JSON 里的 F1）：8 序 0.964002 / 24 序 0.947469。

## 3. 两条「上轮的顾虑」本轮的结论

| 上轮顾虑 | 本轮结论 | 证据 |
|:--|:--|:--|
| 回退把 `max_jitter` 抬到 273.8 / 294.98 px ⇒ 不敢进交付档 | **伪影**。`servo_metrics` 用 records 列表下标当帧号，而 001324 的 records 覆盖帧 0..11809 却只有 10662 条（**缺 1148 帧**）⇒ 跨 105 帧的 271 px 合法位移被当成「一帧速度」做二阶差分。按真实帧号重算：**真实抖动 59.98、真实单帧位移 30.00 px**（限幅生效） | `diag_jitter_audit.py` · `runs/lookahead/final/rb8/diag_jitter_audit_rb8.json` |
| 24 序 FAR 破线（0.0304~0.0321） | **峰值门可解**：mp0.80 → 0.03012 ❌、mp0.85 → 0.03001 ❌、**mp0.90 → 0.02984 ✓**、mp0.95 → 0.02982 ✓。且回退只带来 **0 个新增误报**（8 序 TP +81 / FP +0） | `diag_rollback_attrib.py` · `runs/lookahead/final/final8/diag_rollback_attrib_rb16mp090.json` |

## 4. 新机理（已在交付档里）

```
源帧（t−Δ）检测器漏检 ⇒ 外挂没有锚点 ⇒ 该帧必然脱靶
  ↓ 新：退到最近一次有检测的帧当锚点，多走几步重推（--replay-rollback-max-frames 16）
  ↓ 再叠一道回退链峰值置信门（--replay-rollback-min-peak 0.90）
     峰值低 ⇒ 目标早已飞走、配准锁到背景结构（"幽灵坐标"）⇒ 该帧不上报
```

* **纯增益**：只在新出现的帧上多报，不动任何已有帧（命中→脱靶 = 0、被挤掉 = 0）；
* 深度在 mp0.90 下是「FAR 恒定」旋钮：8 序 FAR 恒 0.01511、24 序恒 0.02984；
* 交付档取深度 **16**：rb24/rb32 只多 0.06/0.12 pp F1，却把 8 序 CLE p95 尾部推高 0.24/0.50 px。

## 5. 复现（服务器）

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. PY=/home/huangzhe/anaconda3/envs/uav/bin/python
bash manu/tracking/lookahead/run_delivery.sh both      # 交付档（8 + 24 序）
bash /mnt/data/codex-overnight/gateA_live.sh            # 标准 Gate A（8 序）
bash /mnt/data/codex-overnight/gateA24_live.sh          # 标准 Gate A（24 序）
bash manu/tracking/lookahead/check_newknobs_off.sh 8    # 新旋钮默认关 ⇒ 逐位一致（8 序）
bash manu/tracking/lookahead/check_newknobs_off.sh 24   # 同上（24 序）
bash manu/tracking/lookahead/render_osd_3arm.sh 8       # OSD 三臂对照视频
$PY -m pytest manu/tracking/tests -q
python3 manu/memory/validate_nav_table.py
```

## 5.1 ★ 第 4 轮（05:20–06:40）：**旋钮空间穷举证伪**（交付档**不动**）

诊断（新增脚本 `manu/tracking/lookahead/diag_gap_buckets.py`）：
好锚点（源帧 t−8 检测在 5px 内）的 23859 帧里，外挂命中 22904（96.00%）、失手 955；
**失手的 955 帧里 563 帧（59%）只差 5~10 px**；失手帧的「8 帧真位移」**42.0% 超过 32 px**
（= xcorr 一次大匹配的搜索半径），而命中帧只有 10.6%。
⇒ **唯一残存的失效模式**：目标 8 帧内走太远 ⇒ 一次大匹配够不着 ⇒ 落回质心链 ⇒ 幽灵坐标。

据此在**当前冠军配置**下扫了 5 个旋钮族、**18+ 个变体**（每个只改一个旋钮）：
**没有一个在 hit@5px 上超过交付档**（最好 +0.015 pp ≈ 4 帧，噪声）。

| 旋钮族 | 扫描 | 结论 |
|:--|:--|:--|
| xcorr 搜索半径 | 32 / 36 / 40 / 48 / 64 | **32 最优**（64 → hit −0.61 pp、FAR +0.0065） |
| xcorr 档位 | direct / chain / chain_direct | **direct 最优**；chain 在**极性修好之后**仍打不过（历史否决复核成立） |
| 模板尺寸 | 28 / 32 / 36 | 32 最优；36 几乎持平但 FAR 更低（8 序 0.01511 → 0.01476） |
| xcorr `th_k` | 1.0 / 1.5 / 2.0 | **1.5 最优** |
| xcorr `min_peak` | 0.25 / 0.40 / 0.60 / 0.80 | **0.25（默认）最优** |
| ROI 前推上限 | 4 / 8 / 12 / 16 / 24 | **4 最优** |

⚠️ **口径陷阱（踩过一次）**：`servo_metrics` 里有两个 p95 ——
`cle_pooled_p95`（**池化**，8 序 9.99 px）与 `cle_p95`（**最差单序列** 001324，25.74 px），
**差 2.6 倍**；而 **console 打印的 `CLE p95=` 用的是「最差单序列」口径** ⇒ 拿 console 的数
跟 JSON 的 `cle_pooled_p95` 比会误判「尾部被拉差 2.6 倍」。跨配置比较必须同字段。

## 6. 唯一剩余方向

**图像重定位推翻管线锚点**（§28 的结论）：锚点 oracle 天花板 8 序 **87.804%** / 24 序 **85.522%**
（零延迟天花板 88.132% / 85.896%），当前交付档 85.2206% / 81.5379%
⇒ 还差 **2.58 pp（8 序）/ 3.98 pp（24 序）**。
继续调重推参数（深度 / 门限 / ROI / 模板尺寸…）**最多只能到锚点 oracle**。

已被实测否决、**不要再试**的旋钮家族：观测层收缩 / 位移层收缩 / 低速冻结 / 位移自适应融合 /
peak_centroid / refine / 一致性门单独用 / `--replay-anchor pred` / `--fast-min-contrast` /
rolling 链 / chain 全家族 / core_frac / th_k≥2.5 / 大半径重试 / 唯一性比单门 / 模板位置-链位置分歧单门 /
链位移单门 / 峰值单门 / 二次意见采纳就弃权 / 精修失败就弃权 / dev 单门 / disp 单门 / rematch 弃权 /
xcorr_fail 弃权。

⚠️ 24 序 FAR 只剩 **0.000162** 余量 —— 任何新增机制必须**先过 FAR 再看精度**。
⚠️ `max_jitter` 在有 records 空洞的序列上是伪影，看伺服抖动请看 `diag_jitter_audit.py` 的「真实单帧位移」。
