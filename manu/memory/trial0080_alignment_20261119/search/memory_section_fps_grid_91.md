### 三十七、🔴 交付帧率网格（第 91 轮）：**「降到多少帧率，跟踪器就不能用了」——v3 基座的可用下限落在 14~17 fps 之间**

**一句话结论**：`s1`~`s4` 只回答了「实测节拍有多糟」，**没有**回答交付决策真正关心的那一个问题；本轮补上 `f25/f20/f17/f14/f12` 五档帧率后，规则与观测**逐点吻合**：**`v3` 基座在
14.10 fps 就已经 F1 精确为 0**，而 17.11 fps 尚能到 0.854244；**交付参数组 `cc2_md10` 把这个下限压到 12.11 fps 仍然全档 ≥ 0.943001**。

**业务含义**：板卡实测节拍是 8.03~8.13 fps（Δn 均值 3.08~3.11）⇒ **`v3` 基座在真实板卡上不可用**，**必须换参数组**（`cc2_md10`），而不是「板卡快一点」就能救。

#### 1) 这轮补的是哪个洞

| 已有 | 回答什么 | 没回答什么 |
| --- | --- | --- |
| `s1`~`s4`（第 89/90 轮） | 实测节拍**有多糟** | **它坏在哪个速率上** |
| 逐帧臂（Δn≡1） | 最快锚点 | 中间 12~20 fps 这一段**从未被采样过** |
| **`f25`~`f12`（本轮）** | **交付速率阶梯 + 可用下限** | — |

实测达成 fps 区间 **12.1067 ~ 23.1319**；⚠️ **「实测达成 fps」才是可引用的数字**：目标帧率是采样器的请求值，而相邻帧乱序会让引擎丢掉晚到的那一帧 ⇒
**跟踪器实际看到的速率低于请求值**（生成器已按此标定，达成 fps 残差 < 1%）。

#### 2) 帧率阶梯：F1 / FAR 随交付速率的变化

| 臂 | 口径 | 场景 | 实测达成 fps | 跳帧 F1 | 跳帧 FAR |
| --- | --- | --- | ---: | ---: | ---: |
| `2b_jump_physical_untuned` | v3 基座 · 未调参 | `f25` | 23.1319 | 0.944969 | 0.018627 |
| `2b_jump_physical_untuned` | v3 基座 · 未调参 | `f20` | 20.0779 | 0.934636 | 0.013161 |
| `2b_jump_physical_untuned` | v3 基座 · 未调参 | `f17` | 17.1141 | 0.854244 | 0.003762 |
| `2b_jump_physical_untuned` | v3 基座 · 未调参 | `f14` | 14.1043 | 0.000000 | 0.000000 |
| `2b_jump_physical_untuned` | v3 基座 · 未调参 | `f12` | 12.1067 | 0.000000 | 0.000000 |
| `3_jump_optimised` | cc2_md10（交付参数组） | `f25` | 23.1319 | 0.945263 | 0.027166 |
| `3_jump_optimised` | cc2_md10（交付参数组） | `f20` | 20.0779 | 0.945240 | 0.026267 |
| `3_jump_optimised` | cc2_md10（交付参数组） | `f17` | 17.1141 | 0.945313 | 0.025338 |
| `3_jump_optimised` | cc2_md10（交付参数组） | `f14` | 14.1043 | 0.944927 | 0.024398 |
| `3_jump_optimised` | cc2_md10（交付参数组） | `f12` | 12.1067 | 0.943001 | 0.024830 |
| `2a_jump_dense_timebase` | 密集时间基（SDK 现状口径） | `f25` | 23.1319 | 0.947148 | 0.020686 |
| `2a_jump_dense_timebase` | 密集时间基（SDK 现状口径） | `f20` | 20.0779 | 0.946308 | 0.020082 |
| `2a_jump_dense_timebase` | 密集时间基（SDK 现状口径） | `f17` | 17.1141 | 0.945541 | 0.019688 |
| `2a_jump_dense_timebase` | 密集时间基（SDK 现状口径） | `f14` | 14.1043 | 0.942694 | 0.018769 |
| `2a_jump_dense_timebase` | 密集时间基（SDK 现状口径） | `f12` | 12.1067 | 0.941190 | 0.017782 |

（跳帧 F1 / 跳帧 FAR：物理帧基、逐序列 micro-pool；FAR 红线 0.03）

**锚点（逐帧，Δn≡1）**：F1=0.947406、FAR=0.021034。

#### 3) 🔴 机制被中间量**证实**（不再是假设）

门限语义已**读码**确认：`manu/evaluation/system_sota_longquan.py` 的 `apply_consecutive_confirmation()` ——
一条航迹的**真实观测帧**按「与前一观测帧的物理帧差 ≤ `max_gap = 1 + confirm_gap_tol`」切成游程，**最长游程 < `confirm_consec` 的整条航迹被丢弃**。

| 参数组 | `confirm_consec` | `confirm_gap_tol` | `max_gap` | 连续性要求 |
| --- | ---: | ---: | ---: | --- |
| `v3` 基座 | 7 | 0 | 1 | 需 7 次观测且**相机帧相邻（Δn=1）** |
| `cc2_md10`（交付参数组） | 2 | 8 | 9 | 2 次观测、间隔 ≤ 9 帧即可 |

⇒ 于是规则是**可证伪的**：若某场景的 Δn=1 最长连长 < 7，则 `v3` 在该场景的 F1 **必须精确为 0**；否则必须 > 0。

#### 4) 机制量化表：规则 → 观测，逐点吻合

聚合口径：`p(Δn=1)` 取 3 seeds 均值；`maxRun` 与 `#run≥k` 取 3 seeds 的**最大值**（保守：只要有一个 seed 存在这样的游程，规则就不允许 F1 恒为 0）。

**(a) `max_gap=1` 侧（`v3` 的判据）**

| 场景 | 来源 | p(Δn=1) | maxRun@1 | #run≥7@1 | v3 F1 | 规则预测 v3≡0 | 观测 | 吻合 |
| --- | --- | ---: | ---: | ---: | ---: | :-: | :-: | :-: |
| `f25` | fps 网格（合成节拍） | 0.9199 | 127 | 92911 | 0.944969 | 否 | 否 | ✅ |
| `f20` | fps 网格（合成节拍） | 0.7837 | 26 | 29543 | 0.934636 | 否 | 否 | ✅ |
| `f17` | fps 网格（合成节拍） | 0.6078 | 11 | 2103 | 0.854244 | 否 | 否 | ✅ |
| `f14` | fps 网格（合成节拍） | 0.3897 | 6 | 0 | 0.000000 | 是 | 是 | ✅ |
| `f12` | fps 网格（合成节拍） | 0.2263 | 5 | 0 | 0.000000 | 是 | 是 | ✅ |
| `s1` | 实测节拍（板卡重建） | 0.0202 | 4 | 0 | 0.000000 | 是 | 是 | ✅ |
| `s2` | 实测节拍（板卡重建） | 0.0107 | 3 | 0 | 0.000000 | 是 | 是 | ✅ |
| `s3` | 实测节拍（板卡重建） | 0.0045 | 2 | 0 | 0.000000 | 是 | 是 | ✅ |
| `s4` | 实测节拍（板卡重建） | 0.3238 | 13 | 112 | 0.336489 | 否 | 否 | ✅ |

**(b) `max_gap=9` 侧（`cc2_md10` 的判据）**

| 场景 | maxRun@9 | #run≥2@9 | 间隙总数 | 覆盖率 | cc2_md10 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f25` | 153110 | 153109 | 153110 | 100.00% | 0.945263 |
| `f20` | 133221 | 133220 | 133221 | 100.00% | 0.945240 |
| `f17` | 113212 | 113211 | 113212 | 100.00% | 0.945313 |
| `f14` | 93193 | 93192 | 93193 | 100.00% | 0.944927 |
| `f12` | 79906 | 79905 | 79906 | 100.00% | 0.943001 |
| `s1` | 5711 | 49210 | 49258 | 99.90% | 0.955383 |
| `s2` | 158 | 30167 | 31971 | 94.36% | 0.953008 |
| `s3` | 46 | 18592 | 23200 | 80.14% | 0.938916 |
| `s4` | 85107 | 85106 | 85107 | 100.00% | 0.958285 |

覆盖率 = `#run≥2@9` / 间隙总数（能形成「≥2 连」游程的间隙占比）。

**吻合 9/9**，其中 **5 个场景 v3 的 F1 精确为 0**（`f14`, `f12`, `s1`, `s2`, `s3`）——规则预测的「零」与观测到的「零」**逐点重合**，不是「相关性不错」。

⚠️ **但不要把 `#run≥7` 线性换算成 F1**：`maxRun`/`#run` 是**全图节拍**的统计量，而一条航迹只在它**被观测到的子区间**里才有观测，所以全图连长是每条航迹连长的**上界**；cc2_md10
那一侧同样非单调（例如 `s3` 的 `#run≥2` 覆盖率明显低于 `f25`，但 F1 仍高达 0.938916）。

#### 5) 交叉验证：两个评测器互证

本轮服务器运行**独立复现**了交付口径（`delivery_caliber.selected`），与第 90 轮已发布的 `cc2_md10` 对账：

| 口径 | F1 | FAR |
| --- | ---: | ---: |
| 本轮运行（`rt_fps_grid_r90.json` → `delivery_caliber.selected`） | 0.944994 | 0.027825 |
| 第 90 轮已发布（`rt_delivery_compare.csv` → `cc2_md10`） | 0.944976 | 0.027859 |
| 差 | +0.0018 pp | 0.000034（取绝对值） |

⇒ 两套独立的评测实现互证一致（F1 差 0.0018 pp、FAR 差 0.000034）。

#### 6) 业务含义

1. **`v3` 基座的可用下限在 14~17 fps 之间**：`f17` 档 （17.1141 fps）F1 尚有 0.854244，到 `f14` 档（14.1043 fps）**精确为 0**。
2. **板卡实测节拍 8.03~8.13 fps 远在下限之下** ⇒ `v3` 基座在真实板卡上**不可用**。
3. **交付参数组 `cc2_md10` 把下限压到 ≤ 12.11 fps**（全档 F1 ≥ 0.943001、FAR ≤ 0.027166，红线 0.03）⇒ **要救的是参数组，不是板卡速度**。
4. 交付口径自检同时给出 F1 下限 0.944406 与 FAR 上限 0.03，本次 `feasible=True`（`f1_ok=True`、`far_ok=True`）。

#### 7) 局限（必须和结论一起读）

1. **`f` 网格固定抖动形状、只改速率**：它把实测节拍的**抖动形状**按目标均值缩放，**与实测 `s1` 不是同一条生成链** ⇒ **`s1` 不是 `f` 网格的延续**，两条链只能并列陈述、不能互相插值。
2. **`f25` 只能到 23.1319 fps**：间隙下限是 1 帧，再往上采样不出来；逐帧锚点仍以 Δn≡1 的口径单列。
3. **本轮不涉及任何板端复核**：全部数字来自服务器侧离线评测；板端不在本轮范围内。

#### 8) 产物 / 命令 / 执行机

| 项 | 值 |
| --- | --- |
| 执行机（重活） | 服务器 `192.168.99.40:32222`（`huangzhe@`，conda env `uav`） |
| 执行机（报告 + 落档） | 本机（纯读产物，无 GPU / 无网络 / 无重活） |
| 服务器运行窗口 | `[r90] start 2026-11-20 13:10:50` → `[r90] exit=0 end 2026-11-20 13:18:24`（`elapsed_s=392.1`） |
| 帧率网格脚本 | `manu/evaluation/rt_fps_jitter_search.py`（新增场景类型 `target_fps`） |
| 网格报告脚本 | `manu/evaluation/rt_fps_grid_report.py`（import 复用 `rt_grid_compare.py` 的口径） |
| 节拍分析脚本 | `/mnt/data/siping/runs/trial0080_onnx/analyse_runs_r90.py` |
| 本轮产物 | `delivery_20261119/artifacts/rt_fps_grid_r90.json` |
| 本轮产物（续） | `delivery_20261119/artifacts/rt_fps_grid_r90.log`、`analyse_runs_r90.csv` |
| 报告产物 | `delivery_20261119/grid_compare/rt_fps_grid.md` / `.csv` |
| 落档生成器 | `manu/memory/trial0080_alignment_20261119/search/make_memory_section_fps_grid_91.py` |

服务器运行命令（帧率网格）：

```bash
cd /mnt/data/siping/runs/npu20ms/nano_search/repo
export PYTHONPATH=/mnt/data/siping/ptq_work/ortgpu:/mnt/data/siping/runs/npu20ms/nano_search/repo
CACHE=/mnt/data/siping/runs/trial0080_onnx/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
timeout 14400 ~/anaconda3/envs/uav/bin/python manu/evaluation/rt_fps_jitter_search.py \
  --cache-root "$CACHE" --labels-root "$GT" --all-seqs \
  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  --opt-params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
  --phase eval --scenarios f25,f20,f17,f14,f12 --seeds 0,1,2 --workers 8 \
  --out-json /mnt/data/siping/runs/trial0080_onnx/search/rt_fps_grid_r90.json
```

本机命令（报告 + 落档）：

```bash
python manu/evaluation/rt_fps_grid_report.py \
  --arm-json manu/memory/delivery_20261119/artifacts/rt_fps_grid_r90.json \
  --out-dir manu/memory/delivery_20261119/grid_compare
python manu/memory/trial0080_alignment_20261119/search/make_memory_section_fps_grid_91.py
```

