# trial_0080 底座对齐产物（2026-11-19）

本目录是 **trial_0080**（= 交付嵌入式 rknn `p0_off_A_none.rknn` / `ir_uav_det.rknn`，md5 `1a3faf2f9459cad8a389ffa81810234c`）
在龙泉山 24 序数据集上的**逐序列对账产物**。详见 `ultralytics/memory/memory.md` **§257**。

## 文件

| 文件 | 说明 |
|:--|:--|
| `cmp_th045.json` / `cmp_v2.json` / `cmp_v3.json` | **本轮核心**：我算的结果 vs 2026-10-22 官方记录的逐序列对账（`overall_*` 汇总 + `per_sequence` 24 行明细 + `missing`） |
| `track_all_th045.json` / `track_all_v2.json` / `track_all_v3.json` | 三个工作点在 24 序上的**全量评测输出**（本轮新算） |
| `th045_search_json_for_render.json` / `v2_...` / `v3_...` | **官方原始记录**（2026-10-22，`osd_sys_sota_trial0080_{th045,v2,v3}_20261022`），逐序列官方指标 |
| `self_check.log` | `--self-check` 闸门输出（**GATE FAIL 属陈旧常量**，详见 §257 第八节） |
| `track_all_v3_gapfit.json` | **gapfit 臂**（v3 + 跳帧适配参数组）全量评测输出 |
| `cmp_v3_gapfit_vs_official_v3.json` | gapfit vs **官方 v3 记录** 逐序列对账 |
| `cmp_v3_gapfit_vs_rebuilt_v3.json` | gapfit vs **重建缓存上的 v3** 逐序列对账（同一份缓存 ⇒ 纯参数效应） |

## 结论速览（24 序 / 165,223 帧 / GT 133,257）

| 臂 | 我算 F1 | 官方 F1 | ΔF1 | FP（我/官） | FAR/帧（我 vs 官） | F1 逐位相同 |
|:--|--:|--:|--:|--:|:--|--:|
| th045 | 94.4134356% | 94.3908503% | +0.022585pp | 4742 / 4790 | 0.0287006 vs 0.0289911 | 20/24 |
| v2 | 94.6231117% | 94.6174551% | +0.005657pp | 4165 / 4165 | **完全相同** | 22/24 |
| **v3（新基准）** | **94.7533347%** | 94.7468518% | +0.006483pp | **3440 / 3440** | **完全相同** | 22/24 |

⇒ 差异全部为**检测级浮点末位差**（TP 多 12~16、FN 少 13~16）⇒ 重建缓存与官方 trial_0080 缓存等价。

## gapfit 臂（跳帧适配参数组）：25fps 逐帧口径下**超 FAR 红线**

参数 = v3 + `confirm_consec 7→2` / `confirm_gap_tol 0→6` / `max_age_phys=10` / `r_slope=3.5` / `alpha_decay=1.0` / `match_dist=16.0`

| 指标 | gapfit | v3（同缓存） | Δ |
|:--|--:|--:|--:|
| F1 | 94.3755% | 94.7533% | −0.3778pp |
| Recall | 93.2416% | 92.3539% | +0.8878pp |
| Precision | 95.5373% | 97.2808% | −1.7435pp |
| FP | 5804 | 3440 | +2364（+68.7%） |
| FN | 9006 | 10189 | −1183 |
| FAR/帧 | **0.035128** | 0.020820 | **+68.7% ⇒ 超 0.03 交付红线** |

⇒ 这组参数是给板端 **Δn≈3 跳帧工况**设计的，在 Δn≡1 下「允许断 6 帧」变成「允许乱认」⇒ 虚警暴涨。
⇒ **必须在仿真 Δn 分布下评测才有意义**；Δn 搜索网格要把 `confirm_consec` / `confirm_gap_tol` / `max_age_phys` 当一等公民。

新 OSD 目录（服务器）：`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_gapfit_20261119/`（pass 21 / un-pass 3）

## 服务器侧原始产物

- 缓存：`/mnt/data/siping/runs/trial0080_onnx/ms_cache/`（24 pkl）
- 评测：`/mnt/data/siping/runs/trial0080_onnx/eval/`
- 重建命令见 `ultralytics/manu/evaluation/run_trial0080_longquan_eval.sh` 与 `memory.md` §257 第六节
