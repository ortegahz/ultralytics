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

## 结论速览（24 序 / 165,223 帧 / GT 133,257）

| 臂 | 我算 F1 | 官方 F1 | ΔF1 | FP（我/官） | FAR/帧（我 vs 官） | F1 逐位相同 |
|:--|--:|--:|--:|--:|:--|--:|
| th045 | 94.4134356% | 94.3908503% | +0.022585pp | 4742 / 4790 | 0.0287006 vs 0.0289911 | 20/24 |
| v2 | 94.6231117% | 94.6174551% | +0.005657pp | 4165 / 4165 | **完全相同** | 22/24 |
| **v3（新基准）** | **94.7533347%** | 94.7468518% | +0.006483pp | **3440 / 3440** | **完全相同** | 22/24 |

⇒ 差异全部为**检测级浮点末位差**（TP 多 12~16、FN 少 13~16）⇒ 重建缓存与官方 trial_0080 缓存等价。

## 服务器侧原始产物

- 缓存：`/mnt/data/siping/runs/trial0080_onnx/ms_cache/`（24 pkl）
- 评测：`/mnt/data/siping/runs/trial0080_onnx/eval/`
- 重建命令见 `ultralytics/manu/evaluation/run_trial0080_longquan_eval.sh` 与 `memory.md` §257 第六节
