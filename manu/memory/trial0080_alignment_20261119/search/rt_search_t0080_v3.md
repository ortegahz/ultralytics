# trial_0080 · Δn（跳帧）搜索报告

> 本文件由 `--out-md` 从**同名 json** 自动生成；**数字以 json 为准**。
> ⚠️ 本文件是**某一次跑批的原始记录**，**跳帧数字不是交付口径**：若该批次产于采样器修复前的 `hash()` 采样器、或只是 objective-best 阶段，其跳帧值**已作废**。交付口径见 `artifacts/shortlist_deterministic.json` 与 `artifacts/rt_eval_delivered_armA.json`。

## 1. 运行元信息

| 项 | 值 |
| --- | --- |
| cache_root | `/mnt/data/siping/runs/trial0080_onnx/ms_cache` |
| labels_root | `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| 序列数 | 24 |
| 情景 | s1, s2, s3, s4 |
| seeds | 0, 1, 2 |
| max_frames | 4000 |
| 耗时 | 2212.4 s |

## 2. 各臂结果

| 臂 | 口径 | mean F1 | worst F1 | mean FAR |
| --- | --- | --- | --- | --- |
| 1_perframe_ceiling | 逐帧 | 0.960186 | - | 0.017558 |
| 2a_jump_dense_timebase | 跳帧 | 0.921253 | 0.874589 | 0.013713 |
| 2b_jump_physical_untuned | 跳帧 | 0.106058 | 0.000000 | 0.000018 |
| 3_jump_optimised | 跳帧 | 0.110650 | 0.000000 | 0.000030 |
| 5_optimised_perframe | 逐帧（截断口径） | 0.959301 | - | 0.021774 |

## 3. 跳帧臂逐情景（截断口径）

| 臂 | 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 2a_jump_dense_timebase | s1 | 0.942860 | 0.012263 | 0.904670 | 0.984418 |
| 2a_jump_dense_timebase | s2 | 0.911584 | 0.011725 | 0.848998 | 0.984133 |
| 2a_jump_dense_timebase | s3 | 0.874589 | 0.015317 | 0.791020 | 0.977902 |
| 2a_jump_dense_timebase | s4 | 0.955980 | 0.015548 | 0.932299 | 0.980895 |
| 2b_jump_physical_untuned | s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s4 | 0.424232 | 0.000073 | 0.269245 | 0.999685 |
| 3_jump_optimised | s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3_jump_optimised | s2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3_jump_optimised | s3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3_jump_optimised | s4 | 0.442602 | 0.000118 | 0.284232 | 0.999515 |

## 4. 逐帧天花板（逐情景）

| 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- |
| s1 | 0.960331 | 0.017470 | 0.942534 | 0.978813 |
| s2 | 0.960034 | 0.017244 | 0.941724 | 0.979070 |
| s3 | 0.960373 | 0.017127 | 0.942230 | 0.979228 |
| s4 | 0.959998 | 0.017501 | 0.941940 | 0.978762 |

## 5. 注入节奏（cadence）

| 情景 | fps | mean Δn | updates | Δn 直方图 |
| --- | --- | --- | --- | --- |
| s1 | 7.3349 | 3.4113 | 1173 | 1:24 2:191 3:555 4:247 5:66 6:41 7:29 8:14 9:2 10:3 |
| s2 | 4.9662 | 5.0366 | 793 | 1:9 2:39 3:151 4:194 5:144 6:102 7:57 8:34 9:27 10:11 11:5 12:11 13:2 14:1 15:3 21:1 24:1 |
| s3 | 3.3971 | 7.3727 | 543 | 1:2 2:21 3:36 4:58 5:72 6:74 7:68 8:45 9:41 10:28 11:27 12:21 13:13 14:6 15:7 16:7 17:7 18:2 19:3 21:2 23:1 27:1 |
| s4 | 13.0845 | 1.9116 | 2093 | 1:703 2:985 3:320 4:57 5:24 6:3 |

## 6. 候选参数（opt_params）

```json
{
  "th_base": 0.48,
  "th_salvage": 0.06,
  "th_ground": 0.26,
  "sky_ratio": 1.0,
  "min_hits": 3,
  "max_age": 3,
  "match_dist": 16.0,
  "max_match_dist": 24.0,
  "instant_conf": 0.25,
  "min_disp": 2.5,
  "min_track_score": 0.08,
  "stitch_gap": 6,
  "infill_gap": 3,
  "min_hits_infill": 5,
  "min_rigid_disp": 2.0,
  "max_rigid_var": 1.0,
  "min_hits_prune": 4,
  "dedup_radius": 768.0,
  "th_deep": 0.0,
  "min_hits_deep": 3,
  "min_rigid_disp_sky": -1.0,
  "hover_vel": 0.0,
  "hover_infill_gap": 15,
  "coast_frames": 0,
  "min_hits_hover": 8,
  "hover_sky_only": 1,
  "stitch_long_gap": 0,
  "stitch_vel_diff": 4.0,
  "confirm_consec": 7,
  "confirm_gap_tol": 0,
  "confirm_trim_prefix": 1,
  "persist_min_hits": 0,
  "persist_window": 0,
  "persist_tol": 4.0,
  "max_age_phys": 18,
  "r_slope": 5.0,
  "alpha_decay": 1.0
}
```

## 7. 搜索轨迹：57 个候选（objective 全量落在 json 的 `search_history` 字段里）

