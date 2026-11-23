# trial_0080 · Δn（跳帧）搜索报告

> 本文件由 `--out-md` 从**同名 json** 自动生成；**数字以 json 为准**。该批次是否为**作废批次**，以交付包 `artifacts/README.md` 的标注为准。

## 1. 运行元信息

| 项 | 值 |
| --- | --- |
| cache_root | `/mnt/data/siping/runs/trial0080_onnx/ms_cache` |
| labels_root | `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| 序列数 | 2 |
| 情景 | s1 |
| seeds | 0 |
| max_frames | 200 |
| 耗时 | 0.5 s |

## 2. 各臂结果

| 臂 | 口径 | mean F1 | worst F1 | mean FAR |
| --- | --- | --- | --- | --- |
| 1_perframe_ceiling | 逐帧 | 0.993711 | - | 0.000000 |
| 2a_jump_dense_timebase | 跳帧 | 0.964912 | 0.964912 | 0.000000 |
| 2b_jump_physical_untuned | 跳帧 | 0.000000 | 0.000000 | 0.000000 |
| 3_jump_optimised | 跳帧 | 0.000000 | 0.000000 | 0.000000 |
| 5_optimised_perframe | 逐帧（截断口径） | 0.993711 | - | 0.000000 |

## 3. 跳帧臂逐情景（截断口径）

| 臂 | 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 2a_jump_dense_timebase | s1 | 0.964912 | 0.000000 | 0.932203 | 1.000000 |
| 2b_jump_physical_untuned | s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3_jump_optimised | s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 4. 逐帧天花板（逐情景）

| 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- |
| s1 | 0.991453 | 0.000000 | 0.983051 | 1.000000 |

## 5. 注入节奏（cadence）

| 情景 | fps | mean Δn | updates | Δn 直方图 |
| --- | --- | --- | --- | --- |
| s1 | 7.4495 | 3.4138 | 59 | 1:1 2:7 3:32 4:9 5:4 6:4 7:1 |

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
  "max_age_phys": null,
  "r_slope": 0.0,
  "alpha_decay": 1.0
}
```

