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
| 耗时 | 2163.0 s |

## 2. 各臂结果

| 臂 | 口径 | mean F1 | worst F1 | mean FAR |
| --- | --- | --- | --- | --- |
| 1_perframe_ceiling | 逐帧 | 0.960186 | - | 0.017558 |
| 2a_jump_dense_timebase | 跳帧 | 0.922890 | 0.875480 | 0.014491 |
| 2b_jump_physical_untuned | 跳帧 | 0.112741 | 0.000000 | 0.000034 |
| 3_jump_optimised | 跳帧 | 0.952784 | 0.942840 | 0.016982 |
| 5_optimised_perframe | 逐帧（截断口径） | 0.957343 | - | 0.029013 |

## 3. 跳帧臂逐情景（截断口径）

| 臂 | 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 2a_jump_dense_timebase | s1 | 0.945468 | 0.011596 | 0.908720 | 0.985313 |
| 2a_jump_dense_timebase | s2 | 0.913921 | 0.013811 | 0.855064 | 0.981478 |
| 2a_jump_dense_timebase | s3 | 0.875480 | 0.016609 | 0.793631 | 0.976154 |
| 2a_jump_dense_timebase | s4 | 0.956691 | 0.015948 | 0.934065 | 0.980440 |
| 2b_jump_physical_untuned | s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s4 | 0.450963 | 0.000138 | 0.291172 | 0.999448 |
| 3_jump_optimised | s1 | 0.956862 | 0.017942 | 0.936516 | 0.978112 |
| 3_jump_optimised | s2 | 0.953808 | 0.016031 | 0.928771 | 0.980233 |
| 3_jump_optimised | s3 | 0.942840 | 0.010873 | 0.903181 | 0.986143 |
| 3_jump_optimised | s4 | 0.957625 | 0.023082 | 0.943474 | 0.972208 |

## 4. 逐帧天花板（逐情景）

| 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- |
| s1 | 0.960405 | 0.017545 | 0.942760 | 0.978724 |
| s2 | 0.959852 | 0.017856 | 0.942054 | 0.978335 |
| s3 | 0.960298 | 0.017143 | 0.942111 | 0.979202 |
| s4 | 0.959920 | 0.017509 | 0.941811 | 0.978739 |

## 5. 注入节奏（cadence）

| 情景 | fps | mean Δn | updates | Δn 直方图 |
| --- | --- | --- | --- | --- |
| s1 | 7.4037 | 3.3795 | 1184 | 1:21 2:199 3:563 4:242 5:71 6:54 7:23 8:8 9:2 |
| s2 | 4.8236 | 5.1896 | 771 | 1:10 2:41 3:110 4:180 5:144 6:125 7:65 8:34 9:25 10:7 11:10 12:11 13:1 14:2 15:2 16:2 17:1 |
| s3 | 3.433 | 7.2956 | 549 | 1:1 2:20 3:32 4:73 5:84 6:70 7:67 8:52 9:33 10:26 11:19 12:14 13:18 14:10 15:7 16:4 17:4 18:2 19:5 20:3 23:2 29:1 33:1 |
| s4 | 12.897 | 1.9394 | 2063 | 1:655 2:988 3:336 4:58 5:22 6:3 |

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
  "confirm_consec": 2,
  "confirm_gap_tol": 8,
  "confirm_trim_prefix": 1,
  "persist_min_hits": 0,
  "persist_window": 0,
  "persist_tol": 4.0,
  "max_age_phys": 10,
  "r_slope": 5.0,
  "alpha_decay": 1.0
}
```

## 7. 搜索轨迹：57 个候选（objective 全量落在 json 的 `search_history` 字段里）

