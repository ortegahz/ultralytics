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
| 耗时 | 152.0 s |

## 2. 各臂结果

| 臂 | 口径 | mean F1 | worst F1 | mean FAR |
| --- | --- | --- | --- | --- |
| 1_perframe_ceiling | 逐帧 | 0.958640 | - | 0.022399 |
| 2a_jump_dense_timebase | 跳帧 | 0.937438 | 0.912085 | 0.037930 |
| 2b_jump_physical_untuned | 跳帧 | 0.952127 | 0.941359 | 0.016176 |
| 3_jump_optimised | 跳帧 | 0.952127 | 0.941359 | 0.016176 |
| 5_optimised_perframe | 逐帧（截断口径） | 0.958640 | - | 0.022399 |

## 3. 跳帧臂逐情景（截断口径）

| 臂 | 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 2a_jump_dense_timebase | s1 | 0.950023 | 0.021842 | 0.927850 | 0.973281 |
| 2a_jump_dense_timebase | s2 | 0.930631 | 0.041868 | 0.912797 | 0.949175 |
| 2a_jump_dense_timebase | s3 | 0.912085 | 0.067121 | 0.904138 | 0.920173 |
| 2a_jump_dense_timebase | s4 | 0.957013 | 0.020889 | 0.939953 | 0.974703 |
| 2b_jump_physical_untuned | s1 | 0.955844 | 0.018325 | 0.934986 | 0.977654 |
| 2b_jump_physical_untuned | s2 | 0.953874 | 0.013078 | 0.925735 | 0.983776 |
| 2b_jump_physical_untuned | s3 | 0.941359 | 0.011870 | 0.901549 | 0.984847 |
| 2b_jump_physical_untuned | s4 | 0.957433 | 0.021430 | 0.941325 | 0.974102 |
| 3_jump_optimised | s1 | 0.955844 | 0.018325 | 0.934986 | 0.977654 |
| 3_jump_optimised | s2 | 0.953874 | 0.013078 | 0.925735 | 0.983776 |
| 3_jump_optimised | s3 | 0.941359 | 0.011870 | 0.901549 | 0.984847 |
| 3_jump_optimised | s4 | 0.957433 | 0.021430 | 0.941325 | 0.974102 |

## 4. 逐帧天花板（逐情景）

| 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- |
| s1 | 0.958985 | 0.022110 | 0.944954 | 0.973438 |
| s2 | 0.958508 | 0.022485 | 0.944481 | 0.972959 |
| s3 | 0.958206 | 0.022848 | 0.944322 | 0.972504 |
| s4 | 0.958547 | 0.022278 | 0.944340 | 0.973188 |

## 5. 注入节奏（cadence）

| 情景 | fps | mean Δn | updates | Δn 直方图 |
| --- | --- | --- | --- | --- |
| s1 | 7.4012 | 3.3807 | 1183 | 1:16 2:204 3:584 4:220 5:71 6:48 7:26 8:8 9:2 10:1 11:1 12:1 |
| s2 | 4.6554 | 5.3774 | 743 | 1:8 2:35 3:116 4:152 5:141 6:106 7:69 8:35 9:26 10:26 11:13 12:5 13:3 14:1 15:2 17:2 18:1 20:1 |
| s3 | 3.6135 | 6.9306 | 577 | 1:6 2:19 3:50 4:82 5:75 6:86 7:61 8:54 9:37 10:27 11:23 12:11 13:9 14:12 15:2 16:6 17:2 18:5 19:2 20:2 21:2 22:1 26:1 29:1 |
| s4 | 12.5844 | 1.9876 | 2013 | 1:604 2:954 3:359 4:68 5:24 6:3 |

## 6. 候选参数（opt_params）

```json
{
  "th_base": 0.48,
  "th_salvage": 0.06,
  "th_ground": 0.26,
  "sky_ratio": 1.0,
  "min_hits": 3,
  "max_age": 3,
  "match_dist": 10.0,
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
  "max_age_phys": null,
  "r_slope": 5.0,
  "alpha_decay": 1.0
}
```

