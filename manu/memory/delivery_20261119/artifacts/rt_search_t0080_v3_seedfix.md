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
| 耗时 | 2132.4 s |

## 2. 各臂结果

| 臂 | 口径 | mean F1 | worst F1 | mean FAR |
| --- | --- | --- | --- | --- |
| 1_perframe_ceiling | 逐帧 | 0.960186 | - | 0.017558 |
| 2a_jump_dense_timebase | 跳帧 | 0.922060 | 0.874592 | 0.013286 |
| 2b_jump_physical_untuned | 跳帧 | 0.084122 | 0.000000 | 0.000025 |
| 3_jump_optimised | 跳帧 | 0.952186 | 0.941208 | 0.016564 |
| 5_optimised_perframe | 逐帧（截断口径） | 0.957343 | - | 0.029013 |

## 3. 跳帧臂逐情景（截断口径）

| 臂 | 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 2a_jump_dense_timebase | s1 | 0.944339 | 0.012540 | 0.907641 | 0.984129 |
| 2a_jump_dense_timebase | s2 | 0.912450 | 0.011030 | 0.849798 | 0.985075 |
| 2a_jump_dense_timebase | s3 | 0.874592 | 0.015497 | 0.791175 | 0.977673 |
| 2a_jump_dense_timebase | s4 | 0.956861 | 0.014078 | 0.932369 | 0.982673 |
| 2b_jump_physical_untuned | s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2b_jump_physical_untuned | s4 | 0.336489 | 0.000101 | 0.202300 | 0.999415 |
| 3_jump_optimised | s1 | 0.955932 | 0.019253 | 0.936160 | 0.976558 |
| 3_jump_optimised | s2 | 0.953736 | 0.013577 | 0.926010 | 0.983173 |
| 3_jump_optimised | s3 | 0.941208 | 0.011176 | 0.900527 | 0.985738 |
| 3_jump_optimised | s4 | 0.957866 | 0.022250 | 0.943021 | 0.973186 |

## 4. 逐帧天花板（逐情景）

| 情景 | F1 | FAR | Recall | Precision |
| --- | --- | --- | --- | --- |
| s1 | 0.960202 | 0.017452 | 0.942262 | 0.978839 |
| s2 | 0.960604 | 0.017311 | 0.942870 | 0.979017 |
| s3 | 0.960072 | 0.018131 | 0.942726 | 0.978069 |
| s4 | 0.959945 | 0.017800 | 0.942160 | 0.978414 |

## 5. 注入节奏（cadence）

| 情景 | fps | mean Δn | updates | Δn 直方图 |
| --- | --- | --- | --- | --- |
| s1 | 7.4475 | 3.3597 | 1191 | 1:21 2:201 3:590 4:231 5:69 6:35 7:28 8:7 9:6 10:2 |
| s2 | 4.7848 | 5.2317 | 765 | 1:4 2:34 3:103 4:186 5:168 6:117 7:53 8:30 9:21 10:22 11:13 12:7 13:4 14:1 17:1 |
| s3 | 3.5759 | 7.0035 | 571 | 1:2 2:28 3:36 4:82 5:85 6:85 7:49 8:50 9:39 10:34 11:20 12:14 13:13 14:6 15:7 16:7 17:4 18:2 20:1 21:1 22:2 23:1 25:1 34:1 |
| s4 | 12.822 | 1.9507 | 2051 | 1:687 2:900 3:379 4:48 5:33 6:3 |

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

