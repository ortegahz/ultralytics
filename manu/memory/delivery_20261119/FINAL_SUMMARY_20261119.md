# trial_0080 × v3 基座 · Δn 跳帧最优参数重搜 + OSD 渲染 · 最终汇总

> 生成：2026-11-20 08:xx（host 侧汇总，数据源全部来自服务器实测产物，校验和已逐一比对）
> 服务器夜班：222 轮自动续跑，08:30 到点自停（rc=0），会话已接回 host（50122 行 / 0 坏行 / 317 turn）

---

## 一、结论一句话

**v3 是基准**（F1 94.7533%、FAR 0.020820，三版里 F1 最高且 FAR 最低）；
在此基座上重搜出的**最优跳帧参数组 `cc2_md10`**（`confirm_consec 2 / confirm_gap_tol 8 / r_slope 5.0 / match_dist 10.0 / min_hits 3`）
**逐帧 FAR 0.027859 ≤ 0.03 红线（余量 7.1%）**，跳帧臂 mean F1 **0.951398**（v3 未调只有 0.084122）；
OSD 已按该参数组渲染完成 **24/24（21 pass + 3 un-pass）**，视频体检 24/24 通过。

---

## 二、交付参数组（`cc2_md10`）

**参数档**：`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json`
（`_provenance` 内含 base / search_source / 两阶段选择说明 / 选择证据文件路径）

| 字段 | 值 | 相对 v3 的变化 |
|:--|:--|:--|
| `th_base` / `th_salvage` / `th_ground` | 0.48 / 0.06 / 0.26 | 不变（v3 基座） |
| `sky_ratio` | 1.0 | 不变（v3 基座；C++ 侧 512×1.0 ⇒ is_sky 恒真，与 Python v3 语义一致） |
| `min_hits` | 3 | 不变 |
| `match_dist` | **10.0** | 16.0 → 10.0（用户原口径 16.0；全口径 FAR 余量优先） |
| `confirm_consec` | **2** | 7 → 2 |
| `confirm_gap_tol` | **8** | 0 → 8 |
| `max_age_phys` | **null**（沿用 `max_age=3`） | — |
| `r_slope` | **5.0** | 3.5 → 5.0 |
| `alpha_decay` | 1.0 | 不变（逐帧下是 no-op，只在板端跳帧生效） |

---

## 三、全口径实测（24 序 / 165,223 帧 / GT 133,257 逐位一致）

| 臂 | F1 | FAR/帧 | Recall | Precision | FP | FN | 判定 |
|:--|--:|--:|--:|--:|--:|--:|:--|
| **v3（基准）** | **94.7533%** | **0.020820** | 92.3539% | 97.2808% | 3440 | 10189 | FAR 最好 |
| v2 | 94.6231% | 0.025208 | — | — | — | — | 次于 v3 |
| th045 | 94.4134% | 0.028991 | — | — | — | — | 三版最差 |
| gapfit（旧、用户指定那组） | 94.3755% | **0.035128** | 93.2416% | 95.5373% | 5804 | 9006 | ❌ **FAR 破 0.03 红线** |
| **cc2_md10（交付）** | **94.4976%** | **0.027859** | 92.6630% | 96.4062% | 4603 | 9777 | ✅ 守住红线（余量 7.1%） |
| cc2_md16n（候选） | 94.5804% | 0.029348 | 92.9827% | 96.2339% | 4849 | 9351 | ✅ 守线但余量仅 2.2% |

**为什么交付 md10 而不是 md16n**：跳帧臂 F1 只差 0.03pp（噪声级），但 FAR 余量 7.1% vs 2.2%（3 倍差距）⇒ 取 md10。

---

## 四、跳帧臂（Δn 仿真，确定性采样器 `zlib.crc32`）

| 口径 | F1 |
|:--|--:|
| v3 未调参 · 物理帧基 | **0.084122**（≈崩掉） |
| v3 基座 · 密集帧基 | 0.922060 |
| **cc2_md10 · 物理帧基** | **0.951398** |
| **cc2_md10 · 密集帧基** | **0.938340** |

---

## 五、与官方 OSD 记录分数的一致性（goal 原始要求）

| 版 | 我的 F1 | 官方 OSD 记录 F1 | Δ | GT |
|:--|--:|--:|--:|:--|
| v3 | 0.9475333 | 0.9474685 | +0.0065pp | 133257 = 133257 ✅ |
| v2 | 0.9462311 | 0.9461746 | +0.0057pp | 133257 = 133257 ✅ |
| th045 | 0.9441344 | 0.9439085 | +0.0226pp | 133257 = 133257 ✅ |

⇒ **三版全部对齐**，差异 ≤ 0.023pp 且 GT 逐位一致。

---

## 六、OSD 产物（服务器，视频本体留服务器）

| 目录 | pass | un-pass | 大小 | 体检 |
|:--|--:|--:|--:|:--|
| `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_20261119`（**交付**） | 21 | 3 | 5.8 GB | **24/24 通过**（tsv_mismatch 0 / group_mismatch 0） |
| `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_md16_20261120`（备选 md16n） | 21 | 3 | 5.8 GB | 24/24 通过 |
| `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_gapfit_20261119`（旧 gapfit，保留对照） | 21 | 3 | 2.1 GB | — |

目录结构与官方 v3 同构：`pass/` + `un-pass/` + `logs/` + `per_sequence_hyperparams.tsv`（含全局参数列）+ `search_json_for_render.json`。
视频名 `<SEQ>_F1-<f1>.mp4`。

---

## 七、搜索过程与两处关键修正

1. **三轮独立搜索**（run1 hashseed / run2 另一 cadence / run3 crc32 固定种子，各 ~36 min），三轮 objective-best 一致收敛到 `confirm_consec 2 / gap_tol 8 / r_slope 5`。
2. **目标函数加「逐帧口径不回退」硬约束** —— 这是旧 gapfit 破 FAR 红线的根因修法。
3. **采样器从 `hash()` 换成 `zlib.crc32` 确定性采样** —— `hash()` 有随机性，旧跳帧数值（0.952127 等）**已作废**，以重测值为准。
4. **两阶段选择**：搜索器在 4000 帧截断口径上最大化跳帧 F1（该口径 FAR 系统性低估 0.005~0.006）；**交付判定在 165,223 帧全口径上做 FAR ≤ 0.03**。搜索器 objective-best（`max_age_phys=10 / match_dist=16`）全口径 FAR 0.0352 ⇒ 被拒。

---

## 八、遗留项（板卡上电后必须做）

1. **板端需显式覆盖 8 个字段**：`th_base 0.48 / th_salvage 0.06 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 2 / confirm_gap_tol 8 / r_slope 5.0 / match_dist 10.0`。
   🔴 `mosaic_pipeline.h:1257-1258` 的注释值 `(6)/(0)` 是陈旧的，**调参别照抄头文件注释**。
2. **`max_age_phys = null`**：跳帧工况下是否需要启用（10 物理帧）需板端实测决定。
3. **`alpha_decay = 1.0`**：逐帧下是 no-op，只在板端真跳帧时起作用，需板端验证。
4. **离线逐帧 OSD 看不出 r_slope / alpha_decay 的差别**（Δn≡1），属正常。

---

## 九、证据文件

| 内容 | 路径（服务器 `/mnt/data/siping/runs/trial0080_onnx/`） |
|:--|:--|
| 最终选择（含两阶段理由） | `search/final_selection.json` |
| 确定性重测记录 | `search/final_selection.json` → `_deterministic_remeasure_20261120` |
| 搜索原始 json | `search/rt_search_t0080_v3.json` / `_final.json` / `_seedfix.json` |
| 确定性短名单 | `search/shortlist_deterministic.json` |
| 逐帧全口径复测 | `eval/track_all_v3_optgapfit.json` |
| 与官方 v3 双向对账 | `eval/cmp_rerun_delivered_vs_official_v3.json` / `cmp_rerun_delivered_vs_rebuilt_v3.json` |
| OSD 视频体检 | `eval/osd_media_check.json` / `osd_media_check_md16_candidate.json` |
| OSD 清单 | `manu/memory/delivery_20261119/osd_manifest.json` |
| 对照表 | `manu/memory/delivery_20261119/COMPARISON.md` |
| 搜索规格（完整） | `manu/memory/trial0080_delta_search_spec_20261119.md` |
| 板卡移植预设 | `manu/memory/trial0080_board_fusion_preset_20261119.md` |
| memory 记录 | `memory/memory.md` §257 十八（+ §257 十四~十六） |
