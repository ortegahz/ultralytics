# trial_0080 板端验证报告（2026-11-20）

**板卡**：`root@192.168.0.64`（ematech，Ubuntu 20.04.6 focal / glibc 2.31 / 8 核 / 7.9 GB）
**模型**：`p0_off_A_none.rknn`，md5 `1a3faf2f9459cad8a389ffa81810234c`（= 交付嵌入式用 **trial_0080**）✅
**序列**：`VIDEO00007_19700101_001324`（11810 帧，GT 10535）
**施工单**：`manu/memory/trial0080_board_fusion_preset_20261119.md`
**交付参数档**：`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json`
（`params` 块 md5 `a1157e013ebb496731be805c72c95dd8` ✅ 与施工单一致）

---

## 一、默认值已按交付组落到板端（运行时实测回读）

改的是 `rk3588_app_base/mosaic_sdk/src/npu/mosaic_track_fusion.h`（`TrackFusionParams` 默认值）
与 `mosaic_pipeline.h`（陈旧注释），备份 `/tmp/mosaic_track_fusion.h.bak_20261120`、
`/tmp/mosaic_pipeline.h.bak_20261120`。

| 字段 | 板端回读（运行时日志） | 交付档 | 一致 |
|:--|--:|--:|:--:|
| `th_base` | 0.4800 | 0.48 | ✅ |
| `th_ground` | 0.2600 | 0.26 | ✅ |
| `sky_ratio` | 1.000 | 1.0 | ✅ |
| `match_dist` | 10.00 | 10.0 | ✅ |
| `max_age_phys` | **-1** | `null`（不设） | ✅（负数=关闭走 legacy，见 §三） |
| `r_slope` | 5.00 | 5.0 | ✅ |
| `confirm_consec` | 2 | 2 | ✅ |
| `confirm_gap_tol` | 8 | 8 | ✅ |
| `alpha_decay` | 1.000 | 1.0 | ✅ |
| `th_salvage` / `max_age` / `min_hits` / `max_match_dist` | 0.06 / 3 / 3 / 24 | 同 | ✅ |

> 板端日志原文：
> `track fusion ON: th_base=0.4800 th_ground=0.2600 sky_ratio=1.000 tags=0x7 reorder_window=0 physical_frame_id=1 max_age_phys=-1 r_slope=5.00 alpha_decay=1.000 confirm_consec=2 confirm_gap_tol=8`

### 1.1 交付档 34 个参数 → C++ 逐字段核对

| 情况 | 参数 | 说明 |
|:--|:--|:--|
| 逐值一致 | `th_base/th_salvage/th_ground/sky_ratio/min_hits/max_age/match_dist/max_match_dist/instant_conf/min_track_score/stitch_gap/infill_gap/min_hits_infill/min_rigid_disp/max_rigid_var/min_hits_prune/th_deep/min_hits_deep/min_rigid_disp_sky/confirm_consec/confirm_gap_tol/confirm_trim_prefix/r_slope/alpha_decay` | 26 项逐值相同 |
| 改名 | `min_disp` → C++ `min_displacement` | 同义 |
| 交付档无 C++ 同名字段、但**在交付值下是无操作分支** | `hover_vel=0.0`（`hover_vel_thresh>0` 恒假）、`hover_infill_gap=15`、`min_hits_hover=8`、`hover_sky_only=1`、`coast_frames=0`（`>0` 恒假）、`stitch_long_gap=0`（`max(0,6)=6` ⇒ `>stitch_max_gap` 恒假 ⇒ 其内部的 `stitch_vel_diff` 也从不生效）、`persist_min_hits=0`/`persist_window=0`（`if not persist_min_hits: return`） | 交付档下**不产生行为差异** |
| 交付档有、C++ 无（🔴 **本行已于 2026-11-20 更正，原结论是错的**） | `dedup_radius=768.0` | 属**候选构建**（多尺度合并半径）参数，**SDK 里根本没有这一步**（原写「在 SDK 的后处理阶段完成」是**错的**；SDK 后处理里没有任何候选级合并配置）⇒ 板端把**未合并**的候选流喂给融合引擎 ⇒ 同一目标多尺度各出一峰 ⇒ 裂成多条航迹 ⇒ 逐帧 FAR 0.3832（红线 12.8 倍）。**根因与修法见 `manu/memory/trial0080_board_gt_scoring_20261120.md` §四 与 `manu/memory/delivery_20261119/trial0080_board_fusion_preset_20261119.md` §2.0.1** |
| **有意不同** | `max_age_phys`: 交付档 `null` ⇒ C++ **-1** | 施工单 §2.1 明令：**「不设」必须是负数**，写 0 会让航迹下一帧即死；`ToFusionParams` 钳制 `[-1,100000]` ⇒ -1 可透传 ✅ |

---

## 二、板端 cadence（检测器 → 跟踪器的更新间隔）

板卡用 `mosaic_cadence_probe` 实测，源固定 25 fps、`npu_workers=1`（2 个 NPU context 会触发
RKNN 6 s 超时 → NPU soft reset，只能物理断电恢复）。

| 项 | 短测（600 源帧） | **长测（4500 源帧）** | 仿真 s1 基线 | 旧记录 |
|:--|--:|--:|--:|--:|
| 交付帧数 | 194 | **1422** | — | — |
| 有效输出 fps | 7.919 | **7.853** | 7.4 | 8.03~8.13 |
| 间隔均值 ms | 126.3 | **127.3** | — | 123 |
| 间隔 p50 / p95 / p99 / max ms | 118.6 / 171.0 / 203.6 / 216.9 | 119.4 / 181.4 / 262.0 / 282.7 | — | — |
| **Δn 均值** | 3.137 | **3.166~3.182** | **3.38** | 3.08~3.11 |
| Δn 直方图（长测） | 1:3 2:30 3:74 4:36 5:9 6:1 | **1:37 2:267 3:732 4:264 5:71 6:30 7:19 8:1** | — | — |
| rknn_run p50 / p95 ms | 117.0 / 120.1 | 117.7 / 121.2 | — | — |
| gmc_fit p50 / p95 ms | 31.4 / 142.6 | 49.3 / 143.2 | — | — |
| post（稀疏后处理）均值 / p99 / max ms | 0.68 / 1.93 / 2.05 | 0.85 / 5.33 / 15.47 | — | — |

⇒ **Δn ≈ 3.17 与仿真 s1（3.38）、旧实测（3.08~3.11）同量级**，交付参数档的 `confirm_gap_tol=8`
（= 允许断 8 个物理帧）在该节奏下有充足余量 ⇒ **施工单 §4 第 2 条通过**。

## 三、交付包同步（嵌入式拿的那份）

交付包路径：`/run/media/manu/SSD960-LOCAL/nfs/mosaic_sdk/`（= 板卡 `/mnt/manu/mosaic_sdk`）。
⚠️ 该分区在 **host 上是只读挂载**（`/dev/sdb[/nfs] ext4 ro`，且 host 无 sudo），
但**板卡挂的是 `rw` + `no_root_squash`** ⇒ 改包必须**经板卡写回**。

已更新（先备份为 `*.bak_20261120`）：

| 文件 | 旧 md5 | 新 md5 |
|:--|:--|:--|
| `src/npu/mosaic_track_fusion.h` | `c222f09e55e1b3a3caa1c84c19c59882` | **`dba851ed2881f2ca647f666a68f7631c`** |
| `src/npu/mosaic_pipeline.h` | `56aa3b281a8da7329247bcf3055aa4c2` | **`47553e374356b680ba5799a0d5a90624`** |
| `out/libmosaic_pipeline.so` | `0baaaaecd12af82606601ce685384e80` | **`e9b37af077cf1033be6e206fbd5407e4`** |
| `out/mosaic_api_test` / `mosaic_async_client_example` / `mosaic_client_example` / `mosaic_contract_test` / `rknn_mosaic_e2e` | 15:00 旧构建 | 09:07 新构建（同一 ocv46 基线） |

- 顶层 `mosaic_pipeline.h` 是指向 `src/npu/mosaic_pipeline.h` 的**符号链接** ⇒ 不用另改。
- 新 `.so` 的 OpenCV 依赖仍是 **`.so.406`（OpenCV 4.6）**，与交付包原基线一致
  （host 上另有两个变体：`ocv48` = `a8d56d26…`、**`ocv42` = `6baf4bc5…`**，后者是本板 `.64` 实测用的）。

### 🔴 交付包与本板 OpenCV 版本不一致（务必转告嵌入式组）

| 板卡 | OS / glibc | 可用 OpenCV | 交付包 .so 依赖 |
|:--|:--|:--|:--|
| **本板 `.64`**（本次验证用） | Ubuntu 20.04.6 / **glibc 2.31** | **只有 focal 自带 4.2.0**（`libopencv_*.so.4.2`） | ❌ 交付包 `.so.406` 在这台板上**跑不了**（需 GLIBC_2.32/2.33）⇒ 本次用 `ocv42` 变体 `6baf4bc5…` |
| 另一台 4.6 板 | — | 4.6 | ✅ 交付包 `.so.406` |

⇒ **交付包保持 4.6 基线不变**（嵌入式那台就是 4.6 板）；但 `.64` 这类 Ubuntu 20.04 板
必须用 `ocv42` 变体，否则 `dlopen` 直接失败。

