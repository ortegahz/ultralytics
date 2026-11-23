# 板卡侧 Δn（跳帧）融合参数 —— 2026-11-19 最优组同步预设

**状态**：🟡 **源码侧已改完，待板卡复测**（板卡 2026-11-19 晚间断电；2026-11-20 已把
`mosaic_track_fusion.h` / `mosaic_pipeline.h` 的默认值与注释**改成交付组**并同步到
各 SDK 树 + `rk3588_app_base/mosaic_sdk/`，并把 **.so 重编译**（md5 `e9b37af077cf1033be6e206fbd5407e4`，
二进制里已核到 0.48/0.26/1.00/10.0/5.0 且**无旧值**）；**板端还差一次实测复测**）
**来源**：`manu/memory/trial0080_delta_search_spec_20261119.md`（trial_0080 重建缓存 + v3 基座上的 Δn 重搜）
**交付参数档**：`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json`（`params_md5 = a1157e013ebb496731be805c72c95dd8`）—— 下表「本次最优」一列即该档 `params` 块的值，**板卡侧施工前请先用这份 md5 核对交付档未被改动**。

**2026-11-20 13:00~13:45 收口**：板卡侧「融合语义」已与服务器 Python **逐字节对齐**（§2.2）、
板端运行时 **9 字段读回已被板卡日志证实**（§2.0）；**仅剩「板端端到端 FAR 复核」一项未闭环**（§4 第 4 条）。
本文件 2026-11-20 新增的所有数据均出自 `manu/memory/memory.md` **§262**（板卡对齐收口），
证据目录 `manu/memory/align_dn_20261120/`。

---

## 1. 为什么要改

板卡 C++ `rk3588_app_base/mosaic_sdk/src/npu/mosaic_track_fusion.h:80-104` 的 Δn 默认值
**是上一轮旧搜索的产物**（跑在 `nogmc_t3/swa` 缓存 + `th045` 阈值上，不是交付底座），
而且那一轮的 `match_dist` 搜索最优是 4.0、板卡却一直用 16.0（用户口径「match_dist 16.0，24 序实测最稳」）。

本轮在**交付底座**（trial_0080 重建缓存 + v3 工作点）上重搜，并且目标函数加了
「逐帧口径不回退」硬约束 ⇒ 得到下面这组**同时守住逐帧 FAR 红线、跳帧臂又能用**的参数。

---

## 2. 需要同步的默认值（2026-11-20 已改完，本表即改后状态）

| 参数 | 板卡当前（旧搜索产物） | **本次最优（要改成这个）** | 含义 |
| --- | --- | --- | --- |
| `confirm_consec` | 2 | **2**（不变） | 连续命中多少次才起批 |
| `confirm_gap_tol` | 6 | **8** | 起批期间允许断链的**物理帧**数 |
| `max_age_phys` | 10 | **null（不设）** | 丢失后按物理帧保留；不设 = 沿用 legacy 按更新次数计龄 |
| `r_slope` | 3.5 | **5.0** | 每多跳 1 帧，关联半径放宽多少 px |
| `alpha_decay` | 1.0 | **1.0**（不变） | 置信度跨帧保留率 |
| `match_dist` | 16.0 | **10.0** | 基础关联半径 px |
| `min_hits` | 3 | **3**（不变） | 最少命中数 |

**阈值类参数（th_base / th_ground / sky_ratio 等）一个都不动**：
`th_base=0.48`、`th_ground=0.26`、`sky_ratio=1.0` 仍是 v3 基准，已冻结。

### 2.0 🔴 板端**调用侧必须显式覆盖**的 9 个字段

> 🟢🟢 **【2026-11-23 已作废：这两节（§2.0 / §2.0.1）的「调用侧必须显式设置」要求已取消】**
> **交付参数组 `cc2_md10`（9 个字段）与跨尺度候选合并半径（20 px）现已内置在库里，且跟踪默认开启**：
> ① `mosaic_pipeline_create()` 自己建好融合引擎（参数 = 交付工作点），**调用侧一行都不用写**；
> ② 跨尺度合并半径是**库常量 20 px**（16~24 为可用带），**公开 setter 已删除**
>    （`mosaic_pipeline_set_candidate_dedup_radius()` **不再是公开接口** ⇒ 公开接口集合回到 2026-11-20 交付版，
>    客户已集成的调用代码**不需要重编**）；
> ③ 标定 / A-B 用环境变量：`MOSAIC_TRACK=0`（关跟踪）、`MOSAIC_TRACK_MERGE_RADIUS=<px>`（覆盖半径，0=关合并）、
>    `MOSAIC_TRACK_QUEUE_MAX=<n>`（`collect_tracked` 队列上限，默认 4096）；
> ④ 验收证据不变：板端启动日志必须有 `[i] candidate dedup ON: radius=20.0 px (cross-scale merge)`。
> ⚠️ **下面 §2.0 / §2.0.1 的正文保留为历史记录**（2026-11-19~11-20 的施工要求与实测），
> **嵌入式组请勿再照抄那两段显式设置代码** ⇒ 照抄会编译不过（接口已删）。
> 权威新版口径：`manu/pipeline/npu/packaging/README.template.md`「🟢 跟踪已内置、默认开启（2026-11-23）」。

头文件默认值已经对了，但**板卡上跑的到底是哪一份 `.so`、头是不是那一版，不能靠猜**
⇒ **调用侧显式写死**，不依赖 `.so` 里编译进去的默认值。
（已写进 `manu/pipeline/npu/mosaic_cadence_probe.cpp` 与 `mosaic_track_dump.cpp`；
嵌入式组自己写的调用侧请照抄这段。）

```cpp
  MosaicTrackFusionConfig tf;
  mosaic_pipeline_track_fusion_defaults(&tf);
  // 🔴 交付工作点（trial_0080 + v3 · cc2_md10）：**显式**覆盖这 9 个字段
  tf.th_base        = 0.48;   // 天空区高置信阈值
  tf.th_salvage     = 0.06;   // 天空区弱脉冲打捞阈值
  tf.th_ground      = 0.26;   // 地面区阈值
  tf.sky_ratio      = 1.0;    // 天地分界 = img_h * sky_ratio
  tf.confirm_consec = 2;      // 连续命中几次算确认
  tf.confirm_gap_tol= 8;      // 连续判定容忍的额外物理帧间隔
  tf.r_slope        = 5.0;    // 每多跳 1 个物理帧，关联门限放宽的像素数
  tf.match_dist     = 10.0;   // 关联基础门限
  tf.max_age_phys   = -1;     // 交付组「不设」⇒ <0 = legacy（写 0 会让航迹下一帧就死；
                              //   写 10 则全口径 FAR 0.0352 ⇒ 破 0.03 红线）
  mosaic_pipeline_set_track_fusion(pipe, &tf);
```

🔴 **为什么是 9 个不是 8 个**：夜班交接报告列的是「检测阈值 4 项 + 跟踪器 4 项」，
**漏了 `max_age_phys`**。而旧 `.so` 里的默认值是 **10**（不是「不设」）⇒ 只覆盖那 8 个的话，
`max_age_phys` 仍会留在 10 ⇒ 板端跑的是**已被交付闸门拒绝**的那组参数
（全口径 FAR 0.0352，破 0.03 红线）。**必须一起覆盖成 -1。**

🟢 板端核验方式：`mosaic_pipeline_set_track_fusion()` 成功时会打印一行
`[i] track fusion ON: th_base=… th_salvage=… th_ground=… sky_ratio=… match_dist=… …`
（2026-11-20 已把 `th_salvage` 与 `match_dist` 补进这行打印）⇒ **照着这行对一遍即可**，
不必反编译 `.so`。

🟢 **2026-11-20 已核验（不再是「待办」）**：板卡运行时读回日志 `/root/verifyB4_20261120_133017.log`
里的原文一行：

```
[i] track fusion ON: th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00 \
    tags=0x1 reorder_window=0 physical_frame_id=1 max_age_phys=-1 r_slope=5.00 alpha_decay=1.000 \
    confirm_consec=2 confirm_gap_tol=8
```

⇒ **9 个字段逐个对上交付参数组 `cc2_md10`**（含 `max_age_phys=-1` = 「不设」；
`physical_frame_id=1` = 引擎吃的是**物理帧号**）
⇒ **「板端必须显式覆盖 9 个字段」这条要求已在真实板卡上落地、并被板端日志证实**。
⚠️ 前提：这次读回用的是**含 `th_salvage` + `match_dist` 的那份 `.so`**（板卡 `/root/build/` 那份，
见 §4.2 的交付风险提示）。

### 2.0.1 🔴🔴 板端**调用侧必须显式打开**的跨尺度候选合并（2026-11-20 新增，第 10 步）

> 🟢 **【2026-11-23 起：合并半径已内置 20 px，默认开启】**
> ⚠️ **下面那段 `mosaic_pipeline_set_candidate_dedup_radius(pipe, 20.0);` 的代码已作废**
> —— 该公开接口**已删除**（加它会改动客户已集成的公开接口）；合并半径现在是**库常量 20 px**，
> 唯一覆盖口是环境变量 `MOSAIC_TRACK_MERGE_RADIUS`（标定 / A-B 用，0 = 关合并）。
> 板端启动日志里的验收行**不变**：`[i] candidate dedup ON: radius=20.0 px (cross-scale merge)`。

只覆盖那 9 个字段**还不够**：板端逐帧 FAR 是 **0.3832**（红线 0.03 的 **12.8 倍**），
根因**不在融合层**（融合层已逐位证明 ≡ 服务器），而在**候选流**：
服务器在喂融合引擎**之前**会把三个尺度的热图峰合并成一个候选
（`merge_scales(dedup_radius)`），板端 SDK **没有这一步**
⇒ 同一个目标在三个尺度上各出一个峰 ⇒ 融合后裂成多条航迹 ⇒ 每条多出来的航迹都是一次误报。

SDK 已补上这一步（`mosaic::MergeCandidatesByRadius`，与服务器 `_merge_scale_rows` **逐条等价**），
但**默认是关的**（半径 0 = 保持原行为），必须由调用侧**显式打开**：

```cpp
  // 🔴 交付值 20 px（16~24 为可用带）。放在 mosaic_pipeline_set_track_fusion() 之后调用。
  //    不开 = 候选流不合并 = 逐帧 FAR 0.3832（破红线 12.8 倍）。
  mosaic_pipeline_set_candidate_dedup_radius(pipe, 20.0);
```

🔴 **半径不能照抄服务器的 768**：768 px 大于 640x512 帧内任意两点距离（最远约 821 px）
⇒ 几乎每帧只剩最高分的那一个候选，把召回从 0.557 压到 0.532（**丢次要目标**）。
板端的正确值是 **16~24 px**，实测（板端真实候选流 + C++ 融合 + 同一套打分器）：

| 合并半径 px | F1 | Recall | Precision | **FAR** | fp |
|--:|--:|--:|--:|--:|--:|
| 0（= SDK 原行为） | 0.5689473 | 0.554959 | 0.583659 | **0.3832** | 958 |
| 16 | 0.6907507 | 0.557025 | 0.908968 | 0.0540 | 135 |
| **20（交付值）** | **0.6930591** | 0.557025 | 0.917007 | **0.0488** | 122 |
| 24 | 0.6927030 | 0.557025 | 0.915761 | 0.0496 | 124 |
| 32 | 0.6930591 | 0.557025 | 0.917007 | 0.0488 | 122 |
| 服务器参考（同一窗口） | 0.6918947 | 0.559091 | 0.907445 | 0.0552 | 138 |

⇒ 开到 20 px 后，板端与服务器**逐帧口径**对齐：F1 差 **+0.0012**、FAR 差 **−0.0064**
（板端反而更干净），**匹配率 82.5%**。

🟢 板端核验方式：`mosaic_pipeline_set_candidate_dedup_radius()` 会打印
`[i] candidate dedup ON: radius=20.0 px (cross-scale merge)`
⇒ **板端启动日志里必须有这一行；没有这一行就是没开**（显式传 0 会打印一条 `[warn]` 警告）。

🟢🟢 **2026-11-20 14:05~14:31 已在真板卡上「在线」验证通过**（不再是「本机推算」）：
在 `evm3588`（`root@192.168.0.64`，aarch64 / OpenCV 4.8）上把**含合并的新 `.so`** 编出来装上，
用**真 2500 帧 NPU 推理**（`fps=4.859`，`dropped=0`）跑同一段 `VIDEO00007_19700101_001324`：

| 板端运行 | FAR | F1 | TP / FP / FN | 板↔服务器轨道匹配率 |
| --- | ---: | ---: | --- | ---: |
| `--cand-dedup-radius 0`（对照） | 0.3832000 | 0.5689473 | 1343 / 958 / 1077 | 53.542% |
| **`--cand-dedup-radius 20`（交付值）** | **0.0488000** ✅ | **0.6930591** | 1348 / **122** / 1072 | **83.673%** |
| 服务器参考（同一窗口 / 同一 GT / 同一匹配器） | 0.0552000 | 0.6918947 | 1353 / 138 / 1067 | —— |

⇒ **板端在线结果 = 本机推算值（0.0488 / 0.6930591），逐位吻合**；
**对照组的两个产物逐字节复现旧交付产物**（fused `5a317dab…` / cands `0e511a4c…`）
⇒ **FAR 的改善唯一来自合并，重编本身零行为漂移**。
板端启动日志实测含 `[i] candidate dedup ON: radius=20.0 px (cross-scale merge)`。
（权威取证：`manu/memory/trial0080_board_gt_scoring_20261120.md` §十四；主档 `manu/memory/memory.md` §269）

⚠️ **本节改动只影响「喂给融合引擎的候选流」**：
`mosaic_pipeline_collect()` 吐出来的**原始候选流**（`_cands.tsv`）不变，
所以板端的 `_cands.tsv` 与改前**逐字节相同**（可以做回归对照）。

⚠️ **ABI 说明**：这一项**没有**加进 `MosaicTrackFusionConfig` 结构体，而是单独的
`mosaic_pipeline_set_candidate_dedup_radius()` 接口
——因为 `MOSAIC_TRACK_FUSION_VERSION` 是**严格校验**的，加字段就得升版本，
所有按 v2 编译的调用侧会被直接 `-2` 拒绝（融合静默不生效）。单开一个接口则**老二进制照常工作**。

---

## 2.1 🔴 C++ 语义对齐对照表（**改默认值之前必须先核这一节**）

服务器上**没有**板卡 C++ 源码（`mosaic_track_fusion.h` 在 host / 板卡侧），
所以下表是**服务器侧 Python 的权威语义**（逐行读过实现），
请**逐条对照 C++ 实现**，确认语义一致后再改默认值。
🔴 **最大的风险是「物理帧」vs「更新次数」**——两个口径下同一个数字行为完全不同。

| 旋钮 | Python 权威实现 | 精确语义 | C++ 必须满足 |
| --- | --- | --- | --- |
| `confirm_gap_tol` | `manu/evaluation/system_sota_longquan.py:382` → `max_gap = 1 + confirm_gap_tol` | 相邻两次**真实观测的帧号差 ≤ `1 + confirm_gap_tol`** 即算同一个 run | ⚠️ **必须用帧号差**（`frame_id` 之差），**不是**「距上次更新的次数」。v3 的 `gap_tol=0` ⇒ `max_gap=1` ⇒ 物理帧差 >1 就断 run ⇒ 跳帧下全断（这正是必须改成 8 的原因） |
| `confirm_consec` | 同上（`apply_consecutive_confirmation`） | 最长 run 的**真实观测帧数** ≥ `confirm_consec` 才确认；**infilled（合成）帧不计数** | ⚠️ 若 C++ 把 infilled 帧也计入连续次数，则 `confirm_consec=2` 的起批会比服务器口径更快 ⇒ 实测 F1 可能低于 94.4976% |
| `r_slope` | `manu/evaluation/eval_bidirectional_track_fusion.py:251` | `cur_gate = min(max_match_dist, match_dist + r_slope*(dt-1) + 0.5*velocity_norm)`，`dt` = **物理帧号差** | ⚠️ `dt` 必须是物理帧差；且门限**还要加 `0.5 × 速度范数`**、并受 `max_match_dist=24` 上限约束 |
| `max_age_phys` | `eval_bidirectional_track_fusion.py:311-312` | 设了它 ⇒ `dead = (frame_idx - last_hit_fid) > max_age_phys`，**接管** `max_age` 分支 | ⚠️ **交付值是「不设」** ⇒ C++ 里应让 `max_age_phys` 处于**未设置/关闭**状态（走 legacy `time_since_update > max_age`），**不是写 0**（写 0 会让航迹下一帧就死） |
| `alpha_decay` | `eval_bidirectional_track_fusion.py:132-141` | `score = 0.65*(score * alpha_decay^dt_hit) + 0.35*new_score`，`dt_hit` = **物理帧号差** | ⚠️ 交付值 `1.0` ⇒ 走简式 `0.65*score + 0.35*new_score`；若 C++ 仍按旧式衰减实现，`alpha_decay=1.0` 也应等价于「不衰减」 |
| `match_dist` | `eval_bidirectional_track_fusion.py:251` | 基础关联门限 px | 门限还要叠加 `0.5 × 速度范数`，上限 `max_match_dist=24` |
| `min_hits` | 跟踪器起判命中数 | 最少命中数 | — |

🟢 **对照用的黄金参照**：服务器端同一套 Python 实现的逐帧口径结果是
**F1 94.4976% / FAR 0.027859（24 序 / 165,223 帧）**。
板卡改完后若**逐帧**（每帧都喂）实测与这个数对不上 ⇒ 就是上面某条语义没对齐，
**先查语义、再调参数**。

⚠️ 另外：本文件第 2 节列的「板卡当前值」（`confirm_gap_tol=6` / `max_age_phys=10` / `r_slope=3.5` / `match_dist=16.0`）
是早前读板卡时记下的；**施工时请以板卡上现读到的值为准**，不要照抄本文件的历史值。

---

## 2.2 🟢 板卡 ↔ Python「逐字节」对齐闸门（2026-11-20 已过）

> 本节数据全部出自 `manu/memory/memory.md` **§262**（板卡对齐收口），无推算、无估计值。

**为什么必须做**：此前唯一的板卡↔Python 对齐产物
`manu/memory/cadence_ab_20261119/vid09_align/`（6 份 TSV 逐字节相同）跑的是 **Δn ≡ 1**
（190 个帧间隔全是 1）**且用的是 `th045` 参数**（2026-11-20 复算证实：th045@Δn1 输出 md5
`93a7d5ec119ce2d52d0745ccccbe9f64` = 那份归档产物的 md5，**逐位复现**）
⇒ **交付参数组 `cc2_md10` 从未在板卡上与 Python 对过账**，而它的全部意义就是跳帧工况。

**闸门脚本**：`manu/evaluation/align_dn_delta_check.py`
- 从**一份**候选 TSV 构造多种 **Δn 图案**（1 / 2 / 3 / 5 / 7 + **板卡实测 cadence 的间隔序列**）；
- 分别跑 **C++**（`rk3588_app_base/mosaic_sdk/src/npu/track_fusion_align_test.cpp`，
  `mosaic::TrackFusionEngine`，**零第三方依赖** ⇒ host 与板卡都能原生编译）
  与 **Python**（`manu/evaluation/sdk_track_fusion_ref.py --physical-frame-index`，
  即**服务器生产流水线**本身）；
- 对两份输出做**逐字节**比对。

**结果（全部逐字节一致）**：

| 数据 | 参数组 | Δn 图案 | 组合数 | 板卡(aarch64) vs 主机(x86_64) vs 服务器 Python |
| --- | --- | --- | ---: | :-: |
| `manu/memory/cadence_ab_20261119/vid09_align/v09_cands.tsv`（板卡实测检测流，191 帧） | 交付 `cc2_md10` / `v3` / `old_gapfit` / `th045` | 1 / 2 / 3 / 5 / 7 / 板卡 cadence | **24** | ✅ **24/24 逐字节一致** |
| **真实 0080 数据** `VIDEO00007_19700101_001324`（板卡用真 `p0_off_A_none.rknn` 跑出的 **2500 帧**检测流） | 交付 `cc2_md10` | 1 / 2 / 3 | **3** | ✅ **3/3 逐字节一致** |

真实 0080 数据那 3 组的 md5（**板卡 / 主机 / 服务器 Python 三方同一个**）：

| Δn | md5 | 检测条数 |
| --- | --- | ---: |
| 1 | `839e1f437be0196e5fb4c54da6105d9e` | 2301 |
| 2 | `026a219ca0055f30c120aaab058caab4` | 2323 |
| 3 | `3321761b02586f5c298c2b08ab91e62f` | 2158 |

⇒ **板卡 ≡ 主机 ≡ 服务器 Python** ⇒ **§2.1 的语义对照表已被实测闭环**，
板端改参数不再有「语义漂移」风险。

**⭐ 附带复现的已知失效模式**：`v3` 参数组在 **Δn ≥ 2 时输出检测数精确为 0**
（板卡 C++ 与服务器 Python 两侧同时复现）⇒ 逐位复现「v3 原参跳帧臂 F1 = 0.084122、
s1~s3 精确为 0」⇒ 这解释了为什么 `v3` 在板卡真实节奏下不可用；
交付组 `cc2_md10` 在同一份真实数据上 180/2301 条正常出图。

**`.so` 融合路径 == 独立对齐引擎**：同一份真实 2500 帧候选流，
`.so` 跑出的 `fused` 与独立引擎跑出的 `fused` 逐字节一致（2301 条，md5 `5a317dab22bb28497e93b3db77927270`）。

⚠️ **口径坑（比对前必读）**：`.so` 融合路径**只写非空帧**，独立对齐引擎**写完整帧轴（含空帧）**
⇒ 比对前**必须先滤掉空行**，否则会得到**假不一致**。

**证据与复现**：
- 证据目录：`manu/memory/align_dn_20261120/`
  （`BOARD_MD5.txt` / `HOST_GOLDEN_MD5.txt` / `host_align_matrix.log` / `params_*.txt` / `real_0080/`）
- 板卡工作目录：`/root/align_dn_20261120/`
- 板卡跳板助手：`rk3588_app_base/board_sh.sh`（本机 → 服务器 → 板卡）
- 编译命令（可直接复制）：

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations \
    -o /tmp/align2/track_fusion_align_test \
    ../rk3588_app_base/mosaic_sdk/src/npu/track_fusion_align_test.cpp \
    ../rk3588_app_base/mosaic_sdk/src/npu/mosaic_track_fusion.cpp
```

⚠️ `-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations` 是**逐字节一致的前提**，
**不要省略或改写**（浮点合并 / FMA 会改末位 ⇒ 字节级不等）。

---

## 3. 服务器侧证据

| 口径 | v3 原参 | 本次最优 |
| --- | --- | --- |
| **逐帧（25 fps，全量 24 序 / 165,223 帧）** | F1 94.7533% / FAR 0.020820 | F1 **94.4976%** / FAR **0.027859**（≤0.03 ✔） |
| **跳帧·物理帧基（4000 帧仿真均值）** | **0.084122**（轨迹全判死） | **0.951398** |
| 跳帧·密集帧基（4000 帧仿真均值） | 0.922060 | **0.938340** |

- 🔴 **本表的跳帧数一律用「确定性采样器重测」口径**（`artifacts/rt_eval_delivered_armA.json` 与
  `artifacts/shortlist_deterministic.json`，sampler = `zlib.crc32` 确定性、一个进程内跑完）。
  早先本文件写过 **0.9521 / 0.9374**（§3.1 还写过 0.951990 / 0.952015）——
  那是**采样器修复前**的数（cadence 采样用 `hash(str)` ⇒ 跨进程不可复现），**已作废**。
  ⇒ **板卡实测请对照本表的 0.951398 / 0.938340**；若实测值与 0.951398 相差 0.07pp 左右，
  那是口径差（板卡的跳帧节奏不是 4000 帧仿真那套），**不是板端改错了**。
- 🔴 **最关键的一条**：v3 原参在跳帧下 F1 = **0.084122**（`rt_search_t0080_v3_seedfix.json` → `2b_jump_physical_untuned`），
  因为确认层的 run 跨度阈值 = `1 + confirm_gap_tol` = **1 帧**（v3 的 `confirm_gap_tol=0`），
  而板卡真实节奏 Δn ≈ 3.4（实测 7.4 fps @ 25 fps 源）⇒ **run 直接断裂、轨迹全被判死**。
  ⚠️ 早先本文件写过 **0.106**，那是**第一次搜索**（cadence 采样用 `hash(str)` ⇒ 不可复现）的数字，**已作废**；
  唯一可复现的规范值是 seedfix 的 **0.084122**（优化臂同源 0.952186）。
- 🟢 改成上面这组之后，跳帧臂 mean F1 = 0.951398，**甚至高于同参数下的密集帧基 0.938340**（+1.31pp）
  ⇒ 说明「让引擎看见真实物理帧号」本身是净收益。
- ⚠️ **代价必须说清**：逐帧口径 F1 回退 **−0.2557pp**（94.7533% → 94.4976%），FAR 从 0.0208 升到 0.0279。
  FAR 增量几乎全部来自病态样本 `VIDEO00012_19700101_002251`（单序列 FAR 0.0997 → 0.1261）。

### 3.1 若想把 FAR 余量再放大（备选）

把 `match_dist` 从 **10.0** 改成 **8.0**（其余不动）：
全量口径 F1 **94.484%**（−0.013pp）/ FAR **0.0265**（余量 0.0035，比 10.0 的 0.0021 大 67%），
跳帧臂 F1 0.951215（与 10.0 的 0.951398 实质相同，差 0.018pp）。
**默认仍用 10.0**（逐帧 F1 最优）；只有当板端 FAR 实测比服务器高、逼近红线时才切 8.0。

### 3.2 🔴 别直接抄搜索器的 objective-best

搜索器的 objective-best 是 **`max_age_phys=10 / match_dist=16.0`**（第二、三次搜索都选它），
但它在**全量口径 FAR = 0.0352 ⇒ 破线**，**已被交付闸门拒绝**。
板卡侧请用本文件第 2 节的表，**不要**用搜索器输出的那组。

---

## 4. 板卡复电后的验证清单

1. ~~把 `mosaic_track_fusion.h` 的 7 个默认值按上表改掉~~ ⇒ ✅ **2026-11-20 已改完**
   （仓库源 + 4 个 SDK 树 + `rk3588_app_base/mosaic_sdk/` 全部对齐；
   板端 `.so` = `out/libmosaic_pipeline.so` md5 `e9b37af077cf1033be6e206fbd5407e4`，
   链接板卡自带 OpenCV 4.6（`libopencv_*.so.406`）—— **别错发 ocv48 / ocv42 那两份**，
   它们链接的是 `.so.408` / `.so.4`，板卡上没有）。
   ⇒ 板端仍要做的是**调用侧显式覆盖那 9 个字段**（见 §2.0），别只依赖头文件默认值。
   ⚠️ **OpenCV 版本口径提醒（2026-11-20 板卡实测后更新）**：本条里的「板卡自带 OpenCV 4.6 /
   `libopencv_*.so.406`」是**按 `.211` 目标板基线**写的；**当前唯一在线的板卡 `.64`（`evm3588`）
   自带的是 OpenCV 4.8（`.so.408`）** ⇒ 该条**不适用于 `.64`**，详见 §4.1。
2. ~~跑一次板端回归，确认 `cad_on.tsv` 的 Δn 分布与仿真 s1（mean Δn 3.38 @ 7.4 fps）对得上；
   - 若板端 Δn 明显大于 3.4，考虑把 `confirm_gap_tol` 再放宽一档。~~
   ⇒ ✅ **2026-11-20 已实测**：板卡与仿真两侧都拿到了直方图

   | 来源 | 平均 Δn | 直方图 | fps |
   | --- | --- | --- | ---: |
   | **板卡实测** `cad_on.tsv`（1930 个间隔） | **3.109** | — | — |
   | **板卡实测** cadence 探针 `cadence_run2.tsv` | **3.043** | `1:8 2:135 3:402 4:133 5:18 6:4` | 8.20 |
   | 仿真 **s1**（measured 1x，board idle） | **3.360** | `1:21 2:201 3:590 4:231 5:69 6:35 7:28 …` | 7.45 |
   | 仿真 **s2**（1.5x + jitter，shared SoC） | **5.232** | 峰在 4~6 | 4.78 |
   | 仿真 **s3**（2x + jitter，heavy load） | **7.004** | 峰在 5~6、长尾到 34 | 3.58 |
   | 仿真 **s4**（doc 口径，2 workers） | **1.951** | `1:687 2:900 3:379 …` | 12.82 |

   ⇒ 板卡实测（3.04~3.11）**对应 s1**（3.36，偏差 ~9%）；**s2/s3 是「SoC 负载更高」的压测工况**、
   **s4 是「两 worker」的乐观假设**，与板卡**空闲态**实测不是同一档
   ⇒ **板卡空闲态 ≈ s1**，**不需要再放宽 `confirm_gap_tol`**（交付组保持 `confirm_gap_tol=8`）。
   ⚠️ 板卡上线后若 CPU 被占，Δn 会往 s2 漂 ⇒ 这正是交付组必须耐跳帧的原因。
3. 对照 `osd_sys_sota_trial0080_v3_optgapfit_20261119/`（服务器已渲好，24 个 mp4）肉眼复核 OSD。
4. 板卡端 FAR 复核：目标是 ≤ 0.03，与服务器逐帧口径 0.0279 同量级。
   ⇒ 🔴 **2026-11-20 仍未闭环（唯一离线排不掉的风险）**。已探明的阻塞与绕过路径：

   | 项 | 2026-11-20 实测 |
   | --- | --- |
   | 板卡 `.64` | ✅ **在线**（hostname `evm3588`、Rockchip RK3588、Ubuntu 20.04.6 / kernel 5.10.160、OpenCV **4.8**、librknnrt **2.3.0**、8 核 / 7.9 GB） |
   | NPU 可用性 | 🔴 **旧结论被证伪**：RKNPU 内建在内核里并注册在 **DRM minor 1** ⇒ 设备节点是 **`/dev/dri/renderD129`**（不是 `/dev/rknpu*`）；整条流水线已在板上实跑通 **2500 帧**（`rknn runtime api=2.3.0 driver=0.9.8`、`RGA native-tile self-check: OK`、`pipeline_frames=2500 fps=4.856 dropped=0`）⇒ **「`/dev/rknpu*` 缺失 ⇒ 跑不了 rknn」不成立** |
   | 还没做完的真正原因 | 🔴 要把 **24 序全部帧**搬上板：板卡根分区 **95% 满（13:0x 复测剩 866M；11:0x 首测 871M）**、`/tmp` tmpfs 仅 **3.9G**；host 的 `/proc/fs/nfsd/exports` **为空** ⇒ 走 NFS 需用户 `sudo exportfs -ra` + `sudo mount -o remount,rw /run/media/manu/SSD960-LOCAL` |
   | 🟢 **绕过路径** | 板卡 → 服务器 `192.168.99.40:32222` **TCP 实测可达** ⇒ 帧可由**服务器直接 scp 到板卡**，不必等 NFS |
   | ✅ **已完成的等价验证** | 融合**语义**侧已用**真实 0080 检测流逐字节对齐**（见 §2.2）；板端运行时 **9 字段读回**也已证实（见 §2.0） |
| 🟢 **2026-11-20 14:05~14:31 新增：板端「在线」复测已做** | 在真板卡 `evm3588` 上把**含跨尺度合并的新 `.so`** 编出来、跑**真 2500 帧 NPU 推理** ⇒ **逐帧 FAR 0.3832 → 0.0488000**（红线 12.8× → 0.63×）、**F1 0.5689473 → 0.6930591**，**优于同窗口服务器参考**（FAR 0.0552 / F1 0.6918947）；且「合并关」的对照组**逐字节复现**旧交付产物 ⇒ **重编零行为漂移**。详见 §2.0.1 与 `manu/memory/trial0080_board_gt_scoring_20261120.md` §十四 |

   ⇒ 剩余动作 = 「板卡跑 24 序 → 与服务器 `track_all_v3_optgapfit.json` 对账」，
   依赖用户 `sudo` 或改走服务器 scp 通道。

   #### 4.0 ✅ 2026-11-20 14:00~15:10：**根因已锁定并定量**（不再只是「未知风险」）

   本机对**板卡自己产出的 2500 帧候选流**做了第一次「对 GT 打分」，并与并发线的板端在线打分
   **两条独立路径互证**（权威专题档：`manu/memory/trial0080_board_gt_scoring_20261120.md`）：

   | 板端候选流处理 | F1 | **FAR/帧** | ΔF1 vs 服务器 |
   | --- | --- | --- | --- |
   | **板端现状（SDK 不做跨尺度合并）** | 0.568947 | **0.383200** | −0.122947 |
   | **+ 跨峰去重 R = 16 px** | 0.690751 | **0.054000** | −0.001144 |
   | **+ 跨峰去重 R = 20 px** | **0.693059** | **0.048800** | **+0.001164** |
   | **+ 跨峰去重 R = 24 px** | 0.692703 | **0.049600** | +0.000808 |
   | 服务器参考（同窗口 / 同 GT / 同匹配器） | 0.691895 | 0.055200 | — |

   ⇒ 🔴 **根因**：交付档里的 `dedup_radius=768.0` 属**候选构建**（多尺度合并）参数，
   **SDK 里根本没有这一级** ⇒ 同一目标的多尺度峰各自成航迹 ⇒ **每帧多一条 FP**。
   **不是**融合层参数错 ⇒ **§2.0 的 9 个字段与 §2.2 的逐字节对齐结论依然成立、可照常交付**。
   ⇒ 🟢 **交付动作**：① 板端调用侧 `MosaicTrackFusionConfig.tag_mask` 设 **`0x1`**（仅原生尺度）；
   ② **正解**：SDK 候选阶段补一步**跨尺度合并，半径 16~24 px**（**不要**照抄 768 px）——
   平台区 R 20~64 px ⇒ FAR 0.0476~0.0496 / F1 0.6930~0.6966（≈ 服务器同窗口）。
   ⇒ 新增可复现脚本：`manu/evaluation/board_far_parity_check.py`、`manu/evaluation/board_far_dedup_probe.py`；
   证据 `manu/memory/board_far_20261120/`；板端原始产物已抢救归档到 `manu/memory/board_far_20261120/board/`。
   ⚠️ 0..2499 帧是**困难片段**（服务器自己这段 FAR 0.0552，全序列仅 0.0262）⇒ **绝对 FAR 不可拿来判 0.03 红线**，
   只有**相对差**可用；**24 序全量板端 FAR 仍未做**（板卡 tmpfs 只放得下 2500 帧）。
5. 核对待测 `.so` 的参数：读板端启动时打印的 `[i] track fusion ON: …` 那一行，
   确认 `th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00
   max_age_phys=-1 r_slope=5.00 confirm_consec=2 confirm_gap_tol=8`。
   （`th_salvage` 与 `match_dist` 两项是 2026-11-20 才补进这行打印的 —— **嵌入式组重编自己的应用后**
   才能一次看全 9 项；用旧二进制打印的那行**没有**这两项，别当成漏配。）

   ✅ **2026-11-20 板端实测行**（`/root/verifyB4_20261120_133017.log`，逐位照抄）：

   ```
   [i] track fusion ON: th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00 \
       tags=0x1 reorder_window=0 physical_frame_id=1 max_age_phys=-1 r_slope=5.00 alpha_decay=1.000 \
       confirm_consec=2 confirm_gap_tol=8
   ```

   ⇒ **9 项全部命中交付组 `cc2_md10`**（`max_age_phys=-1` 即「不设」）。

### 4.1 🔴 复测前的环境前置（2026-11-20 11:0x 实测卡住的地方）

| 前置 | 实测 | 谁来解 |
| --- | --- | --- |
| **host 的 NFS 交换区可写 + 内核导出表生效** | `findmnt /run/media/manu/SSD960-LOCAL` ⇒ **`/dev/sdb ext4 ro`**；`/proc/fs/nfsd/` **空** ⇒ 板上 `mount` 报 **`access denied by server while mounting`**；host 上 `touch` 报 **`Read-only file system`** | 用户 `sudo`：`mount -o remount,rw /run/media/manu/SSD960-LOCAL` +（`nfsd` 当前也是 ro ⇒ 可能还要 `mount -o remount,rw /proc/fs/nfsd`）+ `exportfs -ra` |
| **目标板 `192.168.0.211` 上电** | `No route to host`（断电）—— **2026-11-20 13:0x 复测仍断电** | 用户上电（它才是 OpenCV 4.6、对应 `.so.406` 基线） |
| **备选板 `.64`（`evm3588`）的 NPU 可用** | ⚠️ **本行原结论已作废**（2026-11-20 13:0x 实测）：RKNPU **内建在内核里**、注册在 **DRM minor 1** ⇒ 设备节点是 **`/dev/dri/renderD129`**（板上确实存在，226,129），**不是** `/dev/rknpu*` ⇒ **NPU 可用**；整条流水线已在板上实跑通 **2500 帧**（`rknn runtime api=2.3.0 driver=0.9.8`、`RGA native-tile self-check: OK`、`pipeline_frames=2500 fps=4.856 dropped=0`） | **无需再「恢复 rknpu 设备节点」**；直接用 `/root/build/` 那份 `.so` 跑（板卡工具走 `LD_LIBRARY_PATH=/root/build:…`），或改走 §4 第 4 条的「服务器 scp 帧到板卡」通道 |
| **板卡 OpenCV 版本 ⇒ 到底发哪份 `.so`** | ⚠️ **原文「板卡链接 OpenCV 4.6 ⇒ 只发 `.so.406` 那份」当前不适用于 `.64`**：`.64` 自带 **OpenCV 4.8（`.so.408`，在 `/root/mosaic/opencv/`）** ⇒ `.64` 应发 **ocv48** 变体；**`.so.406` 变体对应 `.211`**，而 `.211` 仍未上电 ⇒ **现无可验** | 先确认目标板是哪一台再选变体：`.64` → ocv48、`.211` → ocv46 |
| **板上磁盘余量** | `.64` 根分区 **95% 满（11:0x 首测剩 871M；13:0x 复测剩 866M）**、**`/tmp` tmpfs 仅 3.9G** | 重编/跑大产物前先清；**24 序全帧放不进根分区**，需 NFS 或走服务器 scp 通道 |

⇒ **交付包本身已核实正确**（部署件 md5 逐位对上：`mosaic_track_fusion.h` = `dba851ed…`、
`mosaic_pipeline.h` = `47553e37…`、`out/libmosaic_pipeline.so` = **`e9b37af077cf1033be6e206fbd5407e4`**），
**板端运行时复测在上述前置补齐前做不了**。
⚠️ **2026-11-20 更新**：本句的「做不了」**仅指端到端 FAR 复核**（需把 24 序帧搬上板）；
**融合语义侧已逐字节对齐**（§2.2）、**板端 NPU 已实跑通 2500 帧**、**板端运行时 9 字段读回已证实**（§2.0 / §4 第 4 条）。

---

## 4.2 🔴 交付风险：板卡上现有**两份** `libmosaic_pipeline.so`，诊断打印行不一致

板卡上现有**两份**功能等价的 `libmosaic_pipeline.so`，差别只在 `[i] track fusion ON:` 这行
**打印字段的多少**：

| 路径 | md5 | `[i] track fusion ON:` 打印行 |
| --- | --- | --- |
| `/root/build/libmosaic_pipeline.so`（板卡 13:08 构建） | `b383291e31ccbd71c46430ca69869360` | ✅ **含 `th_salvage` + `match_dist`**（9 字段可核） |
| `/root/mosaic/libmosaic_pipeline.so`（板卡 13:16 构建） | `a8d56d26d3604d5c33d69a859c598a78` | ⚠️ **缺 `th_salvage` + `match_dist`** |
| 本地 `build/mosaic_sdk_20261120/out/libmosaic_pipeline.so` | `e9b37af077cf1033be6e206fbd5407e4` | ⚠️ **旧版打印**（本地 09:07 构建；源码 11:58~11:59 又改过） |

⇒ 两份**功能等价**（默认值都是交付组 `cc2_md10`：0.48 / 0.26 / 1.00 / 10.0 / 5.0，
**无旧值** 16.0 / 3.5 / 0.45 / 0.22），差别只在打印字段数。
⇒ 板端工具走 `LD_LIBRARY_PATH=/root/build:…` ⇒ **板上实际生效的是打印全字段那份**
（§2.0 与 §4 第 5 条的实测行即来自它）。
🔴 **交付给嵌入式组前必须统一**：若对方只拿到 `/root/mosaic/` 那份，核验行会**少两个字段**
⇒ 会被误判成「少配了 `th_salvage` / `match_dist`」。
⇒ **建议统一发 `/root/build/` 那份**（或在 `/root/mosaic/` 那份也补上这两个字段后重编）；
本地 `build/mosaic_sdk_20261120/out/libmosaic_pipeline.so`（09:07 那份）**打印行也是旧的**，
需按当前源码重编后再发。

---

## 5. 交付给嵌入式的模型

**rknn 模型不变，仍是 `trial_0080`**（md5 `1a3faf2f9459cad8a389ffa81810234c`）。
本轮**只动后处理（时空融合）参数**，模型权重、阈值工作点（th_base 0.48 / th_ground 0.26 / sky_ratio 1.0）全部不动。
