# trial_0080 板端 **GT 打分** 验证报告（2026-11-20 13:00~13:45）

> **补上早班报告 `trial0080_board_validation_20261120.md` 缺的那一环**：早班只测了**节奏与时延**，
> **从未把板端输出对 GT 打过分**。本文件是**第一次**板端端到端精度数字，
> 并且它揭出一个**交付阻塞级**缺口（见 §四）。

**板卡**：`root@192.168.0.64`（`evm3588`，Ubuntu 20.04.6 focal / glibc 2.31 / kernel 5.10.160 / 8 核 / 7.9 GB）
⚠️ `manu/AGENTS.md §4` 写的是「RK3588 板卡 = `192.168.0.211`，`.64` 已作废（2026-11-10 换板）」，
但 **`.211` 至今 `No route to host`（断电）**，本轮按**用户当轮明确指定**用 `.64`。
**模型**：`/root/mosaic/runtime/ir_uav_det.rknn`，md5 `1a3faf2f9459cad8a389ffa81810234c`（= trial_0080 ep3 p0-off fp16 ✅）
**序列**：`VIDEO00007_19700101_001324`（11810 帧），**本轮只跑前 2500 帧**（板卡 tmpfs 只放得下）
**GT**：`lqcache_rebuild/lookahead_residual/data/labels/`（与服务器龙泉山 GT 同名同格式）

---

## 一、环境事实更正（推翻 §261 与早班报告的 4 处）

| 项 | §261 / 早班报告写的 | **本轮实测（13:0x~13:4x）** | 取证命令 |
|:--|:--|:--|:--|
| NPU 可用性 | 「板上无 `/dev/rknpu*`，NPU 推理跑不起来」 | **可用**。rknpu 0.9.8 走 **DRM** 注册：节点 `/dev/dri/renderD129`（card1=`fdab0000.npu`，DRIVER=RKNPU）。`/dev/rknpu*` 不存在是**正常的**（那是旧驱动口径） | `dmesg \| grep -i rknpu` ⇒ `[drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1` |
| 板上 OpenCV | 「本板只有 focal 自带 4.2.0」 | **4.8**：`/root/mosaic/opencv/libopencv_core.so.408` ⇒ 本板对应 **ocv48 变体**（`a8d56d26…`），不是 ocv42 | `ls -l /root/mosaic/opencv/` |
| 交叉工具链 | 「host 上没有 aarch64 交叉编译器」 | 板上有 **`aarch64-linux-gnu-g++`** ⇒ **在板上直接重编 `.so`**，不需交叉工具链 | `which aarch64-linux-gnu-g++` |
| 板端跑 SDK | （未记录） | **必须** `export LD_PRELOAD=/root/mosaic/libstatcompat.so`，否则 `libopencv_core.so.408: undefined symbol: stat`（板端既有配方 `/root/ab_run.sh` 就是这么干的） | 见 §六 命令 |

> 🔴 **§261 的「三个硬阻塞」里，「NPU 跑不起来」「板上没有 OpenCV」两条是不成立的**；
> 真正的阻塞只有「host NFS 交换区只读 + 导出表空」（需用户 sudo）和「服务器公钥未授权」。

## 二、板卡上的一颗地雷（已修，**任何人上板前先看这条**）

| 文件 | 原 md5 | 烤进去的参数 | 处置 |
|:--|:--|:--|:--|
| `/root/mosaic/libmosaic_pipeline.so` | `96fb8e98…` | **旧组**：`th_base 0.45 / th_ground 0.22 / sky_ratio 0.60 / match_dist 16.0 / r_slope 3.5` ⇒ **全口径 FAR 0.0352，破 0.03 红线** | 已备份 `/root/mosaic/libmosaic_pipeline.so.bak_20261120_oldparams`，换成交付 **ocv48** `a8d56d26d3604d5c33d69a859c598a78`（已复验烤入新组） |
| `/root/build/libmosaic_pipeline.so` | —— | 板上用新源码重编，新组 | md5 `b383291e31ccbd71c46430ca69869360` |

⚠️ **板端默认加载路径里躺着的是已被交付闸门拒掉的参数组** —— 早班那份报告没发现它，是因为早班**只测时延**，没看烤进去的常量。

## 三、板端参数组运行时回读（9 字段逐位吻合）

板端日志原文（`mosaic_pipeline.cpp` 的 `track fusion ON` 行）：

```
[i] track fusion ON: th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000
    match_dist=10.00 tags=0x7 reorder_window=0 physical_frame_id=1 max_age_phys=-1
    r_slope=5.00 alpha_decay=1.000 confirm_consec=2 confirm_gap_tol=8
```

⇒ 与交付档 `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json`
（`params_md5 a1157e013ebb496731be805c72c95dd8`）**逐位一致**（`max_age_phys` 交付值「不设」= C++ 的 `-1` ✅）。

## 四、🔴🔴🔴 核心发现：**板端候选流缺「跨尺度合并」⇒ 一个目标裂成多条航迹**

### 4.1 现象

板端逐帧输出（`tag_mask=0x7`，SDK 默认）**每帧 0~3 条航迹**，其中 756/2500 帧是 **2~3 条**；
服务器参考（同一段视频）**每帧最多 1 条**。板端逐帧口径：

**FAR 0.3832**（= 红线 0.030 的 **12.8 倍**），精度掉到 0.584。

### 4.2 逐帧取证（frame 0，GT 框 = 中心 (309.0, 253.5) / 104×39 px）

| 来源 | 候选（坐标, 分数） |
|:--|:--|
| 服务器 cache 原始行 | tag0 `(311.6,251.8) 0.8330` · tag1 `(307.1,255.3) 0.7441` · tag2 `(311.6,255.3) 0.6753` · fused255 `(311.3,255.7) 0.8330` |
| 板端 `_cands.tsv` | `(311.1,252.1) 0.834` · `(305.8,249.0) 0.743` · `(318.7,239.6) 0.031` |
| 板端 `_fused.tsv` | `(311.1,252.1) 0.834`（track 1）· `(305.8,249.0) 0.743`（track 2）⇒ **同一个目标，两条航迹** |
| 服务器离线参考 `_fused` | `(311.6,251.8) 0.833`（**1 条**） |

⇒ 板端候选的**前两个峰与服务器 cache 的 tag0 / tag1 行分数几乎逐位相同**（0.834 vs 0.8330、0.743 vs 0.7441），
说明**底模与后处理没问题**；差的是**服务器侧在喂融合引擎前做了一步跨尺度合并**
（`manu/evaluation/eval_multiscale_cache_th.merge_scales`，`dedup_radius = 768.0`），
**SDK 侧没有这一步**：
- `mosaic_pipeline.cpp` 的 `FeedFusion()` 把**每一个尺度**的连通域（`mosaic::Comp`，`tag` = 0 原生 / 1 thumb160 / 2 thumb80）都喂给融合引擎；
- `MosaicPipelineConfig` **没有任何候选级合并/去重字段**（已逐字段核过）；
- 融合引擎的 Hungarian 匹配是 **1 轨迹 ↔ 1 检测** 的一对一分配 ⇒ 第二个峰无处可去 ⇒ **新开一条航迹** ⇒ 判 FP。

### 4.3 归因链（同一套 Python 融合参考，换候选流）

脚本：`manu/evaluation/board_candidate_attribution.py`

| 组 | 候选流 | 候选/帧 | F1 | Recall | Precision | **FAR** | tp | fp | fn |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **A** | 服务器 cache 的每尺度原始行（**不合并**） | 3.45 | 0.5549514 | 0.695868 | 0.461496 | **0.7860000** | 1684 | 1965 | 736 |
| **B** | **板端原始候选流** | 10.60 | **0.5689473** | 0.554959 | 0.583659 | **0.3832000** | 1343 | 958 | 1077 |
| **B**（板端 C++ 实测） | 同上（板端 `.so` 自己跑） | —— | **0.5689473** | 0.554959 | 0.583659 | **0.3832000** | 1343 | 958 | 1077 |
| **C** | 服务器 cache **合并后**（= 交付口径） | 1.00 | 0.7080000 | 0.585124 | 0.896203 | **0.0656000** | 1416 | 164 | 1004 |

🔴 **B 的两行逐位相同（tp/fp/fn/F1/FAR 全等）⇒ 板端 C++ 融合引擎与服务器 Python 参考实现**完全等价**，
**板端融合引擎没有 bug**；差异 100% 出在**候选流是否跨尺度合并**。

### 4.4 合并半径扫描（给嵌入式组的**可落地**参数；板端候选流，离线推算）

| 合并半径 px | 候选/帧 | F1 | Recall | Precision | FAR |
|--:|--:|--:|--:|--:|--:|
| 0（不合并，等于 SDK 现状） | 10.60 | 0.5689473 | 0.554959 | 0.583659 | **0.3832000** |
| 8 | 10.15 | 0.6606224 | 0.557025 | 0.811559 | 0.1252000 |
| 12 | 9.82 | 0.6773869 | 0.557025 | 0.864103 | 0.0848000 |
| **16** | 9.44 | 0.6907507 | 0.557025 | 0.908968 | 0.0540000 |
| **20** | 9.07 | **0.6930591** | 0.557025 | 0.917007 | **0.0488000** |
| 24 | 8.76 | 0.6927030 | 0.557025 | 0.915761 | 0.0496000 |
| 32 | 8.25 | 0.6930591 | 0.557025 | 0.917007 | 0.0488000 |
| 768（服务器口径） | 1.00 | 0.6853035 | 0.531818 | 0.963323 | **0.0196000** |

⇒ **服务器的 768 px 半径 > 图像对角线（821 px）几乎 ⇒ 每帧只剩 1 个候选**：
它把召回从 0.557 压到 0.532（**丢次要目标**），只对「单目标」场景成立。
**半径 16~24 px 才是正确值**：召回不掉（0.5570），FAR 0.049~0.054，且候选/帧仍有 8.8~9.4（**多目标能力保留**）。

### 4.5 板端实测 `tag_mask=0x1`（只用原生尺度）—— 今天就能上的缓解

板端把 `--tag-mask 1` 跑满 2500 帧（其余参数不变）：

| 口径 | tp | fp | fn | gt | Recall | Precision | **F1** | **FAR** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `tag_mask=0x7`（SDK 默认） | 1343 | 958 | 1077 | 2420 | 0.554959 | 0.583659 | 0.5689473 | **0.3832000** |
| **`tag_mask=0x1`（仅原生尺度）** | 1287 | 120 | 1133 | 2420 | 0.531818 | **0.914712** | **0.6725895** | **0.0480000** |
| 离线推算（板端候选 + 20 px 合并） | 1348 | 122 | 1072 | 2420 | 0.557025 | 0.917007 | 0.6930591 | 0.0488000 |

⇒ **`tag_mask=0x1` 把 FAR 从 0.3832 压到 0.0480（8 倍）、精度从 0.584 拉回 0.915**，
且与「20 px 合并」的离线推算**几乎同档**（FAR 0.0480 vs 0.0488、F1 0.6726 vs 0.6931）。

## 五、板端 vs 服务器：**同一交付帧集合**上的对照（交付口径）

板端用 cadence 实测的交付帧号表（`mean Δn = 3.041`）喂 `--ids`，让融合引擎走物理帧轴；
服务器用**同一张 741 帧交付表**跑离线参考（交付参数组）：

| 项 | 板端（`.so`，tag_mask=1，跳帧） | 服务器离线参考（交付参数组） | 差 |
|:--|--:|--:|--:|
| 交付帧数 | 741（其中 420 帧有输出） | 741 | —— |
| tp / fp / fn / gt | 368 / 68 / 363 / 731 | 377 / 34 / 354 / 731 | —— |
| Recall | 0.503420 | 0.515732 | −1.2 pp |
| Precision | 0.844037 | 0.917275 | −7.3 pp |
| **F1** | **0.6306769** | **0.6602452** | **−2.95 pp** |
| **FAR** | **0.0917679** | **0.0458839** | **2.0×** |

⇒ **板端全链路（真 RKNN fp16 + 真 GMC + 真融合）复现了服务器交付口径的行为**：
F1 差 2.95 pp，FP 多一倍（板端候选流比服务器 cache 更「毛边」）。

### 5.1 🔴 诚实边界：本轮用的 2500 帧窗口**是困难片段**，绝对 FAR 不可与红线直接比

| 口径（同一 2500 帧窗口） | F1 | FAR |
|:--|--:|--:|
| 服务器交付口径（合并后） | 0.7080000 | 0.0656000 |
| 服务器 gateA 参考（match_dist=16） | 0.6972010 | 0.0560000 |
| 板端 tag_mask=1 | 0.6725895 | 0.0480000 |

⇒ **同一窗口上服务器自己的参考 FAR 也是 0.056~0.066**（>0.03），
而全量 165,223 帧口径的交付 FAR 是 **0.0279** ⇒
**红线是定义在全量口径上的，本轮的窗口 FAR 只能做「相对板 vs 服务器」比较，不能做「过线/破线」判定。**
（板端在同窗口上甚至比服务器参考**更干净**：0.0480 vs 0.0656。）

## 六、复现命令（铁律九）

### 6.1 板卡侧（执行机 = RK3588 板卡 `root@192.168.0.64`）

```bash
# ① 抽帧（8bit 灰度 BMP；ffmpeg %06d 从 000001 起 ⇒ 源帧号 = 文件名 - 1）
bash manu/evaluation/board_runs/board_prepare_frames.sh \
  /tmp/lq_ir_640x512.mp4 /tmp/lqframes 2500

# ② 交付参数组逐帧融合轨迹导出（tag_mask 0x7 = SDK 默认；0x1 = 仅原生尺度）
bash manu/evaluation/board_runs/board_track_dump.sh \
  /tmp/lqframes /tmp/board_td 0x7 2500
bash manu/evaluation/board_runs/board_track_dump.sh \
  /tmp/lqframes /tmp/board_tm1 0x1 2500

# ③ 交付节奏实测（生产 25fps 推帧 / 消费者收结果 ⇒ Δn 语义）
bash manu/evaluation/board_runs/board_cadence_probe.sh /tmp/lqframes /tmp/cadence_run2.tsv 150

# ④ 跳帧（交付口径）轨迹导出：ids 表 = ③ 输出的 fid 列（只取 <2500 的）
#    软链目录：第 i 个软链指向 /tmp/lqframes/{id+1}.bmp
bash manu/evaluation/board_runs/board_track_dump.sh \
  /tmp/lqframes_j /tmp/board_jf 0x1 740 /tmp/delivered_run2.ids
```

环境：`LD_PRELOAD=/root/mosaic/libstatcompat.so`、
`LD_LIBRARY_PATH=/root/build:/root/mosaic/opencv:/root/mosaic:/root/hm_speed/lib:/lib:/lib/aarch64-linux-gnu:/usr/lib`
（两个脚本里已内置）。

### 6.2 主机侧打分（执行机 = host，工作目录 = `<workspace>/ultralytics`）

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python   # py3.8.20 / numpy 1.23.5
L=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/lqcache_rebuild/lookahead_residual/data

# ① 板端融合轨迹打分（逐帧口径 + 交付口径）
PYTHONPATH=. $PY manu/evaluation/eval_board_fused_tsv.py \
  --fused-tsv /tmp/boardwork/board_tm1_fused.tsv \
  --labels-root $L/labels --sequence VIDEO00007_19700101_001324 \
  --total-frames 2500 --out /tmp/boardwork/board_score_tm1.json

PYTHONPATH=. $PY manu/evaluation/eval_board_fused_tsv.py \
  --fused-tsv /tmp/boardwork/board_jf_fused.tsv \
  --labels-root $L/labels --sequence VIDEO00007_19700101_001324 \
  --total-frames 2500 --delivered-ids /tmp/boardwork/delivered_run2.ids \
  --out /tmp/boardwork/board_jf_score_fixed.json

# ② 候选流归因 + 合并半径扫描（离线复算板端候选）
PYTHONPATH=. $PY manu/evaluation/board_candidate_attribution.py \
  --board-cands /tmp/boardwork/board_td_cands.tsv --cache-root $L/ms_cache \
  --labels-root $L/labels \
  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
  --sequence VIDEO00007_19700101_001324 --window 2500 \
  --out-json /tmp/boardwork/attrib.json
```

⚠️ `eval_board_fused_tsv.py` 里的 `ULTRALYTICS_ROOT = "/media/manu/1TB-Volume/…"` 是**失效旧路径** ⇒ 必须靠 `PYTHONPATH=.` 才能 import 到 `manu.evaluation.*`。
⚠️ 本轮已修 `eval_board_fused_tsv.py` 的 **delivered 口径**：旧实现会把「已交付但融合无输出」的帧**从召回与 FAR 分母里一起删掉**，
现按服务器约定（`rt_fps_jitter_search.score_run`）把它们计入 FN。

## 七、板端节奏复测（与历史同带）

| 项 | run1（板上重编 `.so` `b383291e…`） | run2（交付 ocv48 `a8d56d26…`） | 早班报告 | §183 旧记录 | 离线仿真 s1 |
|:--|--:|--:|--:|--:|--:|
| pipeline_fps | **8.127** | **8.198** | 7.853 | 8.03~8.13 | 7.4 |
| Δn 均值 | **3.074** | **3.043** | 3.166~3.182 | 3.08~3.11 | 3.38 |
| 间隔均值 ms | 123.10 | 121.99 | 127.3 | 123 | —— |
| rknn_run p50 / gmc_fit p50 / post 均值 ms | 119.33 / 48.14 / 1.634 | —— | 117.7 / 49.3 / 0.85 | —— | —— |
| fallback summary | **clean** | **clean** | —— | —— | —— |

`fallback summary: clean` = 无相机队列丢帧 / 无跳特征 / 无跳检测 / 无降级 GMC / 无 VOIDED。

⚠️ **`mosaic_track_dump` 的 1:1 推帧模式实测 fps = 4.856**（2500 帧 / 514.8 s）：
那是**每帧等齐**的**延迟受限**值，**不是部署吞吐**（部署是相机异步推帧 ⇒ 8.1 fps）。两者不要混用。

## 八、结论与建议（分级）

| 级别 | 结论 | 动作 |
|:--|:--|:--|
| 🔴 **P0 交付阻塞** | SDK 默认 `tag_mask=0x7` ⇒ 同一目标多尺度峰各自成航迹 ⇒ 逐帧 FAR **0.3832**（红线 12.8×）。**服务器侧的 FAR 0.0279 是在「跨尺度合并后」的候选流上标定的，板端没有这一步** | ① **立刻**：板端调用侧把 `MosaicTrackFusionConfig.tag_mask` 设为 **0x1**（仅原生尺度）—— 本轮实测 FAR 0.0480 / F1 0.6726；② **正解**：SDK 候选阶段补一步**跨尺度合并**（半径 **16~24 px**，不是服务器的 768 px），本轮离线推算 FAR 0.0488~0.0540 / F1 0.6908~0.6931 —— ✅ **已于 §十二 落地并逐位验证**（C++ ≡ 服务器 Python 镜像；交付值 **20 px**） |
| 🔴 P0 | 早班报告「`dedup_radius=768` 在 SDK 的后处理阶段完成」是**错的** —— SDK 里根本没有候选级合并配置 | 改文档；**嵌入式组要拿这条施工单** |
| 🟢 P1 | 板端 C++ 融合引擎 ≡ 服务器 Python 参考（逐位一致） | 无需动作（这是好消息：融合层可放心交付） |
| 🟢 P1 | 板端交付口径 F1 0.6307 vs 服务器 0.6602（−2.95 pp，同一 741 帧交付集） | 板端 FP 多一倍，指向**候选流质量**（不是融合层） |
| 🟡 P2 | 板卡 `/root/mosaic/libmosaic_pipeline.so` 曾烤着**旧参数组**（FAR 0.0352 破线） | 已换 + 已备份；**上板前先核 md5** |
| 🟡 P2 | `eval_board_fused_tsv.py` 的 delivered 口径会删掉「无输出」的已交付帧 | 已修为服务器约定 |

## 九、产物清单

| 内容 | 路径 | md5 |
|:--|:--|:--|
| 板端融合轨迹（tag_mask=0x7，1397 行） | `/tmp/boardwork/board_td_fused.tsv` | `5a317dab22bb28497e93b3db77927270` |
| 板端融合轨迹（tag_mask=0x1，1341 行） | `/tmp/boardwork/board_tm1_fused.tsv` | `ce56c5c693364ae60d059db68a247788` |
| 板端跳帧融合轨迹（1170 行，含 775 个 infilled） | `/tmp/boardwork/board_jf_fused.tsv` | `e49dfb18d3e17cb1d51b81c204f9efe5` |
| 板端原始候选流（2500 行） | `/tmp/boardwork/board_td_cands.tsv` | `0e511a4c66a7f2842c79c0a9f73fe613` |
| 交付帧号表（741 帧，mean Δn 3.041） | `/tmp/boardwork/delivered_run2.ids` | `0cbab6c102e62acaffbf5f53707c4901` |
| 候选流归因 + 半径扫描结果 | `/tmp/boardwork/attrib.json` | `58c751df0be736d5a3ed43458c6afdf9` |
| 板端打分（tag_mask=0x7） | `/tmp/boardwork/board_score.json` | `2bcd2cf15774f361f7d7f30a0c0b8398` |
| 板端打分（tag_mask=0x1） | `/tmp/boardwork/board_score_tm1.json` | `c14b8f682c5151c0c3eaec0785854c8c` |
| 板端跳帧打分（交付口径） | `/tmp/boardwork/board_jf_score_fixed.json` | `990afc45df8fdd4e1a42480a4dc70ae5` |
| 板上重编 `.so` | `/root/build/libmosaic_pipeline.so`（板卡） | `b383291e31ccbd71c46430ca69869360` |
| 交付 ocv48 `.so`（换掉旧参数组那份） | `/root/mosaic/libmosaic_pipeline.so`（板卡） | `a8d56d26d3604d5c33d69a859c598a78` |
| 交付模型 | `/root/mosaic/runtime/ir_uav_det.rknn`（板卡） | `1a3faf2f9459cad8a389ffa81810234c` |
| 源视频 | `/tmp/lq_ir_640x512.mp4`（板卡） | `dfe12ac9a9271d7c1a4f0d73360f08c5` |

⚠️ `/tmp/boardwork/` 是本轮的**临时**目录（重开机会丢）。要长期保留请把上面 9 个文件拷进
`manu/memory/` 或 `manu/evaluation/` 下的固定目录。

## 十、遗留 / 本轮未做

1. **板卡磁盘**：根分区 95% 满（剩 871 MB）；`/tmp/lqframes`（792 MB，tmpfs）用完即删。
2. **服务器 `192.168.99.40:32222` 公钥仍未授权** ⇒ 拿不到服务器侧守望进程与 `arms/` 队列 ⇒
   本轮**没能**核实 `board_t4_watch` 的让路判定为何没拦住并发（实测它在 13:18 / 13:33 两次与本会话**并发**跑 NPU，
   幸而 dmesg 全程无 `job timeout` / `soft reset`）。
3. **host NFS 交换区** `/run/media/manu/SSD960-LOCAL/nfs/mosaic_sdk` 仍 **只读 + 导出表空** ⇒ 交付包没到板卡，需用户 sudo。
4. **交付包里的 `out/mosaic_cadence_probe` / `out/mosaic_track_dump` 是 09:07 产物**，**不含** 11:59 源码里的显式 9 字段覆盖（字节级审计已确认）⇒ 若要交付物与源码严格对齐，需重编这两个二进制并刷新 MANIFEST/README md5。
5. **未跑全序列**：板卡 tmpfs 只放得下 2500 帧 ⇒ 全量 11810 帧的板端 GT 口径**未做**（且本窗口是困难片段，绝对 FAR 不可判红线）。
6. **目标板 `.211` 仍断电**（`No route to host`）⇒ 真正的交付板（OpenCV 4.6）**未验证**。

**耐久登记点**：本条（主档 §262 / 精简档 §262）可能被并行任务整档重写覆盖，
**权威入口 = 本文件 `manu/memory/trial0080_board_gt_scoring_20261120.md`**。

---

## 十一、附录：第二条独立路径的复算互证（2026-11-20 14:00~15:10，主档 §264）

> 本节**不是**本报告的复述，而是**另一条独立的计算路径**（本机离线复算，非板卡在线打分），
> 目的是让「根因 + 修复区间」不依赖单一路径。两条路径在半径扫描上**四位小数全等**。

### 11.1 路径差异

| | 本报告（§四~§八） | 本附录（§264） |
|:--|:--|:--|
| 计算位置 | **板卡在线**（`/tmp/boardwork/`，板端打分） | **本机离线**（`manu/evaluation/board_far_dedup_probe.py`） |
| 融合执行 | 板卡 C++ `mosaic_track_fusion` | 服务器 Python 镜像 `sdk_track_fusion_ref.py --physical-frame-index` |
| 去重施加位置 | 板端候选流 | 同一条规则（服务器 `_merge_scale_rows`：按分数降序、R px 内丢弃） |
| GT 来源 | `lqcache_rebuild/lookahead_residual/data/labels/` | 服务器 `Final_Labels_merged_20261010`（同名同格式） |

⇒ 输入数据同源（板卡候选流 md5 `0e511a4c…` 逐位相同），但**计算路径、融合实现、GT 路径都不同**。

### 11.2 半径扫描互证表

| R (px) | 本附录 F1 / FAR | 本报告（§八 / attrib.json） | 一致？ |
|--:|:--|:--|:--:|
| 0（板端现状） | 0.568947 / **0.383200** | FAR **0.3832** | ✅ |
| 16 | 0.690751 / **0.054000** | FAR 0.0540 | ✅ |
| 20 | 0.693059 / **0.048800** | FAR 0.0488 | ✅ |
| 24 | 0.692703 / **0.049600** | FAR 0.0496（F1 0.6908~0.6931） | ✅ |
| 768（服务器口径） | 0.685304 / 0.019600 | — | — |
| 服务器参考（同窗口同 GT） | 0.691895 / 0.055200 | — | ✅ |

⇒ **平台区 R 20~64 px** 两条路径一致 ⇒ **根因（候选流缺跨尺度合并）与修复区间（16~24 px）双向确认**。

### 11.3 本附录新增的脚本（可复现）

| 脚本 | 作用 |
|:--|:--|
| `manu/evaluation/board_far_parity_check.py` | 板端轨道 ↔ 服务器轨道**逐帧最近邻比对**（位置/分数偏差分布）+ 同窗口同 GT 同匹配器打分 |
| `manu/evaluation/board_far_dedup_probe.py` | 跨峰去重半径扫描：每档都重跑服务器 Python 融合镜像再打分 ⇒ 排除「只有板端融合错」的可能 |

### 11.4 板端产物**已抢救归档**

本报告第九节列的 `/tmp/boardwork/` 在本轮结束时**已随板卡 tmpfs 消失** ⇒ 已从板卡 `/tmp/board_*.tsv` 抢救，
落档 **`manu/memory/board_far_20261120/board/`**（含 `README.md` md5 清单 + 板卡原生重编的 SDK 源码与 `.so`）：
`board_td_cands.tsv` `0e511a4c…` / `board_td_fused.tsv` `5a317dab…` / `board_tm1_fused.tsv` `ce56c5c6…` /
`board_jf_*.tsv` / `delivered_run2.ids` `0cbab6c1…` / `src/mosaic_track_fusion.h` `dba851ed…`（**与交付包逐位相同** ✅）。

### 11.5 补一条**可绕过 NFS** 的搬帧路径（给「24 序全量板端 FAR」用）

板卡 → 服务器 `192.168.99.40:32222` **TCP 实测可达** ⇒ 帧可由**服务器直接 scp 到板卡**
（服务器侧有 `ptq-board` 的 ssh 配置），**不必等 host 的 `exportfs -ra`**；
host `192.168.0.115:22` 对板卡是 `Connection refused` ⇒ 走 host 中转反而不通。

---

## 十二、SDK 侧的正解：跨尺度候选合并已落地（2026-11-20 13:5x~14:0x，本机实现 + 逐位验证）

> §四 定位的根因（候选流缺跨尺度合并）已经在 **SDK 源码里补上了**，
> 并且**逐位证明**与服务器 Python 镜像等价 ⇒ §4.4 的离线推算数字**从此可在板端直接复现**。

### 12.1 改了什么（4 个文件 + 2 个接口）

| 文件 | 改动 |
|:--|:--|
| `manu/pipeline/npu/mosaic_track_fusion.h` | 新增 `void MergeCandidatesByRadius(std::vector<TrackFusionCand>*, double radius)`（**候选构造**阶段，零第三方依赖 ⇒ host / 板卡都能原地编译） |
| `manu/pipeline/npu/mosaic_track_fusion.cpp` | 实现该函数：候选池按**分数降序稳定排序** → 贪心保留（与**已保留**候选距离 <= radius 的丢弃）⇒ 与服务器 `eval_multiscale_cache_th._merge_scale_rows` **逐条等价**；`radius <= 0` 原样返回（与改前逐位一致） |
| `manu/pipeline/npu/mosaic_pipeline.h` / `.cpp` | 新增公开接口 `mosaic_pipeline_set_candidate_dedup_radius(MosaicPipeline*, double)`；`FeedFusion()` 在 tag_mask 过滤**之后**、喂引擎**之前**调用合并；`radius<=0` 打 `[warn]`、`radius>0` 打 `[i] candidate dedup ON: radius=…` |
| `manu/pipeline/npu/mosaic_track_dump.cpp` / `mosaic_cadence_probe.cpp` | **显式**打开合并（交付值 **20 px**，可用带 16~24），CLI 新增 `--cand-dedup-radius` |

🔴 **为什么是单独一个接口、而不是往 `MosaicTrackFusionConfig` 里加字段**：
`MOSAIC_TRACK_FUSION_VERSION` 是**严格校验**的（`!=` 直接 `-2`）⇒ 加字段必须升版本 ⇒
所有按 v2 编译的调用侧会被**直接拒绝**（融合静默不生效）。单开接口则**老二进制照常工作**。

### 12.2 验证（本机，x86_64，**无 GPU / 无 NPU / 无训练**）

**编译**（零第三方依赖，与交付构建同款数值开关）：

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics/manu
g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations \
    -o /tmp/align3/track_fusion_align_test \
    pipeline/npu/track_fusion_align_test.cpp pipeline/npu/mosaic_track_fusion.cpp -I pipeline/npu
```

**验证脚本**（新增，仓库相对路径 `manu/evaluation/board_cand_merge_verify.py`）：
对每个半径跑 **C++ 合并 + C++ 融合**，与**服务器 Python 镜像**的输出**逐字节比对**，再用同一套打分器打分。

```bash
PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python
PYTHONPATH=. $PY manu/evaluation/board_cand_merge_verify.py \
  --cands manu/memory/align_dn_20261120/real_0080/real_cands.tsv \
  --params manu/memory/align_dn_20261120/params_delivery.txt \
  --labels-root /tmp/boardfar/gt --sequence VIDEO00007_19700101_001324 \
  --window 0 2500 --radii 0,16,20,24,32 \
  --python-mirror-dir /tmp/boardfar/probe --out manu/memory/board_far_20261120/merge_verify.json
```

**结果**（`manu/memory/board_far_20261120/merge_verify.json`）：

| R (px) | 与 Python 镜像逐字节 | F1 | Recall | Precision | **FAR** | tp / fp / fn |
|--:|:--|--:|--:|--:|--:|--:|
| 0 | ✅ 一致 | 0.5689473 | 0.554959 | 0.583659 | **0.3832** | 1343 / 958 / 1077 |
| 16 | ✅ 一致 | 0.6907507 | 0.557025 | 0.908968 | 0.0540 | 1348 / 135 / 1072 |
| **20** | ✅ 一致 | **0.6930591** | 0.557025 | 0.917007 | **0.0488** | 1348 / 122 / 1072 |
| 24 | ✅ 一致 | 0.6927030 | 0.557025 | 0.915761 | 0.0496 | 1348 / 124 / 1072 |
| 32 | ✅ 一致 | 0.6930591 | 0.557025 | 0.917007 | 0.0488 | 1348 / 122 / 1072 |
| 服务器参考（同窗口） | —— | 0.6918947 | 0.559091 | 0.907445 | 0.0552 | 1353 / 138 / 1067 |

⇒ **R=0 的输出与已归档的 `python_fused_dn1.tsv` 逐字节相同**（md5 `839e1f43…`）
⇒ 回归零风险（不改默认行为）。
⇒ **R=16/20/24/32 与服务器 Python 镜像逐字节相同** ⇒ §4.4 的离线数字**就是板端会出的数字**。
⇒ 开到 **20 px** 后板端 vs 服务器：F1 **+0.0012**、FAR **−0.0064**、匹配率 **82.5%**（改前是 −0.1229 / +0.3280）。

### 12.3 同步与产物

- **派生树已同步**（仓库源为准）：`manu/pipeline/npu/build/src/npu/`、`manu/pipeline/npu/build/mosaic_pipeline.h`、
  `rk3588_app_base/mosaic_sdk/src/npu/`（**交付给嵌入式组的那份**，6 个文件 md5 逐个相同）。
- **产物**：`manu/memory/board_far_20261120/` —— `merge_verify.json`（`9418616a…`）、
  `fused_cpp_r0.tsv`（`839e1f43…`）、`fused_cpp_r20.tsv`（`dd55cb09…`）。
- **编译检查**：`mosaic_pipeline.cpp` / `mosaic_track_dump.cpp` / `mosaic_cadence_probe.cpp`
  本机 `-fsyntax-only` 全过（`-I build/include -I build/include/rga` + 本地 OpenCV 4.10 头，仅 `mosaic_core.h` 一条**既有**的 `%ld` 格式告警）。

### 12.4 还没做的（**需要授权**）

1. **板端在线复测**：把新 `.so` 编到板卡、跑 2500 帧 NPU 推理（≈8.5 min）⇒ 属**推理**，按铁律一/十三**需用户当轮授权**；
   预期结果就是上表 R=20 那一行（F1 0.6930591 / FAR 0.0488）。
2. **全序列（24 序 / 165,223 帧）板端 FAR**：板卡 tmpfs 只放得下 2500 帧；按 §11.5 的新路径可由**服务器直接 scp 帧到板卡**，绕开 host 的 NFS 导出。
3. **交付包刷新**：`.so` 与 `out/mosaic_track_dump` / `out/mosaic_cadence_probe` 需重编 + MANIFEST/README md5 基线重算（**既有**遗留项，非本次新增）。

---

## 十三、交付包（as-shipped）板端闭环：两道等价闸门 ✅ ＋ 交付包自洽性根因修复

> **本节回答的问题**（与 §一~§十二 都不同）：**客户拿到手的那份交付包
> （`build/mosaic_sdk_20261120*`），是不是板卡上真正跑过、且与它自己的源码自洽的那一份。**
> 执行时间 2026-11-20 13:45~15:00；主档 `manu/memory/memory.md` §267。

### 13.1 C1 闸门：as-shipped 交付包 2500 帧全量验证 ✅

跑的是**包内自带**的 `.so` + 探针（探针 `RUNPATH=$ORIGIN` ⇒ 自找同目录 `.so`）⇒ 验证的是「整包搬过去就能跑」这件事本身。

| 项 | 实测 |
|:--|:--|
| 脚本 / 日志 | 板卡 `/root/measure_hm_speed_delivery_verify_C1.sh` / `/root/verifyC1_20261120_135043.log`（副本已归档） |
| 跑的东西 | `/mnt/manu/mosaic_sdk/out/mosaic_track_dump`（`5a854a07…`）+ `/mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so`（`b383291e…`） |
| 规模 | 2500 帧：`pushed=2500 updates=2500 cands=26493 fused_dets=2301 dropped=0 features_dropped=0` |
| 节奏 | `wall_ms=516335.5 fps=4.842` —— **1:1 推帧 ⇒ 延迟受限值，不是部署吞吐**（异步推帧仍 8.1 fps） |
| **融合输出 md5** | `5a317dab22bb28497e93b3db77927270` ⇒ **= 参照 `board_td_fused.tsv`，逐字节相同** |
| **候选输出 md5** | `0e511a4c66a7f2842c79c0a9f73fe613` ⇒ **= 参照 `board_td_cands.tsv`，逐字节相同** |
| 运行时参数回读 | `th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00 tags=0x7 reorder_window=0 physical_frame_id=1 max_age_phys=-1 r_slope=5.00 alpha_decay=1.000 confirm_consec=2 confirm_gap_tol=8` ⇒ **9 字段全在** |
| 板卡健康 | 全程 `dmesg` **无 `job timeout` / 无 `soft reset`**；uptime 连续 2 h+ 未重启 |

### 13.2 C2b 闸门：交付树 09:10 产物 `.so` vs 11:59 源码重编 `.so` ⇒ 行为逐字节等价 ✅

| 变体 | `.so` md5 | 实测加载（`LD_TRACE_LOADED_OBJECTS=1`） | 融合 md5 | 候选 md5 |
|:--|:--|:--|:--|:--|
| oldso | `a8d56d26`（交付树 09:10 产物） | `/root/c2_oldso/libmosaic_pipeline.so` | `b1e8eb4401fb6dcd518d827cdec2a2dc` | `959963cad3d9d520d01d7b84e123b834` |
| newso | `b383291e`（11:59 源码板上重编） | `/root/c2_newso/libmosaic_pipeline.so` | **同上** | **同上** |

- 同 250 帧 ⇒ **两版 `.so` 输出 md5 完全相同** ⇒ **行为逐字节等价**；
  且与 **C1 前 250 帧切片**的 md5 相同 ⇒ **三方一致**（整包 / 独立跑 / 旧 `.so`）。
- **唯一差异**：旧 `.so` 的回读行少打 `th_salvage` 与 `match_dist`（格式串老），
  其余字段值**全部正确** ⇒ **结构体无错位、ABI 安全**
  （`MosaicTrackFusionConfig` 自 v2 起只在**尾部追加**字段，且 `version` 由 `.so` 自己写、自己校）。
- 导出符号集**完全相同**；`b383291e` 比 `a8d56d26` 少依赖 `libopencv_features2d` / `libopencv_flann`。

### 13.3 🔴 踩坑（差点误判）：**`RUNPATH($ORIGIN)` 排在 `LD_LIBRARY_PATH` 之后**

C2 第一版（`phaseC2.sh`）把变体目录只放进 `RUNPATH`，而 `LD_LIBRARY_PATH` 含 `/root/mosaic`
（那里也有一份 `libmosaic_pipeline.so`）⇒ **两个变体实际加载的是同一个 `.so`** ⇒ 输出自然相同，
**差点被当成「两版等价」写进结论**。修正版 `phaseC2b.sh` 把变体目录放在 `LD_LIBRARY_PATH` **最前**，
并用 `LD_TRACE_LOADED_OBJECTS=1` 实证加载路径。

> **新增操作纪律（换 `.so` / 换库做 A/B 时强制）**：① 变体目录放 `LD_LIBRARY_PATH` 最前；
> ② 每次跑用 `LD_TRACE_LOADED_OBJECTS=1` **打印实际加载的文件**；
> ③ 别拿「输出相同」当「A 与 B 等价」——**先证明加载的是不同的东西**。

### 13.4 交付树回灌 ＋ 交付包自洽性修复

| 层 | 改了什么 |
|:--|:--|
| 探针 | `3fe28c17`（cadence）/ `5a854a07`（track_dump）—— **零 OpenCV 依赖** ⇒ **ocv46/ocv48/ocv42 三变体通用** |
| 库 | `.so` `b383291e` + `.o` `470de6a1` / `3a13ee43` ⇒ **只属 ocv48 变体**（链 `libopencv_*.so.408`） |
| 回灌位置 | 探针 → 5 棵树的 `out/`；`.so`+`.o` → **仅** `build/mosaic_sdk_20261120_ocv48/out/` |
| 口径缺陷 | MANIFEST **漏登记 2 个探针源码**（声称 67 实为 **69**）、「`out/` 8 个产物」实为 **10**、README「共 76 个文件」实为 **80** |
| 🔴 根因 | `build/mosaic_sdk_*` 由 **`packaging/make_build_dir.sh` 从 `MANIFEST.template.md` / `README.template.md` 生成** ⇒ 手改生成物会被下次打包覆盖 ⇒ 改的是**模板 + 打包脚本** |
| 回验 | 三棵树的 README/MANIFEST **由模板重新生成** ⇒ **69 行 md5 逐条 0 不一致**、零残留 `{{}}`、计数 69/10 与磁盘实况一致；`bash -n` 通过；27 个占位符全部有替换规则 |

### 13.5 归档产物（本目录）

| 文件 | md5 | 说明 |
|:--|:--|:--|
| `ship_td_fused.tsv` | `5a317dab22bb28497e93b3db77927270` | C1 整包 2500 帧融合输出（= 参照） |
| `ship_td_cands.tsv` | `0e511a4c66a7f2842c79c0a9f73fe613` | C1 整包 2500 帧候选流（= 参照） |
| `ship_td_tracked.tsv` | `7aa855aa01ce593162bfb5369227a506` | C1 逐帧航迹输出 |
| `c2b_oldso_td_fused.tsv` | `b1e8eb4401fb6dcd518d827cdec2a2dc` | 旧 `.so` 的 250 帧融合输出 |
| `c2b_newso_td_fused.tsv` | `b1e8eb4401fb6dcd518d827cdec2a2dc` | 新 `.so` 的 250 帧融合输出（**与上一行 md5 相同 ⇒ 等价**） |
| `verifyC1_20261120_135043.log` | — | C1 板卡日志（含 9 字段回读原文） |
| `verifyC2b_20261120_140423.log` | — | C2b 板卡日志（含两变体 `LD_TRACE_LOADED_OBJECTS` 实证） |
| `phaseC1.sh` / `phaseC2b.sh` | — | 两个闸门脚本原件 |

### 13.6 遗留

1. 🔴 **需用户 sudo**：host NFS 交换区只读 + `/proc/fs/nfsd/exports` 空 ⇒ **NFS 交付通道断**（本轮用 scp 绕过）；
   目标板 `.211` 仍 `No route to host` ⇒ **真交付板（OpenCV 4.6）未验证**。
2. **上游两棵树**（`ultralytics/manu/pipeline/npu/build/` = 打包 `DEST`、`rk3588_app_base/mosaic_sdk/`）的
   MANIFEST/README **未刷新** —— 13:58~14:01 并发线正在改这两个树的 `src/npu`
   （落地 §十二的跨尺度合并，探针源码已变成 `f7040045…` / `7f8e9038…`）⇒ **本节不与其抢文件**；
   等并发线收工后按 §13.4 的模板口径重生成一次即可。
3. **ocv46 / ocv42 变体的预编译 `.so` 仍是 09:1x 产物**（板卡只有 OpenCV 4.8）⇒ 行为已证等价，
   但**回读行少 2 个字段**；要 9 字段回读需在带对应 OpenCV 的机器上跑 `build_lib.sh` 重编。
4. **未跑全序列**（板卡 tmpfs 只放得下 2500 帧）。
5. **交付包需再打一次**：并发线的跨尺度合并（§十二）还没进交付包；本节结论对应 **11:59 源码**这一版。

---

## 十四、板端「在线」复测：真 NPU 跑通 2500 帧（2026-11-20 14:05~14:31，主档 §269）

> §十二 把修复写进了 SDK 源码并在本机逐位验证，但当时标注「**板端在线复测**未做（属推理 ⇒ 需用户当轮授权）」。
> 本节把它**做掉了**：授权依据 = 本会话目标「完成遗留的板卡验证和对齐工作，注意只要 0080 和 v3 osd」。
> **未跑训练、未跑全 24 序、未下载数据集。**

### 14.1 流程（铁律零：本地改 → 推送 → md5 双侧核验；板上**不直接改代码**）

| 步骤 | 动作 | 校验 |
|:--|:--|:--|
| ① 源码上板 | 仓库源 `manu/pipeline/npu/`（§十二 的合并实现）6 个文件 → host `/tmp/merge_push_20261120_b/` → 服务器 → **板卡 `/root/merge_20261120/src/`** | 板卡 md5 逐个 == 仓库源（`b3f7d9b2…` / `7691fd38…` / `98403eaa…` / `95cfea75…` / `7f8e9038…` / `cafd28e4…`） |
| ② 板上重编 | `cp -a /root/sdk /root/merge_20261120/sdk` → 覆盖合并后的源 → 按 `build_sdk.sh` **同一套选项**编到 `/root/merge_20261120/out` | `libmosaic_pipeline.so` **`e1a55e9bde7cb3235dee1cb0ae61e133`**、`mosaic_track_dump` `5cd9a01b…`、`mosaic_cadence_probe` `ca940674…`、`track_fusion_align_test` `2821a093…` |
| ③ 符号自检 | `nm -D` ⇒ `MergeCandidatesByRadius` **在**；`strings` ⇒ `--cand-dedup-radius` **在** | 合并**真的**编进去了（不是死代码） |
| ④ 在线两轮 | 同一段 2500 帧 `/tmp/lqframes`、同一模型 `ir_uav_det.rknn`（`1a3faf2f…`）、同一交付参数组、`--tag-mask 7`：**r=0 对照** + **r=20 实测** | 各 ≈8.6 min（`fps=4.859 / 4.860`），`dropped=0 features_dropped=0` |

⚠️ **板卡串行铁律**：同一时刻只允许一个 NPU 上下文（并发 ⇒ RKNPU job timeout ⇒ soft reset ⇒ 只能物理断电）
⇒ 跑之前确认 `ps -eo comm | grep -E "rknn|e2e"` 空。

### 14.2 在线结果（打分器 = `manu/evaluation/board_far_parity_check.py`；GT 2,420 帧；窗口 0..2499；`inbox` tol=8）

| 指标 | **板卡 r=0（对照）** | **板卡 r=20（交付值）** | 服务器参考（同窗口同 GT 同匹配器） |
|:--|--:|--:|--:|
| FAR（逐帧） | 0.3832000 | **0.0488000** ✅ | 0.0552000 |
| F1 | 0.5689473 | **0.6930591** | 0.6918947 |
| recall / precision | 0.554959 / 0.583659 | 0.557025 / 0.917007 | 0.559091 / 0.907445 |
| TP / FP / FN | 1343 / 958 / 1077 | 1348 / **122** / 1072 | 1353 / 138 / 1067 |
| 板↔服务器轨道匹配率 | 53.542% | **83.673%** | —— |
| Δ（板 − 服务器） | F1 −0.1229474 / FAR **+0.3280** | F1 **+0.0011645** / FAR **−0.0064** | —— |

⇒ **板端开合并后已反超服务器参考**（FAR 低 0.0064、F1 高 0.0012、FP 少 16）。
⚠️ **绝对 FAR 不可判 0.03 红线**（本窗口服务器自己就是 0.0552，全序列口径才是 0.0262）⇒ 只做同口径「板 vs 服务器」比较。

### 14.3 🔴 对照组的价值：重编**零行为漂移**

| 产物 | 板卡 r=0（新 `.so`，合并关） | 已归档旧交付产物 | 结论 |
|:--|:--|:--|:--|
| 融合轨迹 | `mg_td_r0_fused.tsv` **`5a317dab22bb28497e93b3db77927270`** | `board_td_fused.tsv` `5a317dab…` | **逐字节相同** ✅ |
| 原始候选流 | `mg_td_r0_cands.tsv` **`0e511a4c66a7f2842c79c0a9f73fe613`** | `board_td_cands.tsv` `0e511a4c…` | **逐字节相同** ✅ |
| r=20 的候选流 | `mg_td_r20_cands.tsv` `0e511a4c…` | —— | 与 r=0 相同 ⇒ 合并在**候选构造阶段**（引擎之前），原始候选流不受影响 |

⇒ **FAR 从 0.3832 掉到 0.0488，唯一来自跨尺度合并**（候选流逐字节未变、融合层逐位未动）。

### 14.4 三侧逐字节一致（**合并开启**状态下同样成立）

| 层 | r=0（合并关） | r=20（合并开） |
|:--|:--|:--|
| 主机 C++（x86_64）↔ 服务器 Python 镜像（4 参数组 × 6 Δn 图案） | **24/24 逐字节** | **24/24 逐字节** |
| 主机 C++ r=0 ↔ 已归档 `HOST_GOLDEN_MD5.txt` | **24/24 md5 相同** | —— |
| **板卡 C++（aarch64）↔ 主机 C++** | **24/24 md5 相同** | **24/24 md5 相同** |

- 闸门 `manu/evaluation/align_dn_delta_check.py` 新增 `--dedup-radius`（**同时**喂 C++ `--dedup-radius` 与 Python `--dedup-radius`）；
  Python 镜像 = `manu/evaluation/sdk_track_fusion_ref.py --dedup-radius`。
- 🔴 **参数 JSON 的 `dedup_radius=768` 默认不采用**（768 px ⇒ 每帧只留 1 候选；本机实测 3.377 → **0.995** 候选/帧 ⇒ 丢次要目标）；
  **板端正确值 16~24 px（交付 20）**；闸门要显式 `--dedup-radius-from-params` 才采用 768。

### 14.5 板端运行时回读（一行对账）

```
[i] track fusion ON: th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00
                     tags=0x7 reorder_window=0 physical_frame_id=1 max_age_phys=-1 r_slope=5.00
                     alpha_decay=1.000 confirm_consec=2 confirm_gap_tol=8
[i] candidate dedup ON: radius=20.0 px (cross-scale merge)
```
⇒ §三 要求的 **9 个交付字段**在新 `.so` 上全部显式可见（`max_age_phys=-1` = 不设），**合并开关也可见**。

### 14.6 产物（本轮新增，已拉回本机 `manu/memory/board_far_20261120/`）

| 内容 | 路径 | md5 |
|:--|:--|:--|
| 板端 r=0 融合轨迹（对照） | `board_merge/mg_td_r0_fused.tsv` | `5a317dab22bb28497e93b3db77927270` |
| 板端 r=0 原始候选流（对照） | `board_merge/mg_td_r0_cands.tsv` | `0e511a4c66a7f2842c79c0a9f73fe613` |
| 板端 r=20 融合轨迹（**实测**） | `board_merge/mg_td_r20_fused.tsv` | `51c4ef144a172f829c2d40e05ebd89e1` |
| 板端 r=20 原始候选流（与 r=0 同） | `board_merge/mg_td_r20_cands.tsv` | `0e511a4c66a7f2842c79c0a9f73fe613` |
| 板端 r=0 / r=20 tracked 输出 | `board_merge/mg_td_r0_tracked.tsv` / `mg_td_r20_tracked.tsv` | `7aa855aa01ce593162bfb5369227a506` / `34dd9c12616246fbbf1b55bf0ae4bf44` |
| 板端对齐闸门 md5（r=0 / r=20，各 24 条） | `board_merge/BOARD_R0_MD5.txt` / `BOARD_R20_MD5.txt` | —— |
| 板端两轮运行日志 | `board_merge/board_merge_run.log` | —— |
| 打分结果（板 r=0 / 板 r=20） | `parity_board_r0_merge.json` / `parity_board_r20_merge.json` | —— |

### 14.7 复现命令（铁律九）

```bash
# ── 板卡侧（root@192.168.0.64，经服务器跳板；助手 = rk3588_app_base/board_sh.sh）─────────
bash rk3588_app_base/board_sh.sh 'bash /root/merge_20261120/build.sh'   # 重编 .so + 探针
bash rk3588_app_base/board_sh.sh 'bash /root/merge_20261120/align.sh'   # 板卡侧 24 组对齐（r=0 + r=20）
bash rk3588_app_base/board_sh.sh 'bash /root/merge_20261120/run.sh'     # 2500 帧 NPU 两轮

# ── 主机侧（执行机 = 本机 x86_64；工作目录 = <workspace>/ultralytics）────────────────
PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python
g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations \
    -o /tmp/align_merge/track_fusion_align_test \
    manu/pipeline/npu/track_fusion_align_test.cpp manu/pipeline/npu/mosaic_track_fusion.cpp \
    -I manu/pipeline/npu
for R in 0 20; do
  PYTHONPATH=. $PY -u manu/evaluation/align_dn_delta_check.py \
    --cands manu/memory/cadence_ab_20261119/vid09_align/v09_cands.tsv \
    --cpp-bin /tmp/align_merge/track_fusion_align_test --workdir /tmp/align_gate_merge/r$R --python $PY \
    --params delivery=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
    --params v3=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
    --params oldgapfit=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_gapfit_params.json \
    --params th045=manu/evaluation/system_sota_longquan_trial0080_delivery_params.json \
    --cadence-tsv manu/memory/cadence_ab_20261119/cad_on.tsv --dedup-radius $R
done     # 期望：r=0 与 r=20 各「24/24 个 (参数组 x delta-n 图案) 组合逐字节一致」
PYTHONPATH=. $PY manu/evaluation/board_far_parity_check.py \
  --board-tsv manu/memory/board_far_20261120/board_merge/mg_td_r20_fused.tsv \
  --server-preds /tmp/boardfar/preds_cc2_md10_VIDEO00007.json --labels-root /tmp/boardfar/gt \
  --sequence VIDEO00007_19700101_001324 --window 0 2500 \
  --out manu/memory/board_far_20261120/parity_board_r20_merge.json
```

### 14.8 仍未闭环

---

## 十五、跳帧（交付口径）板端打分：跨尺度合并的板端收益确认（2026-11-20 15:00~15:15，主档 §275）

### 15.1 工况

- 板卡 `evm3588`（`root@192.168.0.64`，aarch64），`.so` `libmosaic_pipeline.so` = `e1a55e9bde7cb3235dee1cb0ae61e133`（与 §十四 同一份，含 `MergeCandidatesByRadius`）。
- **741 帧交付表** `manu/memory/board_far_20261120/board/delivered_run2.ids`（md5 `0cbab6c102e62acaffbf5f53707c4901`，mean Δn 3.041，帧号 0..2250）；GT = **731**。
- **两组严格可比**：**同一 `tag_mask=7`**，仅合并半径不同 —— **r=0（关）** vs **r=20（开，交付值）**。
- 板端：`pushed=740 updates=740 cands=9273`，`fps=4.335/4.334`，`dropped=0 features_dropped=0`（`board_merge/board_jf_merge.log`）。

### 15.2 结果（交付口径 = 只在 741 个交付帧上打分）★主判据

| 项 | r=0（合并关） | **r=20（合并开）** | 变化 |
|:--|--:|--:|:--|
| 有输出帧 | 493 | 460 | −33 |
| tp / fp / fn / gt | 442 / 364 / 289 / 731 | **408 / 73 / 323 / 731** | **FP −291（−80%）** |
| Recall | 0.604651 | 0.558140 | −4.65 pp |
| Precision | 0.548387 | **0.848233** | **+29.98 pp** |
| **F1** | 0.575146 | **0.673267** | **+9.81 pp** |
| **FAR** | 0.491228 | **0.098516** | **−80.0%（5.0×）** |

全窗口口径（2500 帧，仅参考）：FAR 0.4008 → 0.0756；F1 0.53322 → 0.60779。

### 15.3 🔴 方法学坑：「all 口径」含**从未推送的帧**

独立复算 `fused.tsv`（第 1 列 = 物理帧号，第 2 列 = 该帧检测数）：
**r=0 输出帧 1,380 = 交付帧 492 + 未推送帧 888**；**r=20 = 459 + 未推送帧 818**。

原因：跳帧稀疏（mean Δn 3.041）+ `confirm_gap_tol=8` ⇒ 跟踪器在**跳过的帧上插值出检测行**（`fused.tsv` 只写非空帧）⇒ **「all」口径把 60% 的输出帧算在根本没推过检测的帧上**。

⇒ **判 FAR 只能用交付口径**；「all」口径的 0.4008 / 0.0756 **不可用于过线判定**，也不可与 §5.1 的窗口 FAR 直接比。
⚠️ 打分器自报 `delivered_frames_with_output` 493 / 460，独立复算 492 / 459（差 1，交付表末帧 `2250` 边界）⇒ 以打分器 JSON 为准。

### 15.4 与 §五 基线的对照（⚠️ `tag_mask` 不同，不可直接相减）

| 来源 | 口径 | F1 | FAR |
|:--|:--|--:|--:|
| 本轮 **r=20** | tag_mask=**7** + 合并开 | **0.673267** | **0.098516** |
| 本轮 r=0 | tag_mask=**7** + 合并关 | 0.575146 | 0.491228 |
| §五 基线（2026-11-20 早） | tag_mask=**1** + 合并关 | 0.6306769 | 0.0917679 |
| 服务器离线参考 | 交付参数组 | 0.6602452 | 0.0458839 |

- **★ 核心结论**：**跨尺度合并让全尺度 `tag_mask=7` 首次成为可用口径**（合并关时其 FAR 高达 0.4912）；r=20 的 F1 **0.6733** 比 §五 基线（0.6307）**高 4.26 pp**，FAR 基本持平（0.0985 vs 0.0918）。
- ⚠️ 板端 vs 服务器参考的**结构性差距未变**（FAR 2.15×，§五 为 2.0×）⇒ 与 §5 结论一致（板端候选流更「毛边」），**非本轮引入**。

### 15.5 产物与复现

- `manu/memory/board_far_20261120/board_merge/`：`mg_jf_r0_fused.tsv`（`9d79405be899c06a3911f500002b4bda`）/ `mg_jf_r20_fused.tsv`（`7ceb41c3a7bbbdeb35a31083ee98dd8b`）/ `mg_jf_r0_cands.tsv` 与 `mg_jf_r20_cands.tsv`（**md5 相同 `7b7759be224b0a273d14d33b71565dc1`** ⇒ 合并在候选构造阶段、原始候选流未改）/ `mg_jf_r0_tracked.tsv`（`c825d624051a9d22f67b8488418152fd`）/ `mg_jf_r20_tracked.tsv`（`45b99d228afe20bb6ac9a1cb9cf7b87e`）/ `board_jf_merge.log` / **`jf_score_r0_merge.json` / `jf_score_r20_merge.json`**。
- 复现（主机 x86_64，**无 GPU / 无训练 / 无重活**；单组 ≈ 6 min，IO 密集 ⇒ 跑主机别跑服务器）：
  ```bash
  cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
  PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python
  D=manu/memory/board_far_20261120/board_merge
  for R in r0 r20; do
    PYTHONPATH=. $PY manu/evaluation/eval_board_fused_tsv.py \
      --fused-tsv $D/mg_jf_${R}_fused.tsv \
      --labels-root ../../lqcache_rebuild/lookahead_residual/data/labels \
      --sequence VIDEO00007_19700101_001324 --total-frames 2500 \
      --delivered-ids manu/memory/board_far_20261120/board/delivered_run2.ids \
      --out /tmp/boardfar/jf_score_${R}_merge.json
  done
  ```

### 15.6 仍未闭环

- 🔴 **24 序全量（165,223 帧）板端 FAR 未做** —— 唯一可做「过线/破线」判定的口径；本轮 2500 帧窗口是**困难片段**，绝对 FAR 不可与红线 0.03 直接比。
- 🔴 三个交付包的 `out/*.so` 仍为 **pre-merge** ⇒ **本节收益在 as-shipped 包上尚不可复现**（需板上重编 + 刷新 MANIFEST/README md5）。
- 🔴 目标板 `.211`（OpenCV 4.6 / `.so.406`）仍离线 ⇒ 未验证。

1. 🔴 **24 序 / 165,223 帧全量板端 FAR**（红线的**唯一有效**判定口径）：板卡 tmpfs 只放得下 2500 帧
   ⇒ 走 §十一/§264 §5 的绕过路径（**服务器直接 scp 帧到板卡**，绕开 host 的 NFS 导出表为空）。
2. 🟡 **交付包 `out/*.so` 仍是 pre-merge 产物**（§十三 已记）：`rk3588_app_base/mosaic_sdk/out/libmosaic_pipeline.so`
   `nm` 无 `MergeCandidatesByRadius`、`out/mosaic_track_dump` 无 `--cand-dedup-radius`
   ⇒ **源码已对齐、产物未对齐**；本机无 aarch64 交叉编译器 ⇒ 只能在板上重编（配方见 §14.1）并刷新 `MANIFEST.md`/`README.md` md5。
3. 🟡 **目标板 `.211`（OpenCV 4.6 / `.so.406`）仍断电未验证**；本节新编 `.so` 是板卡 **OpenCV 4.8** 变体。
