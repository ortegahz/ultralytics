# RK3588 板端 f16 multi-scale 拼图（Canvas Mosaic）推理 —— 对照报告

> **实验目标**：在 RK3588 上用 C++ 实现「**一次前向喂满 640×640、把三个尺度塞进同一张画布**」的 f16 热图模型推理，
> 并与 PC 侧已冻结的 PT float32 multi-scale 基准做**逐口径**对照。
>
> **实验范围**：**龙泉山两个 val 轨**（no track / 单帧臂）
>
> | 轨 | 序列 | 帧数 | GT | 说明 |
>|:--|:--|--:|--:|:--|
> | `track2_golden` | 7 条黄金池序列 | **17,187** | 16,638 | T2，空域白热基准 |
> | `track3_blind` | `VIDEO00007_19700101_001324` | **11,810** | 10,535 | T3，盲测难例留出 |
>
> 合计 28,997 帧 —— 正好是 official val 全量。
>
> **结论一句话**：**拼图打包本身几乎无损（f32 下只损失 0.0027 F1，却保住了多尺度的 +0.016~0.023 增益）**；
> 板端 f16 的代价也可以干净地隔离出来，**拼图装配（RGA）只占 ~1 ms，单次 NPU 前向与单尺度完全一致**。

---

## 1. 实验设计

### 1.1 为什么需要「拼图」

PC 侧 multi-scale 的做法是**逐尺度各跑一次前向**：native(640×512)、thumb160(160×128)、thumb80(80×64)
各自 letterbox 到 640×640 再进模型。也就是 **3 次 NPU 前向**。

拼图方案把三张画布压进**同一张 640×640**：native 贴左上、thumb160 贴左下、thumb80 贴在 thumb160 右侧，
**只跑 1 次前向**。因为 RKNN 的输入张量静态是 `[1,3,640,640]`，卷积代价由形状决定、与像素无关，
所以 **NPU 时间与单尺度完全相同**（实测 `rknn_run` 100~102 ms，见表 C）。

### 1.2 画布布局（v1）

```
canvas 640 x 640 x 3 (RGB, 模型通道序 [residual, diff, I_t]), 背景 114 填满 3 通道
┌────────────────────────────────┬──────────────┐
│                                │              │
│   native 640x512  (1:1)        │  背景 114    │
│   = 画布 (0, 0, 640, 512)      │              │
│                                │              │
├──────────┬──────────┬─────────┤  ← 行 512
│ thumb160 │ thumb80  │ 背景114 │              │
│ 160x128  │  80x64   │         │              │
└──────────┴──────────┴─────────┴──────────────┘
   0      160        240
```

三个 tile 由 **RGA `improcess` ×3** 从同一张 640×512 源图一次装配完（缩放 + 落位一步到位），
逐帧**零 CPU 图像拷贝**：stb_image 解出来的缓冲直接 `importbuffer_virtualaddr` 交给 RGA。

### 1.3 七个 arm（全部用同一个打分器重算）

| arm | 计算图 | 用途 |
|:--|:--|:--|
| **A** | PC f32，逐尺度 3 次前向 | **冻结基准**（gate 产物，0.897694） |
| **A0** | PC f32，逐尺度，仅 native | 多尺度增益的逐尺度口径基线 |
| **B** | PC f32，拼图 1 次前向（缩略图 INTER_AREA） | **隔离「打包损失」** |
| **B1** | PC f32，拼图 1 次前向（缩略图 INTER_LINEAR ≈ 板端 RGA 语义） | 隔离「缩略图重采样语义」 |
| **B0** | PC f32，拼图 1 次前向，仅 native | 多尺度增益的拼图口径基线 |
| **C** | **板端 f16，拼图 1 次前向** | **本次交付的产物** |
| **C0** | 板端 f16，拼图 1 次前向，仅 native | 隔离「画布里塞了缩略图」对 native 检测的扰动 |

> **B 臂是本次实验的关键补件**：没有它，C 与 A 的差值无法归因（那是「打包 + 量化」的合计）。
> 加上 B 之后，A→B 是打包损失、B→C 才是 f16/量化损失。

### 1.4 两条轨的 PC 基准（冻结值，已逐位复现）

| 轨 | TP | FP | FN | GT | F1 | 来源 |
|:--|--:|--:|--:|--:|--:|:--|
| `track2_golden`（7 序列） | 16636 | 33 | 2 | 16638 | **0.9989491698441769** | `gate/epoch3/track2_golden.json` |
| `track3_blind`（1 序列） | 8972 | 482 | 1563 | 10535 | **0.8976937315523539** | `gate/epoch3/track3_blind.json` |

两条轨**各自**用自己 gate 产物里的 `config.params`（`th_base=0.3` / `dedup_radius=768`），
用脚本默认值（0.22 / 32）算出来的数**不可与上表并列**。

---

## 2. 全部对照表

### 2.1 轨道 `track3_blind` —— 1 sequences / 11810 frames

> 序列 1 条 / 11810 帧 / GT 10535 · 口径参数来自 `/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json`（th_base=0.3, dedup_radius=768.0）

表 A：精度对照

| arm | 计算图 | TP | FP | FN | GT | F1 | Recall | Precision | det | ΔF1 vs A |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **A** | PC f32, per-scale, 3 forwards | 8972 | 482 | 1563 | 10535 | 0.897694 | 0.851637 | 0.949016 | 9454 | +0.000000 |
| **A0** | PC f32, per-scale, native only | 8530 | 286 | 2005 | 10535 | 0.881608 | 0.809682 | 0.967559 | 8816 | -0.016086 |
| **B** | PC f32, mosaic, 1 forward | 8735 | 249 | 1800 | 10535 | 0.895025 | 0.829141 | 0.972284 | 8984 | -0.002668 |
| **B1** | PC f32, mosaic, INTER_LINEAR thumbs | 8733 | 262 | 1802 | 10535 | 0.894316 | 0.828951 | 0.970873 | 8995 | -0.003377 |
| **B0** | PC f32, mosaic, native only | 8334 | 239 | 2201 | 10535 | 0.872305 | 0.791077 | 0.972122 | 8573 | -0.025389 |
| **C** | board f16, mosaic, 1 forward | 8733 | 235 | 1802 | 10535 | 0.895555 | 0.828951 | 0.973796 | 8968 | -0.002139 |
| **C0** | board f16, mosaic, native only | 8309 | 230 | 2226 | 10535 | 0.871238 | 0.788704 | 0.973065 | 8539 | -0.026455 |

表 B：损失分解（每一段都有中间量支撑）

| 段 | 口径 | ΔF1 | ΔTP | ΔFP | 含义 |
|:--|:--|--:|--:|--:|:--|
| A → B | 逐尺度 → 拼图 | -0.002668 | -237 | -233 | **拼图打包损失**（f32，单次前向 vs 三次前向） |
| B → B1 | 拼图 → 拼图 | -0.000709 | -2 | +13 | 缩略图重采样语义影响（INTER_AREA → INTER_LINEAR≈RGA） |
| B → C | 拼图 → 拼图 | +0.000529 | -2 | -14 | **f16 + 板端量化损失**（拼图对拼图） |
| A0 → A | 逐尺度 → 逐尺度 | +0.016086 | +442 | +196 | 多尺度增益（逐尺度口径） |
| B0 → B | 拼图 → 拼图 | +0.022721 | +401 | +10 | 多尺度增益（拼图口径） |
| A0 → B0 | 逐尺度 → 拼图 | -0.009303 | -196 | -47 | 拼图打包损失（仅 native tile） |
| B0 → C0 | 拼图 → 拼图 | -0.001066 | -25 | -9 | f16 + 板端量化损失（仅 native tile） |

表 D：各 arm 的按 tag 检测数

| arm | tag0 native | tag1 thumb160 | tag2 thumb80 | 三尺度合计 | tag255 融合行（不参与评测） |
|:--|--:|--:|--:|--:|--:|
| A | 20109 | 11092 | 8140 | 39341 | 11017 |
| A0 | 20109 | 0 | 0 | 20109 | 0 |
| B | 19147 | 10587 | 3847 | 33581 | 0 |
| B1 | 19118 | 10779 | 4021 | 33918 | 0 |
| B0 | 19147 | 0 | 0 | 19147 | 0 |
| C | 19129 | 10636 | 3683 | 33448 | 0 |
| C0 | 19129 | 0 | 0 | 19129 | 0 |

表 E：逐序列（no track）

| 序列 | arm A (PC f32 逐尺度) | arm B (PC f32 拼图) | arm C (板端 f16 拼图) |
|:--|--:|--:|--:|
| VIDEO00007_19700101_001324 | 8972/482/1563 F1=0.897694 | 8735/249/1800 F1=0.895025 | 8733/235/1802 F1=0.895555 |

### 2.2 轨道 `track2_golden` —— 7 sequences / 17187 frames

> 序列 7 条 / 17187 帧 / GT 16638 · 口径参数来自 `/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track2_golden.json`（th_base=0.3, dedup_radius=768.0）

表 A：精度对照

| arm | 计算图 | TP | FP | FN | GT | F1 | Recall | Precision | det | ΔF1 vs A |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **A** | PC f32, per-scale, 3 forwards | 16636 | 33 | 2 | 16638 | 0.998949 | 0.999880 | 0.998020 | 16669 | +0.000000 |
| **A0** | PC f32, per-scale, native only | 16636 | 5 | 2 | 16638 | 0.999790 | 0.999880 | 0.999700 | 16641 | +0.000840 |
| **B** | PC f32, mosaic, 1 forward | 16636 | 5 | 2 | 16638 | 0.999790 | 0.999880 | 0.999700 | 16641 | +0.000840 |
| **B1** | PC f32, mosaic, INTER_LINEAR thumbs | 16636 | 5 | 2 | 16638 | 0.999790 | 0.999880 | 0.999700 | 16641 | +0.000840 |
| **B0** | PC f32, mosaic, native only | 16636 | 5 | 2 | 16638 | 0.999790 | 0.999880 | 0.999700 | 16641 | +0.000840 |
| **C** | board f16, mosaic, 1 forward | 16636 | 5 | 2 | 16638 | 0.999790 | 0.999880 | 0.999700 | 16641 | +0.000840 |
| **C0** | board f16, mosaic, native only | 16636 | 5 | 2 | 16638 | 0.999790 | 0.999880 | 0.999700 | 16641 | +0.000840 |

表 B：损失分解（每一段都有中间量支撑）

| 段 | 口径 | ΔF1 | ΔTP | ΔFP | 含义 |
|:--|:--|--:|--:|--:|:--|
| A → B | 逐尺度 → 拼图 | +0.000840 | +0 | -28 | **拼图打包损失**（f32，单次前向 vs 三次前向） |
| B → B1 | 拼图 → 拼图 | +0.000000 | +0 | +0 | 缩略图重采样语义影响（INTER_AREA → INTER_LINEAR≈RGA） |
| B → C | 拼图 → 拼图 | +0.000000 | +0 | +0 | **f16 + 板端量化损失**（拼图对拼图） |
| A0 → A | 逐尺度 → 逐尺度 | -0.000840 | +0 | +28 | 多尺度增益（逐尺度口径） |
| B0 → B | 拼图 → 拼图 | +0.000000 | +0 | +0 | 多尺度增益（拼图口径） |
| A0 → B0 | 逐尺度 → 拼图 | +0.000000 | +0 | +0 | 拼图打包损失（仅 native tile） |
| B0 → C0 | 拼图 → 拼图 | +0.000000 | +0 | +0 | f16 + 板端量化损失（仅 native tile） |

表 D：各 arm 的按 tag 检测数

| arm | tag0 native | tag1 thumb160 | tag2 thumb80 | 三尺度合计 | tag255 融合行（不参与评测） |
|:--|--:|--:|--:|--:|--:|
| A | 1254 | 1132 | 1480 | 3866 | 1286 |
| A0 | 1254 | 0 | 0 | 1254 | 0 |
| B | 1258 | 773 | 1 | 2032 | 0 |
| B1 | 1257 | 793 | 5 | 2055 | 0 |
| B0 | 1258 | 0 | 0 | 1258 | 0 |
| C | 1256 | 741 | 0 | 1997 | 0 |
| C0 | 1256 | 0 | 0 | 1256 | 0 |

表 E：逐序列（no track）

| 序列 | arm A (PC f32 逐尺度) | arm B (PC f32 拼图) | arm C (板端 f16 拼图) |
|:--|--:|--:|--:|
| VIDEO00006_19700101_000703 | 1742/27/0 F1=0.992310 | 1742/1/0 F1=0.999713 | 1742/1/0 F1=0.999713 |
| VIDEO00007_19700101_000818 | 8263/0/0 F1=1.000000 | 8263/0/0 F1=1.000000 | 8263/0/0 F1=1.000000 |
| VIDEO00008_19700101_001307 | 3660/2/1 F1=0.999590 | 3660/2/1 F1=0.999590 | 3660/2/1 F1=0.999590 |
| VIDEO00009_19700101_001531 | 115/2/0 F1=0.991379 | 115/1/0 F1=0.995671 | 115/1/0 F1=0.995671 |
| VIDEO00029_19700101_014053 | 592/1/0 F1=0.999156 | 592/0/0 F1=1.000000 | 592/0/0 F1=1.000000 |
| VIDEO00033_19700101_015000 | 1025/1/1 F1=0.999025 | 1025/1/1 F1=0.999025 | 1025/1/1 F1=0.999025 |
| VIDEO00034_19700101_015052 | 1239/0/0 F1=1.000000 | 1239/0/0 F1=1.000000 | 1239/0/0 F1=1.000000 |

### 2.3 表 C：板端每帧耗时分解（单次拼图前向，28,997 帧合并）

| 段 | ms/帧 | 说明 |
|:--|--:|:--|
| decode | 11.932 | BMP 解码（含 NFS 读 983 KB/帧） |
| rga_import | 1.041 | RGA importbuffer_virtualaddr |
| rga_improcess | 1.095 | RGA improcess ×3（拼图装配） |
| fp16_or_upload | 12.108 | uint8→fp16 + rknn_inputs_set |
| rknn_run | 100.259 | **rknn_run（单次拼图前向）** |
| outputs_get | 0.617 | rknn_outputs_get + dequant |
| backproject_cc | 11.699 | CPU float32 反投影 + 8 连通域 |
| write | 0.022 | TSV 写出 |
| **total_wall** | **141.413** | 墙钟合计 |
| **fps** | **7.071** | |

> 拼图装配（RGA）只占 ~1 ms，相对一次 NPU 前向可忽略 ⇒ **「拼图一次前向」与「单尺度一次前向」的 NPU 代价相同**，差异全部来自输出侧的反投影与连通域。

### 2.4 口径声明（必读）

1. **打分器唯一**：`manu/evaluation/system_sota_longquan.py --no-pipeline`（**no track / 单帧臂**），`match_mode=inbox`、`tol=8.0`、`pipeline=false`，只用 tag 0/1/2。
2. **每条轨道用自己 gate 产物里的 `config.params`**（`track2_golden.json` / `track3_blind.json`），**不能手抄、不能用脚本默认值**（默认 `th_base=0.22` / `dedup_radius=32` 会给出完全不同的数）。
3. **A 与 B/C 不是同一个计算图**：A 是逐尺度 3 次前向，B/C 是单次拼图前向。⇒ ΔF1(A→B) 是打包损失，ΔF1(B→C) 才是 f16/量化损失，**不能把 A→C 整段记在量化账上**。
4. **画布几何差异**：PC native 用 letterbox（上下各 64 行 pad=114）；拼图里 native 直接贴 (0,0) 无 pad；PC 缩略图画布背景是模型序 (residual=0, diff=0, I_t=mean)，拼图统一背景 114。⇒ **已知口径断裂**，B1 臂量化了它的上限。
5. **反投影方向不同**：PC 是「320→640 先放大再裁」，板端是「先裁 320 格再放大」，亚像素相位差 ≈0.25 输入像素。**非逐比特等价**。
6. **评测本身接近「每帧一个候选」**：`merge_scales(dedup_radius=768)` 的半径大于帧内任意两点距离，所以每帧基本只保留最高分的那一个候选 ⇒ 指标主要由「最高分候选是否落进 GT 框」决定，**不是常规的多框平均精度**。
7. **特征溯源闭环**：板端 BMP = `cv2.imwrite(cv2.imread(official JPG))` 的无损重编码，与 gate 缓存所用的 official 特征逐像素一致（`manu/pipeline/npu/ptq/06_verify_reference_anchor.py` 的 pixel_proof：mismatches=0 / max_abs_diff=0）。⇒ 板端臂与 PC 臂吃的是同一批像素，**不是两个特征源**。
8. **gate/epoch3 与 lq_eval 不可混用**：gate 是 official 逐 lag 拟合特征、8 序列；lq_eval 是 FPV 仿射连乘特征、24 序列，逐像素差 26~70/255。本表只用 gate 口径。

---

## 3. 实现要点

### 3.1 板端 C++（`manu/pipeline/npu/rknn_mosaic_heatmap.cpp`）

| 段 | 说明 |
|:--|:--|
| 解码 | stb_image 直读 BMP（板卡无 OpenCV）。**stb 返回的 RGB 字节序 == 模型通道序**，无需翻转 |
| 拼图 | `importbuffer_virtualaddr` + `wrapbuffer_handle` + `improcess` ×3 |
| 输入 | `rknn_inputs_set` uint8 NHWC（`--input-mode set`） |
| 输出 | `rknn_outputs_get` + dequant → float32 320×320 热图 |
| 反投影 | CPU float32：裁出该 tile 在 320 格上的窗口 → INTER_LINEAR 放大到 640×512 |
| 连通域 | CPU float32：阈值 `>= 0.06`、8 连接、area ≥ 4、按 **sum 降序** 取 top 100 |

**两个消融模式**：`--canvas-mode {mosaic,single}`（single = PC 同构的 native letterbox 画布）、
`--tiles 0,1,2`（只输出部分 tile 的检测）。两者组合可用于把「画布扰动」和「多尺度贡献」分开。

**自带保护**：
- RGA 自检：native tile 是 1:1 拷贝，**必须逐字节等于源图**，否则判定 RGA「返回 SUCCESS 但没写数据」并切到对齐 staging 缓冲。
- 阈值比较在 **double** 里做（对齐 numpy 的 float32-vs-float64 比较语义）。
- 8 段计时（decode / rga_import / rga_improcess / fp16_or_upload / rknn_run / outputs_get / backproject_cc / write）。

### 3.2 PC 侧对照臂（`manu/pipeline/npu/ptq/37_mosaic_pc_control.py`）

复用**同一批 BMP**（不是另读 JPG），保证像素逐字节相同；同一张画布布局、同一次前向、
同一套反投影 + **直接调用 PC 侧的 `detect_components()`**（不是重写），输出**与板端相同格式的 TSV**，
因此两侧可以走**同一个打分器**。

---

## 4. 踩坑与证伪（详见 `manu/memory/MOSAIC_RK3588_20261026.md` §5）

| # | 现象 | 根因 | 结论 |
|:--|:--|:--|:--|
| 1 | 板端 tag1 弱、tag2 归零 | 反投影把「320 宽热图里的窗口」当成紧凑小图，按 packed stride 读 ⇒ 串到邻居 tile | **不是 f16 损失，是 stride bug**。tag0 跨满全宽不受影响 ⇒ 症状只在小尺度出现，极易误判 |
| 2 | `io-mem-nchw` 检测数 19.02/帧（`set` 是 2.35） | 该模型的 NCHW 填充路径是错的 | **只用 `--input-mode set`** |
| 3 | `sum_of_segments > total_wall` | `t_outputs` 从 `rknn_run` 之前开始计时 | 改为从 `rknn_run` 之后计 |
| 4 | 「缩略图重采样语义（INTER_AREA vs RGA 双线性）导致小尺度崩掉」 | — | **被证伪**：B1 臂只差 0.0007 F1 |
| 5 | PC 基准复现不出 0.897694 | 用了脚本默认 `th_base=0.22` / `dedup_radius=32` | 必须用 gate 的 `config.params` |

---

## 5. 复现

```bash
bash manu/pipeline/npu/run_mosaic_lq_val.sh board     # 板端全量 11,810 帧（~28 min）
bash manu/pipeline/npu/run_mosaic_lq_val.sh pc        # PC 侧拼图对照臂（~3 min）
bash manu/pipeline/npu/run_mosaic_lq_val.sh report    # 生成对照表
```

口径守卫（必须逐位 PASS）：

```bash
ssh -p 32222 huangzhe@192.168.99.40 \
  'source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav && cd /tmp/pycharm_project_10ae9e2e && \
   python manu/pipeline/npu/ptq/35_mosaic_board_eval.py --self-check'
```
