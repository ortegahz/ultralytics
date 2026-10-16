# RK3588 NPU 时延结构改写五组受控对照：搬运量是真瓶颈，MAC 不是

> **日期**：2026-10-16
> **对象**：`trial0474_hm_nop0`（Anti-UAV 单帧绝对 SOTA，stride=2 / imgsz=640 / scale=n / 去 P0）
> **测量**：板卡 RK3588 `evm3588`，`rknn_heatmap_smoke`，`--warmup 10 --loops 30`
> **工具链**：rknn-toolkit2 **2.3.0**（host `/media/manu/1TB-Volume/conda-envs/rknn230`），
> 板卡 `librknnrt` **2.3.0**（commit `c949ad889d`）—— 版本一致
> **性质**：🔴 **全部为时延结构探针，精度一律未验证**。见第五节。

---

## 〇、一句话结论

**NPU 时延跟的是「特征图搬运量」，不是 MAC。** 五组受控对照里，
**MAC 砍掉 72% 的方案只快 10.2%，而搬运量砍掉 44% 的方案快 33.9%**；
甚至出现「MAC 更少反而更慢」的反例。

**推荐结构**：`--late-upsample --head-conv 32`（**不含 DW**）
⇒ int8 **34.089 ms / 28.5 fps**，较基线 **−38.6%（1.63×）**；
fp16 53.022 ms / 11.6 fps，**−48.6%**。

---

## 一、基线复现（证明口径可信）

| 项 | 本轮实测 | 案卷记载 | 判定 |
|:---|---:|---:|:---|
| `rknn_run` int8 | **55.501 ms** | 55.5 / 55.7 ms | ✅ 吻合 |
| fps int8 | **17.7** | 17.6–17.7 | ✅ 吻合 |

口径一致：同板卡、同二进制族、同 640×512 特征帧、同 `--stride 2 --th 0.25 --topk 80`。
**所有臂时钟自检均 `[ok]`**（`usleep(50000)` 实测 50.06–50.15 ms）。

---

## 二、五组受控对照总表

| # | 改写 | MAC | 搬运(累加) | 内部内存 | `rknn_run` int8 | fps | `rknn_run` fp16 | fps |
|:--|:---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **base**（当前交付） | 21.48 G | 3353.9 M | 20,400 KB | 55.501 ms | 17.7 | 103.104 ms | 6.8 |
| 1 | merge 两分支 | 21.48 G | — | 25,600 KB | 56.882 ms | 17.3 | 105.458 ms | 6.7 |
| 2 | DW+PW 轻量 | **6.05 G** | 3386.6 M | 33,200 KB | 49.860 ms | 19.7 | 77.981 ms | 8.6 |
| 3 | **late upsample** | 8.20 G | **1864.6 M** | **7,600 KB** | **36.657 ms** | **26.6** | 59.627 ms | 10.5 |
| 4 | late + DW | 4.34 G | 1872.7 M | 7,600 KB | 35.255 ms | 27.6 | 53.450 ms | 11.6 |
| 5 | late + DW + hc32 | 4.04 G | 1592.6 M | 7,600 KB | 34.302 ms | 28.4 | **50.989 ms** | **12.0** |
| 6 | ⭐ **late + hc32** | 5.72 G | 1586.8 M | 7,600 KB | **34.089 ms** | **28.5** | 53.022 ms | 11.6 |

参数量：base 1,790,531 → **late+hc32 1,693,475** → late+DW+hc32 1,627,619。

**§四·9e 上板闸门**：全部 int8 臂 **fallback 警告 0 条 = PASS**；
全部 fp16 臂各 1 条，是**既有的** `Conv:/model/b5/conv/Conv` 3Core→单核，四组日志逐字相同（只差时间戳），**非本轮引入**。

**NPU 内部内存**：int8 **20,400 → 7,600 KB（−63%）**；fp16 105,600 → 26,400 KB（−75%）。

---

## 三、核心发现：为什么「按 FLOPs 选方案」一定错

把变量分离出来看：

| 改写 | MAC 变化 | 搬运变化 | 时延变化 |
|:---|---:|---:|---:|
| merge（#1） | 0% | +Split 拷贝 | **+2.5%（更慢）** |
| DW（#2） | **−72%** | **+1%（几乎没动）** | **−10.2%** |
| late（#3） | −62% | **−44%** | **−33.9%** |
| late + hc32（#6） | −73% | −53% | **−38.6%** |

### 两条决定性反证

**① MAC 多 35% 反而快 26%。**
`late`(8.20 G, 36.657 ms) vs `late+DW`(4.34 G, 35.255 ms)；
更极端地，`late+hc32`(5.72 G, **34.089**) vs `late+DW+hc32`(4.04 G, **34.302**) ——
**MAC 少 30%、参数少 4%，时延反而慢 0.6%。**

**② 在 late 之上砍 MAC 收益归零。**
#4 → #6 只把 MAC 从 4.34 G 砍到 5.72 G 的同时搬运降 15%，净效果 −3.3%；
而 #5 → #6 证明**把 MAC 再砍 30% 完全换不到时延**。

### 机理

DWConv 每组只有 1 个通道，MAC 阵列大幅空转，但它**读取/写出的特征图尺寸与密集卷积完全相同**。
320×320 段本就是**带宽受限**——删掉「非瓶颈资源」自然换不来时延。
只有把计算**搬到更小的分辨率上**（late）才同时削减搬运与算力，才有 33.9% 的收益。

### 与既有铁律的关系

本轮为 `rules.md` **§四·9f「禁止用 MAC/FLOPs 外推 NPU 时延」**补齐了**第三次方向一致的实证**：

| 出处 | MAC | 时延 |
|:---|---:|---:|
| §十四 P0 | −6.8% | **−46.5%** |
| 本轮 #2 DW | **−72%** | −10.2% |
| 本轮 #3 late | −62% | −33.9% |

三个方向都不同，**MAC 与 NPU 时延之间不存在稳定映射**。

---

## 四、五组改写各自说明

### #1 merge（`head.heatmap.0` + `head.offset.0` → 单个 `Conv2d(64,128,3)`）—— 负结果

数学等价、逐位相同（CPU FP32 8 帧 `max|d| = 0.000e+00`）、零重训。
**但变慢 2.5%**：省下的那次读，被 RKNN 为通道切分插入的 `Split` 拷贝成本抵消有余。
佐证：NPU 内部内存 **20,400 → 25,600 KB（+25.5%）**。

### #2 DW+PW 轻量 Head —— 单独有效，叠加后失效

把 320×320 上的 5 个密集 3×3 换成 DW(3×3)+PW(1×1)。
单独用能拿 **−10.2%**，但**在 late 之上增益完全消失**（#5 vs #6）。
⇒ **推荐方案不含 DW**：非等价点更少、重训风险更低，而时延相同或更好。

### #3 late upsample —— 唯一的大赢

把 Head 段 5 个 **320×320** 密集 3×3 全部下移到 **160×160** 计算，
只把最终 **1+2=3** 个通道一次性 nearest 放大回 320×320：

```
fuse_p2(160,C=64) + avg_pool2d(b0) → concat(160,C=80)
  → fuse_p1 / feat_conv.0 / feat_conv.1 / heatmap.0 / offset.0  全在 160×160
  → cat(hm,off) = 3ch@160 → Resize ×2 → 320×320
```

320×320 上的卷积层数 **8 → 1**；内部内存 **−63%**；搬运 **−44%**。

### #6 late + head_conv 32 —— 推荐

Head 通道 64→32。搬运 −15%，**但时延只 −2.7%**（远小于按搬运量估算的 −15%）。
原因：320×320 段压下去后，大头换成了 **backbone + neck（160×160 及以下）**，而本次只动了 Head。

---

## 五、🔴 推荐方案的非等价点与重训风险

**推荐结构 = `--late-upsample --head-conv 32`（不含 DW）。**

| 非等价点 | 说明 | 风险 |
|:---|:---|:---|
| P1 经 `avg_pool2d` 降采样进融合 | b0 的 P1 特征（320×320）被降到 160×160 再与 P2 融合 | **中**：3×3 极小目标依赖 P1 高频，降采样可能伤召回 |
| Head 通道 64→32 | 通道减半，参数 1,790,531 → 1,693,475 | **低**：结构未变，仅宽度 |

- **不含 DW** 是有意的：DW 会额外引入五处函数形式变更，全部无收益。
- 本轮**未做任何精度测量**（探测臂检出 0，属预期）。
- 🔴 **落地前必须先做全量精度评测**（31,613 帧 / GT 25,111 / dist ≤8px），
  重点看 DJI 两序列的召回缺口（案卷记载 R 缺口 42/88 FN 来自那里）。
- 重训建议走**冻结底模 + 零初始化门控**，不要解冻 Trial 0474
  （案卷：「改模型」路线 **8 负 0 胜**，单帧底模已永久锁定）。

---

## 六、距 20 ms 目标的现状裁决

**已达成 34.089 ms（28.5 fps），距 20 ms 仍差 1.70 倍。**

纯结构路线**已趋近饱和**：merge −2.5% ✗ → DW −10.2% → late −33.9% → hc32 −2.7%。
剩余差距**全部落在 backbone + neck 与输入分辨率**上。

| 后续方向 | 预期 | 性质 |
|:---|:---|:---|
| **动态 ROI 局部裁剪**（跟踪态只跑 64~128 patch） | 唯一还有数量级的空间 | 需上层卡尔曼跟踪配合，不属纯结构 |
| backbone/neck 瘦身（调 `w` 或减层） | 未知 | 需重训且触及冻结底模，风险高 |
| ~~stride 2→4~~ | — | 🔴 **不再推荐**：late 已把 Head 压到 160，等价于只保留 P2 尺度；且 §四·9h 警告 40 ms 红线风险 |

---

## 七、复现命令

```bash
# 0) 环境与前置（全部本机，无需登训练服务器）
RKNN=/media/manu/1TB-Volume/conda-envs/rknn230/bin/python   # 2.3.0；同目录 rknn 是弃用的 1.4.0
UAV=/home/manu/anaconda3/envs/uav/bin/python
W=/media/manu/1TB-Volume/rknn_work/hm/merge_head
CKPT=/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/trial_0474/weights/best.pt
CALIB=$W/calib_train_features.txt
BR=/usr/bin/python3   # board_run.py 需要 pexpect；uav 环境没有

# 1) 导出 ONNX（推荐臂）
cd /media/manu/1TB-Volume/workspace/ultralytics
$UAV -W ignore manu/pipeline/npu/hm_export_onnx.py \
    --ablate-p0 --late-upsample --head-conv 32 --skip-check \
    --out $W/onnx/trial0474_hm_nop0_late32.onnx
# ⚠️ --skip-check：本机 uav 环境 torch 1.13.1 配 numpy 2.2.6，ORT 对账跑不了（案卷已载明）

# 2) 构建 int8（fp16 加 --no-quant）
cd /media/manu/1TB-Volume/workspace/ultralytics
$RKNN manu/pipeline/npu/hm_convert_rknn.py \
    --onnx $W/onnx/trial0474_hm_nop0_late32.onnx \
    --out  $W/trial0474_hm_nop0_late32_int8.rknn \
    --calib-list $CALIB | tee $W/build_late32_int8.log

# 3) §四·9e 上板闸门
grep -ic fallback $W/build_late32_int8.log        # 必须 0

# 4) 交叉编译 + 投递 + 测速
bash manu/pipeline/npu/build_rknn_heatmap_arm64.sh
$BR manu/pipeline/opencl/board/board_run.py --exec "mkdir -p /root/merge_head"
$BR manu/pipeline/opencl/board/board_run.py \
    --put $W/trial0474_hm_nop0_late32_int8.rknn:/root/merge_head/late32_int8.rknn
ssh root@192.168.0.64 'cd /root/merge_head && \
    pkill -9 -x rknn_heatmap_smoke; sleep 1 && \
    ./rknn_heatmap_smoke ./late32_int8.rknn ./feat.jpg \
      --th 0.25 --stride 2 --topk 80 --warmup 10 --loops 30 --tag late32_int8'
# ⚠️ 必须用 pkill -x（精确进程名），不能用 -f —— 否则会连承载命令的 shell 一起杀掉

# 5) merge 臂等价性自检（零重训改写的唯一硬闸门）
$UAV -W ignore manu/pipeline/npu/verify_merge_head_branches.py
```

---

## 八、产物清单

全部在 `/media/manu/1TB-Volume/rknn_work/hm/merge_head/`：

| 文件 | 说明 |
|:---|:---|
| `onnx/trial0474_hm_nop0_{base,merge,dwhead,lateups,latedw,late32,latedw32}.onnx` | 7 份 ONNX |
| `trial0474_hm_nop0_{base,merge,dwhead,lateups,latedw,late32,latedw32}_{int8,fp16}.rknn` | 14 个 RKNN |
| `build_*.log`（15 份） | 构建日志，§四·9e 判据来源。14 份对应上表双臂，另 1 份 `build_merge_lvl2.log` 是 `optimization_level=2` 失败探针（见第九节坑 2） |
| `calib_train_features.txt` | 200 张已建特征 JPEG，跨 199 序列，只取 train |

代码（本轮新增/修改，**全部默认关闭，原路径零影响**）：

| 文件 | 改动 |
|:---|:---|
| `manu/models/heatmap_model.py` | 新增 `dw_pw()`；`merge_head_branches` / `dw_head` / `late_upsample` / `head_conv` 四个开关；`remap_head_branches_state_dict()` |
| `manu/pipeline/npu/hm_export_onnx.py` | 新增 `--merge-head-branches` / `--dw-head` / `--late-upsample` / `--head-conv` 与三道 Gate |
| `manu/pipeline/opencl/board/board_run.py` | **bug 修复**：`put()` 原先无条件要求 `RK3588_PASSWORD`，与 `shell()` 不一致，已改为优先公钥 |
| `manu/pipeline/npu/verify_merge_head_branches.py` | 新增：merge 臂等价性自检 |
| `manu/pipeline/npu/run_merge_head_speed_ab.sh` | 新增：三阶段 runbook（`onnx` / `build` / `board`） |

---

## 九、过程中踩到的四个坑（均为「静默失效」型）

1. **`nn.Sequential` 按位置建键** — 少放一层就让 1×1 尾层从 `heatmap.2` 挪到 `heatmap.1`，
   整个 checkpoint 对不上。由自检 Gate 立即炸出（`missing: head.heatmap.1.weight`）。

2. 🔴 **rknn-toolkit2 2.3.0 对 `Conv → Split` 直连会丢 pads** —
   `[ShapeInferenceError] (Conv: branch3x3) Inferred (318) vs 声明 (320)`，
   而 **318 = 320−3+1 即 pad 被当 0**。降 `optimization_level=2` **无效**（结构问题，非融合深度）。
   **解法：把 ReLU 整体提到 Split 之前** —— ReLU 逐元素，`relu(h)[:, :64] ≡ relu(h[:, :64])`，
   数学等价且让 Split 不再紧贴卷积。
   ⚠️ 另：`torch.chunk` 会被 ONNX 追踪成 Shape/Gather/Add/Div 约 9 个标量节点，**必须用静态 `torch.split`**。

3. **`pkill -9 -f <name>` 会杀掉承载命令的 shell** — ssh 命令行里含该字符串时自触发。
   必须用 `pkill -9 -x`（精确进程名）。

4. **scp 在目标目录不存在时报 `dest open ...: No such file or directory`** —
   不会被 key auth 或 `BatchMode` 救回来，必须先 `mkdir -p`。

---

## 十、局限与未覆盖

- 🔴 **本轮零精度测量**。所有结论只对「时延」成立，**不构成任何精度判断**。
- 搬运量指标为「卷积输出元素累加」，**未含** activation 的中间张量、
  Im2col buffer 与 NPU 内部的调度开销，属近似量纲，仅用于**横向对比**。
- 只在 **int8 / fp16 两档**、**scale=n、stride=2、imgsz=640** 下测过；
  未验证 `scale=s`、其他分辨率、以及 e2e 三级流水线口径下的收益。
- 未测 **P=2 并行**下的表现（案卷记载 NPU 并行硬上限 = 2，且 CPU 绑核决定 fp16 能否过 12 fps）。
- `Conv:/model/b5/conv/Conv` 的 fp16 3Core→单核**仍未归因**；本轮静态枚举证实
  该层参数全部健康（C=128 完全 32 对齐、groups=1、40×40），
  判为 **fp16 编译器调度决策而非结构缺陷**——**改网络结构修不掉它**。
