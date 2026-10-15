# ARM GMC 精度损失评估 · 第二轮（anchor_step=10，e2e demo 实际参数）

- 日期：2026-10-15
- 承接：第一轮见 [`2026-10-15_arm_gmc_precision_eval.md`](2026-10-15_arm_gmc_precision_eval.md)（`anchor_step=2` 精确等价档）
- memory 对应：`memory.md` §13 / `memory_compact.md` §15

---

## 一句话结论

**e2e 实际配置（`anchor_step=10`）没有精度损失。**
三臂峰值 F1 落在 **0.9021 ~ 0.9025** 的 0.0004 窄带内，两两差异都在噪声下限（±3 TP ≈ ±0.0002）以内。
**anchor_step 从 2 换到 10，在这份 val 上分不出差异。**

---

## 1. 实验设计

| 维度 | PC-BMP 臂 | ARM-S10 臂 | 对齐 |
|:---|:---|:---|:---:|
| 特征来源 | `build_full_median_dataset.py`（PC） | `gmc_stream_ocl`（板卡 GPU，`anchor_step=10`） | **唯一变量** |
| PC 臂数据 | **完全复用第一轮，未重跑** | — | — |
| 原始输入 | `anti_uav_bmp_all/val`（BMP） | 板卡 raw（同批） | 是 |
| 容器 | BMP | BMP | 是 |
| 帧集合 / 标签 | 31,613 帧 | 31,613 帧 | 逐项一致 |
| 权重 / 抽峰 / 推理 / 匹配 | 同第一轮 | 同第一轮 | 是 |
| 评估脚本 | `eval_arm_pc_gmc_ab.py` | 同 | 是 |

用户要求「只替换特征，其他计算保持 sota model 的 pc 版本的脚本」——PC 臂 cache
`pc_bmp_full_cache.pkl`（md5 `59c0a19bc8fa4705d37e51fdd8f4a0c5`）本轮**未动**。

### 1.1 先核实 e2e demo 到底用不用 step=10（三处独立证据，非引 memory）

| # | 证据 | 内容 |
|--:|:--|:--|
| 1 | 编排脚本 | `manu/pipeline/npu/run_hm_demo.sh:170-172` 调 `gmc_stream_ocl` 时**根本没传 `--anchor-step`** ⇒ 走二进制默认 |
| 2 | 源码默认 | `manu/pipeline/cpp/gmc_stream_ocl.cpp:2217`：`int limit = 0, anchor_step = 10, window = 21, stride_step = 2, downscale = 2;` |
| 3 | 板卡运行时 | 按 demo 写法实跑 30 帧探针，二进制自报 `[BUILD] ... anchor_step=10 ... mode=tree` + `[NOTE] mode=tree is the shipping arm; it is NOT equivalent to the frozen native set` |

源码文件头 `:74-77` 亦注明 `--anchor-step 10 (default) ... The shipping config`。

### 1.2 为什么必须重算特征

第一轮跑的是 `anchor_step=2`（零连乘、逐 lag 直接拟合），该档与 PC/SOTA **逐字节相同**；
`anchor_step=10` 走 `{2,12,22,32,42}` 五锚点 + 最大深度 4 的连乘近似，**特征本身是另一份**，
板卡上原本没有。因此特征**必须重算**——本轮全量重跑了。

---

## 2. 结果

### 2.1 三臂峰值（`th = 0.25`，Distance ≤ 8.0px，GT 25,111）

| 臂 | anchor_step | F1 | P | R | TP | FP | FN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| PC-BMP（SOTA 参照） | — | 0.902316 | 94.87% | 86.02% | 21,601 | 1,167 | 3,510 |
| ARM-BMP（第一轮） | 2 | 0.902116 | 94.84% | 86.01% | 21,598 | 1,174 | 3,513 |
| **ARM-BMP（本轮）** | **10** | **0.902513** | **94.91%** | **86.03%** | **21,603** | **1,159** | **3,508** |

### 2.2 差值汇总

| 对比 | ΔF1 | 折合 TP | 判定 |
|:--|--:|--:|:--|
| PC vs ARM-step2 | −0.000201 | −3 | 噪声内 |
| **PC vs ARM-step10** | **+0.000197** | **+2** | **噪声内** |
| ARM-step2 vs ARM-step10 | +0.000397 | +5 | 噪声内 |

### 2.3 跨 15 档门限

| th | PC-BMP F1 | ARM-S10 F1 | ΔF1 |
|--:|--:|--:|--:|
| 0.05 | 0.845796 | 0.845801 | +0.000005 |
| 0.10 | 0.870055 | 0.869402 | −0.000653 |
| 0.15 | 0.894054 | 0.893840 | −0.000214 |
| 0.20 | 0.900752 | 0.900824 | +0.000072 |
| **0.25** | **0.902316** | **0.902513** | **+0.000197** |
| 0.30 | 0.901631 | 0.901722 | +0.000091 |
| 0.35 | 0.897835 | 0.898107 | +0.000272 |
| 0.40 | 0.892610 | 0.892568 | −0.000042 |
| 0.45 | 0.885318 | 0.885395 | +0.000077 |
| 0.50 | 0.874095 | 0.874159 | +0.000064 |
| 0.55 | 0.854934 | 0.854746 | −0.000188 |
| 0.60 | 0.822496 | 0.822378 | −0.000118 |
| 0.65 | 0.772025 | 0.771722 | −0.000303 |
| 0.70 | 0.695142 | 0.695230 | +0.000088 |
| 0.75 | 0.575898 | 0.575390 | −0.000509 |

均值 **−0.000078**，区间 [−0.000653, +0.000272]，**负档 7/15**，正负各半
⇒ 无系统性差异。峰值门限两臂同为 `th=0.25`，无工作点漂移。

---

## 3. bad case handle（默认开启，本轮即含真实回退行为）

`run_step10_board.sh` **未传任何闸门参数**，三层 handle 全部默认开启。

| 层 | 源码位置 | 默认 | 触发后行为 |
|:--|:--|:--:|:--|
| 合理性闸门 `fitgate` | `check_fit()` `:670-716` | **开** | `W(.) = IDENTITY`，分 `shift`/`other` 计数 |
| 连乘链二次闸门 `chain` | `:1651-1656` | **开** | `W(.) = IDENTITY`；与「坏锚点」分开计数 |
| 超时预算 `fitdeadline` | `--fit-deadline-ms` `:2046-2052` | **关** | `identity` 或 `hold`（复用上一次**成功**拟合，不外推） |

### 3.1 全量 24 序列触发统计

| 序列 | fitgate | chain | | 序列 | fitgate | chain |
|:--|--:|--:|:--|:--|--:|--:|
| wg2022_ir_012_split_08 | **116** | **74** | | wg2022_ir_020_split_07 | 57 | 24 |
| wg2022_ir_052_split_08 | 101 | 64 | | 01_4485_1167-2666 | 57 | 8 |
| wg2022_ir_041_split_06 | 79 | 65 | | wg2022_ir_040_split_04 | 55 | 29 |
| wg2022_ir_015_split_01 | 71 | 29 | | wg2022_ir_020_split_05 | 47 | 31 |
| 02_6321_0274-2773 | 30 | 10 | | wg2022_ir_010_split_01 | 18 | 0 |
| wg2022_ir_020_split_01 | 11 | 7 | | 其余 13 条 | 0 | 0 |

`fitdeadline` 全程 **0**。坏帧集中在大晃动序列，正是闸门该起作用的地方。

01_4485 细节：`rejected=57/8838 (0.645%) [shift=13 other=44]` —— 多数被拒不是「平移超标」
而是 scale/extent 类，印证 `scale×extent+translation` 不独立才是真凶。

### 3.2 为什么回退 IDENTITY 而不是 clamp

源码 `:641-668`：龙泉山业务域实测出现过**平移 4863 px 的有限但物理荒谬解**，`all_finite()` 放行 →
`required_pad()` 忠实报告 → 设备图 10366×10238 = 106 Mpixel/槽 × 45 槽 ≈ **4.8 GB** →
板卡 OOM killer 以 `file-rss:7623308kB` 杀进程，**无断言、无报错、无非零 rc**。
实测真实运动峰值仅 ~76 px（640 px 帧宽），`kDefaultMaxShiftFrac = 1.0` 留约 8× 余量。
注释原话：`Clamping the translation to the cap still yields a wildly wrong warp`。

### 3.3 为什么默认关闭超时臂

源码 `:2047-2049`：`OFF is deliberate: a wall-clock trigger makes the output depend on machine load.`
⇒ 开了它，同一份 raw 在不同负载下出不同特征，**精度对账失效**。
部署需保 40 ms 硬上限时再开，降级选 `identity`（注释：`never propagate a degraded warp forward`）。

---

## 4. 附带结论：step=10 比 step=2 快约 3×

| 配置 | min | p50 | max | fits/frame | compose/frame |
|:--|--:|--:|--:|--:|--:|
| `anchor_step=2` | 37.385 | **111.109** | 216.527 | 21.00 | 0.00 |
| `anchor_step=10` | 13.519 | **35.127** | 63.870 | 6.02 | 40.00 |

单位 ms/frame，板卡全量 24 序列。`fits/frm=6.02 / compose/frm=40.00` 结构上印证
5 锚点 + 深度 4 连乘（step=2 为 21.00 / 0.00 零连乘）。

> ⚠️ 该数字受 memory 既有「时延 5 倍不自洽」问题约束，**不得直接进嵌入式预算**。

---

## 5. 复现命令清单（`rules.md` 铁律九登记）

### 5.1 执行环境

| 项 | 值 |
|:--|:--|
| 板卡 | `ssh -p 22 root@192.168.0.64`（evm3588，Mali-G610） |
| 板卡二进制 | `/root/gmc_stream_ocl` md5 `aa99da811356694d66a0bce6de108b92`（ELF machine `0xb7` = AArch64，**两轮共用同一份**） |
| 训练服务器 | `ssh -p 32222 huangzhe@192.168.99.40`，项目 `/tmp/pycharm_project_10ae9e2e` |
| 解释器 | `/home/huangzhe/anaconda3/envs/uav/bin/python`（py3.10.20 / cv2 4.10.0 / numpy 1.26.4） |
| GPU | RTX 4090 D |
| 冻结构建器 | `manu/data/build_full_median_dataset.py` md5 `7729e77299f8933faffa73577e8e9400`（**全程未改**） |

### 5.2 板卡特征生成

脚本已固化在仓库 `manu/pipeline/npu/run_step10_board.sh`
（与 `run_step2_board.sh` 仅 `--anchor-step` 与输出目录不同）

```sh
scp -P 22 manu/pipeline/npu/run_step10_board.sh root@192.168.0.64:/root/run_step10.sh
ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step10.sh && \
    nohup /root/run_step10.sh > /mnt/manu/armfeat_s10.log 2>&1 &'
```

内部逐序列调用：

```sh
cd /mnt/manu/npu/out/demo/$seq && /root/gmc_stream_ocl \
  --raw-root raw --sequence "$seq" \
  --anchor-step 10 \
  --out-dir /mnt/manu/armfeat_s10/$seq/feat --out-ext bmp \
  --fused gpu --kernel-dir /root/kernels --timing
```

产出 31,661 帧（= raw 总帧数，每序列含前 2 帧），日志尾输出 `STEP10_ALL_DONE`。

### 5.3 传输（host → 训练服务器，6 流并行）

```sh
split -n l/6 -d /tmp/s10list.txt /tmp/s10part_
for p in /tmp/s10part_*; do
  rsync -az -e "ssh -p 32222" --files-from="$p" --relative \
    /run/media/manu/SSD960-LOCAL/nfs/armfeat_s10/ \
    huangzhe@192.168.99.40:/mnt/data/siping/datasets/manu/armfeat_stage_s10/ &
done; wait
```

实测 31,661 帧 / **325 s**（≈81 MB/s）。单流仅 13.8 MB/s，6 流提升 5.7×。

### 5.4 组装（`<seq>_<k>` → `<seq>__<k+1>`，丢弃 k<2 的 48 帧）

```sh
$PY manu/data/assemble_arm_gmc_dataset.py \
  --stage       /mnt/data/siping/datasets/manu/armfeat_stage_s10 \
  --out         /mnt/data/siping/datasets/manu/arm_gmc_full_s10 \
  --labels-from /mnt/data/siping/datasets/manu/ref_bmpval \
  --expect-total 31613 --verify-md5 60
```

产出 31,613 帧，md5 抽样 **60/60 一致**。

### 5.5 ARM-step10 臂推理（PC 臂不重跑）

```sh
$PY manu/inference/cache_trial0474_inferences.py \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --data /mnt/data/siping/datasets/manu/arm_gmc_full_s10/data.yaml \
  --output runs/gmc_eval/arm_gmc_s10_cache.pkl \
  --imgsz 640 --stride 2 --batch 32 --device 0 --conf-thresh 0.08 --top-k 80
```

### 5.6 A/B 评估

```sh
$PY manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/gmc_eval/pc_bmp_full_cache.pkl  --label-a PC-BMP \
  --arm-b runs/gmc_eval/arm_gmc_s10_cache.pkl  --label-b ARM-S10 \
  --tol 8.0 \
  --out-json /mnt/data/siping/datasets/manu/gmc_arm_s10_vs_pc_ab.json
```

评估脚本带自校准闸门：必须先复现冻结 headline（`F1 0.906361 / TP 21,643 / FP 1,004`）才允许继续，
复现不了以退出码 2 中止。

### 5.7 产物登记

| 产物 | 路径 | md5 / 规模 |
|:--|:--|:--|
| ARM-S10 cache | `runs/gmc_eval/arm_gmc_s10_cache.pkl` | `721f53fd39bf944587d1adf5d6bf33de` |
| A/B 报告 | `/mnt/data/.../manu/gmc_arm_s10_vs_pc_ab.json` | `02d0a66dad12e89a2d1db45a3e8a8c8b` |
| 板卡脚本 | `manu/pipeline/npu/run_step10_board.sh` | `a03f6460c2f952b37225d199b4258bb8` |
| PC 臂 cache（本轮未动） | `runs/gmc_eval/pc_bmp_full_cache.pkl` | `59c0a19bc8fa4705d37e51fdd8f4a0c5` |
| ARM-step2 cache（保留未动） | `runs/gmc_eval/arm_gmc_full_cache.pkl` | `7b02d122a2c569e615d88dffb36c4bbc` |

### 5.8 路径隔离

step=10 产物**全部带 `_s10` 后缀**：`armfeat_s10` / `armfeat_stage_s10` / `arm_gmc_full_s10` /
`arm_gmc_s10_cache.pkl`。

**step=2 产物一条未删**（用户明确要求保留，后续可能复用）：
板卡 `armfeat_s2` 31,661 帧、服务器 `arm_gmc_full` 31,613 帧、`armcmp_pcbmp` 31,613 帧。

---

## 6. 限定（引用前必读）

1. **只覆盖 `anchor_step=10` 与 `2` 两档**。`{4,6,8}` 未测，**不得声称 10 最优**。
2. **未做 ARM(step10) vs PC-side tree 交叉校验**。若要断言「移植在 step=10 下仍保真」，
   需另建 PC-side tree 参照臂；本轮 `0.902513` 与旧记 `0.906050` **不可直接相减**。
3. **§4 时延不得直接进嵌入式预算**（memory 旧记录已两度指出该线 5 倍不自洽，
   且 `-O0` Debug 与优化构建不可比）。
4. **三臂共同少 ~0.004 F1**（相对冻结 headline 0.906361），源于输入为
   `anti_uav_bmp_all` 的 BMP 原帧，属共有因素、在 A/B 中抵消，
   但**绝对 F1 不得与 headline 直接比大小**。

---

## 7. 已取消项

- **step10-vs-step2 特征逐像素比对**：用户明确要求不做，已终止。
  本轮结论仅基于 F1 口径，不含特征级差分量化。
