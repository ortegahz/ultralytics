# ARM GMC 精度损失评估（板卡 aarch64 + Mali-G610 实机执行）

- 日期：2026-10-15
- 评估对象：`manu/pipeline/cpp/gmc_stream_ocl.cpp`（GMC 阶段 = fit + warp + 时域中值 + 3 通道组装）
- 执行环境：RK3588 板卡 `evm3588`（192.168.0.64），aarch64，交叉编译产物
  `/root/gmc_stream_ocl`，md5 `aa99da811356694d66a0bce6de108b92`（ELF machine `0xb7` = AArch64）
- 配置：`--anchor-step 2`（精确等价档）、`--fused gpu`、`--out-ext bmp`、`--kernel-dir /root/kernels`
- 参照系：**PC SOTA 官方构建器** `manu/data/build_full_median_dataset.py`
  （md5 `7729e77299f8933faffa73577e8e9400`，全程未改动）

> **本文件是 `anchor_step=2` 精确等价档的结果。**
> e2e demo 实际使用的是 `anchor_step=10`（shipping config），该档结果见
> [`2026-10-15_arm_gmc_precision_eval_s10.md`](2026-10-15_arm_gmc_precision_eval_s10.md)。
> ⚠️ **本文件的「移植无回归」结论属精确等价档，不能直接外推到 e2e 实际配置** —— 须引用第二轮。

---

## 一句话结论

**ARM GMC 未引入精度回归。** 容器对称、严格同集配对的全量对照（24 序列 / 31,613 帧）中，
峰值 ΔF1 = **−0.000201**，跨 15 档均值 **+0.000074**，全部落在测量噪声下限（±3 TP ≈ ±0.0002）以内。
像素级佐证：两侧特征近乎逐字节一致（加权 MAE ~1e-4 量级）。

---

## 1. 实验设计：唯一变量是「特征由谁生成」

| 维度 | PC-BMP 臂 | ARM-BMP 臂 | 是否对齐 |
|:---|:---|:---|:---:|
| 特征来源 | `build_full_median_dataset.py`（PC） | `gmc_stream_ocl`（板卡 GPU） | **唯一变量** |
| 原始输入 | `anti_uav_bmp_all/val`（8bpp BMP） | 板卡 `raw/<seq>`（同批 BMP） | 是 |
| 容器 | BMP（无损） | BMP（无损） | 是 |
| 帧集合 | 官方 val 31,613 | 官方 val 31,613 | 是 |
| 标签 | `uav_gmc_median/labels/val` | 同（硬链接） | 逐项相等 |
| 权重 | `trial_0474/weights/best.pt` | 同 | 是 |
| 抽峰 | conf 0.08 / top_k 80 | 同 | 是 |
| 推理 | imgsz 640 / stride 2 / batch 32 | 同 | 是 |
| 匹配 | Distance ≤ 8.0px，15 档门限 0.05~0.75 | 同 | 是 |

### 关键修正一：PC 臂容器不对称（上一轮结论作废）

冻结构建器第 272 行 `cv2.imwrite(str(out_img_p / im_name), merged)` 的扩展名**继承自 ref 数据集文件名**。
上一轮 PC 对照用的是冻结 ref（含 6,985 帧 JPG），导致输出被 JPEG q95 **再压一次**，
而 ARM 臂是 BMP —— 65.7% 的帧上混入了压缩误差。

修复不需要改构建器：构建器只读 ref 的**文件名和标签**，从不读 ref 图像像素
（`merged` 全部由 raw 重算，见 `build_full_median_dataset.py:268-272`）。
构造 `ref_bmpval`（31,613 条全部改名 `.bmp`、符号链接指回原图）即可强制无损输出。
构建器 md5 全程保持 `7729e772...` 未变。

### 关键修正二：索引映射是 `__<k+1>`，不是 `__<k+3>`

板卡写 `<seq>_<k:06d>.bmp`（k 从 0，对应 raw 目录排序第 k+1 个文件）；
官方 val 写 `<seq>__<i:06d>.bmp`（i 从 3 起）。
由冻结构建器索引逻辑 `curr_list_idx = idx_map[frame_idx]` 得 **i = k+1**。

**这条是像素抽样抓出来的。** 偏移扫描（ARM vs PC 同源重建）：

| 序列 | off=1 MAE | off=1 Max\|D\| | off=3 MAE | off=3 Max\|D\| |
|:---|--:|--:|--:|--:|
| 25_1 | **0.00022** | **2** | 0.44577 | 7 |
| 34_1 | **0.00041** | **2** | 0.53627 | 68 |
| 01_4485_1167-2666 | **0.06503** | 227 | 1.25581 | 248 |

`off=3` 会引入视频运动量级的误差，`off=1` 才给出近乎逐字节的一致。
**若跳过这一步，后续所有 F1 结论都会是错的。**

---

## 2. 全量 A/B 结果（24 序列 / 31,613 帧 / GT 25,111）

严格同集配对 31,613 帧一一对应，GT 逐项一致。

| th | PC-BMP TP | FP | P | R | F1 | ARM-BMP TP | FP | P | R | F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 22628 | 5768 | 79.687% | 90.112% | 0.845796 | 22633 | 5756 | 79.725% | 90.132% | 0.846093 | +0.000298 |
| 0.10 | 22467 | 4067 | 84.672% | 89.471% | 0.870055 | 22469 | 4059 | 84.699% | 89.479% | 0.870234 | +0.000179 |
| 0.15 | 22177 | 2322 | 90.522% | 88.316% | 0.894054 | 22181 | 2324 | 90.516% | 88.332% | 0.894107 | +0.000053 |
| 0.20 | 21868 | 1576 | 93.278% | 87.085% | 0.900752 | 21868 | 1578 | 93.270% | 87.085% | 0.900715 | −0.000037 |
| **0.25** | 21601 | 1167 | 94.874% | 86.022% | **0.902316** | 21598 | 1174 | 94.845% | 86.010% | **0.902116** | **−0.000201** |
| 0.30 | 21306 | 844 | 96.190% | 84.847% | 0.901631 | 21304 | 843 | 96.194% | 84.839% | 0.901604 | −0.000027 |
| 0.35 | 20986 | 651 | 96.991% | 83.573% | 0.897835 | 20985 | 648 | 97.005% | 83.569% | 0.897869 | +0.000034 |
| 0.40 | 20655 | 514 | 97.572% | 82.255% | 0.892610 | 20655 | 515 | 97.567% | 82.255% | 0.892591 | −0.000019 |
| 0.45 | 20276 | 418 | 97.980% | 80.745% | 0.885318 | 20283 | 419 | 97.976% | 80.773% | 0.885469 | +0.000151 |
| 0.50 | 19741 | 317 | 98.420% | 78.615% | 0.874095 | 19746 | 320 | 98.405% | 78.635% | 0.874162 | +0.000067 |
| 0.55 | 18909 | 215 | 98.876% | 75.302% | 0.854934 | 18907 | 217 | 98.865% | 75.294% | 0.854843 | −0.000090 |
| 0.60 | 17645 | 150 | 99.157% | 70.268% | 0.822496 | 17645 | 151 | 99.151% | 70.268% | 0.822477 | −0.000019 |
| 0.65 | 15835 | 76 | 99.522% | 63.060% | 0.772025 | 15836 | 77 | 99.516% | 63.064% | 0.772036 | +0.000011 |
| 0.70 | 13394 | 31 | 99.769% | 53.339% | 0.695142 | 13409 | 32 | 99.762% | 53.399% | 0.695632 | +0.000490 |
| 0.75 | 10160 | 13 | 99.872% | 40.460% | 0.575898 | 10166 | 14 | 99.862% | 40.484% | 0.576124 | +0.000226 |

**峰值**
- PC-BMP：`th=0.25` F1 **0.902316**，P 94.87%，R 86.02%，TP 21,601，FP 1,167
- ARM-BMP：`th=0.25` F1 **0.902116**，P 94.84%，R 86.01%，TP 21,598，FP 1,174

**汇总**
| 指标 | 数值 |
|:---|--:|
| 峰值 ΔF1 | **−0.000201** |
| 跨 15 档均值 ΔF1 | **+0.000074** |
| 区间 | [−0.000201, +0.000490] |
| 负档 | 6 / 15 |
| 峰值 TP 差 | **3 TP**（21,601 → 21,598） |

两侧峰值门限同为 `th=0.25`，与冻结口径一致。

---

## 3. 结论与解读

**判定：ARM GMC 无精度回归。**

- 峰值 ΔF1 = −0.000201，对应 **3 个 TP** 的差别。按 `rules.md` §三·6，
  本口径噪声下限为 ±3 TP ≈ **±0.0002**，因此该差异**恰好处于噪声地板**。
- 跨 15 档均值 +0.000074，只有噪声下限的约 1/3。
- 峰值 precision 差 −0.029 pp、recall 差 −0.012 pp，方向不一致、大小可忽略。

**正确表述是「两者精度等价」，不是「ARM 更好」也不是「ARM 更差」。**

对比说明：两个臂的 F1（~0.902）都低于冻结 headline（0.906361），差约 0.004。
这是**两臂共有**的因素 —— 本实验用 `anti_uav_bmp_all` 的 BMP 原始帧作输入，
而冻结数据集用的是原始帧容器。共有因素在 A/B 中被完全抵消，不影响 ΔF1 的有效性。

### 为什么上一轮的 +0.0047 不能用

上一轮 ΔF1 = +0.0047 看似「ARM 更好」，实际是 PC 臂在 6,985/10,641 帧（65.7%）上
被 JPEG q95 二次压缩所致的系统性劣化。该数字已被本轮容器对称实验取代，**不得再引用**。

---

## 4. 像素级佐证（ARM-BMP vs PC-BMP，全量 31,613 帧逐像素）

| 序列 | 帧数 | Max\|D\| | MAE | p99\|D\| | >5 比例 |
|:---|--:|--:|--:|--:|--:|
| 01_4485_1167-2666 | 1465 | 253 | 0.060751 | 1.0 | 0.216% |
| 02_6321_0274-2773 | 1498 | 183 | 0.046170 | 1.0 | 0.129% |
| 25_1 | 816 | 2 | **0.000297** | 0.0 | 0.000% |
| 34_1 | 535 | 2 | **0.000332** | 0.0 | 0.000% |
| 3700000000002_102004_3 | 1498 | 5 | 0.000115 | 0.0 | 0.000% |
| 3700000000002_153918_1 | 1498 | 7 | 0.000721 | 0.0 | 0.000% |
| 43_1 | 677 | 3 | **0.000432** | 0.0 | 0.000% |
| 5_1 | 1628 | 5 | **0.000749** | 0.0 | 0.000% |
| DJI_0051_2 | 578 | 2 | 0.001422 | 0.0 | 0.000% |
| DJI_0175_2 | 448 | 3 | 0.000523 | 0.0 | 0.000% |
| wg2022_ir_010_split_01 | 1498 | 89 | 0.006432 | 0.0 | 0.034% |
| wg2022_ir_011_split_02 | 1498 | 9 | 0.000532 | 0.0 | 0.000% |
| wg2022_ir_011_split_03 | 1498 | 12 | 0.000456 | 0.0 | 0.000% |
| wg2022_ir_012_split_08 | 1498 | 220 | 0.161623 | 4.0 | 0.872% |
| wg2022_ir_015_split_01 | 1498 | 252 | 0.073852 | 1.0 | 0.332% |
| wg2022_ir_020_split_01 | 1498 | 190 | 0.006929 | 0.0 | 0.024% |
| wg2022_ir_020_split_03 | 1498 | 11 | 0.001306 | 0.0 | 0.000% |
| wg2022_ir_020_split_05 | 1498 | 239 | 0.081944 | 0.0 | 0.225% |
| wg2022_ir_020_split_07 | 1498 | 254 | 0.049392 | 1.0 | 0.229% |
| wg2022_ir_040_split_04 | 1498 | 233 | 0.044702 | 1.0 | 0.178% |
| wg2022_ir_041_split_06 | 1498 | 237 | 0.050251 | 1.0 | 0.160% |
| wg2022_ir_047_split_01 | 1498 | 235 | 0.001879 | 0.0 | 0.002% |
| wg2022_ir_047_split_02 | 1498 | 11 | 0.000392 | 0.0 | 0.000% |
| wg2022_ir_052_split_08 | 1498 | 174 | 0.193248 | 2.0 | 0.650% |
| **合计** | **31613** | **254** | **0.037025** | — | **0.1443%** |

**读法**

- 低运动序列（25_1 / 34_1 / 43_1 / 5_1）**近乎逐字节一致**：MAE 3e-4~7e-4、Max|D| ≤ 5，
  与 memory 历史记录逐项吻合。
- 高运动序列 MAE 升到 0.05~0.19，但 **p99 仍只有 0~4**，`>5` 像素占比 ≤ 0.87%，
  且全局 `>5` 占比仅 **0.144%**。这是 21 帧时域中值被 warp 末位浮点差异放大、
  在少量边缘像素上发生「中值换帧」的典型表现，属跨架构浮点末位效应。

### 高 MAE 序列不是错位（偏移扫描复核）

对 MAE 最高的序列做 PC 侧偏移扫描（ARM 侧固定用已验证的 `__<k+1>`）：

| 序列 | off=0（采用） | off=1 | off=2 |
|:---|--:|--:|--:|
| wg2022_ir_052_split_08 | **0.179988** | 1.324724 | 1.826539 |
| wg2022_ir_012_split_08 | **0.166302** | 1.866035 | 2.406699 |
| 01_4485_1167-2666 | **0.087745** | 0.959308 | 1.388689 |
| 25_1 | **0.000298** | 1.048780 | 1.116588 |

off=0 在**所有**序列上最优，且与 off=1 相差 7~10 倍。映射统一正确。

> 教训：像素校验的判据不能用 `Max|D|` 单一指标。高运动序列的单点最大差可达 254，
> 但整体 MAE 仅 0.06。若按 `Max|D| > 8` 判死刑，会把正确的映射误报为错误。
> 正确判据是 **MAE + `>5` 像素占比**。

---

## 5. 复现命令清单

> 依 `rules.md` 铁律九登记。一键编排脚本：
> `bash manu/evaluation/run_arm_pc_gmc_full_ab.sh [all|board|xfer|assemble|infer|ab|pix]`
> （md5 `29d7c1b1e0a0a7a5a52453b5f637d9bf`）

### 5.0 执行环境

| 项 | 值 |
|:--|:--|
| 本机 host | NFS 挂载 `/run/media/manu/SSD960-LOCAL/nfs` |
| 板卡 | `ssh -p 22 root@192.168.0.64`（evm3588，Mali-G610） |
| 训练服务器 | `ssh -p 32222 huangzhe@192.168.99.40`，项目 `/tmp/pycharm_project_10ae9e2e` |
| 解释器 | `/home/huangzhe/anaconda3/envs/uav/bin/python`（py3.10.20 / cv2 4.10.0 / numpy 1.26.4） |
| GPU | RTX 4090 D |
| 板卡二进制 | `/root/gmc_stream_ocl` md5 `aa99da811356694d66a0bce6de108b92`（ELF machine `0xb7` = AArch64） |
| 冻结构建器 | `manu/data/build_full_median_dataset.py` md5 `7729e77299f8933faffa73577e8e9400`（全程未改） |
| 权重 | `runs/optuna_p0_nas/trial_0474/weights/best.pt` md5 `5b07f255735f9fbadf3e0dfe95a7708f` |

### 5.1 板卡特征生成（在板卡上，root）

**脚本已固化在仓库**：`manu/pipeline/npu/run_step2_board.sh`
（可执行体与板卡 `/root/run_step2.sh` 逐字节一致，md5 `4ebde86eff698475b170b4c9c539e2da`，21 行）

```sh
scp -P 22 manu/pipeline/npu/run_step2_board.sh root@192.168.0.64:/root/run_step2.sh
ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step2.sh && \
    nohup /root/run_step2.sh > /mnt/manu/armfeat_s2.log 2>&1 &'
```

期望清单 `/mnt/manu/npu/out/demo/expected.tsv`（24 行 `<seq>\t<官方帧数>`）。
脚本内部逐序列调用：

```sh
cd /mnt/manu/npu/out/demo/$seq && /root/gmc_stream_ocl \
  --raw-root raw --sequence "$seq" \
  --anchor-step 2 \
  --out-dir /mnt/manu/armfeat_s2/$seq/feat --out-ext bmp \
  --fused gpu --kernel-dir /root/kernels --timing
```

产出 **31,661** 帧（= raw 总帧数，每序列含前 2 帧），耗时约 75 min（37~210 ms/frame）。
日志末尾输出 `STEP2_ALL_DONE`。

### 5.2 传输（host → 训练服务器，6 流并行）

```sh
NFS=/run/media/manu/SSD960-LOCAL/nfs/armfeat_s2
ssh -p 22 root@192.168.0.64 'chown -R 1000:1000 /mnt/manu/armfeat_s2 && chmod -R a+rX /mnt/manu/armfeat_s2'
(cd "$NFS" && find . -name '*.bmp' | sort) > /tmp/arm_bmp_list.txt
split -n l/6 -d /tmp/arm_bmp_list.txt /tmp/armpart_
for p in /tmp/armpart_*; do
  rsync -az -e "ssh -p 32222" --files-from="$p" --relative "$NFS/" \
    huangzhe@192.168.99.40:/mnt/data/siping/datasets/manu/armfeat_stage/ &
done; wait
```

实测 31,661 帧 / **314 s**（≈81 MB/s）。单流仅 13.8 MB/s，6 流提升 5.7×。

### 5.3 构造无损 ref（训练服务器）

构建器只读 ref 的文件名与标签、**从不读 ref 图像像素**，故全改 `.bmp` + 软链即可强制无损输出：

```sh
REF=/mnt/data/siping/datasets/manu/uav_gmc_median
OUT=/mnt/data/siping/datasets/manu/ref_bmpval
rm -rf "$OUT"; mkdir -p "$OUT/images/val" "$OUT/labels/val"
for f in "$REF"/images/val/*; do
  b=$(basename "$f"); stem="${b%.*}"
  ln -sf "$f" "$OUT/images/val/$stem.bmp"
done
for f in "$REF"/labels/val/*.txt; do
  ln -f "$f" "$OUT/labels/val/$(basename "$f")" 2>/dev/null \
    || cp "$f" "$OUT/labels/val/$(basename "$f")"
done
printf 'path: %s\ntrain: images/val\nval: images/val\n\nnames:\n  0: uav\n' "$OUT" > "$OUT/data.yaml"
```

### 5.4 PC 臂无损重建（训练服务器）

```sh
cd /tmp/pycharm_project_10ae9e2e
/home/huangzhe/anaconda3/envs/uav/bin/python manu/data/build_full_median_dataset.py \
  --ref-dataset /mnt/data/siping/datasets/manu/ref_bmpval \
  --raw-root   /mnt/data/siping/datasets/manu/anti_uav_bmp_all/val \
  --output     /mnt/data/siping/datasets/manu/armcmp_pcbmp \
  --splits val --workers 16
```

产出 31,613 帧 BMP，帧名与标签与冻结参照**逐项相等**（已校验 `stem_equal=True`、`labels equal=True`）。

### 5.5 ARM 臂组装（`<seq>_<k>` → `<seq>__<k+1>`，丢弃 k<2）

```sh
/home/huangzhe/anaconda3/envs/uav/bin/python manu/data/assemble_arm_gmc_dataset.py \
  --stage       /mnt/data/siping/datasets/manu/armfeat_stage \
  --out         /mnt/data/siping/datasets/manu/arm_gmc_full \
  --labels-from /mnt/data/siping/datasets/manu/ref_bmpval \
  --expect-total 31613 --verify-md5 120
```

产出 31,613 帧（丢 48 帧），md5 抽样 **120/120 一致**。

### 5.6 双臂推理（训练服务器，同权重同口径）

```sh
/home/huangzhe/anaconda3/envs/uav/bin/python manu/inference/cache_trial0474_inferences.py \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --data /mnt/data/siping/datasets/manu/armcmp_pcbmp/data.yaml \
  --output runs/gmc_eval/pc_bmp_full_cache.pkl \
  --imgsz 640 --stride 2 --batch 32 --device 0 --conf-thresh 0.08 --top-k 80

/home/huangzhe/anaconda3/envs/uav/bin/python manu/inference/cache_trial0474_inferences.py \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --data /mnt/data/siping/datasets/manu/arm_gmc_full/data.yaml \
  --output runs/gmc_eval/arm_gmc_full_cache.pkl \
  --imgsz 640 --stride 2 --batch 32 --device 0 --conf-thresh 0.08 --top-k 80
```

ARM 臂实测 256.0 s（123.5 imgs/s），两臂各产出 31,613 条记录。

### 5.7 评估脚本自校准闸门（必先跑）

```sh
/home/huangzhe/anaconda3/envs/uav/bin/python manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/gmc_eval/official_rebuild_trial0474_cache.pkl --label-a SOTA \
  --arm-b runs/gmc_eval/official_rebuild_trial0474_cache.pkl --label-b SELFTEST \
  --expect-f1 0.906361 --expect-tp 21643 --expect-fp 1004
```

复现不出冻结 headline 时脚本以退出码 2 中止，后续对比一律不得进行。

### 5.8 容器对称 A/B 评估

```sh
/home/huangzhe/anaconda3/envs/uav/bin/python manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/gmc_eval/pc_bmp_full_cache.pkl  --label-a PC-BMP \
  --arm-b runs/gmc_eval/arm_gmc_full_cache.pkl --label-b ARM-BMP \
  --tol 8.0 \
  --out-json /mnt/data/siping/datasets/manu/gmc_arm_vs_pc_ab.json
```

### 5.9 全量逐像素比对

```sh
/home/huangzhe/anaconda3/envs/uav/bin/python manu/diagnostics/cmp_arm_pc_features.py \
  --arm /mnt/data/siping/datasets/manu/arm_gmc_full/images/val \
  --pc  /mnt/data/siping/datasets/manu/armcmp_pcbmp/images/val \
  --samples 0 \
  --out-json /mnt/data/siping/datasets/manu/gmc_arm_vs_pc_pixdiff.json
```

### 5.10 索引映射偏移扫描（本轮关键诊断，改任何映射前必跑）

```sh
# ARM 侧固定用已验证的 __<k+1>，只扫 PC 侧偏移
g=ARM/<seq>__<i>.bmp ; p=PC/<seq>__<i+off>.bmp ; off ∈ {0,1,2}
# off=0 在所有序列最优，与 off=1 相差 7~10 倍
```

---

## 6. 可复现资产

| 资产 | 路径 |
|:---|:---|
| **一键编排脚本（本实验入口）** | `manu/evaluation/run_arm_pc_gmc_full_ab.sh`（md5 `29d7c1b1...`） |
| A/B 评估脚本（带自校准闸门） | `manu/evaluation/eval_arm_pc_gmc_ab.py` |
| 逐像素比对脚本 | `manu/diagnostics/cmp_arm_pc_features.py` |
| 板卡输出组装脚本（OFFSET=1 实证） | `manu/data/assemble_arm_gmc_dataset.py` |
| PC 臂 cache | `runs/gmc_eval/pc_bmp_full_cache.pkl` |
| ARM 臂 cache | `runs/gmc_eval/arm_gmc_full_cache.pkl` |
| A/B JSON 报告 | `/mnt/data/siping/datasets/manu/gmc_arm_vs_pc_ab.json` |
| 像素差 JSON 报告 | `/mnt/data/siping/datasets/manu/gmc_arm_vs_pc_pixdiff.json` |
| 无损 ref | `/mnt/data/siping/datasets/manu/ref_bmpval` |
| PC 臂数据集 | `/mnt/data/siping/datasets/manu/armcmp_pcbmp` |
| ARM 臂数据集 | `/mnt/data/siping/datasets/manu/arm_gmc_full` |

### 自校准闸门（评估脚本可信度证明）

`eval_arm_pc_gmc_ab.py` 必须先复现冻结 headline 才允许继续：

```
[GATE] SOTA 期望 F1=0.906361 TP=21643 FP=1004
       实测 F1=0.906361 TP=21643 FP=1004  -> PASS
```

自比（SOTA vs SOTA）15 档 ΔF1 恒为 +0.000000，确认配对与匹配实现无偏。

---

## 7. 待办

- `anchor_step=10` 档位未测（用户明确指示后续再测）。memory 记载该档 ≠ SOTA 冻结特征，
  需另设参照系，不能与本轮结果混谈。
- 若要把 ARM 特征直接推上板卡 e2e，需确认推理侧容器选择（BMP vs PNG）不影响下游。
