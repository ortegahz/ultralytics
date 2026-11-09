# 「hc32 死模型」真相：不是容量问题，是 **loader 静默丢权重** 的代码 bug

> 2026-11-06 13:20~13:50 定位并修复。
> **一句话**：上一轮 `struct4` 用来判定「head_conv=32 是死模型」的证据**完全失效**；
> 那条结论是从一个**代码 bug** 里读出来的，��不是从模型能力里读出来的。
> 连带后果：**所有 `late_upsample`（320×320 输出）结构此前从未被有效评测过**，
> 而它们正是「NPU ≤21 ms + 保精度」的主力家族。

---

## 1. 当时的结论（❌ 已作废）

`nogmc_struct4` 轮（study `mix_ratio_search_nogmc_struct4`）跑完 4 条有效臂：

| trial | 结构 | T1 | T2 | T3 | 当时的解读 |
|:--|:--|--:|--:|--:|:--|
| trial_0000 | p1 + SiLU（对照） | 0.888847 | 0.998769 | 0.899319 | 复现基线 ✅ |
| trial_0001 | p1 + SiLU（对照） | 0.887495 | 0.998679 | 0.905923 | 复现基线 ✅ |
| trial_0002 | p2_fused + hc64 + relu | 0.874786 | 0.996764 | 0.898111 | 结构 B 可用 |
| trial_0003 | p2_fused + **hc32** + relu | 0.873964 | **0.000000** | **0.000000** | ★「**死模型**」 |

当时的解读（写进了 `build_npu20ms_plan.py` 并据此降级了搜索计划）：
> 「p2_fused + head_conv32 + 160 输出 ⇒ T2/T3 全 0 ⇒ 头通道太窄 ⇒ 热图学不出来
> （32 通道不足以同时承载 320 目标语义）⇒ 160 输出的 hc32 只留 1 条。」

**这个解读是错的。**

---

## 2. 真相：T2/T3 推理端把模型**重建成另一个结构**，head 权重被静默丢弃

### 2.1 证据链（逐条可复现）

| # | 证据 | 位置 | 说明 |
|:--|:--|:--|:--|
| 1 | ckpt 自描述是 **hc32** | `trials/trial_0003/weights/epoch20.pt` 的 meta | `head_conv=32, late_upsample=False, head_src=p2_fused, stride=4` |
| 2 | 训练侧确实按 hc32 建了模型 | `logs/trial_0003.log:7` | `[head_conv] 32` |
| 3 | ★ T2/T3 缓存路径重建时打印 **`[head_conv] 64`** | `trials/trial_0003/gate/epoch20/ms_cache_shard0.log:2` | **ckpt 说 32，推理端建的是 64** |
| 4 | 同一次门禁的 T1 路径打印 `[head_conv] 32` | `trials/trial_0003/gate/epoch20/track1_infer.log:12-13` | `[STRUCT] ckpt 自描述: … head_conv=32` ⇒ **T1 口径是对的** |
| 5 | 加载张量数 **478 vs 494** | `ms_cache_shard0.log` vs trial_0002 | 差 **16** = head 的全部权重（feat_conv×2 各 5 + heatmap.0×2 + heatmap.2.w + offset.0×2 + offset.2.w） |
| 6 | 8 个序列全部 `fused=0` | `ms_cache_shard0.log` | 连 FP 都是 0 ⇒ 不是阈值问题，是**根本没有候选** |
| 7 | 训练动态健康 | `trials/trial_0003/logs/train_log.csv` epoch20 | `loss_hm=0.170985`（epoch1 是 30.7），与 trial_0002 同量级 ⇒ **排除训练崩塌** |
| 8 | T1 数字是活的 | `trials/trial_0003/trial_result.json` | `track1_f1=0.873964` ⇒ **权重本身没问题** |

### 2.2 Bug 的代码位置

`manu/data/preannotate_hm_bbox.py` 的 `load_hm_model()`：

```python
_ck_no_p2     = bool(ckpt.get("no_p2", False))       # ← 读了
_ck_head_level= str(ckpt.get("head_level", "p3"))     # ← 读了
_ck_act       = str(ckpt.get("act", "silu"))         # ← 读了
_ck_head_src  = str(ckpt.get("head_src", "") or "")   # ← 读了
_ck_p2s       = float(ckpt.get("p2_out_scale", 1.0))  # ← 读了
# ★★ head_conv 和 late_upsample **没读** ⇒ 只能靠调用方显式传
```

而 T2/T3 的调用方 `manu/data/multiscale_heatmap_cache.py:318`：

```python
model, stride = load_hm_model(
    resolve_weights(args.hm_weights), device, use_p0_highway=not args.ablate_p0
)   # ★ 没传 head_conv / late_upsample ⇒ 回落到默认 64 / False
```

⇒ 重建出的模型结构与 ckpt **不同** ⇒ head 权重 shape 不匹配 ⇒ 而 `load_hm_model` 的加载循环
**只对 `p0_highway` 前缀计数丢弃，其余 shape 不匹配静默跳过** ⇒
**head 全部保持随机初始化** ⇒ 热图恒为 0 ⇒ `det=0` ⇒ 「死模型」。

**对照**：T1 走的 `manu/inference/cache_trial0474_inferences.py:169/188` 显式读了 ckpt
（`head_conv = ckpt.get("head_conv", None)` / `late_upsample = bool(ckpt.get("late_upsample", False))`），
所以 T1 一直是好的。

### 2.3 影响面（比「hc32 一条臂」严重得多）

| 结构 | 是否受影响 | 后果 |
|:--|:--|:--|
| `head_conv ≠ 64` | ❌ T2/T3 全废 | hc32 / hc48 / hc40 的 T2/T3 数字**从来不可信** |
| `late_upsample = True` | ❌ T2/T3 全废 | ★ **stride 落回 4 ⇒ 输出变 160 而非 320 ⇒ 解码尺度全错** |
| `p2_out_scale ≠ 1.0` | ❌ T2/T3 全废 | 落回 1.0 ⇒ neck 形状不同 ⇒ head 输入通道不同 ⇒ 同样丢权重 |

⇒ **上一轮所有「结构 B 家族」的 T2/T3 数字，只有 hc64 / 无 late / sc=1.0 这一种配置是可信的**
（trial_0002 恰好命中，所以它反而是对的）。

⇒ **≤21 ms 的主力家族（320 输出的 late 变体：20.06 / 20.83 ms）此前一次都没被有效评测过。**
它们的精度上限是**未知**，不是「已知为差」。

---

## 3. 修复（2026-11-06）

1. `load_hm_model()` 补读 ckpt 的 `head_conv` / `late_upsample`；
   `late_upsample` 采用**ckpt 优先**（显式 False 无法与默认 False 区分，
   而漏掉 True 会让**结构被静默改错**，比反向覆盖严重得多）。
2. **加硬门禁**：非 `p0_highway` 的 shape 不匹配 ⇒ 直接 `raise RuntimeError`。
   宁可响亮地崩，也不要静默产出「随机头 ⇒ det=0 ⇒ 被误判成死模型」的假结论。
3. `multiscale_heatmap_cache.py` 显式传入这两个参数（双保险）。

---

## 4. 方法论教训（比这个 bug 本身更值钱）

> **「模型学不出来」和「评测时模型被换掉了」在指标上长得一模一样**（都表现为 F1=0）。

三条可复用的判别动作：

1. **看重建日志的结构打印**：`[head_conv] X` / `[STRUCT] ckpt 自描述` 是否与 ckpt 一致？
   —— 两条路径（T1 vs T2/T3）必须逐位一致。
2. **看加载张量数**：与同结构正常臂差几个？差值应能对上「被丢的张量」。
3. **看 FP 是不是 0**：`det=0, fp=0, fn=16638` ⇒ **连误报都没有** ⇒ 不是阈值问题、
   不是分布偏移，而是**输出恒定/无候选** ⇒ 优先怀疑「模型没被正确加载」。

**并且**：训练 loss 正常收敛（`loss_hm` 从 30.7 → 0.17）+ T1 正常 ⇒ 几乎可以直接排除
「容量不足 / 训练崩塌」这两个解释 —— **因为它们都该在 T1 上也坏**。

---

## 5. 对本轮搜索计划的连带修正

| 项 | 原（基于错误前提） | 现（基于真相） |
|:--|:--|:--|
| hc32 / hc48 / hc40 | 「hc32 死模型」⇒ 按头宽降级 | **不再降级** ⇒ 精度未知，值得投预算 |
| late（320 输出）家族 | 第二梯队（因 hc32 容量疑虑） | ★ **第一梯队**（stride=2 ⇒ 门禁口径与基线 cs_t0043 完全一致） |
| 搜索空间 choices | `head_conv=(64,32)` | `(64,48,40,32)` |
| `p2_out_scale` choices | `(1.0,0.75,0.5)` | `(1.0,0.75,0.5,0.25)` |
| `lr_scale` 上界 | `3.0`（plan 要 3.5/4.0 ⇒ enqueue 抛错） | `4.5` |
| study 名 | 复用 | **换新名**（categorical 不支持动态值空间） |
| plan 的 `epochs` | 走 `fixed_params` 但无人读 ⇒ **全部按 15ep 跑** | 走 `user_attrs` ⇒ 逐臂生效 |

---

## 6. 复现命令（只读，可直接跑）

```bash
# ① 看 T2/T3 路径重建成了什么结构（应与 ckpt 一致；此处显示 64 = bug）
grep -m1 'head_conv' /mnt/data/siping/runs/nogmc_struct4/trials/trial_0003/gate/epoch20/ms_cache_shard0.log

# ② 看 T1 路径（对照组：这里是对的）
grep -m2 'head_conv\|STRUCT' /mnt/data/siping/runs/nogmc_struct4/trials/trial_0003/gate/epoch20/track1_infer.log

# ③ 看加载张量数（478 = 丢了 16 个 head 张量）
grep -m1 'HM weights loaded' /mnt/data/siping/runs/nogmc_struct4/trials/trial_0003/gate/epoch20/ms_cache_shard0.log

# ④ 看训练动态（健康 ⇒ 排除容量/崩塌）
tail -3 /mnt/data/siping/runs/nogmc_struct4/trials/trial_0003/logs/train_log.csv
```

## 7. 未闭环项

- **旧轮次的数据要不要重跑**：`nogmc_struct4` 的 trial_0003（hc32）**只重跑 T2/T3 门禁即可**
  （权重还在，约 17 min 纯推理，无需重训）⇒ 能零成本判定「hc32 到底行不行」。
  ⚠️ 属评测脚本 ⇒ 按铁律一**由用户执行**（命令见 `manu/memory/nogmc_struct_line_20261105.md` 的
  「重跑 T2/T3 门禁」一节）。
- 本轮新搜索的臂**全部**走修好的 loader ⇒ T2/T3 数字可信。

---

## 8. 追加（14:20）：服务器侧完整交付链路已验证 + 「速度只取决于结构」得到实证

### 8.1 服务器可独立完成最终交付（宿主关机后不阻塞）

实测（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，脚本
`manu/training/board_measure_from_server.sh`）：

| 步骤 | 环境 | 结果 |
|:--|:--|:--|
| 导出 ONNX | conda **uav** | ✅ 6.68 MB，`output heatmap [1,1,160,160]`，`matched 494`（**全部权重加载，无丢弃**） |
| 转 RKNN int8 | conda **rknn230**（rknn-toolkit2 2.3.0） | ✅ 3.17 MB，stride=4 |
| 暂存 + 上板实测 | ssh `-i ~/.ssh/id_board` | ✅ **rknn_run 18.028 ms**，null control 全部 CHANGED，determinism IDENTICAL |

⚠️ **两个环境必须分工**（踩过）：
- `uav` 有 torch/PIL/numpy ⇒ 导出用它；
- `rknn230` 有 rknn-toolkit2 但**缺 PIL** ⇒ 在它里面跑导出会在
  `ultralytics/utils/patches.py` 挂掉（`ModuleNotFoundError: No module named 'PIL'`）。

⚠️ **标定列表路径**：`calib_nogmc_train.txt` 里是**宿主路径** `/home/manu/mnt/data/...`
⇒ 服务器上必须改写成 `/mnt/data/...`（脚本已自动处理 + 逐张校验存在性）。

⚠️ **板卡工具的 `--stride` 语义**（`rknn_heatmap_smoke.cpp:1159`）：
校验 `input/grid ratio == stride` ⇒ **必须按臂传**，不能用默认值 2
（160 输出的模型 ratio=4 ⇒ 默认值会报 `[FATAL] input/grid ratio 4 does not match --stride 2`）。

### 8.2 ★「NPU 推理时间只取决于结构、不取决于权重」得到实证

| 对象 | ms | 说明 |
|:--|--:|:--|
| 探针（cs_t0043 权重搭出的**未训练**结构） | **17.973** | 今天的结构谱系表 |
| 真实训练权重（trial_0003/epoch20，**hc32 + 160 输出**） | **18.028** | 刚刚在服务器上实测 |
| 差 | **+0.055 ms** | ✅ 0.3% ⇒ **结构决定速度**成立 |

⇒ ⇒ **周末训出来的最终模型，其 NPU 耗时可直接查表得到**，
**不需要在宿主上再测一次**（宿主 18:30 关机不影响交付）。
⇒ 且 **null control 全部 CHANGED** ⇒ 探针的权重没让输出退化 ⇒ 18 ms 是可信的。

---

## 9. ★★ 由 struct4 数据推出的**结构性结论**（决定了整个搜索策略）

### 9.1 三档 ΔT1 的分层（相对基线 cs_t0043）

| 臂 | 结构 | ms | ΔT1(pp) | ΔT3(pp) | ΔFP_T2 |
|:--|:--|--:|--:|--:|--:|
| trial_0000 | **p1 + SiLU（结构不变，只重训）** | 55.48 | **−0.192** | −0.464 | +5 |
| trial_0001 | **p1 + SiLU（结构不变，只重训）** | 55.48 | **−0.327** | +0.196 | +8 |
| trial_0002 | p2_fused + hc64 + relu + uf1（**160 输出**） | 21.28 | **−1.598** | −0.585 | **+70** |
| trial_0003 | p2_fused + hc32 + relu + uf1（**160 输出**） | 17.97 | −1.680 | *(T2/T3 数据无效)* | *(无效)* |

### 9.2 三条推论

1. **「结构不变、只重新训练」的 ΔT1 下界 = −0.19 ~ −0.33 pp**
   ⇒ 这是**种子 + 训练噪声**的量级 ⇒ **ΔT1 判据（−0.5 pp）就在噪声带边缘**。
2. **p2_fused（160×160 输出）的 ΔT1 = −1.60 pp，是噪声的 ~5 倍**
   ⇒ ★★ **这是结构性的掉点，靠换种子 / 调 lr / 调 mix 都救不回来**。
3. **ΔFP_T2 = +70**（判据 +10）⇒ 160 输出在长泉山序列上的**虚警显著上升**
   ⇒ 同向 ⇒ 160 输出的定位质量确实变差了（与 T1 掉点一致，不是巧合）。

### 9.3 ⇒ 策略结论（**本轮搜索的单一最重要判断**）

> **唯一能守住 ΔT1 ≥ −0.5 pp 的路，是把输出分辨率拉回 320×320**
> —— 也就是 **`late_upsample`**（头在 160 上算、输出 2× 上采样回 320，stride 仍为 2 ⇒
> **门禁/评测口径与基线 cs_t0043 完全相同**）。
>
> ⇒ ⇒ **搜索重心必须压在 late 家族上**，且 **160 输出的 p2_fused 家族只作为对照**。
> ⇒ ⇒ 这也解释了为什么 seed 臂要补给 **late 家族**（`p2f_hc32_late_sc025` / `p2f_hc32_late_sc05`）：
> 它们是**唯一**既 ≤21 ms、又能保住 320 输出的家族。

### 9.4 附：hc64 不是可靠候选

`p2f_hc64_relu` 的 **21.277 ms 是「预测值」**（板卡 null control FATAL ⇒ 没实测）
⇒ 实测很可能 >21 ms ⇒ **hc64 不能进候选**（trial_0002 用的正是它，且它 ΔT1 −1.6 pp、ΔFP_T2 +70）。

⇒ **≤21 ms 的可靠候选只有 hc32 / hc40 / hc48 系列**，其中 **hc32+late 是主力**。

---

## 10. ★ 剪枝：把「结构性注定失败」的臂换掉（2026-11-06 14:45）

### 10.1 依据（§9 的结构性结论）

- **160 输出的 p2_fused ⇒ ΔT1 = −1.60 pp**（噪声带的 5 倍）⇒ **靠超参/种子救不回来**
- ⇒ 把 160 输出家族的预算换成 **late（320 输出）家族的多种子**

### 10.2 剪了什么、留了什么

| 家族 | 剪前 | 剪后 | 说明 |
|:--|--:|--:|:--|
| late（320 输出） | 18 臂 | **18 臂（全留）** | ★ 唯一有希望的家族，一臂不能少 |
| p2_fused 160 输出 | 15 臂 | **6 臂** | 只留 hc48 的 6 条（160 vs 320 的**严格对照**，只差 late 开关） |
| p3（80 输出） | 1 臂 | 0 | 精度已证伪（T3 −3.2 pp） |
| p1 对照 | 2 臂 | **2 臂（全留）** | 量噪声带（ΔT1 = −0.19~−0.33 pp） |
| **合计** | 36 臂 | **26 臂** | 释放 **1900 机时（≈7.9 h 墙钟）** |

⇒ 第一轮收口从 ~29.3 h 降到 **~21.4 h**（周六 ~11:30）⇒ 释放的预算给第二轮种子臂。

### 10.3 ⚠️ 两个坑（实测踩到）

1. **前缀匹配会误伤 late 家族**
   ⇒ 用前缀 `S_p2_fused_hc32_relu_` 剪枝时，**`hc32_relu_late_*` 也被匹配上**
   ⇒ 一次误剪 **18 条最有希望的臂**！
   ⇒ ★ **剪枝必须用「显式 role 全名集合」**（`prune_roles.json`），不能用前缀。

2. **Optuna 4.9 的 `study.tell()` 剪不掉 WAITING 臂**
   ⇒ `ValueError: Cannot tell a WAITING trial.`
   ⇒（旧脚本 `enqueue_extra_arms.py` 的 `--prune-roles` 因此一直是**无效**的）
   ⇒ 正确做法：底层 `storage.set_trial_state_values(trial_id, TrialState.PRUNED)`
   （⚠️ state 要**直接传枚举**，传元组会报 `'tuple' object has no attribute 'is_finished'`）
   ⇒ 新脚本：`manu/training/prune_waiting_trials.py`（自带安全护栏：
   **拒绝剪 late 家族 / 拒绝剪非 WAITING / 拒绝前缀匹配**）

### 10.4 追加臂的两步（不改 dispatcher 也能加臂）

1. `enqueue_trial` 把新臂排进 WAITING 队列（`study.ask()` 会取它们）；
2. ★ **放宽派单上限**：dispatcher 的 `--n-trials` 是**启动参数、不可变**
   ⇒ 追加的臂会永远停在 WAITING ⇒ 必须抬高上限
   ⇒ 已加 `runtime_config.json` 的 `n_trials` 键（dispatcher **每轮重读** ⇒ 运行期可调）
   ⇒ `enqueue_extra_arms.py --bump-n-trials-to N` 一并做掉。
   ⚠️ 但**运行中的 dispatcher 是旧代码** ⇒ 热更新要重启才生效
   ⇒ **本次采用「另起第二轮」**（新 out_root + 新 study 名）⇒ 零损失零污染。
