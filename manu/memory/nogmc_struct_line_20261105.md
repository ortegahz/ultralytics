# no-GMC 结构线（cs_t0043 → YOLO26n 原生拓扑 + heatmap + all relu）

> **开线时间**：2026-11-05 18:34（服务器时钟）｜**收口**：2026-11-06 上午
> **用户目标（2026-11-05 亲述）**：
> 「在『候选：cs_t0043（3 粒子同配方 Model Soup）』这个模型的基础上，**删除 p2 分支以及 wh 预测分支**，
> 直接用 cs_t0043 这个模型初始化，并使用 **all relu**，继续搜索最优模型，
> 期望得到一个**速度大幅提升、精度微量下降**的模型。」
> 追加澄清：「**核心目的就是基本退化到了 yolo26n 原始版本，只是改成 heatmap，使用 all relu**」
> 追加强调：「**你要在 cs_t0043 这个模型上微调**」
> 追加约定：「**/mnt/data/siping/runs/ 就放这个路径下，以后都是**」
> **执行机**：训练 = 服务器 `ssh -p 32222 huangzhe@192.168.99.40`（项目根 `/tmp/pycharm_project_10ae9e2e`，conda `uav`）；
> 导出/转换/板卡 = 宿主 `/media/manu/1TB-Volume/workspace/ultralytics`（rknn230 环境）；
> 板卡 = `ssh -p 22 root@192.168.0.64`（RK3588，librknnrt 2.3.0）

---

## 一、目标结构的定义（本线唯一的新东西）

### 1.1 删了什么

| 删掉的 | 在代码里是 | 尺度 |
|:--|:--|:--|
| **neck 的 P2 融合级** | `up3` / `c19` / `down1` / `c22` / `up_p2` / `fuse_p2` | 160×160 |
| **P1 320 头分支** | `up_p1` / `fuse_p1` | 320×320 |
| **bbox/wh 回归分支** | —— | **本 checkpoint 本来就没有**（见 §1.3） |

⇒ 剩下的拓扑 = **标准 YOLO26n 的 P3/P4/P5**（骨干 b0..b10 + FPN c13/c16 原封不动）。

### 1.2 加了什么

- **head_level ∈ {p3, p2}**：头吃哪一级
  - `p3` = FPN 的 P3（**80×80**，stride 8）⇒ 最省搬运，最贴标准 YOLO26n
  - `p2` = **骨干 /4 stem 的 `p2_in`**（160×160，stride 4）⇒ 保住极小目标分辨率
  - 两者源通道都是 64 ⇒ 用一个 **1×1 adapter（`head_in`：64→48）** 统一，
    **好处**：头的 `feat_conv.0` 权重形状仍是 `(64,48,3,3)` ⇒ **cs_t0043 的头 20/20 张量全部继承**
- **`act=relu`**：全部 SiLU→ReLU（`heatmap_model.py::swap_act_to_relu`）
- **`unfreeze_b0b4`**：all-relu 会改变冻结底模 b0..b4 的函数 ⇒ 冻结着就**永久退化**；
  解冻它们只有 74,848 参数（占全网 2.9%）⇒ 设成一个**搜索维度**，让数据裁决

### 1.3 ⚠️ 一条必须写下来的澄清

**`cs_t0043` 的 checkpoint 里没有任何 wh/bbox 张量**（实测：523 个张量，头只有
`head.feat_conv.*` / `head.heatmap.0,2` / `head.offset.0,2`）。
⇒ 用户说的「删掉预测 bbox 那个分支」在**这个 checkpoint 上没有对应物**：
本线把 YOLO26 的 bbox 头**替换**成 heatmap(+offset) 头，**不引入任何 bbox 回归**。
⇒ 所以「删 wh」在本线的落地形式 = **目标结构里只有 heatmap(+offset)**，没有 wh。

### 1.4 参数与结构实测（宿主 CPU 前向验证）

| 结构 | stride | heatmap 输出 | 参数 | 继承张量 |
|:--|--:|:--|--:|--:|
| 原结构（cs_t0043） | 2 | 320×320 | 1,790,531 | —— |
| **no_p2 + p3** | 8 | **80×80** | **1,661,571（−7.2%）** | 368（head 20/20） |
| **no_p2 + p2** | 4 | **160×160** | **1,661,571（−7.2%）** | 368（head 20/20） |

---

## 二、微调口径（钉死 cs_t0043 的母配方）

cs_t0043 = trial_0043 + trial_0049 + trial_0050 的 epoch20 三粒子同配方 soup，
三者配方**逐项相同**（取自 `nogmc_t3/optuna_summary.csv`）：

```
mix_spec=w2.0  lr_scale=3.5  epochs=20  init_from=rc2+soup_strict3
warmup_epochs=1.5  weight_decay=2.69e-4  focal_beta=2.75  offset_weight=0.40
mosaic=0.15  scale=0.10  translate=0.05  clip_max_norm=5.65  freeze_bn=False
backbone_lr_scale=1  freeze_backbone_epochs=0
```

⇒ 本线**只改两处**：① 初始化权重换成 **cs_t0043** ② 结构维度变化。
⇒ **其余全部钉死**（`--struct-space` 开、`--extended-space` 关 ⇒ `epochs` 由 launcher 给 20、
`backbone_lr_scale`/`freeze_backbone_epochs` 取 dispatcher 默认值 1.0/0，与母配方逐项一致）。

### 2.1 12 条臂（确定性枚举，不用 TPE）

| 臂 | seed | 结构 | 角色 |
|:--|--:|:--|:--|
| `STRUCT_CTRL_REPRO_silu_old_s1000` | 1000 | 原结构 + SiLU | **流水线验证臂** |
| `STRUCT_CTRL_REPRO_silu_old_s1001` | 1001 | 原结构 + SiLU | **流水线验证臂** |
| `STRUCT_p3_uf1_hc64` | 1010 | P3 + relu + 解冻 + hc64 | 结构网格 |
| `STRUCT_p3_uf1_hc32` | 1011 | P3 + relu + 解冻 + hc32 | 结构网格 |
| `STRUCT_p3_uf0_hc64` | 1012 | P3 + relu + 冻结 + hc64 | 结构网格 |
| `STRUCT_p3_uf0_hc32` | 1013 | P3 + relu + 冻结 + hc32 | 结构网格 |
| `STRUCT_p2_uf1_hc64` | 1014 | P2 + relu + 解冻 + hc64 | 结构网格 |
| `STRUCT_p2_uf1_hc32` | 1015 | P2 + relu + 解冻 + hc32 | 结构网格 |
| `STRUCT_p2_uf0_hc64` | 1016 | P2 + relu + 冻结 + hc64 | 结构网格 |
| `STRUCT_p2_uf0_hc32` | 1017 | P2 + relu + 冻结 + hc32 | 结构网格 |
| `STRUCT_P3_uf1_hc64_lr3.0` | 1018 | P3 + relu + 解冻 + hc64，lr 3.0 | 配方邻域 |
| `STRUCT_P3_uf1_hc64_lr4.0` | 1019 | P3 + relu + 解冻 + hc64，lr 4.0 | 配方邻域 |

---

## 三、⭐ 流水线验证臂复现成功（本线数字可信的依据）

| | T1 | T2 | T3 |
|:--|--:|--:|--:|
| **cs_t0043（参照）** | 0.890766573 | 0.998859 (FP 34) | **0.903960496** |
| `trial_0000`（复现臂，seed 1000） | 0.889048 | 0.998829 | **0.903330** |
| `trial_0001`（复现臂，seed 1001） | 0.887877 | 0.998799 | **0.903903** |

⇒ **ΔT3 = −0.063 pp / −0.006 pp** ⇒ **参照系、门禁口径、训练链路全部正确**，
后续结构臂的数字可以信。**两个种子的 T3 都落在 0.9033~0.9039，种子噪声 ≈ ±0.0003 pp。**

---

## 四、夜里实际发生了什么（三个坑，全部已修）

### 4.1 ❌ 第一次启动：`n_trials` 名额被污染

* 第一次启动把 20 条臂（含 `late_upsample=1` 的 8 条）全入队；其中 4 条崩、5 条 `late=1` 必崩；
* `n_trials=20` 统计的是「已完成+在跑」，⇒ **11 条 FAIL + 5 条必崩臂会吃光名额，
  12 条有效臂可能一条都跑不到**。
* ⇒ **解法：换全新 study**（Optuna 4.9 无 `delete_trial`，换 study 是唯一干净解法）。
  最终：**study `mix_ratio_search_nogmc_struct2`，输出根 `/mnt/data/siping/runs/nogmc_struct2`**。

### 4.2 ❌ `late_upsample=1` 在 no_p2 结构下**必然崩**（不是 bug，是不兼容）

`HeatmapHead.late_upsample` 会把输出 **2× 上采样到 stride 2**（320×320），
而 no_p2 的目标在 stride 8/4（80/160）⇒ 形状对不上。
⇒ **no_p2 结构里没有 320 头可推迟 ⇒ 这个维度作废**，已从计划里剔除。
（教训：搜索维度要**先验证结构相容性**，否则一个维度能吃掉全部预算。）

### 4.3 ❌ `head_conv=32` 一开始必崩（我的实现错）

第一版在 worker 里**事后改模块**改 `head_conv` ⇒ 只换了 conv 权重、没换 BN 的 `running_mean`
⇒ `RuntimeError: running_mean should contain 32 elements not 64`。
⇒ **正解**：检测器**本来就支持 `head_conv` 参数**（`heatmap_model.py:759` 的 `hc`）⇒ 直接走构造器，不许事后改模块。

### 4.4 ❌ 门禁全崩：`stride` 断言没放行 8（**p3 结构的 6 条臂被误判成 CRASHED**）

```
[STRUCT] ckpt 自描述: no_p2=True head_level=p3 act=relu head_conv=64
AssertionError: Only stride 4 (P2) or stride 2 (P1) is supported, got 8.
```

⇒ **训练本身完全正常（权重完好）**，只是门禁侧 `cache_trial0474_inferences.py`
用 `ckpt["stride"]`(=8) 重建模型时撞上断言 ⇒ 缓存生成失败 ⇒ 门禁报告缺字段
⇒ 被记成 `GATE_REPORT_INCOMPLETE`（**基础设施失败，不是指标不达标**）⇒ dispatcher 记为 CRASHED。
⇒ **已修**（`stride in (2,4,8)`，并加注释说明）⇒ **权重一个字没动，补跑门禁即可**，
补跑脚本：`manu/training/rerun_struct_gate.py`（基线数字**全部从参照系 JSON 读**，不手抄）。

### 4.5 ❌ 导出侧：P0 highway 与 no_p2 不兼容

`P0ResidualHighway` 在 **stride 2（320×320）** 尺度上输出 ⇒ `feat + p0_highway(x)`
与 80/160 宽的特征加不起来：
`RuntimeError: The size of tensor a (80) must match the size of tensor b (320)`。
⇒ 训练侧本来 `use_p0_highway=False`、门禁侧带 `--ablate-p0`，**只有导出路径还在开 P0** ⇒ 已修。
⇒ **教训：一个结构改动要过的闸门至少有 4 个**（训练 / 门禁缓存 / 导出 / 板卡），
**只验训练会漏掉 3 个**。

---

## 五、p2 结构（头吃骨干原始 /4 特征）的失败机理

4 条 p2 臂**训练完成但精度塌了 ~20 pp**（T1 ≈ 0.730 / T2 ≈ 0.926 / T3 ≈ 0.691，vs cs_t0043 的
0.891 / 0.999 / 0.904）。

**机理（明确）**：删掉 `fuse_p2` 之后，头只能吃**骨干原始 /4 特征 `p2_in`** ——
它**没有经过 FPN 的语义融合**，是纯低层特征 ⇒ 热图头在上面学不出来。
⇒ **"头吃 P2"要成立，必须保留一个轻量的 P2 语义融合**（不是裸骨干特征）。
⇒ **本线 p2 臂作废**，但**机理留档**：下一版若要保 160×160，正确形态是
「删 P1 320 头 + 保留 `fuse_p2`」而不是「删整个 P2 融合」。

---

## 六、新增产物

| 类别 | 路径 |
|:--|:--|
| 结构旗标（模型） | `manu/models/heatmap_model.py`：`no_p2` / `head_level` / `act` / `swap_act_to_relu` |
| ckpt 自描述（训练/缓存/导出三处一致） | `manu/models/heatmap_model.py`、`manu/data/preannotate_hm_bbox.py`、`manu/inference/cache_trial0474_inferences.py`、`manu/pipeline/npu/hm_export_onnx.py` |
| 搜索空间 | `manu/training/optuna_mix_ratio_search.py`：`--struct-space` + `STRUCT_*` 维度 |
| 计划 | `manu/training/nogmc_struct_plan.json`（12 条确定性臂） |
| 启动器 | `manu/training/launch_nogmc_struct_search.sh` |
| 门禁补跑 | `manu/training/rerun_struct_gate.py` |
| 验收合并表 | `manu/training/nogmc_struct_acceptance.py` |
| 板卡测速 | `manu/pipeline/npu/run_hm_speed_board.sh` |
| 产物根 | **`/mnt/data/siping/runs/nogmc_struct2/`**（用户 2026-11-05 定：**以后都放这里**） |

---

## 七、待续（下一轮接手须知）

1. **门禁补跑**（6 条 p3 臂）：`rerun_struct_gate.py` ⇒ 产出 T1/T2/T3
2. **板卡测速**（8 臂同板同日）⇒ 产出 `rknn_run` 对照表（基线 = `trial_0000/0001` 复现臂，**同板同日**）
3. **验收合并**：`nogmc_struct_acceptance.py` ⇒ 一张表回答"速度大幅提升 + 精度微量下降"
4. **下一版结构**（若继续）：把 p2 臂改成「**删 P1 320 头 + 保留 `fuse_p2` 语义融合**」，
   预计能同时拿到 160×160 的分辨率与 P3 级的搬运量

---

## 八、⭐⭐⭐ 板卡 NPU 实测（2026-11-06，同板同日、同工具、同输入）

> 工具：板卡 `/root/rknn_heatmap_smoke`（热图契约），`--loops 30 --warmup 10`，
> 输入 `feat.jpg` 640×640，int8。
> ⚠️ **口径陷阱（先记这个）**：板卡上
> - `base_int8.rknn`（历史 merge_head base，**P0 关**）⇒ **55.4 ms**
> - 本线训练出的控制臂 `trial_0000`（原结构，**P0 在**）⇒ **103.6 ms**
> 两者**架构相同**，差的是 **P0 highway**（全分辨率 640×640 残差流）⇒ **55.4 与 103.6 不可混用**。

### 8.1 全表

| 臂 | 结构 | `rknn_run` | 算子数 | NPU 周期 | RW |
|:--|:--|--:|--:|--:|--:|
| `ctl_trial_0000_nop0` | **原结构 + P0 关**（同口径基线） | **55.376 ms** | — | — | — |
| `trial_0000` | 原结构 + P0 在（控制臂） | 103.661 ms | 734 | — | — |
| `trial_0001` | 同上，seed 1001 | 103.643 ms | | | |
| `trial_0002` | **no_p2 + P3 + hc64 + uf1** | **10.080 ms** | 101 | 4.90 ms | 30.9 MB |
| `trial_0003` | **no_p2 + P3 + hc32 + uf1** | **9.392 ms** | 101 | 4.90 ms | 30.9 MB |
| `trial_0004` | no_p2 + P3 + hc64 + uf0 | 10.083 ms | 101 | 4.90 ms | 30.9 MB |
| `trial_0005` | no_p2 + P3 + hc32 + uf0 | 9.406 ms | 101 | 4.90 ms | 30.9 MB |
| `trial_0006` | no_p2 + P2 + hc64 + uf1 | 8.446 ms | **19** ⚠️ | 6.02 ms | 20.1 MB |
| `trial_0007` | no_p2 + P2 + hc32 + uf0 | 5.705 ms | **19** ⚠️ | 6.02 ms | 20.1 MB |

**9/9 个 null control（反相图必改输出）全部 PASS** ⇒ 模型真实、非死模型。

### 8.2 ⭐ 归因分解（**三条必须分开记的杠杆**）

| 杠杆 | 证据 | 收益 |
|:--|:--|--:|
| **① P0 highway 的成本** | 103.65（开） vs 55.38（关），同配方同初始化 | **48.3 ms（47%）** |
| **② 结构净收益**（同为 P0 关） | 55.38 → **9.39**（hc32）/ 10.08（hc64） | **5.9× / 5.5×** |
| **③ 合计** | 103.65 → 9.39 | **11.0×** |

⇒ **对外引用必须写清是"含 P0 关闭"还是"结构净收益"**，否则会把 P0 的账算到结构头上。
⇒ **对照标准 YOLO26n（P3-P5 检测头）= 23.7 ms** ⇒ 本线 P3 头 **9.39 ms 仍快 2.5×**
（热图头比 Detect 头轻 + 没有 P0 全分辨率流）。

### 8.3 ⚠️ **p2 臂的"更快"是假象，必须作废**

p2 臂（160×160 输出）实测 5.71–8.45 ms，**比 p3 臂还快** —— 这与"分辨率更大 ⇒ 更慢"矛盾。
⇒ **根因：p2 臂的 RKNN 算子表只有 19 个算子（p3 是 101）** ⇒ **它的图在 int8 量化时被折叠了**
（塌陷模型的权重退化 ⇒ conv 被常量折叠）⇒ **速度优势是"模型坏掉了"的副产品**。
⇒ **p2 臂既塌精度（T1 0.73）又是坏图 ⇒ 整条臂作废，其 5.71 ms 不可引用。**

### 8.4 板卡测速的三个坑（都要写进操作手册）

1. **stride 必须按臂传**：板卡工具校验 `input/grid ratio == 640/stride`，
   不传就默认 2 ⇒ 新结构臂被 `[FATAL] input/grid ratio 8 does not match --stride 2` 全拒。
   ⇒ 做法：导出日志里的 `[model] stride=N` ⇒ 写成 `<arm>.stride` 边车文件 ⇒ 随 .rknn 一起 scp 到板卡 ⇒ 板卡侧读。
2. **边车文件命名**必须与 .rknn **替换后缀**对齐（`<arm>.rknn` ⇒ `<arm>.stride`，不是 `<arm>.rknn.stride`）。
3. **同口径基线要现场测**：脚本每次会话都重测 `base_int8.rknn`，因为 **RKNPU 跨天会漂移**，
   历史 55.5 ms 不能直接当今天的对照。

### 8.5 速度侧还有一条未结

`p3 + hc64` 与 `p3 + hc32` 的 NPU 周期**完全相同**（4,896,800）⇒ **head_conv 从 64 改 32 不省算力**
（头在全分辨率上，周期表按 MAC 算），但实测省 ~0.7 ms ⇒ 差在**搬运量**（30.9 MB → 更少）。
⇒ 与 §77.9b 的成本模型一致：**降 RW 有效，降 MAC 无效**。

---

## 九、⭐⭐⭐ 第一轮结论（结构 A = 删 P2 融合、头降到 P3）：**速度达标，精度不达标**

### 9.1 三轨实测（同板同日、同参照系、同母配方）

| 臂 | 结构 | T1 | ΔT1(vs cs_t0043) | ΔT1 带均 | T2 | T3 | **ΔT3(vs cs_t0043)** |
|:--|:--|--:|--:|--:|--:|--:|--:|
| `trial_0000` | 原结构+SiLU（**复现对照**） | 0.889048 | −0.17 pp | | 0.998829 | **0.903330** | **−0.063 pp** ✅ |
| `trial_0001` | 原结构+SiLU（复现对照，seed2） | 0.887877 | −0.29 pp | | 0.998799 | **0.903903** | −0.006 pp ✅ |
| `trial_0002` | **no_p2 + P3 + hc64 + uf1** | 0.876023 | **−1.47 pp** | **−4.57 pp** | 0.994082 | 0.871750 | **−3.22 pp** ❌ |
| `trial_0003` | no_p2 + P3 + hc32 + uf1 | 0.877320 | −1.34 pp | −3.26 pp | **0** | **0** | — **T2/T3 死模型**（absent=全部）❌ |

⇒ **结构 A（头降到 P3/80×80）不满足"精度微量下降"**：T3 −3.2 pp、T1 −1.5 pp（带均 −4.6 pp）。
⇒ **机理**：P1 头（320×320）本来就是为了 1~3 px 极小目标；降到 P3（80×80，每个 cell 覆盖 8×8 像素）
⇒ 小目标定位崩塌。**删 P2 分支在精度上是硬伤。**

### 9.2 ⭐ 由此得到的结构 B（下一轮主角）

**既然"删 P2 融合级"会伤精度，而真正的搬运大头是 320×320 的 P1 头 ⇒ 那就只删 P1 头。**

**结构 B = 保留 P2 融合级（`up3/c19/down1/c22/up_p2/fuse_p2`），只删 P1 320 头分支（`up_p1/fuse_p1`），
头吃 `fuse_p2` 的 160×160（有 P3 语义）** ⇒ 320×320 的大张量照样消失，但热图仍在 stride 4。

⚠️ **与本线第一版的 p2 臂有本质区别**：
| | 第一版 p2 臂（已证伪） | **结构 B** |
|:--|:--|:--|
| 头吃的特征 | 骨干**原始** /4（`p2_in`） | **P2 融合后**（`fuse_p2`，带 P3 语义） |
| 结果 | T1 0.73 / T3 0.69（塌陷） | 待测 |

⇒ 本地 CPU 验证（4 结构全通）：

| head_src | stride | heatmap | 继承张量 | head 继承 | 参数 |
|:--|--:|:--|--:|--:|--:|
| `p1`（现状） | 2 | 320×320 | 494 | 20/20 | 1,790,531 |
| `p2_raw`（已证伪） | 4 | 160×160 | 368 | 20/20 | 1,661,571 |
| **`p2_fused`（结构 B）** | 4 | **160×160** | **488** | **20/20** | **1,759,043（只少 1.8%）** |
| `p3`（结构 A） | 8 | 80×80 | 368 | 20/20 | 1,661,571 |

⇒ **结构 B 几乎全继承（488/494）⇒ 微调起点极好。**

### 9.3 第二轮搜索已启动

* study `mix_ratio_search_nogmc_struct4`，输出根 `/mnt/data/siping/runs/nogmc_struct4`
* dispatcher **PID 2845104**（setsid 独立 session，宿主关机不影响），**4 卡全开，零崩溃**
* 10 条臂：2×复现对照(p1+SiLU) + 6×结构B(uf{0,1}×hc{32,64}) + 1×结构A对照(p3) + 1×all-relu单独贡献对照(p1+relu) + 2×配方邻域(lr 3.0/4.0)
* **计划：10 臂 × 20 ep ≈ 200 min/臂 ÷ 4 卡 ≈ 8.3 h ⇒ 约 19:15 收口**

---

## 十、这一夜的四个坑（全部已修，全部要留档）

1. **`n_trials` 名额被污染** ⇒ 旧 study 的 FAIL + 必崩臂吃光名额 ⇒ 有效臂一条跑不到
   ⇒ **解法：换新 study**（Optuna 4.9 无 `delete_trial`）。
2. **`late_upsample=1` 在 no_p2 结构下必然崩** ⇒ 头会 2× 上采样到 stride 2，与 stride 8/4 的目标对不上
   ⇒ **no_p2 结构里没有 320 头可推迟 ⇒ 这个维度作废**。
3. **`head_conv` 不能事后改模块** ⇒ 只换 conv 权重、BN 的 `running_mean` 通道数不匹配
   ⇒ `RuntimeError: running_mean should contain 32 elements not 64`
   ⇒ **正解：检测器本来就支持 `head_conv` 参数 ⇒ 走构造器**。
4. **一个结构改动要过的闸门有 4 个**（训练 / 门禁缓存 / ONNX 导出 / 板卡），**只验训练会漏掉 3 个**：
   * 门禁：`stride` 断言没放行 8 ⇒ 6 条 p3 臂的**门禁**全被打成 `GATE_REPORT_INCOMPLETE`（训练正常、权重完好）
   * 导出：P0 highway 在 stride 2 输出，与 no_p2 的 80/160 宽特征加不起来
     ⇒ `RuntimeError: The size of tensor a (80) must match the size of tensor b (320)`
   * 板卡：工具校验 `input/grid ratio == 640/stride` ⇒ 新结构臂被全拒

⇒ **⇒ 铁律四的"事实必须实时入档"在这里的含义：不只记结论，还要记"哪个闸门会拦住新结构"。**
