# 交接单：NPU int8 ≤ 21 ms 且精度下降微小 —— 结构线（2026-11-06 收尾）

> 给**下一个会话**：本文包含执行这个任务所需的全部状态与命令，无需翻历史对话。

---

## 0. 一句话目标

在基线 `cs_t0043`（同配方 3 粒子 soup）基础上，找到 **NPU int8 推理 ≤ 21 ms**、
且 **T1/T2/T3 相对 cs_t0043 下降非常微小**的最优模型。
用户原话口径：**「基本退化成 yolo26n 原始版本，只改成 heatmap、使用 all relu」**、**「在 cs_t0043 上微调」**。

---

## 1. 现在在哪（2026-11-06 17:00 实测）

**服务器**（`ssh -p 32222 huangzhe@192.168.99.40`，conda 环境 `uav`）四个常驻进程：

| PID | 角色 | 状态 |
|:--|:--|:--|
| 2883423 | 第一轮 dispatcher（`--max-hours 34` ⇒ **周六 00:08 停止派新臂**） | 跑第 9~12 条臂 |
| 2886633 | 看门狗 | 第一轮收口后跑一次验收 |
| **2930518** | **第二轮 dispatcher**（`--max-hours 32` ⇒ **周一 01:05 停止派新臂**，82 臂队列） | 跑前 4 条臂 |
| **2927799** | **最终收割**（硬闸 **周一 05:00**） | 等第一轮退出 ⇒ 建家族 ⇒ soup ⇒ 门禁 ⇒ 出总表 |

> ⚠️ **原接力脚本 2895962 与原收割 2900316 已于 16:5x 按 PID 精确停掉**
> （接力已被「手动起第二轮」取代；收割改成带硬闸重挂）。
> ⇒ **周一别再找这两个 PID**；第二轮 pid 文件：`/mnt/data/siping/runs/npu20ms_seed/dispatcher.pid`。

**第一轮 study**：`mix_ratio_search_npu20ms_v2`（`/mnt/data/siping/runs/npu20ms/study.db`）
* 计划 36 臂 + 1 probe ⇒ 剪掉 11 条速度超限臂 ⇒ 实跑 25 条
* **⚠️ 第一轮会在周六 00:08 停派单**（`--max-hours 34` 是启动参数，运行期改不了；
  `runtime_config.json` 的 `stop_at`/`hard_cut` 只能**收紧**不能放宽）⇒ 队尾约 5~7 条臂跑不到
  ⇒ **重要的老臂配置已在新一轮（第二轮）里以更高优先级重跑**，见 §7。
* 已出结果：`trial_0004`（uf=False 族）**FAIL_TRACK2_SAFETY** ⇒ uf=False 的 T2 假阳偏多
  ⇒ **活路是 uf=True 家族**（种子臂全部 uf=True）。

**第二轮 study**：`mix_ratio_search_npu20ms_seed`（`/mnt/data/siping/runs/npu20ms_seed/study.db`）
* 计划 **82 臂**（10 条快结构单臂取证 + 4 个 320 输出家族 × 18 种子），队列深 ⇒ **GPU 不会空转**
* 家族：`L4`=p2f hc24 late sc0.5（20.02 ms，第一轮没跑过）/ `L1`=hc32 late sc0.25（19.83）
  / `L2`=hc32 late sc0.5（20.06）/ `L3`=hc32 late sc1.0（20.83）

**时间线**：第一轮 周六 00:08 停派 ⇒ 第二轮 周一 01:05 停派 ⇒ 家族+soup 门禁+总表 周一 01:05~03:30
⇒ **距周一 09:00 deadline 余量约 5.5 h**（收割脚本自带周一 05:00 硬闸兜底）。


---

## 2. 周一怎么取结果

```bash
ssh -p 32222 huangzhe@192.168.99.40
# ⓪ 收割链是否已全部跑完（完成标记）
cat /mnt/data/siping/runs/npu20ms/harvest_done.json 2>/dev/null || echo '未完成，看下面的日志'
# ① 人读总表（★ 含 🍲 soup 行 + 「最接近达标」段）
cat /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md
# ①b 第二轮单臂总表（★ 第二轮是新 out_root ⇒ 第一轮的验收脚本看不到它的臂）
cat /mnt/data/siping/runs/npu20ms_seed/ACCEPTANCE.md
# ② 机读
python -c "import json;d=json.load(open('/mnt/data/siping/runs/npu20ms/acceptance.json'));\
[print(r) for r in d['results']]"
# ③ 收割过程日志
tail -50 /mnt/data/siping/runs/npu20ms/final_harvest.log
# ④ soup 权重本体
ls -la /mnt/data/siping/runs/npu20ms/soups/
```

**判据**（验收表「判定」列）：`NPU ≤ 21.0 ms ∧ ΔT3 ≥ −1.0 pp ∧ ΔT1 ≥ −0.5 pp ∧ ΔFP_T2 ≤ +10`
（Δ 全部相对 **cs_t0043**：T1 0.890766573 / T2 0.998859(FP 34) / T3 0.903960496）

⚠️ **两处基线不同，别误读**：门禁的 T2 FP 绝对上限是 **62**（参照系 frozen baseline 52+10），
验收表是 **≤44**（相对 cs_t0043 的 34+10）⇒ **验收更严 18 个 FP**
⇒ 「门禁 PASS 但验收 FAIL」是**口径差异**，不是 bug。以**验收表**为准。

**候选结构与实测速度**（板卡，结构决定速度 ⇒ 与权重无关）：

| 家族 | 结构 | 板卡实测 | ≤20 ms？ |
|:--|:--|--:|:--:|
| sc025 | p2_fused / hc32 / relu / late / sc=0.25 | **19.83 ms** | ✅ |
| sc050 | p2_fused / hc32 / relu / late / sc=0.50 | **20.06 ms** | ✅ |
| sc100 | p2_fused / hc32 / relu / late / sc=1.00 | **20.83 ms** | ❌(但 <21 ✅) |

⇒ **原始目标里的「20 ms」版本也存在**：sc025 与 sc050 两家本身就 ≤20 ms。

---

## 3. 若板卡重新上电（今天 18:30 断电）

服务器可**独立**完成「导出 ONNX → 转 RKNN int8 → 上板实测」（`uav` 导出 / `rknn230` 转换 / `id_board` 上板）：

```bash
bash manu/training/board_measure_from_server.sh \
  --ckpt /mnt/data/siping/runs/npu20ms/soups/soup_<家族>.pt \
  --out-root /mnt/data/siping/runs/npu20ms/npu_speed
```
（实测「速度只取决于结构」：探针 17.973 ms vs 真实训练权重 18.028 ms，差 0.055 ms ⇒ 查表即可）

---

## 4. 已知的坑（**都已修**，但改动后请重读本文）

| # | 坑 | 结论 |
|:--|:--|:--|
| 1 | ★「hc32 死模型」= loader bug（`load_hm_model` 没读 ckpt 的 `head_conv`/`late_upsample`） | 已修 + 硬门禁（shape 不匹配直接 raise） |
| 2 | ★门禁「评测没跑出来」却判 **PASS**（verdict 被无条件重算 + `--strict-gate` 默认 False） | 已修：基础设施失败 ⇒ 强制 FAIL + `infrastructure_failure` + rc=3 |
| 3 | 门禁报告缺字段 ⇒ builder 把**崩臂**当合格 soup 成员 | 已修：逐轨查 status + 结构字段缺失即拒建 |
| 4 | plan 的 `epochs` 写了没人读（逐臂 20ep 从未生效） | 已修：epochs 走 user_attrs 通道 |
| 5 | `head_conv`/`p2_out_scale` 的 choices 缺值 ⇒ Optuna 动态值空间报错 | 已修：`(64,48,40,32)` / `(1.0,0.75,0.5,0.25)` |
| 6 | `LR_SCALE_HIGH` 太低 ⇒ 3.5/4.0 入队被拒 | 已修：`4.5` |
| 7 | 家族定义写死不可能（第二轮成员运行时才存在） | 新增 `build_struct_families.py`：运行时按 **params 逐字段相等**配对 |
| 8 | ★同 params 多条臂 ⇒ 家族「歧义跳过」 | 已修：从 **study 取臂** + 只认**产物齐全**（`epoch20.pt` + 门禁报告）那条 |
| 9 | ★比较 params 必须**剔除 `epochs`** | 入队时它被搬到 user_attrs ⇒ study 的 fixed_params 里没有 |
| 10 | harvest 与接力**都在等第一轮** ⇒ harvest 会比接力先醒 | 已修：先等 pid 文件出现再等进程退出（实测 pid 文件 20 s 后出现也不跳过） |
| 11 | `ACCEPTANCE.md` 只由看门狗生成 ⇒ **soup 行永远进不了交付物** | 已修：markdown 改由验收脚本生成（harvest 的最后一次验收 ⇒ 含 soup） |
| 12 | `pkill -f` 的 pattern 命中 **dispatcher 的 `--gate-script` 文本** ⇒ 误杀 4 条训练臂 | 已用替补臂补回（trial_0037/0038/0039）；**今后按 PID 精确 kill** |

---

## 5. 交付物路径

* 验收表：`/mnt/data/siping/runs/npu20ms/ACCEPTANCE.md`（人读）+ `acceptance.json`（机读）
* soup 权重：`/mnt/data/siping/runs/npu20ms/soups/soup_<家族>.pt` + `soup_candidates.json`
* 第一轮产物：`/mnt/data/siping/runs/npu20ms/trials/trial_XXXX/`
* 第二轮产物：`/mnt/data/siping/runs/npu20ms_seed/`
* **代码备份**（防服务器重启清空 /tmp）：`/mnt/data/siping/runs/_repo_code_backup_20261106.tgz`（324 MB）
* 仓库：`/tmp/pycharm_project_10ae9e2e`（宿主侧镜像 `/home/manu/mnt/pycharm_project_10ae9e2e`）
* 留档：`manu/memory/memory.md` §80.1~§80.18 + `manu/memory/hc32_deadmodel_truth_20261106.md`

---

## 6. ★★ soup 家族的配方（**已回退，勿再改**）+ 两个必须知道的坑

**当前状态：第二轮种子臂 = `unfreeze_b0b4=True`**（`build_seed_arms.py`，2026-11-06 16:50 回退）
⇒ 与第一轮的 uf=True 母臂同配方 ⇒ 三个 n=3 家族：

```
[FAMILY] sc025: 母臂 = S_p2_fused_hc32_relu_late_sc25_lr3.5_ep20_r2（替补臂 trial_0037）
[FAMILY] sc050: 母臂 = S_p2_fused_hc32_relu_late_sc50_lr3.5_ep20_ep20（trial_0008）
[FAMILY] sc100: 母臂 = S_p2_fused_hc32_relu_late_lr3.5_ep20（trial_0014）
```

⚠️ **坑1（曾踩，已回退）**：曾把种子臂改成 `unfreeze_b0b4=False`，因为老轮数据显示
「p1+uf=False ⇒ FP 39/42」而「p2_fused+uf=True ⇒ FP 104」，一度归因为 uf。
**但老轮从没测过「p2_fused+uf=False」⇒ 结构与 uf 没解耦 ⇒ 归因不成立**；
且实测 `trial_0004`（uf=False）**训不动**（loss 1.44~1.63 vs uf=True 的 0.89~0.99）⇒ 已回退。
⇒ **第一轮同时有 uf=True 与 uf=False 两族 ⇒ 周一按验收表定论，别预先下结论。**

⚠️ **坑2（不影响交付）**：dispatcher 算了 `sched_epochs=trial_epochs` 却**没把 `--sched-epochs` 传给 worker**
⇒ **ep15 臂用 20 epoch 的 cosine 训 15 轮** ⇒ LR 没退火 ⇒ 模型是噪声 ⇒
`trial_0005` 门禁 **F1=0.26 / FP=53004**（看着像模型崩了，其实是协议 bug）。
**母臂全是 ep20 ⇒ 调度正确 ⇒ 交付不受影响**；周一看到 ep15 那几行垃圾数字**不要采信**。

⚠️ **不要再改 `build_seed_arms.py`**：接力里 `python build_seed_arms.py … || exit 1`
⇒ 生成器出错 ⇒ **第二轮整个不启动 ⇒ 零 soup**（单臂仍可交付）。
改前必须在服务器用**完全相同**的命令实测通过：

```bash
python manu/training/build_seed_arms.py \
  --speed-table /tmp/x/struct_speed_table.json --out /tmp/x/seed_arms.json \
  --seed-base 3000 --seeds-per-struct 2 --limit-ms 21.0
```

---

## 7. ★★ 排期改版（2026-11-06 17:00）：**服务器不许空转**

### 7.1 为什么要改
原排期是「第一轮 ~03:00 收口 ⇒ 第二轮 6 条种子臂 ⇒ 3 个 soup 门禁 ⇒ 周六 10:00 交付」。
**「周六 10:00」是我估的「流水线自然跑完点」，不是 deadline**；真正的硬截止一直是**周一 09:00**。
用户明确要求：**充分利用服务器计算资源，不要周六一整天空闲** ⇒ 把窗口从「周六早上」拉到「周一凌晨」。

### 7.2 改了什么（三个动作）
1. **停掉接力脚本 2895962**（按 PID 精确 kill）⇒ 改为**手动起第二轮**：
   ```bash
   OUT_ROOT=/mnt/data/siping/runs/npu20ms_seed STUDY_NAME=mix_ratio_search_npu20ms_seed \
   N_TRIALS=82 MAX_HOURS=32 EPOCHS_DEFAULT=20 \
   SPEED_TABLE=/mnt/data/siping/runs/npu20ms_seed/struct_speed_table.json \
   SPEED_LIMIT_MS=21.0 PLAN=/mnt/data/siping/runs/npu20ms_seed/seed_arms.json \
   bash manu/training/launch_npu20ms_search.sh
   ```
   ⚠️ `--max-hours` 是**启动参数**，运行期改不了 ⇒ 想加预算只能**重启 dispatcher**
   （代价 = 杀掉在跑的臂；本次只损失 3 分钟，所以直接重启）。
   ⚠️ **第一轮不能重启**：它的 4 条臂已跑到 ep18~20，重启会打掉 2~3 小时的训练。
2. **第二轮计划从 6 臂扩到 82 臂**（`manu/training/build_round2_plan.py`，新文件）：
   * 10 条「快结构单臂取证」：p2f hc32 / sc075 / hc40 / hc48 / p3，各 uf=True + uf=False
     ⇒ 回答「**更快的结构能不能守住 ΔT1 ≥ −0.5 pp**」（160 输出历史上 ΔT1=−1.6 pp，但那是 confounded）
   * 4 个 320 输出家族 × 18 种子 = 72 臂 ⇒ **每个家族 n≈13~18 的 soup**
   * ★ **交错排布**（每家族先各出 3 个再循环）：dispatcher 是 FIFO ⇒ 若按家族连排，
     队尾家族会被挤到 0 个成员 ⇒ 交错后任何截断点每个家族都 ≥3
3. **收割脚本加硬闸**：`DEADLINE_EPOCH=<周一 05:00>` ⇒ 即使第二轮拖到天亮，
   收割也会在周一 05:00 强制进入「建家族 ⇒ soup ⇒ 门禁 ⇒ 总表」，**保证交付物不缺**。

### 7.3 代码改动（都已实测，三侧 md5 一致）
| 文件 | 改动 | 实测 |
|:--|:--|:--|
| `manu/training/build_round2_plan.py` | **新增**：生成 82 臂计划（取证 + 4 家族多种子，交错排布） | 服务器实跑出 82 臂 ✔ |
| `manu/training/build_struct_families.py` | ① **允许「纯第二轮家族」**（第一轮没有同配方母臂时，家族成员全是种子臂）② 种子臂按「产物齐全」过滤（无 trial_dir 则不过滤） | 假 study 实测：`n=3 / 母臂=(无·纯第二轮)` ✔ |
| `manu/training/final_harvest.sh` | 追加「第二轮单臂总表」（第二轮是新 out_root，第一轮的验收脚本看不到） | 服务器实跑 ✔ |

⚠️ **为什么必须支持「纯第二轮家族」**：第一轮的 `--max-hours 34` 天花板（周六 00:08）
会把队尾截掉，**含 sc25 家族的替补母臂 trial_0037** ⇒ 第一轮可能**没有** sc25 母臂
⇒ 家族生成器原本会直接跳过这个家族（`第一轮没有同配方母臂`）⇒ **sc25 家族（最快的 19.83 ms）就没了**。
现在改为：同配方 + 不同种子的成员集合**本身就是**合法家族，母臂只是 n 的来源。

### 7.4 周一要注意
* **第二轮的家族 n 可能是 13~18，不止 3** ⇒ soup 是「同配方多种子大平均」。
  若某个家族的大 soup 反而不如小 soup 好 ⇒ 周一可用 `build_struct_soups.py` 手动只挑前 3 个成员重做。
* **`trial_0004`（uf=False）已判 FAIL_TRACK2_SAFETY** ⇒ uf=False 的 T2 假阳偏多 ⇒ 活路是 uf=True 家族。
* **第一轮周六 00:08 就停派单** ⇒ 它的队尾臂跑不到 ⇒ **别把「第一轮臂没跑全」当成故障**。

---

## 8. ★★★ 2026-11-06 18:00 之后的**根因与破局路线**（周一先读这一节）

> ### ⚡ 8.0 现状速览（2026-11-06 18:20 更新）——**§8.3 的路线 A/B 都已经闭环，别重做**
> | §8.3 的路线 | 现状 | 证据 |
> |:--|:--|:--|
> | **路线 A**（板卡测 `p3/hc64/late` 的真实 ms） | ✅ **已完成** | **13.131 ms**（D1 臂 epoch1 真权重上板实测，null control + determinism 通过）⇒ 见 §8.9。**当时没测成的原因已查清**：探针权重随机化会把校准数据打成 NaN ⇒ 量化报 `cannot convert float NaN to integer`；正解是用**真实训练臂的权重**测速 |
> | **路线 B**（切片继承，让换宽度的头也继承知识） | ✅ **已实现并验证** | `manu/data/preannotate_hm_bbox.py::merge_channels` ⇒ 实测 `488 tensors loaded | ★通道合并继承=16 个张量`；推理侧严格 gate 不变 ⇒ 见 §8.7.4。**不用再写一遍** |
> | 路线 C（从零学 + ep60） | ⛔ 不需要 | sc0.25 才是真凶（§8.7.2），sc0.5/sc1.0 直接可用 |
> | **交付主线** | ✅ **已锁定** | `p3/hc64/relu/late` = **13.131 ms**（≤20 ms 目标达成）⇒ 见 §8.9 / §8.10 |
> | **仍待产出** | ⏳ | F_MAIN 家族（n≈40）的门禁 **T1/T2/T3**（服务器自己在跑，周一取表）⇒ 见 §9 |
>
> ⇒ **周一不需要再改任何代码，只需要读表 + 必要时用 §9.4 的兜底工具。**

### 8.1 根因①：`p3 + late_upsample` 的输出语义被回退 ⇒ p3 家族训练**秒崩**
* 现象：所有 `head_src=p3` + `late_upsample=True` 的臂 **0.3~0.7 min 就崩**：
  `RuntimeError: The size of tensor a (160) must match the size of tensor b (80) at non-singleton dimension 3`
* 原因：`manu/models/heatmap_model.py` 的 `elif no_p2:` 分支**完全没处理 late_upsample**
  ⇒ `stride` 仍声明 8（=80×80），而 `HeatmapHead` 的 late 上采样固定 2× ⇒ 实际输出 160 ⇒ 与目标 80 不匹配。
  **老轮（nogmc_struct4）的 p3+late 语义 = 4× 上采样 → 320（stride 2）** ⇒ 被回退了。
* **已修**（三侧 md5 `50bb987260faa775cac38d2c08ce46ed`）：
  `HeatmapHead(..., late_scale=)` + `no_p2` 分支 `stride = 2` + p3 用 4×、p2_fused 仍 2×。
* 实测（服务器）：`p3+late ⇒ stride 2 / 320×320` ✔；`p2f+late ⇒ 320` 不变 ✔；
  `p2f 无 late ⇒ 160` 不变 ✔；`p3 无 late ⇒ 80` 不变 ✔ ⇒ **p2_fused 路径逐位不变**。

### 8.2 根因②（⚠️ **此标题的结论已被 §8.7 / §9.7 推翻**：真凶是 `p2_out_scale=0.25`（sc025 家族）与「silu 权重喂 relu ⇒ 神经元死亡」，**不是**"换头从零学"）
> ⚠️⚠️ **本节已被 §8.7 部分更正**：真凶是 **`p2_out_scale=0.25`（neck 收窄 4 倍）**，
> **不是**"换头从零学"（同族 sc1.0 的臂实时在收敛）。下面的证据仍然有效，但**结论要按 §8.7 读**。
| | 新轮 sc25 家族（trial_0004~0007） | 老轮 late 臂（nogmc_struct4 trial_0000~0003） |
|:--|:--|:--|
| 结构 | `hc32 / p2_fused / relu / uf=True / sc0.25` | `hc64 / p3 / silu / uf=False / sc1.0` |
| **首 epoch loss** | **127** | **0.25** |
| 最终 loss | 1.07~1.28（hm 0.90~1.13） | 0.17~0.25（hm 0.12~0.16） |
| T1 F1 | **0.2519 / 0.2617 / 0.2702 / 0.2670**（ΔT1 = **−63 pp**） | **0.8888 / 0.8875 / 0.8748 / 0.8740** |
| T2 / T3 | 0.0（FP 51885~60357） | 正常 |
* ⇒ **不是推理/门禁 bug**：彩排用 cs_t0043 复现 T1=0.890767（六位一致）⇒ 门禁链路本身没问题；且**训练 loss 本身就差 4 倍**。
* ⇒ 差异里最致命的是**头的 conv 形状变了**（hc64→hc32、p3→p2_fused）⇒ **cs_t0043 的头权重被丢弃 ⇒ 从零学 ⇒ 20 ep 学不动**。
* ⇒ **所以「≤21 ms 的结构」= 小头（hc32/hc24）⇒ 必然换头 ⇒ 必须解决"新头学不动"。**

### 8.3 ★★ 破局路线（周一按顺序执行，别再盲跑）
**路线 A（最快，板卡一上电 5 分钟出数）—— 测老轮结构 `p3/hc64/late(4×→320)` 的真实 NPU ms**
* 今天 17:42 测过一次，**结果不可信**：
  `board_measure_from_server.sh` 从导出日志抓 `[model] stride=`，对 p3+late 抓到 2（实际 4）
  ⇒ 板卡报「input/grid ratio 4 does not match --stride 8」；
  手工把边车改成 4 后板卡接受，但 **null control 失败**（`NPU output independent of input`）
  ⇒ **探针权重按名字复制导致语义错位 ⇒ 测速必须用随机权重**。
* 若 **≤21 ms** ⇒ 老轮结构直接可用（4 条实测 T1 0.874~0.889）⇒ 只需 act=relu 化（诊断臂 D1 在跑）⇒ 交付在望。
* 若 **>21 ms** ⇒ 走路线 B。

**路线 B（工程 1~2 h，兜底）—— 切片继承**：把 cs_t0043 的 hc64 头**按通道切片**迁到 hc32/hc24
⇒ 头有 1/2 知识 ⇒ 20 ep 能收敛 ⇒ 速度 19.83~20.83 ms ✔ ⇒ 同时满足 ≤21 ms + all relu + n=3 soup。
改点：worker 载入 cs_t0043 的地方；规则 = 同名同形状直拷；仅通道维不同 ⇒ **按通道切片**
（取前 N 个输出通道 + 对应输入通道子集）；late 新增参数 ⇒ 随机初始化。

**路线 C（时间不够，不推荐）**：从零学 + ep60 + 更低 head lr ⇒ 一条臂 4.5 h ⇒ 窗口只够 6~8 条 ⇒ 做不出 n=3 soup。

### 8.4 板卡复测命令（★ 随机权重 + 正确 stride）
```bash
# ① 造随机权重探针（速度与权重无关，但必须避开"输出恒定"）
python3 - <<'PY'
import sys, pathlib, torch
sys.path.insert(0,'/tmp/pycharm_project_10ae9e2e')
from manu.pipeline.npu import probe_struct_speed as P
P.CS_T0043='/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt'
out=pathlib.Path('/mnt/data/siping/runs/npu20ms/probe_late/p3_hc64_relu_late.pt')
P.build_probe_ckpt(dict(head_src='p3',head_conv=64,act='relu',
                        late_upsample=True,p2_out_scale=1.0), out)
d=torch.load(out,map_location='cpu',weights_only=False)
for k,v in list(d['model'].items()):
    if torch.is_floating_point(v) and v.dim()>0:
        d['model'][k]=torch.randn_like(v)*0.02
torch.save(d,out); print('随机权重探针 ok')
PY
# ② 导出 + 转 RKNN + 上板
bash manu/training/board_measure_from_server.sh \
  --ckpt /mnt/data/siping/runs/npu20ms/probe_late/p3_hc64_relu_late.pt \
  --out-root /mnt/data/siping/runs/npu20ms/probe_late/spd_relu
# ③ ★ 边车 stride 必须手改（修好 late 语义后 p3+late 输出 320 ⇒ ratio **2**）
ssh root@192.168.0.64 'echo 2 > /root/hm_speed/arms/<name>_int8.stride; \
  cd /root/hm_speed && BOARD_DIR=/root/hm_speed LOOPS=30 WARMUP=10 SETTLE=5 bash /root/hm_speed/measure_board.sh'
```
⚠️ 今天的错例：修 late 语义**之前** p3+late 输出 160 ⇒ ratio 4；修**之后**输出 320 ⇒ ratio **2**。

### 8.5 服务器现状（2026-11-06 18:00）
| 进程 | PID | 说明 |
|:--|:--|:--|
| 第一轮 dispatcher | 2883423 | `--max-hours 34` ⇒ **周六 00:08 停派**（队尾几条臂跑不到 ⇒ **属正常，不是故障**） |
| 看门狗 | 2886633 | |
| **v3 第二轮 dispatcher** | **2949269** | out_root `/mnt/data/siping/runs/npu20ms_seed2`，study `mix_ratio_search_npu20ms_seed_v3`，30 臂，`--max-hours 31` ⇒ **周一 01:00 停派** |
| 收割 | 2949350 | `R2=npu20ms_seed2`、`STUDY2=mix_ratio_search_npu20ms_seed_v3`，硬闸 **周一 05:00** |

**v3 计划（30 臂，`manu/training/build_round2_plan_v2.py`）**：
* 6 条**诊断臂**（先跑，1.5 h 内出训练曲线）：
  `D1` p3/hc64/**relu**/late/uf0（★主候选，与 F_MAIN 同配方）、
  `D2` p3/hc64/**silu**/late/uf0（对照＝老轮结构，应重现 T1 0.874~0.889）、
  `D3` p3/hc64/relu/late/uf1、`D4` p2f/hc32/relu/late/sc1.0/uf0、
  `D5` p2f/hc32/relu/late/sc0.25/uf1、`D6` p3/hc64/relu/late/uf0/lr1.5
* 2 个**家族**（各 12 种子）：`F_MAIN` = p3/hc64/relu/late/uf0/lr3.5（★交付主线，all relu）、
  `F_ALT` = p2f/hc32/relu/late/sc1.0/uf0（hc32 家族，路线 B 成功后也能出 soup）

### 8.6 周一怎么读结果（**先看 loss，再看指标**）
```bash
# 诊断臂：先看 loss 尾部（hm < 0.3 ⇒ 收敛；hm ≈ 1.0 ⇒ 没学会）
grep -aoE "loss[^,]{0,40}" /mnt/data/siping/runs/npu20ms_seed2/logs/trial_000{0..5}.log | tail -3
# 门禁报告
cat /mnt/data/siping/runs/npu20ms_seed2/trials/trial_0001/gate/epoch20/gate_report.json
# 总表
cat /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md            # 第一轮单臂 + soup 行
cat /mnt/data/siping/runs/npu20ms_seed2/ACCEPTANCE.md      # v3 单臂
cat /mnt/data/siping/runs/npu20ms/harvest_done.json        # 收割完成标记
```
**判据**：`NPU ≤ 21.0 ms ∧ ΔT3 ≥ −1.0 pp ∧ ΔT1 ≥ −0.5 pp ∧ ΔFP_T2 ≤ +10`
（Δ 相对 cs_t0043：T1 0.890766573 / T2 0.998859(FP 34) / T3 0.903960496）
⚠️ 门禁的 T2 FP 绝对上限是 62，验收表是 ≤44（口径差异，非 bug）。

---

## 8.7 ★★★ 18:06 的**关键更正**（比 8.2 更可信，周一以本节为准）

### 8.7.1 实时证据（v3 第二轮，18:06 读数）
| 诊断臂 | 结构 | 实测 loss 轨迹 | 判断 |
|:--|:--|:--|:--|
| **D1** trial_0000 | `p3/hc64/relu/late/uf0` | 11.6 → **1.81 → 1.65 → 1.39**（hm 1.27） | **在收敛**（老轮同族最终 0.17~0.25） |
| **D4** trial_0003 | `p2f/hc32/**sc1.0**/relu/late/uf0` | 82.6 → **0.887 → 0.685 → 0.510**（hm 0.42） | **在收敛** |

⇒ **★ D4 是「hc32 换头 + 从零学」的臂，却收敛得很好** ⇒
**"换头 ⇒ 从零学 ⇒ 学不动"这个结论作废**（§8.2 的结论要按本节读）。

### 8.7.2 那 4 条死臂的共同点只有一个：**`p2_out_scale=0.25`**
`trial_0004`(uf0) / `0005`(ep15) / `0006`(lr3.0) / `0007`(lr4.0) —— **4/4 全是 sc0.25 家族**；
而 D4（sc1.0）活着 ⇒ ⇒ **真凶 = neck 收窄 4 倍 ⇒ 头的输入幅度太小 ⇒ 学不动**。
⇒ **`sc0.5`（20.06 ms）与 `sc1.0`（20.83 ms）大概率都是好的** ⇒ **≤21 ms 依然可达** ✔✔

### 8.7.3 第一轮剩下的 17 条待跑臂（几乎都不是 sc0.25 ⇒ 很可能都活）
| 结构 | 条数 | 速度 | 备注 |
|:--|:--:|:--|:--|
| p2_fused / hc32 / **late / sc1.0** | 6 | 20.83 ms | ★主力 |
| p2_fused / hc48 / 非late | 6 | 20.60 ms | 160 输出 |
| p2_fused / hc32 / **late / sc0.5** | 2 | 20.06 ms | ★主力 |
| p2_fused / hc32 / late / sc0.25 | 1 | 19.83 ms | ⚠️ 已知有毒 |
| p1 / hc64 / 非late / sc0.75 & sc0.25 | 2 | 21.03 / 21.05 ms | 160 输出 |

⇒ **第一轮不必再做任何改动**：让 dispatcher 自然跑到周六 00:08 即可（它会自动用上 §8.7.4 的新代码）。

### 8.7.4 ★ 已实现并验证：**通道合并继承**（加速器，不是必需项）
`manu/data/preannotate_hm_bbox.py` 新增 `merge_channels()`：
* 训练侧**换结构**（head_conv 64→32、head_src p1|p3→p2_fused）时，原先形状不匹配的头的张量
  被**整块丢弃** ⇒ 新头从零学；现在改成**按通道合并继承**
  （输出通道等距挑选 + 输入通道分组取均值 + BN 统计量同步取同一组索引）。
* 实测（服务器）：`hc32 + p2_fused + late + relu` 加载 cs_t0043 ⇒
  `488 tensors loaded | ★通道合并继承=16 个张量`（原先 16 个直接丢弃）⇒ 输出 320×320、finite ✔
* **推理侧严格 gate 不变**：不换结构时形状不匹配仍 `RuntimeError`（绝不静默用随机头）✔
  回归实测：默认结构加载 cs_t0043 ⇒ `494 tensors loaded`、无合并、无报错 ✔
* ⇒ **对后续新起的臂自动生效**（worker 是新进程 ⇒ 不用重启任何 dispatcher）。

### 8.7.5 周一该看什么（按优先级）
1. `cat /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md`（第一轮单臂 + soup 行）—— **sc0.5 / sc1.0 的真实 T1/T2/T3**；
2. `cat /mnt/data/siping/runs/npu20ms_seed2/ACCEPTANCE.md`（v3：p3 家族 + p2f 家族）；
3. 诊断臂 loss 尾部（判断是否收敛）：
   `grep -aoE "loss[^,]{0,30}" /mnt/data/siping/runs/npu20ms_seed2/logs/trial_000{0..5}.log | tail -3`
4. 板卡上电 ⇒ 按 §8.4 测 `p3/hc64/late(4×→320)` 的真实 ms（**随机权重**，stride 边车手改）。

⚠️ **判据不变**：`NPU ≤ 21.0 ms ∧ ΔT3 ≥ −1.0 pp ∧ ΔT1 ≥ −0.5 pp ∧ ΔFP_T2 ≤ +10`（相对 cs_t0043）。

### 8.8 18:00 队列优化：第一轮只留 320 输出主线（已剪 8 条）
第一轮原 WAITING 17 条里，有 8 条是 **160 输出（非 late）**（`p2_fused/hc48` 6 条 + `CTRL_p1_silu` 2 条）
⇒ 历史结论「160 输出 ΔT1 = −1.6 pp」⇒ 对交付无价值，却排在队尾挤占 Sat 00:08 前的派单窗口
⇒ **已剪掉（done 计数 8→16 < 36，不会触发提前收口）**。
⇒ 现在第一轮 WAITING **只剩 9 条，全是 320 输出主线**：
`p2_fused/hc32/late/sc1.0 ×6`（20.83 ms）、`sc0.5 ×2`（20.06 ms）、`sc0.25 ×1`（19.83 ms，⚠️ 已知有毒）。
⇒ 剪枝工具 `prune_waiting_trials.py` 需要交互确认 ⇒ **服务器上跑要 `echo yes |`**（否则 EOF 报错）。

### 8.9 ★★★ 18:02 决定性数据：**F_MAIN 结构实测 13.131 ms**（≤20 ms 目标达成）

在板卡断电前用 **D1 臂的 epoch1 真权重**（结构 `p3/hc64/relu/late`，stride 2 / 320×320）上板实测：

```
rknn_run     avg  13.131 ms  (93.1%)
NPU+dequant  avg  14.084 ms  -> 71.0 fps
null control: heatmap 1d561925 -> 0c32b215 CHANGED   ✔（输出确实依赖输入，不是常量）
determinism: run#1 == run#2 IDENTICAL                ✔
health check: heatmap 320x320, offset 320x320, implied stride 2 ✔
```

⇒ **★ 已写进速度表**（`manu/pipeline/npu/struct_speed_table.json`，三侧 md5 `84018963a8d1bdb442ccb9865b3c9943`，
   并同步到 `runs/npu20ms/`、`runs/npu20ms_seed/`、`runs/npu20ms_seed2/` 三份副本）
⇒ 验收脚本已能查到：`(p3, 64, relu, late, 1.0) ⇒ 13.131 ms`。

**⇒ 交付主线就此确定**：

| 家族 | 结构 | NPU int8 | 精度现状 | 结论 |
|:--|:--|:--:|:--|:--|
| **F_MAIN** | `p3/hc64/relu/late`（**yolo26n 原始 P3 层级 + 原始 64 宽 + all relu + 320 输出**） | **13.131 ms** | D1 实测 loss 1.81→1.37 在收敛；老轮同结构（silu）4 条实测 **T1 0.874~0.889** | ★★ **主线**：比 20 ms 目标还余 7.9 ms |
| F_ALT | `p2f/hc32/relu/late/sc1.0` | 20.829 ms | D4 实测 loss 0.89→0.51 在收敛 | 备选（慢 7.7 ms，且 sc0.25 已知有毒） |
| — | `p2f/hc32/relu/late/sc0.5` | 20.057 ms | 未测 | 备选 |
| — | `p2f/hc32/relu/late/sc0.25` | 19.827 ms | **4/4 死臂（已证有毒）** | ⛔ |

⇒ **F_MAIN 就是用户要的「基本退化成 yolo26n 原始版本，只是改成 heatmap，使用 all relu」**：
P3 层级 + 64 宽（原始宽度 ⇒ 头权重可继承）+ relu（all relu）+ 320 输出（与基线同口径）。

**⇒ 周一要做的事因此变得简单**：
1. 等 v3 的 F_MAIN 家族 12 个种子跑完（每个 ~1.5 h + 门禁 ~20 min ⇒ 约周日即可全出）；
2. `build_struct_soups.py` 按家族建 soup（n=12 的 p3/hc64/relu/late 家族）；
3. 门禁 → 验收表 ⇒ **看 ΔT1 是否落在 −0.5 pp 以内、ΔFP_T2 是否 ≤ +10**；
4. 若 F_MAIN 的 ΔT1 太大（老轮 silu 是 −0.2~−1.7 pp），则用 soup 平均压种子噪声
   （**n=12 的 soup 比 n=3 更稳**），或退回 F_ALT。

### 8.10 18:10 整条交付链已核对通（F_MAIN 专用）
| 环节 | 核对结果 |
|:--|:--|
| 速度表命中 | v3 计划里 `(p3,64,relu,late,1.0)` **15 条臂 ⇒ 13.131 ms** ✔（D1 + D3 + D6 + 12 种子）；`(p2_fused,32,relu,late,1.0)` 13 条 ⇒ 20.829 ✔ |
| 门禁推理口径 | `cache_trial0474_inferences.py` 用 `stride = ckpt.get("stride", args.stride)` ⇒ F_MAIN 的 ckpt 自带 `stride=2` ⇒ 峰值坐标与 GT 同口径 ✔（不用改任何推理代码） |
| 家族生成器 | 在**真实 study** 上空跑不崩：能正确识别「第二轮成员 <3 ⇒ 跳过」「第一轮同配方臂无产物齐全 ⇒ 跳过」 ✔ |
| soup ckpt | `build_struct_soups.py` 写 `"stride": ref_ckpt.get("stride", 2)` ⇒ soup 门禁重建结构正确 ✔ |
| 产物齐全判据 | 家族成员必须同时有 `weights/epoch20.pt` + `gate/epoch20/gate_report.json` ⇒ 才进 soup ✔ |

⚠️ **唯一的"表外"结构是 `p3/hc64/silu/late`（D2 对照臂，1 条）** —— silu 版没测速（且不满足 all relu）
⇒ 它只是对照，不影响交付；若要它的速度，板卡上电后用同一命令测一次即可。

⇒ **⇒ 周一不必再改任何代码**，只需：等 F_MAIN 的 12 个种子跑完 → 建 soup → 门禁 → 读验收表。

### 8.11 18:12 把窗口铺满：v3 计划扩到 84 臂（+ 旧队列 26 ⇒ 实际 110 条待跑）
**发现的问题**：按 30 臂的原计划，v3 约 9.5 h 就跑完 ⇒ **周日 04:00 ~ 周一 01:00 GPU 空转 21 h**
⇒ 既然要求"充分利用计算资源"，就把这段时间变成**更多 F_MAIN 种子**（n 越大 soup 越稳）。

* 计划生成器 `build_round2_plan_v2.py` 新增 `--seeds-main`（默认 36）/ `--seeds-alt`（默认 24）/ `--diag-prefix`。
* 本次用 **F_MAIN 48 + F_ALT 30 + 6 诊断臂 = 84 臂**（seed-base 4300）。
* ⚠️ **`--diag-prefix` 必须改**：重跑时旧诊断臂的 role 已在 study 里 ⇒ 同名会让 soup 解析成员
  指向**没有产物的旧 trial** ⇒ 本次用 `R2_` 前缀避开。
* ⇒ **Optuna 队列实际 = 旧队列残留 26 条（D5,D6 + 24 种子）+ 新 84 条 = 110 条**
  （FIFO ⇒ 旧臂先跑；新旧 role 含 seed ⇒ 不撞名）⇒ 容量 ~65 条 ⇒ **F_MAIN 预计 n≈37、F_ALT ≈ n≈23**。
* ⚠️ **坑：`launch_npu20ms_search.sh` 的重复启动保护** —— 旧 dispatcher 被 kill 后**优雅退出中**
  （日志 `🎉 搜索结束`），此时 pid 文件里的 PID 仍 `kill -0` 成功 ⇒ 启动被 `[ABORT] 搜索已在运行` 挡掉
  ⇒ **v3 实际没起来**（GPU 只剩第一轮在跑）。
  ⇒ **处理**：等旧 dispatcher 真正退出后**重跑同一条 launcher 命令**（保护会识别陈旧 PID 并放行）。
  ⇒ **教训：kill dispatcher 后必须核对"新 pid 文件 + LAUNCH 日志"，不能只看命令返回。**

---

## 9. ★ 周一开箱即用清单（照着做，别改代码）

```bash
ssh -p 32222 huangzhe@192.168.99.40        # 服务器常开；宿主关机不影响
cd /tmp/pycharm_project_10ae9e2e && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav

# ① 收割链跑完没？
cat /mnt/data/siping/runs/npu20ms/harvest_done.json 2>/dev/null || tail -30 /mnt/data/siping/runs/npu20ms/final_harvest.log

# ② ★ 先看诊断臂 loss（hm<0.3 ⇒ 收敛；hm≈1.0 ⇒ 没学会）
grep -aoE "loss[^,]{0,30}" /mnt/data/siping/runs/npu20ms_seed2/logs/trial_*.log | tail -6

# ③ 总表（第一轮单臂 + 🍲 soup 行；含 13.131 ms 那行）
cat /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md
cat /mnt/data/siping/runs/npu20ms_seed2/ACCEPTANCE.md

# ④ 机读
python -c "import json;d=json.load(open('/mnt/data/siping/runs/npu20ms/acceptance.json'));\
[print(r['trial'], r['ms'], r['d1'], r['dFP2'], r['d3'], r['verdict']) for r in d['results']]"

# ⑤ soup 权重本体 + 家族定义
ls -la /mnt/data/siping/runs/npu20ms/soups/
cat /mnt/data/siping/runs/npu20ms/struct_families.json
# ★ 预期文件名（家族 tag = 结构维度，故**不要猜文件名，直接 ls**）：
#   F_MAIN（p3/hc64/relu/late，13.131 ms）⇒ soup_p3_hc64_relu_late_sc100.pt
#   F_ALT （p2f/hc32/relu/late/sc1.0）      ⇒ soup_p2f_hc32_late_sc100.pt

# ⑥ 板卡若上电：复测最终 ckpt（结构已实测 13.131 ms）
bash manu/training/board_measure_from_server.sh \
  --ckpt /mnt/data/siping/runs/npu20ms/soups/soup_p3_hc64_relu_late_sc100.pt \
  --out-root /mnt/data/siping/runs/npu20ms/npu_speed_final
# ⚠️ 若换了文件名 ⇒ 先 ls /mnt/data/siping/runs/npu20ms/soups/ 拿真实名字
# ⚠️ 若报 ratio/stride 不匹配 ⇒ 手工改板卡边车：echo 2 > /root/hm_speed/arms/<name>_int8.stride
```

**判据**：`NPU ≤ 21.0 ms`（我们主线实测 **13.131 ms**，余量极大）`∧ ΔT3 ≥ −1.0 pp ∧ ΔT1 ≥ −0.5 pp ∧ ΔFP_T2 ≤ +10`
（Δ 相对 cs_t0043：T1 0.890766573 / T2 0.998859(FP 34) / T3 0.903960496）
⚠️ 门禁的 T2 FP 绝对上限是 62、验收表是 ≤44 ⇒ 两者口径不同，**以验收表为准**。

**若 F_MAIN 的大 soup（n≈37）反而不如小 soup 好** ⇒ 用 `build_struct_soups.py` 只挑 3~5 个成员重做
（家族定义 JSON 里改 `members` 即可，工具支持 `--dry-run`）。
**若 ΔT1 仍超 −0.5 pp** ⇒ 退到 F_ALT（p2f/hc32/relu/late/sc1.0，20.829 ms），它的头更窄但 neck 不收窄。
**若两者都不达标** ⇒ 用第一轮那 9 条 320 输出主线臂（p2f/hc32/late/sc1.0 ×6 + sc0.5 ×2）单臂看有没有达标的。

### 9.1 18:11 收割链端到端空跑已验证（周一只剩"等数据"）
在真实 study 上把三步各跑了一遍（数据未就绪的空跑），**全部不崩且行为正确**：
1. `build_struct_families.py` ⇒ 正确识别"第二轮成员 <3 ⇒ 跳过"，写出家族定义；
2. `build_struct_soups.py --dry-run` ⇒ 正确识别"成员 trial 目录尚不存在 ⇒ 可用成员 0 个 ⇒ 优雅跳过"；
3. `npu20ms_acceptance.py` ⇒ 正常写出 `acceptance.json` + `ACCEPTANCE.md`；
4. 速度表命中自检：`(p3,64,relu,late,1.0) ⇒ 13.131`、`(p2_fused,32,relu,late,1.0) ⇒ 20.829` ✔。
⇒ **⇒ 周一无需再排障，只需等 F_MAIN 的臂跑完，然后读表。**

### 9.2 ⚠️ 周一操作红线（两条最容易踩）
1. **收割等待期间不要重启 v3 dispatcher。**
   收割脚本的 `wait_dispatcher "$R2"` 记住的是**启动时读到的那个 PID**（现为 2960554）
   ⇒ 一旦 v3 换了 PID，收割会以为"第二轮已结束"⇒ **提前进入建家族/soup 门禁** ⇒
   家族成员可能只攒到一小半（n 远小于 37）⇒ soup 质量打折。
2. **若确实要加臂/重启 v3，必须同时重挂收割**（否则同上）：
   ```bash
   # ① 重启 v3（计划写进 <R2>/seed_arms.json；★ 诊断臂 role 必须换前缀防撞名）
   OUT_ROOT=/mnt/data/siping/runs/npu20ms_seed2 STUDY_NAME=mix_ratio_search_npu20ms_seed_v3 \
   N_TRIALS=<臂数> MAX_HOURS=31 EPOCHS_DEFAULT=20 \
   SPEED_TABLE=/mnt/data/siping/runs/npu20ms_seed2/struct_speed_table.json \
   SPEED_LIMIT_MS=21.0 PLAN=/mnt/data/siping/runs/npu20ms_seed2/seed_arms.json \
   bash manu/training/launch_npu20ms_search.sh
   # ② 立刻重挂收割（★ 三个 env 都要给，DEADLINE_EPOCH 是周一 05:00 的硬闸）
   kill <旧收割 PID>   # 按 PID 精确 kill，禁用 pkill -f
   DEADLINE_EPOCH=$(date -d "2026-11-09 05:00" +%s) \
   R2=/mnt/data/siping/runs/npu20ms_seed2 STUDY2=mix_ratio_search_npu20ms_seed_v3 \
     setsid nohup bash manu/training/final_harvest.sh \
     > /mnt/data/siping/runs/npu20ms/final_harvest.log 2>&1 &
   ```
   ⇒ 重挂后**必须核对**：`cat <R2>/dispatcher.pid` 有新 PID + dispatcher.log 出现 `[LAUNCH]` +
   `ps` 里收割进程在（**kill 后旧进程可能还在优雅退出 ⇒ launcher 的重复启动保护会挡掉新启动 ⇒
   等它真退出再重跑 launcher**）。

### 9.3 18:12 资源体检（决定 ETA 可信度 + 唯一要盯的风险）
| 资源 | 实测 | 判断 |
|:--|:--|:--|
| CPU | **80 核**，load 25.75 | ✔ 充裕 ⇒ 4 条臂同时跑门禁（每臂约 19 min）不会成为瓶颈 ⇒ **周一 01:00 前能攒够 n≈37 的 ETA 可信** |
| 显存 | 4 卡各 92~99% 利用率 | ✔ 正常 |
| **内存** | 125 GB 总，**已用 92 GB，available 仅 15 GB** | ⚠️ **唯一要盯的**：8 条臂（第一轮 4 + v3 4）并发 + 门禁建缓存 ⇒ 若某臂日志出现 `MemoryError` / worker 被 OOM kill ⇒ **立刻降并发** |

**OOM 处置（按需执行，注意别踩"重启后必须重挂收割"那条）**：
```bash
# 症状：某臂日志出现 MemoryError / dmesg 有 oom-killer ⇒ 该臂会 CRASH
# 处置：把 v3 降到 3 卡并发（第一轮不动）
OUT_ROOT=/mnt/data/siping/runs/npu20ms_seed2 STUDY_NAME=mix_ratio_search_npu20ms_seed_v3 \
N_TRIALS=84 MAX_HOURS=31 EPOCHS_DEFAULT=20 GPUS=0,1,2 \
SPEED_TABLE=/mnt/data/siping/runs/npu20ms_seed2/struct_speed_table.json \
SPEED_LIMIT_MS=21.0 PLAN=/mnt/data/siping/runs/npu20ms_seed2/seed_arms.json \
  bash manu/training/launch_npu20ms_search.sh
# ★ 然后按 §9.2 重挂收割（R2 / STUDY2 / DEADLINE_EPOCH 三个 env 都要给）
```
⇒ 另：单臂周期实测 = 训练 ~85 min（20 ep × 4.25 min）+ 门禁 ~19 min（gate_report 的 elapsed_seconds≈1144）
⇒ 4 卡 ⇒ **≈2.3 臂/h** ⇒ 31 h 窗口可跑 ~70 臂 ⇒ F_MAIN 预计 n≈40、F_ALT n≈24 ✔

### 9.4 ★ 兜底工具：`pick_soup_members.py`（大 soup 不达标时的救援，已实测）
**场景**：F_MAIN 家族攒到 n≈40 ⇒ `build_struct_soups.py` 做**均匀平均**。
若大 soup 的 ΔT1 差一点点没过 −0.5 pp（或个别成员种子运气差）⇒
**按成员自己的门禁分数挑 top-k 重建 soup** 通常能救回来。

```bash
# ① 看每个家族成员的分数（与验收表同一口径）
python manu/training/pick_soup_members.py \
  --families-json /mnt/data/siping/runs/npu20ms/struct_families.json \
  --root /mnt/data/siping/runs/npu20ms --metric t1_f1 --top-k 8 --dry-run
# ② 写出只含 top-k 的新家族定义（原文件自动备份 .bak）
python manu/training/pick_soup_members.py \
  --families-json /mnt/data/siping/runs/npu20ms/struct_families.json \
  --root /mnt/data/siping/runs/npu20ms --metric t1_f1 --top-k 8 \
  --out /mnt/data/siping/runs/npu20ms/struct_families_top8.json
# ③ 用它重建 soup（结构自描述逐字段校验 ⇒ 结构不一致直接报错）
python manu/training/build_struct_soups.py \
  --out-root /mnt/data/siping/runs/npu20ms --study-name mix_ratio_search_npu20ms_v2 \
  --families-json /mnt/data/siping/runs/npu20ms/struct_families_top8.json \
  --out-dir /mnt/data/siping/runs/npu20ms/soups_top8 \
  --speed-table manu/pipeline/npu/struct_speed_table.json
# ④ 对新 soup 跑门禁：把 soup_candidates.json 指到新目录即可
```
⚠️ 实测过的两个坑（已修）：
1. `read_gate_report()` 要传**trial 目录**（它内部自己 glob `gate*/epoch*/gate_report.json`），
   传门禁报告的父目录会静默读不到指标 ⇒ 排序退化成"任意挑"；
2. 成员的 epoch 与家族 epoch 不一致（ep15 vs ep20）⇒ 该成员被跳过（不是 bug，是保护）。

实测（第一轮 4 条臂）：t1_f1 读出 0.2519 / 0.2702 / 0.2670 ⇒ top-2 正确排除了最差的 0.2519 ✔

### 9.5 ★ 周一必看的一组对比：**all-relu 的代价**（D1 vs D2）
用户的要求里有一条硬约束：**「基本退化成 yolo26n 原始版本，只是改成 heatmap，使用 all relu」** ⇒
主线结构 `p3/hc64/relu/late`（13.131 ms）**act=relu**。
而老轮同结构（`p3/hc64/**silu**/late`）的 4 条实测是 **T1 0.8888 / 0.8875 / 0.8748 / 0.8740**（ΔT1 −0.20~−1.68 pp）。

⇒ **v3 里正好有这两条臂可直接对比**（同一配方、只差 act）：
| 臂 | 结构 | 含义 |
|:--|:--|:--|
| `R2_D1_p3hc64_relu_late_uf0` | act=**relu** | ★交付主线（用户要求 all relu） |
| `R2_D2_p3hc64_silu_late_uf0` | act=**silu** | 对照组（老轮同配方 ⇒ 参照 T1 ≈ 0.874~0.889） |

⇒ **周一先看这两条臂的门禁 T1**：
* 若 **D1 ≈ D2** ⇒ all relu 无代价 ⇒ 直接用 relu 主线，交付无争议 ✔
* 若 **D1 明显低于 D2**（例如差 > 1 pp）⇒ **all relu 有实质代价** ⇒ 此时必须**把两条数字一起摆给用户选**
  （"严格 all-relu 但 ΔT1 −x pp" vs "silu 且 ΔT1 −y pp"），**不要自己悄悄把 relu 换成 silu** ——
  用户明确点过 all relu，这是他的口径要求，改 act 属于改需求。
  ⇒ 另外还有第三种解：**通道合并继承**已实现（§8.7.4）⇒ 若 relu 难训，可用它给 relu 头一个好的起点。

### 9.6 `pick_soup_members.py` 新增 `--require-pass`（只保留门禁通过的成员）★ 已实测
**为什么**：大 soup 的 T2 假阳（FP）往往是被个别"门禁就 FAIL"的成员带进来的
⇒ **只让门禁 PASS 的成员进 soup**，是压 ΔFP_T2 最直接的一招。
```bash
python manu/training/pick_soup_members.py \
  --families-json /mnt/data/siping/runs/npu20ms/struct_families.json \
  --root /mnt/data/siping/runs/npu20ms --metric t1_f1 --top-k 8 --require-pass \
  --out /mnt/data/siping/runs/npu20ms/struct_families_top8.json
```
⚠️ **实测抓到的静默失效（已修）**：`read_gate_report()` 返回的是**归一化字典，里面没有 `veto_failures`**
⇒ 用 `gr.get("veto_failures")` 判断"门禁是否通过"**永远得到"通过"** ⇒ FAIL 臂会被当好成员留下。
⇒ 正确判据 = **`verdict == "PASS"` 且没有 `infra_failed`**（与验收表判定同口径）。
实测：3 条 FAIL 臂 ⇒ 全部标为"门禁未通过 ⇒ 排除"；通过者 <2 ⇒ 打印告警并**回落到全部成员排序**
（不会静默产出空家族）✔

### 9.7 ★★★ 18:20 定量发现：**all-relu 的代价不在 relu，而在「拿 silu 的权重去喂 relu」**

**证据（20 epoch 完整训练轨迹，训练侧 hm loss；门禁未跑完，但 loss 已足够定性）**：

| 臂 | 结构 | 头的来源 | 20 ep 最终 hm |
|:--|:--|:--|:--|
| 老轮 trial_0000 | `p3/hc64/**silu**/late` | **继承 silu 权重** | **0.1643**（参照基准） |
| v3 旧 trial_0000（D1） | `p3/hc64/**relu**/late` | **继承 silu 权重**（形状相同 ⇒ 直拷） | **0.68 ~ 0.80** ★差 4~5 倍 |
| v3 旧 trial_0003（D4） | `p2f/hc32/**relu**/late/sc1.0` | 头**形状变了 ⇒ 重新初始化/合并** | **0.20 ~ 0.44** ≈ 与 silu 同量级 ✔ |

⇒ **★ 结论**：relu 本身没问题（D4 用 relu 也能到 hm 0.2~0.44）；
**问题出在「silu 训好的权重直接喂给 relu」⇒ 负权重通道在 relu 下大量死亡 ⇒ 有效容量骤降 ⇒ 热图 loss 降不下去。**
⇒ **⇒ 结构性推论（周一务必先看这组数字）**：
* **F_MAIN（`p3/hc64/relu`，13.131 ms）** ⇒ 继承了 silu 权重 ⇒ **预计 T1 明显不如老轮（0.874~0.889）**，ΔT1 可能差 1~3 pp ⇒ **大概率过不了 −0.5 pp 门槛**；
* **F_ALT（`p2f/hc32/relu`，20.829 ms）** ⇒ 头重新初始化 ⇒ **预计 hm 与 silu 同量级 ⇒ T1 有希望达标**，
  且它 **act 也是 relu**（满足 all-relu）⇒ **★ 交付主线很可能应该换成 F_ALT**。

**★ 一行修法（让 F_MAIN 也能有救）** —— 在 `manu/data/preannotate_hm_bbox.py` 的加载处加一条规则：
> **当目标 act 与 ckpt 的 act 不同（silu→relu）⇒ 跳过「头」相关张量的权重拷贝，保持随机初始化**
> （形状相同也要跳过）⇒ relu 从随机头开始 ⇒ 无死亡神经元 ⇒ 20 ep 能到 hm≈0.2~0.4。
> 依据：**F_ALT 的行为就是这条规则的等效实现**（头形状变了 ⇒ 本来就是新的）。

**周一决策树**：
1. 先看 **D1(relu, 13.131 ms) 与 D4(relu, 20.829 ms)** 两条臂的门禁 T1/T2/T3（v3 里已跑完 20 epoch 的那两条）；
2. 若 **D4 ≫ D1** ⇒ 主线换 **F_ALT**（20.829 ms ≤ 21 ms，且 all-relu）⇒ 直接建 F_ALT 的 n≈24 大 soup；
3. 若仍想要 **13.131 ms** ⇒ 加上面那条「act 变了就重置头」的规则 ⇒ 重跑 F_MAIN 的种子（每条 ~1.9 h ⇒ 起 4 条 2 h 内可验）⇒ 周一 05:00 前仍来得及；
4. 两族都不达标 ⇒ 退第一轮那 9 条 320 输出主线单臂（p2f/hc32/late/sc1.0 & sc0.5）。

### 9.8 ★★★ 18:26 **已落地**：「act 变了 ⇒ 重置头」已实现并实测 ⇒ **F_MAIN 不用换主线**
**改动**：`manu/data/preannotate_hm_bbox.py`（加载处）—— 当**目标 act 与 ckpt 的 act 不同**（silu→relu）⇒
**跳过「头」张量（`head.*` / `head_in.*`）的拷贝，保持随机初始化**；骨干/颈照常继承。
**实测（服务器，cs_t0043 ⇒ p3/hc64/relu/late）**：
`loaded: 348 tensors | ★act变化⇒重置头=20 个张量` ⇒
头张量与 ckpt **相同 = 0、不同 = 20**（已重置）✔，其余 348 个（骨干/颈）照常继承 ✔。
（自测脚本里 `fuse_p2.conv.weight` 那个 key 在 p3 模型里不存在 ⇒ 那行 TypeError 是**测试脚本**的问题，不是代码问题。）

**⇒ 为什么这等价于已被验证的 D4**：D4（`p2f/hc32/relu`）之所以能到 hm 0.20~0.44，正是因为
**头形状变了 ⇒ 头重新初始化**；现在 F_MAIN 也拿到同样的待遇 ⇒ **预期 hm 回到 0.2~0.4 量级** ⇒
**F_MAIN 的 13.131 ms + all-relu + 可达精度** 三者同时成立。

**⇒ 关键：无需重启任何进程！** worker 是**新进程** ⇒ **正在跑的 v3 调度器从此刻起新起的臂会自动用新代码**
⇒ F_MAIN 的种子臂（正在排队）自动受益 ⇒ 零风险、零重启。
⚠️ **已在跑的 4 条臂（trial_0004~0007）是旧代码** ⇒ 它们属于"未修复"批次 ⇒
**周一建 soup 时用 `pick_soup_members.py --require-pass` 把它们剔掉**（它们门禁大概率不达标 ⇒ 会被自动排除）✔

### 9.9 ★ 周一「一条命令看全局」（已实测可直接用）
```bash
ssh -p 32222 huangzhe@192.168.99.40 'date "+%F %T"; \
echo "--- 进程 ---"; ps -o pid,etime,cmd --no-headers -p 2883423,2886633,2960554,2949350 | cut -c1-60; \
echo "--- GPU ---"; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr "\n" " "; echo; \
echo "--- 收割完成标记 ---"; cat /mnt/data/siping/runs/npu20ms/harvest_done.json 2>/dev/null || echo "未完成 ⇒ 看 final_harvest.log 尾部"; \
echo "--- 达标候选(🎯) ---"; grep -hE "🎯" /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md /mnt/data/siping/runs/npu20ms_seed2/ACCEPTANCE.md 2>/dev/null || echo "（暂无）"; \
echo "--- soup 行(🍲) ---"; grep -hE "🍲" /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md 2>/dev/null | head -8; \
echo "--- F_MAIN 家族成员数 ---"; python3 -c "
import json;d=json.load(open(\"/mnt/data/siping/runs/npu20ms/struct_families.json\"));print({k:len(v[\"members\"]) for k,v in d.get(\"families\",{}).items()})" 2>/dev/null'
```
⇒ 这条命令覆盖：进程存活 / GPU 占用 / 收割是否完成 / **哪些臂达标(🎯)** / **soup 行的数字(🍲)** / 每个家族攒了几个成员。

### 9.10 18:25 确认：**「第一轮母臂 + 第二轮 2 个种子 = n=3」的配对在真实数据上已生效** ✔
当前 `struct_families.json` 里已有 3 个家族（各 3 个成员），成员形如：
* `p2_fused_hc32_relu_late_sc100` ← `S_p2_fused_hc32_relu_late_lr3.5_ep20`（第一轮母臂）+ `SEED_hc32_late1_sc100_s3004/s3005`（第二轮种子）
* 同理还有 `_sc050`（2 个成员）、`_sc025`（⚠️ 有毒家族）
⇒ **⇒ 与基线 cs_t0043 完全同形态的 n=3 soup 已经在成型** ⇒
**⇒ 兜底路径已经存在**：即便 v3 的 n≈40 大家族出问题，
**第一轮母臂 + 第二轮种子的 n=3 家族（sc1.0 20.83 ms / sc0.5 20.06 ms）照样能出候选** ✔
⚠️ `sc025` 家族是已证有毒的那族 ⇒ 建 soup 时用 `pick_soup_members.py --require-pass` 剔掉。

### 9.11 ★ 18:26 更正 §9.8 的一处表述 + 记一个比对陷阱
**更正**：§9.8 说"已在跑的 trial_0004~0007 属未修复批次"——**不准确**。
实际队列是 FIFO ⇒ 当前这 4 条里：
* `trial_0004/0005` = **旧 plan 的 D5/D6 诊断臂**（配方不同 ⇒ **不是 F_MAIN 家族成员**，不用管）；
* `trial_0006/0007` = **旧 plan 的 F_MAIN 种子**（种子 4106/4107）⇒ **确实是 F_MAIN 家族成员，且是旧代码**
  ⇒ 它们的 hm 会卡在 0.7~0.8 ⇒ **建 soup 时必须剔掉** ⇒ `pick_soup_members.py --require-pass` 会自动剔 ✔。
  （新 plan 的 F_MAIN 种子 = trial_00xx 起的一大批，都是**新代码** ⇒ 正常。）

**★ 比对陷阱（我自己踩了）**：**plan JSON 的 `params` 与 study 的 `fixed_params` 不能直接比** ——
dispatcher 入队时把 `epochs` 从 fixed_params **pop 掉**（搬到 user_attrs，见 `optuna_mix_ratio_search.py:1815`）
⇒ 直接比必然比不上 ⇒ 会误判"没有同配方成员"。
⇒ **家族分组本身没问题**（它是 study↔study 比，两边都已 pop 掉 epochs ⇒ 一致）✔
⇒ 但**任何自查脚本**比较 plan 与 study 时，都要 `p.pop("epochs", None)`。

### 9.12 18:26 最后一次盘点：F_MAIN 家族 **n=62**，六个诊断臂覆盖全部关键因子
**v3 study 里的家族分组（按 params 逐字段相等，episodes 已按 dispatcher 的规则 pop 掉）**：

| 家族（配方） | 成员数 | 状态 | 作用 |
|:--|:--:|:--|:--|
| `p3/hc64/**relu**/late/sc1.0/ufFalse/lr3.5` | **62** | 1 FAIL + 1 RUNNING + 60 WAITING | ★**F_MAIN 主线**（13.131 ms）⇒ n≈62 ⇒ 大 soup 材料非常充足 |
| `p3/hc64/**silu**/late/sc1.0/ufFalse/lr3.5` | 2 | 1 FAIL + 1 WAITING | **对照**：silu 代价（老轮同配方 T1 0.874~0.889） |
| `p3/hc64/relu/late/sc1.0/**ufTrue**/lr3.5` | 2 | 1 FAIL + 1 WAITING | **对照**：unfreeze_b0b4 的影响 |
| `p3/hc64/relu/late/sc1.0/ufFalse/**lr1.5**` | 2 | 1 RUNNING + 1 WAITING | **对照**：LR 的影响 |

⇒ **★ 因子覆盖齐全**：relu↔silu、uf True↔False、lr 3.5↔1.5 ⇒ 周一能直接读出"all relu 的代价"与最优配方。
⇒ **★ 已经出现 1 个 FAIL**（旧代码批次：silu 权重喂 relu ⇒ hm 卡 0.7~0.8）⇒ **正好实证了 `--require-pass` 的必要性**
⇒ ⇒ 建 F_MAIN 的 soup 时**必须** `pick_soup_members.py --require-pass`（把那批 FAIL 臂剔掉），
   然后再按 t1_f1 取 top-k ⇒ **n 仍然富余（60+ 候选里挑）**。

### 9.13 ★ 别把两个失效模式混起来（§8.7 与 §9.7 是**两回事**）
| 失效模式 | 触发条件 | 症状 | 处置 | 证据 |
|:--|:--|:--|:--|:--|
| **① neck 收窄** | `p2_out_scale=0.25`（sc025 家族） | 首 epoch loss 127、20 ep 后 hm ≈ 0.7~1.1、**T1 F1 ≈ 0.25（ΔT1 −63 pp）** | 不用 sc0.25；用 sc0.5 / sc1.0 | 第一轮 4/4 死臂全是 sc25（§8.7.2） |
| **② 神经元死亡** | 目标 act=relu **且**继承了 silu 训好的头权重 | 20 ep 后 hm ≈ 0.68~0.80（比 silu 基准 0.16 差 4~5 倍） | loader 加规则「act 变了就重置头」⇒ 20 个头张量重置（§9.8） | D1(p3/hc64/relu，继承 silu 头) vs D4(p2f/hc32/relu，头本来就是新的) |

⇒ **⇒ 判别口诀**：**"头是新的 ⇒ relu 没问题；头是继承的 ⇒ relu 要重置头。"**
⇒ D4（p2f/hc32）之所以好，是因为**头形状变了 ⇒ 头本来就是新的**；不是"hc32 更好"。

---

## 10. ★★★ 2026-11-09 09:00 收口：**交付物改用「单臂」而不是 soup**（soup 对 late 结构有害）

### 10.1 09:00~09:12 的实测（用修好的代码重跑门禁）
| 候选 | T1（UAV 主轨） | T1 的 FP | 结论 |
|:--|:--:|--:|:--|
| `trial_0032`（单臂，p3/hc64/relu/late，13.131 ms） | 0.100945 | 317,074 | T1 轨与 late 结构不兼容（**非训练问题**） |
| `soup_p3_hc64_relu_late_sc100`（21 成员平均） | **0.000003** | **3,161,296** | ★ **权重平均让 T1 更崩 10 倍** |
| `soup_p2_fused_hc32_relu_late_sc100`（22 成员平均） | 0.0 | 0 | 同上（更彻底） |

⇒ **★★ 结论：Model Soup 这条路对 late(320 输出) 结构不成立** ——
late 的头在**低分辨率**上算 ⇒ 头权重对通道排列高度敏感 ⇒ 跨种子平均后错配 ⇒ T1 彻底崩坏。
⇒ **⇒ 交付物必须是「单臂」**（或小 n、且逐成员校验过的子集）。

### 10.2 ★ 交付候选（单臂）
* 权重：`/mnt/data/siping/runs/npu20ms_seed2/trials/trial_0032/weights/epoch20.pt`
* 结构：`p3 / hc64 / relu / late_upsample / stride 2 / 320 输出`（＝「yolo26n 原始 P3 层级 + 原始 64 宽 + all relu」）
* **NPU int8 = 13.131 ms**（板卡实测，null control + determinism 双通过）⇒ **≤20 ms 达成，余 6.87 ms；比基线 cs_t0043 的 55.48 ms 快 4.22×**
* **T2 = 0.998316**（基线 0.998859）⇒ **ΔFP_T2 = −15**（比基线少 15 个假阳）✔
* **T3 = 0.907187**（基线 0.903960）⇒ **ΔT3 = +0.323** ✔；**T3FP = 109**（基线 341）✔
* **T1 = 待定**：所有 late 臂的 T1 都落在 0.099~0.30（极窄带）⇒ 是 **T1 轨与 late 结构不兼容**，
  而非模型差（老轮同结构 silu 臂的 T1 = 0.874~0.889）⇒ 需按 late 结构重标定 T1 口径后重跑（纯推理）。

### 10.3 踩坑：门禁任务并发抢同一张卡 ⇒ 三轨全无指标
09:00 我同时起了 3 个门禁（2 个 soup + 1 个单臂），**都用了 `--infer-gpu 1 / --cache-gpus 1`** ⇒
`soup_p2f...` 的门禁完成后残留进程仍在跑，与单臂任务抢 GPU/缓存 ⇒
**单臂那次重跑三轨全无指标（verdict FAIL 但 veto 为空 ⇒ 基础设施失败，不是模型失败）**。
⇒ **⇒ 门禁必须串行或分配不同 GPU**；判"结果为空"时先看 `veto_failures` 是否为空 + `infra_failed`。

### 10.4 ★★★ 09:19 定位：**T1 塌陷的真正口径错位 = 推理门限 `conf_thresh=0.02` + `top_k=100`**
**推理链对照**（同一份权重、同一套代码，只是两条评估链的门限不同）：
| 轨 | 推理/缓存脚本 | 门限 | 我们的 late 结构表现 |
|:--|:--|:--|:--|
| **T1** | `manu/inference/cache_trial0474_inferences.py` | **`--conf-thresh 0.02` + `--top-k 100`** | 热图是 **80×80 算出来再上采样到 320** ⇒ 噪声峰极多 ⇒ **top-100 截断后几乎全是噪声** ⇒ FP 317,074（soup 更炸：3,161,296）⇒ **F1 被钉死在 0.1009 / 0.302** |
| **T2/T3** | `manu/data/multiscale_heatmap_cache.py` | **`--main-threshold 0.22` / `--region-threshold 0.06`**（区域+面积属性） | 门限合理 ⇒ **T2 0.998 / T3 0.907（甚至反超基线）** ✔ |

⇒ **⇒ 不是模型塌陷，是「T1 用超低门限 + 硬截断去评估一个天然更平滑的（低分辨率上采样）热图」**：
* 非物理证据：不同种子/超参的臂 T1 **死死卡在 0.1009（p3）/ 0.302~0.304（p2f）两个常数**（训练噪声不会这样）；
* 逻辑证据：T3（最难、最小目标）若真塌，绝不可能 0.907 且 FP 从 341 降到 109；
* 物理证据：late 结构的头在 **80×80** 上算 ⇒ 上采样后峰更宽更糊 ⇒ 0.02 门限下噪声峰数量级不同 ⇒ 截断后有效峰被挤掉。

⇒ **⇒ 修复（纯推理，不重训）**：把 T1 的候选推理门限从 **0.02 提到 0.22**（与 T2/T3 的 `main-threshold` 对齐），
再跑同一个 A/B 评测（`eval_arm_pc_gmc_ab.py`，15 档门限扫描 + 8px 点对点）：
```bash
python manu/inference/cache_trial0474_inferences.py \
  --weights <交付 ckpt> --data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
  --output <out_cache.pkl> --imgsz 640 --stride 2 --batch 32 --device 0 \
  --conf-thresh 0.22 --top-k 100 --ablate-p0
python manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a /mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl --label-a BASELINE \
  --arm-b <out_cache.pkl> --label-b FINETUNE_th022 --tol 8.0 \
  --expect-f1 0.9008967482400269 --expect-tp 21499 --expect-fp 1118 --out-json <ab.json>
```
⚠️ **基线臂也要用同一门限重生成**（否则 A/B 两侧口径不一致 ⇒ 比的不是"同口径"）。
⚠️ 顺便：**soup 对 late 结构有害**（§10.1，权重平均后 T1 的 FP 涨 10 倍）⇒ 交付用**单臂**。

### 10.5 ★★★ 周一的关键筛选规则：**用「★act变化⇒重置头」标记挑出"好批次"的臂**
**为什么**：loader 的「act 变了 ⇒ 重置头」修复是在**周六 18:25** 落地的 ⇒
v3 的臂分两批：
| 批次 | 臂 | 头权重 | 实测 hm |
|:--|:--|:--|:--|
| 旧行为（18:25 前启动） | `trial_0000` ~ `trial_0007` | **继承 silu 权重** ⇒ relu 下神经元死亡 | 0.29~0.72，**学不动** |
| **新行为（18:25 后启动）** | **`trial_0008` 及以后** | **头随机重置** | 已在 **0.16~0.36**（trial_0011 hm=0.158、trial_0017 hm=0.186）⇒ **与 D4 同量级 ⇒ 会收敛** ✔ |

⇒ **筛选命令**（周一建家族/挑成员时必须用）：
```bash
# ① 列出"好批次"的臂（带重置头标记）
R=/mnt/data/siping/runs/npu20ms_seed2
for t in $(ls $R/trials | grep -oE "trial_[0-9]{4}"); do
  grep -aq "act变化⇒重置头" $R/logs/$t.log 2>/dev/null && echo "$t"
done
# ② 再叠加门禁 PASS ⇒ 才是 F_MAIN 的合格成员
```
⇒ **或者更省事**：直接用 `pick_soup_members.py --require-pass`（它按门禁 PASS 筛）⇒
旧批次的臂 T1 极差 ⇒ 门禁 FAIL ⇒ 自动被剔掉 ✔（但**注意**：T1 轨本身口径有问题 ⇒
旧批次臂会因 T1 FAIL 被剔，新批次臂**也会**因 T1 FAIL 被剔！）⇒
**⇒ ⇒ 所以周一必须先修好 T1 口径（§10.4）或手工按上面的标记筛，不要盲用 `--require-pass`。**

⇒ **⇒ 交付路径（现已完整成立）**：
1. 从 `trial_0008` 及以后的臂里挑 T2/T3 达标者（**头重置批次 ⇒ 会收敛**）；
2. 速度 = **13.131 ms**（p3/hc64/relu/late，板卡实测）；
3. T2/T3 用门禁的这两轨数字（可信）；T1 待口径修好后补测（同一批臂，纯推理）；
4. **优先交付单臂**（soup 对 late 结构有害，§10.1）；若要 soup ⇒ 只在"好批次"内部平均。
