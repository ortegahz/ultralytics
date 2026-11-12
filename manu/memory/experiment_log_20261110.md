# 路线二实验总账（2026-11-10）

> 目标：RK3588 上 **int8 ≤ 30 ms** 且 **微小目标召回不降**（靶子 nano ≥ 0.6978 = P1 冻结基线水平）
> 判定权威：**官方三轨门禁**（T1/T2/T3）+ **板卡实测延迟** + **NanoBench 分层 nano/micro**
> 交付参照：`trial_0031`（P3/hc64/silu/late，int8 19.63 ms）

---

## 1. 结论摘要

**没有一个候选比 `trial_0031` 更好。** 交付件保持 `trial_0031`，它是联合门禁「速度 ≤30 ms **且** nano 不降（对等线 0.6666）」的**唯一 PASS 臂**。

- **速度目标达成**：真 2px（P1 全路径 320 头）也能进 30 ms —— `p1_hc32 + 颈收窄0.5` = **28.714 ms**；`p1_late` 22.66 ms；P2 系 17.9~21.0 ms；最快臂 13.045 ms。
- **召回目标在同预算内不可达**：P1/P2 家族 nano 天花板 **0.6493** < 交付件 0.6666 < P1 冻结基线 0.6978；且官方 **T2 虚警 FP 118（红线 36）**判死、还慢 3 ms。
- **机理**：nano 瓶颈是「**头对点源的灵敏度 + 训练配方**」，**不是分辨率**。

---

## 2. 臂级台账（本次共 14 条训练臂 + 21 个结构探针）

| trial | 结构 | 起点 init | 混训 | 周期 | 板卡 int8 | nano | 官方门禁 | 判定 |
|:--|:--|:--|:--|:--|--:|--:|:--|:--|
| **trial_0031（★交付件）** | P3/hc64/silu/late | RC2 | w2.0 | 20 ep | **19.63 ms** | **0.6666** | T1 0.885997 / T2 0.997531 (FP10) / T3 0.948330 (FP110) | ✅ **PASS（唯一）** |
| 9603 | P2-fused hc32 + late | RC2 | w2.0 | 50 ep（停@10） | 20.761（探针） | ep3 0.6281 → ep9 **0.5971** | T1 0.882469（Δ −2.33 pp） | ❌ 起点错 + 周期错 |
| 9613 | 同 9603 | RC2（reset 头） | w2.0 | 早停 | — | — | — | ❌ init 策略已定案为 merge |
| 9622 | P2-fused hc64（160 输出） | RC2 | w2.0 | 早停 | 探针 FATAL | — | — | ❌ 无实测速度 |
| 9634 | P2-fused hc48 + late | RC2 | w2.0 | 停@5 | ≈21.5（外推） | — | — | ❌ 同族冗余 |
| 9643 | P1-late hc32 + late | RC2 | w2.0 | 停@5 | 22.590（探针） | ep1 **0.6103** | — | ❌ A/B 判负（起点差） |
| 9644 | **真 2px**（p1_hc32+颈0.5） | RC2 | w2.0 | 停@7 | 28.714（探针） | ep7 0.6252 | — | ❌ F1 0.829 / FP 2293（全面判负） |
| 9645 | P1-late hc32 + late | **冻结基线 474** | w2.0 | 50 ep（停@25） | 22.590（探针） | ep1 0.6267 → ep5 0.6444 → ep10 0.6422 → **ep25 0.5463** | — | ❌ 周期错（50 轮崩塌） |
| 9646 | P1-late hc32 + late | **冻结基线 474** | **w8.0** | 50 ep（停@18） | 22.590（探针） | ep10 **0.6016** | — | ❌ 混训否证 |
| **9647** | P1-late hc32 + late | **冻结基线 474** | w2.0 | **10 ep 全退火** | **22.663（真权重）** | **0.6493** | T1 0.886292 / **T2 FP 118** / T3 0.942899 | ❌ **P1/P2 家族最优**（T2 撞红线） |
| 9648 | P1-late hc32 + late | 冻结基线 474 | w8.0 | 10 ep 全退火 | 22.59（探针） | 0.6193 | — | ❌ 混训否证（10 轮下同向） |
| 9649 | P1-late hc32 + late | 冻结基线 474 | w2.0 | **20 ep 全退火** | — | 0.6184 | — | ❌ 20 轮不如 10 轮 |
| 9651 | P3/hc64/relu/late | 冻结基线 474 | w8.0 | 10 ep（停@9） | — | ep3 0.6146 | — | ❌ 配方污染（w8.0 已被否证，重开） |
| 9652 | P3/hc64/relu/late | 冻结基线 474 | w2.0 | 10 ep 全退火 | **13.045（真权重）** | 0.6304 | T1 0.860488 / T2 FP 721 / T3 0.884503 | ❌ 精度全线退步 |
| **9653** | P1-late **hc64** + late | 冻结基线 474 | w2.0 | 10 ep 全退火 | **25.425（真权重）** | **0.6047** | — | ❌ **加宽头反而更差**（−4.46 pp） |

> 训练配方固定项：`act=relu`｜`--head-init merge`｜`--unfreeze-b0b4 1`｜`--train-on-native-grid 1 --min-radius 0`｜lr_scale 3.5｜warmup 1.5｜mix 口径1（Batch 内采样概率）｜no-GMC 特征集。

---

## 3. 板卡结构探针表（一次会话测完，锚点漂移 <0.4% ⇒ 可复现）

| 结构 | rknn ms | 整帧 ms | ≤30 ms |
|:--|--:|--:|:--|
| control cs_t0043（P1 320 头, silu） | 55.566 | 56.773 | 参照 |
| **P2-fused hc24 / hc32** | **17.956** | 18.657 | ✅ |
| P2-fused hc40 / hc48 | 20.563 / 20.570 | 21.267 / 21.273 | ✅ |
| P2-fused hc32 + late | 20.761 | 21.998 | ✅ |
| P2-fused hc32 + late + 颈0.5 | 20.038 | 21.418 | ✅ |
| **P1-late hc32**（P1 细节参与融合，160 算/320 输出） | **22.590**（真权重 22.663） | 23.823 | ✅ |
| **P1-late hc32 + 颈0.5** | **21.236** | 22.473 | ✅ |
| **P1-late hc64**（真权重） | **25.425** | — | ✅ |
| **P1-hc32 + 颈0.5（真 2px）** | **28.714** | 29.954 | ✅ 余量仅 1.3 ms |
| P1-hc24 / P1-hc32（不收窄颈） | 32.793 / 32.897 | 34.024 / 34.124 | ❌ |
| P2-fused / P1 的 hc64 变体（**探针权重**） | FATAL | — | 探针伪影，真权重正常（9653 25.425 ms） |

**已排除的结构**：
- **P0 highway**：+17.13 ms / +62 MB 读写；且只在真 2px（320 特征）下形状兼容 ⇒ 28.714+17.13 ≈ 46 ms。
- **dw_head**：DW 在 NPU 上是算力反模式（每组 1 通道、MAC 阵列空转）⇒ 导出侧已 `if False` 停用。

---

## 4. 机理杠杆（全部单变量对照 ⇒ 四个确认 / 三个否证）

| 杠杆 | 对照实验 | 效应 | 判定 |
|:--|:--|:--:|:--:|
| **起点权重** | 同结构同配方同轮：RC2 init vs 冻结基线 474 init | +1.64 pp，且 nano 趋势由**降**（0.6281→0.5971）转**升**（0.6267→0.6444） | ✅ **确认（主因）** |
| **周期长度** | 10 ep 全退火 / 20 ep / 50 ep | 10 ep 0.6493 ｜ 20 ep 0.6184 ｜ 50 ep(ep25) **0.5463** ⇒ **+10.3 pp** | ✅ **确认（次因）** |
| **头继承的尺度匹配** | P1 头 → P1 路径 vs P1 头 → P3 80² 路径 | 前者有效；后者 T1 −4.52 pp、T2 FP 721 | ✅ **确认（条件性）** |
| **混训比例** | w2.0 vs w8.0（50 ep 与 10 ep 两档） | −4.06 pp / −3.00 pp | ❌ **否证** |
| **头容量** | p1_late hc32 vs **hc64**（真权重、可上板） | 0.6493 → **0.6047**（更宽更差） | ❌ **否证** |
| **分辨率** | P1(160²/320²) / P2 / P3 三族同起点同配方对照 | nano 由起点与配方决定，结构只决定延迟 | ❌ **否证** |
| **P0 highway** | 板卡消融 | +17.13 ms ⇒ ≤30 ms 内不可得 | ❌ **排除** |

---

## 5. 交付件画像（可直接对客户讲）

| 维度 | trial_0031 |
|:--|:--|
| int8 板卡延迟 | **19.63 ms**（会话 19.705 / 19.629 互证）⇒ 50 FPS 级 |
| 官方 T1（GMC） | **0.885997**（FP 1336） |
| 官方 T2（龙泉山黄金池） | **0.997531（FP 10）** |
| 官方 T3（001324 盲测） | **0.948330（FP 110，FN 936 vs 基线 2523）** |
| NanoBench no-GMC | F1 **0.887122** ｜ nano **0.6666** ｜ micro **0.8370** ｜ regular **0.9445** |
| T2 孤立微小假峰（17187 帧） | **0**（占 FP 0.00%；尺寸分档 nano 0 / micro 0 / small 1 / regular 1191）⇒ 虚警全是大块 |
| 板上 e2e（真实 BMP 帧） | `--native-scale 4` ⇒ 6 检测 / TP 6 / FP 0 / **FP=1.000** |
| **诚实缺口** | vs P1 冻结基线 nano **−3.12 pp**（4px 严格档 −9.33 pp）；**micro +2.59 pp**、regular 稳；对 cs_t0043 / RC2 的 nano **都不降**（+1.68 / −0.12 pp） |

---

## 6. 本轮沉淀的可复用资产

| 资产 | 用途 |
|:--|:--|
| `manu/evaluation/tiny_target_subset.py` | NanoBench 评测台（分桶清单 / 分层评测 / 参照卡 / 自检） |
| `manu/evaluation/dual_constraint_gate.py` | **双约束联合门禁**（板卡 ms × nano 召回 ⇒ PASS/FAIL/FAIL 优先于 UNKNOWN） |
| `manu/evaluation/t2_isolated_tiny_fp.py` | T2 孤立微小假峰台（与官方判定器 **13 字段逐位一致**） |
| `manu/pipeline/npu/route2_acceptance.py` | 多 epoch 缓存 → NanoBench → 取 nano 优先的 epoch → 导 rknn → 板卡 → 联合门禁 |
| `manu/pipeline/npu/route2_final_gate.sh` / `final_board_session.sh` | 官方三轨门禁编排 / 板卡一次会话（先归档再上传） |
| `manu/pipeline/npu/board_speed_search.py` | 结构探针 → rknn → 一次会话测速（支持自定义探针集） |
| `manu/training/launch_route2_p2.sh` | 参数化启动器（`INIT_WEIGHTS` / `MIX_SPEC` / `EPOCHS` / `UNFREEZE`） |
| `manu/memory/route2_search_space_20261110.md` | 三轴定价的搜索空间设计（速度表 + 杠杆 + 排除项 + 换血史） |

**顺带修掉的工程缺陷**
1. 混样上报把 UAV 全算成 LQ：LQ 判定标记用了 `Path(longquan_data).resolve().parent` ⇒ 命中 `"manu"` ⇒ 已改用**数据集目录本身**（修复后实测 1:2.004 / 1:8.006 精确对齐目标；采样本身一直是对的）。
2. 板卡归档顺序：归档在上传**之后** ⇒ 自扫本轮臂（会话 9 s 结束、全 null）⇒ 已改为**先归档再上传**。
3. 板卡 FATAL 归因：**结构探针的 FATAL 是随机初始化头在 relu 下全灭造成的伪影** ⇒ 真权重同结构正常（9653 25.425 ms）⇒ **探针只能测延迟，不能测可用性**。

---

## 7. 后续（若 nano ≥ 0.6978 是硬指标）

该档的真实代价 = **55.48 ms + P0 highway + 3 轮全退火 + UAV 单域**；
而 **P0 在 ≤30 ms 内不可得**（+17.13 ms）⇒ 只能另立路线：

- **双模型 ROI**：0031 打主体（19.6 ms）+ 局部 P1 精修（输入面积 1/4~1/2 ⇒ 预计 12~15 ms）；
- **分档出图**：远距小目标单独走一支。

两条都不在「裁减通道」上继续微调（已被实测否证）。

---

## 8. ★★ Wave 1：NanoBench 定向搜索（2026-11-10 16:40 开跑，接手 §1~§7）

> 起因：用户追加要求「专门测小目标召回的数据子集」⇒ 把 **NanoBench-T1**（`manu/evaluation/tiny_target_subset.py`）
> 从"观测指标"升级为**选型主判据**，并以 `trial_0031` 真实配方为基线做**单变量**搜索。
> 全链：训练 → no-GMC 缓存 → NanoBench 分层（8px 主档 + 4px 严格档）→ **按 nano 下限选 epoch** → 官方三轨（GMC）
> → 导 rknn（act 跟随 ckpt 自描述）→ **一次**板卡会话 → `dual_constraint_gate.py` 联合门禁。
> 驱动脚本 `manu/pipeline/npu/nano_search.py`（幂等/台账/硬截止 `--stop-at`）。

### 8.1 臂表（全部 = trial_0031 配方 + 单变量；seeds 191xx；4 卡并发）

| trial | GPU | seed | 单变量 | 依据 | 状态 |
|:--|:--:|--:|:--|:--|:--|
| **9801** | 1 | 19101 | **对照**（零改动） | 给出 nano / T2-FP 的**种子噪声基线**（±0.0002 F1） | 训练中 |
| **9802** | 2 | 19102 | `--size-weight-small 1.5` | 方向三① 下限（与 9803 构成**剂量-响应对**） | 训练中 |
| **9803** | 3 | 19103 | `--size-weight-small 3.0` | 方向三① 上限 | 训练中 |
| **9804** | 0 | 19104 | `--train-on-native-grid 1 --min-radius 0` | 直击 §88 根因（监督分辨率 ≠ 头原生网格 ⇒ 峰被稀释）= **9504 从未跑完的格** | 训练中 |

- 启动：`setsid nohup … nano_search.py --wave W1 --epochs 5,10,15,20 --stop-at "2026-11-11 08:00" > /mnt/data/siping/runs/npu20ms/nano_search/nano_search.out 2>&1 < /dev/null &`
- 台账：`/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json`；日志 `nano_search.log`；
  臂权重 `/mnt/data/siping/runs/npu20ms_offcal/trials/trial_98xx/weights/epoch{5,10,15,20}.pt`（**每个 epoch 都存档**）。
- 训练实测节奏：Ep01 约 **115 s / 1779 iter**，4 臂并发单臂约 **185 s/epoch** ⇒ 20 ep ≈ **62 min**。

### 8.2 选 epoch 规则（`nano_search.py::pick_epoch`）
先取 **nano ≥ 0.6666** 的 epoch 集合里 **F1 最高**者；若一个都没有 ⇒ 取 **nano 最高**者。
（late_upsample 的 nano 随训练下滑 ⇒ 纯 F1 选臂会挑到最差 epoch。）

### 8.3 判据（对等线口径；**必须同时报 gate 原始 verdict**）
| 轨 | 门槛 | 0031 实测 |
|:--|:--|:--|
| NanoBench F1 | > 0.887122 | 0.887122 |
| NanoBench nano / micro / small / regular | ≥ 0.6666 / 0.8370 / 0.8654 / 0.9445 | 同左 |
| 官方 T1（GMC） | ≥ 0.885997 | 0.885997（**但 gate 原始 verdict = FAIL**，T1 绝对红线 0.905736 Δ −1.97 pp） |
| 官方 T2 | F1 ≥ 0.985 且 **FP ≤ 36** | 0.997531 / FP 10 |
| 官方 T3 | ≥ 0.948330 | 0.948330 |
| 板卡 int8 | ≤ 20 ms | 19.629 ms |

### 8.4 本轮顺手修掉的工程缺陷
1. `route2_acceptance.py` **act 硬编码 relu** ⇒ silu 权重导 relu 图 = **死模型（F1 = 0.000000）** ⇒ 已改为跟随 ckpt 自描述。
2. `heatmap_loss.py` 新增尺寸加权 focal；`verify_size_weight.py` 与旧实现**逐位对比 ALL PASS**；
   **尺寸口径对齐校验**：`(w_norm×640, h_norm×512)` 与 NanoBench `gt_physical_size` 分桶 **0/3360 不一致** ⇒ 加权口径 == 基准口径 ✓。
3. **基准口径缺陷入档（暂不修）**：NanoBench 把 GT 高度再 ×0.8（GT 本是原图像素）⇒ 物理 nano 实为 **14.64%**，
   参照卡写的 **16.85% 偏高 2.2 pp**；修了会与 0031 及全部历史臂失配 ⇒ 下一版统一修并重刷参照卡。

---

## 9. ★★ W1 实测 + 整晚排定（2026-11-10 17:50）

| 臂 | 变更 | 选中 ep | F1 | nano | nano@4px | micro | regular | 官方门禁 |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| 9801 | 对照（0031 配方换种子） | 5 | 0.889960 | 0.6673 | 0.5780 | 0.8325 | 0.9493 | 见台账 |
| 9802 | size-weight 1.5 | 5 | 0.891083 | 0.6843 | 0.6153 | 0.8300 | 0.9476 | 见台账 |
| **9803** | **size-weight 3.0** | 5 | **0.892343** | **0.6919** | 0.6210 | 0.8291 | 0.9416 | 见台账 |
| 9804 | 原生网格 + min-radius 0 | 15 | 0.892028 | 0.6749 | **0.6609** | 0.8386 | 0.9424 | 见台账 |
| — | **交付件 0031（参照）** | 20 | 0.887122 | 0.6666 | 0.6033 | 0.8370 | 0.9445 | T1 0.885997 / T2 FP 10 / T3 0.948330 |

**读法**：① size-weight 剂量-响应单调为正，sw=3.0 **全 epoch 稳定**（sw1.5 会衰减）⇒ 方向三①成立；
② 原生网格把 **nano@4px 抬 +8.3 pp** ⇒ §88 根因成立，但它修"定位"不修"检出" ⇒ 与 ① **机理正交，必须组合**；
③ **判据基准换成对照臂 9801**（同配方换种子）——它的 F1 已比 0031 高 0.0028 ⇒ 种子噪声比假设大一个量级。

**整晚（服务器自主，08:30 收工）**：W2~W9 共 **32 臂**，方向 = 剂量上探(4/5/6/8) + 原生网格×sw 组合 + hf_skip×sw + 周期网格(6/8/10/12)
+ LR 保底 + 大目标降权 + 12ep 短周期 + lr_scale/混样/warmup + 种子复现。调度器 `/tmp/night_runner.sh`，
台账 `/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json`。

## 2026-11-11 09:00 · 整晚结果收口（host 在场，服务器续跑）
- **整晚 36 臂全部完成（04:57 收工）**，台账 `nano_search_ledger.json` 36 臂齐全（nanobench/official/picked/dual）。
- **T1 全臂缺失 = 基线口径用错**（no-GMC 缓存当 GMC 基线用），已修 `nano_search.py`（`OFFICIAL_BASE_CACHE`），
  09:04 起对 8 个关键臂重跑官方三轨（T1 数值同时用 `track1_ab.log` 兜底解析出来，36 臂 T1 已齐）。
- **板卡离线**（192.168.0.64 No route to host）⇒ int8 ms 未测，已挂 5 min 轮询 watcher，板卡一上线自动测。
- **W10 补跑队列**：sw 5.5/6.0/7.0/10.0 剂量续探 + sw5.0/sw4.0 种子复现 + sw5.0×更宽小目标定义。
- 结论：**方向三① size-weight 是唯一真杠杆**（sw5.0 → nano 0.7086，比 0031 的 0.6666 高 **+4.2 pp**，
  F1 0.889712 > 0.887122，T1 0.888086 > 0.885997，T2 FP 5（0031 是 10），T3 0.948489 ≥ 0.948330）；
  hf_skip 否证；原生网格因 T2 FP 68~138 判死。

## 2026-11-11 09:30 · 换板后测速打通（板卡身份问题 → 延迟链路全部恢复）
- 用户告知「板卡更换了」⇒ 查 memory `board_swap_20261110.md`：**新板 192.168.0.211**（旧 .64 作废）。昨晚「板卡离线」= 连错板。
- 新板 NPU 实测可用：NPU 由 **DRM `/dev/dri/renderD129`** 暴露（dmesg `Initialized rknpu 0.9.8 ... on minor 1`）⇒ 旧档「缺 /dev/rknpu ⇒ 跑不了」证伪；
  NFS 已挂；良品 librknnrt（md5 56e425dc）已就位；host 重编侧车 `rknn_heatmap_smoke` 并推板；服务器免密已装。
- 两次板卡会话（s1: 7 候选臂；s2: control + 9801 + 9863 + **交付件 t31_silu_int8**）⇒ 新板候选臂 21.0~22.6 ms，
  **交付件本体 23.528 ms** ⇒ 新板整体慢 ~11~22%，绝对 20 ms 铁律需重新定标（判据改为「同会话不慢于 0031」）。
- 结构等价性取证：9863 与 t31 的 ONNX 逐项一致（stride2 / 320 输出 / 6.68MB）⇒ 交付板 19.306 ms 由构造保证。
- Nano 门禁清理（任务书 guardrail #2）：9811/9813/9844/9853（nano ≤ 0.6666）权重已移入 `rejected/`，台账标记。
- W10 波次 09:21 起训练中（9910~9916 + 9917/9918）。

## 2026-11-11 09:40 · 基线升轨：`trial_9863` ⇒ **Baseline_v2.0**（用户裁决）
- 用户裁决：9863 从此刻起是**全新绝对基线模型**。理由四条：① 打破「时延 vs 精度」妥协（轻量拓扑下 Nano 召回反超离线天花板）；
  ② 向下兼容（计算图/量化表/导出/前后处理 100% 沿用 0031，下游零改动）；③ 门禁通过（T2 FP 腰斩至 5、孤立假峰 0）；
  ④ 确立新的训练方法论母版。
- **资产归档动作已执行**：
  - `trial_0031` → `[Status: Superseded / Deprecated Baseline]`；
  - `trial_0080 (RC2)` → `[Status: Retired / Offline Historical Reference]`；
  - `9863` → **`Baseline_v2.0`（Non-GMC / <21 ms 唯一官方基线）**。
- **落地**：`manu/pipeline/npu/nano_search.py` 增 `BASE_V2` / `BASELINE_REGISTRY` / `BASELINE_NANO_FLOOR`，
  `NANO_FLOOR` 默认切到 9863（CLI `--nano-floor` 可切回）；`pick_epoch` 改 Beat-to-Pass 选 epoch；
  每臂记 `beat_to_pass_ref`；`final_report` 增 `baseline_v2`/`baseline_registry`/`beat_to_pass` 段与 `*_ge_base_v2` 检查列。
  新增 `manu/pipeline/npu/promote_baseline.py`（幂等落盘台账 + 回灌板卡会话实测值）。
  本地 md5 `8810c88b…`(nano_search) / `7577d3de…`(promote_baseline) = 服务器同 md5。
- **踩坑修复**：`phase_report` 的 Beat-to-Pass 阈值若用 6 位截断常量，**基线臂自己会被 `f1_ge_base_v2` 判负**
  （0.889712 > 0.8897117）⇒ 改为从台账取 9863 臂**精确读数**做阈值。
- **待裁决**：新板绝对 ≤20 ms 铁律重新定标（连在位交付件 0031 自身 = 23.528 ms）。
- **诚实口径补正（④）**：核对 `NanoBench_T1_summary_20261110.md` 参照卡 ⇒ 9863 的 **nano@4px = 0.6472 低于 0080 的 0.6668 与 base_frozen 的 0.6966**
  ⇒「反超离线天花板」**只在 8px 主档成立**；micro@4px 9863 0.7871 亦低于 0031 0.7938 / 0080 0.8215。
  同时 0080 / base_frozen 的官方三轨属**路线 B 混训线口径**（T3 序列集不同：0.8977 vs 0.9485）⇒ 不可与 NPU 线直接比大小。
- **独立核对（只读探针）后的五条口径补正**：
  ⑤ 官方门禁 `verdict=FAIL`，**0031 本体同样 FAIL**（veto 轨要求 T1 ≥ 0.905736 绝对红线）⇒ `official_ok/deliver_ok=true` ≠ 门禁通过；
  ⑥ **19.306 ms 是 0031 的数，9863 未在交付板实测**（同图推得）；交付板 11-10 复测 0031 = 19.665 ms（+1.9%）⇒ 20 ms 裕度仅 1.7%；
  ⑦ **基线绑死 ep5**（nano ep5 0.7086 / ep10 0.6942 / ep15 0.6897 / ep20 0.6909）；
  ⑧ 9863 `trial_result.json` = `NO_REPORT`（infrastructure_failure，gate_epoch=20）⇒ 权威门禁是 `gate/NANO_9863_ep5_official/`；
  ⑨ 两套参考卡并存，FP 口径可差 800+ ⇒ 统一用 `nanobench/NanoBench-T1-reference-card.json`。

## 2026-11-11 11:10 · 开新目标：闭环自适应搜索（击败 9863）
- 用户指令：继续搜索更优秀的模型争取击败 9863；host 18:00 关机，写自动程序在服务器跑，deadline 明早 09:00。
- **新增两条此前未探过的轴**：
  ① **Model Soup / SWA**（`ckpt_synth.py`）：零训练成本的后处理式权重平均；已实测 9863ep5+9822ep10+9884ep5
     → 403 key、结构自描述字段完整、19.66 MB，结构不一致直接拒绝。
  ② **闭环自适应调度器**（`adaptive_runner.py`）：读台账选 champion → 合成臂 + 以冠军为 init 的 8 个单变量训练臂
     → 训练/NanoBench/官方三轨/联合门禁/板卡(仅候选臂) → 重选 champion → 下一轮。
- 服务器自动化链已挂：`day1_runner`(W10) → `day3_w11`(W11) → `day4_adaptive`(自适应)，全 setsid 脱离；硬截止 11-12 09:00。

## 2026-11-11 12:30 · W10 结果 + 种子噪声量级实测（重大发现）
- W10 九臂（sw 5.5/6.0/7.0/10.0 + 种子复现 ×3 + 宽小目标 + 周期10ep）NanoBench 读数已出（gate 阶段进行中）。
- 🔴 **同配方只换种子的 nano 散布**：sw1.0 = 0.07 pp / sw4.0 = 1.04 pp / **sw5.0 = 1.53 pp**
  ⇒ **噪声量级随 sw 放大** ⇒ sw5.5(9910)=0.6607 是噪声不是剂量效应；**sw∈[3,8] 实为宽平台**。
- 🔴 **9863 的 nano 0.7086 含显著运气**（同臂其它 epoch 0.6897~0.6942）⇒ **打平线双报**：
  严格线 0.708648（对外）/ 稳健线 0.70098（sw5.0 两种子均值，内部识别可复现增益）。基线身份不变。
- 🟢 双正向信号：**宽小目标定义 thr25/maxside6**（micro 0.8395 +1.38 pp）· **sw6~10 的 F1/micro/nano@4px 更优**。
- 自适应调度器已据此升级：champion 评分加 nano4/F1 修正 · soup 增至 6 个（+soup_multiseed/soup_top_f1）· 训练臂 11 个
  （+swl=0.90 收回 micro、小目标定义三档、两个双正向组合）。md5 `9bf40843ab3488b37fa0e3667bce6966`。

## 2026-11-11 13:20 · W10 官方三轨全景 + Soup 材料分析
- W10 九臂官方三轨已齐：**9863 仍是 nano 第一（0.708648）**，无人击败；第二名 9822(sw8.0) 0.705104。
- **9915(sw10.0) 几乎全维优于基线**：F1 0.893426(+0.37pp) · T1 0.891316(+0.32pp) · T2FP **2**(基线5) · micro 0.8325(+0.67pp)；
  **9822** nano@4px 0.670605(+2.34pp)；**9916** micro 0.8395(+1.38pp)；**9884** T2FP 0。
- ⇒ **soup(9863+9822+9915+9916) 是最高概率胜出路径**（各维优势互补，零训练成本）。
- soup 配方扩到 9 个（+soup_multiseed / soup_top_f1 / soup_dim_top2 / soup_dim_pair），自检全部生成成功。
  `adaptive_runner.py` md5 `13697048a8a02e4725370574f49cb94a`（本地=服务器）。

## 2026-11-11 15:05 · SWA 实测 + W11 证伪 + 自适应调度器 v3 重构
- **SWA/合成实测（14:46）**：跨模型 Soup 全部崩盘（soup_multiseed 0.2422 / soup_dim_top2 0.3046 /
  soup_same_sw 0.4627 / soup_top3 0.6545）⇒ 不同配方/种子的权重不在同一 loss 盆地；
  **同臂相邻 epoch 的 SWA 成立**（swa_late = 9863 的 ep5~ep9 ⇒ nano 0.708885，已略超基线 0.708648）。
- **W11 全负 ⇒ 证伪**：「以 9863 ep5 为 init 再训练」4 臂 nano 0.6498~0.6779（低 3~6 pp）⇒ 二次退火伤 nano；
  v2 调度器的 11 个训练臂（12009~12019 / 12024~12034）与之同型 ⇒ 15:00 停掉 v2，避免整晚算力进死路。
- **v3 重构**（四个改动，均有代码级依据）：① 训练臂一律 base init(cs_t0043.pt) 起训；
  ② **epochs 20→5 逐位等价**（LR 是全局步号的函数、采样种子=seed+trial_number 且与 `--epochs` 无关、
  `total_steps` 只用于打印 ⇒ ep1~ep5 一致）⇒ 单臂 56 min → 14 min，算力 ÷4；
  ③ **配方级判优**（recipe_key 21 参指纹 + 多 seed 均值；冠军配方须 n≥2 且均值 nano 严格优于基线配方；
  打平线双报：严格 0.708648 / 稳健 0.700969；新增「待复现队列」给单臂过线格自动补种子）；
  ④ **官方三轨门禁只跑候选臂**（nano ≥ 基线−2 pp 或当轮前 8），`nano_search.py` 新增 `phase_pick()`。
  自检 `--selftest` CPU-only PASS；md5 `9098918e0135445cf7b0fe66350fc48e`（本地=服务器）。
- **夜间程序已拉起**：`setsid nohup bash manu/pipeline/npu/run_adaptive_night.sh "2026-11-12 09:00" 6 18 &`
  （PID 398941，SID 独立 ⇒ SSH 断开 / host 18:00 关机都不影响）；每轮 24 训练臂（6 种子扫描 + 18 突变格）
  + 4 SWA 合成臂 + 训练后自动 SWA top6；硬截止 11-12 09:00，剩余时间不足自动缩臂。
- 本夜第一优先突变格：**swl=0.5**（依据 9884：nano +1.70 pp / nano@4px +3.69 pp；正对 9863 的 micro 短板 −1.13 pp）。

## 2026-11-11 15:30 · 升轨工具通用化 + 板端参照修复
- 新增 `manu/pipeline/npu/promote_candidate.py`：`--list` / `--trial N --dry-run` / `--trial N --name Baseline_v3.0`。
  预检 = 六项判据 + 板端（≤0031 同会话 ×1.02）+ 种子复现（配方 n≥2 且均值 nano 严格优于基线配方）；单臂过线 = PASS\*（需 --force）。
  执行写 `baseline_v2`/`baseline_registry`/`beat_to_pass` + 臂标记 + `promotion_<trial>.json`，并打印需回填的 `BASE_V2` 常量。
  实测 `--trial 9915 --dry-run` 正确拒绝（nano/nano4/T3 不过）。
- 🔴 板端参照失效：台账 `board_sessions=null`（被 runner 的 save_ledger 覆盖）⇒ 「ms ≤ 0031×1.02」永远判不出。
  改为直接读会话文件（`ns.board_reference_ms` / `board_ms_for` / `parse_board_session_file`），beat_report + 升轨脚本改用之。
- 📌 新板读数：session2 `t31=23.528` / `9863=22.009`（快 1.519 ms）/ `9801=21.766`；session1 的 0031 因 `.stride=8` FATAL。
  ⇒ 9863 从未在交付板（19.x ms）实测，19.306 ms 是 0031 的数；绝对 20 ms 铁律在新板连 0031 自身都达不到 ⇒ 判据暂用同会话相对值。
- 🔴 修 `beat_report.py` winners 表漏「六项全平但未测速」的臂（索引用错）⇒ 改按六项全平入选 + ms 另算 PASS/PASS\*。

## 2026-11-11 15:40 · 修「打平基线」判定的缺项漏检（假 beat）
- 现象：`core_checks()` 对缺项是**跳过** ⇒ `all(chk.values())` 把「只过 nano/nano4/F1、没测 T1/T2FP/T3」的臂判成打平基线。
- 修：新增 `full_checks()`（六项齐全且全 True 才算 beat）· `champion_score()` 缺项计违约 · 板卡靶子只取六项齐全的臂 · beat_report 同步。
- 处置：15:08 带修复重启调度器（代价 ≈8 min×4 卡半截训练）；重启前清掉作废臂（12035~12062）的权重与台账条目，
  避免半截 ckpt 被 `phase_train` 的「末 epoch 已存在 ⇒ 跳过」误判；`next_mutations()` 的「已试过」判据改为
  **只有跑出读数的格才算试过** ⇒ 重启不丢格。自检新增两条回归断言并 PASS。

## 2026-11-11 15:25 · 夜间保险丝（守护 + 代码热升级）+ 系统网格生成器
- 新增 `run_adaptive_watchdog.sh`：调度器不在 ⇒ 重启；日志静默 >45 min ⇒ 判卡死 ⇒ 杀+重启（重启前删写一半的 epoch5.pt）；
  **代码 md5 变了且刚跑完一轮 ⇒ 在轮次边界重启**（避免浪费一轮的官方门禁）；续跑轮号 = 台账 rounds 最大轮号 + 1。
  状态 `watchdog_state.json` / 日志 `watchdog.log`，setsid 脱离。
- 新增 `generate_mutations()`：固定 43 格用完后转系统网格（sw × swl × 小目标定义 / lr × LR保底 / warmup × focal / offset × 混样），
  跳过台账里已跑出读数的配方键 ⇒ 整晚不熄火。
- 代码 md5（本地=服务器）：adaptive_runner `557a233e…` / nano_search `c85d29d1…` / beat_report `a466ef76…` /
  promote_candidate `b43b3909…` / run_adaptive_night `4a812829…` / run_adaptive_watchdog `ba70199c…`。
  `--selftest` CPU-only PASS（含三条回归断言）。

## 2026-11-11 16:10 · 调度器 v3.1（时间预算实测校准 + 可升轨候选落盘）+ 🔴 T1 读数缺失根因
- 🔴 **根因定位（官方三轨 T1 整片消失）**：`auto_epoch_gate_check.py` 的 Track1 A/B 用 `--baseline-cache` 的基线臂缓存
  去复现冻结 headline（F1 0.905736 / TP 21624 / FP 1014）；早期误传 no-GMC 缓存 `nogmc_ref_weekend/uav_cache_base.pkl`
  （实测峰值 F1 0.900897）⇒ `status=ab_failed rc=2` ⇒ 台账 `t1=0.0` ⇒ 六项判据永远缺一项 ⇒ **永远判不出打平**。
  中招臂实测：9802/9804/9812/9813/9821/9823/9824；正确缓存 `runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl`
  ⇒ 9803/9811/9814/9822 拿到有效 T1。本夜代码用的是正确缓存。
- 🟢 `adaptive_runner.py` 加：① `plan_minutes()` / `shrink_plan()`（按 R1 实测校准：训练 16 min/4 臂、
  NanoBench 1.3 min/臂、门禁/dual/板卡/报告固定 25 min ⇒ 24 臂一轮 = 152 min；收工阈值 70→60 min）；
  ② `promotion_candidates()` + `promotion_ready.json`（三关：n≥2 seed / 六项全平 / 均值不劣于基线配方）。
- 服务器端 `--selftest` PASS；md5 `adaptive_runner.py 8ee4f738…` 本地=服务器一致；经 watchdog 轮次边界热升级在 R2 生效。

## 2026-11-11 17:03 · 🔴 台账竞态事故复盘 + 修复 + 重启（恢复 R1 的 24 个臂）
- 事故：16:52 `phase_nanobench` 抛 `FileNotFoundError: nano_search_ledger.json.tmp -> nano_search_ledger.json`
  ⇒ R1 整轮报废。损伤：24 臂全训完，仅 5 臂有 NanoBench 读数、0 臂有 picked ⇒ 门禁/板卡/报告全丢、`rounds` 账目错位。
- 根因：`save_ledger()` 全线程共用固定临时文件 `LEDGER+'.tmp'` ⇒ 并发 `os.replace` 互踩；
  NanoBench 阶段每臂每 epoch 存一次（84 次/轮）⇒ **每轮必撞**。
- 修复：① `save_ledger` 线程安全（唯一临时名 + RLock + dump 重试 6 次）；② `phase_nanobench/phase_gate/phase_train`
  单臂 try/except；③ `adaptive_runner`：`safe_phase()` 阶段容错 + `find_orphan_arms()` 崩溃恢复
  （孤儿臂在下一轮开始前并入本轮补评 ⇒ 冠军配方也能用）。
- 处置：17:03 重启调度器（`--start-round 2`，新 md5 作守护基线）；启动日志 `[R2] 崩溃恢复：35 个孤儿臂`
  （12009~12019 + 12039~12062）。服务器端 `--selftest` PASS。md5：adaptive_runner `292539a4…` / nano_search `f85348e4…`。

## 2026-11-11 17:45 · R1 抢救完成（swl0.3 = 最强格 0.7271）+ 冠军规则「均值稀释」修正 + 队列重排
- 恢复的 35 个孤儿臂全部补评完成。关键读数：12047(sw5.0×swl0.3) ep3 **0.7271**/ep4 0.7136/ep5 0.7169（三 epoch 全过基线线）；
  12044(sw5.0) 0.7072；12045(swl0.5) 0.6923；12046(swl0.7) 0.6779；12048(swl0.9) 0.6876；
  12053(sw5.0×lrfloor0.10) ep4 nano 0.7096 / nano4 0.6567 / F1 0.892338（**NanoBench 侧三项全平基线**）。
- ⚠️ swl 降权拿 F1 换 nano（12047 F1 ≤0.8860 < 0.889712 ⇒ F1 项不过）⇒ 方向 = swl0.3 + lrfloor0.10 组合。
- 🔴 冠军规则 bug：基线配方组 seed 变多（n=8）⇒ 均值被稀释到 0.696568 ⇒ n=2 的 sw4.0 组（0.6986）被误升格。
  修（v3.2）：「单臂最佳格直通通道」（≥基线最佳+0.5 pp 且同臂 ≥2 epoch 过线）⇒ 搜索中心切到 sw5.0×swl0.3。
  队列最前插 14 个 swl0.3 族群格。R2 已开跑（24 臂 + 3 SWA 合成臂）。
- md5：adaptive_runner `ea154cf5…` / nano_search `f85348e4…`（本地=服务器一致）；`--selftest` PASS。

## 2026-11-11 17:52 · 修 watchdog 续跑轮号（len(rounds) → max(round)）
- `rounds_now()` 返回记录条数而非最大轮号 ⇒ 跑完 R2 后下一轮会算成 2 ⇒ 触发重启就把 R2 重跑（浪费 ~2.5 h）⇒ 改为 `max(round)`。
- md5 `run_adaptive_watchdog.sh 4ab0965e…`（本地=服务器一致）；17:52 重启守护（state 保留 code_md5=ea154cf5 / rounds=0）。

## 2026-11-11 18:05 · 选 epoch 规则升级（六项合取）+ 守护双文件监控
- 旧规则漏洞：选 epoch 只在「nano 过线」里取 F1 最高者 ⇒ 忽略 nano4 ⇒ 12047 的 ep5（nano4 0.646739，差 0.0005）
  被选中，而 ep4（nano4 0.6569）是过的 ⇒ 白丢候选。新规则：三项全过优先、否则退回 F1 最高；阈值用台账精确读数。
  回归：全台账 140 臂重算无退化；当前三项全平臂 = 9863 + 12053(lrfloor0.10)。
- 守护改为监控 `adaptive_runner.py + nano_search.py` 的组合 md5（`71bb80ce…`）；state 留旧 md5 ⇒ R2 边界自动重启生效。
- md5：nano_search `ebea0513…` / run_adaptive_watchdog `c4c8d762…`（本地=服务器一致）。

## 2026-11-11 18:15 · 突变队列重排（70 格）+ 系统网格以冠军为中心
- 队首新增 11 个组合格（swl0.3×lrfloor0.10×thr12ms3 / swl0.3×thr12ms3 / lrfloor0.10×thr12ms3 / swl0.25/0.35×lrfloor0.10 /
  swl0.3×mr2 / fb2.5 / mixw2.5 / thr32ms8 / lrfloor0.10×ow0.5 / thr12ms3×ow0.5）；11 个低先验单变量格降到队尾。
- 系统网格改为以冠军配方（sw/swl）为中心 ⇒ 样例由 grid_sw5.0_swl0.5_thr18 变为 grid_sw5.0_swl0.3_thr18。队列 43→70 格。
- md5 `adaptive_runner.py cc230097…`（本地=服务器一致）；`--selftest` PASS。R3 起生效（守护在轮次边界热升级）。

## 2026-11-12 09:00~09:30 · 定位「昨夜整夜零升轨」的真根因（swl 毁 T2），并修 8 处
- **现象**：昨夜 R2~R6 五轮、375 臂，best_nano 0.755~0.758（基线 0.708648，+4.7 pp）却 `promotion_ready` 五轮零候选。
- **排查链**：① 门禁窗口只取 nano 前 8 ⇒ 三项全平的臂从未跑官方三轨（4 个臂受害）② 补跑 triad 三轨后发现高 nano 臂官方 T1/T2/T3 全崩
  ③ 按 swl 分层统计 23 个有官方读数的臂 ⇒ **T2 随 swl 单调塌陷**（1.0→0.996 / 0.5→0.99 / 0.35→0.957 / 0.3→0.957 / 0.2→0.927 / 0.1→0.82），
  **而 NanoBench(no-GMC) 完全无感**（swl0.1 的 nano 0.7578 全场最高）⇒ 搜索中心被 nano 单指标拖进死区。
- **结论**：`epochs=5` 捷径安全（12053 epochs=5 的 T2 0.998104 > 基线 0.997440）；T2 塌的唯一驱动是 swl。
- **修复**：冠军加 T2 安全闸门 + 最佳臂须达严格线 + 突变/网格/待复现队列全面剔除 swl<1.0 + 队首改 lr_floor 阶梯
  + `save_ledger` 跨进程合并/flock（修「gate 脚本写回旧台账 ⇒ R6 轮记录被覆盖丢失」事故，`restore_r6_record.py` 已补回）
  + 新增 `gate_specific_arms.py`。
- **triad 实测**：12053(sw5.0×swl1.0×lr_floor0.10 @ep4) T1 0.887102 / T2 0.998104(FP4) / T3 0.949250(FP76) ⇒ 六项过五项，**只差 T1 0.098 pp**；
  12252/12263/12300 因 swl 0.5/0.35/0.2 ⇒ T2 0.9907/0.9711/0.9216 全 FAIL。
- **口径修正**：`lr_floor=0.10` 组 n=9 均值 nano 0.6884 < 基线配方 epochs=5 均值 0.6951 ⇒ 「四项全上」是运气臂 ⇒ 结论须 n≥2 复现。
- md5：`adaptive_runner.py 501df94c…` · `nano_search.py 8da45aed…`（本地=服务器一致）；服务器 `--selftest` PASS。

## 2026-11-12 09:29~09:35 · 夜搜 R7 正式启动（服务器自主跑到 11-13 09:00）+ §149 收尾补写
- **一键启动**：`manu/pipeline/npu/launch_night_20261112.sh`（补 R6 记录 → 起调度器 → 写守护 state → 起守护 v2 → 核验），
  实跑参数 `"2026-11-13 09:00" 7 6 18`（deadline / start-round=7 / n-seed=6 / n-mut=18）。
- **脱离 host 核验**：调度器 PID 1140198 **PPID=1**、SID 1140195；守护 PID 1143269 **PPID=1**、SID 1143269（**互不相同**）
  ⇒ 18:00 host 关机后服务器继续跑。4 卡满载 51/45/44/32%。`adaptive.log` 出现 `[R7] 训练 24 臂（合成 4 臂免训练）`。
- **搜索中心 = 基线配方**（`冠军配方 n=8 均值nano=0.696568 最佳=0.708648 | 已升格=False`）⇒ T2 安全闸门生效（无更优且 T2 已验证的格）。
- **台账自检**：arms=375、rounds=[2,3,4,5,6]（R7 轮末才写）、pending=0；
  9863 nano 0.708648 / T2 0.997440；12053 nano 0.709594 / T2 0.997983。
- **§149 补写内容**：12053 ep5/ep3、12041 ep5、12040 ep4 的官方三轨读数；**「同臂不同 epoch 官方最优与 nano 最优不在同一处」**
  ⇒ 升轨必须同臂同 epoch 六项齐全，不许分两次选 epoch 拼读数；板卡 R5/R6 假失败根因（192.168.0.211 不可达）与 09:31 恢复；
  第 9 处修复（基线组不再整组排除）与第 10 处修复（gate_specific_arms 报告重复行 + official 槽覆盖）。
- **代码 md5（本地=服务器）**：`adaptive_runner.py 212283d3…` · `nano_search.py 8da45aed…` · `promote_candidate.py f1fe810b…` ·
  `gate_specific_arms.py 649fbee5…` · `gate_triad_arms.py 85b45815…` · `restore_r6_record.py 8ce8830f…`；
  守护基线组合 md5 `d760c06e39bd326a48f27822bf3b07ff` / rounds=6。

## 2026-11-12 09:40~09:55 · 板卡体检 + 今夜板卡阶段主动禁用（保护板卡上的其它任务）
- **板卡本体实测（09:40~09:45）**：新板 `orangepicm5` / RK3588 / `5.10.160-rockchip-rk3588`；11-12 08:35 重启过（uptime 65 min）。
  重启后 **`/dev/rknpu` 设备节点消失**（`/proc/devices`、`/sys/class/misc`、`/proc/misc` 均无 rknpu；
  `lsmod` 空、`find /lib/modules -name '*rknpu*'` 空 ⇒ 无 rknpu.ko 可加载），
  **但驱动本体完好**：`/sys/class/rknpu_dev.8.auto` + `/sys/kernel/debug/rknpu` 存在、
  `CONFIG_ROCKCHIP_RKNPU=y`、dmesg 有完整 RKNPU 初始化 ⇒ **rknnrt 走 DRM/GEM 路径，不依赖 `/dev/rknpu`**。
  实跑 `rknn_heatmap_smoke arms/t31_silu_int8.rknn feat.jpg --loops 5 --warmup 2` ⇒ **NPU 正常出结果**，
  `rknn_run avg 22.606 ms`。
- ⚠️ **该读数作废**：用户 09:50 告知「板卡有别的任务在跑，现在测试的时间不一定准确」（板卡 load 2.19/8 核）⇒ 不进任何判据。
- **时延参照不变**：`t31_silu_int8 = 23.528 ms`（11-11 干净会话，`nano_search.BOARD_SESSION_FILES` 读静态文件，不受今夜影响）。
- 🔴 **审计出我们板卡阶段会破坏板卡上的其它任务**（三条硬伤）：
  ① `nano_search.clean_board_residue()` → `pkill -9 -f "rknn_heatmap_smoke|e2e_par2_demo|e2e_par_demo|e2e_demo"`；
  ② `board_speed_search.archive_board_arms()` → 把板上 `arms/*.rknn` `mv` 走；
  ③ 代码注释自述板卡驱动脆弱「长会话 → soft reset → rknn_init 132，**只能物理断电恢复**」。
- **处置（零代码改动 ⇒ 不触发守护热升级、不打断整夜算力）**：
  `mv /home/huangzhe/.ssh/id_board /home/huangzhe/.ssh/id_board.disabled_20261112_nano_search`
  ⇒ 流水线对板卡的 ssh/scp **0.17 s 内立刻失败**，一个字节都不碰板卡；
  标记文件 `$W/BOARD_PHASE_DISABLED.txt`（含恢复命令）；**明早板卡空闲后 mv 回去**。
- **对升轨无影响**：`promote_candidate.check_arm()` 的 `res["ok"]` 在板卡检查之前算完 ⇒ 六项齐全即 `ok=True`，板卡只出 warning。
  ⚠️ 明早待修：`promote_candidate.py:85` 的 `c["ms"]` 加在 `ok` 之后 ⇒ 板卡超预算也不挡升轨（口径漏洞）。

## 2026-11-12 10:00~10:45 · 嵌入式交付收口：L2/L3 实测闭合 + 官方 PASS + 🔴 RGA 驱动内核 oops 定位与修复
- **L2 新板 golden（`192.168.0.211`）**：e2e 路径 detached 跑，08:53→09:46（52 min，**3.781 fps**），
  喂入 11810 / 产出 **11808**，`dropped=0`，**2 帧被引擎跳过**（000199/000241）。产物 `regress/new_golden211/`。
- **Q1 位级对齐（新板 golden vs push-all 全量）**：共同 **11806** 帧 ⇒ feature md5 **11806/11806 逐位相同**、
  检测 TSV **11806/11806 逐帧全等**；**官方校验器 VERDICT PASS**（`regress/align_newgolden_pa11810.md`，
  5 项判据全过，TP/FP/FN/R/P/F1 全等，**ΔF1 = +0.000000**）。
- **Q2 换板环境差**：旧 golden vs 新 golden md5 **19.96%** / TSV **24.04%**（与旧 push-all 对旧板 19.97% 同量级）。
- **跨环境容差新量化**：检出数变化 **0.56%**；坐标差 中位 **0.0000 px** / p95 0.11 / p99 0.31；**94.17%** 帧 ≤0.1 px、**99.48%** ≤0.5 px。
- **L3 `--inflight 1` 实测**（不再是推断）：3000 帧喂入 → **3000 帧产出**，零跳过零溢出；与 serial 基线 **md5 3000/3000 + TSV 逐行全等**。
- **交付门禁复跑时发现板卡进程挂死**（7.5 min 只用 13 s CPU）⇒ 挖出 🔴 **RGA 驱动内核 oops**：
  `Internal error: Oops: 96000146`，两次（08:52:47 `Comm: rknn_mosaic_e2e` / 10:10:54 `Comm: mosaic_contract`），
  栈同：`__clean_dcache_area_poc → iommu_dma_sync_single_for_device → dma_sync_single_for_device →
  rga_mm_sync_dma_sg_for_device → rga_mm_get_buffer → rga_mm_get_channel_handle_info → rga_mm_map_job_info → rga_job_commit`，
  解引用 `ffffff8000000000`；前置日志 `rga_mm: This is not a buffer that has been imported, handle = -1`。
  手工复跑契约测试正常（30 s，`[RESULT] 199 passed, 0 failed`）⇒ **低概率竞态，非必现**。
- **根因定位**：① 子代理取证板卡有两份 librga（Debian `librga2 2.2.0-1` rga_api **1.10.0_[9]** md5 `32c4b08f…` = 运行时实际加载；
  Orange Pi rockit 捆绑 `librga.so` rga_api **1.9.1_[3]** md5 `55198ac0…` = SONAME 不匹配、运行时不用）；
  内核 RGA 驱动 **新板 `rockchip_rga v1.3.0` vs 旧板 `rga v1.3.1`** ⇒ **用户态 API 同版、内核驱动不同版**。
  ② 板卡上直接编译微基准 `manu/pipeline/npu/tools/rga_pat_ab_test.cpp`（6 种模式）证明：
  pat import 与否 **canvas 哈希完全相同**（`7b066d7c` / `0bb05135`）⇒ import 不改输出；
  且**每种模式都恰好 1 次** `not a buffer that has been imported` ⇒ 该内核报错与 pat 的 `handle=-1`、与 churn 都无关。
- **两处修复**：① `mosaic_core.h` `AllocCanvasBuffers()` 增加 `BufImport(&s->pat)`（原来 `pat` 从未 import，
  以 `wrapbuffer_handle(-1,…)` 传给 `improcess()`）；**A/B 证明输出零影响** —— 同一 e2e 二进制
  （`731f1244…` 未改）挂旧库 `03e1e301…` / 新库 `b39c2cf7…` 各跑 300 帧，`feature_md5.txt` md5 同为 `015649b0…`、
  mosaic TSV md5 同为 `b1882de0…`（`diff -q` 两项 IDENTICAL）。
  ② `copy_to_nfs.sh` 的 `run_board_test()` 加板卡侧硬超时 `timeout -k 10 $BOARD_TEST_TIMEOUT`（默认 600 s），
  超时打印 `[timeout]` + 指向 RGA oops 的判据（`dmesg | grep -c 'Internal error: Oops'`），强杀后重试一次。
- **交付包**：`make_build_dir.sh`（72 文件、MANIFEST 自洽）→ `copy_to_nfs.sh --verify` **5/5 门禁全过（EXIT=0，跑两轮）**；
  板卡侧实测新库 md5 `b39c2cf72aca357d6ad74086f9980365`，5 个测试程序 md5 不变；
  README 新增「🔴 板卡 RGA 驱动会偶发内核 oops（新板特有，嵌入式组必读）」一节。
- **新增遗留 L6**：RGA 驱动 oops —— 用户态已防御（pat import + 门禁超时），**根因需厂商侧升内核驱动**。
- **新脚本**：`manu/pipeline/npu/tools/rga_pat_ab_test.cpp`（板卡侧 A/B 微基准）。

## 2026-11-12 11:10~11:50 · 板卡频率锁 max：CPU/GPU/NPU/DMC 六个域 100 % 满频 + 温控钳位解除（L4/L5 闭合）
- **现象**：用户贴出 `policy0 scaling_max=600000 / p4,p6 max=2016000 / p0 cur=1416000`（**cur > max**）⇒ 判断为异常；空载复查发现 **六个域里 5 个不在 max**（CPU 三档 408 MHz、GPU 300 MHz、DMC 528 MHz，只有 NPU 已在 1000 MHz）。
- **排查链**：① 空载采样（2 s 粒度，10 点）⇒ 证实低频但 `scaling_max == cpuinfo_max`、cooling 全 0、SoC 35~37 °C ⇒ **空载低频是 governor 行为，不是温控**；
  ② 负载采样（97 点）⇒ policy6 仅 **20.6 %** 时间在 2400 MHz、GPU **0 %** 在 1 GHz、DMC **15.5 %** 在 2112 MHz ⇒ 定位 ①**cpufreq governor** ②**devfreq governor**；
  ③ 读 `thermal_zone0`：`passive 75/85/critical 115 °C`，单区驱动 4 个 cooling device（`thermal-cpufreq-0/1/2` + `thermal-devfreq-0`），governor `power_allocator`
  ⇒ 定位 ③**thermal cooling device 改写 `max_freq`**（`600000`/`2016000` 恰是 A55/A76 OPP 相邻档 ⇒ 与 cooling 降频步长吻合；`cur>max` 是两次 `cat` 之间的采样撕裂）；
  ④ 读设备树 `pwm-fan/rockchip,temp-trips` + `cooling-levels` ⇒ 确认**风扇曲线与 trip point 独立**（70 °C 即 PWM=255），抬 trip point 不会让风扇失速。
- **结论**：三个执行者全部中和。**只锁 `max_freq` 不够** —— cooling device 会把它改回去，**必须同时抬 passive trip point**；且 **`min_freq` 也必须钉到 max**（`max` 被改写时 `min` 成为新天花板）。
- **修复**：`/usr/local/bin/maxfreq-lock.sh`（幂等）→ CPU 三档 `performance`+`min=max=cpuinfo_max`；GPU/NPU/DMC `performance`+`min=max`（1000/1000/2112 MHz）；
  `thermal_zone0` passive 75000→90000、85000→100000，critical 115000 保留。持久化：systemd `maxfreq-lock.service`（`After=cpufrequtils.service loadcpufreq.service`、`Before=multi-user.target`）+ `/etc/rc.local` 兜底（实测 RC=0）。
- **实测**：同一 e2e 二进制 `--limit 600` before/after ⇒ 吞吐 **3.666 → 4.109 fps（+12.1 %）**、`run` 段 138.873 → 102.202 ms/帧（−26.4 %）、端到端时延 mean/p50/p95 538.9/445.2/1070.1 → 484.2/395.8/1011.0 ms；
  CPU p0/p4/p6 在 max 占比 79.4/49.5/20.6 % → **100/100/100 %**，GPU/NPU/DMC 0/100/15.5 % → **100/100/100 %**；**输出 md5 逐位相同**（feature_md5 `80dc6d0e…`、TSV `090f253b…`）⇒ 纯性能改动、零正确性影响。
- **长时 soak**：6 轮 e2e 连跑 ≈ 14 min（11:23→11:38，全 rc=0）⇒ **cur≠max 采样数 = 0 / 155**，cooling 全程 0，SoC 峰值 **82.23 °C**（风扇 255 满速），距 90 °C 钳位点 7.8 °C、距 115 °C critical 32.8 °C。
  ⚠️ **代价**：满频后稳态温度比 ondemand 高约 **40 °C**（36~59 → 44~82 °C）；封闭机箱/无风环境需重评 90 °C 阈值。
- **口径修正**：A/B 用的 `--limit 300`（前段帧更便宜）实测 fps **6.90**，**不可与 `--limit 600` 的 4.109 fps 混用比较**；before/after 一律同参数同二进制。
- **「power 模式」澄清（用户提问）**：板卡上三个同名不同物 —— `powersave` **governor** 的语义是**钉死 `min_freq`**（单独用 = 全线最慢，对固定供电板卡有害）；
  `thermal_zone*/policy=power_allocator` 是**温控 governor**；GPU 平台节点 `power_policy` 是**只读状态输出**（`[coarse_demand] always_on`，写 8 个值全被拒）。
  实测机制表：`performance`+min=max ⇒ max；`powersave`+min=max ⇒ **也是 max（等效）**；`powersave`+min=408000 ⇒ 钉 408000；`ondemand`+min=408000 ⇒ 随负载爬升
  ⇒ **决定频率的是 `min_freq`/`max_freq` 两个数，不是 governor 名字**；当前配置下误改成 `powersave` 也掉不下来（e2e fps 6.897 vs 6.910，差异在噪声内）⇒ 额外保险。
- **未做**：未改任何业务代码、未动板卡上的其它任务、未 reboot（板卡共享 ⇒ 只验证 `systemctl start rc-local.service` 与 unit 手工 start 均 RC=0，开机链路靠 `After=` 排序保证）。
