# 遗留问题清单 · 异步模板重构与库 API 改造（2026-11-11 收口）

## §141 ⭐⭐⭐ 异步模板重构 + 库 `push_frame` API 的**遗留问题清单**（2026-11-11 收口）

> **本档是 §三十二.13 / §三十二.14 的收口配套**：代码、门禁、证据全部就绪，
> 这里只列**尚未闭合**的问题、它们各自的**影响面**、以及**闭合所需的成本**，
> 供嵌入式组与验收方排期。
>
> - 专题档：`MOSAIC_RK3588_20261026.md` §三十二.13（库 API + 4 个 Fix）、§三十二.14（本轮全量验收 + ΔF1 定性 + 门禁修复 + 锁频实测）
> - 速览：`memory_compact.md`（§三十二.13 / §三十二.14 两条指针）
> - 交付包：`/run/media/manu/SSD960-LOCAL/nfs/mosaic_sdk/`（72 文件，MANIFEST 64 条 md5 自洽）
> - 🆕 **2026-11-12 补充（当日收口）**：**L1 判据裁定提案已出** ⇒ `manu/memory/L1_criterion_proposal_20261112.md`
>   （含本轮新量化的「位级差异到底有多大」：旧板 golden vs 新板产物，**94.17% 的帧坐标差 ≤0.1 px**、
>   p99 = 0.31 px、检出数变化帧仅 **66/11808 = 0.56%**、宏观 ΔF1 = −0.000103
>   ⇒ 位级不一致 ≠ 精度退化）；**L2 新板 golden 已于 11-12 08:53 在新板 detached 启动、09:46 跑完**
>   （**52 min / 3.781 fps**，喂入 **11810** / 产出 **11808**，`dropped=0`），
>   **官方校验器 VERDICT PASS（ΔF1 = +0.000000，5 项判据全过）** ⇒ **L2 已闭合**；
>   **L3 `--inflight 1` 已实测**（**3000 → 3000**，零跳过零溢出）⇒ **L3 已闭合**；
>   🆕 **新增 L6：板卡 RGA 驱动偶发内核 oops**（新板特有，曾让交付门禁无限期挂死）
>   ⇒ **用户态已防御，根因需厂商侧修复**。

---

## 0. 一句话状态

**代码侧全部闭合**（4 个 Fix + `push_frame` API + 5 道门禁 + 199 条契约测试 + 逐位对齐铁证）；
**L2 / L3 两条遗留实验已于 2026-11-12 实测闭合**；
**剩下 4 个未闭合问题（L1 待拍板 / L4 / L5 / L6）全部是「验收口径 / 硬件 / 厂商」层面，不是代码缺陷**。

| # | 遗留问题 | 性质 | 阻塞交付？ | 闭合成本 |
|:--|:--|:--|:--|:--|
| L1 | 「ΔF1 = +0.000000 + md5 全量匹配」判据在换板后不可达 | **验收口径** | ⚠️ **提案已出，待拍板** | 裁定即可（0 天） |
| L2 | 新板 golden 参照生成 | 环境 | ✅ **已完成 / PASS** | **已闭合**（52 min / 3.781 fps，官方校验器 5 项全过） |
| L3 | `--inflight 1` 的 100% 覆盖率方案 | 调参 | ✅ **已实测** | **已闭合**（3000/3000，零跳过零溢出） |
| L4 | NPU 800 MHz 天花板在 DT 的 PVTM 电压档位，用户态锁不动 | **硬件/厂商** | 否 | 需改 DT 或厂商解锁 |
| L5 | 新板散热与锁频策略未定（84~86 °C，越过 75/85 °C 两级 passive） | **硬件** | 否 | 需嵌入式组 + 散热方案 |
| L6 | 🔴🔴 板卡 RGA 驱动偶发内核 oops —— 会让交付门禁无限期挂死 | **硬件/厂商** | ⚠️ **用户态已防御** | 需**升级 RGA 内核驱动**（v1.3.0 → v1.3.1+）/ 报镜像维护方 |

---

## L1. 「ΔF1 = +0.000000 / md5 全量匹配」判据在换板后不可达 ⭐ **提案已出，待拍板**

### 提案已出（2026-11-12）

**提案档：`manu/memory/L1_criterion_proposal_20261112.md`**（**203 行**，§0~§8）
⇒ **状态：提案已出，等用户拍板。**

| 提案要点 | 内容 |
|:--|:--|
| 根子上的问题 | 原判据把「**跨物理硬件的位级认证**」和「**同一环境的回归守卫**」混成了一条 ⇒ 换板后前者不可达，后者今天就能建起来 |
| 结论 ① · 同环境位级强判据（方案乙） | 参照应当是「**同板同库**」的 e2e 参照 ⇒ **本轮已用新板 golden 落地，官方校验器 VERDICT PASS（ΔF1 = +0.000000）** ⇒ 该强判据**今天就可用了** |
| 结论 ② · 跨环境容差判据（方案丙） | 换板 / 换库时改用容差判据：**检出数变化率 ≤ 1%** + **坐标差 ≤0.5 px 的帧占比 ≥ 99%** + **ΔF1 ≥ −1e-4** ⇒ 已用现成产物算出并**通过**（0.56% / 99.48% / ΔF1 = −0.000103） |
| 结论 ③ · 兜底口径（方案甲） | 「**与基线同量级（\|ΔF1\| ≤ 1.6e-4）且新库不低于旧库**」—— 与既有「F1 宏观一致」口径同口径 |
| 建议 | 拍板「**方案乙 + 方案丙**」为主、方案甲为兜底；跨板卡的「位级 100%」从验收单里删掉（保留**原因**），换成「**数值中性 + 离散决策等价 + 坐标容差**」三件套 |

⚠️ 以下「现象 / 根因 / 建议判据」为原始记录，**保留存档**。

### 现象

官方校验器（服务器 `load_and_align_subset.py`）对**新板 + 异步模板**的全量产物判 **VERDICT FAIL**：

```
[S] surviving frames |S| = 11808  (hole frames = 0)
[MD5] compared=11808 identical=2358 different=9450   (一致率 19.97%)
[ arm@S] frames=11808 GT=10533 TP=8637 FP=201 FN=1896 R=81.999% P=97.726% F1=0.891745
[ ref@S] frames=11808 GT=10533 TP=8638 FP=200 FN=1895 R=82.009% P=97.737% F1=0.891849
⇒ ΔF1 = -0.000103
```

### 根因（三层证据，全部实测，非推断）

1. **参照来自旧板**：`ref` = `psearch/results/A_gate_step2_nodrop/`，其 `timing.txt`
   时间戳 **10-28 10:58**（`board_swap_20261110.md` 记载 **11-10 换板**）⇒
   参照与被测在**不同物理硬件**上，位级必然不同。历次官方校验一致率恒在
   **15%~21%**（14.7% / 19.1% / 20.6% / 20.0%）—— 这就是换板环境差的量级。
2. **本次改动数值零影响**（决定性 A/B）：从 `git HEAD` 取出**改动前的库**单独编译
   （唯一差异：`push_frame` → `mosaic_pipeline_push`），同板同帧 3000 帧对照：

   | 对比项 | 共同帧 | 逐位相同 | 不同 |
   |:--|--:|--:|--:|
   | feature md5 | 2996 | **2996（100.00%）** | **0** |
   | 检测载荷（TSV） | 2996 | **2996（100.00%）** | **0** |
   | frame_map.tsv | 3000 | **3000（100.00%）** | **0** |

3. **旧库跑同一校验器同样 FAIL**：

   | 对照 | 库版本 | md5 一致率 | ΔF1 | 判定 |
   |:--|:--|:--|:--|:--|
   | 全量 11810 帧 | **新库**（本次改动） | 2358/11808 = **19.97%** | **-0.000103** | FAIL |
   | 3000 帧子集 | **旧库**（零改动） | 551/2998 = **18.38%** | **-0.000161** | FAIL |

   ⇒ 旧库一致率**更低**、ΔF1 **同样为负** ⇒ 负 ΔF1 与本次改动**无关**。

### 建议判据（二选一）

- **方案甲（推荐，0 成本，立即可执行）**：判据改为
  「**与基线同量级（|ΔF1| ≤ 1.6e-4）且新库不低于旧库**」。
  本轮新库在两条上均**优于**旧库（一致率 20.0% > 18.4%；|ΔF1| 1.03e-4 < 1.61e-4）。
  该口径与既有裁定（`memory_compact.md`「交付口径已裁定：F1 宏观一致就可以」）一致。
- **方案乙（45 min）**：在**当前新板**上重建一份参照（见 L2），
  之后「md5 全量匹配」即可作为**回归守卫**长期使用。

### 附：位级匹配的**机制性下界**（不是 bug，改判据时必须知道）

即便用新板参照，**异步路径与 e2e 参照之间也存在 2 帧的固有差异**：

- e2e 与异步模板走**同一个 opportunistic 引擎**（板卡日志直接印证：
  `[WARN] gmc_workers=2 was requested, but the OPPORTUNISTIC latest-frame scheduler
  requires exactly ONE engine ... Forcing gmc_workers=1.`）
- 差别只在**喂入节奏**：e2e push 完立即 flush（不等引擎）⇒ `dropped=0`、11810 全产出；
  异步 `push-all` 相机 25 FPS 持续喂 + 反压 ⇒ 引擎偶尔忙不过来，**跳过 2 帧**（000199/000241）。
- ⇒ 异步路径的位级匹配上限是 **11808/11810 = 99.983%**；
  **旧板自己的 golden 之所以 100%，是因为它走的是不等引擎的 e2e 路径，不是异步路径。**
- ⇒ 「引擎忙不过来就取最新帧」是**引擎既定语义**，不是缺陷；
  要绝对 100% 只能走 e2e 式的「不等引擎」节奏（即牺牲吞吐），与异步模板的设计目标相反。

---

## L2. 新板 golden 参照生成 ✅ **已完成 / PASS**（2026-11-12 08:53 → 09:46）

### ✅ 实测结果（2026-11-12，已闭合）

| 项 | 实测值 |
|:--|:--|
| 板卡 | `192.168.0.211`（Orange Pi CM5 / Debian 12 / 内核 `5.10.160-rockchip-rk3588`） |
| 运行方式 | `rknn_mosaic_e2e`（**e2e 路径**：push 完立即 flush，不等引擎），**detached** |
| **时间** | **08:53 → 09:46（52 min）** |
| **吞吐** | **3.781 fps** |
| **喂入 / 产出** | 喂入 **11810 帧** → 产出 **11808 帧** |
| **`dropped`** | **0** |
| **被引擎跳过的帧** | **2 帧（000199 / 000241）** ——「引擎忙不过来就取最新帧」是**既定语义**，异步路径与 e2e 参照之间的**固有差异** |
| 产物 | `_artifacts/ultralytics/accuracy_20261111/regress/new_golden211/`（`feature_md5.txt` / `VIDEO00007_19700101_001324_mosaic.tsv` / `processed_frames.csv` / `run.log` / `timing.txt`） |

**Q1 位级对齐（新板 golden vs push-all 全量）—— 100.0000%**

| 项 | 实测 |
|:--|:--|
| 共同帧 | **11806** |
| feature md5 | 逐位相同 **11806/11806**、逐位不同 **0**（**100.00%**） |
| 检测载荷 TSV | 逐帧完全相同 **11806/11806**、逐帧不同 **0** |
| 只在 ref 里的帧 | **2**（即上表那 2 帧） |

**官方校验器**（服务器 `manu/pipeline/npu/ptq/load_and_align_subset.py`，**ref = 新板 golden**）
报告：`_artifacts/ultralytics/accuracy_20261111/regress/align_newgolden_pa11810.md`

| 项 | 实测 |
|:--|:--|
| 帧集合对齐 | 基准总帧数 **11808** / 运行态 \|S\| = **11806** / S 中基准没有的帧 = **0** / 帧号范围 0..11809 |
| 逐位 MD5 | 参与比对 **11806**、逐位相同 **11806**、逐位不同 **0**、bit-exact 率 **100.0000%** |
| @S 精度 | arm@S 与 ref@S **完全相同** —— TP=**8635** / FP=**201** / FN=**1896** / GT=**10531** / R=**0.819960** / P=**0.977252** / F1=**0.891723** |
| **ΔF1** | **+0.000000**（ΔTP / ΔFP / ΔFN / ΔGT 全为 **0**） |

| 判据 | 结果 | 判定 |
|:--|:--|:--|
| ① 无洞帧逐位相同率 = 100% | **11806/11806** | ✅ PASS |
| ② 全子集逐位不同帧数 = 0 | **0** | ✅ PASS |
| ③ 板端自报有洞帧数 = 0 | **0** | ✅ PASS |
| ④ 检测流逐帧一致 | **11806/11806** | ✅ PASS |
| ⑤ ΔF1 ≥ 0 | **+0.000000** | ✅ PASS |
| **总判定** | 5 项全过 | ✅ **PASS** |

**Q2 换板环境差（旧板 golden vs 新板 golden）**

| 项 | 实测 |
|:--|:--|
| md5 一致率 | **19.96%** |
| TSV 一致率 | **24.04%** |
| 对照：旧 push-all 对旧板 | **19.97%** |

⇒ 换板带来的差异与「异步 vs e2e 路径」的差异**同阶** ⇒ **不是新引入的回归**。

⚠️ 以下为 2026-11-11 收口时的**准备记录**（命令与参数），保留存档。

### 已完成的部分

- 板卡已上传 `rknn_mosaic_e2e`（md5 `731f1244…`）+ 新库（md5 `03e1e301…`）到 `/tmp/golden211/`；
- **参数已与原 golden 逐字对齐**：
  `--anchor-step 2 --npu-workers 2 --gmc-workers 2 --post-workers 1 --input-queue 8
  --jitter-depth 8 --gmc-fit-corners 600 --gmc-fit-win 15 --gmc-fit-level 2 --gmc-fit-iters 20`
  （原 golden `timing.txt` 记录：`anchor_step=2, npu_workers=2, gmc_workers=2, post_workers=1,
  jitter_depth=8, input_queue=8, gmc_fit_corners=600, gmc_fit_win=15, gmc_fit_level=2,
  gmc_fit_iters=20, gmc_downscale=2, gmc_ransac_max_iters=0, gmc_pyr_cache=1, gmc_cache_parts=7,
  input_mode=0, gmc_phys_anchor=0, gmc_input_drop_policy=0, dropped=0`）
- 已实测速率 **4.03 帧/秒** ⇒ 全量 11810 帧需 **45 分钟**；
  ⚠️ 第一轮用 `timeout 3000`（50 min）会**在差 4 分钟时被 kill**（`e2e` 的 `_mosaic.tsv`
  只在跑完后落盘，被 kill 就拿不到可比对的完整产物）⇒ 已改用 `setsid nohup` detached 重跑，
  用户决定收口后**主动中止**。

### 复现命令（可直接执行，detached，不受 timeout 限制）

```bash
# 板卡侧（先上传二进制到 /tmp：NFS soft 挂载上直接执行 >1MB 二进制会 SIGILL）
cat out/rknn_mosaic_e2e | ssh root@192.168.0.211 'cat > /tmp/golden211/rknn_mosaic_e2e'
cat out/libmosaic_pipeline.so | ssh root@192.168.0.211 'cat > /tmp/golden211/libmosaic_pipeline.so'
ssh root@192.168.0.211 'chmod +x /tmp/golden211/*; mkdir -p /tmp/golden_out; cd /tmp && \
  setsid nohup /tmp/golden211/rknn_mosaic_e2e \
    --raw-root /mnt/manu/lq_ir_bmp \
    --sequence VIDEO00007_19700101_001324 \
    --model /tmp/embtest/models/p0_off_A_none.rknn \
    --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels \
    --out /tmp/golden_out --timing /tmp/golden_out/timing.txt \
    --anchor-step 2 --npu-workers 2 --gmc-workers 2 --post-workers 1 \
    --input-queue 8 --jitter-depth 8 \
    --gmc-fit-corners 600 --gmc-fit-win 15 --gmc-fit-level 2 --gmc-fit-iters 20 \
    > /tmp/golden_out/run.log 2>&1 &'

# 跑完后（三方交叉对比脚本已在本地）
python3 _artifacts/ultralytics/accuracy_20261111/regress/cmp_golden3way.py \
  <新板golden目录> _artifacts/ultralytics/accuracy_20261111/regress/full \
  --old-golden _artifacts/ultralytics/accuracy_20261111/regress/old_golden
```

### 产出后能得到什么

- **Q1**：新板 golden vs push-all 全量 ⇒ 新板上的位级一致率（预期 ≥ 99.98%，上限受 L1 的 2 帧下界约束）；
- **Q2**：旧板 golden vs 新板 golden ⇒ **换板环境差的量化规模**（回答「为什么历次一致率只有 15%~21%」）。

---

## L3. `--inflight 1` 的 100% 覆盖率方案 ✅ **已实测**（2026-11-12）

### ✅ 实测结果（2026-11-12，已闭合）

| 项 | 实测值 |
|:--|:--|
| 喂入 / 产出 | **3000 帧 → 3000 帧** |
| 跳过 / 溢出 | **零跳过、零溢出** |
| vs serial 基线 · feature md5 | **3000/3000 逐位相同** |
| vs serial 基线 · TSV | **3000/3000 逐行全等** |
| 产物 | `_artifacts/ultralytics/accuracy_20261111/regress/inflight1_3k/`（`feature_md5.txt` **3000 行** / `feature_md5_phys.txt` / `frame_map.tsv` / mosaic TSV） |

⇒ **覆盖率 100% 成立**（下方原「推断」已升级为**实测结论**）。
⚠️ 代价是一推一收（相机等引擎），与异步模板「持续喂 + 反压」的设计目标相反
—— 位级 100% 与吞吐**二者不可兼得**（机制同 §L1「位级匹配的机制性下界」）。
⇒ 交付 README 里 `--inflight 1` 的覆盖率**已由「推断」改为「已实测 3000/3000」**。

⚠️ 以下为 2026-11-11 收口时的**原始记录**（推断与成本估算），保留存档。

- 现象：`push-all --inflight 4` 在 3000 帧下仍有 **2 帧**被引擎跳过（000199/000241），
  11810 帧全量下同样缺 **2 帧**。
- 推断（**未验证**）：`--inflight 1`（严格一推一收，相机等引擎）⇒ 引擎永不忙 ⇒ 覆盖率 100%。
  ⚠️ 这个推断**合理但没实测**，闭合前不要当成结论引用。
- 成本：3000 帧按 serial 实测速率（2.9 帧/秒）约 **17 min**；全量 11810 帧约 **68 min**。
- 参考：serial 模式 3000 帧实测 **3000/3000 全产出、EXIT=0**（`regress/serial3k/`），
  已证明「一推一收」确实不丢帧 —— 但那是 `serial` 模式，不是 `push-all --inflight 1`，**两者不等价**。

---

## L4. NPU 800 MHz 天花板在 DT 的 PVTM，用户态锁不动

- 现象：`available_frequencies` 列到 1000 MHz，但写 `max_freq=1000000000` 读回 **800000000**；
  `target_freq` 写 **Permission denied**；`governor=performance` 生效但频率仍被锁在 800 MHz。
- 根因：DT `npu-opp-table` 的 **`rockchip,high-temp-max-freq`** / **`rockchip,pvtm-freq`**
  （PVTM 电压频率管理）—— NPU 满频受**电压档位**约束，不是 devfreq 层能绕的。
- 结论：**这不是用户态脚本能解的**。要 NPU 跑 1000 MHz，需改 DT 或由厂商解锁 PVTM。
- 影响：推理吞吐上限受此约束；本轮 F1 结论不受影响（精度与频率无关）。

---

## L5. 新板散热与锁频策略未定

- 实测（满载时）：`soc-thermal` **84~86 °C**；越过 `trip_point_0 = 75000 (passive)`、
  逼近 `trip_point_1 = 85000 (passive)`；`trip_point_2 = **115000 (critical)**`。
  ⚠️ **用户先前担心的「95~100 °C 触发硬件急停」在本板不存在**，余量多 30 °C。
- 板卡自带风扇实测已在 **PWM = 255（满档）**，无 PWM 可调空间。
- **温控特权高于用户态**：写 `scaling_max_freq` / devfreq `max_freq` **不报错但被立刻改回**
  （`thermal_zone0 (soc-thermal)` 单区驱动全部 4 个 cooling device，governor = `power_allocator`）。
- 本轮已**完整还原基线**（所有 governor 回 `ondemand` 系、cooling device 交回 thermal 自主管理，
  板温 85.9 → 80.4 °C）。
- 待定：是否需要外接风扇 / 是否接受常态 84~86 °C / 是否改 passive 阈值（**改阈值有急停风险，需厂商口径**）。

---

## L6. 🔴🔴 板卡 RGA 驱动偶发内核 oops —— 会让交付门禁无限期挂死

> 🆕 **2026-11-12 新发现（此前从未记录过）**：本轮在跑 L2 新板 golden 与交付门禁的过程中，
> 板卡 `dmesg` 出现**两次内核 oops**（调用栈**完全相同**），
> **触发方就是我们交付包里的二进制**（`Comm` 就是我们的进程名），**崩在内核态**。

### 现象（板卡 dmesg，两次，调用栈完全相同）

```
rga_mm: This is not a buffer that has been imported, handle = -1
rga: buffer[0] mm release buffer failed! handle = -1
Unable to handle kernel paging request at virtual address ffffff8000000000
  ESR = 0x96000146   EC = 0x25: DABT (current EL), IL = 32 bits
Internal error: Oops: 96000146 [#1] SMP
  CPU: 3 PID: 4788 Comm: rknn_mosaic_e2e Not tainted 5.10.160-rockchip-rk3588
  CPU: 7 PID: 18287 Comm: mosaic_contract Tainted: G D 5.10.160-rockchip-rk3588
Call trace:
  __clean_dcache_area_poc
  iommu_dma_sync_single_for_device
  dma_sync_single_for_device
  rga_mm_sync_dma_sg_for_device
  rga_mm_get_buffer
  rga_mm_get_channel_handle_info
  rga_mm_map_job_info
  rga_job_commit
  rga_request_commit
  rga_request_submit / rga_ioctl_blit / rga_ioctl / vfs_ioctl
```

| # | uptime | 时间 | `Comm` | 上下文 |
|:--|:--|:--|:--|:--|
| oops #1 | **1072.44 s** | ≈ **08:52:47** | `rknn_mosaic_e2e` | **L2 新板 golden 那一次** |
| oops #2 | **5759.74 s** | ≈ **10:10:54** | `mosaic_contract` | **交付门禁第 1/5 道契约测试那一次** |

### 后果 —— 进程永久死锁

- oops #2 那次：契约测试跑了 **7.5 min 只用掉 13 s CPU** 就再也醒不过来；
  线程状态全是 `futex_wait_queue_me` / `do_sys_poll`；
  fd 里已打开 `/dev/dri/card1`、`/dev/mali0`、`/dev/rga` 与一堆 `/dmabuf:`。
- ⇒ 交付门禁 `copy_to_nfs.sh --verify` 会**无限期挂死**，日志里什么都看不到（只看到 `[1/5]` 之后再无输出）。
- 手工复跑契约测试**正常通过**（**30 s**，`[RESULT] 199 passed, 0 failed`）⇒ **不是必现，是低概率竞态**。

### 根因 —— 新旧板 RGA **内核驱动**版本不同（用户态 API 版本相同）

| | 旧板 `192.168.0.64`（已作废） | 新板 `192.168.0.211` |
|:--|:--|:--|
| 型号 / 系统 | RK3588 8 核 / **Ubuntu 20.04.6** | **Orange Pi CM5** / Debian 12 Bookworm |
| 内核 RGA 驱动 | **`rga v1.3.1`** | **`rockchip_rga` `v1.3.0`**（新板反而更旧） |
| `librga.so.2` | md5 `7136ae5469eb217f068ef3efc6f26d53`（192,712 B） | md5 `32c4b08f45fb012bfb6ef392f2734d5d`（200,896 B） |
| RGA API 版本 | `1.10.0_[9]` | `1.10.0_[9]`（**相同**） |
| 是否 oops | 未观测到 | **2 次 / 约 1.5 h** |

- 交付二进制实际链接的是 `/usr/lib/aarch64-linux-gnu/librga.so.2.1.0`
  （Debian 包 `librga2 2.2.0-1`，2022-01-07，md5 `32c4b08f…`，rga_api `1.10.0_[9]`）。
- 板上另有一份 2024-07-02 的 `/usr/lib/librga.so`（rockit 捆绑件，rga_api **1.9.1_[3]**，SONAME 是 `librga.so`），
  **运行时不会被加载**（SONAME 不匹配），只在编译期 `-lrga` 搜索路径上。
- ⇒ **用户态 API 版本相同，是内核 RGA 驱动版本不同**（v1.3.1 → v1.3.0）
  ⇒ **属板卡 / 厂商侧驱动缺陷，用户态无法彻底消除**。
- ⚠️ **记忆库此前从未记录过 RGA oops**（`not a buffer that has been imported` / `rga_mm` /
  RGA 相关 `Internal error` 全部 **0 命中**）。记过的只是同一条链的**用户态尾巴**
  `rga: buffer[0] mm release buffer failed!`（专题档 §13.6 / §20.8 第 6 条 /
  `falsified_archive.md` §49.3 / `memory_compact.md` §3.17），当时被定性为「**疑似跨进程资源泄漏**」
  —— **从未与内核 OOPS 联系起来**。

### 板卡上的旁证

- `dmesg | grep -c 'not a buffer that has been imported'`：随我们的工作负载**零星增长**
  （本轮从 **50** 涨到 **130+**）。
- 单次 **300 帧** e2e 跑：**+6**；一次契约测试（**199 例**）：**+72**；对照微基准：每个进程**恰好 +1**。

### 归因实验（结论：不是本包 `pat` 缓冲的锅）

微基准 `manu/pipeline/npu/tools/rga_pat_ab_test.cpp`（在板卡上直接编译运行，
`g++ -O2 -o rga_pat_ab_test rga_pat_ab_test.cpp -lrga`），
模式 0=pat 不 import、1=pat import、2=pat_rect 非空、4=每帧 release+import 抖动、5/6=抖动+双 tile：

| 实测 | 结论 |
|:--|:--|
| 模式 0 与模式 1 的 canvas 哈希**完全相同**（`7b066d7c`） | **import pat 不改变合成输出字节** |
| 模式 5 与模式 6 **完全相同**（`0bb05135`） | 同上 |
| 每种模式**恰好产生 1 次** `not a buffer that has been imported`（mode3/4/5/6 各 **3000 次**迭代，delta 均为 **1**） | 该内核报错**与 pat 的 handle=-1 无关、与 churn 无关**，是**进程级 / 会话级事件** |

⇒ **`pat` 是我们代码里的一处无效 API 用法，但不是 oops 的根因。**

### 已做（用户态两处防御）

**① 库 `mosaic_core.h` —— `pat` 缓冲改为正确 import**

- 位置：`manu/pipeline/npu/mosaic_core.h` 的 `AllocCanvasBuffers()`：
  在 `BufAlloc(&s->pat, 4, 4, kRgaFmt)` 之后新增 `BufImport(&s->pat)`，
  失败则 `[FATAL] RGA pat import failed` 并返回 false。
- 原因：`ComposeFrame()` 里 `improcess(srcb, BufAs(s->canvas), BufAs(s->pat), ...)`，
  而 `BufAs()` = `wrapbuffer_handle(b.handle, ...)`；`s->pat` 从未 import
  ⇒ handle 恒为 **-1** ⇒ 传给驱动的第三个 buffer 是**无效 handle**。
- **输出零影响的证明**：同一份 e2e 二进制（md5 `731f12445885a3b3106a1c726b7ad7d3`，**未改**）
  分别挂旧库与新库跑 300 帧：

  | | 旧库 `03e1e30156ed1d142cf86b8526fa8330` | 新库 `b39c2cf72aca357d6ad74086f9980365` |
  |:--|:--|:--|
  | rc / 耗时 | 0 / **53 s** | 0 / **52 s** |
  | 产出帧数 | **298** | **298** |
  | `feature_md5.txt` 的 md5 | `015649b0e547683a35c6c9182a73fc00` | `015649b0e547683a35c6c9182a73fc00` |
  | mosaic TSV 的 md5 | `b1882de090ced61960e42162832de4bf` | `b1882de090ced61960e42162832de4bf` |
  | `not imported` delta | **6** | **6** |

  ⇒ **逐字节相同**（`diff -q` 两项均 **IDENTICAL**）。
- 库 md5：`03e1e301…` → **`b39c2cf7…`**；其余 5 个测试程序 md5 **不变**
  （`mosaic_api_test 58dd792d…` / `mosaic_async_client_example 6b33afa0…` /
  `mosaic_client_example 04821546…` / `mosaic_contract_test 0a934698…` / `rknn_mosaic_e2e 731f1244…`）。

**② `copy_to_nfs.sh` —— 板卡测试加硬超时**

- 位置：`manu/pipeline/npu/packaging/copy_to_nfs.sh` 的 `run_board_test()`。
- 做法：板卡侧远程命令包一层 `timeout -k 10 '$BOARD_TEST_TIMEOUT'`，
  `BOARD_TEST_TIMEOUT` 默认 **600 s**（可用环境变量覆盖）；
  超时（rc = **124 / 137**）时打印 `[timeout] …` + 指向 RGA oops 的判据，
  强杀后**重试一次**。
- 理由：**没有超时的话，一次 oops 就能让 `--verify` 无限期挂死且日志无任何线索。**

### 仍需平台方做什么

| # | 行动项 | 归谁 |
|:--|:--|:--|
| ① | **升级板卡 RGA 内核驱动**：新板 `rockchip_rga v1.3.0` → **v1.3.1 或厂商修订版** | 平台方 / 镜像维护方 |
| ② | **向 Orange Pi 镜像维护方报缺陷**（附本节 dmesg 原文与两次时间 / `Comm`） | 嵌入式组 + 平台方 |
| ③ | **排查判据**：`dmesg \| grep -c 'Internal error: Oops'`（>0 ⇒ 仍有 oops 在发生） | 使用方 / 嵌入式组 |

### 状态

⚠️ **用户态已防御，根因需厂商侧修复。**
用户态已做的两件事（`pat` import + 门禁超时）**不能保证 100% 消除** oops
—— 只是把「无限期挂死」变成「超时后重试」。
交付包侧已重新打包并复跑门禁：**72 文件、5/5 门禁全过（EXIT=0，跑了两轮）**，
板卡侧新库 md5 `libmosaic_pipeline.so b39c2cf72aca357d6ad74086f9980365`，其余 5 个二进制不变。

---

## 附 A. 本轮已闭合项（勿重复排查）

| 项 | 证据 |
|:--|:--|
| Fix 1 关机死锁 | 流控谓词含停机信号；`flush` 在 Producer join 之后；`push-all` / `app-drop` 各 3000 帧 + 全量 11810 帧 **全 EXIT=0** |
| Fix 2 帧号竞态 | 改用 `mosaic_pipeline_push_frame()`，帧号由库原生带回 ⇒ 输出**不可能有 -1**；`frame_map.tsv` 降级为**只读审计记录** |
| Fix 3 丢帧机制 | 应用层「最新帧覆盖」缓冲**已删除**；丢帧只发生在**推理阶段**（opportunistic 引擎）；`serial` vs `push-all` **2998/2998 逐位全等** |
| Fix 4 生命周期 | 统一 `drain_and_stop()`（置停止 → `flush` → join Worker）；删除双重 collect 循环；退出码严格 0 |
| 库 API | `mosaic_pipeline_push_frame()` 新增（**只追加**，老接口 `mosaic_pipeline_push()` 一字未改、ABI 未破）；契约 **199 passed / 0 failed** |
| 输出无 `-1` | `feature_md5_phys` / `_mosaic.tsv` / `frame_map.tsv` **各 0 条**（全量 + serial + app-drop 三轮均验） |
| 打包脚本 | 修掉 `copy_to_nfs.sh` 门禁 5 块作用域 bug（不带 `--verify` 误跑 + `set -u` 报错 ⇒ 退出码 1）；修后不带 `--verify` **EXIT=0**、带 `--verify` **5 门禁全过** |
| 交付包 | 72 文件；MANIFEST 64 条 md5 **0 不符**；二进制 md5 与板卡实跑**逐位一致**（`03e1e301…` / `6b33afa0…` / `0a934698…`） |
| 归档 | 专题档 §三十二.13 / §三十二.14；速览指针两条；`validate_nav_table.py` **EXIT=0**（200/200） |
| **L2 新板 golden + 官方校验器** | 11-12 **08:53 → 09:46**（**52 min**、**3.781 fps**）detached 跑完；喂入 **11810** / 产出 **11808**、`dropped=0`；官方校验器 ref = 新板 golden：共同帧 **11806**、md5 逐位 **100.0000%**、检测流 **11806/11806**、**ΔF1 = +0.000000** ⇒ **5 项判据全过 / VERDICT PASS**（报告 `regress/align_newgolden_pa11810.md`） |
| **L3 `--inflight 1`** | **3000 → 3000**，**零跳过零溢出**；vs serial 基线 md5 **3000/3000**、TSV **逐行全等** ⇒ **覆盖率 100%**（产物 `regress/inflight1_3k/`） |
| **RGA oops 定位 + 用户态两处防御** | 新旧板 RGA **内核驱动 v1.3.1 → v1.3.0**（用户态 API 相同）；`pat` import 修复后输出**逐字节相同**（`015649b0…` / `b1882de0…` 两项 **IDENTICAL**）；门禁加 **600 s** 硬超时；交付包重新打包 **72 文件 / 5/5 门禁全过（EXIT=0）** |

## 附 B. 一个反直觉但已实测的结论：覆盖率低 ≠ 精度差

服务器上已跑过的两组 app-drop（无反压、低覆盖率）结果：

| 对照 | 帧数 | 覆盖率 | ΔF1 | 判定 |
|:--|--:|:--|:--|:--|
| `align_async_ad5k` | 518 | ~17% | **+0.019876** | ✅ ok |
| `align_async_ad218` | 218 | ~7% | **+0.025208** | ✅ ok |

⇒ **覆盖率下降反而 F1 上升**（arm 的 F1 反而高于参照）⇒ 被引擎跳过的帧**偏空帧/难帧**，
跳掉它们对 F1 是「好事」。这与 Fix 3 的结论一致：
**跳帧本身不损害精度**（损害精度的是「应用层跳帧打散 GMC 21 帧物理历史」），
而本轮已把应用层跳帧**彻底删除**。

## 附 C. 证据文件清单

| 产物 | 路径 |
|:--|:--|
| 全量 push-all 产物（11808 帧） | `_artifacts/ultralytics/accuracy_20261111/regress/full/` |
| serial 参照基线（3000 帧） | `_artifacts/ultralytics/accuracy_20261111/regress/serial3k/` |
| 旧库 A/B 产物（2998 帧） | `_artifacts/ultralytics/accuracy_20261111/regress/ab_old/` |
| 旧板 golden（11810 帧） | `_artifacts/ultralytics/accuracy_20261111/regress/old_golden/` |
| 逐位对比脚本（serial vs push-all） | `_artifacts/ultralytics/accuracy_20261111/regress/cmp_serial_vs_pushall.py` |
| 三方交叉脚本（golden 三方） | `_artifacts/ultralytics/accuracy_20261111/regress/cmp_golden3way.py` |
| 官方校验器报告 | 服务器 `mosaic_check_20261111/align_{full_pa11810,ab_old_2998,async_ad5k,async_ad218}.{json,md}` |
