# RK3588 板卡 T4 时延门禁 + 六项升轨链路 · 权威口径与调用方式

> 📄 生成日期：2026-11-23　　📍 工作目录：`workspace/ultralytics`（本机）
> 🔴 本报告**只读**得出：不 SSH、不连 `192.168.99.40`、不跑训练/推理/评估、不做 git 写操作。
> 🔴 本报告**不修改** `manu/memory/{rules,memory,memory_compact,.slim_memory}.md`（有别的代理在写）。

---

## 0. TL;DR 速查

| 事项 | 权威位置 | 一句话 |
| --- | --- | --- |
| 板卡 T4 判据**唯一真源** | `manu/pipeline/npu/t4_criterion.py:97-109` | `decide()` = 绝对口径 **OR** 同会话相对基线口径 |
| 红线 `19.989 ms` | `t4_criterion.py:38` | `trial_0031 19.597 ms × 1.02`（注释见 `:37`） |
| 基线干净参考 `19.629 ms` | `t4_criterion.py:39` | `BASE_REF_DEFAULT`，相对口径容差换算基准 |
| 六项判据权威实现 | `manu/pipeline/npu/six_criteria_audit.py:172-183`（`judge()`） | 六轴全部 `>=` 基线（T2 轴 `<=` 上限） |
| 「至少一轴严格优于基线」 | `auto_promote_watch.py:386-400` / `trackb_core.py:155-156` / `promote_candidate.py:390-396` | ⚠️ **不在** `six_criteria_audit` 里 |
| 🔴 **手工通道比自动通道严一轴** | `auto_promote_watch.py:386-400` vs `promote_candidate.py:390-396` | 自动通道有 **t2 轴**（T2 轨 F1，`:398`），手工 `_strictly_better` **没有 t2 轴** ⇒ 人工核对清单必须自己补 t2 |
| 台账路径 | `nano_search.py:54` | `/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json` |
| 门禁报告路径 | `nano_search.py:987` | `/mnt/data/siping/runs/npu20ms_offcal/gate/NANO_<trial>_ep<ep>_official/gate_report.json` |
| 板端实测记录 | `board_t4_watch.py:43-44,74` | `/mnt/data/siping/runs/npu20ms/board_measure_auto/{measured.json,t4_samples.json,session_*.txt}` |
| 🔴 **必须人工确认** | `baseline_guard.py:196-204` | 手工升轨后**必须回填源码常量**，否则 60 s 内被守卫回滚 |
| 🔴 **回填必须落权威根** | `nano_search.py:94`（`BASE_V2_CKPT`）/ `:95-101`（`BASE_V2`） | 守卫 `baseline_guard.py:57` 解析**实际被 import 的那一份** ⇒ SSHFS 导出根要另 `cp -f`（见 §5.1 步骤 9） |
| 🔴 **升轨前先停守望** | `auto_promote_watch.py` 常驻实例 | 它会自己「回填+升轨+写 memory+重启调度器」⇒ 与手工通道抢写同一份 24 MB 台账 ⇒ **同一臂可能被升两次** |

---

## ① 板卡 T4 时延门禁

### 1.1 T4 是什么意思 / 测的是什么

`manu/pipeline/npu/t4_criterion.py`（166 行，纯函数、零第三方依赖）是**判据唯一真源**。

| 常量 | 行号 | 值 | 含义 |
| --- | --- | --- | --- |
| `LIMIT_MS` | `:38` | `19.989` | 交付时延**绝对红线** |
| `BASE_REF_DEFAULT` | `:39` | `19.629` | 基线在**干净板卡**上的参考绝对值 ⇒ 相对口径容差基准 |
| `DIFF_TOL_MIN` | `:40` | `0.05` | 相对容差下限（防参照缺失时误判 PASS） |

**被测的量** = RK3588 板卡上 `rknn_run_avg_ms`（int8 量化 RKNN 的 NPU 推理平均耗时，单位 ms）。
证据：`board_t4_watch.py:43`（`@@SUMMARY@@ (\S+)\|rknn_run_avg_ms\|([\d.]+)` 正则，见 `:417-418`）。

**两条口径（OR 关系）**，判据实现 `decide()` @ `t4_criterion.py:97-109`：

```python
ok_abs = ms_median is not None and float(ms_median) <= LIMIT_MS          # :105
ok_rel = diff_median is not None and float(diff_median) <= float(tol)     # :106
ok     = ms_median is not None and (ok_abs or ok_rel)                     # :108
```

- **绝对口径**：`ms ≤ 19.989`
- **同会话相对口径**：`ms − 同会话基线臂读数 ≤ LIMIT_MS − 基线干净参考值`
  容差 `diff_tol()` @ `t4_criterion.py:88-94`：`max(0.05, 19.989 − base_ref_ms)`。
  参照臂 = **当前基线臂**（`session_ref_ms()` @ `:71-85`），必须与候选**同一次板卡会话**。
- 🔴 **没有绝对读数 ⇒ 不判定**（`:107-108`，"只有差值没有读数"不是放行理由）。

**为什么要有相对口径**（`t4_criterion.py:4-16`）：板卡读数**热漂移 ±0.5 ms**
（逐小时中位数 11-13 19.68 → 11-19 18 时 20.005 → 11-20 13 时 19.831，批内 max 21.462），
而红线 19.989 **正落在漂移带中央** ⇒ 纯绝对口径把同一张计算图的基线克隆臂全部误判 FAIL ⇒ 升轨链断链。

**中位数口径**：`pool_median()` @ `t4_criterion.py:133-139`；每臂攒够 `MIN_SAMPLES=5` 个样本才判，
最多 `MAX_SAMPLES=9`（`board_t4_watch.py:72-73`）。差值池键必须带参照身份 `|d@{ref}`（`t4_criterion.py:142-166`）。

**消费者必须走 `stored_pass()`**（`t4_criterion.py:117-130`）：有 `pass_t4` 字段就用它，
缺了才退回绝对口径 —— 自己重算 `ms <= LIMIT_MS` 会把相对口径结果整个丢掉（2026-11-19 15:27 / 15:30 两次现场事故）。

### 1.2 红线 19.989 ms 的出处

- `t4_criterion.py:37`（注释）：`红线沿用交付口径：trial_0031 19.597 ms × 1.02 = 19.989 ms`
- `board_t4_watch.py:46`：`LIMIT_MS = T4.LIMIT_MS  # trial_0031 19.597 ms × 1.02（交付时延红线）· 唯一真源 = t4_criterion`
- `promote_candidate.py:38`：同样从 `t4_criterion` 导入（不抄第二份）
- `board_t4_rejudge.py:38-40`：同样从 `t4_criterion` 导入

⚠️ **另一条更松的任务书口径**：`nano_search.py:177-178` 的 `DELIVER = dict(..., budget_ms=20.0, ...)`
（板端 ≤ 20 ms）。`board_t4_watch.py:158-159` 的 `DELIVER` 同样是 `ms: 20.0`。
**升轨链用的是 19.989（更严），不要混用 20.0。**

### 1.3 `board_t4_watch.py`（生产者，962 行）

| 项 | 内容 | 行号 |
| --- | --- | --- |
| 测什么 | 候选臂 int8 RKNN 在板卡上的 `rknn_run_avg_ms` 中位数 | `:370-440` |
| 产物 | `measured.json`（每臂最终读数+判定）、`t4_samples.json`（样本池+差值池）、`session_<trial>_ep<ep>_<时间戳>.txt`（会话原文） | `:43-44`, `:74`, `:413-415` |
| 台账回写 | `write_back_ledger()` 单臂 / `write_back_ledger_batch()` 批量（一次 load/save，防并发丢更新） | `:473-509`, `:512-554` |
| 台账字段 | `board_ms` / `board_ms_epochs` / `board_pass_t4` / `board_limit_ms` / `board_ms_ref` / `board_diff_ms` / `board_ref_trial` / `board_ref_ep` / `board_measured_at` | `:489-502`, `:533-546` |
| PASS/FAIL 怎么判 | `T4.decide(ms, diff_ms, tol)` ⇒ `pass_t4`；**绝对 OR 同会话相对**（`pass_t4_abs` / `pass_t4_rel` 分别落盘） | `:434`，`:438-439` |
| 候选怎么选 | `candidates()` 五桶排序：① 六项全过（家族六项全过数 ≥4 的整体提前）② Track-B 桶 ③ 5/6 近失桶 ④ 交付红线桶 | `:595-680` |
| 板卡忙则跳过 | `board_busy()`：**只 SSH 查板卡上**跑着别人的 `rknn_mosaic/e2e_v/measure_hm` ⇒ 读数会被污染 ⇒ **宁可不测**；🔴 **查不到本机第二实例**（见 §5.1 步骤 4） | `:706-726` |
| 样本不足要补测 | 不是"测过一次就跳过"：样本 < `MIN_SAMPLES` 就继续测 | `:747-785` |
| 严格优轴加权 | `todo` 排序：严格优于基线的轴多者优先 + 「可升轨家族」加权 | `:755-765`, `:787-820` |

**命令行**（`def main()` @ `:939`，argparse `:940-945`，轮询循环 `:951-958`）：

```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

$PY -u manu/pipeline/npu/board_t4_watch.py --once                      # 跑一轮
$PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 1200 \
     --deadline "2026-11-23 21:00"                                     # 常驻（默认 20 min/轮）
```

🔴 **常驻实例已在跑时，不要再手拉 `--once`**：`board_busy()`（`:706-726`）**只 SSH 查板卡**上的进程，
**查不到本机第二实例** ⇒ 双实例会并发导 RKNN / 上传 / 测速 ⇒ 污染 `measured.json` / `t4_samples.json`。
先 `ps -eo pid,etimes,args | grep '[b]oard_t4_watch.py'` 确认（见 §5.1 步骤 4）。

启动时先 `sync_measured_to_ledger()`（`:947`）把 `measured.json` 里的历史读数补进台账
（`T4.stored_pass()` 口径，`:588`）。

### 1.4 `board_t4_rejudge.py`（复判，217 行）—— 🔴 **不是单臂接口**

**权威命令行**（docstring `:12-15`，argparse `:91-93`）：

```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

$PY -u manu/pipeline/npu/board_t4_rejudge.py --dry    # 只看，不写盘
$PY -u manu/pipeline/npu/board_t4_rejudge.py          # 重判并写回
```

| 参数 | 含义 | 行号 |
| --- | --- | --- |
| `--dry` | 只打印判定变化，**不写** `measured.json` / `t4_samples.json` / 台账 | `:93`, `:205-207` |
| （**无 `--trial`**） | 🔴 该脚本**只能全量重判**：扫 `OUT/session_*.txt` 里所有出现过基线臂的会话，重算全部臂 | `:105-131`, `:134-178` |

**它做什么**：
1. 解析 `board_measure_auto/session_*.txt` 的 `@@SUMMARY@@` 行（`:108-130`）；
2. **只保留「同会话里有基线臂读数」的会话**（`:123-124`）⇒ 相对口径才有意义；
3. 绝对池 + 差值池合并（`:133-148`）；
4. `T4.decide()` 重判（`:152`）⇒ `pass_t4` / `pass_t4_abs` / `pass_t4_rel`；
5. 台账回写 `board_ms` / `board_pass_t4` / `board_diff_ms` 等（`:180-197`）。

**结论**：**单臂复判不能靠它**。
- 单臂要重新上板测 ⇒ 走 `board_t4_watch.py`（把该臂排进 `candidates()` 队列，见 `:595-680`），
  或直接调 `board_t4_watch.measure(trial, ep)`（`:370-440`）；
- 只是想用已有会话数据重算判定 ⇒ `board_t4_rejudge.py --dry`（全量）。

### 1.5 板端测时的前置条件

`board_t4_watch.measure()` @ `:370-440` 的完整链路：

| 步 | 动作 | 位置 |
| --- | --- | --- |
| 1 | 权重路径 `{CKPT_DIR}/trial_{trial}/weights/epoch{ep}.pt`（`CKPT_DIR = /mnt/data/siping/runs/npu20ms_offcal/trials`，`:173`） | `:372-376` |
| 2 | **导出 RKNN**：`RA.export_rknn(ck, name)` = `hm_export_onnx.py` → **`.onnx`** → `hm_convert_rknn.py` → **int8 `.rknn`** | `:379`；`route2_acceptance.py:97-127` |
| 3 | 上传到板卡 `root@192.168.0.64:/root/hm_speed/arms/` + 写 `.stride` 文件 | `:390-397`；板卡常量 `board_speed_search.py:32-34` |
| 4 | 🔴 **同会话参照臂上板**：把**当前基线臂**的 RKNN 也放上板（否则相对口径恒失效） | `:403` → `ensure_base_ref_on_board()` `:283-367` |
| 5 | 板卡测速：`CONTROL_NAME=base_int8.rknn LOOPS=30 WARMUP=10 SETTLE=5 bash measure_hm_speed_board.sh` | `:404-409`；`board_speed_search.py:136-139` |
| 6 | 落盘会话原文 + 解析 `@@SUMMARY@@` | `:413-418` |

🔴 **关键澄清：`hm_plain.onnx` / `ir_uav_det_rk3588_v1.0.rknn` 不在 T4 测量链上**。
- `hm_plain.onnx` 是 **PTQ 路线**（`p0_off`）的 ONNX，见 `manu/evaluation/run_trial0080_longquan_eval.sh:28,48`；
- `ir_uav_det_rk3588_v1.0.rknn` 是 **`trial_0080` 的已冻结交付产物**（`archive/19_主档正文_§283-298_20261120-20261123.md:1054-1055（§288）`）；
- T4 测量链用的是**每次现导**的 `n{trial}_ep{ep}_int8.rknn`（`route2_acceptance.py:111-112`）。

⇒ **升轨不会动板卡上挂载的 `ir_uav_det_rk3588_v1.0.rknn`**，也不动 `trial_0080` 交付底座。

### 1.6 🔴 两个基线板端数字的考证：`19.7505` vs `19.619`

| 数字 | 出处（行号） | 模型 | 时间 | 语境 |
| --- | --- | --- | --- | --- |
| **19.7505 ms** | `archive/19_主档正文_§283-298_20261120-20261123.md:1047（§288）` | **`trial_1990055`**（= Baseline_v3.1，**当前**基线） | 2026-11-23（§288 周末搜索） | "板端 T4：当前基线 `1990055` 实测 **19.7505 ms**（红线 19.989 ms，余量 0.24 ms）" |
| **19.619 ms** | `archive/16_主档正文_§236-257_20261118-20261119.md:266（§240）` | **`trial_1622005`**（= Baseline_v2.9，用户表里的"最新训练最优"） | 2026-11-19（§240 v2.9 升轨记录） | "\| 板卡 T4 \| **19.619 ms**（红线 19.989） \| 19.668 ms \| −0.049 ms \|" |

**结论：不是"同一个模型测两次"，而是"两个不同模型、两次独立板卡会话"。**
差值 `19.7505 − 19.619 = 0.1315 ms` ⇒ **两个不同模型之间的差**。

佐证链（同一模型重复测量的抖动幅度，与 0.13 ms 同量级）：

| 模型 | 全部板端读数 | 出处 |
| --- | --- | --- |
| `trial_1622005`（v2.9） | 19.619 / 19.668 ms | `archive/16_主档正文_§236-257_20261118-20261119.md:266（§240）` |
| `trial_1990055`（v3.1） | **19.721 / 19.682 ms**（2026-11-20 升轨时）→ **19.7505 ms**（2026-11-23） | `memory.md:12`, `:16`（索引头）＋ `archive/17_主档正文_§257-265_20261119-20261120.md:1425`, `:1430`（§260）＋ `archive/19_主档正文_§283-298_20261120-20261123.md:1047`（§288） |
| `trial_2020035`（v3.0） | 19.629 / 19.629 ms | `archive/16_主档正文_§236-257_20261118-20261119.md:958`, `:991`, `:1012`（§251） |

⇒ `trial_1990055` 单模型自身三次读数极差 = `19.7505 − 19.682 = 0.0685 ms`；
而 `1622005 → 1990055` 差 `0.1315 ms`。**两个数都是真实板端读数，不要混用、不要当成"重测"。**

🔴 **另有一处头部表不一致（需警惕）**：`manu/memory/memory.md:20` 的头部摘要写
`| 板端 int8 | **19.629 ms**（板卡 .64 实测，红线 19.989 ⇒ PASS） |`，
但同一张表的 `Status` 行（`:21`）说当前基线是 `trial_1990055`。
`19.629` 是 **`trial_0031` / v3.0 的参照值**（= `t4_criterion.BASE_REF_DEFAULT`，`:39`），
**不是 1990055 的读数**。基线快照的正确值在 `archive/18_主档正文_§268-282_20261120-20261120.md:25`（§268）：**19.682 ms**。
⇒ **引用板端数字请用 `archive/18_主档正文_§268-282_20261120-20261120.md:14-25`（§268） 的基线快照表，不要用头部摘要行。**

### 1.7 🔴 与基线快照的一致性核对（memory 记录 vs 代码算出来的）

memory 记录的基线口径（`archive/18_主档正文_§268-282_20261120-20261120.md:14-25`（§268））：
nano `0.7242438563327032` / nano4 `0.6609168241965974` / F1 `0.8906832039622916` /
T1 `0.8909238824339264`（fp 1473）/ T2_fp `3` / T3 `0.9488855869242199` / 板端 `19.682 ms`。

**代码侧交叉验证**：

| memory 数字 | 代码里的来源 | 一致？ |
| --- | --- | --- |
| T1 `0.8909238824339264` | `six_criteria_audit.py:38`（`BASE["t1"]`）/ `nano_search.py:1019` 从 `gate_report.json` 的 `tracks.track1_uav.finetune_peak.f1` 取 | ✅ 逐位一致 |
| T1 `tp=21355 fp=1473` | `eval_arm_pc_gmc_ab.py:113-119`（`sweep` 逐帧 `match_at` 累加）+ `:118` 落 `{"tp","fp","fn"}`；`archive/16_主档正文_§236-257_20261118-20261119.md:1265`（§254） 记 "T1 顶格 .8909238824（TP 21355 / FP 1473）" | ✅ |
| T1 `fn=3756` | **派生值**：`fn = gt − tp = 25111 − 21355 = 3756`（gt=25111 见 `auto_epoch_gate_check.py:55`）。台账只存 `t1`/`t1_fp`（`nano_search.py:1024`），**不存 fn** | ✅ 可复算 |
| T2 `t2_fp=3` | `auto_epoch_gate_check.py:508`（`overall["fp"]`）→ `nano_search.py:1025`（`t2_fp`）→ `six_criteria_audit.py:39`（`BASE["t2_fp"]=3`） | ✅ |
| T3 `0.9488855869242199`（fp 76） | `auto_epoch_gate_check.py:509`（`overall["f1"]`）→ `nano_search.py:1026`（`t3`）；`archive/16_主档正文_§236-257_20261118-20261119.md:967`（§251） 记 "T3（盲测轨） .9488856（fp 76）" | ✅ |
| 🔴 T2 **FP=4** 的另一处记录 | `memory.md:19` / `memory_compact.md:96` 写 "T2 F1 .9931331 / FP **4**"（v3.1 升轨当时），而 `six_criteria_audit.py:39` 写 `t2_fp=3`（后测值） | ⚠️ **两处数字不一致，以 `six_criteria_audit.py:39`（当前判据真源）为准** |

---

## ② 六项门禁 → 升轨链路

### 2.1 六项判据的权威实现

**`manu/pipeline/npu/six_criteria_audit.py:172-183`**：

```python
def judge(r: dict, base: dict | None = None) -> tuple[int, list[str]]:
    B = base if base is not None else live_base()
    flags = [
        (r["nano"]  is not None and r["nano"]  >= B["nano"]),    # :175
        (r["nano4"] is not None and r["nano4"] >= B["nano4"]),   # :176
        (r["f1"]    is not None and r["f1"]    >= B["f1"]),      # :177
        (r["t1"] >= B["t1"]),                                     # :178
        (r["t2_fp"] <= t2_cap(B)),                                # :179
        (r["t3"] >= B["t3"]),                                     # :180
    ]
    miss = [n for n, ok in zip(CRIT, flags) if not ok]
    return 6 - len(miss), miss
```

| 判据 | 比较方向 | 容差 | 行号 |
| --- | --- | --- | --- |
| `nano` | `≥` 基线 | 无容差（逐位比较） | `:175` |
| `nano4` | `≥` 基线 | 无容差 | `:176` |
| `f1` | `≥` 基线 | 无容差 | `:177` |
| `t1` | `≥` 基线 | 无容差 | `:178` |
| `t2_fp` | `≤ max(基线 t2_fp, 5)` | **上限 = `max(base, 5)`**（永不比基线更严，也不低于用户口径 5） | `:179` + `t2_cap()` `:70-72` + `T2_USER_CAP=5` `:67` |
| `t3` | `≥` 基线 | 无容差 | `:180` |

- 轴顺序常量 `CRIT = ["nano","nano4","f1","t1","t2fp","t3"]` @ `:52`
- **基线值从台账 `baseline_v2` 取**（`live_base()` `:129-169`）：优先取基线臂自身的精确 `picked`/`official` 读数，
  再退回 `baseline_v2` 摘要，最后退回模块常量 `BASE`（`:33-40`）⇒ 升轨后判据自动跟着新基线走。
- ⚠️ `judge()` 里 `t1`/`t3` **没有 `is not None` 保护**（`:178`/`:180`）⇒ 缺读数会抛异常而不是判不过；
  `collect()` 已保证这两项必有（`parse_report()` `:84-85` 过滤掉没有 t1/t3 的报告）。

### 2.2 🔴🔴 「至少一轴严格优于基线」（`strictly_better`）到底在哪

**这是本次调查最重要的发现 —— 四个消费者的口径并不一致。**

| 消费者 | 是否要求 strictly_better | 位置 | 结论 |
| --- | --- | --- | --- |
| `six_criteria_audit.py` | ❌ **完全没有** | 全文 grep `strictly_better`/`strict` **零命中** | 🔴 它的"可升轨"标签**最宽松** |
| `auto_promote_watch.py` | ✅ 有 | `_strictly_better()` `:386-400`，过滤 `:402-404` | 自动路径安全 |
| `trackb_core.py` | ✅ 有 | `qualifies()` `:155-156`（`if not win: return False`） | Track-B 路径安全 |
| `promote_candidate.py` | ⚠️ **只对非汤臂生效**，且 ⚠️ **无 t2 轴** | `_strictly_better` 定义 `:390-396`；使用点 `:399`（`not _is_soup`） | 🔴 **手工调它可绕过**；且比自动通道**少一轴** |

**`promote_candidate.py` 的关键逻辑**（`check_arm()` @ `:190-413`）：

```python
# :379-398
res["single_arm_strict"] = False
_is_soup = bool(st.get("synth") or (st.get("soup_members") or []))    # :380
_auth_axes = {...}                                                      # :382-389 六轴，>= 权威口径
_strictly_better = (nano > base.nano) or (nano4 > base.nano4) or (f1 > base.f1)
                    or (t1 > base.t1) or (t3 > base.t3)
                    or (t2_fp < base.t2_fp)                             # :390-396
# :399-412
if (not _is_soup) and all(_auth_axes.values()) and _strictly_better and c.get("ms") is True:
    res["single_arm_strict"] = True
    res["seed_replicated"] = True          # 免种子复现关
    c["t2fp"] = _auth_axes["t2fp"]
    res["ok"] = bool(c) and all(c.values())
```

而 `cmd_promote()` 的放行条件只有三条（`:456-493`）：
1. `r["ok"]`（= 六轴 + 板端，全 True）—— `:456-458`
2. `--require-board` 的三项附加检查 —— `:463-490`
3. `seed_replicated` 或 `--force` —— `:491-493`

⇒ 🔴 **对一个"汤臂（soup / synth）"，`promote_candidate.py` 本身不要求任何轴严格优于基线**：
只要六轴全 `≥`、板端 PASS、且**同族有 ≥2 个六项全过成员**（`:335`：`res["seed_replicated"] = bool(len(six_ok) >= 2)`），
就会被升轨 —— **零增量升轨 / 基线名通胀**（2026-11-18 曾 82 次重复升轨同一臂，事故记录见 `auto_promote_watch.py:356-361`）。
自动守望靠 `_strictly_better` 拦住了这条，但**手工调用 `promote_candidate.py` 没有这层保护**。

🔴🔴 **手工通道比自动通道严一轴（`t2` 轴缺失）**：

| 通道 | `_strictly_better` 位置 | 胜出轴 |
| --- | --- | --- |
| 自动（`auto_promote_watch.py`） | `:386-400` | **六轴**：nano / nano4 / f1 / t1 / t3 / **t2（T2 轨 F1）**（`:398`）＋ 另有 `t2_fp < 基线`（`:396`） |
| 手工（`promote_candidate.py`） | `:390-396` | **五轴**：nano / nano4 / f1 / t1 / t3 ＋ `t2_fp < 基线` ⇒ **没有 t2 轴** |

⇒ 🔴 **「仅 T2 轨 F1 严格优（t2 ↑）」的臂，自动通道会升，手工通道不会**。
⇒ 人工核对清单（§5.1 步骤 3）**必须自己补上 t2 轴**，否则会把这种臂误判为「不可升轨」；
反过来，手工通道放行的一切，自动通道都放行（自动更松 ⇒ 更易误升轨）。

**所以：`strictly_better` 的真正权威落点是 `auto_promote_watch.py:386-400`，
而不是 `six_criteria_audit.py`，也不是 `promote_candidate.py` 的通用路径。**

### 2.3 升轨的非精度条件（逐条 + 行号）

| # | 条件 | 位置 | 判定内容 |
| --- | --- | --- | --- |
| 1 | **六项全过** | `promote_candidate.py:290`（`res["ok"] = bool(c) and all(c.values())`） | 六轴 + 板端 |
| 2 | **板端 T4 PASS** | `promote_candidate.py:248,261-272` | 读台账 `board_pass_t4`（绝对 OR 同会话相对）；无读数 ⇒ `c["ms"]=False`（`:263-264`） |
| 3 | **板端判定新鲜度**（R3） | `promote_candidate.py:250-270` | `board_ref_trial != 当前基线 trial` ⇒ 判 `board_stale` ⇒ `c["ms"]=False`，要求**重测** |
| 4 | **--require-board** 三查 | `promote_candidate.py:463-490` | ① 无 `board_ms` ⇒ 拒；② `board_pass_t4` 为假 ⇒ 拒；③ 🔴 **R3b 新基线不得比旧基线慢**：`bms > 旧基线 ms_new_board` ⇒ 拒（`:480-490`） |
| 5 | **种子复现关**（普通训练臂） | `promote_candidate.py:342-349` | 同 `recipe_key` 组 `n ≥ 2` **且** 组均 nano **>** 基线组均 nano |
| 6 | **种子复现关**（抛光臂） | `promote_candidate.py:294-321` | 同一抛光配方作用在 **≥2 个不同锚点**（`n_anchor ≥ 2`），`polish_family_key()` `:58-70` |
| 7 | **种子复现关**（汤臂） | `promote_candidate.py:326-341` | 同 `soup_family` 成员数 ≥2 **且** 同族**六项全过成员 ≥2**（`soup_family_six_ok()` `:170-187`，以磁盘 `gate_report.json` 为权威） |
| 8 | **单臂严格支配通道**（训练臂/抛光臂专用，免第 5~7 条） | `promote_candidate.py:379-412` | 自身六轴全过 **且** 至少一轴严格优 **且** `c["ms"] is True`；**汤臂不走此通道**（`:380` `_is_soup`） |
| 9 | `--force` 可跳过第 5~7 条 | `promote_candidate.py:491-493` | ⚠️ 手工强推的开关 |

**家族 n≥4？** —— 那是 **`board_t4_watch.candidates()` 的排队权重**，不是升轨条件：
家族计数 @ `:628-639`，**n≥4 排序键 @ `:642-643`**（六项全过桶里"同家族六项全过臂 ≥4"的整体提前）。
**真正的升轨家族门槛是 `n ≥ 2`**（第 5~7 条），`auto_promote_watch.py` 的老通道才用 `n≥4 + 板卡 PASS`
（**n≥4 判定在 `:315-320`**（`:319` `if len(members) < 4: continue`）；
`:344-347` 是「把板卡 PASS 成员塞候选队列」的循环，**不是 n≥4 判定**）。

### 2.4 `promote_candidate.py --trial <N> --name <NAME>` 完整用法

**参数**（`main()` @ `:562-577`）：

| 参数 | 默认 | 含义 | 行号 |
| --- | --- | --- | --- |
| `--list` | — | 列出所有臂（按 nano 排序），标注六项 PASS | `:564`, `cmd_list()` `:416-437` |
| `--trial N` | — | 要升轨的 trial 号 | `:565` |
| `--name NAME` | `Baseline_v3.0` | 新基线名 | `:566` |
| `--dry-run` | — | **只预检不写台账** | `:567`, `:496-498` |
| `--force` | — | 允许未过种子复现关的臂强推 | `:568`, `:491-493` |
| `--require-board` | — | 强制要求板端 T4 有读数且 ≤ 19.989 ms | `:569-570`, `:463-490` |

```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

# 预检（只读）
$PY manu/pipeline/npu/promote_candidate.py --trial 1910071 --name Baseline_v3.2 --require-board --dry-run
# 真升轨
$PY manu/pipeline/npu/promote_candidate.py --trial 1910071 --name Baseline_v3.2 --require-board
```

**它会写哪些文件 / 改哪些台账字段**（`cmd_promote()` `:500-559`）：

| # | 写点 | 内容 | 行号 |
| --- | --- | --- | --- |
| 1 | **台账** `{ns.LEDGER}` = `/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json` | `baseline_v2` = 新基线摘要（`name/trial/epoch/ckpt/nano/micro/f1/nano4/t1/t2_fp/t3/ms_new_board/ms_clean_ref/promoted_at/previous_baseline`） | `:504-521`, `:543` |
| 2 | 同上 | `baseline_registry`：新条目插首位 `ACTIVE / OFFICIAL BASELINE`，旧 ACTIVE 降为 `SUPERSEDED / DEPRECATED BASELINE` | `:522-528` |
| 3 | 同上 | `beat_to_pass`：`ref`/`nano_min`/`f1_min`/`t1_min`/`t2_fp_max`/`t3_min` 全换成新基线 | `:529-534` |
| 4 | 同上 | 该臂 `is_baseline_v2=True` / `status="ACTIVE / OFFICIAL BASELINE"` / `baseline_name` / `beat_to_pass_ref`；**其他旧基线臂** `is_baseline_v2=False` + `status=SUPERSEDED` | `:535-542` |
| 5 | **升轨记录** `{ns.WORK}/promotion_<trial>.json` = `/mnt/data/siping/runs/npu20ms/nano_search/promotion_<N>.json` | `promoted_at/from/to/trial/epoch/checks/nano/nano4/f1/t1/t2_fp/t3/board_ms/seed_replicated` | `:544-551` |
| 6 | **终端输出**（不是写盘） | 提示手工回填 `nano_search.BASE_V2` 常量的完整代码块 | `:554-559` |

🔴 **`promote_candidate.py` 自己不会改任何源码、不会动 `/tmp/pycharm_project_10ae9e2e` 下的交付产物、
不会碰板卡上挂载的 `ir_uav_det_rk3588_v1.0.rknn`、不会动 `trial_0080` 冻结底座。**
但它**只改台账**，而台账的基线身份必须与**源码常量**一致（见 §2.7 守卫）⇒ **必须回填常量**。

### 2.5 `promote_ready_cmd.py`（只读命令生成器，166 行）

- **用途**：把「哪个臂 / 六项读数 / 差哪一关 / 复制粘贴跑哪条命令」直接印出来。
- 🔴 **只读铁律**（docstring `:19-22`）：只 `json.load`（台账）+ `glob`（门禁报告）+ 读文件，
  **不写台账、不写任何文件、不碰权重** ⇒ 流水线正在写台账时也能安全运行。
- **建议新基线名**：`next_baseline_name()` `:51-59` ⇒ 当前 `Baseline_v3.1` ⇒ **`Baseline_v3.2`**（尾号 +1）。
- **命令拼装**：`cmd_line()` `:62-71`。

```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

$PY manu/pipeline/npu/promote_ready_cmd.py          # 默认 top 3
$PY manu/pipeline/npu/promote_ready_cmd.py --top 6
```

输出结构（`:96-157`）：
- 六项全过臂（`npass==6`）按 `(nano, f1)` 降序 —— `:103-104`
- 5/6 近失臂 —— `:105-106`
- 每个冠军臂打印：六项读数 + 板端 T4 状态（**未测 / 过期 / 通过**三分支 `:137-143`）+ 种子复现状态（`:144-145`）
  + ① 预检命令 + ② 真升轨命令（`:150-153`）
- ⚠️ `--force` 会被自动加上（`:148`）当种子关未过；`--require-board` 仅在板端**已测且不 stale** 时加（`:149`）
- 尾部提示回填基线常量（`:159-160`）

🔴 **注意**：`promote_ready_cmd.py` 用的 `SIX_KEYS` 检查（`:133`）只覆盖六项，
**不含 `strictly_better`** ⇒ 它列出的"🏆 六项全过候选"**同样可能是基线克隆**（与 §2.2 同病）。
🔴 **活证据（2026-11-23 实测）**：`promote_ready_cmd.py --top 4` 输出的前 4 名
`1990027 / 1990054 / 1990055 / 1990080`，六项读数与基线**逐位相同**（仅 `t2_fp` 4 vs 3），
其中 `1990055` **就是基线本体** ⇒ 它列的命令**不可直接抄**。

### 2.6 `auto_promote_watch.py`（自动升轨守望，554 行）

**"剔除基线本体 / 纯打平"过滤逻辑**：

| 位置 | 内容 |
| --- | --- |
| `:362-366` | 取当前基线五轴值 `_bm = {nano, nano4, f1, t1, t3}` |
| `:378-384` | T2 参照取**基线臂自己的 `official`**（`arms[base_trial].official`）而不是 `baseline_v2` 摘要 —— 两者可能来自不同批次，差 ~6e-5 |
| **`:386-400`** | **`_strictly_better(r)`**：① `trial == base_trial` ⇒ False（**剔除基线本体**，`:387-388`）；② nano/nano4/f1/t1/t3 任一 `>` 基线 ⇒ True（`:389-394`）；③ `t2_fp < 基线 t2_fp` ⇒ True（`:396`）；④ `t2 > 基线 t2` ⇒ True（`:398`）⇒ **六个胜出轴：nano / nano4 / f1 / t1 / t3 / t2**（比 `promote_candidate._strictly_better` `:390-396` **多一轴 t2**） |
| **`:402-404`** | `cands = [c for c in cands if _strictly_better(c[0])]` + 打印剔除数 ⇒ **这就是防"基线名通胀"的那一刀**（2026-11-18 曾 82 次重复升轨同一臂，事故记 `:356-361`） |

**它调用 `promote_candidate` 的条件**：

1. 老通道：家族 `n≥4`（**判定在 `:318-320`**（`:319` `if len(members) < 4: continue`））
   + 板卡 T4 PASS（`:321-332`）；`:344-347` 是「把板卡 PASS 成员塞候选队列」的循环，**不是 n≥4 判定**
2. 单臂严格支配通道（`:428-459`）：
   - `_t2cap = max(base.t2_fp, 5)`（`:428`）
   - 跳过汤臂（`:435-436`）
   - 六轴全 `≥` 且 `_strictly_better` 且 **板端 `board_pass` 为真**（`:441-454`）
3. 逐个候选试（`:480-501`）：先 `--dry-run` 预检，通过后**不带 `--force`** 真升轨（`:486-495`）
4. 命名：v3.x ⇒ v3.(x+1)（`:465-477`）
5. 升轨成功后：回填常量（`:511`）→ 重启调度器（`:515`，需 `PROMOTE_RESTART_SCHEDULER=1`，`:258`）
   → 写 `promotion_ready.json`（`:522-523`）→ **追加 memory.md**（`:272-277`, `:524`）

```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
$PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "2026-11-23 21:00"
$PY -u manu/pipeline/npu/auto_promote_watch.py     # 只跑一轮
```

🔴 **常驻实例已在跑时，手工升轨必须先按 PID 停它、升轨后再重启**：
它会每 10 分钟自行「回填 + 升轨 + 写 memory + 重启调度器」，与手工 `promote_candidate.py`
**抢写同一份 24 MB台账** ⇒ 同一臂可能被升两次（2026-11-18 **82 次重复升轨**事故）；
且它内存里的 `ns.BASE_V2` 是**旧基线**，不重启会继续拿旧基线判。⇒ 命令见 §5.1 步骤 5 / 步骤 11。

### 2.7 🔴 `baseline_guard.py` —— 手工升轨的**隐形杀手**（必读）

`manu/pipeline/npu/baseline_guard.py`（常驻，每 60 s 一轮）：

- **真源 = 源码常量**，不是台账：`_read_consts()` `:70-109` 用 `ast` **实时解析 `nano_search.py` 的 `BASE_V2` 字面量**
  （`:57` `NS_SRC = ns.__file__` ⇒ 解析**实际被 import 的那个文件**），**绝不回退到 `ns.BASE_V2`**（`:112-121`，事故记 `:4-17`）
- 比较：`base_ok = (cur.trial == want.trial) and |cur.t1 − want.t1| < 1e-9`（`:196-197`）
- 不一致 ⇒ **把台账 `baseline_v2` 改回源码常量的值**（`:201-204`），并修 `baseline_registry`（`:205-209`）
- 副作用：`:210-211` `ns.save_ledger(led)`；`:213-217` 写 sidecar `.baseline_v2_meta.json`

**守卫解析的是哪个文件 —— `BASE_V2` 的权威位置与现值**：

| 项 | 位置 | 现值（回填前） |
| --- | --- | --- |
| `BASE_V2_CKPT` | `manu/pipeline/npu/nano_search.py:94` | `'/mnt/data/siping/runs/npu20ms_offcal/trials/trial_1990055/weights/epoch5.pt'` |
| `BASE_V2` | `manu/pipeline/npu/nano_search.py:95-101`（回填正则 `^BASE_V2_CKPT = .*?\nBASE_V2 = dict\(.*?^\)\n` 实测命中 **94-102**） | 见下方逐项 |
| `BASE`（六轴字典） | `manu/pipeline/npu/six_criteria_audit.py:33-40` | `nano/nano4/f1/t1/t3/t2_fp` 六个数 |

`BASE_V2` 现值（逐项）：
`name='Baseline_v3.1', trial=1990055, epoch=5, ckpt='/mnt/data/siping/runs/npu20ms_offcal/trials/trial_1990055/weights/epoch5.pt', nano=0.7242438563327032, nano4=0.6609168241965974, f1=0.8906832039622916, t1=0.8909238824339264, t1_fp=1473, t2=0.9931631171345595, t2_fp=3, t3=0.9488855869242199, t3_fp=76, ms_delivery_board=19.306, ms_new_board=19.682, ms_new_board_t31_ref=None, recipe='[SB3] soup(1608003@5:w0.6524,1612001@5:w0.0360,1530006@5:w0.2116,1623003@5:w0.1000)'`

守卫解析的**具体文件**由 `:57` `NS_SRC = abspath(ns.__file__)` 决定 ⇒ 解析的是「**实际被 import 的那一份**」，
不是硬编码路径。服务器日志实证 `NS_SRC=/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/nano_search.py`
⇒ 🔴 **回填必须落在权威根那一份**（`/tmp/pycharm_project_10ae9e2e` 是 symlink → `/mnt/data/siping/runs/npu20ms/nano_search/repo`，
inode 211703125）；SSHFS 导出根 `/tmp/pycharm_project_a7d966e6/manu/pipeline/npu/nano_search.py` 是**另一个 inode**
⇒ 改权威根**不会**自动同步过去 ⇒ 必须另做服务端 `cp -f`（见 §5.1 步骤 9）。
（远端 `/tmp/baseline_guard.py` 与仓库副本 md5 一致：`4e32fb02…`。）

⇒ 🔴 **回填窗口 = 60 秒**：`_read_consts()`（`:70-109`）**每轮都用 `ast` 重新解析、无缓存**
⇒ **回填后无需重启守卫**（下一轮自动生效）；但**必须在 60 s 内**完成回填（`time.sleep(60)` @ `:268`），
否则台账已被回滚成旧值（`:201-204`）、registry 也被修回旧基线（`:205-209`）。

⇒ 🔴 **手工升轨后如果不回填 `nano_search.py` 的 `BASE_V2` 块，`baseline_guard` 会在 60 s 内把基线回滚**（见上「回填窗口 = 60 秒」），
并连带把 T4 相对口径的容差参照改回旧基线（`:13-14` 记录了这个连带损伤）。

**回填怎么做**：`auto_promote_watch.backfill_constants(trial, epoch)` `:139-215`
（改 `nano_search.py` 的 `BASE_V2`/`BASE_V2_CKPT` 块 + `six_criteria_audit.py` 的 `BASE` 字典，
各留一份 `.bak_<STAMP>` 备份 `:170`/`:194`，并同步 watchdog 的 `code_md5` `:197-212`）。
`promote_trackb.py:176-177` 也调用它。

**核对**：
```bash
$PY manu/pipeline/npu/baseline_guard.py --status   # 只打印现状，不写盘（`:243-245`）
$PY manu/pipeline/npu/baseline_guard.py --once     # 跑一轮自检（`:246-250`）
```

### 2.8 `six_criteria_audit.py` 的 `--top` 与「可升轨」标签

**`--top` 参数**（`:190`，默认 25）：

- 只影响 **「最接近候选榜 TOP N」** 的打印条数（`:231` `for r in rows[: a.top]`），排序键 = `(-npass, -nano)`（`:199`）
- **不影响**「六项全过」清单（`:201-227`，**全量打印**）
- 🔴 `--top` **不是**"只审前 N 个臂" ⇒ 传 `--top 12` 仍然会**全盘扫**（`collect()` `:94-123` 扫整个 `gate/` 目录 + 24 MB 台账）

**「可升轨」标签**（`:224`）：

```python
tag = ("  ← 基线本体" if r["trial"] == _btrial else "  🔴 **新候选（可升轨）**")
```

🔴🔴 **这个标签比 `promote_candidate.py` 宽松得多**：

| 维度 | `six_criteria_audit.py:224` | `promote_candidate.py` |
| --- | --- | --- |
| 六项全 `≥` 基线 | ✅ 要求 | ✅ 要求（`:290`） |
| **至少一轴严格优于基线** | ❌ **不要求** | ⚠️ 只对非汤臂要求（`:399`） |
| 板端 T4 PASS | ❌ **不要求** | ✅ 要求（`:261-272`） |
| 种子复现关 | ❌ **不要求** | ✅ 要求（`:342-349`） |

⇒ **差异定位**：`six_criteria_audit.py:224` 是**纯精度标签**；
`promote_candidate.py:290 + :399 + :261-272 + :342-349` 才是**完整升轨判据**。

⚠️ 实证：`archive/19_主档正文_§283-298_20261120-20261123.md:1024`（§288） 记「有门禁的 2,250 个 ⇒ **六项全过 57 个，全部是基线逐位克隆 ⇒ 零增量**」
—— 这 57 个在 `six_criteria_audit.py` 输出里**全部带「🔴 新候选（可升轨）」**，但**一个都不能升轨**。

🔴🔴 **活证据（2026-11-23 实测）**：`promote_ready_cmd.py --top 4` 输出的前 4 名
`1990027 / 1990054 / 1990055 / 1990080`，六项读数与基线**逐位相同**（仅 `t2_fp` 4 vs 3），
其中 `1990055` **就是基线本体** ⇒ 这个标签会把**基线克隆臂 / 基线本体**一并标成「可升轨」。

⇒ 🔴 **不要信「🔴 新候选（可升轨）」这个标签**：它**只表示「六轴全 ≥ 基线」**，
**不查板端、不查种子关、也不查 `strictly_better`**（全文 grep `strictly_better|strict` = **0 命中**）。
**唯一可信的判据是 §5.1 步骤 3 的人工严格优核对 + 步骤 6 的 `--dry-run` 预检。**

**另一条升轨路径**（备用）：`manu/pipeline/npu/promote_trackb.py`（Track-B 路）
- `--t2-max`（默认 `T2_USER=5`，`trackb_core.py:83`）、`--min-family`（默认 2）、`--dry-run`（`:199-206`）
- 判据 `trackb_core.qualifies()` `:150-174`：**五轴全 `≥`** **且 `win` 非空（= 至少一轴严格优）** **且 `t2_fp ≤ t2_max`**
- 🔴 它会额外写 `baseline_v2_strict_backup`（`promote_trackb.py:132-133`）⇒ **有严格轨一键回退点**

### 2.9 板端 T4 口径速查（升轨时怎么读台账字段）

| 项 | 位置 | 口径 |
| --- | --- | --- |
| 样本数 | `board_t4_watch.py:72` / `:73` | `MIN_SAMPLES = 5`（攒够 5 个样本才判）、`MAX_SAMPLES = 9`（上限）⇒ 判定用**中位数**（`pool_median()` `t4_criterion.py:133-139`） |
| 判定入口 | `board_t4_watch.py:434` | `T4.decide(ms, diff_ms, tol)` ⇒ `pass_t4` = **绝对 OR 同会话相对**（`pass_t4_abs` / `pass_t4_rel` 分别落 `:438-439`） |
| 绝对口径 | `t4_criterion.py:105` | `ms ≤ 19.989`（`LIMIT_MS` @ `:38`） |
| 相对口径 | `t4_criterion.py:88-94` | `diff ≤ max(0.05, 19.989 − base_ref)`；**参照臂 = 当前基线臂**（`session_ref_ms()` `:71-85`），必须与候选**同一次板卡会话** |
| 无绝对读数 | `t4_criterion.py:108` | 🔴 **不判定**（"只有差值没有读数"不是放行理由） |
| `board_limit_ms` 来源 | `board_t4_watch.py:494` | = `LIMIT_MS`（`:46`）→ `t4_criterion.py:38` = **19.989**（= `trial_0031` 19.597 × 1.02，注释 `:37`） |

### 2.10 家族通道 / 汤臂规则（哪条通道管哪种臂）

| 项 | 位置 | 规则 |
| --- | --- | --- |
| 家族通道（老通道） | `auto_promote_watch.py:318-320` | `len(members) < 4` ⇒ **跳过**（`:319` `if len(members) < 4: continue`） |
| 家族通道还要板卡 PASS | `auto_promote_watch.py:321-332` | 家族成员里**必须有板卡 T4 PASS**（`with_board`），否则 `continue` |
| 家族键 | `auto_promote_watch.py:61-70` | `family_key.arm_family_key`（**唯一权威口径**，委托给 `manu/pipeline/npu/family_key.py`） |
| 种子关（家族内） | `auto_promote_watch.py:309-313` | `seed_ok` = 同 `soup_family` 标签下**六项全过臂 ≥ 2** |
| **汤臂走家族通道** | `auto_promote_watch.py:435-436` | 单臂严格支配通道**显式排除汤臂**：`if _st.get("synth") or (_st.get("soup_members") or []): continue  # 汤臂走老通道（家族 n≥4）` |
| 手工通道的家族门槛 | `promote_candidate.py:326-341`（汤臂）/ `:342-349`（训练臂）/ `:294-321`（抛光臂） | **n ≥ 2**（比自动通道的 n≥4 宽）；训练臂另加「组均 nano **>** 基线组均 nano」 |

---

## ③ nanobench 的 `nano` / `nano4` / `micro` / `small` / `regular` / `f1` / `f1_4`

### 3.1 尺寸分桶（物理像素，原图 640×512）

**真源：`manu/evaluation/tiny_target_subset.py`**

| 桶 | 定义 | 行号 | T1 目标数（占比） |
| --- | --- | --- | --- |
| `nano` | `max(w,h) ≤ 4px` **或** `area ≤ 16px²` | `:10`, `bucket_of()` `:73-81` | 4232（16.8%） |
| `micro` | `4px < max(w,h) ≤ 8px` | `:11`, `:78` | 3283（13.1%） |
| `small` | `8px < max(w,h) ≤ 16px` | `:12`, `:80` | 11796 |
| `regular` | `max(w,h) > 16px` | `:13`, `:81` | 5800 |

分位数参考（`:14-15`, `:43`）：max-side p1=2.5 / p10=4.0 / p30=8.8 / p50=12.5 / p90=26.3 px。
`BUCKETS = ("nano","micro","small","regular")` @ `:54`。

### 3.2 🔴 `nano` / `nano4` 的真实含义（极易误解）

**`nano` 不是「≤8px 召回」**，而是「**≤4px 桶**的召回，在 **8px 匹配容差**下测的」。
**`nano4` 是同一个 ≤4px 桶，在 **4px 匹配容差**下测的。**

证据链（`manu/pipeline/npu/nano_search.py:720-750`）：

```python
res = RA.nanobench(pkl, label)
p8 = res["peak_tol8"]; p4 = res["peak_tol4"]      # :727-728
rb  = p8["recall_by_bucket"]; rb4 = p4["recall_by_bucket"]   # :729-730
return {"f1": p8["f1"], "th": p8["th"], "fp": p8["fp"], "fn": p8["fn"],
        "nano": rb["nano"], "micro": rb["micro"], "small": rb["small"],
        "regular": rb["regular"],
        "nano4": rb4["nano"], "micro4": rb4["micro"], "f1_4": p4["f1"]}   # :731-734
```

| 字段 | 精确含义 | 来源 |
| --- | --- | --- |
| `nano` | `peak_tol8.recall_by_bucket["nano"]` = **≤4px 桶**召回，**8px** 容差 | `nano_search.py:732` |
| `nano4` | `peak_tol4.recall_by_bucket["nano"]` = **≤4px 桶**召回，**4px** 容差 | `nano_search.py:734` |
| `micro` / `small` / `regular` | 同 `nano` 的容差（8px），不同尺寸桶的召回 | `nano_search.py:732-733` |
| `micro4` | 4px 容差下的 micro 桶召回 | `nano_search.py:734` |
| `f1` | `peak_tol8.f1` = **全量 T1 的 F1**（在 8px 容差下扫阈值取峰值），**不是某个尺寸桶的 F1** | `nano_search.py:731` |
| `f1_4` | `peak_tol4.f1` = 同一全量 F1，但匹配容差收紧到 **4px**（更严格） | `nano_search.py:734` |

**双容差口径**：`tiny_target_subset.py:21`（"子集指标一律报双容差（4px / 8px）"）、`:44-45`。
容差实现 `eval_cache(cache, "nogmc", tols=(4.0, 8.0), img_w=640, img_h=512)` @ `route2_acceptance.py:86`。
匹配算法 `match_at()` @ `tiny_target_subset.py:73-98`（按分数降序贪心、tol 内取最近**未匹配** GT，**与权威 T1 评估器逐位一致**，`:17-20`）。

⚠️ **memory 里的旧表述「nano (≤8px)」是不准确的**（例如 `archive/16_主档正文_§236-257_20261118-20261119.md:259`（§240））。
正确表述应是「nano = ≤4px 桶召回 @8px 容差」。**代码口径为准。**

### 3.3 跑在哪个数据集上

| 项 | 值 | 位置 |
| --- | --- | --- |
| **NanoBench 数据集** | T1（Anti-UAV 官方域）**no-GMC** 口径，val **31613 帧 / 25111 个目标** | `tiny_target_subset.py:34`, `:38`；`route2_acceptance.py:21` |
| 数据 yaml | `/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml` | `route2_acceptance.py:39` |
| 缓存目录 | `/mnt/data/siping/runs/tiny_ref/`（**`tiny_ref` = 参照卡目录**） | `route2_acceptance.py:38`（`TINY`）；`nano_search.py:877` |
| 建缓存命令 | `manu/inference/cache_trial0474_inferences.py --imgsz 640 --stride 2 --batch 32 --conf-thresh 0.02 --top-k 100 --ablate-p0 [--no-peak-attrs]` | `route2_acceptance.py:69-72`；`nano_search.py:878` |
| 语义声明 | **GMC 与 no-GMC 缓存严禁混用** | `tiny_target_subset.py:22-23`；`route2_acceptance.py:19` |

**与官方门禁三轨是不是同一批图？—— 🔴 不是。**

| | NanoBench（`nano`/`nano4`/`f1`） | 官方三轨 T1/T2/T3 |
| --- | --- | --- |
| 域 | Anti-UAV **T1 val**（31613 帧） | T1 = Anti-UAV 官方大盘（GMC 语义）；T2/T3 = **龙泉山**序列 |
| 语义 | **no-GMC**（GMC 已被消融） | **GMC**（`--ablate-p0` 关掉 P0） |
| 数据源 | `uav_gmc_median_nogmc` | T1: `uav_gmc_median`；T2/T3: `longquanshan_ft_gmc_median_official` |
| 缓存脚本 | `cache_trial0474_inferences.py` | T1 同脚本；T2/T3: `manu/data/multiscale_heatmap_cache.py`（多尺度热图） |
| 匹配 | 中心欧氏距离 8px/4px | T1 同 8px 点对点；T2 point-in-bbox；T3 同 T2 |

证据：`auto_epoch_gate_check.py:61-62`（`FT_ROOT` / `UAV_DATA`）、`:65-76`（GOLDEN 7 条 + BLIND 1 条）、
`:427-460`（T2/T3 用 `multiscale_heatmap_cache.py --scales 160,80 --ablate-p0`）、
`:478-497`（`system_sota_longquan.py --no-pipeline`）。
⇒ **同一个臂的 `f1`（NanoBench no-GMC 全量 F1）与 `t1`（官方 GMC 轨 F1）不是同一个数，也不能互相替代。**

---

## ④ 官方门禁三轨（track1_uav / track2_golden / track3_blind）

**产出脚本：`manu/training/auto_epoch_gate_check.py`（586 行）**，产物 `{out_dir}/gate_report.json`（`:569`）。
调用方：`nano_search.run_official_gate()` `:983-1033`（`:1010-1012` 组命令），
输出目录 `{ns.ROOT}/gate/NANO_{trial}_ep{ep}_official`（`:987`）。

### 4.1 三轨定义与红线

| 轨 | 数据 | 脚本 | 判据 | 行号 |
| --- | --- | --- | --- | --- |
| **track1_uav** | Anti-UAV 官方大盘（GMC），31613 帧 / 25111 GT | `cache_trial0474_inferences.py` → `eval_arm_pc_gmc_ab.py --tol 8.0` | `F1 ≥ min_f1_required`，**一票否决** | `:344-402` |
| **track2_golden** | 龙泉山黄金池 **7 条**序列（`GOLDEN` `:65-75`） | `multiscale_heatmap_cache.py --ablate-p0` → `system_sota_longquan.py --no-pipeline` | `F1 ≥ min_track2_f1`（默认 0.985）**且** `FP ≤ max_track2_fp`（默认 36 = 基线 26 + 10），**一票否决** | `:478-536` |
| **track3_blind** | 龙泉山盲测 **1 条**（`VIDEO00007_19700101_001324`，`:76`） | 同 T2 | `F1 ≥ 基线`（观察轨），**默认不熔断**（`--track3-strict` 才 veto，`:543`） | `:537-550` |

**T2 口径 = point-in-bbox**（落框内即 TP），见 docstring `:11-13`。

### 4.2 track1_uav：`baseline_peak` vs `finetune_peak` vs `delta.negative_bands/bands`

`auto_epoch_gate_check.py:344-402`：

- **A/B 双臂评测**：基线臂（冻结缓存 `--baseline-cache`）作 `arm_a`，新臂（本次权重重算的缓存）作 `arm_b`
  （`:348-367`）。基线臂**复用不重算**（docstring `:9-10`）。
- `baseline_peak` = `ab["arm_a"]["peak"]`（`:381`）⇒ 基线臂在 **15 档阈值**上 F1 最大的那一档
- `finetune_peak` = `ab["arm_b"]["peak"]`（`:380`）⇒ 新臂同理
- `delta` = `ab["delta"]`（`:386`），由 `eval_arm_pc_gmc_ab.py:199-206` 产出：

```python
"delta": {
    "peak_f1": bb["f1"] - ba["f1"],   # :200
    "mean_f1": float(np.mean(dfs)),   # :201
    "min_f1":  float(min(dfs)),       # :202
    "max_f1":  float(max(dfs)),       # :203
    "negative_bands": int(neg),       # :204
    "bands": len(ths),                # :205
}
```

**🔴「15 个 band」是什么分法？—— 是 15 个**置信度阈值**档，不是尺寸档。**
- `DEFAULT_THS = [round(0.05 * i, 2) for i in range(1, 16)]` @ `eval_arm_pc_gmc_ab.py:34`
  ⇒ **0.05, 0.10, 0.15, …, 0.75**（步长 0.05，共 15 档）
- 每档都跑一遍全量 `match_at`（`:73-98`）算 (tp, fp, fn) → (p, r, f1)（`:101-119`）
- `negative_bands` = **这 15 档里 finetune 的 F1 比 baseline 低（`df < 0`）的档数**（`:158-164`）
- `peak` = 15 档中 F1 最大的那一档（`:169-170`）
- ⚠️ `delta` / `negative_bands` **只是信息项**，`auto_epoch_gate_check.py` 的 T1 判据**只看 `finetune_peak.f1`**（`:389`）

**`min_f1_required` 的来源**（`:387`）= `args.min_track1_f1`，默认 = `BASE["track1_uav"]["f1"] = 0.905736`（`:54-55`, `:154`）。

🔴🔴 **重大口径提醒**：
- 生产链路的门禁命令（`nano_search.py:1010-1012`）**没有传 `--min-track1-f1`** ⇒ `min_f1_required = 0.905736`
- 而**当前基线的 T1 只有 0.8909238824339264 < 0.905736** ⇒ **`gate_report.json` 的 `verdict` 对当前基线自己都是 FAIL**
- 代码里已明确写了这句警告：`board_t4_watch.py:158`
  「⚠️ gate_report 的 verdict 用的是冻结基线 0.905736（连当前基线都过不了）⇒ **不能用它判交付**」
- ⇒ **升轨判据只看 `finetune_peak.f1` 这个数（台账 `official.t1`），绝不能看 `verdict` 字段。**
- （其它脚本确实传了更贴合的 `--min-track1-f1`，如 `manu/training/gate_struct_soups.sh:89-103`、
  `manu/training/seal_candidate.py:303` 用 `0.8958967482400269`；生产 nano 链路没沿用。）

> 🔴 **不要信 `gate_report.json` 的 `verdict`（不可用于升轨判断）**
>
> 链条：`manu/training/auto_epoch_gate_check.py:54-55`（`BASE["track1_uav"]["f1"] = 0.905736`）
> → `:154`（`--min-track1-f1` 默认值）→ `:387`（写入报告的 `min_f1_required`）。
> 生产门禁命令 `nano_search.py:1010-1012` **不传** `--min-track1-f1`
> ⇒ 阈值恒为 0.905736 ⇒ **连基线自己 0.8909238824339264 都过不了** ⇒ `verdict` 对基线本体就是 FAIL。
> ⇒ 升轨只看 `tracks.track1_uav.finetune_peak.f1`（`nano_search.py:1019-1024`）；
> `board_t4_watch.py:158` 已有同款警告。

### 4.3 `official_ok` / `deliver_ok` / `veto_failures` 分别由什么决定

| 字段 | 位置 | 判据 |
| --- | --- | --- |
| **`official_ok`** | `nano_search.py:1029-1030` | `t1 ≥ DELIVER["t1_ge"] (0.885997)` **且** `t2_fp ≤ DELIVER["t2_fp_le"] (36)` **且** `t3 ≥ DELIVER["t3_ge"] (0.948330)`；`DELIVER` 定义 `:177-178` |
| **`deliver_ok`** | `nano_search.py:1031-1032` | `official_ok` **且** `t2_fp ≤ DELIVER["t2_fp_doc"] (20)`（**任务书级** T2 虚警红线，比官方 36 更严） |
| **`veto_failures`** | `auto_epoch_gate_check.py:553-555` | 所有 `gates` 里 `veto=True` 且 `pass=False` 的轨名（`track1_uav` / `track2_golden` 恒 veto；`track3_blind` 仅 `--track3-strict`） |
| **`verdict`** | `auto_epoch_gate_check.py:563-567` | 有 `infra_failures` ⇒ **FAIL**（基础设施失败不能当达标，`:557-562`）；否则 `veto_failures` 非空 ⇒ FAIL；无 soft_fail ⇒ PASS；有 soft_fail ⇒ PASS_WITH_WARNING |
| **`observation_warnings`** | `auto_epoch_gate_check.py:554-556` | 非 veto 轨失败（默认只有 track3_blind） |
| **`gates`** | `auto_epoch_gate_check.py:390-395`, `:524-530`, `:539-544` | 每轨一条：`rule` / `value`(或 `f1`,`fp`) / `pass` / `veto` |
| **退出码** | `auto_epoch_gate_check.py:578-582` | `0`=通过（可带观察告警）；`1`=**指标未达标**（需 `--strict-gate`）；`3`=**基础设施失败**（推理 OOM / 缓存重建失败 / 评测脚本报错）—— 与质量失败严格区分 |

🔴 **六项判据用的是 `finetune_peak.f1` / `track2_golden.f1` / `track3_blind.f1` 三个原始数**，
解析在 `nano_search.py:1019-1026` 与 `six_criteria_audit.py:81-89`（两处独立实现，**但取的是同一份 `gate_report.json`**）。
⚠️ `night_train_axis.py:1351-1354` 明确警告：**不能用门禁自带的 `official_ok` 当 `six_ok`** ——
它 `t2_fp ≤ 36`，会把 `t2_fp=3` 的臂误标成六项全过。

---

## ⑤ 给主代理的结论

### 5.1 把一个「六项全过 + 至少一轴严格优于基线」的臂升轨成 `Baseline_v3.2`

> 🔴 **这是唯一操作顺序（0 → 11 不可跳步）**。每步都带判据，判据不满足就**停**，不要往下走。

**步骤 0 · 环境**

```bash
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
N=<TRIAL>            # 要升轨的 trial 号
EP=5
```

**步骤 1 · 只读审计（确认 6/6 臂清单 + 最接近候选榜）**

```bash
$PY manu/pipeline/npu/six_criteria_audit.py --top 12
```

判据：`--top` **只裁剪 TOP 榜打印**（`six_criteria_audit.py:231`），**不裁剪扫描范围**（仍全盘扫 `gate/` + 24 MB 台账）。
🔴 输出里的「🔴 新候选（可升轨）」**不代表可升轨**（见 §2.8）⇒ 必须自己做步骤 3。

**步骤 2 · 只读命令生成器（拿现成的预检/升轨命令 + 该臂差哪一关）**

```bash
$PY manu/pipeline/npu/promote_ready_cmd.py --top 6
```

判据：它会打印「建议新基线名：Baseline_v3.2」+ 每个臂的六项 / 板端三分支 / 种子复现 / 预检命令。
🔴 **它列的「🏆 六项全过候选」实测全是基线克隆，含基线本体 `1990055` ⇒ 命令不可直接抄**（见 §2.5）。

**步骤 3 · 人工严格优核对（自己逐轴与基线逐位比）**

```bash
$PY - "$N" <<'PYEOF'
import json,sys
N=sys.argv[1]
L=json.load(open('/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json'))
A=L['arms']; B=A[str(L['baseline_v2']['trial'])]
p,o=A[N].get('picked') or {}, A[N].get('official') or {}
bp,bo=B.get('picked') or {}, B.get('official') or {}
g=lambda d,k: float(d.get(k) or 0)
ax={k:(g(p,k),g(bp,k)) for k in ('nano','nano4','f1')}
ax.update({k:(g(o,k),g(bo,k)) for k in ('t1','t3','t2')})
ax['t2_fp']=(int(o.get('t2_fp') or 999), int(bo.get('t2_fp') or 999))
print('arm %s  vs  基线 %s (%s)'%(N,L['baseline_v2']['trial'],L['baseline_v2']['name']))
for k in ('nano','nano4','f1','t1','t2','t2_fp','t3'):
    v,b=ax[k]; print('  %-6s %-22r 基线 %-22r Δ=%+.8f'%(k,v,b,v-b))
w=[k for k in ('nano','nano4','f1','t1','t3','t2') if ax[k][0]>ax[k][1]+1e-12]
if ax['t2_fp'][0]<ax['t2_fp'][1]: w.append('t2_fp')
print('  严格胜出轴:', w if w else '无 ⇒ 纯打平，禁止升轨')
print('  板端:', {k:A[N].get(k) for k in ('board_ms','board_pass_t4','board_ref_trial','board_ref_ep','board_ms_ref','board_diff_ms','board_limit_ms','board_measured_at')})
PYEOF
```

判据（**三条全满足**才继续）：
1. 「严格胜出轴」**非空**；
2. `t1 ≥ 基线`（六项门槛：nano / nano4 / f1 / t1 / t3 ≥ 基线，`t2_fp ≤ max(基线 t2_fp, 5)`）；
3. `board_pass_t4=True` **且** `board_ref_trial == 1990055`（非 stale）。

🔴 **必须自己补 t2 轴**：自动通道 `auto_promote_watch._strictly_better()`（`:386-400`）有**六个**胜出轴
（nano / nano4 / f1 / t1 / t3 / **t2**，`:398`）＋ 另有 `t2_fp < 基线`（`:396`）；
而手工 `promote_candidate._strictly_better`（`:390-396`）**没有 t2 轴**
⇒ **手工通道比自动通道严一轴** ⇒ 漏了 t2 会把「仅 T2-F1 严格优」的臂误判为不可升轨（见 §2.2）。
实测样例：`2110001` 胜出 `nano/nano4/f1/t3`，但 **t1 −7.85e-5** ⇒ 5/6 ⇒ **不可升轨**；
`1990055`（基线本体）判「无 ⇒ 纯打平」。

**步骤 4 · 板端：先确认守望在跑，别开第二实例**

```bash
ps -eo pid,etimes,args | grep '[b]oard_t4_watch.py' && tail -3 /mnt/data/siping/runs/npu20ms/nano_search/board_t4_watch.log
```

🔴 **不要再手拉 `board_t4_watch.py --once`**：常驻实例 PID 777962 已在跑，而 `board_busy()`（`board_t4_watch.py:706-726`）
**只 SSH 查板卡**上的进程，**查不到本机第二实例** ⇒ 双实例会并发导 RKNN / 上传 / 测速
⇒ 污染 `measured.json` / `t4_samples.json`。
确需全量复判用（🔴 **没有 `--trial` 参数，只能全量**；`--dry` 只看不写）：

```bash
$PY -u manu/pipeline/npu/board_t4_rejudge.py --dry
```

判据：该臂 `board_pass_t4=True` 且 `board_ref_trial == 1990055`（步骤 3 已看过）。

**步骤 5 · 停自动升轨守望（按 PID，禁用 `pkill -f`）**

```bash
for p in $(ps -eo pid,args | grep '[a]uto_promote_watch.py' | awk '{print $1}'); do kill $p && echo "stopped auto_promote_watch $p"; done
```

判据：🔴 **必须先停**。常驻实例（`--watch --interval 600 --deadline 2026-11-23 21:00`）会每 10 分钟自行
「回填 + 升轨 + 写 memory + 重启调度器」，与手工 `promote_candidate.py` **抢写同一份 24 MB 台账**
⇒ 同一臂可能被升两次（2026-11-18 有过 **82 次重复升轨**事故）。
🔴 铁律禁用 `pkill -f`（会误杀本进程 / 其它同名进程）⇒ 一律**按 PID 停**。

**步骤 6 · 升轨预检（只读台账，不写任何东西）**

```bash
$PY manu/pipeline/npu/promote_candidate.py --trial $N --name Baseline_v3.2 --require-board --dry-run
```

判据：**7 行 `[✔]` 全绿**（nano / nano4 / f1 / t1 / t2fp / t3 / **ms**），
且「种子复现」行**不是「未复现」**（若 `seed_note` 含「单臂严格支配路径」亦安全）；
否则加 `--force` = **人工承担风险**。

**步骤 7 · 真升轨**

```bash
$PY manu/pipeline/npu/promote_candidate.py --trial $N --name Baseline_v3.2 --require-board
$PY -c "import json;print(json.load(open('/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json'))['baseline_v2']['trial'])"
```

判据：输出「✔ 台账已升轨 → /mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json」
+「✔ 升轨记录 → …/promotion_$N.json」+ 末尾打印待回填的 `BASE_V2` 块；
且第二条命令打印的 trial **== `$N`**。

**步骤 8 · 回填源码常量（权威根，两个文件）**

```bash
$PY -c "from manu.pipeline.npu import auto_promote_watch as A; A.backfill_constants($N, $EP); print('backfill done')"
```

判据：`nano_search.py` 的 `BASE_V2`（:94-102）trial 变成 `$N`，
`six_criteria_audit.py` 的 `BASE`（:33-40）六轴同步；两份 `.bak_<STAMP>` 备份已生成
（`auto_promote_watch.py:170` / `:194`）。
🔴 **必须在 60 秒内完成**（守卫每 60 s 一轮：`baseline_guard.py:268`），否则台账被回滚成旧值。

**步骤 9 · 同步 SSHFS 导出根（服务端 `cp -f`）**

```bash
cp -f /tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/nano_search.py /tmp/pycharm_project_a7d966e6/manu/pipeline/npu/nano_search.py && cp -f /tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/six_criteria_audit.py /tmp/pycharm_project_a7d966e6/manu/pipeline/npu/six_criteria_audit.py
md5sum /tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/nano_search.py /tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/six_criteria_audit.py /tmp/pycharm_project_a7d966e6/manu/pipeline/npu/nano_search.py /tmp/pycharm_project_a7d966e6/manu/pipeline/npu/six_criteria_audit.py
```

判据：🔴 **权威根改完，导出根不会自动同步** ——
`/tmp/pycharm_project_10ae9e2e` 是 symlink → `/mnt/data/siping/runs/npu20ms/nano_search/repo`（inode 211703125），
而 `/tmp/pycharm_project_a7d966e6/manu/pipeline/npu/nano_search.py` 是**另一个 inode**（SSHFS 导出）
⇒ **必须服务端 `cp -f`**（🔴 禁 rsync 的临时文件 + rename：会换 inode，打断正在跑的 import）。
判据：两侧 `md5sum` **一致**（回填前实测两侧 `nano_search.py` = `8d7382d5…`、`six_criteria_audit.py` = `e4f459e9…`）。

**步骤 10 · 验证基线一致性（只读）**

```bash
$PY manu/pipeline/npu/baseline_guard.py --status
```

判据：「源码常量」的 trial **==**「台账基线」的 trial **==** `$N`，
且 registry 首条 = `Baseline_v3.2 … ACTIVE / OFFICIAL BASELINE`。
🔴 **不用重启守卫**（`_read_consts()` 每 60 s 用 `ast` 重读源文件，`baseline_guard.py:70-109`）
⇒ 回填后下一轮自动生效；但**必须**在 60 s 内完成回填，否则台账会被回滚（旧值）。

**步骤 11 · 重启升轨守望（按 PID）**

```bash
cd /mnt/data/siping/runs/npu20ms/nano_search/repo && nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "2026-11-23 21:00" >> /mnt/data/siping/runs/npu20ms/nano_search/auto_promote_watch.log 2>&1 & sleep 2; ps -eo pid,args | grep '[a]uto_promote_watch.py'
```

判据：🔴 **必须重启**：它内存里的 `ns.BASE_V2` 是**旧基线** ⇒ 不重启会拿着旧基线继续判、继续升轨。

### 5.2 升轨的**不可逆 / 有风险**动作

| 风险项 | 说明 | 位置 | 是否需人工确认 |
| --- | --- | --- | --- |
| 🟢 **不会**动交付产物 | `promote_candidate.py` 不写 `/tmp/pycharm_project_10ae9e2e` 的任何交付文件，**不碰板卡挂载的 `ir_uav_det_rk3588_v1.0.rknn`，不动 `trial_0080` 冻结底座** | `promote_candidate.py:500-559` | — |
| 🔴 **改源码常量** | `auto_promote_watch.backfill_constants()` **重写 `nano_search.py` 的 `BASE_V2` 块 + `six_criteria_audit.py` 的 `BASE` 字典**（各留 `.bak_<STAMP>`） | `auto_promote_watch.py:139-196` | ✅ **必须人工确认**（手改两个文件） |
| 🔴 **回滚陷阱** | 不回填常量 ⇒ `baseline_guard` 每 60 s 把台账基线**改回源码常量** ⇒ 升轨被静默撤销 | `baseline_guard.py:196-211` | ✅ 升轨后**立刻** `--status` 核对 |
| 🔴 **基线名通胀** | 手工调 `promote_candidate.py` 升**汤臂克隆** ⇒ 零增量也升轨（`promote_candidate` 本身不查 strictly_better） | `promote_candidate.py:326-341` vs `:399` | ✅ **升轨前必须自己确认至少一轴严格优** |
| 🔴 **守望抢写台账** | 常驻 `auto_promote_watch.py`（`--watch --interval 600`）会自行「回填 + 升轨 + 写 memory + 重启调度器」⇒ 与手工 `promote_candidate.py` **抢写同一份 24 MB 台账** ⇒ 同一臂可能被升两次 | `auto_promote_watch.py:272-277`, `:480-501`, `:511-524` | ✅ **升轨前按 PID 停守望**（§5.1 步骤 5）；**升轨后必须重启**（步骤 11，否则它内存里是旧 `ns.BASE_V2`） |
| 🔴 **导出根不同步** | 权威根（`/tmp/pycharm_project_10ae9e2e` → symlink → `.../nano_search/repo`）与 SSHFS 导出根（`/tmp/pycharm_project_a7d966e6`）是**两个不同 inode** ⇒ 只回填权威根 ⇒ 导出根仍是旧常量 | `baseline_guard.py:57`（`NS_SRC`） | ✅ 必须服务端 `cp -f`（§5.1 步骤 9）+ 两侧 `md5sum` 一致 |
| 🟡 **板卡双实例污染** | `board_t4_watch.board_busy()` **只 SSH 查板卡**上的进程，**查不到本机第二实例** ⇒ 并发导 RKNN / 上传 / 测速会污染 `measured.json` / `t4_samples.json` | `board_t4_watch.py:706-726` | ✅ **不要手拉 `--once`**；先 `ps` 确认守望在跑（§5.1 步骤 4） |
| 🟡 **交付时延不得回退** | `--require-board` 内含 R3b 闸：新基线板端必须 **≤ 旧基线** `ms_new_board`，否则拒 | `promote_candidate.py:480-490` | — （自动拒绝） |
| 🟡 **板端判定过期** | `board_ref_trial != 当前基线` ⇒ 判 stale ⇒ 拒绝（要求重测） | `promote_candidate.py:250-270` | ✅ 需重测板卡 |
| 🟡 **`--force`** | 绕过种子复现关 ⇒ 单臂彩票也能升轨 | `promote_candidate.py:491-493` | ✅ **必须人工确认** |
| 🟡 **写 memory.md** | `auto_promote_watch` 升轨后**追加** `manu/memory/memory.md` | `auto_promote_watch.py:272-277`, `:524` | ⚠️ 与本报告并发写同一文件，注意冲突 |
| 🟡 **重启调度器** | `auto_promote_watch` 会重启 `adaptive_runner`（需 `PROMOTE_RESTART_SCHEDULER=1` 才允许，`:258`） | `auto_promote_watch.py:217-269` | ⚠️ 会打断正在跑的搜索 |
| 🟡 **台账并发写** | 升轨与门禁通道/板卡守望并发写同一份 24 MB 台账 ⇒ 走 `_merge_disk_ledger`（`:413` 的字段级合并补丁已上） | `nano_search.py:413-423` | ⚠️ 避开正在大批量写台账的窗口 |
| 🟢 **不会**改 `trial_0080` 交付链 | 交付冻结底座与本链路完全独立 | `archive/19_主档正文_§283-298_20261120-20261123.md:1054-1055`（§288） | — |

### 5.3 `six_criteria_audit.py` 的「可升轨」标签比 `promote_candidate.py` 宽松 —— **以谁为准？**

**以 `promote_candidate.py` 为准（且要加上 `auto_promote_watch.py:386-400` 的 strictly_better）。**

理由（三条，按重要性）：

1. **口径层次不同**：`six_criteria_audit.py` 是**纯精度审计器**，它的 `judge()`（`:172-183`）只回答
   「六轴是否全 `≥` 基线」；它**从不检查**板端时延、种子复现、甚至**没有 `strictly_better` 这个概念**
   （全文 grep 零命中）。它的输出标签（`:224`）是**给人看的提示**，不是升轨许可。
2. **`promote_candidate.py` 是唯一的写盘点**：只有它会写 `baseline_v2` / `baseline_registry` /
   `beat_to_pass`（`:504-543`）。任何升轨决策最终都要过它的 `check_arm()`（`:190-413`）。
3. **实证**：57 个六项全过臂在审计输出里全带「🔴 新候选（可升轨）」，但
   `archive/19_主档正文_§283-298_20261120-20261123.md:1024`（§288） 记录它们**全部是基线逐位克隆 ⇒ 零增量 ⇒ 一个都不能升轨**。

**⇒ 实操判据（四项取交集）**：
`六轴全 ≥ 基线`（`six_criteria_audit.py:172-183`）
**∧** `至少一轴严格 > 基线`（`auto_promote_watch.py:386-400`）—— 🔴 **以自动通道的六轴为准（含 t2，`:398`）**，
手工 `promote_candidate._strictly_better`（`:390-396`）**没有 t2 轴** ⇒ 人工核对要自己补（§5.1 步骤 3）
**∧** `板端 T4 PASS 且未 stale`（`promote_candidate.py:250-272`）
**∧** `种子复现关已过（或走单臂严格支配通道，或人工 --force）`（`promote_candidate.py:294-349`, `:399`）

---

## ⑥ 附：本次调查用到的权威文件索引

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `manu/pipeline/npu/t4_criterion.py` | 166 | 🔴 板卡 T4 判据**唯一真源** |
| `manu/pipeline/npu/board_t4_watch.py` | 962 | 板端测速**生产者**（守望） |
| `manu/pipeline/npu/board_t4_rejudge.py` | 217 | 板端读数**批量**复判（无 `--trial`） |
| `manu/pipeline/npu/promote_candidate.py` | 581 | 🔴 升轨**唯一写盘点** |
| `manu/pipeline/npu/promote_ready_cmd.py` | 166 | 只读命令生成器 |
| `manu/pipeline/npu/auto_promote_watch.py` | 554 | 自动升轨守望（`strictly_better` 真正落点） |
| `manu/pipeline/npu/six_criteria_audit.py` | 255 | 六项判据**审计器**（`judge()` / `t2_cap()` / `live_base()`） |
| `manu/pipeline/npu/baseline_guard.py` | 274 | 基线守卫（源码常量 = 真源） |
| `manu/pipeline/npu/trackb_core.py` | 192 | Track-B 判据 `qualifies()`（含 strictly_better） |
| `manu/pipeline/npu/promote_trackb.py` | 220 | Track-B 升轨路径（留 `baseline_v2_strict_backup`） |
| `manu/pipeline/npu/nano_search.py` | 1497 | 台账/搜索器（`BASE_V2` / `DELIVER` / `run_official_gate`） |
| `manu/pipeline/npu/route2_acceptance.py` | ~200 | `nanobench()` / `export_rknn()` |
| `manu/evaluation/tiny_target_subset.py` | ~300 | 🔴 NanoBench 尺寸分桶与双容差 |
| `manu/evaluation/eval_arm_pc_gmc_ab.py` | 214 | T1 A/B 评测（15 档阈值、`negative_bands`） |
| `manu/training/auto_epoch_gate_check.py` | 586 | 🔴 官方三轨门禁（`gate_report.json` 产出者） |
