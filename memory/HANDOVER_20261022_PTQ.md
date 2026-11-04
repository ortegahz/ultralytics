# 交接说明：RK3588 PTQ 隔夜实验（2026-10-22 18:30 工作站关机后）

> 工作站（host）18:30 关机 ⇒ NFS 导出失效、交互会话中断。
> **服务器（192.168.99.40）与板卡（192.168.0.64）不关机**，整条实验链在服务器上自主跑完。
> 本文件同时存在于工作站本地、SSHFS 镜像与服务器（md5 一致）。

## 一、明早按这个顺序读产物

```bash
ssh -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
```

| 顺序 | 产物 | 说明 |
|---|---|---|
| 1 | `/mnt/data/siping/ptq_work/final_selection/final_choice.json` | **最终定案**：胜出模型 + 它的最优阈值 + 全部候选的阈值曲线 |
| 2 | `/mnt/data/siping/ptq_work/selection/selection_table.json` | 选型表（th=0.30 同口径） |
| 3 | `/mnt/data/siping/ptq_work/full_scale/status.json` | 全量三轨逐块状态（`chunks ok / failed`、`frames scored`） |
| 4 | `/mnt/data/siping/ptq_work/full_scale/*_report.json` | 三轨的 P/R/F1/TP/FP/FN |
| 5 | `/mnt/data/siping/ptq_work/full_scale/{blind,golden,anti}_board.tsv` | 逐帧检测（可直接离线重扫阈值） |
| 6 | `/mnt/data/siping/ptq_work/logs/full_scale.log` | 全量运行日志 |
| 7 | `/mnt/data/siping/ptq_work/logs/finalize.log` | 定案过程日志 |
| 8 | `/mnt/data/siping/ptq_work/logs/night_selection.log` | 选型扫描日志 |

## 二、整夜自主链路（已上线，无需人工干预）

```
15_night_selection.sh  （选型扫描，板端 2 ctx 并行，18:24 deadline）
   └─ 16_collect_selection.py  （规则化收集 → selection_table.json + best_model.json）
        └─ 20_finalize_and_launch.py  （等 15 退出 → 离线阈值扫描 → 写死规则定案
             → final_choice.json → 启动 14_run_full_scale_nofs.py 全量三轨）
              └─ 14_run_full_scale_nofs.py  （服务器 tar 流分块推帧到板卡 /userdata
                   → 板端 NPU 推理 → 回传 TSV → 合并 + 逐块校验 + 断点续跑）
```

- `20` 与 `14` 都是 `setsid nohup` 脱离会话 ⇒ **工作站关机不影响**。
- `14` 自带 `flock` 锁（防重复启动）、断点续跑、逐块 md5/行数/帧名集合校验、失败隔离（单块失败不中断整轮，最后非零退出）。
- 全量采集阈值固定 **0.30**（与全部既有实验同口径，可直接对账）。
- PT 参照臂走 `cuda:1`，让开 QAT 训练的 `cuda:0`。

## 三、QAT 支线（并行，服务器 4×RTX 4090 D）

- 目标：LSQ 微调 → 导出 QDQ ONNX → `rknn.build(do_quantization=False)` → 板端验证。
- 已确认前提（`17_probe_qdq_import.py`，verdict `QDQ_ADOPTED`）：toolkit 2.3.2 **采纳** ONNX 的 Q/DQ `scale`/`zero_point` 并带到板端（5/5 案例，scale 逐位一致）。
- 硬约束：Q/DQ ONNX **必须** `do_quantization=False`；**不得**传 `mean_values`/`std_values`。
- 板端 I/O 契约：1 输入 `[1,640,640,3]` uint8 NHWC + 2 输出 `[1,1,320,320]`、`[1,2,320,320]`。
- 产物目录：worker 自建（见 `/mnt/data/siping/ptq_work/` 下的新建目录）与 `manu/pipeline/npu/ptq/` 新脚本。

## 四、明早的待办（按优先级）

1. 读 `final_choice.json` + `status.json` ⇒ 补全量三轨表到 `manu/memory/PTQ_RK3588_int8_报告.md`。
2. 对全量 TSV 跑多阈值扫描，给出**全量口径的部署阈值**：
   ```bash
   python manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
     --manifest <该轨的 rows 清单> --tsv-dir <TSV 所在目录> --skip-pt \
     --out /mnt/data/siping/ptq_work/sweep_full
   ```
   （注意 `19` 的 `--tsv-dir` 模式需要与该轨对应的 manifest；全量轨的 rows 由 `14` 内部生成，如需可从 `14` 的 `--out` 目录重建。）
3. **p0_on 变体板端 ZERO_DETECTIONS 的根因**（量程正常 ⇒ 板端特有失效，非量程削顶）：
   - 现象：PC 模拟器 F1=0.6724 正常，板端 0 检出；`out[0] scale=0.00380537`（上限 0.9704）。
   - 下一步：板端 dump heatmap 原始张量，与 ONNX Runtime 的 float 输出逐点比对，定位是哪个张量/哪一步塌掉。
4. 补 `p0_on` 镜像链的板端两级（若 p0_on 修好）。
5. QAT 结果并入最终模型评估；若 QAT 的 int8 明显优于 PTQ ⇒ 用 QAT 模型重跑全量（`14` 可续跑，改 `--model` 需新 `--out`）。
6. 三个 memory 档 + 报告的**三侧 md5 重新校验**（工作站已关机 ⇒ SSHFS 镜像不可用，直接用 scp 拉回）。

## 五、坑位清单（勿重试）

| 坑 | 结论 |
|---|---|
| toolkit 2.5.0 | 必崩（SDPA `IndexError`）⇒ 固定 2.3.2 |
| toolkit 2.3.2 + uint8 图输入 | `abort rc=134 Not Support Dtype: 2` ⇒ 封死 |
| `kl_divergence` | 输出量程削顶 ⇒ 死模型（0.2399 < 0.3） |
| `mmse` | 151 层 ETA 4h08m ⇒ 放弃 |
| `w8a16` / `gdq` | rk3588 不支持 ⇒ 封死 |
| 模拟器喂 float32 | cos 0.108 ⇒ 一律 uint8 NHWC |
| 板端 TSV | 每检测一个 TAB 列（只读第 1 列会把 FP/Recall 算错） |
| `while read < $LIST` 里跑 python/scp | 继承 stdin 吃掉清单 ⇒ 每个 axis 只跑 1 个模型 |
| 板卡 ctx 计数 | 必须 `ps -eo comm`（`ps -eo args` 把 bash 包装一起数） |
| 板端工作目录 | 必须 per-run（PID 隔离），否则并发互相删 TSV |
| 板卡 FUSE/sshfs | 内核 5.10 拒 fuse2 ⇒ 走 SSH tar 流 |
| `ps -eo args \| grep "<名>"` | 匹配到自己的 ssh 命令行 ⇒ 用 `名[_]pattern` 转义 |
| anti 轨 GT | 标签在 `uav_gmc_median/labels/val`（同名同格式）⇒ 已加 `ANTI_GT_ROOT` |
| scp 推模型到 NFS | 慢（写工作站盘）⇒ 全量用 `14`（推 `/userdata/ptq_models`） |

## 六、核心结论速览（详见 `manu/memory/PTQ_RK3588_int8_报告.md`）

- int8 零检出真因 = `kl_divergence` 把输出量程剪到 0.2399 < 阈值 0.3（与输入 dtype 无关）。
- 量化损失 **100% 在虚警**（int8 的 TP 比 fp16 高 86，FP 527 vs 184）。
- 量化把最优检测阈值推移 **+0.10**：int8 最优 0.45、fp16 最优 0.35 ⇒ **板端 int8 不能沿用 PC 的 0.3**。
- 阈值重标定：int8 F1 0.6448 → **0.7090**（+0.0642），量化损失 −0.1036 → **−0.0379**（回收 63%）。
- 剩余 0.0379 阈值救不回来 ⇒ **QAT 是唯一解**；QDQ 导入已验证被 toolkit 采纳。
- PTQ 算法轴在 rk3588 封闭 ⇒ **校准集维度也基本无效**（A/B/C/D 极差仅 0.048）。
  > 🔴 **（2026-10-23 下午更正）** 这一行的**依据有污染**：实测 `calib/set_B`、`set_C`、`set_D` **全部泄漏评测帧**
  > （`27_check_calib_leak.py`：226 / 240 / 216 路径命中，共享 001324 与黄金池 000703），**只有 `set_A` 干净**。
  > 且「阈值位移 +0.10」「量化损失 100% 在虚警」**只对 001324 的 2000 帧子集成立**（全量逐轨给）。
  > **校准集轴的最终判定**（干净集复评）：**在唯一被全量验证的干净集 set_6 上无实质收益**（黄金池 ≈+0.0000、001324 −0.0064）
  > ⇒ 校准集轴**不是一条能单独兑现的杠杆**；⚠️ **set_4 / set_5 的子集收益尚未做全量，是否成立未知**。
  > 详见 `manu/memory/PTQ_RK3588_int8_报告.md` §19~§24 与 `manu/memory/memory.md` §11.5。
