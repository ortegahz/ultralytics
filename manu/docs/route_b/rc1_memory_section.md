> ⚠️ **本文件不是独立 memory 章节**，它是 `memory.md` **§四十六** 的**逐字源文**（2026-10-21 由
> `manu/docs/route_b/rc1_memory_section.md` 原样追加进 `memory.md` 末尾）。保留此处仅为溯源用。

# §四十六 Epoch 3 = RC1 封存判定（Sweet Spot / 帕累托最优解）（2026-10-21）

## 1. 决策一句话

路线 B 首轮试跑（5 epoch）的最佳产物**不是** F1 最高的 epoch5，而是 **epoch3**。
已固化为 **Release Candidate 1 (RC1)**，并登记为**后续一切微调实验的对账基准**（不再回看 epoch5）。
判定依据全部可在 `runs/mixed_ft_route_b/gate/gate_history.jsonl` 逐条复核。

## 2. 三轨全周期对比（基线 = Epoch-0 去 P0 冻结口径；数字取自 gate_history.jsonl）

| Epoch | T1 峰值 th | T1 F1 | T1 TP | T1 FP | T2 F1 | T2 TP | T2 FP | T2 FN | T3 F1 | T3 TP | T3 FP | T3 FN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0（基线） | 0.25 | 0.905736 | 21,624 | 1,014 | 0.988436 | 16,283 | 26 | 355 | 0.837331 | 8,012 | 590 | 2,523 |
| 1 | 0.25 | 0.902215 | 21,650 | 1,232 | 0.998769 | 16,632 | 35 | 6 | 0.846895 | 8,170 | 589 | 2,365 |
| 2 | 0.30 | 0.901927 | 21,437 | 988 | 0.998889 | 16,634 | 33 | 4 | 0.859587 | 8,341 | 531 | 2,194 |
| **3 = RC1** | **0.30** | **0.901918** | **21,398** | **941** | **0.998979** | **16,635** | **31** | **3** | **0.871684** | **8,495** | **461** | **2,040** |
| 4 | 0.30 | 0.901796 | 21,488 | 1,057 | 0.998859 | 16,635 | 35 | 3 | 0.872648 | 8,555 | 517 | 1,980 |
| 5 | 0.30 | 0.900501 | 21,404 | 1,023 | 0.998949 | 16,636 | 33 | 2 | 0.874924 | 8,583 | 502 | 1,952 |

## 3. 三条量化依据（RC1 为什么赢）

| # | 维度 | 证据 | 出处 |
|:--|:--|:--|:--|
| 1 | **源域虚警最低** | T1 FP **941** 为 Ep1~Ep5 **全场最低**（Ep1 1232 / Ep2 988 / **Ep3 941** / Ep4 1057 / Ep5 1023），**比基线 FP 1014 还少 73**；Ep4/Ep5 已反弹破千，Ep5 F1 跌破 0.901（0.900501）= 源域散架征兆 | `gate_history.jsonl` `tracks.track1_uav.finetune_peak` |
| 2 | **目标域质量最高** | T2 F1 **0.9989791016094163** 为全周期峰值；FP **31** 为 Ep1~Ep5 最低（35/33/**31**/35/33）；FN=3 与 Ep5 的 FN=2 相差 1 帧，无实质差异 | `tracks.track2_golden` |
| 3 | **杂波抑制力最强** | T3 FP **461** 为全周期最低（589/531/**461**/517/502），**净砍 129 个误检**；且已完成 Ep5 全程涨幅的 **91.38%**（(0.8716843671438099−0.8373308251032032)/(0.8749235474006115−0.8373308251032032)=0.9138） | `tracks.track3_blind` |

**Ep3→Ep5 边际收益递减（锁定 Ep3 的核心理由）**：额外跑满 2 个完整 epoch（约 40 分钟算力），
T3 F1 只从 0.871684 爬到 0.874924（**+0.0032**），T3 FP 却由 461 回升到 502；
同期 T1 F1 再损 **−0.001417**（≈−0.14pp）、T1 FP 由 941 回升到 1,023。
train loss 虽继续下探（Ep3 0.4171 → Ep5 0.3828），但 val 端虚警全面反弹 ⇒ 判定 **Ep4~5 进入边际收益递减 / 轻度过拟合区间**。
⇒ **锁定 Ep3 = 强判别 + 低虚警 + 轻泛化的最佳交点。**

## 4. ⚠️ 必读限定（不夸大，登记在案）

- 🔴 **原始 Track1 红线（去 P0 冻结基线 F1 ≥ 0.905736）在 Ep1~Ep5 全部 5 个 epoch 均未通过**，
  RC1 自身 T1 F1 = 0.901917808219178 同样判负（`gate_report.json`：`veto_failures=["track1_uav"]`，verdict=FAIL，rc=1）。
  ⇒ **RC1 的产生依据是「帕累托最优 + 虚警最低」，不是「通过原始三轨门禁」**。
  任何把 RC1 表述为"已通过三轨红线"的说法都是错的。
- ⚠️ **T1 峰值门限漂移**：基线/Ep1 峰值在 th=0.25，Ep2~Ep5 峰值在 th=0.30。
  跨 epoch 的 F1/FP 对比是「各模型自身最优门限下」的比较（标准做法），但门限本身右移了 0.05，
  解读时须连同 `delta.negative_bands = 7/15`、`delta.mean_f1 = +0.001309` 一并看。
- ⚠️ **三轨评测口径天生不对称**（T1 中心点 8px 距离匹配 / T2-T3 框内包含匹配），
  三轨 F1 **不可横向比较**，只做轨道内纵向对比（§四十三 §2）。
- ⚠️ **T2「FP 增量 ≤ 10」红线的唯一攻击点仍是序列 000703**（基线 FP 4 → Ep5 FP 29，Δ+25，占黄金池总误报 88%），
  RC1 之后须定点攻它（把 000703 失效机理并入训练集负样本）。

## 5. RC1 封存登记（机读 + 路径 + 指纹）

| 项 | 值 |
|:--|:--|
| 封存权重（服务器） | `runs/mixed_ft_route_b/RC1/epoch3.pt`（21,301,091 bytes，**mode 0444 只读**） |
| 源权重 | `runs/mixed_ft_route_b/weights/epoch3.pt` |
| md5 | `a845a95b77ac8762b7d033b64f61425e` |
| sha256 | `dcc6c40f012866f588bfb972451a405a61cf0ae72dcc0c80cbcef2b76d8010d9` |
| 机读登记 | `runs/mixed_ft_route_b/RC1/RC1.json`（本地副本 `manu/docs/route_b/RC1.json`，md5 `1555f07353f86468db6054314226deb8`） |
| 人读说明 | `runs/mixed_ft_route_b/RC1/README.md`（本地副本 `manu/docs/route_b/RC1_README.md`） |
| 封存工具 | `manu/training/seal_release_candidate.py`（数字**程序化**取自 gate_history，杜绝手抄转写误差） |
| 证据源 | `runs/mixed_ft_route_b/gate/gate_history.jsonl`、`runs/mixed_ft_route_b/gate/epoch3/gate_report.json` |

## 6. 后续对账基准的用法（与 §26 / §四十二 的冻结基线登记同一套约定）

| 轨 | RC1 基准值 | 后续实验的对账要求 |
|:--|:--|:--|
| T1 Anti-UAV 官方大盘 | 峰值 th=0.30：**F1 0.901917808219178 / TP 21,398 / FP 941 / FN 3,713** | 必须同时看 15 档门限**整条曲线**是否整体位移（`delta.negative_bands`、`delta.mean_f1`），**禁止**用 `eval_arm_pc_gmc_ab.py --expect-*`（那是**基线复现**语义，对任何微调权重都必然 FAIL） |
| T2 黄金池 7 条 | **F1 0.9989791016094163 / TP 16,635 / FP 31 / FN 3** | FP 增量红线仍按「≤ RC1 + 10」即 **FP ≤ 41**；并逐序列看（000703 单列） |
| T3 盲测 001324 | **F1 0.8716843671438099 / TP 8,495 / FP 461 / FN 2,040** | 真灵敏度在此；**FP 不许反弹**（Ep4/5 已证明会反弹） |

下一轮候选方向（用户已点名，**均未启动**）：**冻结 BN**、**L2-SP 正则**、重新配比、加 000703 同类负样本。
这些实验一律**以 RC1 为起手与对账基准**，不再回看 epoch5。

## 7. 可复制命令（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，根目录 `/tmp/pycharm_project_10ae9e2e`）

```bash
# ① RC1 封存（幂等：重跑会重新 copy2 + chmod 0444 + 覆写登记）
python manu/training/seal_release_candidate.py \
  --run-dir runs/mixed_ft_route_b --epoch 3 --out-dir runs/mixed_ft_route_b/RC1

# ② RC1 三轨复测（位置参数 / 旗标两形式皆可）
bash manu/training/auto_epoch_gate_check.sh runs/mixed_ft_route_b/RC1/epoch3.pt 3
bash manu/training/auto_epoch_gate_check.sh --weights runs/mixed_ft_route_b/RC1/epoch3.pt --epoch 3

# ③ RC1 权重变更核查（G1 冻结组 66 张量必须 max|Δ|=0）
python manu/training/verify_finetune_delta.py \
  --weights runs/mixed_ft_route_b/RC1/epoch3.pt \
  --source runs/optuna_p0_nas/trial_0474/weights/best.pt

# ④ 全周期逐序列归因（Ep0~Ep5）
python manu/training/analyze_per_sequence.py --run-dir runs/mixed_ft_route_b \
  --epochs 0,1,2,3,4,5 --out-json runs/mixed_ft_route_b/per_sequence_attribution.json

# ⑤ 指纹双侧核对
md5sum runs/mixed_ft_route_b/RC1/epoch3.pt runs/mixed_ft_route_b/weights/epoch3.pt
# 期望两者均为 a845a95b77ac8762b7d033b64f61425e
```
