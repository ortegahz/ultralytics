# 交接说明：YOLO/ultralytics 工程「图像质量 → 跟踪效果」合格线任务
# 交接时间：2026-11-20 17:57 CST（本机 18:00 断电，服务器 192.168.99.40 不断电）
# 会话线程（服务器上已接管）：01a1f7da-e5d8-71a0-b3aa-b6c189101066
# 服务器工作目录：/tmp/pycharm_project_a7d966e6   conda env：uav (python3.10, 有 cv2)

## 0. 接续命令
export CODEX_HOME=/mnt/data/.codex
cd /tmp/pycharm_project_a7d966e6
codex resume 01a1f7da-e5d8-71a0-b3aa-b6c189101066
# 无头单轮：codex exec resume 01a1f7da-e5d8-71a0-b3aa-b6c189101066 "继续之前的工作"
# 若报 Not inside a trusted directory，往 /mnt/data/.codex/config.toml 加：
#   [projects."/tmp/pycharm_project_a7d966e6"]
#   trust_level = "trusted"

## 1. 本机断电前已完成（全部已落盘，md5 双侧一致）
- 会话已 fork 到服务器（handoff/handoff_to_server.sh --latest --cwd /tmp/pycharm_project_a7d966e6）
- 工作区已 rsync 到服务器：manu/video_quality/ + manu/memory/
- 脚本 md5（本地 = 服务器）：
  vq_sot_compare.py  5beff7e558f629672389687fc226bb4a   ← 本轮改过（修 KeyError）
  vq_sot_link.py     ebc6aa0980d4951765ce95237376803d
  vq_sot_link_driver.py b1de859720c63b617bb01363a9eb183c
  vq_sot_verify.py   3e57e2bfc2c8e23374ca84f2799dce74

## 2. 本轮修的 bug（已修 + 已同步服务器 + 已重跑）
`vq_sot_compare.py` 的 `admission_line()` 返回 dict 用的是 `tpr`/`fpr` 键，
但下游 `derive_thresholds()` 读的是 `good_tpr`/`bad_fpr` ⇒ KeyError: 'good_tpr'。
修法：两条返回分支各加
    best["good_tpr"] = best.pop("tpr")
    best["bad_fpr"]  = best.pop("fpr")
（本地 md5 已是 5beff7e5…，服务器已验证一致）

## 3. 正在服务器后台跑的批任务（断电不影响）
PID 2375743（父 2375732），日志 /tmp/sot_compare_run.log
命令：
  cd /tmp/pycharm_project_a7d966e6 && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh \
    && conda activate uav && for S in long short; do python manu/video_quality/vq_sot_compare.py \
    --sot-dir /tmp/sot_out,/tmp/sot_out_osd --windows manu/video_quality/report/windows.csv \
    --grid-summary manu/video_quality/report/grid_summary.csv \
    --out-dir /tmp/sot_report_$S --score xcorr_${S}_ugr; done
注意：python 输出有缓冲，日志要等进程结束才刷出来，别以为卡死。
产物：/tmp/sot_report_long/{sot_thresholds.csv,sot_thresholds_ir.csv,sot_thresholds_vis.csv,...}
      /tmp/sot_report_short/{同上}
⚠ long/short 的 per-modality 文件同名，拉回本地要分别改名（sot_thresholds_short.csv 等）。

## 4. 剩余待办（按顺序）
1) 取准入线：ssh 上去 grep -E "LINE=" /tmp/sot_compare_run.log；确认每个 scope×metric 都有
   LINE= / good_TPR / bad_FPR / good_cov；若无则查 admission_line 的 fallback 分支。
2) rsync /tmp/sot_report_long/ → 本地 manu/video_quality/report/
   rsync /tmp/sot_report_short/ → /tmp/sot_short2/ 再改名为 sot_thresholds_short.csv 等
3) 改写汇报件 §9.5：主表改为「按 IR / VIS 分列」的准入线（线上跟住率 ≥X% / 坏片误放率 /
   好片覆盖率），混池线只作参考并注明局限；同步更新 §一 的合格线表。
4) 同步改：专题档 §8.5、manu/memory/memory.md §284、manu/memory/memory_compact.md §284、
   manu/memory/.slim_memory.md 登记行（追加/替换该表数字，**不要重写既有内容**）
5) 重跑验收：python manu/video_quality/vq_sot_verify.py（CHECK5 用 raw_threshold，不受新口径影响）
   + 导航表校验 + 表格列数自检
6) md5 双侧比对 4 个脚本；git diff --cached 必须为空（仓库根不是 git 仓，ultralytics/ 是独立 repo，
   **禁止 git add/commit/push**）
7) 收尾后 update_goal(status="complete")

## 5. 必须写进文档的诚实口径（不能丢）
- 本节测的是「**任意点选位置的纹理可跟踪性**」，**不是「跟踪无人机本身的成功率」**
  （测试片里没有无人机，且用的不是交付模型 trial_0080/RC2）。
- 灰度域 IR 与彩色域 VIS 的指标量纲不同（好片 lap_var 中位 IR 100 vs VIS 2102），
  **跨模态不能共用同一条绝对线**。
- 上线前必须统一裁剪/缩放策略；必须扩样重标（坏源片仅 4 条）。
- snr_db / noise_sigma / dog_snr_max 的 AUC 分别只有 0.134 / 0.111 / 0.333 ⇒ 无区分度，
  不能当合格线用。

## 6. 已稳定的核心结论（不随合格线口径变，直接照抄）
- τ_det（纹理地板，只用好片 P10）：IR 4.3279（n=6448）/ VIS 97.6087（n=11056）
- UGR p50：好片 short 0.8125/0.875、long 0.375/0.4375；坏片源视频 short/long 全 0.000
- patch 臂（产品快核）全组 UGR = 0.000 ⇒ 快核要紧凑亮/暗斑块，点背景纹理锁不住
- AUC（好片为正类）：global_contrast 0.919 / edge_density 0.902 / lap_var 0.896 /
  blockiness 0.894 / entropy 0.874
- 与 §271 互证：逐片 |SOT lock − grid tgr_raw| mean 0.082 / worst 0.208；verify PASS
- 坏片 120BT-04-可见光 short 跟住率 0.862 但 DET 仅 0.292 ⇒ 不加纹理地板会伪装成最好片
- gap 最大口径（已稳定）：
  long：all lap_var 2424 / edge_density 0.1947 / global_contrast 0.7746 / blockiness 0.8745 /
        entropy 7.372；ir 222.7 / 0.04768 / 0.7624 / 1.02 / 7.372；vis 2400 / 0.1947 / 0.1984 /
        0.8745 / 6.729
  short：all 2421 / 0.04763 / 0.7746 / 1.05 / 7.378；ir 68.46 / 0.02505 / 0.7559 / 1.02 / 7.355；
        vis 199.4 / 0.04805 / 0.1843 / 1.05 / 5.288

## 7. 好/坏片窗口分布（写文档逐字照抄）
好片 IR：lap_var P10 37.8/P50 100.4/P90 130.5、edge_density 0.0141/0.0496/0.0635、
  global_contrast 0.516/0.771/0.811、blockiness 1.025/1.049/1.618、entropy 6.747/7.371/7.438
好片 VIS：lap_var 699.9/2102/3070、edge_density 0.0468/0.126/0.205、global_contrast 0.562/0.645/0.721、
  blockiness 1.009/1.024/1.068、entropy 5.757/6.732/7.020
坏片 IR：lap_var 22.2/32.1、edge_density 0.0019/0.0019、global_contrast 0.12/0.257、
  blockiness 1.761/2.061、entropy 4.021/5.177
坏片 VIS：lap_var 4.83/30.98、edge_density 0/0.0038、global_contrast 0.118/0.341、
  blockiness 1.020/1.163、entropy 4.426/5.812

## 8. 铁律
- 仓库根不是 git 仓；ultralytics/ 是独立 repo。**禁止 git add/commit/push**。
- 每条结论带：脚本路径 + 可复制命令 + 执行机 + 产物路径。
- ssh 必须用：ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 32222 huangzhe@192.168.99.40
- 服务器 conda：/home/huangzhe/anaconda3（注意非交互 ssh 下 `python` 是 py2，必须 conda activate uav）
- 同步判据只用 md5 双侧比对；memory 文件只追加不改历史；中文回答。

## 9. ⚠️ 追加（18:02 主机断电前最后一手）：准入线跑完了，但几乎全是 FALLBACK
`grep -E "LINE=" /tmp/sot_compare_run.log` 结果：**24 个 (scope × metric) 里只有 1 个
真的满足两条约束**，其余全部 `[FALLBACK 无候选同时满足]`。
唯一合格：scope=ir, metric=blockiness, LINE=1.032, good_TPR=0.906, bad_FPR=0.0000,
good_cov=21.1%（且 good_below_median=0.375 > 0 ⇒ 好片低于线的部分照样跟得住）。

这**不是脚本 bug，是真实结论**，写文档时必须这么表述：
- 「单指标 + 单条绝对线」无法同时做到 线上跟住率≥90% 且 坏片误放率≤1%：
  好片内部的逐秒 UGR 方差太大（好片 P10 质量窗口的 UGR 中位仍 > 0），
  把线压到 TPR≥0.90 只能落在好片 P90 附近 ⇒ 好片覆盖率必然只剩 20~30%。
- 因此主表应给**两条线**，而不是一条：
  (a) **准入线（保守/上线用）**：取满足 bad_FPR ≤ 1% 的最低线 ⇒ 好片覆盖率低但安全；
  (b) **推荐线（覆盖用）**：取 good_TPR 最高且 bad_FPR 可接受的线。
  两者都按 scope=ir / vis 分列，并把 FALLBACK 标记暴露出来，不要藏。
- 现有 CSV 已带 `cover_fallback` 列，可直接用；补列时从 CSV 取数，不要手抄。
- `raw_threshold`（gap 口径，已稳定、与 §271 互证）仍应保留在表里作为**参考线**，
  因为它就是历史上已落档、能对上的那一组数。

补充正确性提示：`admission_line` 返回的键是 good_tpr/bad_fpr（已修），
落盘 CSV 里的列名是 cover_good_tpr / cover_bad_fpr / cover_fallback。

## 10. 🔴 追加（独立复算已完成）：新脚本 + 三条铁证（写 §9.5 / §一 必须用）
新脚本（本地=服务器 md5 一致）：`manu/video_quality/vq_sot_admission_verify.py`
md5 = `a95417e0744a39014c89a2e4d2381a23`（修 bug 后请重新取 md5 再同步）
它 **不 import** `vq_sot_compare.py`，从原始 JSON + windows.csv 独立重算，结果 **[RESULT] PASS**：
- CHECK A 纹理地板：IR 4.3279 / VIS 97.6087 与存档逐位吻合
- CHECK B 逐窗聚合：13,207 窗口数吻合
- CHECK C **24 组(scope×metric)准入线的 line / TPR / FPR / 覆盖率 / fallback 全部精确复现**
- CHECK D gap 参考线（raw_threshold）全部精确复现
- 产物：`/tmp/sot_report_long/sot_admission_transfer.csv`（迁移体检明细）

### 铁证一：混池线会把好片系统性误杀（CHECK E1，long 时程）
用**混池** gap 线分别卡两个模态的好片窗口：
| 混池线 | 卡好红外片 | 卡好可见光片 |
| --- | --- | --- |
| `lap_var ≥ 2423.9` | **403/403 = 100% 误杀** | 408/691 = 59.0% |
| `edge_density ≥ 0.19466` | **403/403 = 100%** | 540/691 = 78.1% |
| `global_contrast ≥ 0.77465` | 210/403 = 52.1% | **691/691 = 100%** |
| `blockiness ≤ 0.87452` | **403/403 = 100%** | **691/691 = 100%** |
| `entropy ≥ 7.3725` | 204/403 = 50.6% | **691/691 = 100%** |
⇒ 混池线**没有一个指标能同时对两个模态成立**，§一/§9.5 必须**按 IR / VIS 分列**，
混池线只能标注「仅供整体参考，跨模态禁用」。

### 铁证二：gap 线是「分界线」不是「准入线」（CHECK E3）
用**本模态自己的** gap 线卡本模态好片窗口，误杀率照样是 100%/45%/42%/93%/51%/100%…
（IR：lap_var 403/403、edge_density 182/403、global_contrast 170/403、blockiness 373/403、
 entropy 204/403、snr_db 403/403、noise_sigma 403/403、dog_snr_max 403/403；
 VIS：lap_var 404/691、edge_density 540/691、blockiness 691/691、entropy 345/691、snr_db 691/691）
原因：gap 口径只最大化「线以上均值 − 线以下均值」，**完全不管好片覆盖率**，
线会跑到好片分布的**上尾之外**，只覆盖极少样本。
⇒ **gap 线只能写成「最优分界线」，绝不能当准入线**；准入线必须用 ROC 式的 admission 线
（它显式约束了 TPR/FPR/覆盖率）。

### 铁证三：ROC 式准入线只有 1/24 真正满足双约束
约束：`P(跟得住|≥L,好片) ≥ 0.90` 且 `P(跟得住|≥L,坏片) ≤ 0.01`
- 唯一满足：`scope=ir, blockiness, L ≤ 1.032, good_TPR=0.906, bad_FPR=0.000, 好片覆盖 21.1%`
- 其余 23 组全部 `[FALLBACK]`（无候选同时满足）⇒ **好片内部逐秒 UGR 方差太大**，
  单指标 + 单条绝对线**做不到**「线上跟住率≥90% 且 坏片误放率≤1%」
- 长时程各 scope 的 admission 线（good_TPR / bad_FPR / 好片覆盖）：
  all：lap_var 2418.7(0.656/0.0000/26.1%)、edge_density 0.19694(0.825/0.0000/13.1%)、
        global_contrast 0.75725(0.639/0.0000/22.0%)、blockiness 1.5001(0.479/0.1152/92.0%)、
        entropy 7.3642(0.731/0.0000/19.0%)、snr_db 47.646、noise_sigma 0.38912、dog_snr_max 218.67
  ir：lap_var 19.565(0.526/0.0742/96.8%)、edge_density 0.059088(0.773/0.3755/24.1%)、
      global_contrast 0.75388(0.636/0.0000/60.0%)、blockiness **1.0319(0.906/0.0000/21.1%)**、
      entropy 7.3546(0.730/0.0000/55.1%)、snr_db 48.082、noise_sigma 0.38483、dog_snr_max 220.3
  vis：lap_var 2384.5(0.659/0.0000/42.0%)、edge_density 0.19763(0.820/0.0000/20.1%)、
       global_contrast 0.57919(0.409/0.0154/84.9%)、blockiness 1.1731(0.405/0.1563/99.0%)、
       entropy 6.9489(0.728/0.0268/26.0%)、snr_db 36.232、noise_sigma 1.7517、dog_snr_max 50.825

### §9.5 / §一 的正确表结构（建议照此写）
每个模态一张表，每行一个指标，列：
`指标 | 准入线(原始量纲) | 线上跟住率 TPR | 坏片误放率 FPR | 好片覆盖率 | 是否满足双约束(FALLBACK 标注) | 参考分界线(gap)`
并在表头写清楚三句：
1. 准入线 = ROC 式，目标是「线上跟住率 ≥ 好片典型水平 且 坏片误放率 ≤ 1%」；
2. **每条线只在它自己那个模态内有效**，跨模态禁用（附 CHECK E1 铁证表）；
3. gap 列是「最优分界线」，**不是准入线**（附 CHECK E3 铁证）。

## 11. 🔴🔴 追加（最终交付口径）：「好片分位数」组合判据 —— 这才是领导要的答案
独立脚本 `vq_sot_admission_verify.py` 新增 CHECK H（同样 PASS）：
把绝对阈值换算成「达到好片分布的第 P 分位」，多指标**同时过线**（AND），
输出 `sot_percentile_rules.csv`。
**判据用 5 个有效指标**（AUC 0.874~0.919）：global_contrast / edge_density / lap_var / blockiness / entropy。

### 红外 IR（长时程 1 秒）
| 好片分位 | 组合规则 | 好片覆盖 | **线上跟住率** | **坏片误放率** |
| --- | --- | --- | --- | --- |
| **P10** | global_contrast ≥ 0.51601 AND edge_density ≥ 0.014121 AND lap_var ≥ 37.825 AND blockiness ≤ 1.0246 AND entropy ≥ 6.7471 | 10.2% | **97.6%** | **0.0000** (0/3220) |
| P25 | gc ≥ 0.57543 AND ed ≥ 0.029497 AND lap ≥ 63 AND blk ≤ 1.0332 AND ent ≥ 7.1918 | 23.6% | 88.4% | **0.0000** |
| P50 | gc ≥ 0.77068 AND ed ≥ 0.049584 AND lap ≥ 100.39 AND blk ≤ 1.0491 AND ent ≥ 7.3706 | 23.6% | 76.8% | **0.0000** |
| P75 | gc ≥ 0.80117 AND ed ≥ 0.0585 AND lap ≥ 117.98 AND blk ≤ 1.1072 AND ent ≥ 7.4135 | 6.7% | 48.1% | 0.0000 |
| P90 | gc ≥ 0.81132 AND ed ≥ 0.063481 AND lap ≥ 130.5 AND blk ≤ 1.6178 AND ent ≥ 7.4384 | 1.2% | 60.0% | 0.0000 |

### 可见光 VIS（长时程 1 秒）
| 好片分位 | 组合规则 | 好片覆盖 | **线上跟住率** | **坏片误放率** |
| --- | --- | --- | --- | --- |
| P10 | gc ≥ 0.56166 AND ed ≥ 0.046798 AND lap ≥ 699.94 AND blk ≤ 1.0089 AND ent ≥ 5.7573 | 6.1% | 69.0% | **0.0000** (0/3138) |
| **P25** | gc ≥ 0.60549 AND ed ≥ 0.074354 AND lap ≥ 1098.3 AND blk ≤ 1.0154 AND ent ≥ 6.1976 | 10.4% | **80.6%** | **0.0000** |
| P50 | gc ≥ 0.64505 AND ed ≥ 0.12611 AND lap ≥ 2102.1 AND blk ≤ 1.0241 AND ent ≥ 6.7322 | 3.5% | 62.5% | 0.0000 |
| P75 | gc ≥ 0.69338 AND ed ≥ 0.18514 AND lap ≥ 2750.6 AND blk ≤ 1.0362 AND ent ≥ 6.9522 | 0.0% | n/a | 0.0000 |
| P90 | gc ≥ 0.72109 AND ed ≥ 0.2055 AND lap ≥ 3070.2 AND blk ≤ 1.0681 AND ent ≥ 7.0201 | 0.0% | n/a | 0.0000 |

### 一句话给领导（建议直接写进 §一）
> **红外**：画面质量达到**好片 P10** 那一档（对比度 ≥ 0.52、边缘密度 ≥ 0.014、拉普拉斯方差 ≥ 37.8、
> 块效应 ≤ 1.02、熵 ≥ 6.75），线上 **1 秒跟住率 97.6%**，坏片**误放率 0.0000**（3220 个坏片窗口 0 个通过）；
> 想更稳就上 P25（跟住率 88.4%、好片覆盖 23.6%）。
> **可见光**：需要达到**好片 P25** 那一档（对比度 ≥ 0.605、边缘密度 ≥ 0.074、拉普拉斯方差 ≥ 1098、
> 块效应 ≤ 1.015、熵 ≥ 6.20），线上跟住率 **80.6%**，坏片误放率 **0.0000**。
> **要点**：① 每条线只在本模态内有效，跨模态禁用（CHECK E1）；② gap 线只是「最优分界线」，
> **不是准入线**（CHECK E3）；③ 组合判据在本批数据上**对坏片 100% 拦截**，但样本只有 4 条坏源片，
> 上线前必须扩样重标。

### 产物（服务器）
/tmp/sot_report_long/sot_admission_lines.csv     ← 红线 / 推荐线 / ROC线 / gap线 总表
/tmp/sot_report_long/sot_combined_rules.csv      ← 组合判据（单指标线 AND）
/tmp/sot_report_long/sot_percentile_rules.csv    ← 好片分位刻度（★最终交付表）
/tmp/sot_report_long/sot_admission_transfer.csv  ← 跨模态误杀体检

## 12. 🔴🔴🔴 追加（最终交付表 · 两个时程都跑完了）：组合判据 = 「一组指标」的最终形态
`vq_sot_admission_verify.py` 的 **CHECK H**（用 5 个有效指标 global_contrast / edge_density /
lap_var / blockiness / entropy **同时过线**），阈值按「好片分布的第 P 分位」标定，
两个时程都已跑完，产物：
- 长时程(1秒)：`/tmp/sot_report_long/sot_percentile_rules.csv`
- 短时程(5帧)：`/tmp/sot_report_short/sot_percentile_rules.csv`
（本地已拉回：report/sot_percentile_rules.csv 与 report/sot_short/sot_percentile_rules.csv）

### 短时程（5 帧）—— **线上跟住率 100%，坏片误放 0.0000**
| 模态 | 好片分位 | 组合规则 | 好片覆盖 | **线上跟住率** | **坏片误放率** |
| --- | --- | --- | --- | --- | --- |
| IR | **P10** | gc ≥ 0.51601 AND ed ≥ 0.014121 AND lap ≥ 37.825 AND blk ≤ 1.0246 AND ent ≥ 6.7471 | 10.2% | **1.000** | **0.0000** (0/3220) |
| IR | P25 | gc ≥ 0.57543 AND ed ≥ 0.029497 AND lap ≥ 63 AND blk ≤ 1.0332 AND ent ≥ 7.1918 | 23.6% | **1.000** | 0.0000 |
| IR | P50 | gc ≥ 0.77068 AND ed ≥ 0.049584 AND lap ≥ 100.39 AND blk ≤ 1.0491 AND ent ≥ 7.3706 | 23.6% | **1.000** | 0.0000 |
| IR | P75 | gc ≥ 0.80117 AND ed ≥ 0.0585 AND lap ≥ 117.98 AND blk ≤ 1.1072 AND ent ≥ 7.4135 | 6.7% | **1.000** | 0.0000 |
| IR | P90 | gc ≥ 0.81132 AND ed ≥ 0.063481 AND lap ≥ 130.5 AND blk ≤ 1.6178 AND ent ≥ 7.4384 | 1.2% | **1.000** | 0.0000 |
| VIS | **P10** | gc ≥ 0.56166 AND ed ≥ 0.046798 AND lap ≥ 699.94 AND blk ≤ 1.0089 AND ent ≥ 5.7573 | 6.1% | **1.000** | **0.0000** (0/3138) |
| VIS | P25 | gc ≥ 0.60549 AND ed ≥ 0.074354 AND lap ≥ 1098.3 AND blk ≤ 1.0154 AND ent ≥ 6.1976 | 10.4% | **1.000** | 0.0000 |
| VIS | P50 | gc ≥ 0.64505 AND ed ≥ 0.12611 AND lap ≥ 2102.1 AND blk ≤ 1.0241 AND ent ≥ 6.7322 | 3.5% | **1.000** | 0.0000 |
| VIS | P75/P90 | — | 0.0% | n/a | 0.0000 |

### 长时程（1 秒）
| 模态 | 好片分位 | 组合规则 | 好片覆盖 | **线上跟住率** | **坏片误放率** |
| --- | --- | --- | --- | --- | --- |
| IR | **P10** | gc ≥ 0.51601 AND ed ≥ 0.014121 AND lap ≥ 37.825 AND blk ≤ 1.0246 AND ent ≥ 6.7471 | 10.2% | **0.976** | **0.0000** (0/3220) |
| IR | P25 | gc ≥ 0.57543 AND ed ≥ 0.029497 AND lap ≥ 63 AND blk ≤ 1.0332 AND ent ≥ 7.1918 | 23.6% | 0.884 | 0.0000 |
| IR | P50 | gc ≥ 0.77068 AND ed ≥ 0.049584 AND lap ≥ 100.39 AND blk ≤ 1.0491 AND ent ≥ 7.3706 | 23.6% | 0.768 | 0.0000 |
| IR | P75 | gc ≥ 0.80117 AND ed ≥ 0.0585 AND lap ≥ 117.98 AND blk ≤ 1.1072 AND ent ≥ 7.4135 | 6.7% | 0.481 | 0.0000 |
| IR | P90 | gc ≥ 0.81132 AND ed ≥ 0.063481 AND lap ≥ 130.5 AND blk ≤ 1.6178 AND ent ≥ 7.4384 | 1.2% | 0.600 | 0.0000 |
| VIS | P10 | gc ≥ 0.56166 AND ed ≥ 0.046798 AND lap ≥ 699.94 AND blk ≤ 1.0089 AND ent ≥ 5.7573 | 6.1% | 0.690 | **0.0000** (0/3138) |
| VIS | **P25** | gc ≥ 0.60549 AND ed ≥ 0.074354 AND lap ≥ 1098.3 AND blk ≤ 1.0154 AND ent ≥ 6.1976 | 10.4% | **0.806** | 0.0000 |
| VIS | P50 | gc ≥ 0.64505 AND ed ≥ 0.12611 AND lap ≥ 2102.1 AND blk ≤ 1.0241 AND ent ≥ 6.7322 | 3.5% | 0.625 | 0.0000 |
| VIS | P75/P90 | — | 0.0% | n/a | 0.0000 |

### 为什么「组合判据」才是最终答案（写 §9.5.9 时必须说清）
- 9.5.2 / 9.5.3 的**单指标单线**在长时程几乎全是「无解」——这是**真实结论**，不要掩盖；
- 但业务要的是「**一组指标**」：把 5 个有效指标**同时**卡，**坏片误放率在所有模态/所有档位都是 0.0000**
  （IR 3220 个坏窗 0 通过、VIS 3138 个坏窗 0 通过），而好片的线上跟住率大幅提升；
- 因此正确的上线口径是 **「5 项同时达标才接入」**（任一项不达标即拒接入），
  而不是「单指标达标即可接入」；
- 短时程（5 帧）口径最漂亮：**红外/可见光只要达到好片 P10 档，线上跟住率 100%、坏片误放 0%**；
- 长时程（1 秒）口径保守：红外 P10 档 97.6%、可见光需到 P25 档才有 80.6%。

### 建议新增 §9.5.9「组合判据（多指标同时过线）—— 上线口径」
表结构：`模态 | 好片分位 | 组合规则 | 好片覆盖 | 线上跟住率 | 坏片误放率`，
两个时程各一张；表头注明：
1. 5 个指标 = 第五节 AUC 0.874~0.919 的那 5 个（global_contrast / edge_density / lap_var / blockiness / entropy），
   `snr_db` / `noise_sigma` / `dog_snr_max` **不进组合**（AUC 0.111~0.333，无区分度）；
2. 阈值按**好片分布分位数**标定 ⇒ 领导可以直接问「要比好片中位数再好一档」；
3. 每条规则**只在本模态内有效**；
4. 🔴 本批坏源片**只有 4 条**（IR 2 + VIS 2），「坏片误放率 0.0000」是**小样本结论**，
   上线前必须扩样重标。

同时更新：**§9.5.8 一句话给领导** 与 **§一 的合格线表**（在「无解」的诚实结论之后，补上组合判据这一条可执行路径）。
