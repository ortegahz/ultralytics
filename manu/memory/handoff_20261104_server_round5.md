# 交接说明 · 2026-11-04 服务器夜间第 3 轮（服务器）→ host

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`）
> **线程 id**：`01a199f3-5523-7591-b039-9e07b7a1fb58`
> **本轮时段**：2026-11-04 02:00 → 03:40（服务器夜间自主推进 · 第 3 轮；host 已关机）
> **上一轮交接**：`handoff_20261104_server_round4.md`（本轮**未改**交付档，冠军仍是第 2 轮那档）

## 一、结论速览

**交付档配置本轮未改**（仍是第 2 轮的 `--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`）。
本轮产出三件事：

| # | 产出 | 关键数字 |
|:--|:--|:--|
| ① | **帧级精度损失对账**（objective 第 3 条的落盘证据） | 8 序：外挂独丢 **1260** 帧 / 独赚 **456** 帧 / **双失 2769（10.19%，检测器上限）**；净差距 **−2.9588 pp** |
| ② | ★ **锚点 oracle 天花板**（外挂精度的理论上界） | 8 序 **87.804%**（零延迟天花板 88.132%）；24 序 **85.522%**（天花板 85.896%）⇒ **再调重推参数最多再拿 +2.63 / +4.02 pp** |
| ③ | ★ 新机理「源帧无检测 ⇒ 再往回退重推」+ 回退链峰值置信门 | 8 序 `rb8 + mp0.50`：F1 **95.624%** / hit@5px **85.592%** / FAR **0.0171**（净收益）；但 **24 序 FAR 红线已饱和 ⇒ 破线**，故**默认关、不进交付档** |

## 二、★ 本轮最重要的结论（下一轮的指路灯）

1. **管线上报的位置 = 检测器的原始检测点**（不做 KF 平滑）：实测 001324 的 8990 个上报点里
   **8858 个（98.7%）与 `ms_cache` 原始点逐位相同**（差值 p50 = 0.0000 px）。
2. **外挂误差 = 源帧锚点误差 e(f) ＋ 位移估计残差 δ**，而 **δ 极小**：
   8 帧内目标位移 ≤ 50 px 时 **δ p50 0.55~1.13 px、p90 1.31~3.13 px**。
   ⇒ 外挂继承的是**检测器在源帧的单帧定位误差**，**不是**重推链的累积漂移。
3. ⇒ **继续调「快核 ROI / 模板 / 链参数」只能抠 δ 的尾巴，收益已封顶**。
   **要突破锚点 oracle，必须用图像证据重新定位、推翻管线锚点** —— 这是下一轮唯一该走的路。

## 三、本轮新产物（全部在服务器磁盘）

| 用途 | 路径 |
|:--|:--|
| 帧级损失归属（8 / 24 序） | `runs/lookahead/final/final8/diag_gap_attrib.json` · `runs/lookahead/final/final24/diag_gap_attrib.json` |
| 误差分解（8 序） | `runs/lookahead/final/final8/diag_err_decomp.json` |
| 锚点 oracle（8 / 24 序） | `runs/lookahead/final/final8/diag_oracle_bound.json` · `runs/lookahead/final/final24/diag_oracle_bound.json` |
| 回退档扫描（8 / 24 序） | `runs/lookahead/final/rb8/*.json` · `runs/lookahead/final/rb24/*.json` |
| **OSD 三臂对照补齐**（8 序 × 3 臂 = **24 段**） | `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4` + `runs/lookahead/osd/windows/<SEQ>_3arm.json` |
| 新脚本 | `manu/tracking/lookahead/diag_gap_attrib.py` · `diag_err_decomp.py` · `diag_oracle_bound.py` · `check_newknobs_off.sh` |
| 文档生成器（数字全来自产物，可重放） | `manu/tracking/lookahead/docgen/gen_docs_round5.py` · `gen_docs_round5b.py` · `land_round5.py` |

## 四、回归防线（本轮实跑，**全部 PASS**）

| 闸门 | 结果 |
|:--|:--|
| Gate A（活树直跑：原脚本 vs 新脚本 `--lookahead off`，整文件 md5） | 8 序 **IDENTICAL=9 / DIFFER=0**，F1 = 0.964002；24 序 **IDENTICAL=25 / DIFFER=0**，F1 = 0.947469 |
| 新旋钮**关闭** ⇒ 与交付档冠军产物逐位一致 | 8 序 **9/9**、24 序 **25/25**（`per_frame` + `metrics`） |
| `--lookahead off` ⇒ 与参考产物逐位一致 | 8 序 **9/9**、24 序 **25/25** |
| 单测 | `python -m unittest discover -s manu/tracking/tests -t .` ⇒ **Ran 152 tests OK (skipped=10)** |
| 导航表 | `python3 manu/memory/validate_nav_table.py` ⇒ **101/101 PASS** |

> ⚠️ 比对口径：`preds_*.json` 里 `config.cache_root` 字段的**相对 vs 绝对路径**会让整文件 md5 不同，
> 但 `per_frame` 与 `metrics` 逐位相同。Gate A 的标准口径是**同一次运行现跑原脚本 vs 现跑新脚本**。
> 复核脚本：`bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]`

## 五、本轮新增 CLI（**默认关 ⇒ 逐位不变**，交付档**未启用**）

```bash
--replay-rollback-max-frames 0   # 源帧（t−Δ）无检测时，最多往回退多少帧找锚点；0 = 不回退
--replay-rollback-min-peak  0.0  # 回退链的相关峰值置信门：峰值低于此值该帧不上报
```

板端契约：`manu/memory/lookahead_board_port_contract_20261103.md` **§3.8 契约八**（+ §4 真值表两行、§7 风险 R18/R19）。

## 六、下一步（host 开机后可直接接续）

1. **攻「图像重定位」**：目标 = 突破锚点 oracle（8 序 87.804% / 24 序 85.522%）。
   候选机理：用**模板在源帧内做一次小范围重定位**（把管线锚点当**搜索中心**而不是**模板原点**），
   拿到比管线锚点更准的 R(f)，再按 `R(f) + 位移` 上报。
   ⚠️ 注意：上报坐标必须与检测器的**框中心约定**一致（纯强度质心与 GT 框中心有 1~3 px 系统偏置）。
2. **不要碰**已被实测否决的旋钮家族（见专题档 §27.4 / §28）。
3. 若团队愿意放宽 24 序 FAR 红线，可直接启用 `rb8 + mp0.50`（24 序 F1 93.277% / hit@5px 81.81% / FAR 0.0321）。
