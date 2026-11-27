# 服务器续跑会话 · 2026-11-26

> host 工作站 18:00 关机 ⇒ 「红外图像质量评估 v2 · 新的 v2 指标搜索 + 详细报告」**已转到服务器上继续跑**。

## 会话标识

| 项 | 值 |
|:--|:--|
| 服务器 | `huangzhe@192.168.99.40:32222` |
| 工作目录 | `/tmp/pycharm_project_a7d966e6`（服务器工程根） |
| **codex thread id** | **`01a21723-d2d3-78b1-8a7b-1101872510ef`** |
| **实时日志** | `/tmp/codex_iqa_v2.log`（JSONL，`codex exec --json`） |
| 启动命令 | `codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C /tmp/pycharm_project_a7d966e6 --json - < /tmp/codex_prompt_iqa_v2.txt`（`setsid` + `nohup` ⇒ **与 host 断连后仍存活**） |
| 启动提示词 | `/tmp/codex_prompt_iqa_v2.txt`（3715 B） |
| 交棒书 | `/tmp/pycharm_project_a7d966e6/iqa_v2/HANDOVER_SERVER_20261126.md`（213 行） |
| 任务书原文 | `/tmp/pycharm_project_a7d966e6/iqa_v2/GOAL_OBJECTIVE.md`（91 行） |

## 监控命令（host 回来后）

```bash
SSH='ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 32222 huangzhe@192.168.99.40'

# 1) 还在跑吗 + 最新进度
$SSH 'ps -eo pid,etime,cmd | grep "codex exec" | grep -v grep; tail -c 3000 /tmp/codex_iqa_v2.log'

# 2) 人话版进度（最近 10 条 agent 消息）
$SSH 'grep -o "\"type\":\"agent_message\",\"text\":\"[^\"]*\"" /tmp/codex_iqa_v2.log | tail -10 | cut -c1-260'

# 3) 缺口留痕
$SSH 'cat /tmp/pycharm_project_a7d966e6/iqa_v2/PROGRESS_SERVER.md'

# 4) 交付物
$SSH 'ls -la /tmp/pycharm_project_a7d966e6/iqa_v2/REPORT_V2_FINAL.md \
        /tmp/pycharm_project_a7d966e6/iqa_v2/MEMORY_SECTION.md \
        /tmp/pycharm_project_a7d966e6/iqa_v2/output/'

# 5) 断线后续跑（会话 id 复用）
$SSH 'export PATH=$HOME/.local/bin:$PATH; cd /tmp/pycharm_project_a7d966e6 && \
      codex exec resume 01a21723-d2d3-78b1-8a7b-1101872510ef "继续：检查 G1~G9 进度，补完剩余缺口并产出 REPORT_V2_FINAL.md"'
```

## 交棒时的状态快照

- 17:49 CST：会话已启动 **~1 min**，已读完交棒书 + 任务书 + `degradation_4d.py` + `target_roi_metrics.py`，
  正在分析 `run_eval_v2.py` 结构并**并行派了 explorer**。
- 代码基线：host 与服务器 `iqa_v2/*.py` **8 个文件 md5 全部一致**（`8622fc0c82 / 6e47b22ab6 / cc66815f9a /
  6bd2fc02dc / 27f7a3df33 / f7e36318a2 / 272bca85d4 / 3fdd3effc8`）⇒ 服务器侧副本即最新，无需再同步。
- 服务器侧 `manu/memory/memory.md` **不要改**（会与 host 权威副本分叉）⇒ 成稿章节写 `iqa_v2/MEMORY_SECTION.md`。

## 追加：host 侧已做的两件准备（不与服务器侧改同一文件，无冲突）

| 文件 | 作用 |
|:--|:--|
| `iqa_v2/collect_from_server.sh` | 回收服务器交付物 + 代跑三道门禁（pytest / verify_recalc / md5sum -c，**host 无 cv2 只能在服务器跑**）+ 两侧逐字节 md5 比对。拉进 `iqa_v2/server_out/`，**不覆盖工作副本**。 |
| `iqa_v2/audit_final.py` | **完成度审计**：把任务书逐条要求变成机器可判的检查（R1 退化轴 / R2 底座纯净度 / R3 序列级切分无交叉 / R4 三分法回归三指标 / R5 三项核心指标约束线 + v1 全图不入选 / R6 免责声明逐字 / R7 小目标 / R8 双算法与跨家族不合并 / R9 交付模型条件项）。**已用旧产物自测：报出 8 FAIL + 1 WARN，与已知缺口一致 ⇒ 工具本身有牙**。 |

## 追加：G7（正式交付模型替换）已**预先查清**（host 侧核查，服务器不必重查）

- `纵横AI数据采集_preannot_rc2/` 下 **58 条** `scores.json` 全部是 **2026-11-09 14:50** 生成的**模型预标注**（软分 0.77~0.85），
  **整个目录最新 mtime 仍是 2026-11-09** ⇒ §315（2026-11-25 发出三份交付）之后**人工标注尚未回收**。
- ⇒ **G7 结论：保持 XCorr + KCF，不替换 `trial_0080`/RC2**；报告须写明「正式交付模型替换待业务标注回收后按同口径复测执行」。

---

## 追加二：2026-11-27 上午 · 会话已接回 + 独立验收 + 记忆库合并

> host 工作站 2026-11-26 18:00 关机，2026-11-27 上午回来。服务器那一轮已于 **11-26 18:48 `turn.completed`**（正常结束）。

### 1) 会话状态与接回方式

| 项 | 值 |
|:--|:--|
| 服务器那一轮 | `codex exec`（pid 2505229）跑 17:15→**18:48**，末条事件 `turn.completed`；日志 `/tmp/codex_iqa_v2.log`（3.5 MB，JSONL） |
| 会话记录 | `/home/huangzhe/.codex/sessions/2026/11/26/rollout-2026-11-26T17-15-53-01a21723-d2d3-78b1-8a7b-1101872510ef.jsonl`（3883 行） |
| **接回命令（已验证可用）** | `codex exec resume --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --json <thread_id> - < prompt.txt`（**必须带 `--skip-git-repo-check`**，否则报「Not inside a trusted directory」） |
| 2026-11-27 复跑 | 已用该命令把同一 thread 接回并追加一轮「跨机字节级核对」，完成后正常退出；服务器已追加留痕到 `PROGRESS_SERVER.md` |
| 对话全文（host 侧） | `iqa_v2/server_out_final/conv_digest.md`（从 rollout jsonl 抽取的 133 条 assistant 消息 + 工具调用统计） |

### 2) 交付物回收（host 侧）

`iqa_v2/server_out_final/files/`：`REPORT_V2_FINAL.md`(77,729 B) · `MEMORY_SECTION.md`(22,288 B) ·
`PROGRESS_SERVER.md`(50,150 B) · `audit_final.py`(33,827 B) · `run_eval_v2.py` · `verify_recalc_v2.py` ·
`test_iqa_v2.py` · `degradation_4d.py` · `target_roi_metrics.py` · `dataset_anti_uav.py` · `make_manifest_v2.py` ·
`output/`（12 CSV，含 `degrade_detail_v2.csv` 8.1 MB）—— 共 18 MB，**与服务器原件 18/18 md5 逐条一致**。

### 3) host 独立验收（详见 `iqa_v2/HOST_ACCEPTANCE_20261127.md`）

- 四道门禁 host 亲跑：pytest **44/44** · verify_recalc 误差 **3.553e-15**、膝点复算 **0.000e+00**、曲线 **162/162** · md5sum -c 全过 · 审计 PASS
- **host 独立审计器自身有 5 处缺陷（已修）**：最关键一条是 `binned_curve` 误用全量均值把曲线压平 ⇒ 产生 3 个假 FAIL；
  修好后独立复算**复现出约束线数值**（0.18917 / 5.82855）⇒ 证明原缺陷在仪器侧、不在交付物侧
- 修好后 host 审计：**PASS 37 / FAIL 0 / WARN 1**（R9 人工判读为通过）
- host 修好的审计器已同步到服务器 `iqa_v2/audit_final_host.py`（**不覆盖**服务器自己的 `audit_final.py`）

### 4) 记忆库合并（host 权威副本，已完成）

`bash iqa_v2/merge_memory.sh iqa_v2/server_out_final/files/MEMORY_SECTION.md` ⇒ 落为 **§319**
（278 行正文逐行全等；roll/check/nav 三绿；备份 `manu/memory/memory.md.bak_20261127_085624`）。

### 5) 本轮唯一成立的红线

`target_local_contrast < 0.1891664`（xcorr/motion）。
`target_lap_var`（任务书 819.55）**本轮无有效红线**、`local_scr`（任务书 2.69~3.21）**盲测区分间隔仅 1.60 pp 不成立**
⇒ 两条都不写进对外免责声明，理由与差异逐条列在报告 §4.2 / §6 / §10.2~10.3 / §12。
