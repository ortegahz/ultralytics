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
