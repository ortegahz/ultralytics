# 服务器夜间作业包 · Look-ahead 快速外推外挂

## 你现在只需要做一件事

在**你自己的终端**（不是 agent 的沙箱，本机或笔记本均可，能 SSH 到服务器）执行：

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics
bash manu/tracking/lookahead/server_overnight/push_and_start.sh
```

它做三件事：
1. `rsync` 把 `manu/tracking/`（含全部外挂代码 + 本作业包）与 `manu/memory/` 两个记忆文档同步到
   `huangzhe@192.168.99.40:/tmp/pycharm_project_10ae9e2e`；
2. 打印关键基线文件的 md5 供核对（本地与服务器基线已验证一致：`system_sota_longquan.py` = `3361e7b7…`）；
3. 在服务器上 `setsid nohup` 拉起夜间作业：
   - **确定性评测**（不依赖 LLM）：环境自检 → 单测 → 24 序列原逻辑/臂1/臂3 三臂 → 对照表；
   - **服务器端 codex 会话**：`codex exec < TASK.md`，TASK.md 是本会话的完整上下文 + 今晚的作业计划 +
     验收标准，服务器上的 agent 会**继续自己推进**（实现 rolling chain、扫参、出表、写记忆文档），
     进度实时写到 `$OUT/progress.md`。

> 本机 18:30 关机不影响服务器作业：远端是 `setsid nohup` 独立会话。

## 如果你想手动分步来
```bash
ssh -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_10ae9e2e
bash manu/tracking/lookahead/server_overnight/start_on_server.sh
```

## 早上 9 点验收
```bash
ssh -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_10ae9e2e
cat runs/lookahead/server_out/comparison_24.md     # 三臂对照表（主交付）
cat runs/lookahead/server_out/progress.md          # 一句话结论 + 卡点
tail -80 runs/lookahead/server_out/codex_session.log
```

## 产物清单
| 路径 | 内容 |
|---|---|
| `runs/lookahead/server_out/original_24.json` | 原代码 24 序列基线（应复现 F1≈94.747%） |
| `runs/lookahead/server_out/gateA_off.json` | Gate A：外挂关闭与原代码逐位一致的证据 |
| `runs/lookahead/server_out/arm1_stale_24.json` | 臂1 纯检测基线（硬吃 320ms，≈77.2%） |
| `runs/lookahead/server_out/arm3_replay_24.json` | 臂3 外挂结果 |
| `runs/lookahead/server_out/comparison_24.md` | **对照表**（F1 / FAR / hit@5px / CLE p50,p95 / max_jitter / frag） |
| `runs/lookahead/server_out/progress.md` | 进度与卡点 |
| `runs/lookahead/server_out/codex_session.log` | 服务器端会话日志 |
| `runs/lookahead/server_out/eval.log` | 确定性评测日志 |

## 包内文件
```
TASK.md                    # ★ 服务器端 agent 的完整任务书（上下文+计划+验收标准）
push_and_start.sh          # ★ 你要执行的那一条
start_on_server.sh         # 服务器端总控（起确定性作业 + 起 codex exec）
run_overnight_eval.sh      # 确定性三臂评测（LLM 挂掉也能出数）
docs/design_spec_user.txt  # 用户原始设计书（滚动链/环形缓冲）
docs/lookahead_extrapolator_20261102.md  # 记忆文档（含 §11~§16 对照表与踩坑记录）
docs/rules.md              # 仓库铁律
```
