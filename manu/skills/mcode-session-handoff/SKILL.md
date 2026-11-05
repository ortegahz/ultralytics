---
name: mcode-session-handoff
description: 把当前 MCode/MiniMax Code 对话交接到训练服务器，让服务器接管同一会话继续干。覆盖 MCode 的会话存储布局（v2/sessions + runtime-state.sqlite）、可见性规则（workspace_dir 精确匹配）、搬运清单（messages.jsonl 为唯一真值、checkpoint 行必搬）、activeRevision 指纹算法（含三个必踩的坑）、一键命令与只读验收。触发语："把 mcode 对话交接到服务器"、"让服务器接着干"、"host 要关机了"、"服务器上看不见这个会话"、"mcode 会话怎么搬"、"会话迁移到训练机"。
---

# MCode 会话跨机交接规程

> 把 host 上的 MCode 对话搬到训练服务器，服务器用 `mcode --session <id>` 接手续干。
> 本文所有结论均在 **MCode 0.6.12（host）/ 0.6.13（server）** 上实测通过。

---

## 0. 一句话速查

```bash
# 一键端到端（export → 传输 → import → verify → 实际加载测试）
bash /media/manu/1TB-Volume/workspace/ultralytics/manu/skills/mcode-session-handoff/scripts/mcode_handoff_to_server.sh \
     --session mvs_<会话ID> --exec
```

反向（把 server 上的活儿拿回 host）：

```bash
# 不带 --force 只做差异预览（哪边更新、会不会覆盖）
bash .../scripts/mcode_pullback_from_server.sh --session mvs_<会话ID>

# 确认后再真正回拉 + 加载测试
bash .../scripts/mcode_pullback_from_server.sh --session mvs_<会话ID> --force --exec
```

⚠️ **推是单向快照，不是双向同步** —— 回拉会覆盖本机较新的轮次，详见 §10。

三步命令（要自己控制每一步时）：

```bash
cd /media/manu/1TB-Volume/workspace/handoff/mcode

# ① 本机抽出（生成含逐文件 sha256 的包）
python3 mcode_handoff.py export --session mvs_<ID> --out /tmp/pack.json

# ② 传到目标机（必须比对 md5）
scp -P 32222 /tmp/pack.json huangzhe@192.168.99.40:/tmp/pack.json

# ③ 目标机上导入（需 node，见 §6）
ssh -p 32222 huangzhe@192.168.99.40 \
  'export PATH=$(ls -d $HOME/.minimax-code/runtime/node-v*/bin | head -1):$PATH; \
   cd /tmp/mcode_handoff_tools && \
   python3 mcode_handoff.py import --pack /tmp/pack.json \
     --data-dir ~/.minimax --workspace /tmp/pycharm_project_10ae9e2e'

# ④ 只读验收
ssh -p 32222 huangzhe@192.168.99.40 \
  'export PATH=$(ls -d $HOME/.minimax-code/runtime/node-v*/bin | head -1):$PATH; \
   cd /tmp/mcode_handoff_tools && \
   python3 mcode_handoff.py verify --session mvs_<ID> --data-dir ~/.minimax'

# ⑤ 真正的加载测试（最有说服力的验收）
ssh -p 32222 huangzhe@192.168.99.40 \
  'cd /tmp/pycharm_project_10ae9e2e && ~/.minimax-code/bin/mcode exec \
     --session mvs_<ID> --output-format json --max-steps 1 "reply OK only"'
```

---

## 1. MCode 会话存在哪（与 Codex 完全不同）

Codex 是单目录 + SQLite 投影；**MCode 是「目录 + SQLite 双写」**，两套东西都得搬。

```
~/.minimax/                                  ← dataDir（可用 $MINIMAX_DATA_DIR 覆盖）
├── v2/
│   ├── sessions/YYYY/MM/DD/HH-MM-SS-mmm-session_<suffix>/
│   │   ├── messages.jsonl          ★★★ 对话真值（每行一条 envelope）
│   │   ├── history-catalog.json    ☆ 派生索引（可自动重建）
│   │   ├── user-message-locators.jsonl  ☆ 派生索引（可自动重建）
│   │   ├── manifest.json           身份校验用（paths.* 是绝对路径）
│   │   ├── snapshots/
│   │   │   ├── g<gen>--ctx_<id>.jsonl   压缩代快照（多代会话必备）
│   │   │   └── env-g<gen>--ctx_<id>.json
│   │   ├── llm-call.json           诊断用
│   │   └── reports/                诊断用
│   └── sqlite/runtime-state.sqlite ← 索引库
│       ├── local_runtime_sessions              会话主表（**列表可见性在这里**）
│       ├── local_runtime_message_rows          UI 显示投影
│       ├── local_runtime_thread_goals           ★ 目标（与 session 1:1 UNIQUE）
│       ├── local_runtime_pi_history_file_migrations  ★★★ checkpoint（见 §4）
│       ├── local_runtime_message_row_migrations      display 回填标记
│       ├── local_runtime_projects               工作区（workspace_dir → project_id）
│       ├── local_runtime_session_locks           执行权租约（接手要清）
│       └── local_runtime_messages                legacy blob（本机实测全空）
└── sessions/  ← 旧版布局，不必管
```

**目录后缀规则**：`session_<suffix>` 中 suffix = `base64url(session_id)`（**含 `mvs_` 前缀，不剥离**）取前 180 字符。
实测：`mvs_dd1c49ef29c74a8c8932256ee8c33175` → `bXZzX2RkMWM0OWVmMjljNzRhOGM4OTMyMjU2ZWU4YzMzMTc1`。

**时间戳段用 UTC**（`getUTC*`），与源机本地时区无关 ⇒ 跨时区搬运目录名不变。

---

## 2. 可见性规则（决定「服务器上看不见这个会话」）

MCode 的会话列表按 **`workspace_dir` 精确匹配**过滤（Codex 是 cwd 精确匹配，语义一致）：

- `zgr()`：POSIX 路径 `workspace_dir = ?`（**区分大小写**，去尾斜杠）；Windows 才做 `lower()` 比较
- `Kgr()`：`visibility <> 'hidden'`
- `Vgr()`：`archived = 0/1`
- `PIe()`：只列根会话（`NOT EXISTS` 子查询）
- `project_id` **不参与过滤**，只用于侧边栏分组

⇒ **交接时必须把 `workspace_dir` 改成目标机上该项目的真实路径**，否则会话存在但列表里不显示。

⚠️ **`workspace_dir` 存在两处，必须一起改**：
1. 独立列 `local_runtime_sessions.workspace_dir`
2. **`record_json` 里的 `workspaceDir` 字段** ← 只改列 ⇒ 运行时仍按旧路径找历史文件

（`record_json` 与独立列本身在 MCode 里就可能不一致 —— 它是 v1 遗留的序列化快照，`archived`/`visibility` 经常与列值不同。**以列为准，但 `workspaceDir` 两处都要改。**）

---

## 3. 搬运清单

### 必搬

| # | 内容 | 位置 | 漏了会怎样 |
|---|---|---|---|
| 1 | **会话目录整棵** | `v2/sessions/<rel>/` | 会话不存在 |
| 2 | `messages.jsonl` | 目录内 | **对话内容丢失** |
| 3 | `snapshots/` | 目录内 | **多代压缩会话直接加载失败** |
| 4 | `manifest.json` | 目录内 | 身份校验失败（`Session history manifest identity mismatch`） |
| 5 | `local_runtime_sessions` 该行 | SQLite | 列表里看不到 |
| 6 | `local_runtime_thread_goals` 该行 | SQLite | **目标（goal）丢失** |
| 7 | `local_runtime_pi_history_file_migrations` 该行 | SQLite | **★ 加载直接失败**（见 §4） |
| 8 | `local_runtime_message_rows` 该会话全部行 | SQLite | TUI 历史回看为空 |
| 9 | `local_runtime_message_row_migrations` 该行 | SQLite | display 行不回填 |

### 不用搬 / 要清理

- `local_runtime_session_locks`：**接手时删掉该会话的锁行**（执行权租约，带过去会挡住运行）
- `history-catalog.json` / `user-message-locators.jsonl`：派生索引，运行时发现不一致会**自动重建**（`loadOrRebuild`）；搬了就必须自洽
- `local_runtime_messages` / `local_runtime_pi_history_rows`：legacy 迁移源，本机实测全空
- `local_runtime_ledger_watermarks` / `local_runtime_session_projection_watermarks`：实测 0 行
- `local_runtime_sessions_fts*`：全文索引，只影响搜索

---

## 4. ★★★ 最深的坑：checkpoint 行缺失 ⇒ 目标机永远加载不了

**症状**（目标机 `mcode exec --session <id>`）：

```json
{"status":"failed","error":{"message":"Conversation history could not be safely updated. Please retry."}}
```

日志里真实原因（`~/.minimax/v2/observability/logs/runtime-*.log`）：

```
"Canonical history materialization is ambiguous: mvs_xxx"
  at QIt (chunk-7ONY7WH4.js:5115)  ← 0.6.13
  at async aY / d / u / run / read / rU.readCommitted
```

**根因**（读 0.6.13 chunk 源码 `aY()` 得到）：

```js
async function aY(e) {
  if (!e.legacyHistory) { ...; return }                    // ① 无 legacy → 直接用 messages.jsonl
  if (await e.legacyHistory.checkpoints.getCheckpoint(e.sessionId)) { ...; return }  // ② 有 checkpoint → 直接用 messages.jsonl
  let n = await e.files.readTarget(e.paths.messages),      // n = 文件里的 255 条
      r = await $br(e),                                     // r = legacy 源（空！）
      i = zbr(e.sessionId, r.records),                      // i = 空
      s = n === undefined ? await qbr(e, i.items) : await QIt(e, n, i.items);
  //                                                              ↑ n 有值 ⇒ 进 QIt ⇒ n(255) vs i(0) ⇒ 指纹不等 ⇒ ambiguous
}
```

`checkpoints.getCheckpoint()` 读的是 **`local_runtime_pi_history_file_migrations`** 表（`session_id` 主键），
**不是** `local_runtime_legacy_migrations`（那个表本机实测一直是 0 行，容易查错地方）。

⇒ **目标机没有这一行 ⇒ 运行时以为这个会话需要 legacy 迁移 ⇒ 从空的 legacy 源重建物化结果 ⇒ 与 `messages.jsonl` 指纹不等 ⇒ 拒绝加载。**

**处方**：搬运时必须带上源机这一行：

```sql
INSERT OR REPLACE INTO local_runtime_pi_history_file_migrations
  (session_id, migrated_at_ms, source, message_count, target_revision)
VALUES (?, ?, 'empty', 0,
        'sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945');
```

（`target_revision` 是空数组的指纹。源机若没有这行 —— 极早期会话 —— 目标机也要造一条等价的 `source='empty'` 行，否则同样会误入 legacy 路径。）

**诊断手法**：给目标机 chunk 打一行 patch 打印三个指纹对比，能一眼定位：

```js
// QIt 里加：
console.error("[PROBE] len(t)="+t.length+" len(r)="+r.length+" len(n)="+n.length
  +" Zr(t)="+Zr(t)+" Zr(r)="+Zr(r)+" Zr(n)="+Zr(n));
```

实测输出（搬运会话）：`len(t)=255 len(r)=255 len(n)=0` ⇒ 确认 legacy 源为空。
（⚠️ 打 patch 后 mcode 可能报 `missing ) after argument list`，恢复原文件即可。）

---

## 5. activeRevision 指纹算法（★ 三个必踩的坑）

MCode 用 `activeRevision` 校验历史文件是否被篡改/是否自洽：

```
activeRevision = "sha256:" + sha256(
    "[" +
    records.map(r => JSON.stringify(sortKeysDeep(r))).join(",") +
    "]"
  )
```

`sortKeysDeep` = 递归按 key 排序（只对 plain object）。

### 坑 1：分隔符

必须用 `JSON.stringify` 的**默认分隔符**（`, ` / `: `），**不是**紧凑格式 `(',',':')`。
用紧凑格式算出来的值和 catalog 对不上。

### 坑 2：Python 不能重写

`json.dumps(..., ensure_ascii=True)`（Python 默认）会把非 ASCII 转成 `\uXXXX`，Node 不会 ⇒ 哈希不同。
**实测**：某会话 281 条记录里 93 条含非 ASCII，Python 算出的值始终对不上；改用 Node 后逐位一致。
⇒ **指纹计算一律用 Node**（`mcode_session_tool.mjs` 就是干这个的），别用 Python 重写。

### 坑 3：不是文件字节的哈希

对 `messages.jsonl` 文件字节做 sha256 **不等于** `activeRevision`。
它是「规范化后的记录数组」的哈希。同一文件也可能因为换行/尾部空白导致字节哈希变化而指纹不变。

### 自校验

```bash
node mcode_session_tool.mjs verify <会话目录>   # 退出码 0 = catalog 与实算一致
```

在 40 个真实会话上实测 **40/40 通过**。

### catalog 不一致会怎样

`history-catalog.json` 校验失败（`sessionId`/`activeGeneration`/`activeRevision` 对不上）⇒
抛 `Stale canonical history index` ⇒ 运行时**自动重建** catalog ⇒ 一般不致命。
但 `locators` 里的 `artifactRevision` 若与 catalog 对不上也会触发重建。

---

## 6. 目标机前置条件

| 条件 | 说明 |
|---|---|
| **node** | 指纹计算依赖 Node。服务器上系统 node 常常没有，用 MCode 自带的：`~/.minimax-code/runtime/node-v24.19.0-linux-x64/bin/node` |
| ⚠️ PATH 写法 | `export PATH=$HOME/.../node-v*/bin:$PATH` **不做 glob 展开**！必须先算出绝对路径：`export PATH=$(ls -d $HOME/.minimax-code/runtime/node-v*/bin \| head -1):$PATH` |
| MCode 已安装 | `~/.minimax-code/bin/mcode`，版本与 host 差一两个小版本不影响（0.6.12 ↔ 0.6.13 实测互通） |
| `dataDir` | 默认 `~/.minimax`；可用 `--data-dir` 指定 |
| workspace 目录 | 目标机上该项目目录必须已存在，且已登记进 `local_runtime_projects`（脚本会自动建 project 行） |
| 登录态 | `~/.minimax/auth` 需要有效，否则 `mcode exec` 会失败（但这与交接无关） |

---

## 7. 只读验收（不消耗模型调用）

`mcode_handoff.py verify` 输出这些字段，**全绿才算交接完成**：

| 字段 | 期望 |
|---|---|
| `ok` | `true` |
| `dir_name_matches_suffix_rule` | `true`（目录名后缀 == base64url(session_id)） |
| `catalog_matches` | `true`（catalog 指纹 == 实算指纹） |
| `project_workspace_dir_ok` | `true`（project 的 workspace_dir == session 的） |
| `lock_rows` | `0` |
| `history_checkpoint_present` | **`true`**（★ 缺这个目标机加载不了） |
| `goal_rows` | ≥1（若源会话有 goal） |

**最终验收 = 真跑一次**：

```bash
mcode exec --session <id> --output-format json --max-steps 1 "reply OK only"
```

- `status: "succeeded"` ⇒ 通过
- `usage.inputTokens` 很大（> 10 万）⇒ 历史上下文确实被载入了（而不是开了个空会话）
- `usage.cacheReadTokens` 很大 ⇒ 命中了 host 侧的 prompt 缓存，说明上下文逐位一致

---

## 8. 只搬 `conversation`，不搬 `task`

`mcode exec --session <task型会话id>` 会报：

```
Sub-agent Sessions are internal and cannot be opened.
```

`session_kind='task'` 是子 Agent 会话，**不能作为交接对象**。
`session_kind='conversation'` 才能被 `--session` 打开。

查询可搬的会话：

```bash
python3 mcode_handoff.py list --workspace /media/manu/1TB-Volume/workspace
```

---

## 9. 与 Codex 交接机制的对应关系

| Codex | MCode |
|---|---|
| rollout jsonl | `messages.jsonl`（同样每行一条，但 envelope 结构不同） |
| `session_meta.cwd` | `workspace_dir` 列 **+ `record_json.workspaceDir`**（两处！） |
| `threads.rollout_path` | `local_runtime_sessions.history_relative_dir` |
| `thread_history_1.sqlite` 物化 | `history-catalog.json` + `user-message-locators.jsonl`（可自动重建） |
| `goals_1.sqlite`（goal 不随交接走） | `local_runtime_thread_goals`（**随会话同库，一行搬过去就行**） |
| `history_mode=paginated` 必须 | `local_runtime_pi_history_file_migrations` checkpoint 行必须（**MCode 独有的第三个坑**） |
| `thread/fork` 产物不可见 | 同理，别用 fork 思路 |
| `codex resume <id>` | `mcode --session <id>` / `mcode exec --session <id>` |

**MCode 独有的两个坑**（Codex 版本里没有）：
1. `record_json.workspaceDir` 与独立列**双写**，不改两处会按旧路径找文件
2. `pi_history_file_migrations` checkpoint 行缺失 ⇒ 误入 legacy 迁移路径 ⇒ ambiguous

---

## 10. 双向语义：推是单向快照，不是双向同步

**两个方向都已经实测跑通**（2026-11-05，host 0.6.12 ↔ server 0.6.13）：

| 方向 | 脚本 | 实测结果 |
|---|---|---|
| host → server | `mcode_handoff_to_server.sh` | 257 条/1 代 + 439 条/13 代，均 `succeeded` |
| server → host | `mcode_pullback_from_server.sh` | 261 条 + 441 条，均 `succeeded` |
| host → server → host | 手动串 | **前 261 行逐字节完全相同**，差异仅为新 append 的 2 条 |
| host → server → host → server | 手动串 | `succeeded`，217572 tokens |

### ⚠️ 三个必须知道的语义

**(1) 推是覆盖，不是合并**

`import` 会 **先 `rmtree` 目标目录再写入**（无自动备份）。⇒ 回拉会用 server 版本**整体覆盖** host 版本，
host 上较新的轮次**直接丢失**。

回拉脚本默认**不带 `--force` 就只做差异预览**，明确告诉你哪边更新：

```
本机   : msg=439 rev=sha256:7bb5882fd52ed7764
服务器 : msg=441 rev=sha256:4f3ab8e6b111d9162
⇒ 服务器更新（多 2 条），回拉会推进本机
```

若显示「本机更新（多 N 条）」，回拉会**覆盖掉本机较新的轮次** —— 此时应先决定是否放弃本机进展。

**(2) 交接后目标机是对话的所有者**

host 再加的轮次不会自动出现在 server 上；重跑 handoff 会用 host 副本**再次覆盖** server。
⇒ **不要在两边同时继续同一个会话**。正确用法是「谁接手谁独占」。

**(3) 版本兼容已验证，但别赌**

0.6.13 写过的 `messages.jsonl` / `history-catalog.json` 被 0.6.12 正常读取并加载（实测 `succeeded`），
反向也通。**但这是两个相邻小版本**；跨大版本（0.6.x ↔ 0.7.x）前应先在隔离目录验一次：

```bash
# 用 MINIMAX_DATA_DIR 指向隔离副本做无破坏验证
MINIMAX_DATA_DIR=/tmp/xxx mcode exec --session <id> --output-format json --max-steps 1 "reply OK only"
```

（MCode 认 `MINIMAX_DATA_DIR`，也认 `MAVIS_DATA_DIR`。隔离副本里需要 `auth/` 与 `config.yaml`，
否则会报 `Sign in to MiniMax to use Agent features`。）

### 身份字段不随路径改写而变

`manifest.json` 的 `sessionId` / `createdAtMs` / `schemaVersion` 在往返中**逐位未变**，
只有 `paths.*` 被改写成当前机器路径 —— 这是必须的，否则运行时按旧路径找历史文件。

### 其他注意事项

- **包的体积**：base64 编码后约为原目录的 1.37 倍（40 MB 目录 → 72 MB 包，13 代会话 224 文件）。
  大会话传输要留时间。
- **别用 symlink 装 skill**：MCode 运行时枚举 skill 目录时用 `entry.isDirectory()`（lstat 语义），
  对符号链接返回 false ⇒ symlink 装的 skill 会被**整批静默跳过**。必须用真实目录 + `cp -r`。
- **备份建议**：真正覆盖前先 `cp -a` 备份会话目录 + 用 `sqlite3` 导出相关表行。
  本次实测的备份方式见 §13。

---

## 11. 已验证的实测记录（2026-11-05）

**单向（host → server）**

| # | 会话 | 规模 | 压缩代数 | 结果 |
|---|---|---|---|---|
| 1 | `mvs_d81a182fa1e5401c8dd21eec4a1317c9` | 257 条 / 29 文件 | 1 代 | `succeeded`，inputTokens=215278 |
| 2 | `mvs_80d99b058e8249fbbcd785c4e2f53064` | 99 条 / 224 文件 / 2708 投影行 | **13 代**（14 artifact） | `succeeded`，inputTokens=115470，cacheRead=116455 |

**双向（往返）**

| # | 场景 | 结果 |
|---|---|---|
| 3 | server → host 回拉（261 条） | host `succeeded`，inputTokens=216461 |
| 4 | host → server → host 往返 | **前 261 行逐字节完全相同**，差异仅为新 append 的 2 条（我实测的那次 exec） |
| 5 | host → server → host → server | `succeeded`，inputTokens=217572 |
| 6 | 439 条 / 13 代会话 host → server | `succeeded`，inputTokens=305517 |
| 7 | server → host 回拉（441 条，13 代） | host `succeeded`，inputTokens=305862 |

指纹校验：**40 个真实会话 40/40 通过**。

---

## 12. 相关文件

| 路径 | 作用 |
|---|---|
| `scripts/mcode_handoff.py` | 核心：export / import / verify / list |
| `scripts/mcode_session_tool.mjs` | 指纹计算 + 打包（**必须用 Node，不要用 Python 重写**） |
| `scripts/mcode_handoff_to_server.sh` | 一键端到端：host → server |
| `scripts/mcode_pullback_from_server.sh` | 一键端到端：server → host（带差异预览 + `--force` 保护） |
| `/media/manu/1TB-Volume/workspace/handoff/mcode/` | 同上三个脚本的工作副本（md5 一致） |
| `/media/manu/1TB-Volume/workspace/handoff/README.md` | Codex 版交接文档（本 skill 的姊妹篇） |
| `~/.minimax/v2/observability/logs/runtime-*.log` | 目标机加载失败时的真实报错出处 |

---

## 13. 覆盖前备份（实测用过，可靠）

```bash
SID=mvs_<会话ID>
BK=/tmp/host_backup_$SID
rm -rf $BK && mkdir -p $BK

# ① 会话目录
REL=$(python3 -c "
import sqlite3
c=sqlite3.connect('$HOME/.minimax/v2/sqlite/runtime-state.sqlite')
print(c.execute('SELECT history_relative_dir FROM local_runtime_sessions WHERE session_id=?',('$SID',)).fetchone()[0])
")
cp -a ~/.minimax/v2/sessions/$REL $BK/sessiondir

# ② 相关表行（JSON 导出，含自增 id 需自行处理）
python3 - <<'EOF'
import sqlite3, json, os
SID = os.environ.get("SID") or "mvs_xxx"
c = sqlite3.connect(os.path.expanduser("~/.minimax/v2/sqlite/runtime-state.sqlite"))
c.row_factory = sqlite3.Row
out = {}
for t in ["local_runtime_sessions","local_runtime_message_rows","local_runtime_thread_goals",
          "local_runtime_session_locks","local_runtime_pi_history_file_migrations",
          "local_runtime_message_row_migrations"]:
    out[t] = [dict(r) for r in c.execute(f"SELECT * FROM {t} WHERE session_id=?", (SID,))]
json.dump(out, open("/tmp/host_backup/db_rows.json","w"), ensure_ascii=False, default=str)
print({k: len(v) for k, v in out.items()})
EOF
```

恢复：目录 `cp -a` 回去，表行按 JSON 逐表 `DELETE` + `INSERT`（`local_runtime_message_rows`
的自增 `id` 建议保留原值以免与其它行冲突）。
