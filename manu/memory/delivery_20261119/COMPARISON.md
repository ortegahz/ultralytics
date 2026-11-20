# trial_0080 + v3 · Δn 重搜 —— 交付包对照表

- 交付参数档：`system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json`（**params_md5 `a1157e013ebb496731be805c72c95dd8`**）
- 新 OSD 目录：`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_20261119`（pass 21 / un-pass 3 = 24 个 mp4）
- 对照 OSD 目录（**保留未动**）：`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_gapfit_20261119`
- 基准：v3 = th_base 0.48 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 7 / confirm_gap_tol 0
  （params_md5 `bba1e948f420efe1b5070269b4587a43`）

## 1. 逐帧口径（全量 24 序 / 165,223 帧，交付口径）

| 工作点 | F1 | FAR | Recall | Precision | TP | FP | FN | FAR≤0.03 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3（基准） | 94.7533% | 0.020820 | 92.35% | 97.28% | 123068 | 3440 | 10189 | ✅ |
| v3 + optgapfit（交付） | 94.4976% | 0.027859 | 92.66% | 96.41% | 123480 | 4603 | 9777 | ✅ |
| old_gapfit（旧 gapfit，仅对照） | 94.3755% | 0.035128 | 93.24% | 95.54% | 124251 | 5804 | 9006 | **❌ 破线** |

- 交付组 vs v3：**F1 -0.2557pp**，**FAR +0.007039**（红线 0.03）
- 交付组 vs 旧 gapfit：F1 +0.1221pp，FAR -0.007269 ⇒ **两个指标都更好**

🔴 **两阶段选型（重要）**：搜索器在 **4000 帧截断口径**上最大化跳帧臂，
但**截断口径系统性低报 FAR ≈ +0.005~0.006（绝对值）** ⇒ **红线必须在全量口径上判**。
实测反例：第二次搜索（换一组 cadence 采样）的 objective-best
（confirm_consec 2 / confirm_gap_tol 8 / max_age_phys 10 / r_slope 5.0 / match_dist 16.0）
在截断口径 FAR 0.029013（看着守线），**全量口径 0.0352 ⇒ 破线 ⇒ 被拒**。
详见 `artifacts/final_selection.json`。

## 2. 跳帧口径（4000 帧仿真，物理帧基 = 板卡真实节奏）

🔴 **口径更正（2026-11-19 23:31~2026-11-20 00:08）**：下表**全部重测过**。
原表（0.9521 / 0.9374）出自 `rt_eval_t0080_v3_opt.json`（19:24）——
那是 `hash()` → `zlib.crc32` 采样器修复（21:23）**之前**跑的，
而 CPython `hash(str)` 每进程随机 ⇒ 那批 cadence 采样**跨进程不可复现** ⇒ 不能当规范值。
重测产物：`artifacts/rt_eval_delivered_armA.json`（`--phase eval`，base=v3 / opt=交付参数）
+ `artifacts/shortlist_deterministic.json`（一个进程内重测 14 组候选，1562.8 s）。
**交叉校验逐位一致**：v3 未调 0.084122 ＝ 两份产物的同一格；交付参数 0.951398 ＝ 两份产物的同一格。

| 臂 | s1 | s2 | s3 | s4 | mean F1 |
| --- | --- | --- | --- | --- | --- |
| 逐帧天花板（v3 基座的 per-frame 臂） | - | - | - | - | **0.960186**（FAR 0.017558） |
| v3 原参 · 跳帧·物理帧基（未调） | 0.0000 | 0.0000 | 0.0000 | 0.336489 | **0.084122**（轨迹全判死） |
| **交付参数 · 跳帧·物理帧基** | 0.955383 | 0.953008 | 0.938916 | 0.958285 | **0.951398** |
| v3 原参 · 跳帧·密集帧基 | 0.944339 | 0.912450 | 0.874592 | 0.956861 | 0.922060 |
| 交付参数 · 跳帧·密集帧基 | 0.949450 | 0.931357 | 0.914469 | 0.958082 | 0.938340 |

结论方向不变：**v3 原参跳帧 0.0841 → 交付参数 0.9514（+0.8673）**；
且交付参数在**密集帧基**下也高于 v3 原参（0.9383 vs 0.9221）⇒ 收益不是靠物理帧号取巧。

🔴 **口径澄清（读表前必看）**：搜索器 `rt_fps_jitter_search.py` 的 `prepare()`（L691）在 `--max-frames 4000` 处
把每条序列**截断到前 4000 帧** ⇒ **上表所有臂（含「逐帧天花板」）都跑在 70,451 帧的截断子集上
（全量 165,223 帧的 42.64%）**，**不是**逐帧交付口径。

| 口径 | 覆盖帧 | F1 | FAR |
| --- | ---: | ---: | ---: |
| 截断（arm5 / 天花板） | 70,451 | 0.958640 | 0.022399 |
| **交付口径（全量）** | 165,223 | **0.944976** | **0.027859** |

⇒ 截断口径**同时高报 F1（+1.37pp）、低报 FAR（−0.00546）**。
**交付放行只看全量口径**（FAR 0.027859 ≤ 0.03 ✅）；arm5 只用于搜索期可行性过滤。
另：arm5 内 `far_delivered_agg`（帧加权 0.022399）与 `far_delivered`（样本均值 **0.030842 > 0.03**）**不是一个量**，别混用。

## 3. 候选短名单（全量口径实测）

| 候选 | 逐帧 F1 | 逐帧 FAR | FAR 余量 | 跳帧 F1（重测，确定性采样器） |
| --- | --- | --- | --- | --- |
| **cc2_md10（交付）** | 94.4976% | 0.027859 | 7.1% | **0.951398** |
| cc2_md8 | 94.4842% | 0.026455 | 11.8% | 0.951215 |
| cc2_md12 | 94.5166% | 0.028440 | 5.2% | 0.951582 |
| cc2_md16n | 94.5804% | 0.029348 | **2.2%** | **0.951693**（最高） |
| cc3_md10 | 94.5097% | 0.026116 | 13.0% | 0.943935 |
| cc3_md16 | 94.5939% | 0.028029 | 6.6% | 0.944615 |
| cc4_md10 | 94.5339% | 0.025112 | 16.3% | 0.935804 |
| cc5_md10 | 94.5462% | 0.024440 | 18.5% | 0.927832 |
| cc6_md10 | 94.5556% | 0.023744 | 20.9% | 0.915936 |
| cc7_md10 | 94.5804% | 0.022872 | 23.8% | 0.905213 |
| rerun_selected | 94.3749% | 0.035152 | ❌ 破线 | 0.952186 |
| old_gapfit | 94.3755% | 0.035128 | ❌ 破线 | 0.950655 |

🔴 **选型措辞更正**：全量口径守线的候选里，**跳帧 F1 最高的是 `cc2_md16n`（0.951693）**，
比交付组 `cc2_md10`（0.951398）高 **0.03pp（噪声量级）**；
但它的全量 FAR 离 0.03 红线只剩 **2.2%**，交付组是 **7.1%**（约 3 倍余量）。
⇒ **交付仍是 `cc2_md10`**：用 0.03pp 跳帧 F1 换 3 倍 FAR 余量。
**若交付方要「跳帧 + 逐帧 F1 都再高一点」**，`cc2_md16n` 同样合规（全量口径已实测：
`/mnt/data/siping/runs/trial0080_onnx/eval/shortlist/track_cc2_md16n.json`），换参数档 + 重渲 OSD 即可。

## 4. 交付参数（叠加在 v3 之上的 7 个 Δn 键）

```json
{
  "confirm_consec": 2,
  "confirm_gap_tol": 8,
  "max_age_phys": null,
  "r_slope": 5.0,
  "alpha_decay": 1.0,
  "match_dist": 10.0,
  "min_hits": 3
}
```

## 5. 目录说明

- `system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json` —— 新参数档（**交付物**）
- `osd_manifest.json` —— 新 OSD 目录逐文件清单（24 个 mp4 逐个登记 group / 文件名 / 体积）
  🔴 **体积单位口径**：`size_mb` / `total_mb` 用**十进制 MB（1 MB = 10⁶ 字节）**，`total_bytes` 是**真实字节数**；
  核对体积请用 `total_bytes`（或 `bytes/1e6`），别拿 MiB（2²⁰）去比，否则会差 4.86%。
- `artifacts/` —— 搜索 json/md、三臂复核 json/md、track json、三份对账 json、memory 章节、搜索规格书、板卡同步预设
- 🔴 **三方一致性已校验**：交付参数档 ↔ OSD 的 `search_json_for_render.json` ↔ OSD 的 TSV 全局参数行，**37 键逐键相等、零差异**（三者会分别传到不同人手里，必须是同一个工作点）
- `artifacts/memory_md5_manifest.txt` —— 6 份 memory 的当前 md5（**改档后需重生成**；`HOST_BACKFILL.sh` 靠它复核）
- 新 OSD 目录的 `per_sequence_hyperparams.tsv` —— 🔴 本轮修掉了渲染器**硬编码 5 键**导致 **6 个 Δn 旋钮没进 TSV** 的缺陷；
  现在 `# global params:` 行记录**完整 36 键工作点**（含 `confirm_gap_tol=8` / `r_slope=5.0` / `match_dist=10.0` /
  `alpha_decay=1.0` / `max_age_phys=None` / `min_hits=3`）。**只改了第 1 行，数据行逐字节一致，24 个 mp4 未重渲。**
  旧 gapfit 对照目录按「不改名不覆盖」要求**未动**（仍是旧 5 键格式，属历史留档）。
- `artifacts/rt_eval_delivered_armA.json` / `artifacts/shortlist_deterministic.json` —— 🔴 **确定性采样器重测产物**
  （跳帧三臂 + 短名单的跳帧列；旧的 0.9521/0.9374 出自修复前采样器，已作废）
- `artifacts/remeasure_shortlist_deterministic.py` —— 上面那份重测的脚本（一个进程内测完 14 组候选）
- `artifacts/verify_all.py` —— 端到端核验脚本（当前 **ALL CHECKS PASSED / RC=0**）
- 🔴 `artifacts/repro_preds_md5.txt` / `repro_track_arm.log` / `repro3_track_arm.log` / `repro3_iostat.log` /
  `track_all_v3_optgapfit_repro.json` / `track_all_v3_optgapfit_repro3.json`
  —— **可复现性闭环证据**（2026-11-20 00:19~00:37）：用**交付参数档原样连跑 3 次** track 臂，
  **3 次预测目录 25/25 文件逐位 md5 相同**、**评测 JSON 320 键中均仅 `elapsed_seconds` 不同**
  （80.06 / 205.77 / 117.75 s），`overall` 与 **24 条 per_sequence 全部逐位一致**，
  且 3 次的 `config.params` 都 == 交付参数档 ⇒ **交付数字可复现，不是单次侥幸**。
  耗时差异有**同期留证**（run3 的 `iostat`：`sda` %util 94~100、读 45.7~84.4 MB/s；`nvme0n1` 仅 0~4.2%）
  ⇒ **缓存所在的 `sda` 是 I/O 瓶颈**，墙钟随磁盘竞争浮动，但**输出字节零差异**。

---

## 6. host 侧回填（host 开机后执行；**两条路都留着**）

### 6.0 两条路一句话

| 方向 | 脚本 | 执行位置 | 前置条件 |
| --- | --- | --- | --- |
| **服务器 → 主机（推）** | `HOST_PUSH.sh` | **服务器**（本夜可自跑） | host 已开机 **且** host 的 sshd 在跑 —— 服务器持有 `~/.ssh/id_ed25519_host_zerg`（ssh config 里就是 `Host host-zerg` = `zerg@192.168.0.115`） |
| 主机 → 服务器（拉） | `HOST_BACKFILL.sh` | host 终端 | host 的公钥须被服务器 `authorized_keys` 授权 —— **目前没有** ⇒ 这条路在认证上大概率不通 |
| **服务器 → 主机（一次性，幂等）** | `HOST_PUSH_ONCE.sh` | **服务器**（每次值守/投递时调一次） | 无需人工：host 可达即推送 + 主机侧复核（exit 0），不可达即跳过（exit 3） |

**为什么优先推**：`HOST_PUSH.sh` 只把本线拥有的「§257 十八」块写进 host 的 memory
（缺失→插入 / 旧版→原位替换 / 已一致→no-op），**块外字节逐字节保留**
⇒ 并发线的 §258/§259/§260 以及 host 独有的内容一律不动，**不存在整文件覆盖掉对方章节的风险**。

🔴 **补齐（2026-11-20 00:08~00:15）**：两条回填路原先**只搬 §257 十八 正文**，
**交付包本体（新参数档 / 本文件 / `osd_manifest.json` / `artifacts/`）没人搬** ⇒ host 收齐 memory 也拿不到这轮的参数档。
现已给两路都加「交付包同步」段（只写**本线自己的** `manu/memory/delivery_20261119/`，**不碰并发线的任何章节**）。
**双路端到端复测通过**（用服务器自身当模拟主机 `ssh 127.0.0.1:32222` + `/tmp/hostsim{,2,3}`）：
推送路 6/6 插入 + 正文 md5 一致 + §258/§259/§260 3/3 + host 独有行 6/6 存活 + 交付包落地 + 再推 6/6 `NO-OP`（幂等）；
拉取路 6/6 合并 + sec_md5 全对 + `concurrent_kept=3/3` + 交付包拉全 + rc=0。
🔴 **第 76 轮（2026-11-20 02:31~02:40）**：推送路原用**手写白名单**（21 个文件）搬交付包，
白名单会随证据增减漂移 ⇒ host 上会缺「规范搜索产物」「OSD 媒体核验脚本」等**本文件已登记的交付物**；
现改为**按 `artifacts/` 目录实际内容整搬**（只排 `.bak*`，保持 scp 依赖，不引入 host 端 rsync 依赖），脚本净减 13 行。
复测（模拟 host `ssh 127.0.0.1:32222`）：推送路 6/6 插入 + §258/§259/§260 3/3 + host 独有章节 6/6 存活 +
**交付包 35/35 逐字节一致** + 二次推送 6/6 `NO-OP`；拉取路同样 **35/35 一致** ⇒ **两条路交付集字节级相同**。
留证：`search/host_push_sim6.log`、`search/host_push_sim6_run2.log`、`search/host_backfill_sim6.log`。
🔴 **第 77 轮（2026-11-20 02:37~02:42）补测「零人工路径」**：`HOST_PUSH_ONCE.sh`（每次值守调一次的那条路）
此前只被测过「host 关机」分支，**从未对着能连的 sshd 跑通过** ⇒ 它坏了每晚值守会静默空转而闸门照样绿。
复测（模拟 host `ssh 127.0.0.1:32222` + `/tmp/hostsim7`）：**可达⇒rc=0**（host 侧 6/6 合并 + 正文 md5 全对 +
并发章节 §258/§259/§260 **3/3** + host 独有章节 **6/6 存活** + 交付包 **35/35 一致**）、**再跑⇒6/6 `NO-OP`**、**不可达⇒rc=3**。
闸门同时加了 1 项「零人工路径真的调 `HOST_PUSH.sh`」（反向对照：摘掉调用 ⇒ rc=1 且恰好 1 FAIL），闸门 146 → **147 项**。
留证：`search/host_push_once_sim7.log`。
⚠️ 另测一个**故意只有 §258** 的模拟 host ⇒ 拉取路闸门如实报 `concurrent_kept=1/3` 且 **rc=1**（闸门会真拦）。
**host 仍关机**（`ping 192.168.0.115` 无响应，00:12 实测）⇒ 本轮**无法真正回填**，host 一上电调 `HOST_PUSH_ONCE.sh` 即闭环。

### 6.1 首选：服务器侧推送（一条命令）

```bash
# 服务器侧（host 已开机且 sshd 可达时）
bash manu/memory/delivery_20261119/HOST_PUSH.sh --check   # 只探测：连通性 + 主机仓库根目录，不写任何文件
bash manu/memory/delivery_20261119/HOST_PUSH.sh           # 正式推送 + 主机侧复核
```

推送内容 = `merge_memory_section.py` + 两份 §257 十八 正文（完整版给主档、精简版给精简档），
落到 host 仓库的 6 份文件上；推送后脚本在**主机侧**复核并打印
`sec_md5`（与 `artifacts/memory_md5_manifest.txt` 的正文 md5 一致才算 OK）与 `concurrent_kept=3/3`。

> **已实测（2026-11-19 22:2x）**：用服务器自身当模拟主机（`ssh 127.0.0.1:32222` + `/tmp/hostsim` 模拟 host 仓库，
> 模拟文件里**故意删掉 §257 十八**并在 §258 里塞了 host 独有行）跑完整链路：
> **6/6 插入成功、host 独有行 6/6 存活、块外字节不变、再推一次 6/6 `SKIP` 幂等**。
> 过程中真抓到两个 bug 并已修：`scp` 的端口参数是大写 `-P`（用 `-p` 会被当成"保留时间戳"）、
> 服务器 `/home` 是只读挂载 ⇒ `~/.ssh/known_hosts` 写不进去（改用可写的 `UserKnownHostsFile`）。

**零人工版**：服务器侧在**每次值守/投递时**调一次（幂等，可无脑重复）：

```bash
bash manu/memory/delivery_20261119/HOST_PUSH_ONCE.sh          # 0=成功 / 3=host 仍关机 / 1=失败
tail -f /mnt/data/siping/runs/trial0080_onnx/search/host_push_watch.log
```

🔴 **为什么不用常驻守望**：这台服务器（容器）**没有可用的守护机制** ——
`cron` 未运行且 `/var/spool/cron`、`/etc/cron.d` 不可写、也没有 `crond` 二进制；
`nohup` / `setsid` / `screen` 起的后台进程**都会在当前 shell 会话结束时被回收**（实测两次，PID 归 1 也一样）。
⇒ 常驻守望活不过一次 exec。`HOST_PUSH_WATCH.sh` 仍随包交付（长驻 screen/tmux 的环境下有用），
但在这台服务器上**不指望它常驻**；本包的行为约定是"每次值守调一次 `HOST_PUSH_ONCE.sh`"。

### 6.2 备选：host 侧拉取（**已改成章节级合并，不再整文件覆盖**）

```bash
bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh --check   # 先预检，不动任何文件
bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh           # 正式回填 + md5 复核
```

脚本会**自动探测 host 仓库根目录**（从自身位置向上三级）、**自动探测可用服务器地址**
（候选 `192.168.99.40`，可用 `SERVER_CANDIDATES="ip1 ip2"` 追加），
然后**只取回「合并工具 + 两份正文 + md5 清单」**，在 host 本机做**章节级合并**：

- 🔴 **不再整文件 rsync** —— 那会把 host 上**更新过的**并发线 §258/§259/§260 换成服务器副本（可能更旧），
  静默吞掉对方章节；现在**块外字节逐字节保留**。
- **闸门 = 「§257 十八 正文 md5 与清单一致」+「§258/§259/§260 仍在位」**；
  整档 md5 只作**信息项**（host 侧有独有内容时本就应不同，不是故障）。
- 密钥可用 `SSH_KEY=<私钥路径>` 指定；`KNOWN_HOSTS=<路径>` 指定主机密钥记录文件。

⚠️ 认证前提目前不成立（见 6.3），所以它是备选。

### 6.3 拓扑与认证事实（两条路都要看）

- 服务器**只有 `192.168.99.40/24` 一张网卡**，host 在 `192.168.0.0/24` ⇒ 两边不同网段，**host→服务器的路由未证实**。
- 服务器 sshd：`Port 32222`、`PasswordAuthentication=no`、`PubkeyAuthentication=yes`。
- 服务器 `authorized_keys` 只有 2 把：`zxthz@126.com`（= `id_board`，用户本人）、`board-ptq`（= `id_ed25519_ptq_board`，板卡）
  ⇒ **host 自己的公钥不在里面** ⇒ host→server 拉取在认证上大概率不通；
  而服务器侧**持有 host 的私钥** `~/.ssh/id_ed25519_host_zerg` ⇒ **服务器→主机推送的密钥是现成的**
  （前提是 host 的 sshd 在跑 —— host 关机时无法验证）。
- 服务器 `/home` 是**只读挂载** ⇒ `~/.ssh/known_hosts` 写不进去（ssh 告警但仍可登录）
  ⇒ `HOST_PUSH.sh` 改用可写的 `UserKnownHostsFile`（默认 `/tmp/.manu_known_hosts`）。
- `HOST_PUSH.sh --check` 会先探测连通性与主机仓库根目录（探测不到可用路径时，用 `REMOTE_ROOT=/path/to/ultralytics` 指定）。
- 🔴 **探测诊断已细化**：`HOST_BACKFILL.sh` 原先把「服务器主机密钥未记录」误报成「密钥未授权」
  （BatchMode 下 `Host key verification failed` 与 `Permission denied` 同分支）⇒ 现按 ssh 真实报错分三类：
  **路由不通 / 主机密钥未记录 / 密钥未授权**，并已加 `StrictHostKeyChecking=accept-new`（首次连自动记录）。

### 6.4 兜底：手工补一节

1. **借板卡中转**：板卡的公钥（`board-ptq`）**已在**服务器 `authorized_keys` 里 ⇒ 板卡能 ssh 上服务器；
   板卡又能挂 host 的 NFS（`192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs`）
   ⇒ 板卡上 `scp` 6 份 memory 到 NFS 目录，host 再从 NFS 取回。
2. **从工作站中转**：工作站能 ssh 进 host（`zerg@192.168.0.115`），也可连服务器
   ⇒ 先把 6 份文件 scp 到工作站，再 scp 进 host 的 `ultralytics/` 仓库，最后跑一次
   `grep -v '^#' manu/memory/delivery_20261119/artifacts/memory_md5_manifest.txt | md5sum -c` 复核。
3. **兜底手工插入**：服务器侧已备好两份正文
   `manu/memory/delivery_20261119/artifacts/memory_section_257_18.md`（主档用）与
   `artifacts/memory_section_257_18_compact.md`（精简档用）
   ⇒ 实在传不过去时，在 host 上直接跑交付包里的合并工具即可（它只动这一块）：

   ```bash
   python3 merge_memory_section.py <host 的主档 memory.md> artifacts/memory_section_257_18.md
   python3 merge_memory_section.py <host 的精简档 memory_compact.md> artifacts/memory_section_257_18_compact.md
   ```

   （**插在 §257 十八末尾、§258 之前**，勿整文件覆盖。）

⚠️ **回填前请确认服务器侧写入已收口**（本轮已收口，`verify_all.py` RC=0），否则拉到的是中间态。

## 7. 第 88 轮复验：搜索器的硬约束搬到**交付口径**（2026-11-20 05:26~07:40）

第 3 节的选型结论**不变**（交付仍是 `cc2_md10`）。本轮做的是**把搜索器的一个真缺陷修掉**，并复核它对交付结论的影响。

1. **缺陷**：搜索器的逐帧硬约束原先量在 4000 帧截断口径上（系统性低报 FAR 约 0.0055）⇒ 会出现「搜索口径守线、交付口径破线」。
   已修：新增 `--perframe-caliber`，**默认 delivery**（全量 165,223 帧）；arm1 天花板与 arm5 显式留在截断口径 ⇒ 历史数字逐位复现。
2. **修完重搜，argmax = `cc2_md16n`**（与第 3 节人工短名单的最高点**逐键相同**）：
   历史选 `max_age_phys=10`（全量 FAR 0.035152 **破线**），修复后选 `max_age_phys=null`（**守线**）。
   修复后的搜索器在 sweep2 **实测过交付组那一档**（md10 → 0.951398、md12 → 0.951582、md16 → **0.951693**）⇒ 交付组排第二。
3. **独立复核（权威交付口径）**：`cc2_md16n` 全量 24 序 ⇒ **F1 0.945804 / FAR 0.029348**，**逐位复现** r87 的
   `eval/shortlist/track_cc2_md16n.json`；`--self-check` ⇒ **[GATE PASS]**；对官方 v3 `GT identical 24/24`。
   ⚠️ 搜索器自带全量臂给的是 0.945673 / 0.029556，**别与权威 track 臂混用**。
4. **交付判定不变**：`cc2_md16n` 的 FAR 余量只有 **2.2%**（交付组 7.1%），而**板端 FAR 复核还没做（板卡断电）**
   ⇒ 薄余量风险今夜无法排除 ⇒ 仍然用交付组（`match_dist=10.0`）。
   **切换就绪**：候选参数档 `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_md16_candidate_params.json`
   （首行 `status: CANDIDATE`）、`eval/track_all_v3_optgapfit_md16.json`、候选 OSD
   `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_md16_20261120/`（pass 21 + un-pass 3 = 24）。
   🔴 **候选参数档故意不进本包**（包内只允许一份 `load_params` 能静默接受的参数文件）。

5. **补充核验（07:20~07:40）：un-pass 归因** —— 官方 v3 参考渲染 `osd_sys_sota_trial0080_v3_20261022/` 的
   pass/un-pass 同为 **21 + 3**，且 un-pass 三条与交付组**完全相同**
   （`…000322` / `…002251` / `…015740`）；逐序列 F1（v3 → 交付）**78.821→78.867 / 66.101→60.691 / 14.286→17.120**，
   且这三条在 v3 里本就是 FAR 最高的三条（0.4277 / 0.0997 / 0.0300）⇒ **un-pass 是场景固有难点，与本次参数改动无关**。
   **渲染判据 = F1 ≥ 88%**（故 89.950% 那条算 pass）；交付 / 候选两个 OSD 的 `per_sequence_hyperparams.tsv`
   与各自 `search_json_for_render.json` 的 **f1/far/tp/fp/fn/gt 逐字段一致（0 处不符）**，`group` 列严格等于 88% 判据。
   🔴 本条**只补文档与证据，交付件一个字节都没动**。
