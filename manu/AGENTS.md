# AGENTS.md — manu/（红外弱小无人机检测 · 领域迁移）

> 本文件与上级 `../AGENTS.md`（Ultralytics 上游开发规约）**合并生效**。
> 上级文件管上游代码风格；本文件管**本项目**的不可协商约束。
> 完整铁律正文：`manu/memory/rules.md`（1,433 行，**动手前必读**）。

## 0. 不可协商约束（违反即事故）

| # | 铁律 | 落地动作 |
|:--|:--|:--|
| 〇 | **本地优先** | 一律先改本地 `/media/manu/1TB-Volume/workspace/ultralytics/`，**严禁直接改服务器代码**；同步判据**只能靠 md5 双侧比对**（本地 / SSHFS 镜像 / 服务器） |
| 〇之二 | **基线冻结** | `manu/data/build_full_median_dataset.py` **严禁修改**（只可调用/读取）——它定义了 baseline 本身 |
| 〇之三 | **任务边界** | 只做用户交代的事；**禁止自启动未授权实验/训练/推理/重型评测**。发现缺口只能记录 + 上报，不得自行执行 |
| 一 | **禁擅自启动重活** | 训练 / 推理 / 缓存构建 / 评测必须用户当轮明确授权 |
| 二 | **缓存精简** | pkl 等 cache 控制在 ~100 MB 级，严禁数十 GB 冗余缓存 |
| 五 | **受控 A/B** | 重大非继承性实验严禁越级对比 SOTA，必须同源受控 A/B；**绝不加权求和**三轨判据 |
| 六 | **禁解冻底模** | 组件扩容 / 微调走受控渐进残差规范 |
| 九 | **可复现记录** | 每条结论必带**脚本相对路径 + 可复制完整命令 + 执行机 + 产物路径**；否定性结论**必须量中间量**；统计量抗离群 |
| ★ | **评估不得与训练同卡** | 单卡同时只允许「训练」或「评估」二者之一（OOM 事故归纳）；起 GPU 活前先查卡是否被占 |
| ★ | **板卡 ctx 计数** | 必须 `ps -eo comm`（`ps -eo args` 会把 ssh/bash 包装一起数）⇒ 闸误判「板卡满」并静默跳过模型 |
| 十一 | **overnight 收尾链** | 墙钟截止必须带**硬截止 + 看门狗豁免**；子步骤 rc 独立可查；封存幂等；门禁开始时刻须晚于最后一条训练臂完成（详见 `rules.md` §一·11） |
| 十二 | **跨机状态迁移** | 搬移会话/状态库必须**对着目标机真实库仿真**、对账**收编期字段**、预检并自动补目标端信任与 provider、覆盖前做方向性保护；**agent 侧出网被封时必须先测绘 host 侧触发器**（本例 = app-server 的 `git worktree add` + `post-checkout` 钩子），别直接甩给用户（详见 `rules.md` §一·12） |
| 十三 | **算力归属：重活一律上服务器** | 用户 host 笔记本 **CPU/内存有限**（12 核 / 14 GB，且与用户日常办公同机）⇒ **训练 / 推理 / 评测 / 缓存重建 / 长跑 / 任何 >5 min 的任务** **必须放服务器 `192.168.99.40`** 执行；host 只做**代码编辑、产物读写、md5 与口径抽检、文档编辑**。**禁止**在 host 上并行起 >1 个重活；**禁止**用 `nohup &` 挂常驻重活（沙箱会回收，主机重启全灭——2026-11-03 11:01 重启实测杀死 8 序列评测 + FPV8 重建）；**禁止**与既有重建任务抢核（FPV8 重建曾把评测拖慢 2~3×）。落地动作见 **§4.1** |

## 1. 事实层 vs 方法层（分工硬约束）

| 内容 | 去处 |
|:--|:--|
| 「做过什么 / 当前是什么」（状态、指标、擂主、结论） | `manu/memory/`（`memory.md` 主档 / `memory_compact.md` 5 分钟卡 / `rules.md` 铁律 / `archive/` 只增不改 / 专题档） |
| 「怎么做」（SOP、方法论、判别标准、搜索规程） | `manu/skills/`（4 个 Skill + `REGISTRY.md` 索引） |

**不要在 skill 里写实验历史，也不要在 memory 里写操作步骤。**

## 2. 动手前必读（顺序固定）

1. `manu/memory/memory_compact.md` — 5 分钟建立上下文（30 秒摘要 / 三条工作线 / 关键数字 / 导航表）
2. `manu/memory/rules.md` — 铁律正文（尤其 §〇~§〇之三、§一、§三 计量纪律、§四 流式协议、§五 交付口径）
3. `manu/memory/memory.md` — 当前状态快照 + 擂主 + 最新结论 + §导航表

历史细节走 `manu/memory/archive/`（**按 §编号查导航表，不要线性读归档**）。

## 3. 关键真值（唯一来源，别处没有第二份）

| 项 | 值 |
|:--|:--|
| 业务域 GT（服务器绝对路径） | `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| 🔴 **GT 分母 ≠ 文件数** | 目录 165,224 = **文件数**；**GT 分母 = 133,279 行**（标注框行数）⇒ 用错指标系统性偏低 ~20% |
| 评测 `--labels-root` | 取**服务器侧路径**，**不要**指本机 SSHFS 只读路径 |
| 底模 | `runs/optuna_p0_nas/trial_0474/weights/best.pt`（单帧 F1 0.906361，永久冻结） |
| 当前擂主 | RC2 = `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt` |
| 业务域交付档 | `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`（`th_base 0.48 / sky_ratio 1.0 / th_ground 0.26 / confirm_consec 7`） |
| 交付判据 | F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030 |

## 4. 环境

| 项 | 值 |
|:--|:--|
| 训练/推理服务器 | `ssh -p 32222 huangzhe@192.168.99.40`（项目目录 `/tmp/pycharm_project_10ae9e2e`，conda env `uav`） |
| SSHFS 代码镜像（**只读 ro**，写不进去，**不是推送通道**） | `/home/manu/mnt/pycharm_project_10ae9e2e` |
| RK3588 板卡 | `ssh -p 22 root@192.168.0.211`（**2026-11-10 换板**，旧 `192.168.0.64` 已作废；口令见 `manu/memory/rules.md` §1b，**不得写入对外材料**） |
| 交叉编译器 | `/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu`（前缀 `aarch64-rockchip-linux-gnu-`） |

### 4.1 算力归属 SOP（重活上服务器）

**前提**：沙箱**出网被封**（`socket: Operation not permitted`，ssh/scp/rsync 全部不可用），
且 `/home/manu/mnt/*` 是 **SSHFS 只读（ro）** 镜像 ⇒ **Codex 无法自行在服务器起活**，
必须由用户在**沙箱外**（自己的终端）执行**一条**命令完成「推代码 → 跑 → 拉产物」。

| 事实 | 值 |
|:--|:--|
| 服务器 ssh | **`ssh -p 32222 huangzhe@192.168.99.40`**（**端口 32222**；22 端口是**关闭**的，用 22 会 `Connection refused`） |
| 服务器项目目录 | `/tmp/pycharm_project_10ae9e2e`（conda env `uav`） |
| 本机 SSHFS 镜像 | `/home/manu/mnt/pycharm_project_10ae9e2e`（**只读 ro**，写不进去，别指望它当推送通道） |
| 服务器数据 | `/mnt/data/...`（本机镜像 `/home/manu/mnt/data`，**只读**） |
| 现成脚本 | `manu/tracking/lookahead/server_overnight/push_and_start.sh`（打包 + scp + 拉起夜间作业）、`start_on_server.sh`、`run_overnight_eval.sh` |

**执行法（用户在 host 终端跑一次）**：

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics
bash manu/tracking/lookahead/server_overnight/push_and_start.sh   # 端口已在脚本里设为 32222
```

**Codex 侧纪律**：

1. 重活先问「这能不能在服务器跑」——能就**不要在 host 上跑**；
2. 确实只能 host 跑时（如依赖 host 本地 sshfs 缓存）：**限线程**（`OMP_NUM_THREADS=2~3`）、
   **前台 tty exec**（不用 `nohup &`）、**单任务串行**、跑完立刻出日志；
3. host 上**不再**跑 FPV8 缓存重建这类数小时任务；
4. 服务器侧产物用 **md5 三侧比对**（本地 / 服务器 / 报告）确认落地，不以「命令没报错」当成功。
| ⚠️ 交叉编译纪律 | **严禁 `-I/usr/include`**；`-ffp-contract=off` 与拒绝 fast-math 必须保留 |

## 5. 同步三侧校验（改完必做）

```bash
md5sum /media/manu/1TB-Volume/workspace/ultralytics/<f> \
       /home/manu/mnt/pycharm_project_10ae9e2e/<f>
ssh -p 32222 huangzhe@192.168.99.40 "md5sum /tmp/pycharm_project_10ae9e2e/<f>"
```

三侧一致才算落地。**「已同步」≠「本地已改」**（真实事故：曾直接写 SSHFS 挂载盘，本地 checkout 一行未变）。

## 6. 记忆体系自检

```bash
python3 manu/memory/validate_nav_table.py    # 归档完整性 + §导航逐条解析
```

## 7. 子 Agent 纪律（本项目专用）

派生子 Agent 时**必须**把上面 §0 的铁律原文写进 `context`，否则子 Agent 不知道自己被禁止启动重活、禁止改服务器、禁止碰冻结基线。子 Agent 的结论是**自述**，落盘产物与外部副作用必须由主控自行复核（md5 / 回读 / 远端重查）后才可对外汇报。
