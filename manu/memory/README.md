# manu/memory/ · 记忆库（入口说明）

> **一句话**：`manu/memory/memory.md` 是**入口 + 活章**，**历史正文**在 `manu/memory/archive/` 分卷里。
> **找历史**走 `memory.md` §6 导航表（不要线性读归档）；**写方法**去 `manu/skills/`，不要写进 memory。
> 本文件回答四件事：**这库有什么 / 新条目写哪 / 怎么找 / 怎么验**。

---

## 1. 体系总览

| 路径 | 是什么 | 写什么 |
|:--|:--|:--|
| `manu/memory/memory.md` | **入口 + 活章**（2,404 行） | 索引头：§0 铁律速览 / §1 状态快照 / §2 GT / §3 擂主 / §4 环境 / §5 最新结论 / **§6 导航表** / §7 专题档 / §8 SOP / §10 实现总结；活章：当前 §299~§309（11 节） |
| `manu/memory/archive/` | **历史正文分卷**（20 卷） | 既有卷 `00`~`07`（§零 ~ §五十八 + `memory_compact` 全量，2026-10-14 ~ 10-23 冻结）；新增卷 `08`~`19`（§11~§298，按 § 区间 + 日期切）。**只增不改** |
| `manu/memory/archive/_roll_manifest.json` | **零丢失基准** | 逐卷 md5 + 每节章节索引 + 归档拼接基准 + **切分前源锚点**（`origin`）与坐标映射（`live_sections` / `head_map`）；由 `roll_memory.py` 生成 |
| `manu/memory/rules.md` | **铁律正文** | 不可协商约束全文，动手前必读 |
| `manu/memory/memory_compact.md` | **5 分钟上手卡** | 新 Session 5 分钟建立上下文 |
| `manu/memory/.slim_memory.md` | **瘦活文档**（411 行） | 极简上下文；`validate_nav_table.py --slim` 校验它 |
| `manu/memory/*.md` + 若干子目录 | **专题档**（顶层 70 个，连子目录共 121 个 `.md`） | 单一主题深挖：某条线、某次事故、某工具的完整记录 |
| `manu/skills/` | **方法层**（6 个 Skill + `REGISTRY.md`） | 「怎么做」：SOP、判别标准、搜索规程、交接规程 |

**新增 12 卷（`08`~`19`）的卷首标记**：每卷 5 行 HTML 注释，写明 § 区间、日期区间、**切出前的原始全局行号**，例如
`<!-- 主档正文分卷 08 ｜ §11-20 ｜ 20261022 ~ 20261103 ｜ 原始全局行 737-3223 -->`。

### 阅读顺序（新人 / 新会话）

1. `manu/memory/memory_compact.md` —— 5 分钟卡（30 秒摘要 / 三条工作线 / 关键数字）
2. `manu/memory/rules.md` —— 铁律正文
3. `manu/memory/memory.md` —— 当前状态（§1）+ 擂主（§3）+ 最新结论（§5）+ **§6 导航表**
4. 需要历史 → 按 **§6.2** 跳到 `archive/NN.md` 的对应节（**只读那一节**）
5. 需要「怎么做」 → `manu/skills/REGISTRY.md`

---

## 2. 新增条目写哪里

| 你要写的东西 | 写到哪 | 怎么写 |
|:--|:--|:--|
| **新章节**（今天做了什么） | `manu/memory/memory.md` **BODY 区** | 在 `<!-- ===== MEMORY BODY START ===== -->` 与 `<!-- ===== MEMORY BODY END ===== -->` **之间**追加 `## §NNN（日期）· 标题`；**同时在 §6.2 索引加一行**（`§编号｜标题（简）｜memory.md｜起始行`），否则该节在下一次滚动前不在索引里 |
| **专题深挖**（一个主题） | `manu/memory/<主题>.md` | 独立文件，并在 `memory.md` §7 专题档清单登记 |
| **铁律** | `manu/memory/rules.md` | 追加到对应铁律小节；同步在 `memory.md` §0 速览补一行 |
| **SOP / 方法** | `manu/skills/<name>/SKILL.md` | **不写进 memory**；在 `manu/skills/REGISTRY.md` 登记 |
| **汇报产物** | `manu/report/` | memory = 做过什么；report = 怎么讲；skills = 怎么做 |

### 活章超窗 ⇒ 滚动归档（**不要手改归档卷**）

- 活区上限 `LIVE_MAX_LINES = 1500`、单卷目标 `VOLUME_TARGET_LINES = 2500`（常量在 `manu/memory/roll_memory.py`）。
- 超窗就跑 `python3 manu/memory/roll_memory.py`：旧节按 § 边界切进新卷 `archive/NN_主档正文_<起§>-<止§>_<起日期>-<止日期>.md`，§6 导航表随之重建。
- **切分点只落在 `## ` 标题前**，绝不把一节劈成两半；脚本**幂等**，重跑不改变结果。
- 归档卷**只增不改**：`00`~`07` 与 `08`~`19` 一律不许原地改写；要补充就新开一卷。
- §6.1 / §6.2 中**归档部分**由脚本生成（`NAV AUTO-GENERATED` 标记区），**不要手工编辑**。

---

## 3. 怎么找

| 你要找 | 去哪 |
|:--|:--|
| 某一节的正文 | `memory.md` **§6.2 章节索引** —— 每节一行 → `archive/NN.md` + **卷内行号** |
| 某一节属于哪一卷 | `memory.md` **§6.1 分卷索引** —— 12 行（卷号 / §区间 / 日期 / 文件 / 行数 / 章节数） |
| §零 ~ §五十八 等旧卷 + 专题档 | `memory.md` **§6.3**（原表保留） |
| 5 分钟上下文 | `manu/memory/memory_compact.md` |
| 硬规则 | `manu/memory/rules.md` |
| 怎么做 | `manu/skills/`（`REGISTRY.md` 索引） |

```bash
grep -n "§299" manu/memory/memory.md          # 找一节（先看 §6.2 的落点）
grep -rn "关键词" manu/memory/memory_compact.md  # 在 5 分钟卡里翻
```

### 🔴 引用格式（硬规定）

- 一律写「**§编号**」；
- **表格导航行**（与 §6.1/§6.2、各归档卷内表**完全同形**）：file 列 = `` `archive/NN_主档正文_*.md` ``（**相对 `manu/memory/`**），
  「起始行」列 = **卷内行号**（该节标题行在卷内的行号），§ 号写在**标题列**里（`主档 §24 …`）；
- **散文引用**：`` `archive/NN_主档正文_*.md:卷内行号`（§编号）`` 或 `` `manu/memory/memory.md:<行号>`（§编号）``；
- **不要写 `memory.md` 的**裸**全局行号** —— 新增章节会让它整体平移，写了必然失效；
- 手上只有**切分前的旧行号**时，用 `python3 manu/memory/resolve_ref.py <旧行号>` 换算成新位置；
- 改完导航行跑 `python3 manu/memory/validate_nav_table.py --nav-file <该文件>` —— 每一行都要能**解析到真实的章节标题行**。

---

## 4. 工具

| 命令 | 作用 | 备注 |
|:--|:--|:--|
| `python3 manu/memory/check_memory_roll.py` | **零丢失校验**（退出码 0 = 全绿） | **六道闸门**：① 逐卷 md5 ② 归档区拼接（+ 2b 每卷正文首行必须是 `## ` 标题）③ `memory.md` 索引头 / 活区一致 ④ §6.2 索引全覆盖 ⑤ **导航行归一化后的正文对照冻结锚点**（证明除导航行外一个字没丢）⑥ **索引头锚点**（NAV 自动块之外的 prefix / suffix 逐字节比对，防「索引头少拼一行」） |
| `python3 manu/memory/roll_memory.py --dry` | **预览**一次滚动（只打印计划，不写盘） | 执行前先看计划 |
| `python3 manu/memory/roll_memory.py` | **执行**滚动（切卷 + 重建导航表） | 幂等；重跑不改变结果 |
| `python3 manu/memory/roll_memory.py --refresh-head-anchor` | **重锚索引头**（你**有意**改过索引头正文后用） | 默认不重算锚点；改过索引头又没带这个开关 ⇒ 闸门 6 变红（`--dry` 会提前告警） |
| `python3 manu/memory/roll_memory.py --check` | 只跑零丢失校验（等价第一条） | |
| `python3 manu/memory/resolve_ref.py 27888` | **旧行号换算**：切分前的 `memory.md` 全局行号 → 切分后的（文件, 行号, §编号） | 手上只有旧引用时用它换算；支持区间 `27890-27891` 与 `--json` |
| `python3 manu/memory/validate_nav_table.py` | **导航表自检**：归档完整性 + § 导航逐条解析 | `--slim` 校验 `.slim_memory.md`；`--nav-file X` 指定被校验文件 |

执行机：**本地**（纯文本 + hash），不碰 GPU。

### 拆卷前的单体底稿（**在仓库外**）

- 拆卷前那份 29,344 行的单体 `memory.md` 已按铁律二十五（中间产物出仓库目录）保存在
  **`workspace/cache/ultralytics/manu/memory/memory_pre_split_20261124.md`**
  （md5 `d6e6916ae65ac22e8616b5f010786b07`）。
- ⚠️ 它**不等于** `git show HEAD:manu/memory/memory.md`（那是 **29,117 行 / md5 `bad6f8e78…`** 的**更早**版本）——
  拆卷用的是**工作区**当时的状态（含 227 行未提交改动）。
- 用途：**万一 `archive/_roll_manifest.json` 丢了**，用它重跑一次即可重建 `origin` 锚点；
  日常不需要碰它。**别把它拷回仓库**（会变成 2.6 MB 的大文件）。

---

## 5. 硬约束

- `archive/` **只增不改**；分卷 md5 记在 `_roll_manifest.json` 里，改卷即被抓。
- **不擅自启动重活**：训练 / 推理 / 缓存构建 / 评测必须用户当轮明确授权。
- **本地优先**：一律改本地 checkout，**严禁直接改服务器代码**；同步判据只能靠 md5 双侧比对。
- **不碰冻结基线**：`manu/data/build_full_median_dataset.py` 严禁修改。
- **`git add` / `commit` / `push` 不归 Agent 管**：只改工作区、不提交；交付 = 改动清单 + 验证命令 + 明说「未提交」。
- 结论必须带**脚本相对路径 + 可复制完整命令 + 执行机 + 产物路径**（铁律九）。
- **单一真值**：事实进 memory，方法进 skills —— 不要在 skill 里写实验历史，也不要在 memory 里写操作步骤。

> ⚠️ 仓库根的 `memory/` 是**另一份历史快照**（2026-11-23 之前），**不是活库**；活库在 `manu/memory/`。详见 `memory/README.md`。
