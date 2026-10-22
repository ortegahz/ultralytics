# Skill Registry · 领域迁移工程方法论（4 个可插拔专业工具包）

> **定位**：本目录是**"怎么做"（SOP）**的唯一真值；`manu/memory/` 是**"做过什么 / 当前是什么"**（事实）。
> 二者严格分工：事实进 memory，方法进 skill。**不要在 skill 里写实验历史，也不要在 memory 里写操作步骤。**
> 单一真值：每个 Skill 的正文都在 `manu/skills/<name>/SKILL.md`，本文件只做**索引与触发入口**，不复制正文。

---

## 1. 四个 Skill 速查

| # | Skill | 一句话能力 | 典型触发语 |
|:--|:--|:--|:--|
| 1 | **`domain-data-curator`**<br>(领域数据规范化) | 新场景数据接入的第一道卡口：人工定界 → 特征基底固化 → 数据划分防护网 | "来了新场景青城山/沙漠/海面，准备切分数据"、"这批新素材怎么标"、"GT 分母到底是多少"、"GMC 通道怎么对齐" |
| 2 | **`mixed-ft-orchestrator`**<br>(混合微调流水线) | 路线 B 全帧对等混训的启动与调度标准：冻结策略、采样器口径、退火周期绑定 | "起一次混训"、"配比怎么定"、"1:4 到底怎么实现"、"正式训练怎么跑"、"换新场景先动哪个参数" |
| 3 | **`three-track-gatekeeper`**<br>(三轨门禁与裁决) | 三轨评测 + 一级双通道红线 → 二级帕累托 → 三级同基准 U 的**串行**漏斗判决 | "给模型过门禁"、"这个 checkpoint 能不能上线"、"候选和 RC1 比谁好"、"三级漏斗判一下"、"是不是复现了" |
| 4 | **`hparam-search-operator`**<br>(超参搜索操作规程) | Optuna 多卡并发搜索的启动、守护、断点续跑、异常熔断与结果收割 | "起一轮超参搜索"、"搜索怎么停了"、"断点续跑"、"trial 报错了怎么办"、"能不能加个参数维度" |

---

## 2. 路由指南（**典型任务 → 该调哪个**）

| 任务形态 | 调用链 |
|:--|:--|
| "新场景来了，走一遍完整流程" | `domain-data-curator` → `mixed-ft-orchestrator` → `three-track-gatekeeper` → （需要时）`hparam-search-operator` |
| "我已经训完了，这个模型能不能上线" | `three-track-gatekeeper` |
| "帮我跑一轮配比搜索" | `hparam-search-operator`（先读 `mixed-ft-orchestrator` 确认搜索空间口径） |
| "为什么这个 trial 分数这么高/这么低" | `hparam-search-operator`（收割与口径） + `three-track-gatekeeper`（判决） |
| "这个数字是哪来的 / 之前做过什么" | **不调 skill** → 读 `manu/memory/memory.md` + `manu/memory/archive/` |

⚠️ **边界**：门禁/搜索 skill **不负责**数据标注，训练 skill **不负责**上线判决。
一条完整链路里，**门禁永远是最后一道**，且**门禁不过就不许进下一步**。

---

## 3. 四个 Skill 的共同铁律（**每个 Skill 正文里都重复了细则**）

| 铁律 | 在 Skill 里的体现 |
|:--|:--|
| **本地优先** | 代码只改本地 `/media/manu/1TB-Volume/workspace/ultralytics/`；同步判据**只能靠 md5 三侧比对**（本地 / SSHFS 镜像 / 服务器） |
| **禁止自启动未授权实验** | 搜索启动、扩搜索空间、改协议、跑重评估，都必须先有用户明确授权 |
| **评估不得与训练同卡** | 单卡同时只允许"训练"或"评估"二者之一（OOM 事故归纳） |
| **绝不加权求和** | 三轨判据是**串行**漏斗（一票否决 → 帕累托 → 同基准 U），禁止把三轨揉成一个"综合分" |
| **否定性结论必须量中间量** | 统计量必须抗离群；不给单点当结论 |
| **结论必带溯源** | 脚本相对路径 + 可复制完整命令 + 执行机 + 产物路径 |

---

## 4. 命名约束（**踩过，别改回去**）

Mavis 运行时**强制** skill 名为 **kebab-case**（`/^[a-z][a-z0-9-]*$/`）**且必须与目录同名**，
`skill-creator` 的 `lint-skill.js` 会对下划线命名直接判 **FAIL**。

| 原始设想名（下划线，**不可用**） | 实际落地名（kebab-case，**唯一真值**） |
|:--|:--|
| `domain_data_curator` | **`domain-data-curator`** |
| `mixed_ft_orchestrator` | **`mixed-ft-orchestrator`** |
| `three_track_gatekeeper` | **`three-track-gatekeeper`** |
| `hparam_search_operator` | **`hparam-search-operator`** |

---

## 5. 安装与自检

```bash
# 【本地】把 4 个 Skill 装进运行时用户 skill 目录（复制 + md5 校验）
bash manu/skills/install_to_runtime.sh

# 【本地】lint 全绿自检
for d in manu/skills/*/; do
  node /home/manu/.minimax/.builtin-skills/skill-creator/scripts/lint-skill.js "$d" | tail -1
done

# 【本地】记忆体系自检（归档完整性 + 55 条 § 导航逐条解析）
python3 manu/memory/validate_nav_table.py
```

安装位置与两条实测约束（2026-10-22 查 mcode 0.6.12 运行时代码得出）：

| 事项 | 结论 |
|:--|:--|
| **唯一生效的用户级根** | `~/.minimax/skills/<name>/SKILL.md`（运行时代码：`{id:"user-global", kind:"global", rootPath: <dataDir>/skills"}`）—— **默认启用**，无需改 config |
| **必须用真实目录，不能用符号链接** | 运行时用 `readdir(dir,{withFileTypes:true})` + `entry.isDirectory()` 枚举；**对 symlink 该判断返回 false** ⇒ 链接装的 skill 会被静默跳过。实测：symlink 版 `skill()` 报 "Local skill not found" |
| **`<repo>/.minimax/skills` 默认不生效** | 属 `external` 源，需 `skills.external.enabled=true`（config.yaml 默认关闭）⇒ 未启用，勿依赖 |
| **生效时机** | 运行时 skill 目录在 **Session 启动时**建目录 ⇒ **新装/新改的 skill 需新开 Session 才出现在 `available_skills`**；当前 Session 内 `skill()` 仍报 not found（这是预期行为，不是安装失败） |

---

## 6. 维护规则

1. **Skill 只写方法，不写历史**：结论、指标、擂主一律回 `manu/memory/`。
2. **每个数字都要可溯源**：写进 SKILL.md 前先回代码/memory 核对，并在正文里标出**文件相对路径 + 行号**。
3. **改完必跑 lint**，PASS 才算交付；再跑 `python3 manu/memory/validate_nav_table.py` 确认记忆体系没被带坏。
4. **铁律正文只在 `manu/memory/rules.md`**，Skill 里只引用不复制（避免两处维护）。
5. **归档只增不改**；`manu/memory/archive/` 的 5 个分卷拼起来必须等于瘦身前原 memory.md（`validate_nav_table.py` 会验）。
6. **不要在本目录建 README/CHANGELOG/scripts/ 之外的脚手架**：lint 会拦 forbidden files；确实需要确定性脚本时才建 `scripts/`。
