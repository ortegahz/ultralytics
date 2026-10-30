# manu/report/ —— 汇报产物归档区

> **定位**：本目录存**「怎么讲」**—— 阶段汇报的结构、论证链、逐页结论与成品文件。
> 与项目其它目录的分工（对齐 `manu/AGENTS.md` §1 事实层 vs 方法层硬约束）：

| 目录 | 装什么 | 变更频率 |
|:--|:--|:--|
| `manu/memory/` | **做过什么**（事实、指标、证伪记录、遗留项） | 每个实验完成即入档 |
| `manu/skills/` | **怎么做**（SOP、判据、操作规程） | 规程变更时 |
| **`manu/report/`** | **怎么讲**（本阶段汇报的叙事、逐页素材、成品、溯源） | **每次对外汇报建/更一版** |

> ⚠️ **唯一真值纪律**：汇报里的每个数字都必须指回 `manu/memory/`。
> 若 `report/` 与 `memory/` 冲突，**以 `memory/` 为准**并回头修 `report/`（铁律〇·本地优先、铁律四·事实实时入档）。

---

## 一、本次归档：工程化阶段汇报（2026-10-27 定稿）

| 项 | 值 |
|:--|:--|
| **主题** | 复杂背景下红外弱小无人机检测 · 工程化阶段汇报 |
| **主线** | 从"能跑"到"可交付"——把一条 PC 上的算法图，变成 RK3588 上可并行、可量化、可验收的实时管线 |
| **结构** | **开篇立两个靶子**（难题一 GMC / 难题二 NPU）→ **战线 A 打 GMC**（A1 OpenCL / A2 step10 / A3 缓存）→ **战线 B 打 NPU**（B1~B4）→ **公共底座**（并行结构 + 库交付）→ 收口遗留与下一步 |
| **页数** | **21 页**（16:9，1280×720，含结尾「谢谢」页；v4 新增「战果 A4 · 多缓存深度帕累托」整页） |
| **生成器** | **ppt-master**（`/media/manu/1TB-Volume/workspace/ppt-master`）· Quick Generate 流程 · mode=`pyramid` + style=`data-journalism` · 产出**原生 DrawingML 形状**（非整页位图），文字可直接在 PowerPoint 里编辑 |
| **内容源** | **ppt-master 工程 `svg_output/*.svg`**（本仓 `assets/svg_source/` 为其归档副本，**唯一真值**） |
| **状态** | ✅ **21/21 页完全通过**布局质量门（0 warning / 0 error / 0 blocking） |

### 核心叙事（讲的人务必先记住这三句）

1. **管线在板卡上跑起来之后，被两个问题卡住了** —— 一个是 GMC（吃掉了端到端时延的 **78.5%** 且时延账算不准），一个是 NPU（让"编译通过"≠"跑在 NPU 上"、让 FLOPs 外推失效）。
2. **后面 9 个工作汇报点全部围绕这两个问题展开** —— 战线 A 打在 GMC 上，战线 B 打在 NPU 上，两项公共底座两条战线共用。
   - **战线 A 的三招按 GMC 的三段成本一一对应**：`warp+median 77.4%` → **A1 OpenCL**；`拟合 21.6%` → **A2 step10**；`fit 内部 65.8%` → **A3 缓存**。
     💡 A1 与 A2 **互补而非重复** —— step10 明确"warp 一次没少"，它砍不动的 77.4% 正是 A1 搬走的那块。
3. **收口时主动摆出未解项** —— 评审会最服的不是"全做完了"，而是"你知道还差什么、为什么差、下一步怎么补"。

---

## 二、目录结构

```
manu/report/
├── README.md                        ← 本文件：索引 + 复现方式
├── 00_汇报正文_工程化阶段.md          ← ★ 汇报正文（评审 / 存档 / 讲稿底稿，v3）
├── 01_幻灯片大纲.md                  ← 19 页逐页素材（自动导出，勿手改）
├── 02_数据溯源.md                    ← ★ 每个数字 → memory 出处 + 修订记录 + 生成链路
├── outline/                         ← 幻灯片大纲的结构化副本（JSON）
└── assets/
    ├── 红外弱小无人机检测_工程化阶段汇报_21页_原生可编辑.pptx  ← ★ 当前成品（ppt-master，原生矢量形状）
    ├── 红外弱小无人机检测_工程化阶段汇报_19页.pptx          ← 上一版（PPT-AGENT，整页位图）
    ├── 红外弱小无人机检测_工程化阶段汇报_19页_可编辑版.pptx   ← 上一版（PPT-AGENT，可编辑版）
    ├── _旧版/                        ← 更早的 20 页版（OpenCL 还在公共底座），勿直接使用
    ├── svg_source/                   ← ★ 21 个 SVG 源页（改字改这里，重跑即出新版）
    ├── html/                         ← 上一版（PPT-AGENT）的 HTML 源页
    └── outline.json / contents.json  ← 上一版内容源的运行期副本
```

> **✅ 当前状态**：**成品已用 ppt-master 生成（21 页，零布局问题，原生 DrawingML 形状）** ——
> `assets/红外弱小无人机检测_工程化阶段汇报_21页_原生可编辑.pptx`（96 KB / **788 个原生形状 / 544 段可编辑文字 / 0 张位图**）
> + `assets/svg_source/`（**21 个 SVG 源页**）+ `slides/`（**21 张 PNG 预览**）。
> 页序见 `00_汇报正文_工程化阶段.md` **§5 幻灯片分页映射**。
>
> **v5 视觉重构（2026-10-29）**：针对"配图简陋、文字过多"的反馈做了整版重做 ——
> ① **13 页新增真实图表 / 结构示意图**（条形图、帕累托四联图、前后对照、流水线时间线、RGA 画布拼图、环节链路图）；
> ② **文字大幅压缩**，每页收敛为"**一个视觉主锚点 + 一句结论 + 来源脚注**"；
> ③ 全部 21 页重跑布局质量门 → **21/21 完全通过（0 warning / 0 error）**。
>
> **两版差别**：19 页版是 PPT-AGENT 生成的**整页位图**（改字要回 HTML 改）；
> 21 页版是 ppt-master 生成的**原生形状**（可直接在 PPT 里编辑，源页在 `assets/svg_source/`）。

### 导航

| 我要…… | 去哪儿 |
|:--|:--|
| 通读这次讲了什么 | `00_汇报正文_工程化阶段.md` |
| 查某一页讲了什么 | `01_幻灯片大纲.md`（按 P01…P18 定位） |
| 查某个数字从哪来 / 怎么复现 | `02_数据溯源.md` |
| 拿成品 PPT | `assets/` |
| 快速过一眼版面 | `slides/` |
| **改文案** | 改 `assets/svg_source/*.svg`（唯一真值）→ 重跑质量门 → 重导出 PPTX；**不要手改 `01_`** |

---

## 三、怎么复现

**执行机**：本地工作站（生成）；板卡 `root@192.168.0.64`（数据来源）

```bash
# 1) 全量生成（当前成品：ppt-master / Quick Generate）
cd /media/manu/1TB-Volume/workspace/ppt-master
P=projects/ir_uav_engineering_20261029
PY=/home/manu/anaconda3/envs/uav/bin/python

# 2) 布局校验（写 validation/svg_quality_report.json，导出端要读它做指纹核对）
$PY skills/ppt-master/scripts/svg_quality_checker.py $P \
    --quick-generate --canonical-authoring --stage final --json

# 3) 导出原生 PPTX
$PY skills/ppt-master/scripts/svg_to_pptx.py $P --quick-generate --no-notes
#    → $P/exports/<name>_<timestamp>.pptx

# 4) 归档：拷回本目录
cp $P/exports/*.pptx  manu/report/assets/红外弱小无人机检测_工程化阶段汇报_21页_原生可编辑.pptx
cp $P/svg_output/*.svg manu/report/assets/svg_source/

# 5) 记忆体系自检（与 PPT 无关，但每次改 memory 都要跑）
python3 manu/memory/validate_nav_table.py    # 期望 65/65
```

> ⚠️ **改文案/改图的五个坑（都踩过）**：
> ① 多行文本必须用 `<text>` 的**直接子** `<tspan x=… dy=…>`，**不能嵌套**（嵌套 → checker ERROR）；
> ② `<tspan>` 的开闭标签数必须配平（写错 → XML 不合法 → 整页导出失败）；
> ③ **行内强调 tspan（无 `x`/`dy`）≠ 换行 tspan（有 `x`+`dy`）** —— 批量脚本把二者混同会��文字挤成一行；
> ④ 同一 `x`/`y` 的两条 `<text>` 会被 checker 判为"段落未合并"（advisory，不阻塞）；
> ⑤ **条形图的数值标签必须放在条外**，压在条上会被同色盖住（肉眼几乎看不见）。
> 改完先跑 `svg_quality_checker.py` 再导出；**改了 SVG 必须重跑质量门**，否则导出端报 `stale` 拒绝导出。

**产物路径**
- 生成中间产物：`ppt-master/projects/ir_uav_engineering_20261029/`
  （`svg_output/` **21 个源页**、`validation/` 校验报告、`exports/` PPTX、`backup/` 备份）
- 归档副本：`manu/report/assets/` 与 `manu/report/slides/`

> **📌 操作纪律**：**内容优先，PPT 按需生成** —— 除非明确要求，**不要**主动跑生成命令
> （每轮 10~20 分钟、消耗 LLM 额度）。改文案只改内容源 + 重新导出 `01_幻灯片大纲.md`；
> 需要新成品时再跑生成命令（见「三、怎么复现」）。

---

## 四、工具链与外部依赖

| 组件 | 位置 / 版本 | 说明 |
|:--|:--|:--|
| **PPT 生成器（当前成品）** | `/media/manu/1TB-Volume/workspace/ppt-master` | 手写 SVG 源页 → 质量门 → **原生 DrawingML** PPTX（**无位图**） |
| **内容源（唯一真值）** | `ppt-master/projects/ir_uav_engineering_20261029/svg_output/*.svg` | 21 个源页，本仓 `assets/svg_source/` 为归档副本 |
| 上一版生成器（留档） | `/media/manu/1TB-Volume/workspace/PPT-AGENT` | LLM 逐页生成 HTML → 截图 → PPTX（整页位图） |
| 并行驱动 | `PPT-AGENT/run_fixed_deck_parallel.py` | Phase A 并发 LLM / Phase B 串行 playwright / Phase C 导出 |
| 串行驱动（对照） | `PPT-AGENT/run_fixed_deck.py` | |
| LLM 端点 | MiniMax OpenAI 兼容（`api.minimaxi.com/v1`） | key 存 `PPT-AGENT/.env`（`chmod 600`，`.gitignore` 已忽略） |
| 渲染 | Playwright + Chromium | `sync_playwright` **非线程安全** ⇒ 校验阶段必须串行 |
| 导出 | python-pptx + 工具自带 DOM exporter | 图片版 = 整页位图；可编辑版 = 形状可拖 |

**对 PPT-AGENT 工程的改动（已留档，见 `02_数据溯源.md` §七）**：
`ai_client.py` 增加了 `_strip_think()`（剥离推理模型的 `<think>` 块，属通用兼容性修复）；其余为新增文件。

---

## 五、工具链小抄

| 需求 | 命令 / 位置 |
|:--|:--|
| **改汇报文案 / 改图** | 改 `ppt-master/projects/.../svg_output/*.svg` → 跑质量门 → 重导出 PPTX |
| 重跑布局质量门 | `svg_quality_checker.py $P --quick-generate --canonical-authoring --stage final --json` |
| 导出原生 PPTX | `svg_to_pptx.py $P --quick-generate --no-notes` |
| 渲染 PNG 预览 | Playwright + Chromium 直渲 `svg_output/*.svg`（1280×720） |
| **改 PPT-AGENT 旧链路文案** | 改 `PPT-AGENT/deck_content.py` → 重新导出 `manu/report/01_幻灯片大纲.md`（**不需要跑 PPT**） |
| 归档产物 | `manu/report/assets/`、`manu/report/slides/` |
| 查数据出处 | `manu/report/02_数据溯源.md` → `manu/memory/` |

---

## 六、写作纪律（本次汇报沉淀）

**必须做到**
1. **开篇先立靶子** —— 后面每一项战果都要明确说"打的是哪一条"
2. **主动摆出未解项**，并给出"为什么差 + 下一步怎么补"
3. **每个数字带口径**（同基准 / 同帧集 / 同阈值 / 同匹配规则）
4. **未测的就说未测**（`anchor_step=10` 是否最优、PC 时延 5 倍分歧、闸门对 F1 的影响、角点截尾对 F1 的影响）

**必须避免**
1. 把 0.9209 说成单帧模型指标
2. 把 Oracle 上界写成可交付指标
3. 把"子集 F1"当选型依据
4. 把"端到端等价"说成"逐位等价"（GMC step10 臂与 SOTA 特征**不等价**，已实测）
5. 把"40 ms"说成全局性质（**只在选定 5 条基准序列成立**）
6. 引用**已作废**的数字（F1 0.824029 / fps 13.905 / fps 13.885）

**边界（本汇报刻意不含）**：领域迁移的**微调实验结果**（候选权重、搜索结论、配比甜点区、效用分、业务域提升数字）一概不出现；方向 B 只到 **SOP 方法论层**。
