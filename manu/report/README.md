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
| **结构** | **开篇立两个靶子**（难题一 GMC / 难题二 NPU）→ 战线 A 打 GMC → 战线 B 打 NPU → 公共底座 → 收口遗留与下一步 |
| **页数** | 18 页（16:9，1280×720 渲染） |
| **状态** | ✅ 全部页通过布局检查 |

### 核心叙事（讲的人务必先记住这三句）

1. **管线在板卡上跑起来之后，被两个问题卡住了** —— 一个是 GMC（吃掉了端到端时延的 **78.5%** 且时延账算不准），一个是 NPU（让"编译通过"≠"跑在 NPU 上"、让 FLOPs 外推失效）。
2. **后面 9 个工作汇报点全部围绕这两个问题展开** —— 战线 A 打在 GMC 上，战线 B 打在 NPU 上，三项公共底座两条战线共用。
3. **收口时主动摆出未解项** —— 评审会最服的不是"全做完了"，而是"你知道还差什么、为什么差、下一步怎么补"。

---

## 二、目录结构

```
manu/report/
├── README.md                        ← 本文件：索引 + 复现方式
├── 00_汇报正文_工程化阶段.md          ← ★ 汇报正文（评审 / 存档 / 讲稿底稿）
├── 01_幻灯片大纲.md                  ← 18 页逐页素材（自动导出，勿手改）
├── 02_数据溯源.md                    ← ★ 每个数字 → memory 出处 + 生成链路命令
├── slides/                          ← 18 张 PNG 预览
└── assets/                          ← PPTX（图片版 + 可编辑版）+ HTML 源页
```

### 导航

| 我要…… | 去哪儿 |
|:--|:--|
| 通读这次讲了什么 | `00_汇报正文_工程化阶段.md` |
| 查某一页讲了什么 | `01_幻灯片大纲.md`（按 P01…P18 定位） |
| 查某个数字从哪来 / 怎么复现 | `02_数据溯源.md` |
| 拿成品 PPT | `assets/` |
| 快速过一眼版面 | `slides/` |
| **改文案** | 改 `PPT-AGENT/deck_content.py`（唯一真值）→ 重跑生成 → 重新导出 `01_幻灯片大纲.md`；**不要手改 `01_`** |

---

## 三、怎么复现

**执行机**：本地工作站（生成）；板卡 `root@192.168.0.64`（数据来源）

```bash
# 1) 全量生成（18 页，并发 5，约 15~20 分钟）
cd /media/manu/1TB-Volume/workspace/PPT-AGENT
/home/manu/anaconda3/bin/python -u run_fixed_deck_parallel.py --workers 5

# 2) 只重跑指定页（保留其它页）
/home/manu/anaconda3/bin/python -u run_fixed_deck_parallel.py --workers 1 --only 9

# 3) 重新导出幻灯片大纲到本目录
cd /media/manu/1TB-Volume/workspace/PPT-AGENT
/home/anu/anaconda3/bin/python -c "
import sys; sys.path.insert(0,'.')
import deck_content
from html_pipeline.pipeline import _get_pages
print(len(_get_pages(deck_content.OUTLINE)), '页')
"
# 正式导出脚本见下方「五、工具链」

# 4) 校验与归档
python3 manu/memory/validate_nav_table.py    # 记忆体系自检（59/59）
```

**产物路径**
- 生成中间产物：`/home/manu/tmp/ppt-output/红外弱小无人机检测算法工程化阶段汇报/`
  （`html/` 18 个源页、`outline.json`、`contents.json`、`slide-status.json`、两个 PPTX）
- 归档副本：`manu/report/assets/` 与 `manu/report/slides/`

---

## 四、工具链与外部依赖

| 组件 | 位置 / 版本 | 说明 |
|:--|:--|:--|
| PPT 生成器 | `/media/manu/1TB-Volume/workspace/PPT-AGENT` | LLM 逐页生成 HTML → 截图 → PPTX |
| **内容源（唯一真值）** | `PPT-AGENT/deck_content.py` | outline + contents，**逐字可溯源** |
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
| 生成 18 页 PPT | `run_fixed_deck_parallel.py --workers 5` |
| 只重跑某页 | `run_fixed_deck_parallel.py --workers 1 --only N` |
| 导出幻灯片大纲 md | `deck_content.py` → `manu/report/01_幻灯片大纲.md`（**自动生成，勿手改**） |
| 渲染预览 PNG | `html_pipeline.html_builder::render_html_with_validation` |
| 归档产物 | `manu/report/{assets,slides}/` |
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
