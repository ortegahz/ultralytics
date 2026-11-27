# HOST 独立验收记录 · 2026-11-27（服务器那一轮收官）

> 执行机：host 工作站（`/run/media/manu/3061aec8-.../workspace/ultralytics`）
> 对象：服务器 `huangzhe@192.168.99.40:32222` 上 2026-11-26 17:15~18:48 跑完的 v2 收口那一轮
> 原则：**不采信服务器自报**，全部证据由 host 亲自跑一遍。

---

## 〇、一句话结论

**服务器侧交付物可交付**：四道门禁 host 亲跑全绿；跨机 18/18 文件字节级一致；host 自带独立审计器
**PASS 37 / FAIL 0 / WARN 1**（该 WARN 已人工判读为通过）；成稿章节已并入 host 权威记忆库为 **§319**。

---

## 一、host 亲跑的四道门禁（2026-11-27 上午，服务器上实跑）

| 门禁 | 命令（服务器） | 实测 |
|:--|:--|:--|
| 机械门禁 | `python -m pytest iqa_v2/test_iqa_v2.py -q` | **44 passed** in 6.36s |
| 独立复算 | `python iqa_v2/verify_recalc_v2.py --samples 120` | ROI 最大绝对误差 **3.553e-15**；膝点复算 6 个最大误差 **0.000e+00**；曲线有效性 **162/162** 一致；分箱/逐档跨度相对误差 **0.000e+00** ⇒ **全部通过（容差 1e-06）** |
| 产物清单 | `md5sum -c iqa_v2/output/manifest_v2.md5` | **全过**（25 条） |
| 完成度审计 | `python iqa_v2/audit_final.py . iqa_v2/REPORT_V2_FINAL.md` | 服务器版 **PASS 39 / FAIL 0 / WARN 0**；host 版 **PASS 37 / FAIL 0 / WARN 1**（见 §三） |

## 二、跨机字节级核对（host 回收副本 ↔ 服务器原件）

host 把交付物回收回本机 `iqa_v2/server_out_final/files/`（18 个文件 + `output/`，共 18 MB），
把 md5 清单交给服务器会话（thread `01a21723-d2d3-78b1-8a7b-1101872510ef`，**已 resume 接回**）逐条重算：
**18/18 全部一致、零差异**（报告 / 章节 / 留痕 / 脚本 / 7 张 CSV / 清单），服务器侧已追加留痕到 `PROGRESS_SERVER.md`。

## 三、host 独立审计器：先修自己的 bug，再复核

> 🔴 **host 那版审计器自身有 bug**，查出来了才敢拿来当验收仪器。**记录在案，免得以后再踩。**

| # | 缺陷 | 症状 | 修法 |
|:--|:--|:--|:--|
| 1 | `binned_curve` 里 Y 写成 `success.mean()`（**全量**均值）而非 `success[sel].mean()`（**箱内**均值） | 每箱 Y 相同 ⇒ 曲线被压成水平线 ⇒ 报出 **3 个假 FAIL**（R5g 退化守卫、R5h 独立复算、R5f/R6c 连带的 nan 行误判） | 改为箱内均值；**R5h 随即复现出约束线数值**（0.18917 / 5.82855），证明原缺陷在仪器侧 |
| 2 | 曲线子集未按「曲线集」过滤（`own_good` / `common_good`） | 用全族窗口建曲线 ⇒ 与流水线口径不符 | 新增 `curve_subset()`：族 × 曲线集（按 `segment_class_v2.csv` 的 `class_<arm>`）× 分层 |
| 3 | 守卫只算了分箱跨度，缺「逐档跨度 / 逐档单调 / 指标单调」 | 与流水线守卫语义不一致 | 新增 `level_profile()` / `curve_guard()`，与 `run_eval_v2.py` 的守卫同语义；并**反查产物列**（R5g2） |
| 4 | `level_profile` 的字典键是字符串、档位表是 float | 取不到值 ⇒ 逐档跨度恒 0 | 键统一成 float |
| 5 | `nan` 线位（本轮无有效红线）仍被要求「出现在免责声明」「匹配膝点」 | 2 个假 FAIL | nan 行跳过；R5i 改为「列在但全 nan ⇒ 报告须声明算法无关性不成立」 |

修好后 host 独立审计：**PASS 37 / FAIL 0 / WARN 1**。
唯一 WARN = **R9「正式交付模型替换状态」**（需人工判读）⇒ **判读为通过**：
报告 §12 明确写「**正式交付模型替换待标注回收后执行**」「**本轮保持 XCorr + KCF，未启动任何训练或模型重训**」，
且 host 侧已核查 58 条业务序列仍是 2026-11-09 的模型预标注、人工标注未回收 ⇒ 与交棒书 G7 一致。

## 四、记忆库合并（host 权威副本）

| 项 | 结果 |
|:--|:--|
| 合并对象 | `iqa_v2/server_out_final/files/MEMORY_SECTION.md`（180 行文件，正文 278 行） |
| 章节号 | 落为 **§319**（下一个空闲号；§318 已被上一轮的「会话已转服务器续跑」占用，与记忆库自身计划一致） |
| 正文完整性 | 278 行正文在 `manu/memory/memory.md` 中**原样连续出现**（逐行比对一致；落盘段多出的 2 行是标题与 `MEMORY BODY END` 标记） |
| 三道校验 | `roll_memory.py` 切分完成 → `check_memory_roll.py` **全绿**（六闸门：分卷 276 节、索引命中 276、缺 0）→ `validate_nav_table.py` **354/354 解析成功** |
| 连带效应 | roll 把 §303/§304 切进 `archive/24_主档正文_§303-304_20261123_20261123.md`（各 1 份，**不重复**）；活区 § 标题数 16 → 15（303/304 出活区 + §319 入活区） |
| 回滚 | `manu/memory/memory.md.bak_20261127_085624`（+ `archive/_roll_manifest.json.bak_20261127_085624` + `archive/.volumes_before_20261127_085624`） |

## 五、交付物（host 侧位置）

| 交付物 | 位置 | 大小 |
|:--|:--|--:|
| 详细报告（12 章） | `iqa_v2/server_out_final/files/REPORT_V2_FINAL.md` | 77,729 B |
| 记忆库成稿章节 | `iqa_v2/server_out_final/files/MEMORY_SECTION.md` → 已并入 `manu/memory/memory.md` §319 | 22,288 B |
| 全部产物 CSV + 清单 | `iqa_v2/server_out_final/files/output/`（含 `degrade_detail_v2.csv` 8.1 MB） | 12 CSV |
| 服务器留痕 | `iqa_v2/server_out_final/files/PROGRESS_SERVER.md` | 50,150 B |
| 服务器会话对话全文 | `iqa_v2/server_out_final/conv_digest.md`（133 条 assistant 消息） | 56,961 B |
| 服务器侧原件 | `/tmp/pycharm_project_a7d966e6/iqa_v2/`（`huangzhe@192.168.99.40:32222`） | — |

## 六、本轮**唯一成立**的红线（与任务书基准的差异，报告已逐条写明）

| 指标 | 任务书基准 | 本轮 | 结论 |
|:--|--:|--:|:--|
| `target_local_contrast` | （未给） | **0.1891664** | ✅ **本轮唯一可对外的红线**（xcorr/motion；真实好片对照 29.33% 错误豁免须随文说明） |
| `target_lap_var` | 819.55 | **无有效红线** | ❌ 分箱曲线跨度仅 1.92 pp（逐档 6.41 pp）⇒ 守卫判其曲线承载不了膝点 ⇒ 降级为物理参照，**不填数值** |
| `local_scr` | 2.69 ~ 3.21 | 5.828548 | ❌ 盲测区分间隔仅 **1.60 pp**（真实好片 98.0% 落线下 vs 真实坏片 99.6%）⇒ **区分不了好坏片，不写入对外免责声明** |

⇒ 免责声明的**对外可发布版**只保留 `target_local_contrast < 0.1891664`，并带三条限定
（单臂反解 / 画质—跟踪分界点而非崩溃点 / 29.33% 错误豁免）。任务书 §二.3 倚重的
「**XCorr ↔ KCF 算法无关性**」在净化底座上**本轮无法复现**（只有 XCorr 臂通过曲线守卫），报告 §4.2/§6 已写明。

## 七、留给用户拍板的三件事（服务器已按「保守且诚实」的方向落笔，可改）

1. **是否接受「只留一条红线」**：更硬的做法是**加大退化档位上限**（blur σ→6~8、contrast α→0.05）让曲线真正塌下来，
   而不是调守卫门槛（放宽到 3 pp 会让 KCF 的 blur 族重新入选）——报告 §12 已给出这条建议。
2. **58 条业务标注回收后**按同口径复测并替换探针（`trial_0080`/RC2）⇒ 届时才谈「正式上板红线」。
3. **是否把本轮结果提交到 git**：`manu/memory/` 与 `iqa_v2/` 当前都是工作区未提交状态（`?? iqa_v2/`、记忆库多文件已改），
   按仓库约定未自动 commit。
