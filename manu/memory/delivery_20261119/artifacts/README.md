# artifacts 清单（trial_0080 + v3 · Δn 重搜）

| 文件 | 是什么 | 注意 |
| --- | --- | --- |
| `self_check_optgapfit.log` | 🔴 **任务书要求的 `[GATE PASS]` 证明**（`--self-check --self-check-baseline system_sota_longquan_trial0080_singleframe_gate_baseline.json` 的原始输出） | 第 79 轮补进包：此前它只在服务器 `eval/` 下，文档反复引用却**没随包交付** ⇒ host 侧只能凭空信「GATE PASS」这句话；现已随包，且 tp/fp/fn/gt/recall/precision/f1/far **8 项 expected==actual**，并与随包冻结 baseline 逐项对齐（已固化成 `verify_all.py` §6d 自动校验） |
| `system_sota_longquan_trial0080_singleframe_gate_baseline.json` | 🔴 **冻结单帧闸门 baseline**（上面那行日志里 expected 数字的出处） | 第 79 轮补进包：没有它，日志不可自证（读者无从判断 expected 是不是对着冻结基线算的）；含 cache_root / 模型 md5 / 24 序 165,223 帧指纹与容差 |
| `space_t0080_v3.json` | 🔴 **搜索空间定义**（7 个 Δn 旋钮的候选值集合） | 第 79 轮补进包：规格书 §10.1 断言「空间逐项精确覆盖任务书要求」，但**空间本身没随包** ⇒ 该断言不可核；现已随包，且由 `verify_all.py` §6d 逐旋钮比对任务书要求集合 + 校验保留 v3 可回退点（confirm_consec=7 / confirm_gap_tol=0） |
| `rt_search_t0080_v3_seedfix.json` | **规范搜索产物**（run3，cadence 采样用 `zlib.crc32` 确定性种子） | 只有它可复现 |
| `rt_search_t0080_v3.json` | run1 历史留档 | 🔴 **作废**：`random.Random(hash(str))` ⇒ CPython `hash(str)` 每进程随机 ⇒ cadence 不可复现 |
| `rt_search_t0080_v3_final.json` | run2 历史留档 | 🔴 同样用 `hash()` ⇒ 不可复现 |
| `opt_params_t0080_v3_flat.json` | 交付工作点的扁平参数（第 75 轮随生成器补进包） | ✅ **它就是交付参数**：37 键与 `system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json` 的 `params` 块**逐键相等**（已实测）。🔴 **第 85 轮起，本包内只剩这一份参数文件**：run2/run3 的 objective-best 扁平参数（原 `opt_params_t0080_v3_final_flat.json` / `opt_params_t0080_v3_seedfix_flat.json`）**已从交付包删除** —— 它们装的是**已被全量闸门拒绝**的那组（`match_dist 16.0` / `max_age_phys 10` ⇒ 全量口径 FAR **0.035152 > 0.03 红线**），又是嵌套 `{"params": {...}}` 形态 ⇒ `system_sota_longquan.py` 的 `load_params`（`payload.get("params", payload)`）**静默接受**它，喂进去不报错、直接跑出破线结果；板卡施工单 §3 正要把 `match_dist 16.0→10.0` / `max_age_phys 10→null` 改掉，**误取该文件 = 把被拒参数同步上板**。**证据未丢**：run2/run3 的 objective-best 全量参数在 `rt_search_t0080_v3_final.json` / `rt_search_t0080_v3_seedfix.json` 的 `opt_params`（37 键逐键相等），选型过程在 `final_selection.json` |
| `rt_eval_t0080_v3_opt.json` | 三臂复核（base = **交付参数**） | 臂 1/2b/3 数值相同 **不是 bug**（base 已是交付组）；🔴 **它的跳帧数字（0.9521/0.9374）产于采样器修复前 ⇒ 不可复现，已被下面两份重测产物取代** |
| `rt_search_t0080_v3.md` | run1 搜索报告（**由同名 json 现渲染**） | 第 81 轮补进包：搜索器的 `--out-md` 原是**死参数**（只解析、从不使用）⇒ 交付文档声称交付 md、实际一份都不存在；现已把渲染器做成纯 stdlib 模块并让 `--out-md` 真出报告；🔴 **批次状态：run1 用 `hash()` 采样器 ⇒ cadence 与跳帧数字均**不可复现**，其跳帧值已作废（以 run3 + `shortlist_deterministic.json` 为准） |
| `rt_search_t0080_v3_final.md` | run2 搜索报告（同上，现渲） | 同上；🔴 **批次状态：run2 同样用 `hash()` 采样器 ⇒ 跳帧数字不可复现、已作废**，仅留档 |
| `rt_search_t0080_v3_seedfix.md` | run3（**规范搜索产物**）的搜索报告 | 同上；`zlib.crc32` 确定性采样器；🔴 **批次状态：run3 是唯一可复现的搜索批次**，但它的 objective-best（0.952186）**全量口径 FAR 0.0352 > 0.03 破线被拒** ⇒ 交付组取的是 `cc2_md10`（0.951398），不是它 |
| `rt_eval_t0080_v3_opt.md` | 三臂复核报告（同名 json 现渲） | 同上；🔴 **批次状态：采样器修复前 ⇒ 跳帧数字不可复现、已作废**，已被 `rt_eval_delivered_armA.json` + `shortlist_deterministic.json` 取代；其正文表格里的旧值属「原始记录」，**以文件头部的口径声明为准**（第 82 轮加的声明） |
| `render_search_report_md.py` | **md 渲染器**（纯 stdlib，无 numpy 依赖）：`python3 render_search_report_md.py <report.json> [report.md]` | 第 81 轮新增并随包交付 ⇒ 拿到包就能从 json 复渲 md；闸门 §6g 用它现渲一遍**逐字节比对** ⇒ md 不可能与 json 漂移 |
| `rt_eval_delivered_armA.json` | 🔴 **跳帧三臂的规范产物**（`--phase eval`，base=v3 / opt=交付参数，确定性采样器） | 逐帧天花板 0.960186 ｜ v3 未调 0.084122 ｜ 交付参数 0.951398 |
| `shortlist_deterministic.json` | 🔴 **14 组候选的规范重测**（一个进程内测完，只 load 一次缓存，1562.8 s） | 跳帧列 / 逐帧截断列全部出自这里；与上一行**逐格交叉一致** |
| `remeasure_shortlist_deterministic.py` | 上面那份重测的脚本 | 可复现重测的唯一入口 |
| `final_selection.json` | **两阶段选型证据**：搜索 objective-best → 全量口径闸门 → 交付参数 | 🔴 **交付参数来自这里的全量闸门，不是 objective-best**；`_deterministic_remeasure_20261120` 键记录了第二轮口径更正与选型措辞修正 |
| `track_all_v3_optgapfit.json` | 交付参数的**全量口径**评测（24 序 / 165,223 帧） | **交付判据就看这个** |
| `track_all_v3_optgapfit_repro.json` | 🔴 **可复现性闭环**：用交付参数档**原样重跑** track 臂的结果 | 与上一行 **320 键中仅 `elapsed_seconds` 不同**，`overall` + 24 条 per_sequence **逐位一致** |
| `repro_preds_md5.txt` | 🔴 上面那次重跑的**逐文件 md5 对照**（25/25 相同）+ JSON 结构 diff | 证明预测输出逐位可复现 |
| `repro_track_arm.log` | 上面那次重跑的**真实 stdout/stderr + `/usr/bin/time -v`**（RC=0，wall 3:27.52，峰值 RSS 378 MB） | |
| `track_all_v3_optgapfit_repro3.json` | 🔴 **第 3 次跑批**（带同期 iostat 采样器）的评测 json | 3 次跑批 320 键均只差 `elapsed_seconds`（80.06/205.77/117.75 s） |
| `repro3_track_arm.log` | 第 3 次跑批的 stdout/stderr + `/usr/bin/time -v`（RC=0，wall 2:00.25，CPU 91%，峰值 RSS 378 MB） | |
| `repro3_iostat.log` | 🔴 **第 3 次跑批的同期 iostat 留证**（60 个采样点） | `sda` %util 94.0~100.2、读 45.7~84.4 MB/s；`nvme0n1` 仅 0.0~4.2% ⇒ **I/O 瓶颈在 sda** |
| `host_push_sim4.log` / `host_push_sim5.log` | 推送路对**模拟 host** 的端到端复测（6/6 INSERT + 正文 md5 一致 + 3/3 并发章节在位 + host 独有行存活 + 二次 6/6 NO-OP） | 真实 host 仍关机时，这是唯一能拿到的链路证据；sim5 另验证了**带时间戳的备份**与**新 artifacts 全部落地** |
| `track_all_v3.json` | v3 原参在同一份重建缓存上的全量评测 | 纯参数效应对照 |
| `cmp_v3_optgapfit_vs_official_v3.json` | 交付组 vs 官方 v3 逐序列对账 | 🔴 `max-df1-pp 0.1` 闸门 **rc=1（预期）**：换了参数组，逐序列 F1 必然变；GT 仍 24/24 一致 |
| `cmp_v3_optgapfit_vs_rebuilt_v3.json` | 交付组 vs 重建缓存上的 v3 | **rc=0**（GT 24/24 一致） |
| `cmp_v3_rebuilt_vs_official_v3.json` | 同参 v3 的**可复现性**对照 | **rc=0** ⇒ 缓存与流水线没坏 |
| `memory_section_257_18.md` | 本条线的 memory 章节正文·**主档版**（与 6 份主档里的 §257 十八逐字节一致） |🔴 含「二十二」小节（主机回填路径）～**「二十八」小节（源码级复核：任务书硬性要求逐条回源码核验 + 抓到并修掉的两处文档不一致）**；🔴 **第 80 轮更正跳帧数字口径**：§5 三臂表 / §7 短名单表的跳帧列 / md8 等价解差值，原写 0.9521 / 0.9374 / 0.9520 / 0.951990 等**作废批次**（修复前采样器）⇒ 全部改成确定性重测规范值 0.9514 / 0.9383（逐工况 0.9554/0.9530/0.9389/0.9583 与 0.9495/0.9314/0.9145/0.9581），md8/md10 差值 0.0025pp → **0.0183pp**；「天花板」标签原指向 0.9586（其实是交付参数的逐帧臂）⇒ 改为「最优参数·逐帧臂（截断口径，非天花板）」并补上真正的天花板 0.9602（FAR 0.0176） |；＋ **第 88 轮补充核验**（un-pass 归因 / 渲染判据 / TSV↔JSON 一致性，见该段第 8 条） |
| `memory_section_257_18_compact.md` | 同上·**精简档版**（与 6 份精简档一致） | 精简档正文与主档**不同**（压缩版），别串档位；同样含「二十八」；第 80 轮与主档**同步**做了同样的跳帧数字口径更正（0.9514 / 0.9383 / 逐工况 / md8 差值 0.0183pp / 天花板标签），且第 80 轮把本档**纳入 `build_delivery.py` 的 copies**（此前只有主档从 `search/` 覆盖、紧凑档是手工副本 ⇒ 两份正文会朝不同方向漂） |
| `trial0080_delta_search_spec_20261119.md` | 搜索规格书（§8.1~§8.8 八条方法论 + §9.1~§9.6 收口 + **§10 源码级复核**） | §8.6 = arm5 口径澄清；§9.6 = 可复现性闭环；**§10 = 硬性要求逐条回源码核验（含两处文档不一致的修正记录）**；**§10.4 = 板卡施工单跳帧数字口径更正（第 78 轮）**；**§10.5~§10.11 = 硬证据补交 / 清单一致 / 跳帧口径 / 同源唯一 / 误引与作废值全包扫描 / 合并工具三方对不上（第 83 轮）/ 独立审计后的判据加固 + 交付包卫生 + 两路交付集一致（第 84 轮）** |
| `trial0080_board_fusion_preset_20261119.md` | 板卡侧 Δn 默认值同步施工单（**板卡断电，未做**） | 第 74 轮补「交付参数档 + `params_md5 = a1157e013ebb496731be805c72c95dd8`」追溯行（修 r73 缺口：施工单未写交付档 md5）；板卡施工前按该 md5 核对交付档未改动，复算式 `md5(json.dumps(params,sort_keys=True,ensure_ascii=False))`；🔴 **第 78 轮口径更正**：§3 跳帧数字原是**作废那批**（0.9521 / 0.9374 / 0.951990 / 0.952015），已改成确定性采样器规范值 **0.951398 / 0.938340 / 0.951215**（板卡工程师正是拿这几个数当验收目标，照错数验收等于白干）；源文件与包内副本 md5 一致（`f1047887470035296b9eab1893ff3c84`） |
| `memory_sync_incident_20261119.md` | 本夜 memory 并发同步事故与修复留痕 | 自造事故，必读 |
| `memory_md5_manifest.txt` | 6 份 memory 的**整档 md5（信息项）** + **「§257 十八 段 md5（真正的闸门）** + 两份正文的 md5 | 🔴 **改档后用 `python3 artifacts/regen_manifest.py` 重生成**；整档会被并发线改写（故只作信息项），闸门用「段 md5」——并发线永远不会碰段 |
| `regen_manifest.py` | 上面那份清单的**生成器**（一条命令重生成，不再手写） | 服务器侧与 host 侧都能跑 |
| `HOST_PUSH_ONCE.sh` | **一次性幂等推送**（服务器侧每次值守时调一次）：host 可达即推送+复核 | 0=成功 / 3=host 仍关机 / 1=失败；第 77 轮**首次对它做端到端复测**（模拟 host）：可达⇒rc=0 且 host 侧 6/6 合并 + 交付包 35/35 一致、二次⇒6/6 `NO-OP`、不可达⇒rc=3 跳过 |
| `HOST_PUSH.sh` | 服务器侧推送（host 上电后一条命令） | 第 76 轮：交付包 `artifacts/` 改为**按目录实际内容整搬**（只排 `.bak*`），不再维护手写白名单 —— 白名单会随证据增减漂移，host 上会缺「规范搜索产物」「OSD 媒体核验脚本」等**已在本 README 登记的交付物**；实测两条路交付集**字节级一致**（**35 件**是**第 77 轮当时口径**，缺失 0 / 多余 0 / md5 不符 0）；🔴 **第 84 轮更正**：拉取路 `HOST_BACKFILL.sh` 此前只排 `__pycache__`、**不排 `*.bak*`** ⇒ 实际会多带 **12 个 `.bak*`**（其中 4 个是**旧版闸门脚本** `verify_all.py.bak_*`）到 host，两路交付集**其实不一致**；现已让拉取路也排 `*.bak*`，两路口径统一，并由闸门新增一条断言盯住 |
| `HOST_PUSH_WATCH.sh` | 常驻守望脚本（长驻 screen/tmux 环境下可用） | 🔴 本容器无可用守护机制，**不指望它常驻** |
| `merge_memory_section.py` | 章节级合并工具：只改写「§257 十八」块，**块外字节逐字节保留** | 两条回填路共用；🔴 备份带时间戳 `*.bak_pre_push_YYYYmmdd_HHMMSS` 且只留最近 2 份（**审计基线不会被下一次合并覆盖**）；🔴 第 83 轮：这份**曾是手工放进包的 00:44 版**，server `search/` 里跑的却是 22:15 旧版（固定名备份 ⇒ 每次覆盖）⇒ **两份不同源**，现已归一并由 `build_delivery.py` 在包根显式刷新（**不进 artifacts/，免得同一份东西在包里出现两份**），闸门 §6i 逐字节兜底；🔴 **第 84 轮**：归一方向写实为「**server `search/` 是源**」，且 `build_delivery.py` 在两端不一致时**直接中止、不静默覆盖**（r82 的失败模式正是包内被手工改后又被 build 悄悄盖掉） |
| `verify_all.py` | 端到端核验脚本 | 当前 **ALL CHECKS PASSED / RC=0**（**190 项 PASS / 0 FAIL**：第 75 轮加「清单体积口径」2 项、第 77 轮加「零人工路径」1 项、第 78 轮加「板卡施工单 ↔ 交付参数档一致性」4 项、第 79 轮加「硬证据随包交付」7 项 + 「交付清单 ↔ README 登记」2 项、第 80 轮加「跳帧数字口径一致」3 项、第 81 轮加「同一份东西只能有一个版本」5 项、第 82 轮加「跳帧差值现算 + 生成器无作废字面量」4 项、第 83 轮加「memory 更新工具链」7 项、第 84 轮加「工具链判据加固」5 项 + 「交付包字节码」1 项 + 「两路筛选口径一致」1 项 + 「交付清单登记改双向」0 项（改判据不加项数）、第 85 轮加「参数文件唯一性 + 整档覆盖防线」5 项、第 86 轮加「闸门项数自证」1 项）；🔴 **仅服务器侧可跑**（依赖 `/mnt/data/siping/runs/trial0080_onnx/` 的跑批产物），host 上会打印 `[SKIP]` 并退出 2 |
| `check_osd_media.py` | **OSD 渲染校验**（ffprobe）：每个 mp4 可解析、**帧数 == 该序列处理帧数**（`per_frame` 长度 ⇒ 无截断/无丢帧）、帧率恒 25.000、1920×1080；**文件名内嵌 F1 / TSV `f1_pct` / TSV `group` 列三者 vs 评测 JSON 逐序列对账** | 第 68 轮新增（"文件数对"不等于"渲染完整"），第 72 轮并入一致性对账；**反向对照**：把它指向 v3 评测 json 会报 19/24 不符 rc=1 ⇒ 检查非空转；🔴 仅服务器侧可跑（依赖 `preds_v3_optgapfit/`）；exit 0=全通过 / 1=有失败 |
| `osd_media_check.json` | 上面那次核验的逐文件结果（**24/24 通过、0 失败**） |
| `COMPARISON.md` | **交付对照表**（v3 / 交付组 / 旧 gapfit 三方逐项对照 + 口径说明），在交付包**根**目录 | 第 84 轮补登记：此前包根 4 个交付物从未进本清单（闸门 §6e 只查 artifacts/ ⇒ 它们是「未登记」状态）；现改为双向校验，artifacts/ 与包根都算 |
| `HOST_BACKFILL.sh` | **host 侧拉取路**：host 开机后由人在 host 终端执行（`bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh`） | 第 84 轮补登记 **+ 补推**：此前**两条路都不搬这个文件** ⇒ 文档让 host 执行它，host 上却根本没有；现由 `HOST_PUSH.sh` 随交付包一起推过去 |
| `osd_manifest.json` | 新 OSD 目录的逐文件清单（24 个 mp4 + pass/un-pass 分组），在交付包**根**目录 | 第 84 轮补登记 |
| `system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json` | 🔴 **交付参数档**（`params_md5 = a1157e013ebb496731be805c72c95dd8`），在交付包**根**目录 | 第 84 轮补登记 | |

| `rt_search_t0080_v3_r88.json` | **r88 搜索产物**：搜索器把逐帧硬约束搬到**交付口径**后的首次搜索 | 第 88 轮补进包：此前搜索器的硬约束量在 4000 帧截断口径上（系统性低报 FAR 约 0.0055）⇒ 会出现「搜索口径守线、交付口径破线」；本产物的 argmax = `cc2_md16n`（**与 r87 人工短名单的最高点逐键相同**） |
| `rt_search_t0080_v3_r88.md` | 同名 json 现渲的搜索报告 | 同上；🔴 **批次状态：可复现**（确定性采样器 + 交付口径约束），是继 `_seedfix` 之后**唯一**既可复现又口径正确的搜索批次 |
| `rt_search_t0080_v3_r88.log` | r88 搜索的**真实运行日志**（每一步 objective 的原样输出） | 第 88 轮补进包：文档里引用的每一个 objective 数字都能在这份日志里逐行对上 |
| `track_all_v3_optgapfit_md16.json` | **候选组 `cc2_md16n` 的全量交付口径权威评测**（24 序 / 165,223 帧） | 第 88 轮补进包：F1 0.945804 / FAR 0.029348，**逐位复现** r87 的 `eval/shortlist/track_cc2_md16n.json`。🔴 **它不是交付评测**（交付的是 `track_all_v3_optgapfit.json`） |
| `cmp_v3_optgapfit_md16_vs_official_v3.json` | 候选组 vs 官方 v3 的逐序列对账 | `GT identical 24/24`；`|dF1|>0.1pp` 共 11 条 = **参数效应**（冻结交付组 r87 同口径也是 11 条），不是故障 |
| `cmp_v3_optgapfit_md16_vs_rebuilt_v3.json` | 候选组 vs 重建缓存上 v3 的逐序列对账（同一份缓存 ⇒ 纯参数效应） | `[CMP GATE PASS]` rc=0 |
| `self_check_optgapfit_md16.log` | 候选组的 `[GATE PASS]` 证明 | 第 88 轮补进包：候选组的 tp/fp/fn/gt/recall/precision/f1/far **8 项 expected==actual** |
| `osd_manifest_md16_candidate.json` | **候选组 OSD 目录**的逐文件清单（24 个 mp4 + pass/un-pass 分组） | 第 88 轮新增：🔴 **候选，不是交付** —— 交付 OSD 清单仍是包根 `osd_manifest.json`（`match_dist=10.0` 那组），候选组切换待板端 FAR 复核 |
| `osd_media_check_md16_candidate.json` | **候选组 OSD 的媒体完整性核验结果**（ffprobe 逐个 mp4：可解析、帧数 == 该序列处理帧数、帧率恒 25.000、1920×1080；文件名内嵌 F1 / TSV `f1_pct` / TSV `group` 列三者 vs 评测 JSON 逐序列对账） | 第 88 轮新增：**24/24 通过、0 失败**，TSV 逐序列 F1 与评测 JSON 一致、group 列与目录分组一致 ⇒ 「切换就绪」这句话有证据，不是空话 |

## 🔴 本轮修掉的渲染器缺陷

`manu/videos/render_sys_sota_osd.sh` 的 `# global params:` 行原是**硬编码 5 键** ⇒ **6 个 Δn 旋钮没进 TSV**
⇒ 拿到 TSV 的人会以为参数"没生效"。已改成**从 `params` 字典派生**（无硬编码列表可漂移），
并把新 OSD 目录的 TSV **只重写第 1 行**（数据行逐字节一致，**24 个 mp4 未重渲**）。
旧 gapfit 对照目录**未动**。

## 🔴 host 回填（host 开机后）

**首选 = 服务器侧推送**（服务器持有 host 的私钥 `~/.ssh/id_ed25519_host_zerg`）：

```bash
bash manu/memory/delivery_20261119/HOST_PUSH.sh --check   # 只探测，不写
bash manu/memory/delivery_20261119/HOST_PUSH.sh           # 正式推送 + 主机侧复核
```

它**只把「§257 十八」块**写进 host 的 6 份 memory（缺失→插入 / 旧版→原位替换 / 已一致→no-op），
**块外字节逐字节保留** ⇒ 并发线的 §258/§259/§260 与 host 独有内容不动，**没有整文件覆盖风险**。
已用「服务器自身当模拟主机」跑通端到端（6/6 插入、host 独有行 6/6 存活、块外不变、二次推送全 no-op）。

**零人工 = 每次值守调一次**：`bash manu/memory/delivery_20261119/HOST_PUSH_ONCE.sh`
（幂等；0=成功 / 3=host 仍关机 / 1=失败；日志 `search/host_push_watch.log`）。
🔴 本容器**无可用守护机制**（cron 未运行且 spool 不可写、nohup/setsid/screen 后台进程随会话结束被回收，实测两次），
所以不做常驻守望，改成"每次值守调一次"。

**备选 = host 侧拉取**：`bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh --check` 先预检再去掉 `--check`
（**已改成章节级合并，不再整文件覆盖**；闸门 = 正文 md5 一致 + §258~§260 仍在位）。
⚠️ 两条路的拓扑/认证事实与兜底路径见 `COMPARISON.md` §6.3~§6.4。

## 🔴 三方一致性校验

交付参数档 ↔ `search_json_for_render.json` ↔ `per_sequence_hyperparams.tsv` 全局参数行，**37 键逐键相等**（已固化成 `verify_all.py` 自动校验）⇒ 三个流向不同人的文件描述的是同一个工作点。

## 🔴 最重要的一句话：**交付参数 ≠ 搜索 objective-best**

搜索器在 **4000 帧截断口径**上最大化跳帧臂，而该口径**系统性低报 FAR ≈ +0.005~0.006**（绝对值）。
⇒ 搜索 objective-best（`confirm_consec 2 / confirm_gap_tol 8 / max_age_phys 10 / r_slope 5.0 / match_dist 16.0`）
在截断口径 FAR 0.029013（看着守线），**全量口径 0.035152 ⇒ 破线 ⇒ 被拒**。
⇒ 最终交付 = 短名单里在**全量口径**上过闸门的 `cc2_md10`
（`confirm_consec 2 / confirm_gap_tol 8 / max_age_phys null / r_slope 5.0 / alpha_decay 1.0 / match_dist 10.0 / min_hits 3`）。
详见 `final_selection.json`。

## 第 75 轮修订（2026-11-20 02:25~02:35）

1. 🔴 **`osd_manifest.json` 的 `total_bytes` 名不副实已修**：原值是「各文件 `size_mb` 求和」（十进制 MB，5797.6），
   字段名却叫 `total_bytes` ⇒ host 侧照名字当字节读会**差 10⁶ 倍**（拿 MiB 去比则差 4.86%）。
   根因在生成器 `/mnt/data/siping/runs/trial0080_onnx/search/build_delivery.py` ⇒ **改生成器而不是改产物**：
   `total_bytes` 现在是**真实字节数**（5,797,668,961），另加 `total_mb`（5797.7）与 `size_unit`（写明 1 MB = 10⁶ 字节）。
   清单已重新生成并逐文件复核：24/24 文件名 / 分组 / 体积与磁盘一致，缺失 0、未登记 0、分组不一致 0。
2. 🔴 **生成器不再回退交付文档**：生成器自带的 `COMPARISON.md` 模板比线上那份**旧**（缺口径更正、缺 s1~s4 分工况列、缺 §6 host 回填章），
   重跑会静默覆盖成旧版 ⇒ 生成器改为「线上文件已含『口径更正』就跳过不覆盖」。
   第 75 轮实测：重跑后 `COMPARISON.md` md5 前后**完全不变**（`d1479c1a747fdc20d5c3c1ff84987caa`）。
3. 🔴 **两份板卡施工单副本原先不一致**：r74 的「交付参数档 md5」追溯行只打在 `artifacts/` 副本上，
   仓库源文件 `manu/memory/trial0080_board_fusion_preset_20261119.md` 没有 ⇒ 重跑生成器会把它覆盖掉。
   第 75 轮把同一行补进**源文件**，两份副本现 md5 一致（`524d769e01fb73b506c597b3de11561d`）。

## 第 76 轮修订（2026-11-20 02:31~02:40）

1. 🔴 **推送路的交付包搬运改为「按目录实际内容整搬」**：原先 `HOST_PUSH.sh` 用**手写白名单**（21 个文件），
   白名单会随证据增减漂移 ⇒ host 上会缺文件，而本 README 早已把它们登记成交付物
   （漏掉的有：`rt_search_t0080_v3_seedfix.json`「唯一可复现的规范搜索产物」、`check_osd_media.py` +
   `osd_media_check.json`「OSD 媒体核验」、三份 `cmp_*.json` 对账、`rt_eval_t0080_v3_opt.json` 等）。
   改为 `find artifacts -maxdepth 1 -type f ! -name '*.bak*'` 后按实际内容 scp（**保持 scp 依赖，不引入 host 端 rsync 依赖**），
   脚本净减 13 行。拉取路 `HOST_BACKFILL.sh` 本来就是 rsync 整个 `artifacts/`，两路现已**语义一致**。
2. **两条路端到端复测（模拟 host，sshd 127.0.0.1:32222）**：
   - 推送路 `/tmp/hostsim6`：**6/6 INSERT** + 正文 md5 一致 + §258/§259/§260 **3/3** + host 独有章节 **6/6 存活**
     + 交付包 **35/35 逐字节一致**；**第二次推送 6/6 NO-OP**（幂等）⇒ rc=0。
   - 拉取路 `/tmp/hostsim6_pull`：6/6 合并 + sec_md5 全对 + `concurrent_kept=3/3` + 交付包 **35/35 逐字节一致** ⇒ rc=0。
   - 留证：`search/host_push_sim6.log`、`search/host_push_sim6_run2.log`、`search/host_backfill_sim6.log`。

## 第 77 轮修订（2026-11-20 02:37~02:42）

1. 🔴 **零人工路径首次端到端复测**：`HOST_PUSH_ONCE.sh`（每次值守调一次的那条路）此前只被测过「host 关机」分支，
   从未对着一个能连的 sshd 跑通过 ⇒ 它一旦坏了，每晚值守会**静默空转**而闸门照样绿。
   复测（模拟 host `ssh 127.0.0.1:32222` + `/tmp/hostsim7`）：
   - **可达** ⇒ rc=0 `[SUCCESS]`：host 侧 6/6 memory 的 §257 十八 正文 md5 全对 + §258/§259/§260 **3/3**
     + host 独有章节 **6/6 存活** + 交付包 **35/35 逐字节一致** + 包根目录 4 件齐全；
   - **再跑一次** ⇒ rc=0，**6/6 `NO-OP`**（幂等）；
   - **不可达** ⇒ rc=3 `[WAIT] … 本次跳过`（本夜对真 host `192.168.0.115` 每轮都是这一支）。
   留证：`search/host_push_once_sim7.log`。
2. **闸门补 1 项**：「零人工路径真的调 `HOST_PUSH.sh`（否则每次值守等于没推）」——
   断言的是**调用链**而不是文件名清单。**反向对照**：把 `HOST_PUSH_ONCE.sh` 里的推送调用摘掉 ⇒
   `verify_rc=1` 且恰好 1 FAIL；还原后 md5 与改前逐字节一致 ⇒ 判据非空转。闸门 146 → **147 项**。

## 第 78 轮修订（2026-11-20 02:45~03:05）

1. 🔴 **板卡施工单 §3 的跳帧数字是「作废那批」，已改成规范值**：
   施工单是**板卡复电后照着施工的唯一凭据**，板卡工程师会拿其中的跳帧 F1 当**验收目标**。
   原 §3（表格 + prose + §3.1）写的是 **0.9521 / 0.9374 / 0.951990 / 0.952015** ——
   这几个数出自**采样器修复前**的那批产物，**不可复现**，且与规范值差 0.07pp 量级。
   规范值（确定性采样器，`artifacts/shortlist_deterministic.json` 逐格交叉一致）应为
   **跳帧·物理帧基 0.951398 / 跳帧·密集帧基 0.938340 / md8 变体 0.951215**。
   ⇒ 已改正文，并在 §3 加「本表跳帧数一律用确定性采样器重测口径」的说明 + 点名 0.9521 / 0.9374 已作废、
   实测与 0.951398 相差 ~0.07pp 属**口径差**（防止板卡端把自己的实测值误判成"没对齐"）。
   源文件 `manu/memory/trial0080_board_fusion_preset_20261119.md` 与包内副本 md5 一致：`f1047887470035296b9eab1893ff3c84`。
2. **闸门补 4 项（第 6c 节「板卡施工单 ↔ 交付参数档一致性」）**：
   施工单是**人工维护的文档**，没有闸门就只会漂 —— 上一轮正是这样漂出了 4 个作废数字。
   新增断言：
   - 「板卡施工单随包交付」（文件存在）；
   - 「施工单里的跳帧数 == 确定性重测规范值」（从 `shortlist_deterministic.json` 取 0.951398 / 0.938340 / 0.084122 比对，**不写死常量**）；
   - 「施工单的 7 个 Δn 参数 == 交付参数档逐键一致」（解析施工单参数表第 3 列 → 与
     `system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json` 的 `params` 逐键比数值，
     `max_age_phys` 比 `null` 字样）；
   - 「施工单的冻结阈值 == 交付参数档」（`th_base=0.48` / `th_ground=0.26` / `sky_ratio=1.0`）。
   **反向对照（3 次，逐条证明非空转）**：把 0.951398 改成 0.952100 ⇒ `rc=1` 且恰好该条 FAIL；
   把 `match_dist` 改成 16.0 ⇒ `rc=1` 且恰好「7 个 Δn 参数」那条 FAIL；
   把 `th_base=0.48` 改成 0.45 ⇒ `rc=1` 且恰好「冻结阈值」那条 FAIL。
   每次还原后施工单 md5 均回到 `f1047887470035296b9eab1893ff3c84`，闸门复跑 **151 PASS / 0 FAIL / rc=0**。
   （`verify_all.py` 备份：`artifacts/verify_all.py.bak_20261120_0247`。）
3. **施工单里 4 处源码行号复核通过**（任务书硬性要求「逐条回源码核验」）：
   `system_sota_longquan.py:382`（`max_gap = 1 + confirm_gap_tol`）、
   `eval_bidirectional_track_fusion.py:251`（`match_dist + r_slope*(dt-1) + 0.5*velocity_norm`）、
   同文件 `:311-312`（`max_age_phys` 接管 `dead` 规则）、`:132-141`（`alpha_decay` 公式）—— 四处行号与语义均与源码一致。

## 第 79 轮修订（2026-11-20 02:52~03:15）

1. 🔴 **三件「被文档当证据引用、却没随包交付」的硬证据补进包**（审计口径：把交付包三份文档里
   所有 `search/*`、`eval/*` 反引号路径抽出来，逐个查「服务器上有 / 包内有没有」）：
   | 证据 | 文档在哪引用 | 此前状态 | 现在 |
   | --- | --- | --- | --- |
   | `self_check_optgapfit.log` | 规格书 §10.1「self-check 必须 `[GATE PASS]`」 | 🔴 只在服务器 `eval/` 下，**没随包** | ✅ 随包（612 B） |
   | `system_sota_longquan_trial0080_singleframe_gate_baseline.json` | 日志正文第一行指名的 baseline | 🔴 没随包 ⇒ **日志不可自证** | ✅ 随包（891 B） |
   | `space_t0080_v3.json` | 规格书 §10.1「搜索空间逐项精确一致」 | 🔴 没随包 ⇒ **该断言不可核** | ✅ 随包（285 B） |
   ⇒ 后果很具体：host / 嵌入式组拿到包后，`[GATE PASS]` 与「空间覆盖 7 个旋钮」这两句话**只能凭空信**。
   **改生成器而不是改产物**（r75 教训）：`/mnt/data/siping/runs/trial0080_onnx/search/build_delivery.py`
   的 `copies` 字典新增 3 条 ⇒ 重跑生成器即自动随包，不会再漏。
2. **闸门加 7 项（§6d「硬证据随包交付」）**：
   - 三件文件在包内；
   - self-check 日志 **8 项 expected == actual**（tp/fp/fn/gt/recall/precision/f1/far_per_frame）；
   - self-check 日志的 expected **逐项 == 随包冻结 baseline**（计数项精确、浮点项 1e-9）；
   - 搜索空间**逐旋钮覆盖任务书要求的值集合**（7 个旋钮，含 `max_age_phys` 的 `null`）；
   - 搜索空间**保留 v3 可回退点**（`confirm_consec=7` / `confirm_gap_tol=0`）。
   **反向对照 4 次，逐条命中**：改日志 actual 的 tp ⇒「expected==actual」FAIL；
   改 baseline 的 expected tp ⇒「日志 == 随包 baseline」FAIL；
   删空间的 `match_dist=12.0` ⇒「空间覆盖」FAIL；
   把 `[GATE PASS]` 改成 `[GATE FAIL]` ⇒「证明随包交付」FAIL。
   四次还原后三件文件 md5 全部回到原值（`689da4a5…` / `8137dec5…` / `4ad32b7f…`）⇒ 判据非空转。闸门 151 → **158 项**。
3. **artifacts 目录 35 → 38 件**（非 `.bak` 计数），两条回填路都按「目录实际内容整搬」发货 ⇒ 自动带上这 3 件。

4. **两条回填路都实测能把新证据送到 host**（不是"应该会"，是真跑过）：
   - 推送路（零人工路径 `HOST_PUSH_ONCE.sh` → `HOST_PUSH.sh`，对模拟 host `ssh 127.0.0.1:32222`）：
     **rc=0**；host 侧 artifacts **35 → 38**；**38/38 逐文件 md5 一致**（缺失 0 / 多余 0 / 不符 0）；
     host 侧 `§257 十八` 在位（1 处）+ 并发线 `§258/§259/§260` **3/3 存活**。
   - 拉取路（host 侧主动拉 `HOST_BACKFILL.sh`，模拟 host 目录 `/tmp/hostsim8_pull`）：**rc=0**；
     artifacts → **38**，三件新证据 md5 与服务器一致；6 份 memory 合并 OK、`concurrent_kept=3/3`。
   - 留证：`search/host_push_once_sim8.log`、`search/host_backfill_sim8.log`（**服务器侧留证，不随包**）。
5. **闸门加 2 项（§6e「交付清单 ↔ README 登记一致」）**：本文件抬头写的是「artifacts 清单」，
   但此前**没有任何东西强制它与目录实际内容一致** ⇒ 证据一多，登记就会悄悄漂（r78 的施工单漂移就是这类）。
   现在断言：① `artifacts/` 里每个交付物都在本清单登记（README.md 自身除外）；② 本清单登记的文件都真实存在。
   **反向对照 3 次**：塞一个没登记的文件 ⇒ ①FAIL；删掉 `space_t0080_v3.json` 那一行 ⇒ ①FAIL；
   登记一个不存在的文件 ⇒ ②FAIL。三次还原后 README md5 均回到 `02c233c4fb8ce23c1c410ae94ac624e1`。闸门 158 → **160 项**。

## 第 80 轮修订（2026-11-20 03:04~03:45）

1. 🔴 **r78 的缺陷还有 3 份同类文档没修**（当时只修了板卡施工单）：
   审计口径 = 把交付包 5 份文档里所有出现**作废跳帧值**（0.9521 / 0.9374 / 0.951990 / 0.952015 / 0.9520 / 0.952127 / 逐工况 0.9558…）的段落全捞出来，
   逐段看它**是不是被当成现值在用**。结果：**主档 §5 三臂表、§7 短名单表的跳帧列、md8 等价解差值、compact 同款表、规格书 §4.2/§8.4 三臂表**，
   全都还在拿作废值当现值 ⇒ host/嵌入式组读到的就是错的数（还错在**结论句**上：「mean F1 = 0.9521 … 差 +1.47pp」）。
   权威真值取 `artifacts/shortlist_deterministic.json`（确定性采样器）：

   | 口径 | 作废值（修复前采样器） | 规范值（重测） |
   | --- | --- | --- |
   | 交付 · 跳帧·物理帧基 | 0.9521 | **0.951398**（逐工况 0.9554/0.9530/0.9389/0.9583） |
   | 交付 · 跳帧·密集帧基 | 0.9374 | **0.938340**（逐工况 0.9495/0.9314/0.9145/0.9581） |
   | 物理基 − 密集基 | +1.47pp（错） | **+1.31pp** |
   | md8 vs md10 跳帧差 | 0.0025pp（错） | **0.0183pp**（md8=0.951215 / md10=0.951398） |
   | 「逐帧天花板」 | 0.9586（标签指错对象） | **0.9602**（FAR 0.0176）；0.9586 是**交付参数的逐帧臂**（截断口径） |

2. 🔴 **正文改动怎么落到 6 份 memory**：走**章节级合并工具** `merge_memory_section.py`（只改写「## §257 十八」块，
   块外逐字节保留）⇒ 6 份 memory 的 §257 十八 块 = 新正文（主档 md5 `2b17e9d5…` / 紧凑档 `548d43fb…`），
   **块外 6 份 md5 前后完全一致**（实测 `4f1796a6…` ×3 + `611cf46a…` ×3），并发线 `§258/§259/§260` **3/3 存活**。
   `memory_md5_manifest.txt` 已用 `regen_manifest.py` 重生成（§257 十八 段 md5 = 真闸门，整档 md5 仍是信息项）。
3. 🔴 **抓到一个流程坑并修掉**：本轮改的是**包内副本** `artifacts/memory_section_257_18.md`，但 `build_delivery.py` 是从
   `search/memory_section_257_18.md` **重新覆盖**的 ⇒ 一重跑生成器，改动就被打回原形（闸门当场报出段 md5 漂移）。
   ⇒ 改**源头**（`search/` 下两份正文），并把**紧凑档也加进 `copies`**（此前只有主档从 `search/` 覆盖、
   紧凑档是手工副本 ⇒ 两份正文必然朝不同方向漂，正是本轮这类缺陷的温床）。
   现在两份正文源与包内副本 md5 **两两一致**。
4. **闸门加 3 项（§6f「跳帧数字口径一致」）**：
   ① 规范值可从 `shortlist_deterministic.json` **现取**（不写死常量）；
   ② 5 份交付文档里，**作废跳帧值只允许出现在明确标注「已作废」的段落**（段落级判定，允许在口径更正段落里引用旧值）；
   ③ 每份文档都写的是**规范值**（0.9514 或 0.951398、0.9383 或 0.938340，用词边界正则，`0.951398x` 蒙不过去）。
   **反向对照 3 次**：把正文 §5 表的现值改回 0.9521 ⇒ ②FAIL（同时段 md5 闸门也报漂移）；
   抹掉规格书某段的「已作废」标注 ⇒ ②FAIL；
   紧凑档把 0.9514 与 0.951398 一起改成别的 ⇒ ③FAIL（+ 段 md5 闸门）。三次还原后 md5 全回原值。闸门 160 → **163 项**。

## 第 81 轮修订（2026-11-20 03:15~04:05）

1. 🔴 **抓到「双份不同源」：有人照 memory 里的指引会跑到旧闸门**
   `§257` 正文明写「脚本：`merge_blocks.py` / `undo_merge_blocks.py` / `verify_all.py`（`/mnt/data/siping/runs/trial0080_onnx/search/`）」，
   而 `search/verify_all.py` 是 **00:15 的旧版**（缺 §6c「施工单↔参数档一致性」/ §6d「硬证据随包交付」/ §6e「交付清单↔README 登记」/ §6f「跳帧数字口径一致」四节），
   包内 `artifacts/verify_all.py` 才是 03:11 的当前版 ⇒ **同一闸门两个版本**。
   同一目录还躺着 `search/memory_md5_manifest.txt`（00:15 版，整档 md5 全是旧值 `aefcfa9f…`/`433f9942…`、正文段 md5 也是旧值），**无任何脚本引用它** ⇒ 谁拿它 `md5sum -c` 就会得到「6 份全不匹配」的假警报。
   ⇒ 处置：`search/verify_all.py` 用包内当前版覆盖（同一闸门只有一个版本）；`search/memory_md5_manifest.txt` **删除**（权威版只有包内一份，由 `regen_manifest.py` 重生成）。

2. 🔴 **抓到 `--out-md` 是死参数 ⇒ 交付文档在声称交付不存在的文件**
   `manu/evaluation/rt_fps_jitter_search.py` 的 `--out-md` **只解析、从不使用**（全文件没有任何 `.md` 写盘代码）⇒ 历次搜索/复核的 md 报告**一份都没生成**；
   但 `COMPARISON.md` §5 明写「artifacts/ —— 搜索 json/**md**、三臂复核 json/**md**」，`build_delivery.py` 的 `copies` 里还挂着 4 条指向这些**从未存在**的 `.md` 的条目（`if s.exists()` 静默跳过 ⇒ 生成器永远不报错，缺口不可见）。
   ⇒ 处置：把渲染器做成**纯 stdlib 独立模块** `manu/evaluation/render_search_report_md.py`（`render_report_md(report) -> str` + CLI），
   搜索器改为引用它并让 `--out-md` **真出报告**；用同一函数从 4 份已存 json 现渲出 4 份 md（run1 / run2 / run3 / 三臂复核）⇒ 文档的声明变真，缺口补齐。
   渲染器随包交付 ⇒ 拿到包的人可从 json 复渲；**md 里的每个数字都取自同一份 json**，不可能自己漂。

3. **闸门加 §6g「同一份东西只能有一个版本」5 项**：
   ① 包内闸门 == `search/` 闸门（逐字节）② `search/` 下没有陈旧的 `memory_md5_manifest.txt` 副本
   ③ 生成器 `copies` **没有死条目**（每个源都真实存在）④ 包内生成物 == 其源（逐字节）
   ⑤ **md 报告 == 由同名 json 现渲染**（逐字节）⇒ 文档/生成器/产物三处任何一处漂移都会被当场抓住。闸门 163 → **168 项**。
4. **本轮不动数值**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**（本轮是文档与工具链的收口，不是新的搜索）。
## 第 82 轮修订（2026-11-20 03:35~04:25）

1. 🔴 **抓到正文里一句误引：「差 0.0025pp」根本不是那两个数的差**
   `§257 十八` 与规格书 §8.2 都写着「objective-best 与交付组跳帧臂只差 **0.0025pp**」。
   实测：**0.0025pp 是 md8↔md10 的差**（旧值 0.951990 vs 0.952015），
   objective-best(`rerun_selected` 0.952186) 与交付组(`delivered_cc2_md10` 0.951398) 的真差 = **0.0788pp**。
   ⇒ 「只差千分之几」被**低估一个数量级**；md8 等价解那条差值被**串到了错误的比较上**。
   处置：正文两份（主档 / 紧凑档）改 **0.0788pp**；md8↔md10 另写 **0.0183pp**（0.951215 vs 0.951398）。
   **差值从此不再手写** —— 一律由 `shortlist_deterministic.json` 现算，闸门现取现比。

2. 🔴 **作废跳帧值扫描扩到全交付包 + 搜索 + 评测**（此前只审 5 份文档）⇒ 无标注命中 5 处，逐处修掉：
   `rt_eval_t0080_v3_opt.md`（+`search/` 源）的各臂表 0.952127 / 0.937438、`build_final_selection.py` 硬编码 0.952015、
   `make_memory_section.py` 硬编码 JUMP_F1 + 差值 0.0025pp、`winner_knobs.json` 的 `_why` 把 0.952015 当现值。
   ⇒ 4 份**自动生成报告**改为在**头部**声明「跳帧数字不是交付口径」+ 指向规范源
   （它们是「某一次跑批的原始记录」，携带旧值是设计如此，但读者必须一眼看出）；
   3 个**生成器脚本**里的旧字面量全部改成「现取规范源」，重跑不会再把旧值写回磁盘。

3. **`final_selection.json` 的对外声明同步**：`delivered.jump_arm_f1_4000f` **0.952015 → 0.951398**，
   新增 `jump_arm_dense_f1_4000f` = 0.93834 + `jump_arm_source`（写明规范源、旧采样器值已作废）。

4. **闸门加 §6h 4 项**（§6f 同时扩面到 9 份文件、自动生成报告改按头部声明判）：
   ① 正文两份 + 规格书的跳帧差值 == 规范源现算值 ② `final_selection.json` 交付跳帧 F1 == 0.951398
   ③ 4 份 md 头部含口径声明 + 规范源指针 ④ 3 个生成器脚本内无作废跳帧字面量。闸门 168 → **172 项**。
5. **本轮不动数值**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**。
## 第 83 轮修订（2026-11-20 03:48~04:35）

1. 🔴 **抓到「文档说的 / 服务器跑的 / 包里带的」memory 合并工具三方对不上**（本轮只修工具链，数值一个没动）
   | | md5 | 时间 | 备份命名 |
   | --- | --- | --- | --- |
   | 包根 `merge_memory_section.py` | `17224adc…` | 00:44 手工放 | `.bak_pre_push_<ts>`，留最近 2 份 |
   | server `search/merge_memory_section.py` | `d0452cdc…` | 22:15 旧版 | 固定名 `.bak_pre_push`，**每次合并覆盖** |

   ⇒ r79~r82 的 6 份合并都走的旧版 ⇒ 「合并前原样」审计基线只剩最后一次；而 `HOST_PUSH.sh` / `HOST_BACKFILL.sh` 推给 host 的**正是包内那份**。
   ⇒ 已归一（两端 md5 相同）；**第 84 轮把归一方向写实**：**server `search/` 是源**，`build_delivery.py` 用它刷新包根（**故意不进 `copies`**：进了会在 `artifacts/` 多出第二份，又回到 r81 的「双份不同源」），且**两端不一致就直接中止、不静默覆盖**（r82 的失败模式正是包内那份被手工改后又被 build 悄悄盖掉）。

2. 🔴 **抓到规格书 §7 点的更新工具会插出重复块，且不可逆**
   §7 仍写「走**外科式插入** `insert_memory_section.py` · 零行删改」—— 那是第 76 轮的做法（当时正文与块内内容逐字节相同，插入恰好幂等）。
   **第 83 轮在临时副本上实测**：正文一旦改过，它把整段再插一遍 ⇒ memory 文件出现**两个 `## §257 十八` 块**（+54,142 B）；
   再用 `merge_memory_section.py` 也**救不回来**（只替换**第一个**块，重复块留着、**新内容被丢弃**）。
   ⇒ §7 标「已作废」+ 写明当前唯一路径 = **`merge_memory_section.py`（章节级·替换式）**；
   文档一旦提到 `insert_memory_section.py`，**必须**同时带重复块 / 不可逆警示（闸门 §6i 强制）。

3. 🔴 **正文「脚本留存」清单指向的是事故工具**
   原清单只列 `merge_blocks.py` / `undo_merge_blocks.py` / `verify_all.py` —— 前两个**就是**那次「把主档灌进精简档」的事故工具，
   且**都未随包交付**（host 拿到包也跑不了）；真正在用、且随包交付的 `merge_memory_section.py` 反而没列上。
   ⇒ 清单重写为「当前可用（随包）」+「事故工具（仅 server 留证，未随包）」两栏；事故记录 `memory_sync_incident_20261119.md` 补一行「现状」（不改写历史）。

4. **闸门加 §6i「memory 更新工具链」7 项**：
   ① 规格书 §7 与正文「脚本留存」段都点名 `merge_memory_section.py`（当前路径）
   ② 文档提到 `insert_memory_section.py` 时必须同时带重复块 / 不可逆警示
   ③ 正文「脚本留存」清单点名的脚本必须真实存在于 server `search/`
   ④ 随包 `merge_memory_section.py` == server 侧那份（逐字节）
   ⑤ `artifacts/` 里**不得**有第二份 `merge_memory_section.py`（同一份东西只有一个版本）
   ⑥ **实测负证据**：闸门现跑 `insert_memory_section.py`（临时副本 + 改过的正文）⇒ 复现「2 个标题块」
   ⑦ **实测负证据**：`merge_memory_section.py` 对重复块**只改第一个** ⇒ 重复块清不掉 ⇒ 文档必须警示
   闸门 172 → **179 项**。
5. **本轮不动数值**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**。

## 第 84 轮修订（2026-11-20 03:58~04:35）

1. 🔴 **独立 reviewer 审计 r83**：3 条声称修掉的缺陷**本体**逐条核实确已修好（md5 逐字节、刷新逻辑、`copies` 无死条目、§7 已作废、清单点名的脚本都存在、179 项闸门实跑 0 FAIL）；
   但 §6i 判据有 **4 处可被绕过的空档**（reviewer 逐条构造了可复现反例），另有 3 个新缺陷。本轮据其反例逐条加固，**不动任何数值**。

2. 🔴 **正文「脚本留存」把 `verify_all.py` 写成「可在 host 侧直接用」—— 而它自带 host 侧守卫，必然 `exit 2`**
   `artifacts/verify_all.py:30-34` 自己写着：跑批产物不在本机就 `[SKIP]` + `exit(2)`；而**同一个包里随包 README 早就写着「🔴 仅服务器侧可跑」**。
   ⇒ r83 新写的清单与随包 README **自相矛盾**；host 照清单做，第一步就是静默失败。
   **处置**：清单分三栏 ——「随包交付、host 侧可直接用」(`merge_memory_section.py` / `regen_manifest.py`)、
   「随包交付、仅服务器侧可跑」(`verify_all.py`)、「事故工具（仅 server 留证，未随包交付）」；
   并加**双向**断言：标「host 侧可直接用」的脚本源码里**不得**依赖 server 跑批路径，标「仅服务器侧可跑」的脚本源码里**必须**依赖它（标注与源码相符）。

3. 🔴 **§6i 的 4 处判据可被绕过**
   ① **极性**：原判据只判「同句共现」⇒「当前更新工具 = merge_memory_section.py（其实已废弃，别用）」照样 PASS。
      现在按**句**判定，并要求那句**没有**把它否掉（勿用/禁用/作废/废弃/别用/不要用/停用/已弃用/不可用）。
   ② **警示词**：原词表含「禁止」「作废」这类**泛用词** ⇒「请用 insert_memory_section.py。（顺便：§258 禁止改动。）」能蒙混过关。
      现在只认**工具特有**的危害词：必须同时含「重复」+（「不可逆」或「只插不替」）；为此把正文与 §7 的两处措辞补成合规。
   ③ **存在性**：原判据把 server `search/` 也算「存在」（只存在于服务器、根本没随包的脚本照样 PASS），
      且正则 `` `([A-Za-z0-9_]+\.py)` `` **漏掉带路径的名字**（写成 `` `artifacts/xxx.py` `` 时抽出来是空集 ⇒ 判据**静默空转**）。
      现在**分栏**校验（basename 归一）：「随包交付」栏的脚本必须在**包内**存在；「未随包」栏的脚本必须**不在**包内。
   ④ **覆盖名单**：原名单只有 5 份包内文档，漏了 `COMPARISON.md`（随包推送，host 真会读）与**6 份 memory 正本**
      （并发线的 §258/§259/§260 就在里面）。现在 **12 份全覆盖**。

4. 🔴 **交付包卫生：2 个 `__pycache__`（其中 1 个是**陈旧字节码**），而且闸门自己每次跑都生成**
   `artifacts/__pycache__/verify_all.cpython-38.pyc` 指向的是 40,535 B 的**旧版** `verify_all.py`（现文件 62,090 B）；
   包根那两个是闸门 `python -m py_compile` 语法检查写进去的（**每跑一次生成一次**）——「同一份东西只能有一个版本」在 .pyc 上的复现。
   **处置**：语法检查改成**进程内 `compile()`**（同样验语法、零磁盘副产物）+ 删掉两处 `__pycache__`
   + 新增一条闸门断言「交付包里没有 `__pycache__` / `.pyc`」。
   两条回填路本来就排除了它（`HOST_PUSH.sh` `-maxdepth 1 -type f`、`HOST_BACKFILL.sh` `--exclude '__pycache__'`）⇒ **未污染 host**，
   危害限于本地包卫生与「同一份东西只能有一个版本」。

5. 🔴 **两条回填路的交付集其实不一致**：推送路排 `*.bak*`，拉取路只排 `__pycache__`
   ⇒ 拉取路会把 **12 个 `.bak*`**（其中 4 个是**旧版闸门脚本** `verify_all.py.bak_*`）带到 host。
   **处置**：拉取路加 `--exclude '*.bak*'`，两路口径统一 + 新增一条断言盯住。

6. 🔴 **`HOST_BACKFILL.sh` 从未被推给 host**，而文档却让 host 用户执行它
   **处置**：`HOST_PUSH.sh` 随交付包一起推 `HOST_BACKFILL.sh` ⇒ 拉取路真的可执行（服务器端无 host 仓库副本，host 上本来就拿不到这个脚本）。

7. **归一方向写实**（r83 文档写「server ← 包内」，实现是「包内 ← server」）：**server `search/` 是源**，
   `build_delivery.py` 用它刷新包根，且**两端不一致就直接中止、不静默覆盖**。

8. **闸门 179 → 184 项**（§6i「存在性」拆 1 换 2 +「host 侧可运行性」2 项 +「交付包字节码」1 项 +「两路筛选口径一致」1 项；
   §6e「交付清单登记」改**双向**（artifacts/ 与包根都算），项数不变）。
   **本轮不动数值**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**。
   §257 十八 段 md5：`0b00856d…` → **`34f0aecd604fa33ec8fa93a3334dbf2f`**（主）/ `cd37ef88786e9061552cd44bb5504ded`（紧凑，未变）；
   6 份整档 `d6d91505e7c50dcccfc23b03114dc9d8` ×3 + `5c8189892d36e940963ba2de44417eb7` ×3。

## 第 85 轮修订（2026-11-20 04:40~05:50）

1. 🔴 **交付包里混进了两份「已被全量闸门拒绝」的破线参数文件（本轮最严重）**
   `artifacts/opt_params_t0080_v3_final_flat.json` 与 `artifacts/opt_params_t0080_v3_seedfix_flat.json`（内容逐字节相同）
   装的是 run2/run3 的 **objective-best**：`match_dist 16.0` / `max_age_phys 10`（与交付组**只差这 2 键**）——
   而这 2 键正是踩线处：**全量逐帧口径 FAR 0.035152 > 0.03 红线**、F1 94.3749%（比交付组低 0.1227pp）。
   危害**实测坐实**：`manu/evaluation/system_sota_longquan.py::load_params` 是 `override = payload.get("params", payload)`
   ⇒ 嵌套 `{"params": {...}}` 与扁平两种形态**都静默接受**，喂错文件**不报错** ⇒ 直接跑出破线结果。
   命名误导：文件名带 `final` 看着像权威版（真正交付那份是 `opt_params_t0080_v3_flat.json`）；
   语境加重：板卡施工单 §3 正要把 `match_dist 16.0→10.0` / `max_age_phys 10→null` 改掉 ⇒ **误取 = 把被拒参数同步上板**。
   闸门之所以漏检：这两份**不在** `build_delivery.py` 的 copies 清单里。
   **处置**：① 交付包内删除；② `build_delivery.py` 加**自清理**（copies 之后扫 `artifacts/*.json`，用镜像 `load_params` 的
   `_effective_params()` 取有效参数，凡「有效参数 ≠ 交付参数」即剔除并打印 `[CLEAN]`）；
   ③ 上面那张清单里对应两行已删，`opt_params_t0080_v3_flat.json` 行的说明栏已改写成「另外两份已删除 + 为什么危险 + 证据在哪」；
   ④ server 侧 `search/README.md` 保留那两行（文件确实还在 server，仅留档）但加「不得上板」警示。
   **反向实测**：注入 `opt_params_REVERSE_TEST.json` ⇒ 被 `[CLEAN]` 剔除（判据非空转）。
   **证据未丢**：run2/run3 的 objective-best 全量参数在 `rt_search_t0080_v3_final.json` / `rt_search_t0080_v3_seedfix.json`
   的 `opt_params`（37 键逐键相等），选型过程在 `final_selection.json`。

2. 🔴 **`manu/videos/upload_to_server.sh` 的整树 rsync 能整档覆盖 memory**
   该脚本第 135 行是 host→server 的整树 `rsync -a --checksum --compress --partial`，**未排除任何 memory 文件**
   ⇒ host 副本与 server 不同即**整档替换**，把并发线的 §258/§259/§260 一并盖掉
   ⇒ 与「memory 只做章节级合并、绝不整文件覆盖」的铁律直接冲突。
   **处置**：`EXCLUDES` 加 6 条 memory 排除（`/memory.md` `/memory_compact.md` `/memory/memory.md` `/memory/memory_compact.md`
   `/manu/memory/memory.md` `/manu/memory/memory_compact.md`）+ 注释写明原因与替代路径（走 `HOST_PUSH_ONCE.sh` → `merge_memory_section.py`）。
   **本地模拟 rsync 实证**：6 份 memory 一份都没被传走，非 memory 文件照传。

3. 🔴 **§6i 探针加 tmpdir 硬断言**
   `insert_memory_section.py`（只插不替，会**插出重复块**且**不可逆**）是整条链里**唯一**还会 exec 危险工具的地方，
   安全性此前只靠「目标在 tmpdir 里」这一条**调用方约定** ⇒ 改成**硬断言**（`relative_to` try/except，Python 3.8 兼容）。

4. **`search/` 卫生**：与源**逐字节相同**的 `remeasure_shortlist_deterministic.py.bak`
   （md5 `30bbbc938a12fe4afd94aa24767756bb` ⇒ 零信息量）与 `__pycache__/`（3 个 `.pyc` 全部相对现源过期）已删（删除前 md5 留证）；
   剩余 6 个 `.bak*` 是**真实历史版本**，保留作留证。

5. **闸门 184 → 189 项**（§6j「参数文件唯一性 + 整档覆盖防线」5 项：参数文件唯一性 / 被拒文件不在包内 /
   反向对照非空转 / `load_params` 静默接受实测 / rsync 排除 6 份 memory），`ALL CHECKS PASSED` rc=0。
   **本轮不动数值**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**。
   §257 十八 段 md5：`34f0aecd…` → **`b7beb9ccb428e4a9ea833d84043201ae`**（主）/
   `cd37ef88…` → **`66da31732745a31a430f7c919be3364c`**（紧凑）。

## 第 86 轮修订（2026-11-20 05:09~05:40）

1. 🔴 **本清单里 `verify_all.py` 那一行的「闸门项数」头条过期 5 项，且与同一份文件自相矛盾**
   头条写的是第 84 轮当时的项数（**184**），而本文件末尾的「第 85 轮修订」写「**闸门 184 → 189 项**」
   ⇒ 一份文件里两个数字并存，读者查交付包时看到的仍是**过期**项数
   （这一行正是「闸门有多少项」的**唯一对外声明**）。
   **根因**：第 85 轮加了 5 项，只在「修订小节」里记了数，**没回头改清单头条**；闸门也**管不到**它
   —— §6e 只对「清单登记 ↔ 实际文件」做**双向对账**，**不看数量**。
   **处置**：① 头条改为第 86 轮的实际项数（**190**），枚举补上第 85/86 轮增量；
   ② **闸门加 §9「闸门项数自证」1 项** —— 末尾现算实际跑出的项数，与本文件头条里的数字对账，
   并要求该表述**全文只出现一处** ⇒ 以后再有人加项忘了改头条，闸门当场 FAIL。
   **反向对照**：头条改回「189 项」⇒ §9 翻 `[FAIL]`；改回「190 项」⇒ 恢复 `[PASS]` ⇒ **判据非空转**。

2. **本轮只修文档一致性，数值一个没动**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 /
   三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**。

3. **本轮另做两轮独立只读审计**（结论均 PASS，**未发现新缺陷**）：memory 同步链
   （6 份块 md5 逐字节对齐 / 无重复块 / 块外字节不变 / manifest 零漂移 / §258–§260 完好 / 并发线**无新段落**）
   ＋ 交付包与板卡施工单（包内**只有一份**等价参数文件 / 清单双向零漏零 ghost /
   施工单 7 个 Δn 默认值逐项相等 / 3 份 cmp 均 **24/24 GT 一致**）。

4. **闸门 189 → 190 项**，`ALL CHECKS PASSED` rc=0。
   §257 十八 段 md5：`b7beb9cc…` → **`143132e5dee6b4c81d9db012d7ccb724`**（主）/
   `66da3173…` → **`2d83e81df48bfbc6acbe5dde34aed03e`**（紧凑）。

## 第 87 轮修订（2026-11-20 05:20~05:35）

1. 🔴 **闸门从来没有证明过自己的判据「非空转」**
   前 86 轮的验收标准一直是「`ALL CHECKS PASSED` rc=0」，但**全 PASS 只说明「没触发失败」，不说明「真的在检查」**
   —— 一条恒真判据（例如 `check("x", True)`）同样会全 PASS ⇒「190 项全过」本身**不构成**闸门有效的证据。
   **处置**：新增 `negctl_sweep.py`（**只留在 server `search/`**、**故意不进本包** —— 它**会故意破坏交付包**）：
   逐个**故意破坏**交付包里的东西 → 跑真闸门 → 看它**是否真的翻红** → **逐字节还原并复核 md5**。

2. **6 例负向对照全部翻红（判据非空转）**

   | # | 破坏对象 | 报警判据 | rc / FAIL 数 |
   | --- | --- | --- | --- |
   | C1 | 本包 README 头条 190→189 | §9 闸门项数自证 | 1 / 1 |
   | C2 | 本包 `opt_params_t0080_v3_flat.json` 的 `match_dist` 10.0→16.0（把**被拒参数**塞回包） | §6g 包内生成物 == 其源 + §6j① 包内唯一参数文件 | 1 / 2 |
   | C3 | 本包 README 清单删掉 `osd_media_check.json` 的登记行 | §6e 交付物未登记 | 1 / 1 |
   | C4 | `upload_to_server.sh` 删掉一条 memory 排除项 | §6j⑤ 整树 rsync 排除 6 份 memory | 1 / 1 |
   | C5 | `search/memory_section_257_18.md` 章节标题改 1 个字符 | §6 章节源一致 + §6g 包内生成物 == 其源 + manifest 段 md5 + merge no-op | 1 / 4 |
   | C6 | 交付参数档 `th_base` 0.48→0.47 | §1 两条 + 参数档 ↔ OSD 逐键（JSON / TSV 各一条）+ §6j① + §8 三次跑批参数 | 1 / 6 |

   ⇒ **6/6 全部翻红**；每例跑完**逐字节还原**，6/6 目标文件 md5 **回到原值**（已复核）。
   工具 `/mnt/data/siping/runs/trial0080_onnx/search/negctl_sweep.py`，日志 `.../search/negctl_sweep.log`。

3. **闸门项数从哪来（190 是稳定的，不是碰巧）**
   - 源码里共 **140 个静态 `check()` 站点**，运行时是 **190 项** ⇒ 差的 **50 项**全部来自**循环里的 `check()`**；
   - 逐个核过**所有**含 `check()` 的循环/条件：**循环长度全是固定字面量**
     （7 个 Δn 旋钮 / 3 份 cmp / 7 个 TSV 全局参数键 / 6 份 memory × 7 检查 / 固定清单元组）
     ⇒ **没有一条依赖数据长度的循环** ⇒ 项数不会随数据漂移；
   - **诚实边界**：`if <产物存在>` 守卫的检查（TSV / 渲染 JSON / arm5 / 施工单 / self-check / 搜索空间 / 三次跑批）
     在**产物缺失时整段消失** ⇒ 项数**会下降**。但这**不是掩盖**：产物缺失时对应检查本就会翻 `[FAIL]`，闸门照样红，
     §9 只是**多报一条**「本文件头条写的项数 vs 实际项数」；
   - **确定性双跑**：同一份产物连跑两次，均 **190 项全过**，`[PASS]/[FAIL]` 标签序列**逐行一致**。

4. **闸门项数本轮不变（仍 190 项）**：本轮**没有新增闸门项**，只新增审计工具与文档记录；
   `negctl_sweep.py` **留在 server `search/`**、**不进本包**（§6i 的「未随包」栏会自动核「它确实不在包内」）。
   **本轮不动数值**：交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 三臂跳帧口径 / OSD pass 21 + un-pass 3 **全部未变**。

## 第 88 轮修订（2026-11-20 05:26~07:40）

1. 🔴 **审出来的真缺陷：搜索器的逐帧硬约束量在了错误的口径上**
   `manu/evaluation/rt_fps_jitter_search.py` 的 `objective()` 里，「逐帧 FAR ≤ 0.03」「逐帧 F1 不回退」两条**硬约束**
   原先量在**截断口径**（每序列前 4000 帧，全库 70,451 帧 = 165,223 帧的 42.64%）。该口径**系统性低报 FAR 约 0.0055 绝对值**
   （同时高报 F1 约 1.37pp）⇒ 会出现「**搜索口径守线、交付口径破线**」的参数组。
   **实测证据**：run2/run3 的 objective-best（`max_age_phys=10 / match_dist=16.0`）截断 FAR 0.029013 看似守线，
   **全量 FAR 0.035152 ⇒ 破线** —— 这正是上一轮 gapfit 组被拒的原因。
   第二个缺陷：`--sweeps 2` 让坐标下降在 sweep1 末**还有改进时提前收工** ⇒ 交付组从来不是搜索器自己选出来的。
   **处置**：新增 `--perframe-caliber {delivery,truncated}`（**默认 delivery** = 全量 165,223 帧）；
   `--sweeps` 默认改 **0 = 收敛即停**；报告新增 `delivery_caliber` 判定（`far_ok` / `f1_ok` / `feasible`）。
   arm1 天花板与 arm5 **显式留在截断口径** ⇒ **历史数字逐位复现**（0.960186 / 0.017558、起点 objective 0.084122）。

2. **修复后重搜：argmax = `cc2_md16n`（与 r87 人工短名单的最高点逐键相同）**
   两轮的 argmax 都是 `match_dist=16.0`（v3 基座原值，搜索从未动过它），真正的差别在 **`max_age_phys`** ——
   历史选 `age=10`（全量 FAR **0.035152 破线**），r88 选 `age=null`（**守线**）。
   r88 的 sweep2 **明确实测过交付组那一档**：md10 → objective **0.951398**、md12 → 0.951582、md16 → **0.951693（选中）**
   ⇒ 修复后的搜索器把**交付组排在 md16 之后一位**。

3. **独立复核（权威交付口径，与搜索器自带口径不是同一个数）**
   `system_sota_longquan.py` 全量 24 序 / 165,223 帧 ⇒ **F1 0.945804 / FAR 0.029348**（TP 123906 / FP 4849 / FN 9351 / GT 133257），
   **逐位复现** r87 的 `eval/shortlist/track_cc2_md16n.json`；`--self-check` ⇒ **[GATE PASS]**；
   对官方 v3 `GT identical 24/24`（`|dF1|>0.1pp` 共 11 条 = **参数效应**，冻结交付组 r87 同口径也是 11 条，不是故障）；
   对重建缓存上的 v3 ⇒ `[CMP GATE PASS]` rc=0。
   ⚠️ 搜索器自带全量臂给的是 0.945673 / 0.029556，**别与权威 track 臂混用**。

4. **交付判定不变（仍是 `match_dist=10.0` 那组）**
   `cc2_md16n` 逐帧 F1 94.5804% / FAR 0.029348（**余量仅 2.2%**）vs 交付组 94.4976% / 0.027859（**余量 7.1%**）；
   跳帧 phys 0.951693 vs 0.951398（+0.000295）。
   **板端 FAR 复核还没做（板卡断电）⇒ 薄余量风险今夜无法排除** ⇒ 用 0.03pp 跳帧 F1 换 1/3 的 FAR 余量不划算
   （r87 `final_selection.json` 已写明同一判断）⇒ **交付参数档 / 评测 / OSD 一个字节都没动**。
   **切换条件**：① 板端 FAR 实测 ≤ 0.03 且余量可接受；② 重跑 track 臂仍 GT 24/24 且 F1 不退；③ 重渲 24 个 OSD（≈ 26 min）。
   候选组三件套**已备好**：参数档 `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_md16_candidate_params.json`
   （37 键，params_md5 `2e41d10be83c1689f7ebb57742a9e3e1`，首行 `status: CANDIDATE`）、
   `eval/track_all_v3_optgapfit_md16.json`、候选 OSD `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_md16_20261120/`
   （pass 21 + un-pass 3 = 24）。🔴 **候选参数档故意不进本包**（§6j①：包内只允许**一份** `load_params` 能静默接受的参数文件）。

5. **本轮新增进包的文件（9 件）**：`rt_search_t0080_v3_r88.json` / `.md` / `.log`（搜索产物 + 真实运行日志）、
   `track_all_v3_optgapfit_md16.json`（候选组权威评测）、`cmp_v3_optgapfit_md16_vs_official_v3.json` /
   `cmp_v3_optgapfit_md16_vs_rebuilt_v3.json`（两份逐序列对账）、`self_check_optgapfit_md16.log`（候选组 GATE PASS 证明）、
   `osd_manifest_md16_candidate.json`（候选 OSD 清单）、`osd_media_check_md16_candidate.json`（候选 OSD 媒体完整性核验：24/24 通过）。**交付 OSD 清单（包根 `osd_manifest.json`）不动**。

6. **闸门判据顺带扩了一条（项数不变）**：§6g 的「md 报告 == 由同名 json 现渲染」那条判据（**循环外单条** check）
   把新批次 `rt_search_t0080_v3_r88` **纳入覆盖** ⇒ r88 的 md 报告同样被证明「不可能与 json 漂移」；
   因为该判据是单条 check，**纳入新批次不增加闸门项数**。
7. **闸门项数本轮不变（仍 190 项）**：本轮**没有新增闸门项**，只修搜索器 + 扩一条既有判据的覆盖面 + 补证据文件 + 补文档；
   交付参数组 / 逐帧 F1 94.4976% / FAR 0.027859 / 跳帧 phys 0.951398 / dense 0.938340 / OSD pass 21 + un-pass 3 全部未变。


8. **补充核验（2026-11-20 07:20~07:40）：un-pass 归因 ＋ 渲染判据 ＋ TSV↔JSON 一致性**
   - **un-pass 与本次参数改动无关（基线固有难点）**：官方 v3 参考渲染 `osd_sys_sota_trial0080_v3_20261022/`
     的 pass/un-pass 划分同为 **21 + 3**，且 un-pass 三条与交付组**完全相同**
     （`VIDEO00006_19700101_000322` / `VIDEO00012_19700101_002251` / `VIDEO00036_19700101_015740`）；
     逐序列 F1（官方 v3 → 交付 `cc2_md10`）**78.821→78.867 / 66.101→60.691 / 14.286→17.120**，
     且这三条在 v3 里本就是 **FAR 最高的三条**（0.4277 / 0.0997 / 0.0300）⇒ **场景本身难，不是 gapfit 参数改坏的**。
     旧 gapfit 对照目录同为此 3 条。
   - **渲染判据 = F1 ≥ 88%**（`manu/videos/render_trial0080_longquan_osd.sh` 头注）⇒ 89.950% 那条计入 pass，
     **别按 90% 口径误判**。
   - **TSV ↔ 评测 JSON 逐字段一致**：交付 / 候选两个 OSD 的 `per_sequence_hyperparams.tsv`（各 24 行）与各自
     `search_json_for_render.json` 的 **f1 / far / tp / fp / fn / gt** 全部吻合（不一致 0 处），
     `group` 列严格等于 88% 判据（两目录皆 pass 21 + un-pass 3）。
   - 本条**只补文档与证据，不动任何交付件**：交付参数档 / 评测 / preds / OSD 均未改。
