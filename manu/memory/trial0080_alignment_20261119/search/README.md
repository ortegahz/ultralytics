# artifacts 清单（trial_0080 + v3 · Δn 重搜）

| 文件 | 是什么 | 注意 |
| --- | --- | --- |
| `rt_search_t0080_v3_seedfix.json` | **规范搜索产物**（run3，cadence 采样用 `zlib.crc32` 确定性种子） | 只有它可复现 |
| `rt_search_t0080_v3.json` | run1 历史留档 | 🔴 **作废**：`random.Random(hash(str))` ⇒ CPython `hash(str)` 每进程随机 ⇒ cadence 不可复现 |
| `rt_search_t0080_v3_final.json` | run2 历史留档 | 🔴 同样用 `hash()` ⇒ 不可复现 |
| `opt_params_t0080_v3_seedfix_flat.json` | run3 的 objective-best 扁平参数 | 🔴 **它不是交付参数**：见下。🔴 **第 85 轮起，这两份已从「交付包」删除**（server 侧保留仅为留档）：它们装的是**已被全量闸门拒绝**的那组（`match_dist 16.0` / `max_age_phys 10` ⇒ 全量口径 FAR **0.035152 > 0.03 红线**），且是嵌套 `{"params": {...}}` 形态 ⇒ `system_sota_longquan.py::load_params`（`payload.get("params", payload)`）**静默接受**，喂错不报错 ⇒ **不得**随包交付、**不得**上板 |
| `opt_params_t0080_v3_final_flat.json` | run2 的 objective-best 扁平参数 | 同上（与上一行内容逐字节相同） |
| `rt_eval_t0080_v3_opt.json` | 三臂复核（base = **交付参数**） | 臂 1/2b/3 数值相同 **不是 bug**（base 已是交付组）；🔴 **它的跳帧数字（0.9521/0.9374）产于采样器修复前 ⇒ 不可复现，已被下面两份重测产物取代** |
| `rt_eval_delivered_armA.json` | 🔴 **跳帧三臂的规范产物**（`--phase eval`，base=v3 / opt=交付参数，确定性采样器） | 逐帧天花板 0.960186 ｜ v3 未调 0.084122 ｜ 交付参数 0.951398 |
| `shortlist_deterministic.json` | 🔴 **14 组候选的规范重测**（一个进程内测完，只 load 一次缓存，1562.8 s） | 跳帧列 / 逐帧截断列全部出自这里；与上一行**逐格交叉一致** |
| `remeasure_shortlist_deterministic.py` | 上面那份重测的脚本 | 可复现重测的唯一入口 |
| `final_selection.json` | **两阶段选型证据**：搜索 objective-best → 全量口径闸门 → 交付参数 | 🔴 **交付参数来自这里的全量闸门，不是 objective-best**；`_deterministic_remeasure_20261120` 键记录了第二轮口径更正与选型措辞修正 |
| `track_all_v3_optgapfit.json` | 交付参数的**全量口径**评测（24 序 / 165,223 帧） | **交付判据就看这个** |
| `track_all_v3_optgapfit_repro.json` | 🔴 **可复现性闭环**：用交付参数档**原样重跑** track 臂的结果 | 与上一行 **320 键中仅 `elapsed_seconds` 不同**，`overall` + 24 条 per_sequence **逐位一致** |
| `repro_preds_md5.txt` | 🔴 上面那次重跑的**逐文件 md5 对照**（25/25 相同）+ JSON 结构 diff | 证明预测输出逐位可复现 |
| `repro_track_arm.log` | 上面那次重跑的**真实 stdout/stderr + `/usr/bin/time -v`**（RC=0，wall 3:27.52，峰值 RSS 378 MB） | |
| `track_all_v3.json` | v3 原参在同一份重建缓存上的全量评测 | 纯参数效应对照 |
| `cmp_v3_optgapfit_vs_official_v3.json` | 交付组 vs 官方 v3 逐序列对账 | 🔴 `max-df1-pp 0.1` 闸门 **rc=1（预期）**：换了参数组，逐序列 F1 必然变；GT 仍 24/24 一致 |
| `cmp_v3_optgapfit_vs_rebuilt_v3.json` | 交付组 vs 重建缓存上的 v3 | **rc=0**（GT 24/24 一致） |
| `cmp_v3_rebuilt_vs_official_v3.json` | 同参 v3 的**可复现性**对照 | **rc=0** ⇒ 缓存与流水线没坏 |
| `memory_section_257_18.md` | 本条线的 memory 章节正文·**主档版**（与 6 份主档里的 §257 十八逐字节一致） | 🔴 含「二十二」小节（主机回填路径） |
| `memory_section_257_18_compact.md` | 同上·**精简档版**（与 6 份精简档一致） | 精简档正文与主档**不同**（压缩版），别串档位 |
| `trial0080_delta_search_spec_20261119.md` | 搜索规格书（§8.1~§8.8 八条方法论 + §9.1~§9.6 收口） | §8.6 = arm5 口径澄清；§9.6 = 可复现性闭环 |
| `trial0080_board_fusion_preset_20261119.md` | 板卡侧 Δn 默认值同步施工单（**板卡断电，未做**） | |
| `memory_sync_incident_20261119.md` | 本夜 memory 并发同步事故与修复留痕 | 自造事故，必读 |
| `memory_md5_manifest.txt` | 6 份 memory + 两份正文的当前 md5 | **改档后需重生成**；`HOST_BACKFILL.sh` / `HOST_PUSH.sh` 都靠它复核 |
| `HOST_PUSH_ONCE.sh` | **一次性幂等推送**（服务器侧每次值守时调一次）：host 可达即推送+复核 | 0=成功 / 3=host 仍关机 / 1=失败 |
| `HOST_PUSH_WATCH.sh` | 常驻守望脚本（长驻 screen/tmux 环境下可用） | 🔴 本容器无可用守护机制，**不指望它常驻** |
| `merge_memory_section.py` | 章节级合并工具：只改写「§257 十八」块，**块外字节逐字节保留** | 两条回填路共用 |
| `verify_all.py` | 端到端核验脚本 | 当前 **ALL CHECKS PASSED / RC=0**（**136 项 PASS / 0 FAIL**，含主机回填链路与可复现性闭环校验） |

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
