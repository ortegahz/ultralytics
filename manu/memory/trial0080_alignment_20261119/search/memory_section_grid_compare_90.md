### 三十六、🔴 Δn 网格搜索的「指标评估 + 对比」统一汇总 + **267 项独立复算**（第 90 轮，2026-11-20）

**本轮交付数字一个没动**（交付组仍是 `cc2_md10` / 逐帧 F1 94.4976% / FAR 0.027859 / 跳帧 phys 0.951398 / OSD 21+3）；
本轮做的是**把网格搜索的结果收成一份可审计的对比**，并把**指标口径第一次从产物里反解出来、写进文档**。

#### 1) 🔴 环境事故（host 重装系统 ⇒ 三条外部依赖同时断）

| 依赖 | 状态 | 证据 |
| --- | --- | --- |
| 服务器 `192.168.99.40:32222` | **网络可达但认证被拒** | TCP + OpenSSH banner 正常；`Permission denied (publickey)` |
| 板卡 `192.168.0.211:22` | **离线** | `ssh: connect to host 192.168.0.211 port 22: No route to host` |
| host 旧挂载 `/media/manu/1TB-Volume/workspace` | **整个消失** | `/media` 现为空目录、属 nobody；handoff 默认 WORKSPACE 指向该路径 ⇒ 失效 |

⇒ 补充证据：板卡 `ping` **100% 丢包**；服务器侧是**沙箱内新生成的 `~/.ssh/id_ed25519` 未在服务器登记**；
handoff 触发器 `handoff/hook_post_checkout.sh` 默认 WORKSPACE 指向那个已消失的路径 ⇒ 失效。

⇒ **本轮全程不依赖服务器**：所有结论都在**已拉回本地的实测产物**上用新脚本重算，**没有重跑任何一条评测**
（铁律十三：不擅自起重活；服务器不可达 ⇒ 也不能起）。

#### 2) 新增脚本与它补上的洞

- 脚本：**`manu/evaluation/rt_grid_compare.py`**（新文件）。它做四件事：①**独立复算**表 1/表 2 的每个聚合值；②**跨文件对账**
  （短名单 vs 臂 json，逐情景 + 构造性等式）；③**量口径差**（截断 vs 交付，同源臂逐一配对）；④出**对比表**。
- **补上的洞**：此前「网格搜索结果」散在 3 个文件、3 种口径里（臂 json / 短名单 json / 交付口径 track json），汇报时**要人工跨文件拼**，
  且**没人独立复算过聚合值**。
- **口径第一次反解出来并写进脚本 docstring**（此前只散落在各脚本里）：

  | 量 | 定义 |
  | --- | --- |
  | precision | `Σtp / Σ(tp+fp)` |
  | recall | `Σtp / Σgt` |
  | F1 | `2PR/(P+R)` |
  | 跳帧 FAR | `Σfp / Σ(delivered_frames × seeds)` |
  | 交付口径 FAR | `Σfp / 总帧数`（全量 165,223 帧）|

  ⇒ 全部是**序列级 micro-pool**（先把 24 条序列的 tp/fp/gt 加起来再算，不是逐序列 F1 求平均）；
  **逐序列 F1 求平均会低 ~2.7pp**（实测 0.928331 vs 0.955383，臂 `3_jump_optimised/s1`）⇒ 报数必须报 pooled。

#### 3) 验收：**267/267 项通过，0 失败 ✅**（脚本自带独立复算闸门，非人眼核对）

- 覆盖：逐臂逐情景的 precision/recall/F1/FAR 复算、跳帧 mean/worst/mean-FAR 复算、短名单 vs 臂 json 跨文件对账、交付口径逐序列 tp/fp/fn/gt 求和 == overall、
  FAR 分母必须是**整数帧数**（实测 165,223）、以及每个臂的红线判定。
- **诚实边界**：交付口径 FAR 的分母只能由 `Σfp / far_per_frame` 反解（逐序列帧数未落盘），**不是**从视频元数据直接读；
  已用「每条序列反解出的帧数都是整数」+「总数 == 165,223」交叉验证。

#### 4) 对比结果（表 1：交付口径 —— FAR 红线定义在这一档）

| 参数组 | F1 | FAR/帧 | 红线 0.03 | 余量 | TP | FP | FN | GT | 帧数* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `v3` | 94.7533% | 0.020820 | ✅ | 30.6% | 123068 | 3440 | 10189 | 133257 | 165223 |
| `v2` | 94.6231% | 0.025208 | ✅ | 16.0% | 123398 | 4165 | 9859 | 133257 | 165223 |
| `th045` | 94.4134% | 0.028701 | ✅ | 4.3% | 123396 | 4742 | 9861 | 133257 | 165223 |
| `old_gapfit` | 94.3755% | 0.035128 | ❌ | -17.1% | 124251 | 5804 | 9006 | 133257 | 165223 |
| `cc2_md10` | 94.4976% | 0.027859 | ✅ | 7.1% | 123480 | 4603 | 9777 | 133257 | 165223 |
| `cc2_md16n` | 94.5804% | 0.029348 | ✅ | 2.2% | 123906 | 4849 | 9351 | 133257 | 165223 |
| `cc2_md8` | 94.4842% | 0.026455 | ✅ | 11.8% | 123239 | 4371 | 10018 | 133257 | 165223 |
| `cc2_md12` | 94.5166% | 0.028440 | ✅ | 5.2% | 123613 | 4699 | 9644 | 133257 | 165223 |
| `cc3_md10` | 94.5097% | 0.026116 | ✅ | 12.9% | 123252 | 4315 | 10005 | 133257 | 165223 |
| `cc3_md16` | 94.5939% | 0.028029 | ✅ | 6.6% | 123744 | 4631 | 9513 | 133257 | 165223 |
| `cc4_md10` | 94.5339% | 0.025112 | ✅ | 16.3% | 123163 | 4149 | 10094 | 133257 | 165223 |
| `cc5_md10` | 94.5462% | 0.024440 | ✅ | 18.5% | 123094 | 4038 | 10163 | 133257 | 165223 |
| `cc6_md10` | 94.5556% | 0.023744 | ✅ | 20.9% | 123014 | 3923 | 10243 | 133257 | 165223 |
| `cc7_md10` | 94.5804% | 0.022872 | ✅ | 23.8% | 122946 | 3779 | 10311 | 133257 | 165223 |
| `old_gapfit(短名单复测)` | 94.3755% | 0.035128 | ❌ | -17.1% | 124251 | 5804 | 9006 | 133257 | 165223 |
| `rerun_selected` | 94.3749% | 0.035152 | ❌ | -17.2% | 124253 | 5808 | 9004 | 133257 | 165223 |

\* 帧数**不是**从视频元数据读的（那在服务器）：由 `Σfp / far_per_frame` 反解，已校验为整数、且所有臂反解出**同一个**值
⇒ 可当「全量 165,223 帧」用。

#### 5) 对比结果（表 2：跳帧网格，物理帧基）

| 臂 | s1 | s2 | s3 | s4 | mean F1 | worst F1 | mean FAR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `2a_jump_dense_timebase` | 0.9443 | 0.9124 | 0.8746 | 0.9569 | 0.922060 | 0.874592 | 0.013286 |
| `2b_jump_physical_untuned` | 0.0000 | 0.0000 | 0.0000 | 0.3365 | 0.084122 | 0.000000 | 0.000025 |
| `3_jump_optimised` | 0.9554 | 0.9530 | 0.9389 | 0.9583 | 0.951398 | 0.938916 | 0.016506 |

情景语义（来自臂 json 的 `meta.scenarios_spec`，**不是**本脚本编的）与实测节奏（来自臂 json 的 `cadence`，即中间量）：
- `s1` = measured 1x (board idle, ~8.0 fps)｜kind=`measured`｜mean Δn=`3.3597`｜fps=`7.4475`
- `s2` = measured 1.5x + jitter (shared SoC, ~5.4 fps)｜kind=`measured`｜mean Δn=`5.2317`｜fps=`4.7848`
- `s3` = measured 2x + jitter (heavy load, ~4.0 fps)｜kind=`measured`｜mean Δn=`7.0035`｜fps=`3.5759`
- `s4` = brief assumption (2 workers, ~13.9 fps)｜kind=`doc`｜mean Δn=`1.9507`｜fps=`12.822`

⇒ `2b`（v3 基座 · 未调参）在 s1~s3 上 F1 **精确为 0**（s4 假设情景 0.3365，均值 0.084122）；`3`（cc2_md10）均值 0.951398、最差情景 0.938916（s3）。

**归因（假设，非因果证明）**：v3 的 `confirm_consec=7` 要求连续 7 帧确认，而 s1~s3 的 mean Δn 分别为 3.36 / 5.23 / 7.00 —— 帧间隔已接近或超过该门限，
跟踪器因此无法建立确认 ⇒ **这是「帧不再连续」的结构性后果，不是检测器变差**。
（依据：上表 cadence 中间量 kind=`measured`、mean Δn=3.3597/5.2317/7.0035、fps=7.4475/4.7848/3.5759）

#### 6) 🔴 表 2b：**截断口径系统性低报 FAR**（第 88 轮那个缺陷的量化闭环）

| 短名单臂 | 交付口径臂 | 检测侧同源 | FAR（截断 4000 帧/序） | FAR（交付 165,223 帧） | 偏移 | 截断判定 | 交付判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v3_base` | `v3` | ✅ | 0.017558 | 0.020820 | +0.003262 | PASS | PASS |
| `v3_base` | `v2` | ⚠️ | 0.017558 | 0.025208 | +0.007650 | PASS | PASS |
| `old_gapfit` | `old_gapfit` | ✅ | 0.029013 | 0.035128 | +0.006115 | PASS | FAIL |
| `delivered_cc2_md10` | `cc2_md10` | ✅ | 0.022399 | 0.027859 | +0.005461 | PASS | PASS |
| `cc2_md16n` | `cc2_md16n` | ✅ | 0.024130 | 0.029348 | +0.005218 | PASS | PASS |
| `cc2_md8` | `cc2_md8` | ✅ | 0.021632 | 0.026455 | +0.004823 | PASS | PASS |
| `cc2_md12` | `cc2_md12` | ✅ | 0.023364 | 0.028440 | +0.005077 | PASS | PASS |
| `cc3_md10` | `cc3_md10` | ✅ | 0.020937 | 0.026116 | +0.005180 | PASS | PASS |
| `cc3_md16` | `cc3_md16` | ✅ | 0.023080 | 0.028029 | +0.004949 | PASS | PASS |
| `cc4_md10` | `cc4_md10` | ✅ | 0.020227 | 0.025112 | +0.004885 | PASS | PASS |
| `cc5_md10` | `cc5_md10` | ✅ | 0.019673 | 0.024440 | +0.004766 | PASS | PASS |
| `cc6_md10` | `cc6_md10` | ✅ | 0.019318 | 0.023744 | +0.004425 | PASS | PASS |
| `cc7_md10` | `cc7_md10` | ✅ | 0.018864 | 0.022872 | +0.004008 | PASS | PASS |
| `rerun_selected` | `rerun_selected` | ✅ | 0.029013 | 0.035152 | +0.006139 | PASS | FAIL |

⇒ **同源配对 13 例**，偏移区间 **[+0.003262, +0.006139]，全部为正** ⇒ 截断口径在本批上**一律低于交付口径**；
**2 例两口径判定相反**（`old_gapfit`、`rerun_selected`：截断 PASS / 交付 FAIL）⇒ 只看截断口径会选出破线参数 ⇒ 硬约束必须量在**交付口径**上。

ℹ️ **区间怎么读**：上面的区间**只统计 `检测侧同源` = ✅ 的 13 例**；表体里 `v2` 那行是**异参行**
（偏移 +0.007650），**已剔除**、**不参与**区间统计 ⇒ 按表自算时不要把它算进来。

⚠️ **异参行 1 例**（检测侧与搜索基座不同，**不参与上面的偏移统计**）：`v2` —— 它的偏移里混着**检测阈值变更**，不能当口径效应引用。

ℹ️ **表 2b 里没有出现的交付臂**：`th045` —— 它在 14 臂短名单网格里**没有同参数签名的对应臂**（签名 = 7 个跟踪旋钮的取值，见表 1）⇒ **没有截断口径读数可配**，故不入表 2b；
**这不是漏测**，是它在短名单网格之外。

⚠️ **偏移的来源要说准**：两个口径的**指标定义完全相同**，差的是**帧宇宙**（截断 = 每序列前 4000 帧；
交付 = 全量 165,223 帧）⇒ 偏移是**帧宇宙不同**带来的子集效应，**不是**口径定义不同、也不是换算错误；长尾帧的 FAR 是否更高，本轮未单独测量，不作断言。

⚠️ **配对口径的诚实边界**：配对只按 **7 个跟踪旋钮**匹配：
`confirm_consec/confirm_gap_tol/max_age_phys/r_slope/alpha_decay/match_dist/min_hits`
`v2` 与 `v3` 的这 7 键**相同**（差异在检测阈值 `th_base/th_salvage/th_ground/sky_ratio`），故两行都挂在 `v3_base` 短名单臂上
—— 这是**真实的口径事实**（同一跟踪旋钮、检测阈值不同）；产物 md 已用 `检测侧同源` 列把它标成 ⚠️ 异参行并剔出偏移统计。

#### 7) 短名单 14 臂（网格本体）

| 臂 | 参数 | 跳帧 mean F1 | 跳帧 worst F1 | 逐帧(截断) F1 | 逐帧(截断) FAR |
| --- | --- | --- | --- | --- | --- |
| `rerun_selected` | cc=2, gt=8, ma=10, rs=5.0, ad=1.0, md=16.0 | 0.952186 | 0.941208 | 0.957343 | 0.029013 |
| `cc2_md16n` | cc=2, gt=8, ma=None, rs=5.0, ad=1.0, md=16.0 | 0.951693 | 0.939229 | 0.958628 | 0.024130 |
| `cc2_md12` | cc=2, gt=8, ma=None, rs=5.0, ad=1.0, md=12.0 | 0.951582 | 0.939025 | 0.958416 | 0.023364 |
| `delivered_cc2_md10` | cc=2, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.951398 | 0.938916 | 0.958640 | 0.022399 |
| `cc2_md10` | cc=2, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.951398 | 0.938916 | 0.958640 | 0.022399 |
| `cc2_md8` | cc=2, gt=8, ma=None, rs=5.0, ad=1.0, md=8.0 | 0.951215 | 0.938840 | 0.958584 | 0.021632 |
| `old_gapfit` | cc=2, gt=6, ma=10, rs=3.5, ad=1.0, md=16.0 | 0.950655 | 0.936110 | 0.957343 | 0.029013 |
| `cc3_md16` | cc=3, gt=8, ma=None, rs=5.0, ad=1.0, md=16.0 | 0.944615 | 0.915707 | 0.958717 | 0.023080 |
| `cc3_md10` | cc=3, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.943935 | 0.915239 | 0.958848 | 0.020937 |
| `cc4_md10` | cc=4, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.935804 | 0.888383 | 0.958954 | 0.020227 |
| `cc5_md10` | cc=5, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.927832 | 0.862536 | 0.959050 | 0.019673 |
| `cc6_md10` | cc=6, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.915936 | 0.821642 | 0.959094 | 0.019318 |
| `cc7_md10` | cc=7, gt=8, ma=None, rs=5.0, ad=1.0, md=10.0 | 0.905213 | 0.786370 | 0.959125 | 0.018864 |
| `v3_base` | cc=7, gt=0, ma=None, rs=None, ad=None, md=16.0 | 0.084122 | 0.000000 | 0.960186 | 0.017558 |

⚠️ **这张表的「逐帧(截断) FAR」列不能用来判红线**（见表 2b）。网格上跳帧分最高的臂在**交付口径**上会破线（见表 1）。**红线判定只看表 1（交付口径）。**
（产物的表 3 还有一列 `跳帧 mean FAR`，此处为控行长未抄，数值见 `grid_compare/rt_grid_compare.md`。）

🔴 **活标本**：`rerun_selected`（`confirm_consec 2 / gap_tol 8 / max_age_phys 10 / r_slope 5.0 / match_dist 16.0`）
在**跳帧网格上分数最高**（mean F1 0.952186），但它的**交付口径 FAR = 0.035152 ⇒ 破线**
（本轮已在表 1 用 `eval/shortlist/track_rerun_selected.json` 实测到，
不再是「服务器实测」转述）⇒ 「只看跳帧臂就会选出破线参数」这件事，现在有**同表两列**的对照证据。

#### 8) 产物 · 复现命令 · 执行机

- 脚本：`manu/evaluation/rt_grid_compare.py`（本轮新增）
- 产物：`manu/memory/delivery_20261119/grid_compare/`（`rt_grid_compare.md` / `rt_grid_scenario.csv` /
  `rt_grid_shortlist.csv` / `rt_delivery_compare.csv` / `rt_far_calibration.csv`）
- 对比正文：`manu/memory/delivery_20261119/COMPARISON.md`（既有）+ 本轮 `rt_grid_compare.md`（新增，带复算验收表）
- **执行机：本机（host）**，纯读本地 json，无 GPU、无网络、无重活。

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
A=manu/memory/delivery_20261119/artifacts
E=manu/memory/trial0080_alignment_20261119/eval
python3 manu/evaluation/rt_grid_compare.py \
  --arm-json $A/rt_eval_delivered_armA.json --shortlist $A/shortlist_deterministic.json \
  --track "v3=manu/memory/trial0080_alignment_20261119/eval/track_all_v3.json" \
  --track "v2=manu/memory/trial0080_alignment_20261119/eval/track_all_v2.json" \
  --track "th045=manu/memory/trial0080_alignment_20261119/eval/track_all_th045.json" \
  --track "old_gapfit=manu/memory/trial0080_alignment_20261119/eval/track_all_v3_gapfit.json" \
  --track "cc2_md10=manu/memory/delivery_20261119/artifacts/track_all_v3_optgapfit.json" \
  --track "cc2_md16n=manu/memory/delivery_20261119/artifacts/track_all_v3_optgapfit_md16.json" \
  --track "cc2_md8=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc2_md8.json" \
  --track "cc2_md12=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc2_md12.json" \
  --track "cc3_md10=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc3_md10.json" \
  --track "cc3_md16=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc3_md16.json" \
  --track "cc4_md10=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc4_md10.json" \
  --track "cc5_md10=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc5_md10.json" \
  --track "cc6_md10=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc6_md10.json" \
  --track "cc7_md10=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_cc7_md10.json" \
  --track "old_gapfit(短名单复测)=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_old_gapfit.json" \
  --track "rerun_selected=manu/memory/trial0080_alignment_20261119/eval/shortlist/track_rerun_selected.json" \
  --out-dir manu/memory/delivery_20261119/grid_compare
```

#### 9) 本轮**没做**的事（外部依赖断，不是遗漏）

1. **板端 FAR 复核**（`cc2_md16n` 余量仅 2.2% 的风险）⇒ 板卡离线，**做不了**；板卡上电后按 `manu/evaluation/eval_board_fused_tsv.py` 跑（脚本已就位）。
2. **服务器侧重跑/扩样** ⇒ 沙箱内无公钥认证，**登不上去**；需要用户在本机终端执行一次`scp`+`ssh` 授权后才有。
3. **帧率扫描更细的网格**（如 25/20/17/14/12 fps 全扫描）⇒ 依赖服务器缓存，**本轮做不了**。

#### 10) 数字来源（本节防漂移约定）

- 表 1 / 表 2 / 表 2b / 表 3 的**每一个单元格**都逐字取自产物 `manu/memory/delivery_20261119/grid_compare/rt_grid_compare.md`（唯一数字来源）；
  结论数字（同源 13 例 / 偏移区间 / 2 例判定相反）由表体**重算**后再与产物正文**逐位对账**。
- 生成器 `search/make_memory_section_grid_compare.py` 带**自检**：每个单元格逐字命中产物 md、每个数字都有声明来源、
  行长 ≤ 120、结论与产物逐位一致 ⇒ 数字对不上就**不写文件、直接 FAIL**（自检计数见脚本 stdout）。

