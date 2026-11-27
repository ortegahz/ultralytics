# W10 · 🔴 交付物实质缺陷（host 侧第三方审计 + **服务器最新产物实跑复核**）

> 投递：2026-11-26 18:05 CST（host 侧，18:00 计划关机前最后一条）
> 数据来源：**服务器 17:49 重算后的 `output/`**（`constraint_lines_v2.csv` / `knee_thresholds_v2.csv` / `correlations_v2.csv` / `regression_v2.csv`），host 侧用服务器解释器**实跑复核**（非引用旧快照）。
> ⚠️ 与 W6~W9 不重复：W6~W9 讲口径/守卫/交棒；**本条讲交付数字本身对不上**。
> 🔴 **host 关机后没有独立复核通道了** ⇒ 这几条请你**自己复核并写进报告**，别指望 host 回来兜底。

## 一、🔴 三项必须处理（按优先级）

### 1. `local_scr` 红线把 **98% 的真实好片**判为坏（实测）

| 集合 | `rate_below_line` | n |
|:--|--:|--:|
| good_holdout | **0.2579** | 380 |
| real_good | **0.9800** | 300 |
| real_bad（拦截率） | 0.9960 | 248 |

⇒ `local_scr` 线位 **5.82855** 高于真实清晰片的实际分布 ⇒ 作为「正式红线」**不成立**。
（对照：`target_lap_var` 误杀 good_holdout 0.0132 / real_good 0.0000 —— 这个指标的红线是好的。）

**要求**：**要么**把 `local_scr` 红线降级为「参考线 / 仅同域使用」并写明依据，**要么**按「同域留出集」重标；
报告 §10 必须给出「三集合 × 三指标」这张误杀率表，**不能只报拦截率**（只报 99.6% 拦截率而不报 98% 误杀，是选择性呈现）。

### 2. 跨臂互证**只剩 xcorr 一臂**（`cross_arm_dev_pct = nan`、`kcf_legacy_line = nan`）

| 指标 | line | xcorr | kcf | csrt |
|:--|--:|--:|--:|--:|
| target_local_contrast | 0.18917 | 0.18917 | **nan** | 0.05359 |
| target_lap_var | 541.09 | 541.09 | **nan** | 1110.81 |
| local_scr | 5.82855 | 5.82855 | **nan** | 2.25680 |

⇒ 任务书 L60 要求的「**XCorr 与 KCF 已证明膝点的算法无关性**」**在当前产物上不再成立**
（KCF 无线位；csrt 与 xcorr 差 2.0~2.6 倍）。

**要求**：报告 §12 明确写「**本轮算法无关性不成立**」，并写清 KCF 臂为何无线位：
是曲线塌陷？被本轮 W8 分类口径改动清空？还是被「y_span 守卫」过滤掉？
**若是本轮分类改动误清的，必须修**（KCF 是任务书指定的第二条腿，不能只剩一条）。

### 3. 红线与任务书常量漂移（任务书 L30/L35/L91-94 vs 交付）

| 指标 | 任务书 | 交付 | 漂移 |
|:--|--:|--:|--:|
| target_lap_var | 819.55 | 541.0926 | **−34.0%** |
| local_scr | 2.69 | 5.8285 | **+116.7%** |

⚠️ **注意**：与 17:39 快照比，`local_scr` 线位从 **2.2568 → 5.8285**（变了 2.6 倍、方向也反了）。
⇒ 报告必须写清「**换了什么底座 / 什么口径导致**」，否则复现者会认为数字是随手改的。

**要求**：按任务书 G6 口径 —— **变了就要写清楚变了多少、为什么**；
免责声明里的门槛必须是**本轮实测值**（当前审计 R6c 已报 FAIL：`541.093` 与 `5.82855` 未出现在免责声明里）。

## 二、🟠 一项需要解释的矛盾：被采纳的膝点，8 成在相关性表里是「弱区分度」

`knee_thresholds_v2.csv` 里 `knee_reliable=1` 共 **37 行**，其中 **30 行**在 `correlations_v2.csv` 里被判
`usable=0` 或 `direction_ok=0`（例：`xcorr / blur / target_lap_var / own_good`）。

⇒ **被采纳成正式红线的膝点，约 8 成在相关性表里方向不对或区分度不足**。

**要求**：**要么**在采纳膝点时**同时要求 `usable=1 ∧ direction_ok=1`**（推荐，顺带解决 §一.1 的误杀问题），
**要么**在报告里逐条说明「为何在相关性弱的情况下仍采纳该膝点」。
这两条口径**必须对齐**，不能各说各话（现在的状态是：相关性表判「不可用」，膝点表判「可靠」）。

## 三、🟢 验收仪器的已知盲区（**这四类必须由你在报告里自证**）

新版 `audit_final.py`（562 行，已装到服务器 `iqa_v2/`）已修：按表头识别产物 / 独立复算膝点 / 退化守卫（y_span）。
**但它不查**：① 任务书常量对账（本条 §一.3）② 跨臂偏差阈值（本条 §一.2）③ 误杀率门限（本条 §一.1）
④ `correlations_v2.csv` 交叉（本条 §二）⑤ `gates/pytest.txt` / `md5.txt` / `verify.txt` 的读取。

⇒ **把服务器 pytest 里已有的「条款→测试名」对应测试，当作条款级证据**，
在报告附录逐条点名（这些测试名可直接 grep 核对）：

```
test_sequence_level_split_is_disjoint_and_reproducible
test_simulation_base_is_pure_and_holdout_never_simulated
test_conditions_cover_three_degradation_families
test_constraint_lines_only_cover_three_core_metrics
test_jpeg_and_white_noise_axes_are_removed
test_regression_covers_three_way_sets
test_small_target_threshold_constant
test_motion_angle_is_deterministic_and_in_range
test_good_segment_threshold_is_task_book_3px
test_boundary_safety_no_nan_and_no_overlap
test_output_manifest_md5_all_ok
```

## 四、host 侧收尾状态（供你知情）

| 项 | 状态 |
|:--|:--|
| 服务器主会话 | **在跑**，host 关机后继续存活（`setsid+nohup`） |
| `iqa_v2/audit_final.py`（新版 562 行） | **已在服务器**，自检：`/home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/audit_final.py .` |
| `iqa_v2/MEMORY_SECTION.md` | 标题**必须**写成 `## §318 · …`（host 侧合并脚本会自动重编号为 §319；**无 `§` 的标题会落不进记忆库索引**） |
| 记忆库合并 | **在 host 做**（host 的 `manu/memory/` 是权威副本）⇒ 服务器**不要**改服务器上的 `manu/memory/memory.md` |
| host 的 `merge_memory.sh` / `collect_from_server.sh` | host 侧正在修独立审计查出的 13 处缺陷（假绿 / 静默吞正文 / 过期 manifest / 目录丢失），**不影响你的评测与报告** |

## 五、自证命令（服务器上直接跑）

```bash
cd /tmp/pycharm_project_a7d966e6
/home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/audit_final.py .        # 要求 FAIL=0
/home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/make_manifest_v2.py    # 重生成清单（现清单已过期）
/home/huangzhe/anaconda3/envs/uav/bin/python -m pytest iqa_v2/test_iqa_v2.py -v
```
