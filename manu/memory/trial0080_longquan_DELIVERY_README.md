# trial_0080 · 龙泉山交付说明（DELIVERY_README）

本目录是 **trial_0080**（领域微调 best model，权重 `weights/epoch3.pt`）在龙泉山数据集上的
**全部评估产物**。机器可读的完整清单见同目录 `DELIVERY_MANIFEST.json`（每条交付物的路径、md5、
关键事实与期望值）。一条命令即可复验全部结论：

```bash
python manu/evaluation/verify_delivery_evidence.py --server huangzhe@192.168.99.40
# 期望输出：PASS 80 / FAIL 0
```

## 交付结论

| 指标 | 冻结系统级基线 (trial_0474) | **trial_0080 @ v3 交付档** | 判据 |
|:--|--:|--:|:--|
| 单帧 F1 | 87.070% | **92.361%** | — |
| 带 track F1 | 90.605% | **94.747%** | ✅ ≥ 88% |
| 带 track Recall | 85.881% | **92.342%** | ✅ ≥ 85% |
| FAR / frame | 0.0298 | **0.0208** | ✅ ≤ 0.030（裕度 30.6%） |
| FP | 4,919 | **3,440** | — |
| pass 序列 | 17/24 | **21/24** | — |

**v3 交付工作点**：`th_base=0.48 / th_ground=0.26 / sky_ratio=1.0 / confirm_consec=7`
（参数文件 `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`，
md5 `b20de759599d96aae9493b93ce6b61e8`；
冻结档与 v1/v2 档原样保留未动。）

## 本目录文件对照

| 文件 | 是什么 |
|:--|:--|
| `single_frame_all.json` | **单帧臂**报告（冻结口径 th=0.22 / radius=768 / inbox）→ F1 92.361% |
| `track_all.json` | 带 track 臂 @ **冻结工作点** th_base=0.30（F1 94.007%，但 FAR 0.0405 破线） |
| `track_all_th045.json` | 带 track 臂 @ **v1** th_base=0.45（F1 94.391%，FAR 0.0290） |
| `track_all_v2.json` | 带 track 臂 @ **v2** 0.42/0.26/7（F1 94.618%，FAR 0.0252） |
| `track_all_v3.json` | 带 track 臂 @ **v3 交付档** 0.48/0.26/7 + sky_ratio=1.0（F1 94.747%，FAR 0.0208） |
| `far_tradeoff_probe.json` | 单轴 `th_base` 扫描（5 候选），v1 的标定依据 |
| `robustness_grid.json` | 三轴网格 `th_base × confirm_consec × th_ground`（54 候选），v2 的依据 |
| `sky_ratio_grid.json` | 第四轴网格 `sky_ratio × th_base`（28 候选，**28/28 三条全过**），v3 的依据 |
| `baseline_recheck.json` | **冻结基线**用同一探针代码路径重跑（65.1 s，逐位复现 FP 4,919 / FAR 0.02977） |
| `loo_check.json` / `loo_check_v3.json` | **留一序列交叉验证**（v2 / v3）：23/24 选出同一工作点 ⇒ 非过拟合 |
| `ms_cache/*.pkl` | 24 序列多尺度热图缓存（特征源 = FPV `longquanshan_ir_gmc_median`，**直接读盘未重算**） |
| `preds_single/` | 单帧臂逐序列预测（OSD 输入） |
| `preds_track/` `preds_track_th045/` `preds_track_v2/` `preds_track_v3/` | 四档带 track 逐序列预测（OSD 输入，含 `track_id`） |
| `pipeline.log` `track_all*.log` | 流水线与各臂运行日志 |

## OSD 视频（在数据卷，不在本目录）

服务器路径 `/mnt/data/siping/datasets/manu/龙泉山/`，本地 SSHFS 视图 `/home/manu/mnt/data/siping/datasets/manu/龙泉山/`：

| 目录 | 工作点 | pass / un-pass |
|:--|:--|:--|
| `osd_sys_sota_20261020/` | 旧基线 trial_0474 | 17 / 7（**只读，全程未动**） |
| `osd_sys_sota_trial0080_20261022/` | th_base=0.30 | 21 / 3（FAR 破线，不可交付） |
| `osd_sys_sota_trial0080_th045_20261022/` | th_base=0.45（v1） | 20 / 4 |
| `osd_sys_sota_trial0080_v2_20261022/` | 0.42/0.26/7（v2） | 21 / 3 |
| **`osd_sys_sota_trial0080_v3_20261022/`** | **0.48/0.26/7 + sky_ratio=1.0（v3，推荐交付）** | **21 / 3** |

每个目录均为 `pass/` + `un-pass/` 两个子目录，24 个视频全部 ffprobe 可读、无零字节。
渲染器：`manu/videos/render_sys_sota_longquan.py`（红=系统检出 / 绿=命中 GT / 橙=漏掉 GT，
标签格式 `T<track_id>:<score>` ⇒ **带跟踪**）。

## 两条口径提醒

1. OSD 字节数用**纯文件字节和**（`find -name '*.mp4' -printf '%s\n'` 求和），
   **不要用 `du -sb`** —— 后者含目录项 apparent size，服务器原生 fs 与 SSHFS 挂载结果会差约 10 MB。
2. 旧基线目录除 24 个交付视频外，`preview/` 子目录里还有 1 个预览片，
   **"24 个"只指 pass/ + un-pass/**。

## 详细沉淀

`manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md`
（§五十四 口径取证 / §55.1–55.5 结果 / §55.6–55.8 交付档标定 / §55.9 归因修正 /
§55.10 LOO 验证 / §55.11 v3 / §55.12 证据索引 / §55.13 遗留项）。
