# 2026-10-22 · trial_0080 龙泉山交付（单帧 / 带 track / 全量 OSD）

> 顶层速查。完整说明见 `runs/optuna_mix_ratio/trials/trial_0080/lq_eval/DELIVERY_README.md`
> 与 `DELIVERY_MANIFEST.json`（46 条交付物逐条列路径 / md5 / 期望值）。
> 归档沉淀见 `manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md`。

## 结论

| 指标 | 冻结系统级基线 (trial_0474) | **trial_0080 @ v3 交付档** | 判据 |
|:--|--:|--:|:--|
| 单帧 F1 | 87.070% | **92.361%** | — |
| 带 track F1 | 90.605% | **94.747%** | ✅ ≥ 88% |
| 带 track Recall | 85.881% | **92.342%** | ✅ ≥ 85% |
| FAR / frame | 0.0298 | **0.0208** | ✅ ≤ 0.030（裕度 30.6%） |
| FP | 4,919 | **3,440** | — |
| pass 序列 | 17/24 | **21/24** | — |

**交付工作点**：`th_base=0.48 / th_ground=0.26 / sky_ratio=1.0 / confirm_consec=7`
（`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`，
md5 `b20de759599d96aae9493b93ce6b61e8`）。冻结档与 v1/v2 档**原样未动**。
模型：`runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`。

## 去哪找

| 我要 | 去这个路径 |
|:--|:--|
| 评估产物（12 份报告 + 24 缓存 + 5 份预测） | `runs/optuna_mix_ratio/trials/trial_0080/lq_eval/` |
| 单帧臂结果 | `.../lq_eval/single_frame_all.json` |
| **交付档**带 track 结果 | `.../lq_eval/track_all_v3.json` |
| OSD 视频 | `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_20261022/`（推荐交付，21 pass + 3 un-pass） |
| 旧 OSD（**只读，未动**） | `/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020/` |
| 完整清单 | `.../lq_eval/DELIVERY_MANIFEST.json` |

> ⚠️ `trial_0080/gate/epoch3/` 是**训练期源域 gate**（8 序列，official 特征语义），
> **不是**业务域评估，**不可用** —— 详见 `trial_0080/gate/DO_NOT_USE_FOR_LQ_EVAL.md`。

## 一条命令复验全部结论

```bash
python manu/evaluation/verify_delivery_evidence.py --server huangzhe@192.168.99.40
# 期望：PASS 80 / FAIL 0
```
不依赖 numpy、不用 GPU、不做推理；覆盖参数档 md5、评估报告逐字段、三条判据从 raw counts 重算、
缓存 meta 一致性、五档 OSD 计数与字节和、旧 OSD 的 mtime、三侧 md5、入口文件存在性，
以及**本卡片/各交付卡片上引用的数字与路径 vs 实测产物**（卡片若与实测漂移，复验即 FAIL）。

## 三条口径提醒

1. OSD 字节用**纯文件字节和**（`find -name '*.mp4' -printf '%s\n'` 求和），**不要用 `du -sb`**
   —— 后者含目录项 apparent size，服务器原生 fs 与 SSHFS 挂载会差约 10 MB。
2. 旧基线 OSD 实际有 **25 个 mp4**（`preview/` 里 1 个预览片）；"24 个"只指 pass/ + un-pass/。
3. `lq_eval` 里的 JSON 报告最大 462 KB（`robustness_grid.json`）、缓存 pkl 每个约 1 MB ——
   **先看 `DELIVERY_MANIFEST.json`（23 KB）再决定要不要打开大文件**。
