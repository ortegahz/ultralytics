# ⚠️ 业务域（龙泉山）评估产物在这里：`lq_eval/`

本 trial 目录里有**两个**容易混淆的评估产物目录，二者**不可互换**：

| 目录 | 是什么 | 序列数 | 能否用于龙泉山业务域评估 |
|:--|:--|--:|:--|
| **`lq_eval/`** | ✅ **龙泉山业务域全量评估产物**（本轮交付） | **24** | ✅ **这就是要找的目录** |
| `gate/epoch3/` | ❌ 训练期**源域 gate** 评估产物 | 8 | ❌ **不可用**，见下方原因 |

## 为什么 `gate/epoch3/` 不能用

`gate/epoch3/ms_cache/` 的 8 个缓存是训练期 gate 用的，它的特征源是
`longquanshan_ft_gmc_median_official`（official 逐 lag 直接拟合），
而冻结基线与本轮交付用的都是 `longquanshan_ir_gmc_median`（**FPV 仿射连乘**）——
两套语义**逐像素差 26~70/255**（非边界帧 official 字节命中 0/3，FPV 3/3）。

⇒ 混用会让"全量 F1"变成两个分布的加权平均，**无法做差值归因**。
判定脚本：`manu/data/verify_longquan_feature_source.py`；
取证过程见 `manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md` §54.2。

## 交付产物在哪

**入口就在 `lq_eval/` 目录里**：

- `lq_eval/DELIVERY_README.md` —— 人读：本目录每个文件是什么、四项交付结论、OSD 五档位置、两条口径提醒
- `lq_eval/DELIVERY_MANIFEST.json` —— 机读：39 条交付物的路径 / md5 / 关键事实 / 期望值

**一条命令复验全部结论**（不依赖 numpy、不用 GPU、不做推理）：

```bash
python manu/evaluation/verify_delivery_evidence.py --server huangzhe@192.168.99.40
# 期望输出：PASS 80 / FAIL 0
```

## 四项交付结论速查

| 指标 | 冻结系统级基线 (trial_0474) | **trial_0080 @ v3 交付档** | 判据 |
|:--|--:|--:|:--|
| 单帧 F1 | 87.070% | **92.361%** | — |
| 带 track F1 | 90.605% | **94.747%** | ✅ ≥ 88% |
| 带 track Recall | 85.881% | **92.342%** | ✅ ≥ 85% |
| FAR / frame | 0.0298 | **0.0208** | ✅ ≤ 0.030（裕度 30.6%） |
| FP | 4,919 | **3,440** | — |
| pass 序列 | 17/24 | **21/24** | — |

交付工作点：`th_base=0.48 / th_ground=0.26 / sky_ratio=1.0 / confirm_consec=7`
（`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`，
md5 `b20de759599d96aae9493b93ce6b61e8`）。

模型权重：`weights/epoch3.pt`（= `trial_result.json` 里的 `weights` / `last_weights`，
`gate_epoch=3`；与 `train_log.csv` 末行 `epoch=3,END` 三方交叉验证一致）。

OSD 视频在**数据卷**，不在本目录：
`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_20261022/`（推荐交付，21 pass + 3 un-pass）。
旧基线 OSD `osd_sys_sota_20261020/` **只读、全程未动**。
