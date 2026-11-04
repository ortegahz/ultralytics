# ⚠️ 本目录不是龙泉山业务域评估产物

这里是 **trial_0080 训练期的源域 gate 评估产物**（`gate/epoch3/`），
**不是**龙泉山业务域（全量 24 序列）的评估结果。

**龙泉山业务域交付产物在**：`runs/optuna_mix_ratio/trials/trial_0080/lq_eval/`
→ 入口文件 `lq_eval/DELIVERY_README.md` 与 `lq_eval/DELIVERY_MANIFEST.json`。

## 差异

| | `gate/epoch3/`（本目录） | `lq_eval/` |
|:--|:--|:--|
| 序列数 | **8** | **24**（全量龙泉山） |
| 特征源 | `longquanshan_ft_gmc_median_official`（official 逐 lag 拟合） | `longquanshan_ir_gmc_median`（**FPV 仿射连乘**） |
| 用途 | 训练期源域 gate 门禁 | 业务域交付（单帧 / 带 track / OSD） |
| 能否混用 | ❌ **不可** | — |

两套特征语义**逐像素差 26~70/255**（非边界帧 official 字节命中 0/3，FPV 3/3），
混用会让"全量 F1"变成两个分布的加权平均，**无法做差值归因**。
判定脚本：`manu/data/verify_longquan_feature_source.py`。
取证过程：`manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md` §54.2。

> 本文件只是**新增的指引**，不修改本目录任何原有产物。
