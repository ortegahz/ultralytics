# 板卡 FAR 收口（2026-11-20）· 根因 → 修复 → 逐位验证

> 权威专题档：**`manu/memory/trial0080_board_gt_scoring_20261120.md`**（板端在线打分报告，§十二 = SDK 侧落地）
> 主档：**`manu/memory/memory.md` §263 / §264 / §265**
> 🔴 **交付件状态**（2026-11-20 14:2x 审计）：修复**只在源码里**，15 个交付二进制**0 个**带修复
> ⇒ 现在发货，板端 FAR 仍是 0.3832。详见 **`delivery_closure_20261120.md`**。

## 根因（一句话）

服务器在喂融合引擎**之前**会把三个尺度的热图峰**合并成一个候选**（`merge_scales(dedup_radius)`），
板端 SDK **没有这一步** ⇒ 同一个目标在三个尺度上各出一个峰 ⇒ 融合后裂成多条航迹 ⇒
每条多出来的航迹都是一次误报 ⇒ **逐帧 FAR 0.3832**（红线 0.03 的 **12.8 倍**）。

## 修复

SDK 新增 `mosaic::MergeCandidatesByRadius()`（候选构造阶段，与服务器 `_merge_scale_rows` **逐条等价**）
+ 公开接口 `mosaic_pipeline_set_candidate_dedup_radius()`（**不**动 `MosaicTrackFusionConfig` ⇒ ABI 不破）。
调用侧**必须显式**打开：`mosaic_pipeline_set_candidate_dedup_radius(pipe, 20.0)`（可用带 16~24，**别抄 768**）。

## 本目录的产物

| 文件 | md5 | 说明 |
|:--|:--|:--|
| `merge_verify.json` | 见下 | **SDK 修复的验证报告**：5 档半径的 C++ 输出 + 与服务器 Python 镜像的**逐字节**比对 + 同口径打分 |
| `fused_cpp_r0.tsv` | `839e1f437be0196e5fb4c54da6105d9e` | C++ 融合输出（**不开**合并）= 板端原行为；与已归档的 `python_fused_dn1.tsv` 逐字节相同 |
| `fused_cpp_r20.tsv` | `dd55cb09a844f41408322374ddc232cf` | C++ 融合输出（**开**合并，半径 20 px）= 交付值 |
| `dedup_sweep.json` | — | 服务器 Python 镜像的半径扫描（离线推算） |
| `parity_board_as_is.json` | — | 板端（未合并）↔ 服务器 逐帧轨道比对 + 打分 |
| `parity_dedup24.json` | — | 板端（合并 R=24）↔ 服务器 逐帧轨道比对 + 打分 |
| `board/` | — | **板卡上的原始产物**（抢救归档，板卡 tmpfs 已丢）+ 板卡原生重编的 SDK 源码与 `.so` |
| `delivery_closure_20261120.md` | — | **交付件闭环审计**（本轮）：15 个二进制 0 个带修复 + 纠正 §265 的一处误判 + 重编路径与阻塞点 |
| `artifact_audit_20261120.md` / `artifact_audit.json` | — | 交付件审计的**机器可读**结果（由 `manu/evaluation/artifact_audit_20261120.py` 生成，可重跑） |

## 关键数字（窗口 = `VIDEO00007_19700101_001324` 的 0..2499 帧，GT 2,420，同窗口同 GT 同匹配器）

| 口径 | F1 | FAR | tp / fp / fn |
|:--|--:|--:|--:|
| 板端原行为（无合并） | 0.5689473 | **0.3832** | 1343 / 958 / 1077 |
| 板端 + 合并 16 px | 0.6907507 | 0.0540 | 1348 / 135 / 1072 |
| **板端 + 合并 20 px（交付值）** | **0.6930591** | **0.0488** | 1348 / 122 / 1072 |
| 板端 + 合并 24 px | 0.6927030 | 0.0496 | 1348 / 124 / 1072 |
| 服务器参考（同窗口） | 0.6918947 | 0.0552 | 1353 / 138 / 1067 |

⇒ 开 20 px 后板端 vs 服务器：F1 **+0.0012**、FAR **−0.0064**、匹配率 **82.5%**（改前 −0.1229 / +0.3280）。
⚠️ 0..2499 是**困难片段**（服务器自己这段 FAR 0.0552，全序列仅 0.0262）⇒ **绝对 FAR 不可判红线**。

## 复现（本机，无 GPU / 无 NPU / 无训练）

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python
PYTHONPATH=. $PY manu/evaluation/board_cand_merge_verify.py \
  --cands manu/memory/align_dn_20261120/real_0080/real_cands.tsv \
  --params manu/memory/align_dn_20261120/params_delivery.txt \
  --labels-root /tmp/boardfar/gt --sequence VIDEO00007_19700101_001324 \
  --window 0 2500 --radii 0,16,20,24,32 \
  --python-mirror-dir /tmp/boardfar/probe --out manu/memory/board_far_20261120/merge_verify.json
```

⇒ 期望输出：`R=0.0 … R=32` 五行，全部 `identical_to_python_mirror = true`。

⚠️ **工作目录必须是 `<ultralytics>`**（不是 `<ultralytics>/manu`）：
脚本在 `manu/evaluation/` 下、`--out manu/memory/...` 也相对 `<ultralytics>` ⇒ `cd` 到 `manu/` 会
`can't open file 'manu/evaluation/board_cand_merge_verify.py'`（2026-11-20 14:2x 实测踩到，已修）。

## 交付件自检（发货前跑一次，重编后应全绿）

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python
$PY manu/evaluation/artifact_audit_20261120.py
```

⇒ 期望：`二进制 15 个；含修复 15 个` / `任何交付件带上修复：是`。
**当前是 `0 / 15`** ⇒ 说明还没重编（`.so` 与两个探针必须**一起**重编，详见 `delivery_closure_20261120.md` §3）。
