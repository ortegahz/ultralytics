# 自适应闭环夜间台账（2026-11-11 夜 → 2026-11-12 09:00）
> 目标：击败 Baseline_v2.0 = trial_9863（nano 0.708648 严格线 / 0.70097 稳健线）

## [2026-11-11 20:19:16] 第 2 轮完成

- 冠军配方：`sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3`（n=1，均值 nano=0.757798，最佳 0.757798）★已升格为搜索基线
- 基线配方稳健线：n=8 均值 nano **0.696568**（严格线 = 单臂 0.708648，trial [9863, 9912, 12039, 12040, 12041, 12042, 12043, 12044]）
- 本轮 62 臂；六项全平基线的臂：**无**
- 待复现配方：['sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=2.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.35|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.2|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.15|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=25.0|maxside=6.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3']

| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | 门禁 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12131 | [R2] ★ 突变 swl0.1：swl0.3 下界（只留极小目标的梯度） | 4 | 0.757798 | 0.679348 | 0.877716 | 0.7993 | 0.8646293723955044 | 0 | 0.8914129463111953 | FAIL |
| 12126 | [R2] ★ 种子扫描 5/6（冠军配方复现：sw=5.0） | 4 | 0.735113 | 0.683129 | 0.887611 | 0.8294 | 0.8830105687580182 | 0 | 0.9327867193035733 | FAIL |
| 12130 | [R2] ★ 突变 swl0.2：swl0.3 继续下探（把大目标梯度霸凌压到更 | 3 | 0.730151 | 0.641541 | 0.879656 | 0.8303 | 0.8763108748742996 | 0 | 0.9250280126311501 | FAIL |
| 12132 | [R2] ★ 突变 swl0.15：swl0.2 与 0.1 之间 | 5 | 0.722353 | 0.652883 | 0.882600 | 0.8346 | 0.8764429600250495 | 0 | 0.9123512134796723 | FAIL |
| 12145 | [R2] ★ 突变 sw8.0：剂量上探 sw=8.0（9822 的 nano@ | 5 | 0.717864 | 0.665643 | 0.885143 | 0.8282 | 0.8810451618195987 | 0 | 0.9358334593477493 | FAIL |
| 12047 | [R1] ★ 突变 swl0.3：大目标降权下界（把大目标梯度霸凌压到最低） | 5 | 0.716919 | 0.646739 | 0.883899 | 0.8209 | 0.8738691009862255 | 4 | 0.9368426349957223 | FAIL |
| 12125 | [R2] ★ 种子扫描 4/6（冠军配方复现：sw=5.0） | 3 | 0.715737 | 0.627127 | 0.889692 | 0.8175 | 0.8799233466525319 | 2 | 0.9311375728401318 | FAIL |
| 12129 | [R2] ★ 突变 swl0.35：swl0.3 右侧细探（0.3 与 0.5  | 3 | 0.713374 | 0.610350 | 0.887576 | 0.8294 | 0.8860780491850868 | 0 | 0.9355959514577773 | FAIL |
| 12121 | ★ 合成臂 swa_early：champion 臂 epoch{~2}~{~+ | 5 | 0.713138 | 0.655955 | 0.883326 | 0.8200 | - | - | - | - |
| 12120 | ★ 合成臂 swa_wide：champion 臂 epoch{~2}~{~+6 | 5 | 0.713138 | 0.655955 | 0.883326 | 0.8200 | - | - | - | - |
| 12119 | ★ 合成臂 swa_neigh：champion 臂 epoch{~2}~{~+ | 5 | 0.713138 | 0.655955 | 0.883326 | 0.8200 | - | - | - | - |
| 12134 | [R2] ★ 突变 swl0.3_thr25ms6：swl0.3 × 宽小目标定 | 3 | 0.712902 | 0.631144 | 0.881901 | 0.8428 | - | - | - | - |
| 12141 | [R2] ★ 突变 swl0.3_warmup1.0：swl0.3 × warm | 3 | 0.712665 | 0.624764 | 0.887591 | 0.8249 | - | - | - | - |
| 12138 | [R2] ★ 突变 swl0.3_mixw2.2：swl0.3 × 混样 w2. | 3 | 0.712665 | 0.630435 | 0.883846 | 0.8279 | - | - | - | - |

## [2026-11-11 23:33:09] 第 3 轮完成

- 冠军配方：`sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3`（n=7，均值 nano=0.731535，最佳 0.757798）★已升格为搜索基线
- 基线配方稳健线：n=8 均值 nano **0.696568**（严格线 = 单臂 0.708648，trial [9863, 9912, 12039, 12040, 12041, 12042, 12043, 12044]）
- 本轮 36 臂；六项全平基线的臂：**无**
- 待复现配方：['sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=2.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.35|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.2|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.15|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=25.0|maxside=6.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3']

| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | 门禁 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12183 | ★ 合成臂 swa_late：champion 臂 epoch{~0}~{~+4 | 5 | 0.755198 | 0.678875 | 0.877572 | 0.7993 | 0.8647465049688395 | 0 | 0.8915637319316689 | FAIL |
| 12185 | ★ 合成臂 swa_early：champion 臂 epoch{~2}~{~+ | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12184 | ★ 合成臂 swa_wide：champion 臂 epoch{~2}~{~+6 | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12182 | ★ 合成臂 swa_neigh：champion 臂 epoch{~2}~{~+ | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12195 | [R3] ★ 种子扫描 2/6（冠军配方复现：sw=5.0） | 4 | 0.739839 | 0.675803 | 0.877115 | 0.8115 | 0.8564534545532887 | 0 | 0.8938869665513263 | FAIL |
| 12209 | [R3] ★ 突变 thr12ms3_ow0.5：更严小目标定义 × offse | 4 | 0.735586 | 0.664225 | 0.875826 | 0.8218 | 0.8566556538420548 | 0 | 0.8865053365740251 | FAIL |
| 12197 | [R3] ★ 种子扫描 4/6（冠军配方复现：sw=5.0） | 3 | 0.732514 | 0.667769 | 0.871091 | 0.8297 | 0.8480166394295053 | 0 | 0.8750800768737989 | FAIL |
| 12214 | [R3] ★ 突变 sw8.0：剂量上探 sw=8.0（9822 的 nano@ | 4 | 0.727552 | 0.677221 | 0.874265 | 0.8227 | 0.8602497663749894 | 0 | 0.8961147764163786 | FAIL |
| 12215 | [R3] ★ 突变 sw12.0：剂量上探 sw=12.0（sw10.0=0.7 | 5 | 0.727316 | 0.654537 | 0.876523 | 0.8175 | - | - | - | - |
| 12198 | [R3] ★ 种子扫描 5/6（冠军配方复现：sw=5.0） | 3 | 0.725898 | 0.651465 | 0.880037 | 0.8151 | - | - | - | - |
| 12194 | [R3] ★ 种子扫描 1/6（冠军配方复现：sw=5.0） | 5 | 0.725189 | 0.645321 | 0.867973 | 0.8175 | - | - | - | - |
| 12217 | [R3] ★ 突变 lrfloor0.05：学习率保底 5% | 3 | 0.724244 | 0.665643 | 0.876709 | 0.8361 | - | - | - | - |
| 12202 | [R3] ★ 突变 lrfloor0.10_thr12ms3：两个保 F1 的  | 5 | 0.724244 | 0.664225 | 0.887051 | 0.8288 | - | - | - | - |
| 12208 | [R3] ★ 突变 lrfloor0.10_ow0.5：LR 保底 10% ×  | 5 | 0.724008 | 0.655246 | 0.871837 | 0.8343 | - | - | - | - |

## [2026-11-12 02:40:47] 第 4 轮完成

- 冠军配方：`sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3`（n=13，均值 nano=0.727788，最佳 0.757798）★已升格为搜索基线
- 基线配方稳健线：n=8 均值 nano **0.696568**（严格线 = 单臂 0.708648，trial [9863, 9912, 12039, 12040, 12041, 12042, 12043, 12044]）
- 本轮 36 臂；六项全平基线的臂：**无**
- 待复现配方：['sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=2.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.35|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.2|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.15|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=25.0|maxside=6.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3']

| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | 门禁 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12219 | ★ 合成臂 swa_late：champion 臂 epoch{~0}~{~+4 | 5 | 0.755198 | 0.678875 | 0.877572 | 0.7993 | 0.8647465049688395 | 0 | 0.8915637319316689 | FAIL |
| 12221 | ★ 合成臂 swa_early：champion 臂 epoch{~2}~{~+ | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12220 | ★ 合成臂 swa_wide：champion 臂 epoch{~2}~{~+6 | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12218 | ★ 合成臂 swa_neigh：champion 臂 epoch{~2}~{~+ | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12249 | [R4] ★ 突变 mixw2.2：混样 w2.0→w2.2（细探） | 3 | 0.750709 | 0.697306 | 0.881879 | 0.8267 | 0.8682497127537342 | 0 | 0.8813541335031306 | FAIL |
| 12238 | [R4] ★ 突变 thr12ms3：更严小目标定义（4px 档压力测试） | 4 | 0.743620 | 0.652647 | 0.877876 | 0.8246 | 0.8627657557049242 | 0 | 0.8951862133989839 | FAIL |
| 12241 | [R4] ★ 突变 warmup2.5：warmup 1.5→2.5 | 5 | 0.741257 | 0.666352 | 0.878290 | 0.8075 | 0.8689179759362247 | 0 | 0.894505033557047 | FAIL |
| 12229 | [R4] ★★ 复现 sw=5.0 seed2（上一轮单臂打平基线） | 3 | 0.741257 | 0.670369 | 0.885301 | 0.8349 | 0.8807514462852758 | 0 | 0.919541406489917 | FAIL |
| 12233 | [R4] ★ 种子扫描 4/6（冠军配方复现：sw=5.0） | 3 | 0.738422 | 0.645321 | 0.879571 | 0.8169 | - | - | - | - |
| 12240 | [R4] ★ 突变 warmup1.0：warmup 1.5→1.0（9871  | 5 | 0.735350 | 0.660444 | 0.874668 | 0.8343 | - | - | - | - |
| 12239 | [R4] ★ 突变 thr40ms10：极宽小目标定义 | 3 | 0.734641 | 0.669896 | 0.879930 | 0.8486 | - | - | - | - |
| 12245 | [R4] ★ 突变 fb3.0：focal beta 2.75→3.0 | 5 | 0.733932 | 0.669423 | 0.878885 | 0.8306 | - | - | - | - |
| 12243 | [R4] ★ 突变 lr4.0：lr_scale 3.5→4.0 | 4 | 0.733932 | 0.648629 | 0.874609 | 0.8325 | - | - | - | - |
| 12234 | [R4] ★ 种子扫描 5/6（冠军配方复现：sw=5.0） | 3 | 0.732042 | 0.665406 | 0.881531 | 0.8337 | - | - | - | - |

## [2026-11-12 06:04:04] 第 5 轮完成

- 冠军配方：`sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3`（n=19，均值 nano=0.724716，最佳 0.757798）★已升格为搜索基线
- 基线配方稳健线：n=8 均值 nano **0.696568**（严格线 = 单臂 0.708648，trial [9863, 9912, 12039, 12040, 12041, 12042, 12043, 12044]）
- 本轮 36 臂；六项全平基线的臂：**无**
- 待复现配方：['sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=1.0|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=2.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.35|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.2|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.1|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.15|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=25.0|maxside=6.0|lr=3.5|lr_floor=0.0|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3', 'sw=5.0|swl=0.3|thr=16.0|maxside=4.0|lr=3.5|lr_floor=0.1|warmup=1.5|mix=w2.0|fb=2.75|ow=0.4|wd=0.000269|mosaic=0.15|scale=0.1|translate=0.05|clip=5.65|sched=5|uf=0|hf=0|hfsrc=b1|hfch=1|native_grid=0|mr=1|backbone_lr=1.0|head_src=p3']

| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | 门禁 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12287 | [R5] ★ 突变 uf1：解冻 b0~b4（更激进的域适配） | 3 | 0.756380 | 0.684074 | 0.884377 | 0.8407 | 0.8600879896282757 | 0 | 0.8534609319775976 | FAIL |
| 12255 | ★ 合成臂 swa_late：champion 臂 epoch{~0}~{~+4 | 5 | 0.755198 | 0.678875 | 0.877572 | 0.7993 | 0.8647465049688395 | 0 | 0.8915637319316689 | FAIL |
| 12257 | ★ 合成臂 swa_early：champion 臂 epoch{~2}~{~+ | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12256 | ★ 合成臂 swa_wide：champion 臂 epoch{~2}~{~+6 | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12254 | ★ 合成臂 swa_neigh：champion 臂 epoch{~2}~{~+ | 5 | 0.751654 | 0.673677 | 0.877097 | 0.7941 | 0.8618244172999979 | 0 | 0.8929152880626017 | FAIL |
| 12277 | [R5] ★ 突变 sched6：cosine 周期 5→6 ep | 6 | 0.750236 | 0.678166 | 0.878300 | 0.8255 | 0.8701146539459369 | 0 | 0.8984370916313835 | FAIL |
| 12267 | [R5] ★ 种子扫描 2/6（冠军配方复现：sw=5.0） | 5 | 0.731333 | 0.675803 | 0.874133 | 0.8048 | 0.867375395776981 | 0 | 0.8913317904481379 | FAIL |
| 12279 | [R5] ★ 突变 wd5e4：weight decay ×1.86 | 5 | 0.731096 | 0.655246 | 0.876198 | 0.8282 | 0.8627235445820827 | 0 | 0.8990078328981722 | FAIL |
| 12266 | [R5] ★ 种子扫描 1/6（冠军配方复现：sw=5.0） | 5 | 0.729915 | 0.646030 | 0.874206 | 0.8182 | - | - | - | - |
| 12289 | [R5] ★ 突变 grid_sw5.0_swl0.1_thr20：系统网格：s | 5 | 0.727079 | 0.649338 | 0.872724 | 0.8191 | - | - | - | - |
| 12285 | [R5] ★ 突变 mr2：min_radius 1→2（监督峰更宽） | 5 | 0.726134 | 0.644849 | 0.876903 | 0.8313 | - | - | - | - |
| 12276 | [R5] ★ 突变 sw6.0_swl0.5_thr25ms6：三变量组合 | 3 | 0.724244 | 0.646030 | 0.892345 | 0.8407 | - | - | - | - |
| 12282 | [R5] ★ 突变 scale0.20：scale 抖动 0.10→0.20 | 5 | 0.723299 | 0.663280 | 0.876176 | 0.8303 | - | - | - | - |
| 12265 | [R5] ★★ 复现 sw=5.0 seed2（上一轮单臂打平基线） | 3 | 0.722590 | 0.642013 | 0.883483 | 0.8182 | - | - | - | - |
