| `grid_compare/rt_grid_compare.md` / `.csv` | 第 90 轮：Δn 网格搜索的指标评估与对比（16 臂交付口径 + 3 臂×4 跳帧情景 + 截断↔交付口径校准），复算 `[VERIFY] 267/267 checks passed` |
| `grid_compare/rt_fps_grid.md` / `.csv` | 第 91 轮：交付帧率阶梯（`f25`~`f12` 五档），机制与观测 9/9 逐点吻合 |
| `artifacts/rt_fps_grid_r90.json` / `artifacts/analyse_runs_r90.csv` | 第 91 轮服务器实测产物与节拍中间量 |
| `manu/evaluation/rt_fps_jitter_search.py` / `rt_fps_grid_report.py` | 第 91 轮帧率网格脚本（报告脚本 import 复用 `rt_grid_compare.py` 的 pooled 口径） |
| `search/make_memory_section_fps_grid_91.py` / `make_delivery_docs_r90_r91.py` | 落档与业务文档生成器（数字机械派生 + 漂移自检） |
