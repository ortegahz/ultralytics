# docgen —— 文档生成器（数字全部来自产物，**不手抄**）

这几个脚本把**产物 JSON**（`runs/lookahead/final/*/arms_delivery.json` 等）渲染成本轮四份文档的
Markdown 片段，再由 `land_round4.py` 追加进 memory / README / 板端契约，并**按标题重算导航表行号**。

| 脚本 | 作用 | 产出 |
|:--|:--|:--|
| `gen_docs_round4.py` | 专题档 §27（完整机理 + 全部表格） | `/mnt/data/codex-overnight/round4_topical.md` |
| `gen_docs_round4b.py` | 主档 §30.9.22 · README §5.11 · 板端契约 §3.7 契约七 | `round4_main.md` / `round4_readme.md` / `round4_contract.md` |
| `gen_handoff_round4.py` | 交接档 | `manu/memory/handoff_20261104_server_round4.md` |
| `land_round4.py` | 落盘 + 导航表行号重算（**幂等**：marker 已存在则跳过） | 直接写 4 个文档 |
| `reland_round4.py` | 「回退 → 重新生成 → 重新落盘」（生成器改动后重放） | 调 `gen_*` + `land_round4.py` |

## 顺序

```bash
cd /tmp/pycharm_project_10ae9e2e          # 服务器；host 侧路径需改
python manu/tracking/lookahead/docgen/reland_round4.py
python manu/tracking/lookahead/docgen/gen_handoff_round4.py
python3 manu/memory/validate_nav_table.py     # 必须 PASS
```

> ⚠️ `land_round4.py` 会**先**按 §27 / §30.9.22 / §5.11 的标题**回退**旧内容再追加（`reland_round4.py` 负责回退），
> 所以**改生成器后必须走 `reland_round4.py`**，直接跑 `land_round4.py` 会被 marker 跳过而保留旧内容。

## 口径提醒

- **CLE 有两套**：`cle_pooled_*`（全帧池化，**伺服可接受性**看这套）/ `cle_p50|95`（最差单序列，只看尾部风险）。
  生成器的 `servo()` 一律取**池化**口径。
- **FAR 红线 0.030** 是**整体**口径；单条序列可以超线（见专题档 §27.6 逐序列表）。
