# 冻结基线轮代码镜像（2026-11-26 基线轮）

**这是什么**：基线轮（baseline）**已交付并冻结**的那一版代码，逐字节拷贝。

**为什么需要它**：冻结的基线清单 `iqa_v2/output/manifest_v2.md5` 里登记了代码条目的
md5（`iqa_v2/*.py`）。深轮（deep）改了 `degradation_4d.py` / `run_eval_v2.py` /
`make_manifest_v2.py` / `test_iqa_v2.py` / `audit_final.py` ⇒ 拿**当前**代码去核对
一份**历史**清单必然假失败。

⇒ 解法：把基线轮的代码冻结一份在这里，清单的**代码条目**在这里核对；
清单的**产物条目**（`iqa_v2/output/*`、报告正文）仍然**就地核对**
（这才是「基线产物没被覆盖」的关键证据，规格书 C3）。

由 `iqa_v2/test_iqa_v2.py` 的 `BASELINE_CODE_DIR` 引用。

🔴 **不要删本目录**：删掉之后清单的代码条目就失去了唯一可信的核对依据，
只剩「拿当前代码核对历史清单」⇒ 必然假失败 ⇒ C4 门禁必然红。

🔴 **本目录里的文件一个字都不许改**：它们是证据。改了就等于改了基线轮的证据链。
逐字节来源：`iqa_v2/server_out_final/files/`（服务器回收的原件）与
`iqa_v2/server_out_final/` 之外的基线工作区拷贝；每个文件的 md5 与
`iqa_v2/output/manifest_v2.md5` 中登记的基线代码条目**逐一相符**。
