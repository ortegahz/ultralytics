# host 侧脚本独立审计结论（2026-11-26 17:50，第三方只读审计）

> **本文件是保底件**：host 关机后若加固补丁未落盘，按本清单重做即可。
> 涉及两个**纯 host 侧**脚本，**与服务器上的评测/报告无关**：
> - `iqa_v2/collect_from_server.sh` —— 从服务器回收产物 + 代跑三道门禁 + 两侧逐字节比对
> - `iqa_v2/merge_memory.sh` —— 把服务器 `iqa_v2/MEMORY_SECTION.md` 并入 host 权威记忆库 `manu/memory/memory.md`

## 一、`merge_memory.sh`

| # | 缺陷 | 症状 | 修法 |
|:--|:--|:--|:--|
| 1 | fallback 分支**整体覆盖**章节正文 | 输入 10 行 → 落盘 1 行，正文全丢，而脚本打印「[OK] 已插入 §319（共 1 行）」、roll+check+validate **三道校验全绿** | 用 `re.subn(r'^##\s*§\d+', '## §%d', sec, count=1, flags=re.M)` 重编号；未匹配上才补标题且必须**前置**（`'## §%d · %s\n\n%s'`），绝不丢正文 |
| 2 | 「插入前后行数/标题数断言」实际未生效 | fallback 场景 10 行 → 1 行仍报 OK | 断言改**硬失败**：正文中 `^## §\d+` 标题**恰好 1 个**、**正文逐行与输入一致** |
| 3 | 章节标题不含 `§` 时落不进记忆库索引 | 该节既不在 §6.2 索引也不在校验里 | 合并前断言标题含 `§`，否则 exit 2 |
| 4 | `cp` 备份失败仍打印「已备份」（无 `set -e`） | 回滚路径不可用 | `cp` 失败即 exit 1；回滚命令带上新增分卷删除 |
| 5 | `check_memory_roll.py` 输出被 `\| tail -8` 截断 | FAIL 明细看不到 | 落文件再打印（`tail -30`） |
| 6 | 下一个空闲 § 号只扫活区不扫 archive | 历史乱序补记会撞号 | 活区 + archive 一起扫（实测今天不会撞，但明天可能） |

## 二、`collect_from_server.sh`

| # | 缺陷 | 症状 | 修法 |
|:--|:--|:--|:--|
| 1 | 服务器清单为空 / ssh 失败仍报「✅ 共有 0 个文件逐字节一致」 | **假绿**：空文件 `diff -q` 判同 | ssh/pytest 失败即 exit 1；服务器清单为空即 exit 1；共有文件数必须 > 0 |
| 2 | `awk 'NR==FNR{...}'` 在**第一个文件为空**时对第二文件每行都为真 | `_srv_sub` 恒空 → 比对形同虚设 | 改 `awk 'FILENAME==ARGV[1]{b[$2]=$1;next} ($2 in b){print b[$2]"  "$2}'` |
| 3 | pytest 失败计数被 ANSI 转义码挡住（`^[[31mFAILED`） | `grep -cE '^(FAILED\|ERROR)'` 恒为 0 | 去行首锚点 `grep -acE '(FAILED\|ERROR)'` + **以退出码为准** + 读 summary 行 |
| 4 | `scp output/*` 遇目录静默丢整棵 `post_check/` 证据树 | 门禁证据缺失但不报错 | `scp -r` + 逐项核对（目录必须在、文件数一致） |
| 5 | 服务器 `manifest_v2.md5` 已过期（清单 12:35 生成，之后 4 个脚本都改过） | `md5sum -c` 与 `test_output_manifest_md5_all_ok` 必然红 | 门禁前先在服务器重生成清单（`make_manifest_v2.py`）；**默认关闭**，用 `REGEN_MANIFEST=1` 开关 |
| 6 | `grep -F -f` 子串误配（`windows.csv` 误配 `old_windows.csv.bak`） | 假警报（多报不一致） | `_hst_sub` 改 awk 按 `$2` 精确匹配 |
| 7 | ssh/scp 全程不看退出码 | 服务器不可达时静默通过 | 逐条判退出码，stderr 落证据文件 |
| 8 | verify 的 grep 模式匹配不到真实输出 | 结论页「最大误差」行是空的 | 模式改 `最大绝对误差\|max.*err\|max_abs` |
| 9 | 结论页只有计数没有判定 | 读者自行脑补 | 加 `GATES: pytest=… verify=… md5=…` 总判定 |
| 10 | `files/` 从不清空、计数是累计值 | 「本轮拉回 N 个」是假的 | 每次跑前清空或写带时间戳的新目录 |
| 11 | 中间产物落固定名 `/tmp/_srv.md5` 等 7 个文件 | 重复/并发跑互相覆盖，且证据跑完即丢 | `mktemp -d` + 两侧清单落 `gates/` |

## 三、自测要求（改完必须逐条验）

- 沙箱桩脚本模拟：ssh 失败 / 清单为空 / 空清单+awk / pytest 非零退出 / 目录拷贝丢失 / grep 子串误配 / verify 模式不命中 ⇒ 每种都要 **exit ≠ 0**。
- `merge_memory.sh` 沙箱：`/tmp/memtest/{manu/memory, iqa_v2}` 复制真实记忆库与脚本 ⇒ 正常章节 / **不以 § 开头**的章节（复现原 bug）/ 畸形输入（空文件）三例，正文必须逐字节一致；**绝不可碰真实 `manu/memory/memory.md`**。
- 服务器只读自检可用（`ls`/`md5sum`/`py_compile`/跑 pytest），**不得**在服务器上重生成 manifest、不得写服务器文件、不得起训练/推理/评测。
- 服务器 pytest 须用 `/home/huangzhe/anaconda3/envs/uav/bin/python -m pytest` 且 `PYTHONPATH=/tmp/pylibs_cvcontrib`。
