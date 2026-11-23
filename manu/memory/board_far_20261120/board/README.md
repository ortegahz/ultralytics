# 板端 GT 打分原始产物（2026-11-20，从板卡 `/tmp` 抢救性归档）

> 板卡 `root@192.168.0.64`（`evm3588`）；本目录是**板卡上的原始输出**，不是本机复算结果。
> 板卡 `/tmp/boardwork/` 已在归档后消失（板卡 tmpfs 会丢），本目录是它与 `/tmp/board_*.tsv` 的**合并快照**。

| 文件 | md5 | 说明 |
|:--|:--|:--|
| `board_jf_cands.tsv` | `7b7759be224b0a273d14d33b71565dc1` | |
| `board_jf_fused.tsv` | `e49dfb18d3e17cb1d51b81c204f9efe5` | |
| `board_jf_tracked.tsv` | `eb59351266ddaf6dc8687427cdeba50a` | |
| `board_td_cands.tsv` | `0e511a4c66a7f2842c79c0a9f73fe613` | |
| `board_td_fused.tsv` | `5a317dab22bb28497e93b3db77927270` | |
| `board_td_tracked.tsv` | `7aa855aa01ce593162bfb5369227a506` | |
| `board_tm1_cands.tsv` | `0e511a4c66a7f2842c79c0a9f73fe613` | |
| `board_tm1_fused.tsv` | `ce56c5c693364ae60d059db68a247788` | |
| `delivered_run2.ids` | `0cbab6c102e62acaffbf5f53707c4901` | |


## src/（板卡上 13:03 同步、板卡原生重编的 SDK 源码与 .so）

| 文件 | md5 |
|:--|:--|
| `src/libmosaic_pipeline.so` | `a8d56d26d3604d5c33d69a859c598a78` |
| `src/mosaic_cadence_probe.cpp` | `4966019433197b845c4d9a0852c26db0` |
| `src/mosaic_pipeline.cpp` | `02f6792df24221c84b3caa93c64c306b` |
| `src/mosaic_pipeline.h` | `2f0aec0600287ae99a56a10c02fbab0e` |
| `src/mosaic_track_dump.cpp` | `043aa7cdf9bb5f796f3d2cc67d53dc5f` |
| `src/mosaic_track_fusion.cpp` | `40de197a8c8de18a213fbf9bb2fe2987` |
| `src/mosaic_track_fusion.h` | `dba851ed2881f2ca647f666a68f7631c` |
