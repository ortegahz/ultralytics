# 交付件审计：跨尺度候选合并（cross-scale candidate merge）是否进了**二进制**

> 由 `manu/evaluation/artifact_audit_20261120.py` 生成（只读，无板卡 / 无 NPU / 无训练）。
> 判定依据：`libmosaic_pipeline.so` 是否导出 `mosaic_pipeline_set_candidate_dedup_radius`、
> 两个探针是否带 `--cand-dedup-radius` 开关。

## 结论

- 审计二进制 **15** 个，**含合并修复 0** 个（不含 15 个）。
- **任何交付件带上修复了吗：否**。
- 源码槽位与仓库修复版不一致的：**2** 处。

## 逐包明细

| 包 | 二进制 | md5 | 含合并修复 |
|:--|:--|:--|:--|
| delivery:mosaic_sdk_20261120 | `out/libmosaic_pipeline.so` | `e9b37af077cf` | ❌ 否 |
| delivery:mosaic_sdk_20261120 | `out/mosaic_track_dump` | `5a854a07dcfb` | ❌ 否 |
| delivery:mosaic_sdk_20261120 | `out/mosaic_cadence_probe` | `3fe28c17ea8d` | ❌ 否 |
| delivery:mosaic_sdk_20261120_ocv48 | `out/libmosaic_pipeline.so` | `b383291e31cc` | ❌ 否 |
| delivery:mosaic_sdk_20261120_ocv48 | `out/mosaic_track_dump` | `5a854a07dcfb` | ❌ 否 |
| delivery:mosaic_sdk_20261120_ocv48 | `out/mosaic_cadence_probe` | `3fe28c17ea8d` | ❌ 否 |
| delivery:mosaic_sdk_20261120_ocv42 | `out/libmosaic_pipeline.so` | `6baf4bc5f155` | ❌ 否 |
| delivery:mosaic_sdk_20261120_ocv42 | `out/mosaic_track_dump` | `5a854a07dcfb` | ❌ 否 |
| delivery:mosaic_sdk_20261120_ocv42 | `out/mosaic_cadence_probe` | `3fe28c17ea8d` | ❌ 否 |
| staging:ultralytics/manu/pipeline/npu/build | `out/libmosaic_pipeline.so` | `e9b37af077cf` | ❌ 否 |
| staging:ultralytics/manu/pipeline/npu/build | `out/mosaic_track_dump` | `5a854a07dcfb` | ❌ 否 |
| staging:ultralytics/manu/pipeline/npu/build | `out/mosaic_cadence_probe` | `3fe28c17ea8d` | ❌ 否 |
| mirror:rk3588_app_base/mosaic_sdk | `out/libmosaic_pipeline.so` | `e9b37af077cf` | ❌ 否 |
| mirror:rk3588_app_base/mosaic_sdk | `out/mosaic_track_dump` | `5a854a07dcfb` | ❌ 否 |
| mirror:rk3588_app_base/mosaic_sdk | `out/mosaic_cadence_probe` | `3fe28c17ea8d` | ❌ 否 |

## 源码是否与仓库修复版一致

| 包 | 文件 | 与仓库一致 |
|:--|:--|:--|
| delivery:mosaic_sdk_20261120 | `mosaic_track_fusion.cpp` | ✅ |
| delivery:mosaic_sdk_20261120 | `mosaic_track_fusion.h` | ✅ |
| delivery:mosaic_sdk_20261120 | `mosaic_pipeline.cpp` | ✅ |
| delivery:mosaic_sdk_20261120 | `mosaic_pipeline.h` | ✅ |
| delivery:mosaic_sdk_20261120 | `mosaic_track_dump.cpp` | ✅ |
| delivery:mosaic_sdk_20261120 | `mosaic_cadence_probe.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv48 | `mosaic_track_fusion.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv48 | `mosaic_track_fusion.h` | ✅ |
| delivery:mosaic_sdk_20261120_ocv48 | `mosaic_pipeline.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv48 | `mosaic_pipeline.h` | ✅ |
| delivery:mosaic_sdk_20261120_ocv48 | `mosaic_track_dump.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv48 | `mosaic_cadence_probe.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv42 | `mosaic_track_fusion.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv42 | `mosaic_track_fusion.h` | ✅ |
| delivery:mosaic_sdk_20261120_ocv42 | `mosaic_pipeline.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv42 | `mosaic_pipeline.h` | ✅ |
| delivery:mosaic_sdk_20261120_ocv42 | `mosaic_track_dump.cpp` | ✅ |
| delivery:mosaic_sdk_20261120_ocv42 | `mosaic_cadence_probe.cpp` | ✅ |
| staging:ultralytics/manu/pipeline/npu/build | `mosaic_track_fusion.cpp` | ✅ |
| staging:ultralytics/manu/pipeline/npu/build | `mosaic_track_fusion.h` | ✅ |
| staging:ultralytics/manu/pipeline/npu/build | `mosaic_pipeline.cpp` | ✅ |
| staging:ultralytics/manu/pipeline/npu/build | `mosaic_pipeline.h` | ✅ |
| staging:ultralytics/manu/pipeline/npu/build | `mosaic_track_dump.cpp` | ❌ 缺失 |
| staging:ultralytics/manu/pipeline/npu/build | `mosaic_cadence_probe.cpp` | ❌ 缺失 |
| mirror:rk3588_app_base/mosaic_sdk | `mosaic_track_fusion.cpp` | ✅ |
| mirror:rk3588_app_base/mosaic_sdk | `mosaic_track_fusion.h` | ✅ |
| mirror:rk3588_app_base/mosaic_sdk | `mosaic_pipeline.cpp` | ✅ |
| mirror:rk3588_app_base/mosaic_sdk | `mosaic_pipeline.h` | ✅ |
| mirror:rk3588_app_base/mosaic_sdk | `mosaic_track_dump.cpp` | ✅ |
| mirror:rk3588_app_base/mosaic_sdk | `mosaic_cadence_probe.cpp` | ✅ |

## MANIFEST 自洽性

| 包 | md5 行数 | 不一致行 |
|:--|--:|:--|
| delivery:mosaic_sdk_20261120 | 69 | — |
| delivery:mosaic_sdk_20261120_ocv48 | 69 | — |
| delivery:mosaic_sdk_20261120_ocv42 | 69 | — |
| staging:ultralytics/manu/pipeline/npu/build | 67 | `README.md`, `build_demo.sh`, `build_lib.sh`, `src/npu/mosaic_pipeline.cpp`, `src/npu/mosaic_pipeline.h`, `src/npu/mosaic_track_fusion.cpp`, `src/npu/mosaic_track_fusion.h` |
| mirror:rk3588_app_base/mosaic_sdk | 67 | `build_lib.sh`, `fetch_board_libs.sh`, `models/p0_off_A_none.rknn` |

