# 交付件闭环审计（2026-11-20 14:2x~14:3x）· 修复在**源码**里，但**没有任何一个交付二进制**带上它

> 本文件由一次**独立复检**产出（只读审计 + 本机重编验证，**无板卡 / 无 NPU / 无训练 / 无网络**）。
> 机器可读结果：`artifact_audit.json`；人读表格：`artifact_audit_20261120.md`；审计脚本：`manu/evaluation/artifact_audit_20261120.py`。
> ⚠️ **并发提示**：本文写于 `manu/memory/memory.md` / `memory_compact.md` **被并发线改动的同一时段**
> （14:24 仍有写入）⇒ 按铁律九.1「单一写者纪律」，**本文没有去改那两个文件**，
> 需要并入主档的段落见本文 §四（可直接粘贴）。

---

## 一、结论（一句话）

**§265 的修复（跨尺度候选合并）只存在于仓库源码里**；
`mosaic_sdk_20261120{,_ocv48,_ocv42}` 三个交付包、工程暂存包、包镜像 —— **15 个二进制，0 个**带修复。

⇒ 现在把包交给嵌入式组 / 集成方，板端 **FAR 仍然是 0.3832（红线 0.03 的 12.8 倍）**，
因为跑起来的 `.so` 里的候选流**仍然没做跨尺度合并**。

| 判据（实测） | 结果 |
|:--|:--|
| `libmosaic_pipeline.so` 导出 `mosaic_pipeline_set_candidate_dedup_radius` | **0 / 5 个 .so** |
| `libmosaic_pipeline.so` 含 `MergeCandidatesByRadius`（C++ 符号） | **0 / 5** |
| 探针含 `--cand-dedup-radius` 开关 / `candidate dedup ON` 日志串 | **0 / 10** |
| 交付包内 `src/npu/*` 与仓库修复版一致 | **0 / 3 个包**（三个包全是修复前源码） |

### 逐包（md5 取前 12 位）

| 包 | `.so` | `mosaic_track_dump` | `mosaic_cadence_probe` | 包内源码 |
|:--|:--|:--|:--|:--|
| `build/mosaic_sdk_20261120` | `e9b37af077cf` | `5a854a07dcfb` | `3fe28c17ea8d` | 修复前 |
| `build/mosaic_sdk_20261120_ocv48` | `b383291e31cc` | `5a854a07dcfb` | `3fe28c17ea8d` | 修复前 |
| `build/mosaic_sdk_20261120_ocv42` | `6baf4bc5f155` | `5a854a07dcfb` | `3fe28c17ea8d` | 修复前 |
| `manu/pipeline/npu/build`（暂存包） | `e9b37af077cf` | `5a854a07dcfb` | `3fe28c17ea8d` | **修复版**（探针源缺） |
| `rk3588_app_base/mosaic_sdk`（包镜像） | `e9b37af077cf` | `5a854a07dcfb` | `3fe28c17ea8d` | **修复版** |

- `.so` 三种变体的编译器指纹：交叉版 = `GCC 9.3.0 (Buildroot)`；两个探针 = `GCC 9.4.0 Ubuntu 20.04`
  ⇒ **探针是在板卡上原生编的**（不是宿主交叉编的），所以它们的 mtime（14:01）是**回灌**时间戳。
- 三个交付包的 `src/npu/mosaic_pipeline.cpp` = `02f6792df242…`（修复前），仓库修复版 = `95cfea757ad2…`。

### 🔴 顺带纠正主档 §265 §4 的一条误判

§265 §4 写「交付包二进制是 09:07 产物，**不含 11:59 源码的 9 字段覆盖** ⇒ 仍需重编」——
**这条不对**：

- 探针**确实**带 9 字段覆盖（板卡侧 13:54 的源码归档 `board/src/mosaic_track_dump.cpp` 与仓库当前版本的
  diff **只有**跨尺度合并那几处 ⇒ 9 字段覆盖在 13:54 就已在内）；
- 真正的陈旧点是**合并修复本身**：`.so`（09:07）与探针（13:54 源码）都**早于** 13:58/13:59 的合并改动。

⇒ 「重编」这件事**要做**，但**理由和范围都变了**：不是补 9 字段，是补**合并修复**，
且**必须 `.so` 与两个探针一起重编**（见 §三）。

---

## 二、本轮**独立复跑**（复现 §265 的核心结论，不依赖上一轮的产物）

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
PY=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python
PYTHONPATH=. $PY manu/evaluation/board_cand_merge_verify.py \
  --cands manu/memory/align_dn_20261120/real_0080/real_cands.tsv \
  --params manu/memory/align_dn_20261120/params_delivery.txt \
  --labels-root /tmp/boardfar/gt --sequence VIDEO00007_19700101_001324 \
  --window 0 2500 --radii 0,16,20,24,32 \
  --python-mirror-dir /tmp/boardfar/probe --out /tmp/boardfar/merge_verify_recheck.json
```

实测输出（与归档 `merge_verify.json` **逐字段相同**，含每个半径的 `fused_tsv_md5`）：

```
R=  0.0  tp= 1343 fp=  958 fn= 1077 F1=0.5689473 FAR=0.3832
R= 16.0  tp= 1348 fp=  135 fn= 1072 F1=0.6907507 FAR=0.0540
R= 20.0  tp= 1348 fp=  122 fn= 1072 F1=0.6930591 FAR=0.0488
R= 24.0  tp= 1348 fp=  124 fn= 1072 F1=0.6927030 FAR=0.0496
R= 32.0  tp= 1348 fp=  122 fn= 1072 F1=0.6930591 FAR=0.0488
```

逐字节核对（独立 `md5sum`）：

| 比对 | 结果 |
|:--|:--|
| C++ R=0 输出 vs 归档 `python_fused_dn1.tsv` | **相同** `839e1f437be0196e5fb4c54da6105d9e` ⇒ 改前行为零回归 |
| C++ R=20 输出 vs 服务器 Python 镜像 `fused_r20.tsv` | **相同** `dd55cb09a844f41408322374ddc232cf` ⇒ 与服务器逐条等价 |
| 复跑 JSON vs 归档 `merge_verify.json` | 全部字段相同（`radius` / `fused_tsv_md5` / `identical_to_python_mirror` / `f1` / `far` / `tp` / `fp` / `fn`） |

### ⚠️ 顺手修掉的**复现命令 bug**

`manu/memory/board_far_20261120/README.md` 与主档 §265 §2 里那条命令**照抄会失败**：

```bash
cd .../workspace/ultralytics/manu          # ← 这里 cd 之后
PYTHONPATH=. $PY manu/evaluation/board_cand_merge_verify.py   # ← 这个路径不存在
```

实测报错：`can't open file 'manu/evaluation/board_cand_merge_verify.py': [Errno 2] No such file or directory`
（脚本在 `<ultralytics>/manu/evaluation/`，而 `--out manu/memory/...` 也是相对 `<ultralytics>` 的 ⇒
**整段命令的工作目录应当是 `<ultralytics>`**，不是 `<ultralytics>/manu`）。
✔ 已按正确工作目录复跑（上文命令）。

---

## 三、重编路径（本轮把「怎么重编」补齐了）

### 3.1 新增：`manu/pipeline/npu/packaging/build_probes.sh`

**此前全网没有任何脚本会编这两个探针**（板上的二进制是手敲 `g++` 编的）。
新脚本做的事：

1. **前置闸门**：先查 `out/libmosaic_pipeline.so` 是否导出 `mosaic_pipeline_set_candidate_dedup_radius`
   —— 没有就**直接失败并说明原因**（探针源码调用它，**链接期必然 undefined reference**，
   不是脚本保守，是硬依赖：`.so` 与探针**必须同版本**）；
2. 用**与库、与 Python 参考实现同一套数值开关**编译
   （`-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations`），
   链 `-lmosaic_pipeline -Wl,-rpath,'$ORIGIN' -Wl,--allow-shlib-undefined`（与 `build_lib.sh` 的 `RELAX` 同口径）；
3. 编译完**自检**：探针里必须能 `strings` 到 `--cand-dedup-radius`。

用法：

```bash
# 板卡上原生编（探针零第三方依赖 ⇒ 三个 OpenCV 变体通用）
cd <包根> && bash <repo>/manu/pipeline/npu/packaging/build_probes.sh

# 宿主交叉编（工具链见 §3.3）
CROSS_CXX=<aarch64 交叉 g++> bash <repo>/manu/pipeline/npu/packaging/build_probes.sh
```

### 3.2 本机实测（三条结论）

| 验证 | 结果 |
|:--|:--|
| 闸门：拿**修复前**的 `.so` 跑 `REQUIRE_MERGE=1` | **按设计失败**（明确提示「先重编 .so」） |
| 链接：拿修复前 `.so` 交叉编探针 | **链接失败**：`undefined reference to mosaic_pipeline_set_candidate_dedup_radius` ⇒ 证明**探针与 `.so` 必须一起重编** |
| **修复代码本身能不能为 aarch64 编出来** | ✅ **能**：`mosaic_track_fusion.cpp` 用交付工具链交叉编译通过，`nm -C` 见 `T mosaic::MergeCandidatesByRadius(...)`（全局可见符号） |

```bash
TC=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu/bin/aarch64-rockchip-linux-gnu-g++
cd .../manu/pipeline/npu
$TC -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -fPIC \
    -c mosaic_track_fusion.cpp -o /tmp/mosaic_track_fusion_fix.o
nm -C --defined-only /tmp/mosaic_track_fusion_fix.o | grep MergeCandidatesByRadius
```

### 3.3 🔴 阻塞点（本机**编不出 `.so`**，三个独立原因）

| # | 阻塞 | 事实 |
|:--|:--|:--|
| 1 | **交叉工具链路径已变** | `build_lib.sh` / `make_build_dir.sh` 里写死 `/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/…`，该卷**已改挂**为 `/run/media/manu/3061aec8-…/rk3588/rk3588_cross_toolchain/…`（工具链**在**，gcc 9.3.0 可用）⇒ 脚本里的默认路径**在本机是死的**，需用 `CROSS_CXX=` / `RK3588_TOOLCHAIN=` 覆盖 |
| 2 | **缺 OpenCV 4.6 的 aarch64 头与库** | 交付 `.so` 链 `libopencv_*.so.406`（OpenCV 4.6）；本机只有 `opencv-4.10.0-arm64-install`（**4.10**，且只有静态 `.a`）⇒ 头与运行库**必须同版本**（4.10 的头配 4.6 的 `.so` 会破 ABI）。取 4.6 头/库要 `fetch_board_libs.sh`（**需要网络/板卡**） |
| 3 | **缺板卡运行时库 + 沙箱禁网** | `librknnrt.so` / `librga.so.2` 不在本机；本沙箱 `socket: Operation not permitted` ⇒ 板卡 `192.168.0.64` / 服务器 `192.168.99.40:32222` **都连不上**（`fetch_board_libs.sh` / 板上原生编都走网络） |

⇒ **`.so` 的重编必须落在板卡（原生，OpenCV 4.8 ⇒ ocv48 变体）或 1TB-Volume 服务器（ocv46/ocv42）**，本机做不了。
**探针**则本机可交叉编（零第三方依赖），但**必须等 `.so` 先重编**（链接硬依赖）。

### 3.4 重编 + 重新打包的完整顺序（铁律九：可复制）

```bash
# ① 板卡上原生重编 .so（ocv48 变体；板卡自带 OpenCV 4.8 ⇒ 产出链 .so.408）
#    —— 需要 SSH 到 192.168.0.64（**本沙箱禁网** ⇒ 需授权）
cd /root/sdk && bash build_lib.sh          # 产出 /root/build/libmosaic_pipeline.so

# ② 用**新** .so 重编两个探针（零第三方依赖 ⇒ 三变体通用）
cd <包根> && bash <repo>/manu/pipeline/npu/packaging/build_probes.sh
#    期望：'[ok] .so 导出 mosaic_pipeline_set_candidate_dedup_radius' + 两行 '[ok] … 带 --cand-dedup-radius 开关'

# ③ 把新 .so + 探针回灌到三个交付包 + 暂存包 + 包镜像，再重新打包（刷新 MANIFEST/README 的 md5）
cd <repo>/manu/pipeline/npu && MODEL_RKNN=<模型路径> bash packaging/make_build_dir.sh

# ④ 审计回环（**必须绿**）：任何交付件带上修复了吗？
cd <repo>/ultralytics && $PY manu/evaluation/artifact_audit_20261120.py
#    期望：二进制 15 个；含修复 15 个；任何交付件带上修复：是
```

### 3.5 打包脚本本机**可离线跑通**（已验证）

```bash
cd <repo>/manu/pipeline/npu
MODEL_RKNN=<一个 .rknn 副本> bash packaging/make_build_dir.sh
```

实测（在 `/tmp` 的副本上跑的，没动真树）：
拷源码 → 清理陈旧文件（**0 个**）→ 生成 README/MANIFEST → **回验「表内 md5 全部与磁盘一致」「无漏登记文件」**
⇒ 文件数 **80 = 70 + 10（产物）/ 17（src）/ 42（include）/ 3（kernel）/ 1（模型）**。

⚠️ 两个坑：
1. `MODEL_RKNN` 默认指向 **NFS 路径**（本机没挂）⇒ 必覆盖，否则脚本 `die`。
2. 脚本自带警告：**不要在它自己运行期间再调它**（覆盖正在被读取的脚本 ⇒ 后半段静默不执行、退出码还可能是 0）。

### 3.6 交付包文档的两处**不一致**（顺带查出来的）

| 现象 | 位置 | 说明 |
|:--|:--|:--|
| 暂存包 MANIFEST 仍写**修复前**的源码 md5 | `manu/pipeline/npu/build/MANIFEST.md`（12:02） | 与磁盘上的修复版源码**不一致**（4 行不一致）⇒ md5 验收会失败 |
| 包镜像 MANIFEST 写 `models/p0_off_A_none.rknn` | `rk3588_app_base/mosaic_sdk/` | 该路径**不存在**；模型实际在 `runtime/ir_uav_det.rknn`（md5 相同 `1a3faf2f…`）⇒ 按文档去 `models/` 找会扑空（板上脚本用的是 `runtime/` 路径 ✔） |

⇒ 两者都应**重新生成**（改模板 + 重跑打包脚本，**不手改生成物**），并在 §3.4 的 ③/④ 里一并回验。

---

## 四、需要并入主档的段落（**给 `memory.md` / `memory_compact.md` 的拥有者**）

> ⚠️ 本轮**没有**改 `manu/memory/memory.md` 与 `manu/memory/memory_compact.md`：
> 审计期间（14:16~14:24）有**并发线**在写这两个文件（铁律九.1「单一写者纪律」）。

**§265 追加小节（建议文案）**：

> ### 5) 2026-11-20 14:2x~14:3x 独立复检：**修复只在源码里，15 个交付二进制 0 个带修复**
>
> - **复跑**：§265 §2 的验证脚本重跑一遍，5 档半径的 F1/FAR/tp/fp/fn 与每个半径的 `fused_tsv_md5`
>   **与归档 `merge_verify.json` 逐字段相同**；C++ R=0 == 归档 `python_fused_dn1.tsv`、C++ R=20 == 服务器
>   Python 镜像 `fused_r20.tsv`（**逐字节**）⇒ §265 的结论**独立成立**。
>   （附带修掉复现命令的**工作目录 bug**：脚本在 `<ultralytics>/manu/evaluation/`，`cd` 应当是 `<ultralytics>`。）
> - **交付件审计**（新增 `manu/evaluation/artifact_audit_20261120.py`，只读）：
>   **15 个交付二进制、0 个**含合并修复（`.so` 不导出 `mosaic_pipeline_set_candidate_dedup_radius`、
>   探针无 `--cand-dedup-radius`）；**三个交付包的 `src/npu` 全是修复前**源码
>   ⇒ **现在发货 = 板端 FAR 仍 0.3832（红线 12.8 倍）**。
> - **纠正 §265 §4 的一处误判**：探针**已带** 9 字段覆盖（板卡 13:54 源码归档可证）；
>   真正的陈旧点是**合并修复本身**，且 **`.so` 与两个探针必须一起重编**
>   （探针链接期就依赖新符号 ⇒ 实测 `undefined reference`）。
> - **可复现的重编路径**：新增 `manu/pipeline/npu/packaging/build_probes.sh`（此前无任何脚本编探针），
>   含「`.so` 必须先有合并 API」的前置闸门；**修复代码本身已实测可用交付工具链交叉编译为 aarch64**
>   （`nm -C` 见 `T mosaic::MergeCandidatesByRadius(...)`）。
> - **本机编不出 `.so`**：① 工具链路径已改挂（`/media/manu/1TB-Volume` → `/run/media/manu/3061aec8-…`），
>   ② 缺 OpenCV **4.6** 的 aarch64 头/库（本机只有 4.10，版本必须配套），
>   ③ 缺 `librknnrt`/`librga` 且沙箱禁网 ⇒ **`.so` 重编需板卡（ocv48）或 1TB-Volume 服务器（ocv46/42）**。
> - **打包脚本可离线跑通**（`MODEL_RKNN=` 必须覆盖），自带 md5 回验；暂存包 MANIFEST 仍写修复前 md5（4 行不一致）、
>   包镜像 MANIFEST 写了一个**不存在的**模型路径（`models/` vs 实际 `runtime/ir_uav_det.rknn`）⇒ 都要重新生成。

**§267 §7 建议更新**：第 5 条「交付包需要再打一次才能带上 §265 的修复」可补上本轮的量化结论
（**哪些包、哪些二进制、缺什么符号**），并把 §7.2「上游两棵树的 MANIFEST/README 未刷新」
与 §3.6 的两处不一致对上。

---

## 五、本轮**没做**的（都需要当轮授权）

| 项 | 为什么需要授权 |
|:--|:--|
| **板端在线复测**（新 `.so` + 2500 帧 NPU 推理，≈8.5 min） | 属推理（铁律一/十三） |
| **24 序 / 165,223 帧**全量板端 FAR | 需板卡 + 搬帧（§264 §11.5 的服务器→板卡 scp 路径） |
| **SSH 到板卡重编 `.so`** / 取 OpenCV 4.6 头库 | 需网络与板卡访问（本沙箱禁网） |

⇒ 上述三项等用户当轮指定后，本文的 §3.4 就是现成的执行清单。
