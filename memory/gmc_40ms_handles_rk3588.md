# GMC 通道准备流水线 40 ms 硬保证：四个 handle 的板卡落地与三条被证伪的因果

> **2026-10-12 实测。全部数字来自 RK3588 板卡 `evm3588` 实跑，非估算、非投影。**
> 语料：5 序列 × 400 帧 BMP（`board/.bench_bmp`，源帧均 640×512），warmup 100，稳态 300 帧/序列。
>
> 📄 **完整交付对账见 [`../pipeline/opencl/board/GMC_HANDLE_DELIVERY.md`](../pipeline/opencl/board/GMC_HANDLE_DELIVERY.md)
> 与 [`../pipeline/opencl/board/RK3588_PREPROCESS_BENCH_REPORT.md`](../pipeline/opencl/board/RK3588_PREPROCESS_BENCH_REPORT.md)。**
> 本文只记录**下次 Session 必须先知道的结论与坑**。

---

## 1. 结论：40 ms 达成，且在干净系统与满载干扰下都达成

交付配置：`--prealloc-pad 1024 --pin-pad 1` + `--fit-corners 200`（**仅 VIDEO00036**）。

| 目标 | 目标 | 基线（闸门版） | 交付实测 | |
| :--- | :--- | ---: | ---: | :--- |
| **全序列稳态 ≤ 40 ms** | 0 帧超线 | **21 / 1500** | **0 / 1500** | ✅ |
| `VIDEO00034` max | < 25 | **53.50** | **14.35** | ✅ |
| `VIDEO00036` p99 | < 30 | **42.88** | **24.34** | ✅ |
| Ch0 位精确 | — | — | 五序列 **100.0000%** | ✅ |

最坏单帧 **31.85 ms**（余量 8.15 ms）。8 臂 × 5 序列 × 400 帧全部首次跑通，
**无 OOM、无 SIGKILL、无重试**。

---

## 2. 四个 handle 及其触发条件（源码行号为 `cpp/gmc_stream_ocl.cpp`）

| handle | 开关 | 触发条件 | 计数 |
| :--- | :--- | :--- | :--- |
| A 预分配 | `--prealloc-pad N` | `:872-898` 仅当 `want_alloc > alloc_pad_` 才重分配 | `devrealloc` |
| **A′ pad 钉死** | **`--pin-pad 1`** | `:850-896` `bucket = max(bucket, quantise_pad(cap))`，`pad_` 从不增长 | `pad_grew` 恒假 |
| B 角点预算 | `--fit-corners N` | `:510-526` 探针多要 1 个点，**精确判定**而非估计 | `cornercap=Nf/Ncalls` |
| C 墙钟看门狗 | `--fit-deadline-ms F` | `:1388-1420` 每 anchor 前查墙钟，超时整帧退化 | `fitdeadline` |

🔴 **A′ 是本轮新增的，且是让 40 ms 达成的关键。A 单独做不够。**

### 2.1 为什么 A 不够（这是本轮最重要的发现）

`--prealloc-pad` 把 `devrealloc` 五序列全部压到 **1**（原 5/6/6/1/3），
**但 `VIDEO00034` 仍以 43.16 ms 超线**。取证（`board/v34_forensics.py`）：

```
img_up p50 1.56 / p99 16.47 / max 31.06，2..10 ms 区间【一帧都没有】
296/300 帧 < 2 ms，2 帧 > 20 ms；最坏帧仍是 frame 169
```

**离散事件，不是长尾。** 成因写在源码注释里（`:876-879`）：
`pad_` 既是 `copyMakeBorder` 的边框，又是 kernel 采样的内容偏移；
**它一增长，全部 45 个驻留槽位就落到错误偏移而必须整体作废重传**——
*"even when the allocation itself does not change"*。

⇒ **这条路径不经过重分配分支，因此 `devrealloc` 根本不计数。**
`devrealloc=1` 只证明「分配尺寸没变」，**不证明「驻留槽位没被清空」**。

🔴 **教训：一个计数器只覆盖它设计要覆盖的事件。覆盖率没被验证时，
「计数器归零」不能当作「该类开销消失」。**

### 2.2 `--pin-pad` 的代价（实测，不是估算）

`--pin-pad` 让 `img_up` 恒按 pad 1024 计费，全序列统一到 **2.44–2.46 ms** p50。
代价：VIDEO00002 p50 **+1.50 ms**、VIDEO00036 +0.98、VIDEO00034 +0.73；
VIDEO00028/00032 本就在 1024，**在噪声内**。

---

## 3. 板卡噪声底噪（本项目第一个可信的波动基线）

arm D 与 arm H 是**同一二进制、同一配置、跑两遍**：

| 序列 | p50 噪声 | p99 噪声 | max 噪声 |
| :--- | --: | --: | --: |
| VIDEO00034 | 0.19 | 0.37 | 0.55 |
| VIDEO00028 | 0.10 | 0.07 | 0.15 |
| VIDEO00032 | 0.17 | 0.21 | 0.11 |
| VIDEO00002 | 0.50 | 0.44 | 0.49 |
| VIDEO00036 | 0.32 | 0.21 | 0.28 |

**板卡 0.07–0.55 ms；x86 是 3–60 ms。** 这个对比决定了哪些 A/B 差异可以当真。
⇒ **板卡上的 A/B 差异大于 0.55 ms 即可视为效应；x86 上的必须先量噪声再说。**

---

## 4. 三条被证伪的因果/结论（已入 `falsified_archive.md`）

### 4.1 🔴「`VIDEO00034` 的 53.50 ms 尖峰是设备重分配造成的」——错

原始反事实预测「减去上传超额 ⇒ max 53.50 → ≈13.0 ms」，
理由是 `devrealloc=5` 与稳态内 4 帧 `img_up>10 ms` 数量吻合。
**板卡实跑证伪**：`devrealloc` 降到 1 之后，尖峰**仍在同一帧 169**。
真因是 `pad_` 增长导致的环形作废（§2.1），两者都能解释「数量吻合」，
**相关不等于因果**。⇒ 详见 `falsified_archive.md` 第 38 节。

### 4.2 🔴「按分配尺寸上传会让 H2D 放大 32×，是笔永久税」——被总线饱和推翻

源码原注释（`:867-871`）据此否决了 `--pin-pad`。板卡实测：

| pad | 上传字节 | img_up p50 |
| --: | --: | --: |
| 32 | 0.405 Mpixel | **0.37 ms** |
| 512 | 2.555 Mpixel | **1.56 ms** |
| 1024 | 6.881 Mpixel | **2.45 ms** |

**17× 的字节只换来 6.6× 的时间**——H2D 带宽饱和、固定开销占主导。
所谓「永久税」是 **~2 ms**。**只有板卡能给总线定价，x86 给不了。**

### 4.3 🔴「x86 的 A/B 时延差可用于判断 handle 效果」——x86 时延双重不可用

- **设备不对**：x86 的 OpenCL 是 **PoCL 跑在 Intel i5-10500 CPU 上**，
  且像素格式从板卡的 `CL_R`(1 B/px) 退化成 `CL_RGBA`(4 B/px)。每帧 110–245 ms。
- **噪声淹没信号**：同配置两跑 p99 波动 **3–60 ms**（相对 15%–60%），而 A-B 差异只有 1.9–63 ms。

⇒ x86 只保留**计数类**与**位级精度类**结论；时延一律只报板卡实测。

---

## 5. 两条方法论教训（写给下次的自己）

### 5.1 「零观测」必须显式区分于「零事件」

本轮开局时 `.bench_out/` 全树 `grep cornercap` 命中数为 **0**——
报告里的触发数**无法从盘上任何产物回查**。补救办法不是「再跑一次」，
而是**把产物留下来 + 用脚本出表**（`board/x86_delivery_tables.py`）。
今后：**凡是报告里出现的数字，其原始产物必须留在盘上且有解析脚本。**

### 5.2 🔴 md5 相同是充分条件，**不是必要条件**

源码里改了一句**注释**（把「32×」换成实测值），
重建后二进制 md5 变了——`.text` 有 **30 字节位移**（分支位移字段），
而**构建本身是确定性的**（同一源码两次构建 md5 相同）。
但**输出逐位不变**（零回归全过，板卡复跑 0 超线）。

⇒ **交付结论必须由零回归与复跑给出，不能由 md5 给出。**
md5 用来回答「板子上跑的是不是这份源码」，不回答「行为是否相同」。

### 5.3 分析器的守卫（本轮新增 3 条，总计 8 条）

- 无数据 ≠ 零；CSV 短于 warmup ≠ 无数据；计数未解析 ≠ 零；A/B 未解析 ≠ 零；
  残缺 CSV 必须标 `PARTIAL`；**配置未声明时拒绝给触发数**。
- 🔴 **合成/干跑数据必须隔离并打哨兵**：`guarantee_report.py` 见到
  `SYNTHETIC_DO_NOT_REPORT.txt` 直接拒报并退出码 2。
  起因是我第一版把合成帧写进了 `.bench_out/G{C,D,E}`——**正是真实板卡数据的落地目录**，
  fetch 一失败就会被当成 RK3588 实测报出去。
- 🔴 **崩溃点会吃掉后面所有表**：`load_seq(path)` 少传一个参数，
  `TypeError` 恰好发生在 TABLE 1 之后、TABLE 2 之前，
  于是**触发普查与精度账两张核心表永远打不出来**，而 `set -e` 让脚本以非零码退出、
  看起来像全盘失败。**装饰性子表里的参数缺漏，能藏住全部交付数字。**

---

## 6. 板卡访问（2026-10-12 已打通）

`ssh -p 22 root@192.168.0.64`，口令**只经 `RK3588_PASSWORD` 环境变量传入，不落盘、不回显**。
本机无 `sshpass`，用 `board/board_run.py`（pexpect）与 `board/fetch_results.py`（scp+pexpect）。

🔴 **禁止 base64-over-ssh**（pty 回显会填满缓冲并卡死会话）——用 scp。
🔴 内联多层引号的 `--exec` 会卡会话——改用脚本文件传输。

⚠️ **2026-10-12 之前的「无法登录」是真实阻塞**：四条认证路径全部走完且失败
（环境变量缺失 / 私钥被拒且板侧 `authorized_keys` 0 字节 / 无 `.netrc` 与 `~/.ssh/config` /
adb 5555·telnet 23·http 80 均不可达）。**不要在没拿到口令时反复重试。**

---

## 7. 复现

```bash
cd manu/pipeline/opencl/board

# 完整 campaign（部署+md5 双侧校验 → arm C/D/E → 回传 → 四张表）
RK3588_PASSWORD=<口令> bash guarantee_run.sh

# --pin-pad 的板卡验证（arm F/G/H）+ 板卡噪声底噪
RK3588_PASSWORD=<口令> bash pin_pad_board.sh
python3 board_noise_floor.py

# 后台抢占下的 40 ms 保证（实验 B）
RK3588_PASSWORD=<口令> bash contended_board.sh

# x86 零回归（双臂：CPU 逐帧 MD5 + GPU A/B）
bash zero_regression_pin.sh

# x86 触发普查 / 精度 / 配置敏感性
bash x86_evidence_regen.sh census
bash corner_budget_full.sh VIDEO00034_19700101_015052 VIDEO00028_19700101_013631 \
     VIDEO00032_19700101_014453 VIDEO00002_19700101_001415 VIDEO00036_19700101_015740
bash corner_config_sensitivity.sh
python3 x86_delivery_tables.py
```

产物：`board/.bench_out/{GC,GD,GE,GF,GG,GH,GI,GJB}/`（板卡）、`board/.bench_out/REGEN{,_REP}/`（x86）。
交付表：`BOARD_FINAL_TABLES.txt`、`CONTENDED_TABLES.txt`、`BOARD_NOISE_FLOOR.txt`、`V34_FORENSICS.txt`。

---

## 8. 遗留未解（**不得读作「没问题」**）

1. **业务 F1 影响未测**：闸门触发、以及 `VIDEO00002` 的既有 600 角点截断
   （板卡实测 **398/398 帧全部顶满**，x86 未截尾真值 **1056.6**）对 Trial 0474 检测 F1 的定量影响。
   **这需要检测指标输出，不在时延交付范围内。** 这是修复上线前最该补的一项。
2. **语料固定 400 帧/序列**，工作区无更多源帧 ⇒ **>400 帧长扫描未做**。
   缓解理由：`--pin-pad` 使 `img_up` 成本**结构上恒定**（`pad_` 被 clamp 在 cap 内，永不增长），
   这比「400 帧内没发生」更强；但仍是未实测项。
3. **温度对照未做**（单次运行，未做冷热对照）。
4. **page cache 冷/热两态未主动对照**。
5. **多线程下 RANSAC/LK 的位可复现性**在 `gmc_ocl_board_run.md` §7 仍是「未验证的担心」，
   本轮未触及；若要放宽等位精确要求，须先补 Python 侧多线程对账。
6. **单像素舍入抖动根因未定位**（Ch2 第 41 帧那 1 个像素走 `max(0, I-B)` 后 `astype(uint8)` 的舍入）。