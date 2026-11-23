# RK3588 板卡换板换 IP（2026-11-10 17:35 实测）

> 用户口述变更并授权核对：`ssh root@192.168.0.211`，密码 `qwe123`。
> 本档是 **2026-11-10 之后**板卡身份与环境事实的唯一权威来源；
> `manu/memory/rules.md` §1b 与 `manu/AGENTS.md` §4 的连接信息表同步指向这里。

---

## 1. 一句话

**板卡换了（新板 ≠ 旧板，不只是改 IP），且新板当前跑不了 rknn 推理、也没挂 NFS。**
旧板 `root@192.168.0.64`（密码 `ematech`）已 **No route to host**，彻底作废。

---

## 2. 身份实测（2026-11-10 17:35，密码 `qwe123` 实测登录通过）

| 项 | 旧板 `192.168.0.64`（**已作废**） | **新板 `192.168.0.211`** |
|:--|:--|:--|
| SSH | `ssh -p 22 root@192.168.0.64` / `ematech` | **`ssh -p 22 root@192.168.0.211` / `qwe123`** |
| hostname | （未记） | **`orangepicm5`** |
| machine-id | （未记） | `09f263d632ef4a77a3b03a83fee822da` |
| 发行版 | "Ubuntu base" arm64（SSH 横幅 `OpenSSH_8.2p1 Ubuntu-4ubuntu0.12`） | **Orange Pi 1.0.0 Bookworm / Debian 12**（`OpenSSH_9.2p1 Debian-2+deb12u2`） |
| 内核 | （未记） | **`5.10.160-rockchip-rk3588`** |
| 架构 / 内存 | arm64 / ~1.7G 级 | arm64 / **7.8Gi** |
| 根盘 | （未记） | `/dev/mmcblk1p2` 28G，**22G 可用**（21% 已用） |
| `librknnrt` | **2.3.0** | **2.2.0**（`c195366594@2024-09-14T12:18:56`） |
| 网口 / IP | end1 `192.168.0.64/24` | end1 **`192.168.0.211/24`** |
| Python | — | 3.11.2 |
| `/root` 内容 | `hm_speed` 边车、`rknn_yolo26n_smoke`、`psearch/`、`kernels/` 等 | **只有 `app/ config/ lib/ storage/` + 两个 `linux-*-legacy-rockchip-rk3588_1.0.0_arm64.deb`** |
| `/mnt/manu` | 已挂 NFS（880G，来自 host） | **存在但为空目录，未挂** |
| `/dev/rknpu*` | 有 | **缺失** |
| 设备树 / dmesg | — | 有 `npu@fdab0000`；dmesg 有 `RKNPU fdab0000.npu: pvtm=860` 等初始化日志 |

⇒ **新板是另一套镜像**（Debian 12 + Orange Pi 内核 + librknnrt 2.2.0），**不是**旧板原地改 IP。

---

## 3. 🔴 当前两个硬阻塞

### 3.1 NPU 设备节点缺失 ⇒ 跑不了 rknn 推理

```
设备树：/proc/device-tree/npu@fdab0000  ✅ 在
dmesg ：RKNPU fdab0000.npu: pvtm=860   ✅ 驱动初始化过
/dev/ ：ls /dev | grep rknpu            ❌ 无 rknpu 设备节点（只有 rga/input/uinput）
```

⇒ 驱动加载了但**没有创建设备节点** ⇒ `rknn_init` 会失败。
⇒ 需要在板卡侧查 `/dev/rknpu` 的 udev 规则 / `rknpu` 内核模块是否随 `linux-image-legacy-rockchip-rk3588` 装齐。

### 3.2 NFS 挂载没了，且 host 侧导出表是空的

- 板卡 `/mnt/manu` 是空目录（旧的 880G 挂载点没了）。
- **host 侧**：`/etc/exports` 条目**还在**
  ```
  /run/media/manu/SSD960-LOCAL/nfs  192.168.0.0/24(rw,sync,no_subtree_check,insecure,no_root_squash)
  ```
  `nfs-server` **active**，但 **`/proc/fs/nfsd/exports` 是空的** ⇒ 内核导出表没生效。
- ⇒ 板卡挂 NFS 前，host 侧必须先 `sudo exportfs -ra`（host 的 `sudo` **需用户执行**，Agent 无法完成，见铁律一）。
- 板卡侧挂载命令（**未变**，host IP 仍是 `192.168.0.115`）：
  ```bash
  mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2
  ```

---

## 4. 口令登录怎么在 host 上自动化

⚠️ host 上 **`sshpass` 未安装**。用 `SSH_ASKPASS` + `setsid`（OpenSSH 10.2 支持 `SSH_ASKPASS_REQUIRE=force`）：

```bash
cat > /tmp/board_askpass.sh <<'EOF'
#!/bin/sh
echo qwe123
EOF
chmod 700 /tmp/board_askpass.sh

SSH_ASKPASS=/tmp/board_askpass.sh SSH_ASKPASS_REQUIRE=force setsid -w ssh \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  root@192.168.0.211 'hostname; uname -a'
```

⚠️ 该脚本**含明文口令**，只能放 `/tmp`（权限 700），**不得进仓库、不得写进对外汇报材料**。

---

## 5. 🔴 代码库里还有 40 处旧 IP（会直接跑挂）

`manu/pipeline/npu/` 下的板端脚本把旧地址写死了，改板后**必然连不上**：

| 形态 | 代表文件 |
|:--|:--|
| `BOARD=root@192.168.0.64`（硬编码） | `board_speed_search.py`、`nano_search.py`、`ptq/deploy_pipeline_search.sh`、`ptq/board_recover_final.sh`、`ptq/board_recover_and_launch.sh` |
| `BOARD="${RKNPU_BOARD:-root@192.168.0.64}"`（有 env 覆盖） | `run_hm_speed_board.sh`、`run_yolo26n_ref_board.sh`、`build_mosaic_pipeline_arm64.sh`、`packaging/make_build_dir.sh` |
| `MOSAIC_BOARD_HOST` env 默认值 | `ptq/35_mosaic_board_eval.py` |
| 交叉编译时从板卡拉头文件/库 | `build_rknn_mosaic_arm64.sh`、`build_rga_probe_arm64.sh`、`build_rknn_heatmap_arm64.sh` |
| 注释/用法示例里的旧地址 | `run_step10_board.sh`、`run_npu_smoke.sh`、`ptq/adb_shell.py` 等 |

⇒ 大部分是「默认值 + env 可覆盖」形态 ⇒ **一条 `sed` 可全量替换**，但**会同时改掉历史实验文档里的取证命令**。
⇒ 处置口径（待定）：**只改脚本、只改当前生效的连接信息，不改历史实验记录**（历史记录里的旧 IP 是"当时在哪台机上跑的"的事实，改了就失真）。

---

## 6. 复现本档全部事实的命令

```bash
# 旧板：确认已作废
ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.0.64 'hostname'
# → ssh: connect to host 192.168.0.64 port 22: No route to host

# 新板：口令登录 + 身份取证
# （见 §4 的 SSH_ASKPASS 写法）
ssh ... root@192.168.0.211 'hostname; cat /etc/machine-id; uname -a; \
  cat /etc/os-release | head -3; ip -4 -o addr show; \
  ls /usr/lib/librknnrt.so; strings /usr/lib/librknnrt.so | grep "librknnrt version"; \
  ls /dev | grep -iE "rknpu|rga"; ls /root; df -h /'

# host 侧 NFS 导出表
cat /etc/exports; cat /proc/fs/nfsd/exports; systemctl is-active nfs-server
```

---

## 🔄 2026-11-18 更正：本文件「旧板已作废」的结论**已失效**（两块板并存）

2026-11-18 17:05 实测：`192.168.0.64` / `ematech` **重新上线**，身份 = **原始 EVM3588 实验板**
（hostname `evm3588`，Ubuntu 20.04.6，内核 5.10.160，**librknnrt 2.3.0**，8 核 7.9 GB，
`/root` 里 `rknn_mosaic_e2e*` / `mosaic_api_test*` / `psearch/` 全套在，NFS 已挂 `/mnt/manu`）。
**10-30~11-10 之间产生全部 E2E / 精度对齐 golden 的就是这块板。**

⇒ 本文件 §1「旧板 192.168.0.64 已 No route to host，彻底作废」**不再成立**。
⇒ 当前是**两块板并存**：

| 板卡 | IP / 口令 | 身份 | 用途 |
|:--|:--|:--|:--|
| **EVM3588（原始实验板）** | `192.168.0.64` / `ematech` | Ubuntu 20.04.6 / rknnrt **2.3.0** / NPU 走 **DRM `/dev/dri/renderD129`**（**无 `/dev/rknpu`**） | 全部 E2E + golden + 精度对齐基线 |
| **Orange Pi CM5（嵌入式同款）** | `192.168.0.211` / `qwe123` | Debian 12 / rknnrt 2.2.0 | §178 的 RGA Oops 复现与 dma-fd 修复验证 |

⚠️ 两块板**不能混用 golden**：位级 golden 只对 EVM3588 成立（OpenCV 4.10 静态 + buildroot 9.3.0 工具链）。
详见 `memory.md` §179。

---

## 🔄 2026-11-20 更正（13:xx 实测）：`.64` 是**唯一活着的板卡**；`.211` 仍离线；NPU 走 DRM 不走 `/dev/rknpu`

> 本节**只增不改**前文任何内容；与前文冲突处**以本节为准**。
> §1「旧板 `.64` 已 No route to host，彻底作废」已在 2026-11-18 作废过一次；**本节再作废一次，并补上实测数据**。
> 完整取证（板卡 C++ ≡ 主机 C++ ≡ 服务器 Python 逐字节对齐）见 `manu/memory/memory.md` §262 与 `manu/memory/.slim_memory.md` §7。

### 1) 两块板当前状态（2026-11-20 13:xx 实测）

| 板卡 | IP | 身份 | 状态 | 关键环境（实测） |
|:--|:--|:--|:--|:--|
| **EVM3588（原始实验板）** | `192.168.0.64` | hostname **`evm3588`**，machine-id `27a1db91af4746b0a2b8ba93b052c60c` | ✅ **在线（当前唯一可用）** | Ubuntu **20.04.6 LTS** / kernel `5.10.160` (rockchip-rk3588) / aarch64 / **8 核 / 7.9 G**；**OpenCV 4.8**（`/root/mosaic/opencv/libopencv_*.so.408`）；librknnrt **2.3.0**（`c949ad889d@2024-11-07`）；**NPU 节点 = `/dev/dri/renderD129`**；根分区 16 G、**95% 满**（剩 <1 G）、`/tmp` tmpfs **3.9 G（剩 3.1 G）** |
| **Orange Pi CM5（嵌入式同款）** | `192.168.0.211` | hostname `orangepicm5`（2026-11-10 记录） | ❌ **仍 `No route to host`（离线）** | 2026-11-20 复测不可达 ⇒ OpenCV **4.6** / `libopencv_*.so.406` / librknnrt 2.2.0 那一套**当前完全不可验** |

### 2) 🔴 更正一：§3.1「NPU 设备节点缺失 ⇒ 跑不了 rknn 推理」**对 `.64` 不成立**

§3.1 那一节描述的是 **`.211`**（Orange Pi CM5）。**`.64` 完全不同**：

```text
dmesg   : [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1   ✅
/dev/dri/: card0  card1  renderD128  renderD129                                 ✅
           minor 1 (=128+1) ⇒ NPU 设备节点就是 /dev/dri/renderD129
```

- RKNPU 驱动**内建在内核**（不是可加载模块），且注册在 **DRM** 名下 ⇒ **没有 `/dev/rknpu*` 是正常现象，不是故障**。
- 实证：`.64` 上整条流水线**真跑通了 2500 帧 NPU 推理**（板端日志原文）：
  ```
  [i] rknn runtime api=2.3.0 (c949ad889d@2024-11-07T11:35:33) driver=0.9.8
  [i] mosaic_pipeline created: model=/root/mosaic/runtime/ir_uav_det.rknn ...
  [i] RGA native-tile self-check: OK
  pipeline_frames=2500 wall_ms=514845.4 fps=4.856 dropped=0 features_dropped=0
  ```
- ⇒ **板端复测不是「做不了」，是「没人去跑」**。

### 3) ⚠️ 更正二：OpenCV 版本口径 —— 「只发 `.so.406` 那份」的 Guidance **当前不适用于 `.64`**

- `.64` 自带的 OpenCV 是 **4.8**（`.so.408`，位于 **`/root/mosaic/opencv/`**；**系统 `/usr/lib` 下没有** `libopencv_core.so.*`）
  ⇒ `trial0080_board_fusion_preset_20261119.md` 里「板卡链接 OpenCV 4.6 ⇒ 只发 `.so.406` 那份」的 Guidance **对 `.64` 当前不适用**。
- OpenCV **4.6**（`.so.406`）是 **`.211`** 那一套 ⇒ **`.so.406` 变体当前无处可验**（对应板未上电）。
- 取证时务必按 `/root/mosaic/opencv/` 取路径，不要写 `/usr/lib/libopencv_core.so.*`（那会取不到）。

### 4) 🟢 更正三：NFS 硬阻塞**有绕过路径**（不必等 `exportfs -ra`）

| 方向 | 2026-11-20 实测 | 含义 |
|:--|:--|:--|
| 板卡 → 服务器 `192.168.99.40:32222` | ✅ **TCP 可达** | **帧可由服务器直接 scp 到板卡** ⇒ **不必等 host 的 NFS 导出表恢复** |
| 板卡 → host `192.168.0.115:22` | ❌ `Connection refused` | 板卡不能直接拉 host 上的文件 |
| host `/proc/fs/nfsd/exports` | ❌ **不存在**（`/proc/fs/nfsd` 挂成 `ro`，导出表未生效） | NFS 硬阻塞**依然存在** |
| host `/etc/exports` | ✅ 条目已在：`/run/media/manu/SSD960-LOCAL/nfs *(rw,sync,no_root_squash)`（2026-11-20 11:49） | 条目在但**内核导出表没生效** |
| host 导出盘 `/run/media/manu/SSD960-LOCAL` | ⚠️ 当前 **ro**（`ro,nosuid,nodev,relatime,errors=remount-ro`） | 需 `sudo mount -o remount,rw /run/media/manu/SSD960-LOCAL` |

⇒ 恢复 NFS 仍需**用户 sudo**：`sudo exportfs -ra` + `sudo mount -o remount,rw /run/media/manu/SSD960-LOCAL`（铁律一：Agent 无法自行提权）；
⇒ 但**板卡端跑评测不必等这一步** —— 走「服务器 → scp → 板卡」即可。

### 5) 结论

- **板卡身份以本节为准**：当前**唯一可用的板卡是 `192.168.0.64`**（EVM3588 / OpenCV **4.8** / librknnrt 2.3.0 / **NPU 可用**）；`192.168.0.211`（Orange Pi CM5）**仍离线**。
- 任何「**板卡不可用 / NPU 不可用 / 必须等 NFS**」的说法，**若指的是 `.64`，一律以本节为准**：NPU 可用；搬帧走服务器 scp 即可，不必等 NFS。
- §2 的身份对照表、§3 的两个硬阻塞，**对 `.64` 均以本节为准**；对 `.211` 仍然成立（它确实离线）。

### 6) 复现命令（2026-11-20 实测，可直接复制执行）

```bash
cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace

# ① 板卡身份 + NPU 设备节点 + OpenCV/librknnrt 版本 + 磁盘（经服务器跳板）
#    ⚠️ 跳板会打印 ssh 的 post-quantum 警告，属正常噪声，不影响结果
bash rk3588_app_base/board_sh.sh 'hostname; cat /etc/machine-id; head -2 /etc/os-release; \
  uname -a; nproc; ls -l /dev/dri/; ls /root/mosaic/opencv/libopencv_core.so.*; \
  strings /usr/lib/librknnrt.so | grep "librknnrt version"; dmesg | grep -i "Initialized rknpu"; \
  df -h / /tmp | tail -2'
# → evm3588 / 27a1db91af4746b0a2b8ba93b052c60c / Ubuntu 20.04.6 LTS / 5.10.160 aarch64 / 8 核
# → /dev/dri/renderD129 存在；libopencv_core.so.408（OpenCV 4.8）
# → librknnrt version: 2.3.0 (c949ad889d@2024-11-07)
# → [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1
# → / 16G 95%（剩 <1 G，随写入波动）；/tmp 3.9G 22% 3.1G

# ② .211 仍离线（No route to host）
timeout 5 bash -c 'cat < /dev/null > /dev/tcp/192.168.0.211/22' || echo "No route to host（.211 离线）"

# ③ 板卡 → 服务器 / host 的 TCP 可达性（NFS 绕过路径的根据）
bash rk3588_app_base/board_sh.sh 'timeout 5 bash -c "cat < /dev/null > /dev/tcp/192.168.99.40/32222" && echo SERVER_TCP_OK; \
  timeout 5 bash -c "cat < /dev/null > /dev/tcp/192.168.0.115/22" || echo HOST_TCP_REFUSED'

# ④ host 侧 NFS 现状（导出条目在 / 内核导出表未生效 / 导出盘 ro）
cat /etc/exports; mount | grep nfsd; findmnt -no OPTIONS /run/media/manu/SSD960-LOCAL

# ⑤ 板端运行时读回 9 个交付字段（交付参数组 cc2_md10）
bash rk3588_app_base/board_sh.sh 'grep -h "track fusion ON" /root/verifyB4_20261120_133017.log'
# → [i] track fusion ON: th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 \
#     match_dist=10.00 tags=0x1 reorder_window=0 physical_frame_id=1 max_age_phys=-1 \
#     r_slope=5.00 alpha_decay=1.000 confirm_consec=2 confirm_gap_tol=8
```
