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
