# 新板（散热改版 Orange Pi CM5）稳温 / 零降频测试

> 执行机：板卡 `192.168.0.211`（Orange Pi CM5，RK3588，散热改版）｜观测主机：本机
> 脚本目录：`manu/pipeline/npu/board/`｜数据目录：`manu/pipeline/npu/board/results/20261112_newboard_thermal/`
> 状态：**第一轮跑到 t=13 min 被 host 关机打断 ⇒ 明天重跑完整 1 h**（本文件已含方法学与已确认结论）

---

## 1. 结论先行（已确认部分）

| 结论 | 证据 |
|:--|:--|
| 新板散热改造后，**CPU 满载 + NPU 满载**下 SoC 稳定在 **69–74 ℃**（均值 72 ℃），**全程 0 次降频** | `results/…/bts_1h_run2_samples.csv`（218 行 / 0 行探针不完整） |
| 全程 **cooling_device 0/0/0/0**（内核温控框架从未出手） | 同上，判据 A |
| CPU0/CPU6、GPU、NPU、DMC **始终 = 允许上限 = 额定值**，无 PM-QoS 钳位 | 同上，判据 B/C2 |
| PVTM **未触发**（上限未被压到额定值以下） | 同上，判据 C |
| dmesg **零温控增量** | `bts_1h_run2_dmesg.txt` |
| **"没降频"不是我自己的锁频脚本造成的假象**：PVTM 走 **PM QoS**（`effective_max = min(用户请求, 热天花板)`），用户态写 `scaling_max_freq` **绕不过去** | `/usr/local/bin/maxfreq-lock.sh` 头部注释 + 实测 `scaling_max_freq == cpuinfo_max_freq` |
| 风险点：峰值 **73.9 ℃** 距 thermal_zone0 第一个 trip（**75 ℃**）仅 **1.1 ℃**；距 PVTM 阈值（85 ℃）11.1 ℃。风扇 **恒 255（满速）** | `bts_1h_run2_samples.csv` |

---

## 2. 板卡环境基线（新板，2026-11-12 实测）

| 项 | 值 |
|:--|:--|
| 板卡 / SoC | Orange Pi CM5（`orangepicm5`），RK3588，8 核（4×A55 + 4×A76），7.9 GB RAM |
| 内核 / 系统 | `5.10.160-rockchip-rk3588`（2026-09-16 build），Debian 12 bookworm，`orangepi-release` 1.0.0 |
| IP / machine-id | `192.168.0.211` / `09f263d632ef4a77a3b03a83fee822da` |
| thermal zones | 0 soc-thermal / 1 bigcore0 / 2 bigcore1 / 3 littlecore / 4 center / 5 gpu / 6 npu |
| zone0 trip | **75 ℃ passive**、**85 ℃ passive**、**115 ℃ critical**（其余 zone 无 trip） |
| cooling devices | 0 `thermal-cpufreq-0` max 7、1 `thermal-cpufreq-1` max 13、2 `thermal-cpufreq-2` max 13、3 `thermal-devfreq-0` max 7（实测 cur_state 全 0） |
| 风扇 | `pwm_fan` → `/sys/class/hwmon/hwmon7/pwm1`；设备树 temp-trip 50/55/60/65/70 ℃ ⇒ 70 ℃ 满速；实测恒 **255** |
| CPU 额定 | policy0 **1800000 kHz**（小核簇）、policy4/6 **2400000 kHz**（大核簇），governor 全 `performance` |
| devfreq 额定 | GPU `fb000000.gpu` **1000000000 Hz**、NPU `fdab0000.npu` **1000000000 Hz**、DMC **2112000000 Hz**，governor 全 `performance` |
| PVTM | 驱动 `rockchip-pvtm pvtm@0..4`；`rockchip,high-temp=85000`、hysteresis 5000 ⇒ 天花板 cluster0 **1608000 kHz**、cluster1/2 **2208000 kHz**、gpu **800000 kHz**、npu **800000 kHz** |
| PVTM 生效方式 | **PM QoS**（`effective_max = min(用户请求, 热天花板)`），用户态写频无法绕过 ⇒ 锁频脚本不会掩盖钳位 |
| 板端存储 | eMMC `/dev/mmcblk1p2` 28 G（实测 used 6.2 G / avail 21 G）；**`/tmp` 是 tmpfs（3.9 G）⇒ 板卡重启必丢**，脚本/产物放 `/root/bts/` |
| NFS | `192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs → /mnt/manu`（nfs4, soft, timeo=5, retrans=2）；**不在 fstab ⇒ 板卡每次重启都要手工重挂** |

---

## 3. 监控程序（三件套，全部只读 /sys、/proc）

| 脚本 | 位置 | 作用 |
|:--|:--|:--|
| `board_thermal_monitor.py` | 板端 `/root/bts/` | 板侧采样器：全 thermal zone、全 cooling device、全 cpufreq/devfreq、每分钟打印；支持 `--stop-cmd` 连续确认后自动停负载 |
| `board_host_watch.py` | **主机侧** | 主机观测器：板卡重启也不丢证据（CSV + dmesg 变化检测 + 每分钟关键状态行）；PVTM 判据在此 |
| `board_thermal_report.py` | 主机侧 | 从 CSV 直接生成报告表格（温度统计 / 四项降频判据 / 每分钟表），保证报告数字机器可复现 |
| `board_thermal_load.sh` | 板端 | 负载发生器：`start [duration] [npu_procs] [cpu_stress] [gpu]`；`STRESS_CORES` 可覆盖 stress 核数 |
| `board_thermal_setup.sh` | 板端 | 板卡重启后重建环境（重挂 NFS、恢复 /root/bts、频率锁）；`COPY_FRAMES=1` 才拷帧（默认不拷） |

### 降频判据（四条，缺一不可）

- **A. cooling 动作**：任一 `cooling_device*/cur_state > 0`
- **B. 频率低于上限**：任一 cpufreq `scaling_cur_freq < scaling_max_freq`、任一 devfreq `cur_freq < max_freq`
- **C. PVTM 钳位**：`scaling_max_freq < cpuinfo_max_freq`、devfreq `max_freq < 额定`
- **D. 内核事件**：dmesg 出现新的 thermal / throttle / cooling / pvtm 行

---

## 4. 负载设计（为什么这么压）

| 域 | 负载 | 说明 |
|:--|:--|:--|
| NPU | **1 个 `mosaic_async_client_example`**（真实交付负载：rknn 推理 + GMC + 拼帧，push-all / 25 fps / inflight 4） | 帧直接读 NFS，**不做任何文件拷贝**（用户明确要求） |
| CPU | `stress --cpu 8`（纯用户态计算，模拟系统侧其它业务） | |
| GPU | **不单独压** | 真实业务里 GPU 负载只有 GMC 的 fit，约 1–4 %（实测 `fb000000.gpu/utilisation` 0–4 %），用户已同意 |

---

## 5. 🔴 附带发现：**硬复位根因**（2 客户端 + 8 核满压）

| 组合 | 结果 |
|:--|:--|
| 1 客户端 + `stress --cpu 8` | **稳定**（t=13 min，72 ℃，0 降频） |
| 2 客户端 + `stress --cpu 8` | **两次硬复位**：17:30:00、17:38:15（相隔 8 min，模式一致） |
| 2 客户端不带 stress | 稳定（288 s，63.8 ℃） |

**排除项**：温度（复位瞬间 SoC 仅 46–52 ℃）｜PVTM（cdev 全 0、频率全满）｜内核 panic（`/sys/fs/pstore` 空、dmesg 无 oops/panic）｜看门狗（板上无可用 watchdog 设备）｜手动 reboot（会话日志无 reboot 指令，wtmp 今日仅 12:42 一次正常关机）｜客户端自身崩溃（两个客户端日志都停在**启动阶段**，一帧未跑完）。

**复位前最后快照**（17:38:13，距死亡 2 s）：SoC 52.7 ℃ / GPU 47.2 ℃ / NPU 49.9 ℃、CPU0 1800 MHz、CPU6 2400 MHz、NPU 1 GHz load 100 %、GPU util 17 %、cdev 0/0/0/0、fan 刚起 50、load1 6.19 ⇒ **stress 一起就死**。

**待判别的两个机理**（明天用 T1/T2 区分）：
- **(a) 功耗/PMIC 跌落**：8 核满载 + 2 份 NPU/RGA/GMC 负载超电源预算 ⇒ 无内核信息的静默复位。
- **(b) 核绑定抢占**：管线把 pre/post 钉在 **0-3**、NPU 钉在 **6-7**、GMC 钉在 **4-7**，两个实例**重复钉同一批核**；叠加 stress ⇒ RGA/NPU 服务线程被饿死，驱动锁竞争自旋 ⇒ 复位。

**判别实验**（各约 5 min，明天跑）：
- **T1** = 2 客户端 + `STRESS_CORES=4 stress --cpu 4`（给 4-7 留管线核）→ 不复位 = (b)；仍复位 = (a)
- **T2** = 2 客户端 + `stress --cpu 8` 但 CPU 限频 800 MHz（需先 `systemctl stop maxfreq-guard.timer`）→ 不复位 = (a)；仍复位 = (b)

---

## 6. 三个工具坑（明天别再踩）

1. **host 侧 "板卡重启！" 误报**：`board_host_watch.py` 原本每逢会话开头就报一次"重启"（拿持久化的旧 boot_id 比对）⇒ **已修**（首次观测只记基线）。今天真实复位只有 2 次。
2. **`/sys/class/devfreq/fdab0000.npu/load` 空载也读 100** ⇒ 该指标**不能**当"NPU 是否在跑"的依据（此前"NPU 100 %"的论据要打折）。
3. **板卡 `/tmp` 是 tmpfs ⇒ 重启必丢**；NFS 不在 fstab ⇒ **板卡重启后必须手工重挂**：
   `mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2`

---

## 7. 明天开工清单

```bash
# 0) host 开机后确认板卡在线、NFS 已挂（板卡重启过就必须重挂）
/tmp/bssh.sh "mount | grep /mnt/manu || mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2"

# 1) 起负载：1 客户端 + stress --cpu 8，帧走 NFS，4200 s
RUN_DIR=/mnt/manu/bts_1h2 /tmp/bssh.sh "RUN_DIR=/mnt/manu/bts_1h2 /root/bts/board_thermal_load.sh start 4200 1 1 0"

# 2) 起主机侧观测器（每分钟一行，板卡重启也不丢证据）
python3 manu/pipeline/npu/board/board_host_watch.py --duration 4000 --interval 2 \
    --print-interval 60 --out /tmp/bts_1h3 --label newboard_1h_1client_cpu8

# 3) 出报告
python3 manu/pipeline/npu/board/board_thermal_report.py --csv /tmp/bts_1h3_samples.csv \
    --dmesg /tmp/bts_1h3_dmesg.txt --label newboard_1h_1client_cpu8 --per-minute
```

**完成判据**：① 覆盖 ≥3600 s 的有效样本；② 判据 A/B/C/D 四项命中数全为 0；③ SoC 峰值 < 75 ℃（或有明确余量结论）；④ 全程 boot_id 不变（无复位）。
