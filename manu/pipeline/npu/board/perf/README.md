# 板卡满频锁（RK3588S / Orange Pi CM5）

**适用板卡**：`root@192.168.0.211`（Orange Pi CM5，Debian 12，内核 `5.10.160-rockchip-rk3588`）

**目标**：CPU / GPU / NPU / DMC 四个域在任何时刻都以最大频率工作。

---

## 1. 结论先讲：降频有两个执行者，只有一个能治

| # | 执行者 | 触发条件 | 钳到哪里 | 谁能改 |
|:--|:--|:--|:--|:--|
| ① | **cpufreq / devfreq governor** | 一直（默认 `ondemand` / `simple_ondemand`） | 各自 `min_freq` | ✅ 用户态，**本目录解决** |
| ② | **Rockchip PVTM 高温降额**（设备树 OPP 表） | **SoC ≥ 85 °C** | A55→1608 MHz、A76→2208 MHz、GPU→800 MHz、NPU→800 MHz | ❌ **设备树常量，只能改 DTB** |

> 🔴 **抬 `thermal_zone0` 的 passive trip point 挡不住 ②**。
> trip point 抬到 90/100 °C 之后，**85 °C 的 PVTM 线反而成了更早触发的瓶颈**。

**实测证据**（2026-11-12，真实交付负载 `rknn_mosaic_e2e` 连跑，2 s 采样）：

| 时刻 | SoC °C | bigcore °C | p0max | p4max | gpumax | npumax | cooling device |
|:--|--:|--:|--:|--:|--:|--:|:--|
| 12:19:44 | 83 | 84 | 1800000 | 2400000 | 1000000000 | 1000000000 | 0/0/0/0 |
| 12:19:48 | 84 | **85** | 1800000 | 2400000 | 1000000000 | 1000000000 | 0/0/0/0 |
| **12:19:50** | 83 | 83 | **1608000** | **2208000** | **800000000** | **800000000** | **0/0/0/0** |
| … | | | | | | | 0/0/0/0 |
| **12:20:47** | 80 | 82 | 1800000 | 2400000 | 1000000000 | 1000000000 | 0/0/0/0 |

⇒ **cooling device 全程 `cur_state=0`** ⇒ 温控 governor 从没介入 ⇒ **写频率的一定是 PVTM**。
⇒ 迟滞带：**< 85 °C 触发，> 82 °C 释放**。

---

## 2. 装法

```bash
# 在板卡上（root）
install -m 755 maxfreq-lock.sh  /usr/local/bin/maxfreq-lock.sh
install -m 755 maxfreq-guard.sh /usr/local/bin/maxfreq-guard.sh
install -m 644 maxfreq-lock.service  /etc/systemd/system/maxfreq-lock.service
install -m 644 maxfreq-guard.service /etc/systemd/system/maxfreq-guard.service
install -m 644 maxfreq-guard.timer   /etc/systemd/system/maxfreq-guard.timer

systemctl daemon-reload
systemctl enable --now maxfreq-lock.service
systemctl enable --now maxfreq-guard.timer
```

**为什么要 `After=cpufrequtils.service`**：开机时 `/etc/default/cpufrequtils` 会把 governor 设回
`ondemand`（`GOVERNOR=ondemand MIN_SPEED=408000 MAX_SPEED=2400000`），必须靠 systemd 的排序压过去。

**兜底**：`/etc/rc.local` 末尾再调一次 `/usr/local/bin/maxfreq-lock.sh`（`rc-local.service` 需 `enabled-runtime`）。

---

## 3. 锁的内容

| 域 | governor | max_freq | 满频值 |
|:--|:--|--:|--:|
| CPU policy0 (A55 ×4) | `performance` | `cpuinfo_max_freq` | 1800000 |
| CPU policy4 (A76 ×2) | `performance` | `cpuinfo_max_freq` | 2400000 |
| CPU policy6 (A76 ×2) | `performance` | `cpuinfo_max_freq` | 2400000 |
| GPU `fb000000.gpu` | `performance` | 1000000000 | 1000000000 |
| NPU `fdab0000.npu` | `performance` | 1000000000 | 1000000000 |
| DMC (DDR) | `performance` | 2112000000 | 2112000000 |

**为什么只写 `scaling_max_freq` 就够**：`performance` governor 的语义是**恒取 `policy->max`**，
所以**空载也是满频**（实测 IDLE：p0=1800000 / p4=2400000 / p6=2400000 / gpu=1 GHz / npu=1 GHz）。
`scaling_min_freq` 写成 max 只是加一层保险；PVTM 钳位期间它会与 max 冲突并让后续 sysfs 写报 EIO，
所以脚本**容忍该失败**。

**看门狗 `maxfreq-guard.timer`**（每 10 s）：
- SoC **< 78 °C** ⇒ 重新施加满频锁（防 `cpufrequtils` / 温控 governor 改回去）；
- SoC **≥ 78 °C** ⇒ **只记录、不覆盖** ⇒ **不与 PVTM 抢**（85/82 °C 的迟滞带上方留了 4 °C 缓冲）。

---

## 4. 「power 模式」到底有没有用

板卡上三个同名不同物：

| 名字 | 是什么 | 有用吗 |
|:--|:--|:--|
| `powersave` **governor** | 语义 = **把频率钉在 `min_freq`**（不是智能省电） | ❌ **有害**：单独用会把 CPU 压到 408 MHz、GPU/NPU 压到 300 MHz、DMC 压到 528 MHz |
| `power_allocator` | 温控 governor（决定 cooling device 状态） | 与频率锁正交；本方案里它**从未介入**（全程 `cur_state=0`） |
| `power_policy` | `/sys/devices/platform/fb000000.gpu/power_policy`，**只读状态输出** | ❌ 写 8 个值全被驱动拒绝，**不是可设模式** |

**实测（负载饱和时，governor 名字完全无差别）**：

| governor | e2e fps | TSV md5 |
|:--|--:|:--|
| `performance` | 7.145 | `b1882de090ced61960e42162832de4bf` |
| `powersave` | 7.093 | 同上 |
| `ondemand` | 7.130 | 同上 |

⇒ **governor 只在「空载」时才有区别**（决定停在 min 还是 max）。
⇒ 要「**任何情况都是 max**」，只有 `performance` 能保证空载也满频；其余一律靠 `min_freq` 兜底。

---

## 5. ⚠️ 已知边界：满频的散热代价

| 项 | 实测 |
|:--|:--|
| 满频锁后 SoC 温度 | **76~85 °C**（真实交付负载连续跑，~5 min 顶到 85 °C） |
| 锁之前的 SoC 温度 | 36~59 °C（governor 空转降频） |
| 距 PVTM 拐点余量 | **≈ 0**（85 °C 触发 / 82 °C 释放） |
| 风扇 | 70 °C 起 PWM=255 满速（设备树 `pwm-fan/rockchip,temp-trips`，与 trip point 无关） |

⇒ **「恒定满频」在当前散热条件下不成立**：一旦 SoC 越过 85 °C，PVTM 就会把
A76 压到 2208 MHz、**NPU 压到 800 MHz**（-20 %）。

**三个可选方向**（按风险从低到高）：

1. **空载别钉 GPU/NPU/DMC**（负载下仍到满频）—— 用 devfreq `simple_ondemand` / `rknpu_ondemand` /
   `dmc_ondemand` 且 `max_freq` 仍设满值 ⇒ 空载省掉一块底噪功耗，负载下频率不变。
2. **改设备树抬 `rockchip,high-temp`**（85 → 95/100）—— 需要重编 DTB；
   ⚠️ **超出原厂降额曲线，稳定性优先于 10 % 频率**（交付要求推理结果逐位可复现，
   偶发不稳比降频更糟）。
3. **改善散热**（散热器 / 风扇 / 风道）—— 唯一不牺牲稳定性的根本解。

---

## 6. 验收命令

```bash
# 六个域是否都在 max
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo "$p gov=$(cat $p/scaling_governor) cur=$(cat $p/scaling_cur_freq) max=$(cat $p/scaling_max_freq)"
done
for d in /sys/class/devfreq/*; do
  echo "$d gov=$(cat $d/governor) cur=$(cat $d/cur_freq) max=$(cat $d/max_freq)"
done

# 是不是 PVTM 在压（cooling device 全 0 就是）
for c in /sys/class/thermal/cooling_device*; do
  echo "$(cat $c/type) cur=$(cat $c/cur_state)/$(cat $c/max_state)"
done
cat /sys/class/thermal/thermal_zone0/temp

# PVTM 常量（设备树）
for t in cluster0 cluster1 cluster2 gpu npu; do
  d=/sys/firmware/devicetree/base/$t-opp-table
  echo "$t high-temp=$(od -An -tu4 --endian=big $d/rockchip,high-temp) \
ceiling=$(od -An -tu4 --endian=big $d/rockchip,high-temp-max-freq)"
done
```
