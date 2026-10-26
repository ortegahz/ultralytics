# 当前 Goal（已暂停）· RK3588 PTQ 精度提升

> 状态：**⏸️ 已暂停**（2026-10-23，用户指令：停止所有 agent 与进程，暂停 goal）
> 恢复方式：用户明确说「继续」并指明从哪一步开始。

## 目标

在 RK3588 上提升 trial_0080 热图模型的 int8 量化精度，评测集 = 三个核心 val：
T1 Anti-UAV（31,613 帧 / 25,111 GT）· T2 黄金池（17,187 帧 / 16,639 GT）· T3 001324 盲测（11,810 帧 / 10,535 GT）。

**硬约束**：输入张量必须保持 INT8（`rknn_inputs_set` 0.34 ms；FP16 输入 = 45.41 ms，25 fps 预算爆掉）。
**口径**：同帧 / 同解码器 / 同 GT / 同阈值 / topk=100；量化损失 = F1(PT float) − F1(arm)，逐轨给。
**范围**：只做 PTQ，**不碰 QAT**。

## 已完成（2026-10-23，全部亲自实测）

| 项 | 结论 |
|:--|:--|
| **PTQ 轴判决表** | ❌ 已证伪/平台死路：`optimization_level` 0/1/2/3（与 base 逐位相同）、`enable_flash_attention`（逐位相同）、`quantized_method=layer`（更差，cos 0.9679 vs 0.9731）、`group{SIZE}` / `w8a16` / `gdq`（toolkit 参数校验拒绝）、`kl_divergence`（4 校准集 F1 全 0） |
| | ✅ **校准集是唯一确认有效的轴**：mse 跨集 1.67e-06 ~ 3.57e-06（差 2.1×） —— 🔴 **（2026-10-23 下午纠正）** 这是**张量层**差异，**不是端到端收益**；校准集轴**不是一条能单独兑现的杠杆**（**唯一被全量验证的干净集 set_6 上** T2 持平、T3 −0.0064），**set_4 / set_5 的子集收益尚未全量验证** |
| **输出量程真相** | 🔴 修正旧认知：冠军臂 out[0] scale=0.00378871 ⇒ 上限 **0.9661**；校准集实测峰 0.9732 → 0.9732/255=0.003817 ✓。**无削顶**。旧的 `scale=0.00094072` 只是 `kl_divergence` 的产物。量化损失真因 = **中间层逐张量激活 scale 被离群激活撑大 ⇒ 噪声地板抬高** |
| **方法论陷阱** | **cos 更高 ≠ F1 更高**（100 帧小样本）：layer cos 最低但 F1 第二高；hyb1 cos 最高但 F1 与 base 相同。⇒ 100 帧 F1 噪声 ≥0.05 ⇒ **cos/mse 排序、F1 只做最终验证** |
| **候选臂（100 帧，仅排序参考）** | hyb1 cos 0.989496 / mse 1.080e-06（最好，比 base 好 3.3×）；cand1 mse 1.669e-06；base mse 3.566e-06；cand3 F1 0.7800（最高） |
| **板卡数据通路** | 新增 `manu/tools/board_frames.sh`。**通路 B**：服务器 → 本机 NFS（192.168.0.115，601 G 可用）→ 板卡 `/mnt/manu` **直读**，板卡零磁盘占用（已实测双向可读写） |
| **环境** | 服务器 80 核 / 125 GB / 4×RTX 4090D 空闲；板卡 ctx=0、`/dev/root` 剩 1.4 G、`/userdata` 剩 12 G |
| **入档** | `manu/memory/memory.md` §11.4（三侧 md5 一致）；计划 `manu/docs/experiments/2026-10-23_ptq_plan.md` |

## 待办（按性价比排序，均已就绪可直接执行）

1. **板端验 hybrid 输入 dtype**（≤200 帧，约 10 分钟）—— 决定 hybrid 路线生死。
   已知 `screen_hyb1_diag/p0_off_A_normal_hyb1.rknn` 构建 status=ok、激活精度最好，但输入 dtype 未裁；
   且历史「hybrid ⇒ FLOAT16」结论**建立在失败脚本（`NameError: quant_hybrid_level`）上**，需重验。
2. **大样本三轨筛选**（服务器 PC 模拟器，不占板卡）—— 7 个臂已建好，**直接跑无需重建**：
   `screen_base/p0_off_A_normal`（对照=当前交付）、`screen_v2/p0_off_{1,2,3}_normal`、`screen_n/p0_off_{B,C,D}_normal`。
   同时需 PT float 参照臂（onnxruntime，同帧）以给出量化损失。

   > 🔴 **前提修正（2026-10-23 实测确认）**：**PC 模拟器无法加载已构建的 `.rknn`** ——
   > `init_runtime` 拒绝 `load_rknn`（toolkit 2.3.2，报 `RKNN model that loaded by 'load_rknn' not support inference`）。
   > ⇒ 模拟器评测**必须按配方在进程内重建**（`load_onnx` + `build`），**不能直接吃现成 .rknn**。
   > 仓库既有的 `manu/pipeline/npu/ptq/25_pc_sim_threshold_sweep.py:23-25` 已写明此限制并实现了重建路径 → **直接复用它，不要另写脚本**。
   > ⇒ 「跑现成 7 个臂」的实际含义 = 按各自配方**重建 7 次**（每个约 5~10 分钟），而非加载 7 个文件。
   > ⇒ 另一条替代路径：**板端评测**可以吃现成 .rknn（板端 runtime 接受），但板卡稀缺，只用于最终验收。
3. **新校准集**（按中间层激活离群度分层，而非人工标注）—— 唯一有效轴，值得做。
   前提：先用 ONNX Runtime 做逐层激活剖面，定位离群 top-10 层。

## 不重试清单（勿再开实验）

`optimization_level` · `enable_flash_attention` · `quantized_method=layer` · `group{{SIZE}}` · `w8a16` · `gdq` · `kl_divergence` ·
`uint8 图输入 ONNX`（toolkit `Not Support Dtype: 2`）· 板端 FUSE/sshfs 挂载（内核 5.10 拒 fuse2）· 直接改服务器代码。

## 中止时的现场状态

- 子 agent：全部已停（存活数 0）
- 服务器：我派发窗口（11:15 后）**零进程**；仅剩另一并行会话的 QAT 任务（未动）
- 板卡：ctx = 0，无残留
- 本轮新增文件：`manu/AGENTS.md`、`manu/tools/{agent_fleet.sh,board_frames.sh}`、`manu/docs/experiments/2026-10-23_ptq_plan.md`（均未 commit）
