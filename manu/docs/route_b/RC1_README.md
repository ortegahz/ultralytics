# RC1 — 路线 B 全帧对等混训（1:3.09）首轮试跑 · Epoch 3

| 项 | 值 |
|:--|:--|
| 封存权重 | `runs/mixed_ft_route_b/RC1/epoch3.pt` |
| md5 | `a845a95b77ac8762b7d033b64f61425e` |
| sha256 | `dcc6c40f012866f588bfb972451a405a61cf0ae72dcc0c80cbcef2b76d8010d9` |
| 大小 | 21,301,091 bytes |
| 模式 | 0444（只读封存） |
| 源权重 | `runs/mixed_ft_route_b/weights/epoch3.pt` |
| 门禁判定 | **FAIL**（rc=1，veto 失败项：['track1_uav']） |

## 三轨指标（RC1 = Epoch 3）

| 轨道 | F1 | TP | FP | FN | ΔF1 vs 基线 |
|:--|--:|--:|--:|--:|--:|
| Track1 Anti-UAV 官方大盘（峰值 th=0.3） | 0.901918 | 21398 | 941 | 3713 | -0.003818 |
| Track2 黄金池 7 条 | 0.998979 | 16635 | 31 | 3 | +0.010543 |
| Track3 盲测 001324 | 0.871684 | 8495 | 461 | 2040 | +0.034354 |

## 为什么是 Epoch 3 而不是 Epoch 5

- Track1 Anti-UAV 官方大盘：RC1 峰值 FP=941，为 Ep1~Ep5 全场最低（Ep1=1232/Ep2=988/Ep3=941/Ep4=1057/Ep5=1023），比基线 FP=1014 还少 73；Ep4/Ep5 已反弹至 1057/1023，Ep5 F1 跌破 0.901（0.900501）= 源域散架征兆
- Track2 黄金池 7 条：RC1 F1=0.9989791016094163 为全周期峰值（Ep1=0.998769/Ep2=0.998889/Ep3=0.998979/Ep4=0.998859/Ep5=0.998949），FN=3（Ep5=2，差 1 帧，无实质差异），FP=31 为 Ep1~Ep5 最低
- Track3 盲测 001324：RC1 FP=461 为全周期最低（Ep1=589/Ep2=531/Ep3=461/Ep4=517/Ep5=502），净砍 129 个误检；RC1 已完成 Ep5 涨幅的 91.4%（(0.8716843671438099-0.8373308251032032)/(0.8749235474006115-0.8373308251032032)=0.9139）
- Ep3→Ep5 边际收益递减：额外 2 个完整 epoch（约 40 分钟算力）仅换来 Track3 F1 +0.0032，代价是 Track1 再损 0.14pp 且全图虚警由 461 回升到 502 ⇒ 判定为边际收益递减/轻度过拟合

**锁定 3 而非 5**：`epoch5.pt（目标域 F1 最高但三轨虚警全面抬头、源域 F1 跌破 0.901）`

## ⚠️ 必读限定

- 🔴 原始红线 Track1 F1 ≥ 0.905736（去 P0 冻结基线）在 Ep1~Ep5 **全部 5 个 epoch 均未通过**，RC1 自身的 Track1 F1=0.901917808219178 同样判负（veto_failures=[track1_uav]）。RC1 的产生依据是**帕累托最优 + 虚警最低**，不是**通过原始 Track1 红线**。任何把 RC1 表述为『通过三轨门禁』的说法都是错的。
- ⚠️ Track1 峰值门限漂移：基线/Ep1 峰值在 th=0.25，Ep2~Ep5 峰值在 th=0.30。跨 epoch 的 FP/F1 对比是『各模型自身最优门限下』的比较（标准做法），但门限本身右移了 0.05，解读时应连同 delta.negative_bands=7/15 一并看。
- ⚠️ 三轨评测口径天生不对称（Track1 中心点 8px 距离匹配 / Track2-3 框内包含匹配），三轨 F1 **不可横向比较**，只做轨道内纵向对比（详见 memory.md §四十三 §2）。
- ⚠️ Track2 的「FP 增量 ≤ 10」红线的唯一攻击点是序列 000703（基线 FP 4 → Ep5 FP 29，Δ+25），RC1 之后须定点攻它（把 000703 失效机理并入训练集负样本）。

## 后续用法

后续任何进一步微调（冻结 BN、L2-SP 正则、重新配比、加负样本等）**一律以本 RC1 为对账基准**，不再回看 epoch5。比较时三轨分别纵向对比，Track1 需同时看 15 档门限曲线的整体位移（禁止用 --expect-* 基线复现语义，那套参数对任何微调权重都必然 FAIL）。

## 复现命令（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，根目录 `/tmp/pycharm_project_10ae9e2e`）

```bash
runs/mixed_ft_route_b/weights/epoch3.pt → 门禁
bash manu/training/auto_epoch_gate_check.sh runs/mixed_ft_route_b/weights/epoch3.pt 3

# 逐序列归因
python manu/training/analyze_per_sequence.py --run-dir runs/mixed_ft_route_b --epochs 0,1,2,3,4,5 --out-json runs/mixed_ft_route_b/per_sequence_attribution.json
```

登记文件的机读版本：`RC1.json`
