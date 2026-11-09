#!/usr/bin/env python
"""no-GMC **结构搜索**的训练器外壳（shim）。

存在的理由
----------
``manu/training/mixed_ft_train.py`` 由主控 agent 负责改造（新增 ``--act`` /
``--no-p2`` / ``--head-level`` / ``--head-conv`` / ``--late-upsample``）。
但本搜索线还需要两个**超参杠杆**：

  * ``lr_scale`` —— 路线 B 各参数组基准 lr 的整体倍率。
    上一轮 no-GMC 的**最强旋钮**：+1.6489 T3 / −0.0578 T1（汇率 28.5 pp T3 / 1 pp T1，
    远优于 5.8 的前沿红线）。
  * ``mix_spec`` —— 批内采样口径 1（加权采样 P(LQ)=1/(1+r)）。
    ⚠️ 已知是**负杠杆**（−0.8073 T3 / +0.0436 T1）⇒ 本线只测 w2.0 / w2.5 两档。

如果训练器**原生**支持 ``--lr-scale`` / ``--mix-spec``（主控已加），本 shim 直接透传；
否则用**运行时打补丁**的方式补上（不改训练器源码，避免与主控并发编辑冲突）：

  * ``lr_scale`` → 改写 ``mixed_ft_train.GROUP_SPECS`` 里各组的基准 lr
    （``build_param_groups`` / ``assign_group`` 在**调用时**解析模块全局 ⇒ 补丁有效）。
  * ``mix_spec`` → 在 ``build_mixed_loader`` 调用窗口内，把
    ``torch.utils.data.dataloader.RandomSampler`` 换成 ``WeightedRandomSampler``。
    **官方 ``build_yolo_dataset`` / ``build_dataloader`` 全程不动**，
    与 ``optuna_mix_ratio_search.py`` 的口径 1 **逐位一致**
    （见该文件 §"加权采样器" 的说明：只换"抽哪些下标"，batch_idx 语义不变）。

G1 解冻兜底
-----------
``--act relu`` 会把 **b0..b4（官方底模部分）** 的 SiLU 全换成 ReLU ⇒
冻结底模的语义被改变，**冻结着就学不回来**。若训练器**没有**自带解冻机制，
本 shim 会把 G1 提到可训练组（lr 取 G2 基准）并取消 ``set_frozen_eval`` 的 eval 保护。
⚠️ 这是**改变底模语义**的重大决定，日志里会显著标注。

用法（一般不由人手敲，dispatcher 调用）::

    STRUCT_LR_SCALE=3.0 STRUCT_MIX_SPEC=w2.5 \
    python manu/training/nogmc_struct_train_shim.py --out-dir ... --act relu --no-p2 ...
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import WeightedRandomSampler  # noqa: E402


# ==============================================================================
# 训练器能力探测（读源码文本，避免 import 期副作用）
# ==============================================================================
def _trainer_text(trainer_path: Path) -> str:
    try:
        return trainer_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _has_flag(text: str, flag: str) -> bool:
    return f'"{flag}"' in text or f"'{flag}'" in text


def _has_unfreeze(text: str) -> bool:
    """训练器是否自带 G1 解冻机制（任一标记即可）。"""
    markers = ("G1_unfrozen", "unfreeze", "unfreeze_backbone", "--no-freeze", "G1_trainable")
    return any(m in text for m in markers)


# ==============================================================================
# 加权采样器（逐位复刻 optuna_mix_ratio_search.py 的口径 1）
# ==============================================================================
@contextlib.contextmanager
def weighted_sampler_patch(weights: np.ndarray, num_samples: int, seed: int):
    """在官方 build_dataloader 调用期间，把索引采样器换成 WeightedRandomSampler。

    torch 1.13 的 ``DataLoader.__init__`` 在 ``sampler is None and shuffle=True`` 时
    使用**模块级** ``RandomSampler(dataset, generator=generator)``。
    我们只在调用窗口内替换这个名字，于是：
      · 数据集仍是官方 ``build_yolo_dataset``（图片/标签扫描、缓存、mosaic 全官方）
      · BatchSampler / collate_fn / worker_init_fn / generator / prefetch 全官方
      · **唯一**变化是"抽哪些下标"由 WeightedRandomSampler 决定
    ⇒ 索引语义（batch_idx）与官方完全一致，不会出现"样本腰斩 + 凭空多假警"。
    """
    import torch.utils.data.dataloader as dl

    original = dl.RandomSampler

    def _factory(dataset, generator=None, **kwargs):  # noqa: ANN001
        g = generator if generator is not None else torch.Generator()
        g.manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=g,
        )

    dl.RandomSampler = _factory
    try:
        yield
    finally:
        dl.RandomSampler = original


# ==============================================================================
# 补丁：mix_spec（加权采样）
# ==============================================================================
def _patch_mix(module) -> None:
    orig_build = module.build_mixed_loader

    def wrapper(args):
        spec = os.environ.get("STRUCT_MIX_SPEC", "natural").strip() or "natural"
        # 先跑原函数（官方管线 + 样本数闸门），拿到 dataset 与样本数
        loader, mixed, n_lq, n_uav = orig_build(args)
        if spec == "natural":
            print("[SHIM-MIX] mix_spec=natural（自然遍历）", flush=True)
            return loader, mixed, n_lq, n_uav
        try:
            ratio = float(spec[1:]) if spec.startswith("w") else float(spec)
        except ValueError:
            print(f"[SHIM-MIX] ⚠️ 无法解析 mix_spec={spec!r}，回落 natural", flush=True)
            return loader, mixed, n_lq, n_uav
        # 口径 1：Batch 内采样概率 P(LQ) = 1/(1+r)，P(UAV) = r/(1+r)
        p_lq = 1.0 / (1.0 + ratio)
        p_uav = 1.0 - p_lq
        weights = np.concatenate(
            [
                np.full(n_lq, p_lq / n_lq, dtype=np.float64),
                np.full(n_uav, p_uav / n_uav, dtype=np.float64),
            ]
        )
        num_samples = n_lq + n_uav  # 每 epoch 总采样量不变 ⇒ 各 trial 算力/耗时可比
        seed = int(getattr(args, "seed", 0))
        with weighted_sampler_patch(weights, num_samples, seed=seed):
            loader = module.build_dataloader(
                mixed, batch=args.batch, workers=args.workers, shuffle=True
            )
        print(
            f"[SHIM-MIX] 口径1 加权采样：mix_spec={spec} P(LQ)={p_lq:.4f} P(UAV)={p_uav:.4f} | "
            f"每 epoch 抽样 {num_samples:,} 次（LQ 期望 {num_samples*p_lq:,.0f} / "
            f"UAV {num_samples*p_uav:,.0f}）",
            flush=True,
        )
        return loader, mixed, n_lq, n_uav

    module.build_mixed_loader = wrapper


# ==============================================================================
# 补丁：lr_scale
# ==============================================================================
def _patch_lr_scale(module, scale: float) -> None:
    module.GROUP_SPECS = [
        (name, pattern, lr * scale, desc) for (name, pattern, lr, desc) in module.GROUP_SPECS
    ]


# ==============================================================================
# 补丁：--act relu 时的 G1 解冻兜底
# ==============================================================================
def _patch_g1_unfrozen(module) -> None:
    """把 G1(b0..b4) 从「严格冻结」改成可训练组（lr = G2 基准）。"""
    specs = list(module.GROUP_SPECS)
    if any(n == "G1_unfrozen" for n, _p, _lr, _d in specs):
        return
    base_lr = next((lr for n, _p, lr, _d in specs if n == "G2_p4p5"), 3e-6)
    specs.insert(0, ("G1_unfrozen", r"^b[0-4]\.", base_lr, "G1 b0..b4（relu 结构下解冻）"))
    module.GROUP_SPECS = specs
    # 让 assign_group 不再把 b0..b4 判成 G1_frozen（该全局在调用时解析 ⇒ 补丁有效）
    module.G1_FROZEN_RE = r"^(?!b[0-4]\.).*$"
    # 取消「冻结段强制 eval()」的 BN 保护 —— 既然解冻，BN 统计量就该更新
    module.set_frozen_eval = lambda model: model.train()


def main() -> int:
    argv = sys.argv[1:]
    trainer_flag = os.environ.get("STRUCT_TRAINER", "manu/training/mixed_ft_train.py")
    trainer_path = Path(trainer_flag)
    if not trainer_path.is_absolute():
        trainer_path = PROJECT_ROOT / trainer_path
    text = _trainer_text(trainer_path)
    if not text:
        print(f"[SHIM] ⚠️ 读不到训练器源码 {trainer_path}，按原生能力继续", flush=True)

    native_lr = _has_flag(text, "--lr-scale")
    native_mix = _has_flag(text, "--mix-spec")
    native_unfreeze = _has_unfreeze(text)

    import manu.training.mixed_ft_train as M  # noqa: PLC0415

    lr_scale = float(os.environ.get("STRUCT_LR_SCALE", "1.0") or 1.0)
    mix_spec = os.environ.get("STRUCT_MIX_SPEC", "natural").strip() or "natural"
    act_relu = "--act" in argv and "relu" in argv

    print("=" * 100, flush=True)
    print(f"  [SHIM] 训练器 {trainer_path}", flush=True)
    print(
        f"  [SHIM] 原生能力：lr_scale={'有' if native_lr else '无(打补丁)'} "
        f"mix_spec={'有' if native_mix else '无(打补丁)'} "
        f"G1解冻={'有' if native_unfreeze else '无(兜底打补丁)'}",
        flush=True,
    )
    print(f"  [SHIM] lr_scale={lr_scale} mix_spec={mix_spec} act_relu={act_relu}", flush=True)
    print("=" * 100, flush=True)

    if lr_scale != 1.0 and not native_lr:
        _patch_lr_scale(M, lr_scale)
        print(f"[SHIM] 已应用 lr_scale×{lr_scale} 到 GROUP_SPECS 基准 lr", flush=True)
    elif lr_scale != 1.0:
        # 训练器原生支持 ⇒ 直接透传（shim 不再打补丁，避免双重生效）
        if "--lr-scale" not in argv:
            argv += ["--lr-scale", str(lr_scale)]
        print(f"[SHIM] 训练器原生支持 --lr-scale ⇒ 透传 {lr_scale}", flush=True)

    if mix_spec != "natural" and not native_mix:
        _patch_mix(M)
        print("[SHIM] 已安装 mix_spec 加权采样补丁", flush=True)
    elif mix_spec != "natural":
        if "--mix-spec" not in argv:
            argv += ["--mix-spec", mix_spec]
        print(f"[SHIM] 训练器原生支持 --mix-spec ⇒ 透传 {mix_spec}", flush=True)

    if act_relu and not native_unfreeze:
        _patch_g1_unfrozen(M)
        print(
            "[SHIM] ⚠️⚠️ --act relu 且训练器无 G1 解冻机制 ⇒ 已兜底解冻 G1(b0..b4)。\n"
            "        这**改变了冻结底模的语义**（SiLU→ReLU 后底模必须能再学）——"
            "训练结果只能与同口径的新结构臂比较，**不能**与旧结构的 cs_t0043 直接比权重。",
            flush=True,
        )

    sys.argv = [str(trainer_path)] + argv
    return int(M.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
