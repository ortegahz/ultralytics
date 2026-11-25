# env
cd /tmp/pycharm_project_10ae9e2e && conda activate uav

# install
pip install -e .

# tune
screen python manu/optuna_parallel_uav.py
tail -f /tmp/pycharm_project_10ae9e2e/runs/optuna_uav_recall_sgpu/logs/trial_0008.log

# train
screen python manu/train_uav.py
screen python manu/train_uav_from_optuna.py

# val
yolo val \
    model=/tmp/pycharm_project_10ae9e2e/runs/detect/train-8/weights/last.pt \
    data=datasets/uav/data.yaml \
    imgsz=640 \
    device=0 \
    plots=True

# pred
yolo predict \
    model=/tmp/pycharm_project_10ae9e2e/runs/detect/train-10/weights/last.pt \
    source=/tmp/pycharm_project_10ae9e2e/datasets/uav/images/val \
    imgsz=640 \
    conf=0.2 \
    iou=0.2 \
    max_det=5 \
    save=True

yolo predict \
    model=/tmp/pycharm_project_10ae9e2e/runs/detect/train-10/weights/best.pt \
    source=/tmp/pycharm_project_10ae9e2e/datasets/uav/images/val \
    imgsz=640 \
    conf=0.001 \
    iou=0.7 \
    max_det=100 \
    save=True \
    save_txt=True \
    save_conf=True \
    project=/tmp/pycharm_project_10ae9e2e/runs/detect \
    name=predict-analysis


# tensorboard
tensorboard \
    --logdir /tmp/pycharm_project_10ae9e2e/runs/detect/train-10 \
    --host 0.0.0.0 \
    --port 6006

# heatmap
python manu/train_uav_heatmap.py \
    --data /mnt/data/siping/datasets/manu/uav/data.yaml \
    --weights yolo26np2.pt \
    --stride 4 \
    --imgsz 640 \
    --epochs 30 \
    --batch 32 \
    --device 2,3 \
    --lr0 0.001 \
    --conf_thresh 0.20 \
    --project runs/heatmap_uav \
    --name uav_gpu23_heatmap

python manu/train_uav_heatmap.py \
    --data /mnt/data/siping/datasets/manu/uav/data.yaml \
    --weights yolo26np2.pt \
    --stride 2 \
    --imgsz 640 \
    --epochs 30 \
    --batch 32 \
    --device 2,3 \
    --lr0 0.001 \
    --project runs/heatmap_uav_s2 \
    --name uav_gpu23_heatmap_stride2

screen -S uav_1280_resume python manu/train_uav_heatmap.py \
    --data /mnt/data/siping/datasets/manu/uav/data.yaml \
    --weights runs/heatmap_uav_s2_1280/uav_stride2_1280_ep70/weights/best_recall.pt \
    --stride 2 \
    --imgsz 1280 \
    --epochs 20 \
    --batch 8 \
    --device 0 \
    --lr0 0.00008 \
    --max_grad_norm 1.0 \
    --project runs/heatmap_uav_s2_1280 \
    --name uav_stride2_1280_from_ep6

python manu/infer_heatmap.py \
    --weights runs/heatmap_uav/uav_gpu23_heatmap/weights/best_recall.pt \
    --source /mnt/data/siping/datasets/manu/uav/images/val \
    --conf 0.25 \
    --device 1 \
    --save-dir runs/heatmap_infer_gt

screen python manu/optuna_parallel_heatmap.py \
    --gpus 1,2 \
    --batch 32 \
    --epochs 10 \
    --n-trials 1024 \
    --weights runs/heatmap_uav_s2/uav_gpu23_heatmap_stride2/weights/best_recall.pt

tail -f runs/optuna_heatmap_stride2_640/logs/trial_0083.log

python manu/diagnose_heatmap_badcases.py \
    --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
    --imgsz 640 \
    --stride 2 \
    --device 2 \
    --conf 0.20 \
    --dist-thresh 4.0 \
    --output-dir runs/badcase_analysis

python manu/eval_distance_tolerances.py \
    --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
    --device 2 \
    --conf 0.20

python manu/eval_adaptive_oks_pck.py \
    --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
    --device 2 \
    --min-radius 4.0 \
    --alpha 0.5

python manu/diagnose_heatmap_badcases.py \
    --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
    --device 2 \
    --dist-thresh 8.0 \
    --conf 0.20

python manu/export_top_missed_sequences.py \
    --csv-file runs/badcase_analysis/fn_missed_analysis.csv \
    --output-dir runs/badcase_analysis/top_missed_sequences \
    --max-per-seq 100 \
    --top-k-seqs 5

python manu/generate_osd_videos.py \
    --cache-file runs/badcase_analysis/inference_cache.pkl \
    --output-dir runs/badcase_analysis/osd_videos \
    --dist-thresh 8.0 \
    --conf 0.20 \
    --fps 25

python manu/stat_recall_with_ensemble.py \
    --heatmap-cache runs/badcase_analysis/inference_cache.pkl \
    --yolo-weights runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt \
    --device 2 \
    --dist-thresh 8.0 \
    --conf-hm 0.20 \
    --conf-yolo 0.20 \
    --size-split 28.0

python manu/make_sequence_video.py \
    --data-root /mnt/data/siping/datasets/manu/anti-uav \
    --seq wg2022_ir_020_split_03 \
    --scale 2.0

python manu/visualize_scr_normalization.py \
    --data-root /mnt/data/siping/datasets/manu/anti-uav \
    --seq wg2022_ir_020_split_03 \
    --lag 3 \
    --output-dir runs/scr_videos

screen python manu/optuna_parallel_3frame_heatmap.py \
    --data /mnt/data/siping/datasets/manu/uav_temporal_3frame/data.yaml \
    --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
    --gpus 0,1,2,3 \
    --n-trials 256 \
    --epochs 10 \
    --batch 32 \
    --stride 2 \
    --output-root runs/optuna_3frame_temporal_10ep

tail -f runs/optuna_3frame_temporal_10ep/logs/trial_0000.log

python3 manu/build_yolo_hybrid_corr_dataset.py \
    --ref-dataset /mnt/data/siping/datasets/manu/uav \
    --raw-root /mnt/data/siping/datasets/manu/anti-uav \
    --output /mnt/data/siping/datasets/manu/uav_hybrid_corr \
    --lag-diff 2 \
    --lag-corr 8

python3 manu/train_uav_heatmap.py \
    --data /mnt/data/siping/datasets/manu/uav_hybrid_corr/data.yaml \
    --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
    --stride 2 \
    --temporal-mode hybrid_corr \
    --device 0,1,2,3 \
    --batch 16 \
    --epochs 20 \
    --project runs/uav_hybrid_corr \
    --name exp_corr_v1

screen python3 manu/optuna_finetune_40ep.py \
    --gpus 0,1,2,3 \
    --epochs 40 \
    --batch 32 \
    --n-trials 64 \
    --output-root runs/optuna_heatmap_stride2_fine40

tail -f /tmp/pycharm_project_10ae9e2e/runs/optuna_heatmap_stride2_fine40/logs/trial_0000.log

python manu/train_irstd_unet.py \
    --data /mnt/data/siping/datasets/manu/uav/data.yaml \
    --device 0,1,2,3 \
    --batch 32 \
    --val_batch 24 \
    --stride 2 \
    --epochs 30 \
    --lr0 0.0003 \
    --project runs/irstd_unet \
    --name uav_unet_stride2

screen python manu/optuna_median_distributed.py \
    --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
    --weights runs/finetune_median/exp_6ep_median/weights/best_f1.pt \
    --gpus 0,1,2,3 \
    --epochs 3 \
    --batch 32 \
    --dist-thresh 8.0 \
    --study-name median_sota_search \
    --project runs/optuna_median_search

tail -f /tmp/pycharm_project_10ae9e2e/runs/optuna_median_search/logs/trial_0000.log

screen python manu/optuna_soft_iou_distributed.py \
    --weights runs/optuna_median_search/trial_0022/weights/best.pt \
    --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
    --gpus 0,1,2,3 \
    --n-trials 1024 \
    --epochs 3 \
    --batch 32 \
    --stride 2 \
    --output-root runs/optuna_soft_iou_search
tail -f /tmp/pycharm_project_10ae9e2e/runs/optuna_soft_iou_search/logs/trial_0001.log
python -c "
import pandas as pd
df = pd.read_csv('runs/optuna_soft_iou_search/optuna_summary.csv')
best = df[df['status'] == 'COMPLETE'].sort_values(by='f1', ascending=False).iloc[0]
print('\n' + '='*50 + ' 🏆 CHAMPION TRIAL ' + '='*50)
for k, v in best.items():
    print(f'{k:<20}: {v}')
print('='*119)
print(f'Best Model Checkpoint: runs/optuna_soft_iou_search/{best[\"trial\"]}/weights/best.pt\n')
"

screen python manu/optuna_p0_nas_distributed.py \
    --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
    --weights runs/optuna_median_search/trial_0022/weights/best.pt \
    --gpus 0,1,2,3 \
    --n-trials 4096 \
    --epochs 3 \
    --batch 32 \
    --output-root runs/optuna_p0_nas
tail -f /tmp/pycharm_project_10ae9e2e/runs/optuna_p0_nas/logs/trial_0000.log
python manu/report_p0_nas_status.py --output-root runs/optuna_p0_nas

screen python manu/optuna_temporal_nas.py \
    --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
    --cache-root /mnt/data/siping/datasets/manu/uav_s2_diff_cache \
    --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
    --output-root runs/optuna_temporal_nas \
    --gpus 0,1,2,3 \
    --n-trials 1000 \
    --epochs 3 \
    --batch 16
tail -f /tmp/pycharm_project_10ae9e2e/runs/optuna_temporal_nas/logs/trial_0000.log

python manu/audit_substandard_cases.py \
    --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
    --dist-thresh 12.0 \
    --th-base 0.22 \
    --th-salvage 0.06 \
    --th-ground 0.35 \
    --min-hits-infill 5 \
    --min-rigid-disp 2.0 \
    --max-rigid-var 0.5 \
    --exclude-avian \
    --exclude-demo

screen python -u manu/training/optuna_p0_nas_distributed.py \
    --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
    --weights runs/optuna_median_search/trial_0022/weights/best.pt \
    --gpus 0,1,2,3 \
    --n-trials 8192 \
    --epochs 3 \
    --batch 32 \
    --imgsz 640 \
    --stride 2 \
    --dist-thresh 8.0 \
    --study-name p0_nas_search_4096 \
    --output-root runs/optuna_p0_nas
python manu/reports/report_p0_nas_status.py

python manu/process_ir_video_sota.py \
  --input "/mnt/data/xxxu/龙泉山/ir/VIDEO00010_19700101_001550.h264" \
  --output "runs/paper_diagnostic_videos/VIDEO00002_001415_sota.mp4" \
  --weights "runs/optuna_p0_nas/trial_0474/weights/best.pt" \
  --device 1

screen python manu/process_ir_videos_parallel.py \
  --input-root "/mnt/data/xxxu/龙泉山/ir" \
  --output-dir "runs/paper_diagnostic_videos/longquanshan" \
  --weights "runs/optuna_p0_nas/trial_0474/weights/best.pt" \
  --gpus 0,1,2,3 \
  --workers-per-gpu 8

python manu/extract_video_frames.py \
  --input-root "/mnt/data/xxxu/龙泉山/ir" \
  --output-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --recursive \
  --format jpg \
  --jpg-quality 95

screen python manu/preannotate_parallel.py \
  --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --output-root "/mnt/data/siping/datasets/manu/龙泉山/preannot_v1" \
  --split train \
  --hm-weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --bbox-weights runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --batch-size 2 \
  --yolo-batch-size 2 \
  --cpu-threads 1 \
  --chunk-size 64 \
  --hm-conf 0.06 \
  --bbox-conf 0.06 \
  --overwrite

python manu/render_yolo_labels_video.py \
  --dataset-root "/mnt/data/siping/datasets/manu/龙泉山/preannot_v1" \
  --split train \
  --seq "VIDEO00004_19700101_002438" \
  --output "runs/preannot_review/VIDEO00004_review.mp4" \
  --fps 25 \
  --scale 2 \
  --show-centers

python manu/diagnostics/probe_heatmap_regions_video.py \
  --features-dir /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train \
  --pattern 'VIDEO00005_19700101_002959__frame_*.jpg' \
  --output-dir runs/heatmap_probe/VIDEO00005_19700101_002959_ds320 \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --spatial-downsample 160 \
  --region-threshold 0.06 \
  --main-threshold 0.22 \
  --min-area 4 \
  --device 0 \
  --fps 25

VIDEO00032_19700101_014453_diagnostic_demo_th0.22.mp4
VIDEO00005_19700101_002959_diagnostic_demo_th0.22.mp4

python manu/diagnostics/probe_heatmap_regions_video.py \
  --features-dir /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train \
  --pattern 'VIDEO00005_19700101_002959__frame_*.jpg' \
  --output-dir runs/heatmap_probe/VIDEO00005_19700101_002959_fusion \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --compare-downsample 160 \
  --fusion-row \
  --fusion-dilate 32 \
  --region-threshold 0.06 \
  --main-threshold 0.22 \
  --min-area 4 \
  --device 0 \
  --fps 25

# issue
1. 悬停目标召回差，特别是复杂背景悬停更差，运动目标相对检出比较稳定。
2. 近距离大目标检出没有中等距离目标检出稳定，感觉还是感受野问题。
3. 路灯和窗户有误检，分数还不低。
# ============================================================================
# 新交付域「纵横AI数据采集」：RC2 中心点 OSD 复核视频 + 序列划分（2026-11-09 新增）
# 详见主档 memory.md §91 与 manu/docs/dataset/zongheng.md
# ============================================================================

# 全量渲染（服务器，4 卡分片，每片一卡；detached 起活）
bash manu/videos/launch_rc2_zongheng_osd.sh

# 单分片冒烟（只渲 150 帧，先验证工作点与画法）
LIMIT=150 SHARDS=1 bash manu/videos/run_rc2_zongheng_osd.sh

# 序列划分（读每帧检测记录 → split 清单 + 人工复核底稿）
python manu/data/build_zongheng_ft_split.py \
  --frames-dir /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109/frames \
  --out manu/data/zongheng_ft_split.json \
  --review-sheet manu/docs/dataset/zongheng-tk.md

# 工作点复扫（CPU-only，不需要 GPU；在真实龙泉山 RC2 缓存上实测 th/radius）
python manu/evaluation/eval_multiscale_cache_th.py \
  --cache-root runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache \
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --tag all --match-mode inbox --merge-radius 768 --th-min 0.30 --th-max 0.55 --th-step 0.02

# ============================================================================
# 纵横AI数据采集：RC2 + 龙泉山时空轨迹后处理档（2026-11-10 新增）
# 详见主档 memory.md §91.10 / §91.11 与 manu/docs/dataset/zongheng.md
# ============================================================================

# 全量渲染（4 卡分片，detached；产物进独立目录，与前两档互不覆盖）
bash manu/videos/launch_rc2_tracked_osd_batch.sh

# 渲染 + 收尾一次性跑（服务器端自动：划分 + 复核底稿 + 预标注 + 核验）
bash manu/videos/run_rc2_tracked_osd_batch.sh

# 单条冒烟（含逐阶段轨迹诊断）
bash manu/videos/run_rc2_tracked_osd_smoke2.sh

# 诊断：是哪一道闸把轨迹剪掉了（逐阶段计数）
python manu/evaluation/diag_zongheng_track_stages.py <ms_cache>/<SEQ>.pkl
python manu/evaluation/diag_zongheng_track_gate.py <ms_cache>/

# 诊断：跟踪档覆盖率损失的三级归因（CPU-only，服务器上跑）
#   1) 逐序列清单：检出帧数 / 轨迹数 / 最长轨迹 / 插补帧，并标出「跟踪后 0 轨迹」的序列
python manu/evaluation/diag_zongheng_track_inventory.py \
  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_th020_20261109/frames \
  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020_20261110/frames \
  /mnt/data/siping/runs/zongheng_rc2_tracked_th020_inventory.json
#   2) 找出「单帧连续 ≥N 帧、跟踪一帧不剩」的段，并用帧内位移区分 static / moving
python manu/evaluation/diag_zongheng_persistent_drop.py \
  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_th020_20261109/frames \
  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020_20261110/frames 7 2.0 \
  /mnt/data/siping/runs/zongheng_rc2_tracked_th020_pdrop.json
#   3) 以龙泉山 v3 的 max_match_dist(=24px) 为界，统计结构性不可关联的目标
python manu/evaluation/diag_zongheng_fast_movers.py \
  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_th020_20261109/frames \
  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020_20261110/frames \
  manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  /mnt/data/siping/runs/zongheng_rc2_tracked_th020_fastmovers.json

# 同阈值逐序列对比（跟踪档 vs 单帧档）
python manu/evaluation/cmp_zongheng_tracked_vs_single.py <single_frames> <tracked_frames>

# ============================================================================
# 本机 → 服务器 upload（2026-11-16 新增）
# 大件产物已外迁 workspace/cache ⇒ upload 传输量 60 GB → 82.5 MB
# 详见 manu/memory/memory.md §174 与 workspace/cache/README.md
# ============================================================================

# 预览（默认 dry-run，不落盘）
DRY_RUN=1 bash manu/videos/upload_to_server.sh

# 真传（排除项写死；默认不带 --delete ⇒ 服务器上已有的数据不会被删）
DRY_RUN=0 bash manu/videos/upload_to_server.sh

# 真传并允许删除服务器侧多余文件（🔴 危险，确认后再用）
ALLOW_DELETE=1 DRY_RUN=0 bash manu/videos/upload_to_server.sh

# ⚠️ 手工 rsync 时必须自己带上这些排除项，否则会抹掉服务器自己的 30 GB runs/：
#   --exclude=/runs --exclude=/manu/pipeline/npu/out --exclude=/manu/pipeline/npu/results
#   --exclude=/manu/pipeline/npu/build --exclude=/manu/pipeline/npu/.demo-build
#   --exclude=/manu/pipeline/npu/.demo-build-mobilenet
#   --exclude='/manu/pipeline/npu/convert/*.rknn'
#   --exclude=/manu/pipeline/opencl/board/out --exclude=/manu/pipeline/opencl/board/.bench_bmp

# ============================================================================
# 周末训练轴（2026-11-20 新增；服务器侧）
# 详见 manu/memory/memory.md §280（汤空间封死）/ §281（结构互斥）/ §282（L5 + L5B 轴）
# ⚠️ 会真的烧 GPU 并写臂号目录（臂号一旦绑定不可改配方 ⇒ 幂等续跑靠「epoch<N>.pt 存在」判定）
# ============================================================================

# 启动器（推荐）：会先杀旧训练轴 + 训练子进程再拉起，并打印自检
#   环境变量：ARMS=并发臂数(默认 3) / GPUS 写死在脚本里
bash manu/pipeline/npu/scripts/weekend_train_axis_20261120.sh --dry-run   # 只打印
bash manu/pipeline/npu/scripts/weekend_train_axis_20261120.sh             # 真执行
#   实配：--arms 3 --gpus 0,2,3 --deadline 2026-11-23 08:00 --polish-arms 600 --pt1-arms 30 --l5-arms 28 --l5b-arms 42

# 直接跑训练轴本体（服务器运行时根 = /tmp/pycharm_project_10ae9e2e → nano_search/repo）
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/night_train_axis.py \
    --arms 3 --gpus 0,2,3 --stagger 60 --deadline "2026-11-23 08:00" \
    --polish-arms 600 --pt1-arms 30 --l5-arms 28 --l5b-arms 42

# 通道（channel）与臂号段（台账 nano_search_ledger.json / arm_recipe_registry.json）
#   LOT  中签扫描      2105000+   prio -2（家族 n）
#   PT1  T1 供体轻抛   2106000+   2 epoch，官方门禁强制
#   L5   T1 供体长训   2107000+   5 epoch + cp + L2 锚定，官方门禁强制（§282 新增）
#   L5B  L5 补网格     2108000+   lr{0.75,2.5} × cp{0.05,0.10,0.15}，5 epoch，官方门禁强制（§282）
#   CP   cp 强度轴     2104000+
#   P    抛光          2100000+   锚点分配 V3 60% / R2 25% / BASE 15%（§281 复核后不改）
#   T    新盆地        2101000+   T2_fp 红线 ≤5

# 看日志 / 看台账
tail -f /mnt/data/siping/runs/npu20ms/nano_search/night_train_axis.log
grep -n "臂计划\|阶段1-C" /mnt/data/siping/runs/npu20ms/nano_search/night_train_axis.log | tail -3

# ⚠️ 改「正在跑的脚本」对旧实例无效 ⇒ 顺序必须是：先改脚本 → 再建开关 → 最后杀进程重启
# 📌 铁律：任何「会 pkill 再拉起」的脚本都必须加 flock（cron heartbeat.sh 与守望是两个竞争拉起器
#   ⇒ 双实例 ⇒ 同臂号目录竞争 ⇒ 台账静默污染）。weekend_train_axis_20261120.sh 已加 flock -n（fd 9，
#   锁 $W/.train_axis_launch.lock）；锁只在脚本执行期持有 ⇒ 顺序重启仍可用。

# ============================================================================
# 视频质量 → 跟踪效果（好片 vs 坏片）质量线挖掘：48 宫格模拟点击 + 双向闭环 + 点选式 SOT
# 详见主档 manu/memory/memory.md §271 / §284，专题档 manu/memory/video_quality_metrics_20261119.md §七 / §八
# 汇报件 manu/video_quality/report/REPORT_视频质量指标.md 第八节 / 第九节
# 数据：好片 /home/manu/mnt/data/xxxu/朗基天香20260903（18 条 .h264）
#       坏片 /home/manu/mnt/data/xxxu/纵横挂飞（7 条 .mp4，含 osd/ 子目录 5 条为桌面录屏参考组）
# 执行机：huangzhe@192.168.99.40:32222    conda env: uav    纯 CPU，不占 GPU
# ============================================================================

# 0) 连服务器（铁律：必须带 -F /dev/null 与 StrictHostKeyChecking=no）
ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_a7d966e6 && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav

# ⚠ 铁律二十五（2026-11-23 用户裁决）：**代码产生的中间产物一律出仓库目录**；仓库只留「代码 + 结论级 report」。
#    ⇒ 产物根统一走 VQ_PROD（服务器侧 canonical = /mnt/data/siping/runs/video_quality）；
#      本机跑时覆盖为：
#      VQ_PROD=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/video_quality
VQ_PROD="${VQ_PROD:-/mnt/data/siping/runs/video_quality}"
mkdir -p "${VQ_PROD}/report"

# 1) 逐帧画质指标提取（25 条片，4 worker，线程钉 1；产物 ${VQ_PROD}/out/*.json）
bash manu/video_quality/vq_launch_server.sh 4
# 1b) 模态重分类 + 采样占比落盘（报告里「81.0% 可见光」等数字的来源，缺这步会 KeyError）
python manu/video_quality/vq_reclassify.py --metrics-dir ${VQ_PROD}/out

# 2) 好/坏对比 + AUC + 阈值挖掘 + 出图（产物落 ${VQ_PROD}/report/；结论级文件由第 9 步「显式挑选」回仓库）
#    ⚠ vq_compare.py 的入参是 --metrics-dir / --report-dir（**没有 --out-dir**，写错会直接 argparse 报错）
python manu/video_quality/vq_compare.py \
  --metrics-dir ${VQ_PROD}/out --report-dir ${VQ_PROD}/report

# --- 第八节：48 宫格「模拟操作员点击」可跟踪性（暴力 NCC 搜索） ---
# 3) 逐片提取网格可跟踪性（4 worker，线程钉 1，内存看门狗 <20% 自刹车）
bash manu/video_quality/vq_grid_launch_server.sh 4
#    产物：${VQ_PROD}/grid_out/<片名>.{json,npz,rows.csv} + run_manifest.json
# 4) 与质量红线对照 + 出图（AUC / 交叉表 / 阈值敏感性 / 6 张图）
python manu/video_quality/vq_grid_compare.py \
  --grid-dir ${VQ_PROD}/grid_out \
  --windows-csv ${VQ_PROD}/report/windows.csv \
  --report-dir ${VQ_PROD}/report
python manu/video_quality/vq_grid_charts.py \
  --grid-dir ${VQ_PROD}/grid_out \
  --report-dir ${VQ_PROD}/report
# 5) LK 光流对照实验（方法无关性验证；合成帧自检不需要视频）
python manu/video_quality/test_vq_grid_lk.py        # 已知位移必须被还原
bash manu/video_quality/vq_grid_lk_launch_server.sh 4
#    产物：${VQ_PROD}/grid_lk_out/<片名>.{json,rows.csv}

# --- 第九节：点选式单目标跟踪（SOT）实测 + 画质合格线 ---
# 6) 逐片模拟点击 + 链式 SOT 跟踪（作业表 = TSV：路径 分组 kind 模态）
python manu/video_quality/vq_sot_link_driver.py --jobs /tmp/sot_jobs.tsv \
  --out-dir /tmp/sot_out --workers 22 --threads 4 --click-mode alternating
python manu/video_quality/vq_sot_link_driver.py --jobs /tmp/sot_jobs_osd.tsv \
  --out-dir /tmp/sot_out_osd --workers 3 --threads 4 --click-mode alternating
#    产物：/tmp/sot_out/<片名>.json（22 条源片）、/tmp/sot_out_osd/<片名>.json（3 条 OSD）
# 7) 聚合 + 画质合格线反解（长 / 短两个口径各跑一次）
python manu/video_quality/vq_sot_compare.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows ${VQ_PROD}/report/windows.csv --grid-summary ${VQ_PROD}/report/grid_summary.csv \
  --out-dir ${VQ_PROD}/report/sot_long  --score xcorr_long_ugr
python manu/video_quality/vq_sot_compare.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows ${VQ_PROD}/report/windows.csv --grid-summary ${VQ_PROD}/report/grid_summary.csv \
  --out-dir ${VQ_PROD}/report/sot_short --score xcorr_short_ugr
#    产物：${VQ_PROD}/report/sot_{long,short}/sot_{groups,discrimination,thresholds,tau_det,clips,windows}.csv
# 8) 独立复算验收（不 import compare；空输入直接 rc=2 拒绝出 PASS ⇒ 不会假通过）
python manu/video_quality/vq_sot_verify.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows ${VQ_PROD}/report/windows.csv --grid-summary ${VQ_PROD}/report/grid_summary.csv \
  --thresholds ${VQ_PROD}/report/sot_long/sot_thresholds.csv

# 9) 🔴 显式挑选（铁律二十五 §4）：**只有结论级 report 回仓库**，中间态永远留在 ${VQ_PROD}/
#    ① 画质结论表 + 图（windows.csv / metrics_table.csv / comparison.json / stats.json 是中间产物，不回仓库）
for f in comparison.csv stats.csv tail_compare.csv tail_slices.csv; do
  [ -f "${VQ_PROD}/report/$f" ] && cp -f "${VQ_PROD}/report/$f" manu/video_quality/report/
done
#    🔴 图表不入库（铁律二十五）：落 cache 的 repo-artifacts 镜像目录（1:1 镜像仓库相对路径，可原路还原）
mkdir -p /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report
cp -f ${VQ_PROD}/report/chart_*.png /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/
#    ② 宫格结论表 + 图（grid_windows.csv = 26490 行逐秒大表，中间产物，不回仓库）
for f in grid_summary.csv grid_compare.csv grid_crosstab.csv grid_sensitivity.csv grid_thresholds.json; do
  [ -f "${VQ_PROD}/report/$f" ] && cp -f "${VQ_PROD}/report/$f" manu/video_quality/report/
done
mkdir -p /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report
cp -f ${VQ_PROD}/report/chart_grid_*.png /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/
#    ③ SOT 结论表（长时程进 report/；短时程进 report/sot_short/，另存一份带 _short 后缀的扁平副本）
for f in sot_groups.csv sot_discrimination.csv sot_thresholds.csv sot_tau_det.csv sot_clips.csv; do
  [ -f "${VQ_PROD}/report/sot_long/$f" ]  && cp -f "${VQ_PROD}/report/sot_long/$f"  manu/video_quality/report/
  if [ -f "${VQ_PROD}/report/sot_short/$f" ]; then
    cp -f "${VQ_PROD}/report/sot_short/$f" manu/video_quality/report/sot_short/
    cp -f "${VQ_PROD}/report/sot_short/$f" manu/video_quality/report/${f%.csv}_short.csv
  fi
done

# 🔴 验收判据（第九节）：[PASS] all checks passed；per-clip |SOT lock - grid tgr_raw| mean ≤ 0.120（实测 0.082）；
#   CHECK 5 的 global_contrast re-derived thr == 存档 thr（实测 0.7746 == 0.7746）；
#   sot_groups.csv 里 bad/source 两行 xcorr_long_ugr_p50 必须是 0.0
# ⚠️ 铁律：未跑任何 git add / commit / push（仅落工作区）

# --- 第十节：板端 vs 服务器「交叉喂数」复核（2026-11-20 §285；零上板，吃已落盘的板端产物）---
# 10) 三方逐字节 + 打分逐位比对（板端 fused ≡ Python 镜像 ≡ C++ 对齐器）
ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_a7d966e6 && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export PYTHONPATH=.
python manu/evaluation/board_crossfeed_verify.py \
  --tsv-dir /mnt/data/siping/runs/trial0080_board24_20261120/tsv \
  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
  --params-flat manu/memory/align_dn_20261120/params_delivery.txt \
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --sequences VIDEO00007_19700101_001324,VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00030_19700101_014144,VIDEO00036_19700101_015740 \
  --radii 20 \
  --total-frames-json /mnt/data/siping/runs/trial0080_board24_20261120/total_frames.json \
  --source-dir manu/pipeline/npu --work-dir /tmp/crossfeed/work \
  --out /tmp/crossfeed/crossfeed_verify.json
# 11) 阈值卡边量化（两侧「阈值前」候选池逐目标比对 8 px 内最强峰）
python manu/evaluation/board_fp16_edge_probe.py \
  --tsv-dir /mnt/data/siping/runs/trial0080_board24_20261120/tsv \
  --cache-root /mnt/data/siping/runs/trial0080_onnx/ms_cache \
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --sequences VIDEO00007_19700101_001324,VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00030_19700101_014144,VIDEO00036_19700101_015740 \
  --out /tmp/crossfeed/edge_probe.json
# 12) 误检画像 + 最终分数门限扫描（FAR 红线的真实抓手）
python manu/evaluation/board_fp_profile.py \
  --board-tsv /mnt/data/siping/runs/trial0080_board24_20261120/tsv/VIDEO00007_19700101_001324_fused.tsv \
  --server-preds /mnt/data/siping/runs/trial0080_onnx/eval/shortlist/preds_cc2_md10/VIDEO00007_19700101_001324.json \
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --sequence VIDEO00007_19700101_001324 --frames 11810 \
  --out /tmp/crossfeed/fp_profile_07.json
# 🔴 验收判据（第十节）：5 序三方 md5 全同（`VIDEO00007` = 9e087698）且 F1 逐位相等（0.8597382）；
#   ⚠️ 比 md5 前必须先滤掉空帧（板端只写有检出的帧，Python 镜像写每一帧）
# ⚠️ 铁律：未跑任何 git add / commit / push（仅落工作区）

# ============================================================================
# SB4 细网格汤融合轴（2026-11-20 新增；服务器侧，独立进程）
# 详见 manu/memory/memory.md §286
# ⚠️ 不训练（只 checkpoint 加权平均 + NanoBench）⇒ 不抢训练轴 GPU；但会写台账（save_ledger 带跨进程 flock）
# ============================================================================
# 启动器（自带 flock + 幂等续跑 + 「本轮新增臂数=0 ⇒ 组合跑完即退出」）：
bash manu/pipeline/npu/scripts/soup_fusion_axis_20261120.sh          # 服务器上跑
#   parents   14963:4,15510:4,1104642:5          （T1 富集）
#   children  1104200:1,1104203:1,1104204:1,1104205:1,1104207:1   （nano 富集，从未门禁）
#   weights   0.10..0.90 步长 0.05（17 档）⇒ 3×5×17 = 255 臂
#   臂号段    2109000~2109399（独占段，flock 占用；与 826 臂训练轴零重叠）
#   GPU       SB4_GPU=1（默认），nice -n 15 降优先级
# 看日志
tail -f /mnt/data/siping/runs/npu20ms/nano_search/soup_fusion_axis.log
grep "⇒ nano=" /mnt/data/siping/runs/npu20ms/nano_search/soup_fusion_axis.log | tail -20
# 📌 SB4 补记（17:47）：跨家族汤实测**混坏 nano**（90% child 仍 −24 阶 < 线性 +12 阶）
#   ⇒ 父本扩为 6 个（加 1104200:1 / 1104203:1 / 1104205:1 ⇒ **同族互汤**）⇒ 6×5×17 = **510 臂**
#   脚本 md5 `c3a6144af93b965694724fcbd9ff57e3`（旧 `91152028…` 作废）
#   周一重点看家族 bs1104200:* / bs1104203:* / bs1104205:*（同族）；三轴全过 ⇒ stage-2 自动送官方门禁

# ============================================================================
# 六项门禁审计 / 升轨 / 台账修复（2026-11-23 新增；服务器侧）
# 详见 manu/memory/monday_pickup_20261123.md 与 manu/memory/memory.md §286
# ⚠️ 运行时根 = /tmp/pycharm_project_10ae9e2e（软链到仓库）⇒ 下面 4 条都要先 cd + export PYTHONPATH=.
#    python = /home/huangzhe/anaconda3/envs/uav/bin/python
#    台账   = /mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json
# ============================================================================

# --- ① 六项门禁全盘复算审计（只读）---
#   六项：nano / nano4 / f1 / T1 / T2_fp / T3，对当前基线复算（Baseline_v3.1 = trial_1990055 @ ep5）
#   口径：台账里的 NanoBench 读数 × 磁盘官方门禁报告配对
#     /mnt/data/siping/runs/npu20ms_offcal/gate/NANO_<trial>_ep<epoch>_official/gate_report.json
#   🔴 这是「有没有更好的模型」的**唯一权威口径**（台账 official 可能 stale/缺项 ⇒ 别只看台账）
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/six_criteria_audit.py --top 12

# --- ② 升轨命令生成器（只读；印出可复制粘贴的升轨命令）---
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_ready_cmd.py --top 6

# --- ③ 正式升轨（真写台账 + 换基线名）---
#   会做板端 T4 实测 + 种子复现判定 ⇒ 慢；--force = 允许未过种子关强推；--require-board = 强制板端 T4 PASS
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_candidate.py \
    --trial <N> --name Baseline_v3.2 [--force] [--require-board]
#   🔴 种子复现关：对「抛光臂」要求 n_anchor ≥ 2（同一配方要跨 ≥2 个锚点）
#      L5/L5B 同一配方跨 7 锚点 ⇒ 天然满足；单锚点抛光臂 ⇒ 只能 --force
#   另：--dry-run 只预检不写台账（先预检再真升轨更稳）

# --- ④ 台账修复：补回被并发写 lost-update 吃掉的 picked 读数 ---
#   根因：台账合并是「整臂覆盖」⇒ 内存快照陈旧的进程会把别的进程刚写的真读数整臂覆盖成 None
#   安全设计：全程持同一把 $WORK/ledger.lock flock ⇒ 与台账写手的「读-合并-写」互斥；持锁后重读盘；
#             只补「picked 缺失」的臂（已有值一律不动）；临时文件 + os.replace 原子写；幂等
#   ⚠️ 已注册为 cron（每 10 分钟一次）⇒ 手工跑与 cron 同时起也安全（同一把锁）
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/repair_soup_ledger.py --dry-run
cd /tmp/pycharm_project_10ae9e2e && export PYTHONPATH=. && \
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/repair_soup_ledger.py
#   只修某段臂号：--min-sid <N> / --max-sid <N>（如只修 SB4 段 ⇒ --min-sid 2109000）

# ============================================================================
# PointAnnotator 标注工具：三份并行交付（拆分 / 验收 / 自检）（2026-11-25 新增）
# 详见 manu/memory/memory.md §315 与 manu/point_annotator/README.md
# ============================================================================
# 母包（58 条视频 + 58 个序列的模型预推点，保持全量形态不动）：
export PA_MASTER=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/PointAnnotator_Windows
export PA_OUT=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/PointAnnotator_Windows_三份交付

# ① 只看分配方案，不落盘（秒级；ffprobe 读容器帧数，不用解码）
python3 manu/point_annotator/split_delivery.py --master "$PA_MASTER" --out-dir "$PA_OUT" \
  --parts 3 --frame-cache /tmp/vq_frames.json --plan

# ② 真正生成三份（每份自包含：tool/ + docs/ + Python 安装包 + 自己的 videos/ 与 preannot/）
python3 manu/point_annotator/split_delivery.py --master "$PA_MASTER" --out-dir "$PA_OUT" \
  --parts 3 --frame-cache /tmp/vq_frames.json
#   ⇒ 产出 "$PA_OUT"/{PointAnnotator_Windows_标注员1,2,3}/ + 交付说明.md + _split_summary.json
#   等量口径：总帧数 / 手工帧数（无预推点须手工落点的帧）/ 工作量（手工 1.0 + 核对 0.35）三项对齐
#   🔴 三份互不重叠、并集 = 母包 58 条 ⇒ 回收时按视频名直接并集，不会互相覆盖

# ③ 验收（67 项：结构 / 配对 / 逐字节 md5 / 等量 / 无重复无遗漏；约 1 分钟）
python3 manu/point_annotator/verify_split.py --master "$PA_MASTER" --parts-dir "$PA_OUT"

# ④ 单份自校验（在任一份的根目录下执行；每份 ~3.2 万个文件）
cd "$PA_OUT/PointAnnotator_Windows_标注员1" && md5sum -c _split_manifest.md5

# ⑤ 工具无窗口自检（本机需要 opencv + numpy；三份都应输出 PASS 101 / FAIL 0）
for d in "$PA_OUT"/PointAnnotator_Windows_标注员*; do (cd "$d" && python3 tool/point_annotator.py --self-test); done
