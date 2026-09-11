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
