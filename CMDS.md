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

python manu/infer_heatmap.py \
    --weights runs/heatmap_uav/uav_gpu23_heatmap/weights/best_recall.pt \
    --source /mnt/data/siping/datasets/manu/uav/images/val \
    --conf 0.25 \
    --device 1 \
    --save-dir runs/heatmap_infer_gt

