# install
pip install -e .

# tune
screen python manu/optuna_parallel_uav.py

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
