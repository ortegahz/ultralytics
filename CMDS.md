# install

pip install -e .

# data

<rm old files>
cp ~/test/manu-pc/ST2000DM005/fire/data/pseudo_filtered/out_all/images/pseudof/ . -rvf
cp ~/test/manu-pc/ST2000DM005/fire/data/pseudo_filtered/out_all/labels/pseudof/ . -rvf

# infer

yolo predict model=yolo11s.pt source='https://ultralytics.com/images/bus.jpg'

yolo predict model=/home/manu/tmp/runs_yolo11/train13/weights/best.pt source=/home/manu/tmp/16_12米-1.mp4 save_txt=False save_frames=False show_labels=False show_conf=False show_boxes=True conf=0.25

yolo predict model=/home/manu/tmp/runs_yolo11/train13/weights/best.pt source="/home/manu/tmp/fire (248).mp4" show=True

yolo predict model=/home/manu/tmp/runs_yolo11/train14/weights/best.pt source=/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai/J-D-30m-001.mp4 show=True
yolo predict model=/home/manu/tmp/runs_yolo11/train14/weights/best.pt source=/home/manu/mnt/ST2000DM005-2U91/fire/data/test/火/正例/08_打火机3距离B.mp4 show=True

yolo predict model=/home/manu/tmp/runs_yolo11/train13/weights/best.pt source=/home/manu/tmp/云盒误报1125-1201/火焰误报.mp4 show=True

yolo predict model=/home/manu/tmp/runs_yolo11/train13/weights/best.pt source=/media/manu/ST2000DM005-2U91/fire/data/aigc_20241230/AI图片/AIFirePicture3

# train

yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=4,5,6,7

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=yolo11s.pt epochs=100 imgsz=640 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train2/weights/best.pt epochs=100 imgsz=640 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train3/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train4/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train5/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train6/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train7/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train8/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train10/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train11/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

screen yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train12/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train13/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/tmp/pycharm_project_278/ultralytics/cfg/models/11/yolo11s.yaml pretrained=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train13/weights/best.pt epochs=10 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

# export

yolo export model=/home/manu/tmp/runs_yolo11/train14/weights/best.pt format=onnx opset=11

# tensorboard

tensorboard --logdir /home/manu/tmp/runs_yolo11/train/