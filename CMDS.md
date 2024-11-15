# install

pip install -e .

# infer

yolo predict model=yolo11s.pt source='https://ultralytics.com/images/bus.jpg'

yolo predict model=/home/manu/tmp/runs_yolo11/train/weights/best.pt source=/media/manu/ST2000DM005-2U91/fire/data/20241112_neg/ save_txt=True save_frames=True show_labels=False show_conf=False show_boxes=False

yolo predict model=/home/manu/tmp/runs_yolo11/train4/weights/best.pt source=/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai/J-D-30m-001.mp4 show=True

yolo predict model=/home/manu/tmp/runs_yolo11/train4/weights/best.pt source=/media/manu/ST2000DM005-2U91/fire/data/20241112_neg/12_0_6858f798d78279fb647f415734da79e9.mp4 show=True

yolo predict model=/home/manu/tmp/runs_yolo11/train4/weights/best.pt source=/home/manu/tmp/fire_037_384_640.bmp show=True

# train

yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=4,5,6,7

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=yolo11s.pt epochs=100 imgsz=640 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train2/weights/best.pt epochs=100 imgsz=640 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

yolo detect train data=ultralytics/cfg/datasets/fire.yaml model=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train3/weights/best.pt epochs=100 imgsz=1280 device=0,1,2,3,4,5,6,7 project=/home/Huangzhe/test/manu-pc/tmp/runs_yolo11

# export

yolo export model=/home/manu/tmp/runs_yolo11/train2/weights/best.pt format=onnx

# tensorboard

tensorboard --logdir /home/manu/tmp/runs_yolo11/train/