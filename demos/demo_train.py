from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model="/tmp/pycharm_project_278/ultralytics/cfg/models/11/yolo11.yaml",
            data="/tmp/pycharm_project_278/ultralytics/cfg/datasets/fire.yaml",
            pretrained="/home/Huangzhe/test/manu-pc/tmp/runs_yolo11/train13/weights/best.pt",
            epochs=3)
trainer = DetectionTrainer(overrides=args)
trainer.train()
