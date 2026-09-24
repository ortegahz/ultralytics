from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

results = model.train(data="coco.yaml", epochs=100, imgsz=640, device=[0, 1, 2, 3])
