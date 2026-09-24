from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained YOLO26n model
results = model("ultralytics/assets/bus.jpg", save=True)
