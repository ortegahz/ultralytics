from ultralytics import YOLO

# 使用 COCO 预训练的 YOLO26n 权重
model = YOLO("yolo26n.pt")

results = model.train(
    # 你的 Frame Difference 数据集配置
    data="/mnt/data/siping/datasets/manu/uav/data.yaml",

    # 输入尺寸
    imgsz=640,

    # 训练周期
    epochs=20,

    # 四张 GPU
    device=[0, 1, 2, 3],

    # 每张 GPU 的 batch size。
    # 4 张 GPU、每卡 32 时，总 batch size 为 128
    batch=32,

    # 优化器先使用自动选择，作为稳定基线
    optimizer="auto",

    # 训练数据加载线程
    workers=8,

    # 保存策略
    save=True,
    save_period=10,

    # 训练日志
    plots=True,

    # 关闭 HSV 增强：
    # FD 图像的三个通道不是普通 RGB，不应改变色调、饱和度和亮度语义
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,

    # 空间增强
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    translate=0.1,
    scale=0.5,

    # 水平翻转可保留；垂直翻转需根据相机运动场景决定
    fliplr=0.5,
    flipud=0.0,

    # 你的任务是单 UAV 检测，不建议使用 Copy-Paste
    copy_paste=0.0,

    # 红外小目标场景不建议一开始使用 MixUp
    mixup=0.0,

    # Mosaic 可保留，但不要过度增强
    mosaic=0.7,

    # 最后若干 epoch 关闭 Mosaic
    close_mosaic=5,

    # 早停
    patience=0,

    # 训练确定性
    seed=42,

    # 单类别任务
    single_cls=True,
)
