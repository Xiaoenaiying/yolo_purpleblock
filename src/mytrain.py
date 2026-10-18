from ultralytics import YOLO

def main():
    model=YOLO(r"runs/detect/yolov8n_block_v3-5/weights/bestright00.pt")
    model.train(
        data='datasets/xz-/block_xz.yaml',#yolo训练数据集
        epochs=100,
        imgsz=640,#图像尺寸
        batch=-1,#批，分批打包发送给模型（一次多少张图片）（-1自动设置）
        cache='ram',#缓存
        workers=3,#多进程开关     
        device=0,#指定用 GPU
        conf=0.8,#置信度阈值
        iou=0.3,#印制重复框
        name='yolov8n_block_v3',  # 换个名字，别覆盖旧的
        # 关键修改点 ↓
        #mosaic=0.0,        # 关掉马赛克增强！之前崩盘就是它害的
        #mixup=0.0,         # 关掉混叠
        #hsv_h=0.015,       # 色调抖动
        #hsv_s=0.7,         # 饱和度抖动
        #hsv_v=0.4,         # 亮度抖动
        #degrees=10.0,      # 稍微加点旋转
        #translate=0.1,     # 平移
        #scale=0.5,         # 缩放
        #patience=20,       # 20轮不提升就早停，别让它无限乱跑
    )



if __name__=="__main__":
    main()   