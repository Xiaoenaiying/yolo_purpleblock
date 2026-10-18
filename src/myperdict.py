from  ultralytics import YOLO

model=YOLO("D:/Code/yolo_learning/yolo_v8purple_block.pt")
model.predict(
    source=0,
    save=False,
    show=True,
    save_txt=True
)