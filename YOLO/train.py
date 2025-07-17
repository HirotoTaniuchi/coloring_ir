from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # "n"はNano。他にs/m/l/xがある
model.train(data='/home/usrs/taniuchi/workspace/datasets/ir_seg_det_dataset/data.yaml', epochs=50, imgsz=640)