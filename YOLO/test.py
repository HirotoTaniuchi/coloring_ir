from ultralytics import YOLO

save_name = 'test_ir3ch'  # Change this to 'test_ugawa' for the other dataset
model = YOLO('runs/detect/train/weights/best.pt')
metrics = model.val(data='/home/usrs/taniuchi/workspace/projects/coloring_ir/YOLO/data_test_ir3ch.yaml'
                    , save_json=True, save_txt=True, save_conf=True, split='val', name=save_name)
print(metrics)
