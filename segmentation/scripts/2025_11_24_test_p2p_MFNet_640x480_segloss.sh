#!/usr/bin/env bash
## DeepLabV3+ RGB テスト用スクリプト

IMAGEPATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/output_img/mfnet_pix2pix_640x480_segloss_20251106/fake_B
MODEL_NAME=deeplabv3plus_resnet101
DOMAIN=rgb
NUM_CLASSES=9
OUTPUT_STRIDE=16
PTH_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation/checkpoints/rgb_deeplabv3plus_resnet101_202510012130/seg_100_100_202510012130.pth
INPUT_SIZE=500
COLOR_MEAN=0.232,0.267,0.233
COLOR_STD=0.173,0.173,0.172
TARGET_DIR=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/labels
FILE_LIST=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/test_day.txt

SAVE_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/output_seg/${MODEL_NAME}_test_$(date +"%Y_%m_%d")

cd /home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation

python seg_test.py \
  --imagepath $IMAGEPATH \
  --model_name $MODEL_NAME \
  --num_classes $NUM_CLASSES \
  --output_stride $OUTPUT_STRIDE \
  --domain $DOMAIN \
  --pth_path $PTH_PATH \
  --input_size $INPUT_SIZE \
  --color_mean $COLOR_MEAN \
  --color_std $COLOR_STD \
  --save_dir $SAVE_DIR \
  --file_list $FILE_LIST \
  --target_dir $TARGET_DIR

cp ./scripts/2025_11_24_test_rgb_deeplab.sh ${SAVE_DIR}/2025_11_24_test_rgb_deeplab.sh


