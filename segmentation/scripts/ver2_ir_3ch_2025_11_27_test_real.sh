#!/usr/bin/env bash
## DeepLabV3+ RGB テスト用スクリプト


MODEL_NAME=deeplabv3plus_resnet101
DOMAIN=ir_3ch
NUM_CLASSES=9
OUTPUT_STRIDE=16
PTH_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation/checkpoints/ver2_ir_3ch_deeplabv3plus_resnet101_2025_11_27/seg_100_100_202511271736.pth
SEG_MODEL_VERSION=2


IMAGE_PATH=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images256_ir_3ch
COLOR_MODEL=Ugawa
COLOR_CONFIG=FLIR_MFNet_day
COLOR_DAY=2025_11_27

INPUT_SIZE=500
COLOR_MEAN=0.232,0.267,0.233
COLOR_STD=0.173,0.173,0.172
TARGET_DIR=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/labels256
FILE_LIST=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/test_day.txt

SAVE_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/output_seg/ver${SEG_MODEL_VERSION}_${DOMAIN}_real
# SAVE_DIR_2=${IMAGE_ROOT_PATH}/ver${SEG_MODEL_VERSION}_${DOMAIN}


cd /home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation

# 出力先を事前に作成
mkdir -p "$SAVE_DIR"

python seg_test.py \
  --imagepath $IMAGE_PATH \
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
  # --save_dir_2 $SAVE_DIR_2   # 使う場合のみ値を設定してから有効化

# 自身のパス・名前取得し、スクリプト自身を保存
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
cp "${SCRIPT_PATH}" "${SAVE_DIR}/${SCRIPT_NAME}"
# [[ -n "$SAVE_DIR_2" ]] && mkdir -p "$SAVE_DIR_2" && cp "${SCRIPT_PATH}" "${SAVE_DIR_2}/${SCRIPT_NAME}"





