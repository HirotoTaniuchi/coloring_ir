#!/usr/bin/env bash
## DeepLabV3+ RGB テスト用スクリプト


MODEL_NAME=deeplabv3plus_resnet101
DOMAIN=rgb
NUM_CLASSES=9
OUTPUT_STRIDE=16
PTH_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation/checkpoints/ver2_rgb_deeplabv3plus_resnet101_2025_11_27/seg_100_100_202511270131.pth
SEG_MODEL_VERSION=2


IMAGE_ROOT_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/output_img/Ugawa_2025_11_25_AAFSTNet2andSegNet2_FLIR_MFNet_day_256x256/ugawa_ni_tekiyou/test_day_latest
# DOMAIN に応じて IMAGEPATH の末尾を付与
case "$DOMAIN" in
  rgb)IMAGEPATH="${IMAGE_ROOT_PATH}/fake_B";;
  rgb_ir)IMAGEPATH="${IMAGE_ROOT_PATH}/fake_4ch";;
esac
COLOR_MODEL=Ugawa
COLOR_CONFIG=FLIR_MFNet_day
COLOR_DAY=2025_11_27

INPUT_SIZE=500
COLOR_MEAN=0.232,0.267,0.233
COLOR_STD=0.173,0.173,0.172
TARGET_DIR=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/labels256
FILE_LIST=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/test_day.txt

SAVE_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/output_seg/ver${SEG_MODEL_VERSION}_${DOMAIN}_${COLOR_MODEL}_${COLOR_DAY}_${COLOR_CONFIG}
SAVE_DIR_2=${IMAGE_ROOT_PATH}/ver${SEG_MODEL_VERSION}_${DOMAIN}

cd /home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation

# 出力先を事前に作成
mkdir -p "$SAVE_DIR"
mkdir -p "$SAVE_DIR_2"  # 使用しない場合はこの行をコメントアウト

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
  # --save_dir_2 $SAVE_DIR_2  # 使用する時のみコメント解除

# 自身のパス・名前取得し、スクリプト自身を保存
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
cp "${SCRIPT_PATH}" "${SAVE_DIR}/${SCRIPT_NAME}"
# cp "${SCRIPT_PATH}" "${SAVE_DIR_2}/${SCRIPT_NAME}"  # 使用する時のみコメント解除


