#!/usr/bin/env bash
## DeepLabV3+ IR 学習用スクリプト

GPU_ID=0
ROOTPATH=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset
DOMAIN=ir_3ch
MODEL_NAME=deeplabv3plus_resnet101
PRETRAINED_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation/model_DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth
NUM_CLASSES=9          # 例: MFNet 用に 9
OUTPUT_STRIDE=16
BATCH_SIZE=4
N_LAYERS=100
NUM_EPOCHS=100
LR=0.001
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
INPUT_SIZE=600
AUX_WEIGHT=0.4
COLOR_MEAN=0.232,0.267,0.233
COLOR_STD=0.173,0.173,0.172

# 学習元ディレクトリ名（ROOTPATH 直下）
IMAGE_DIR_IR=images256_ir
IMAGE_DIR_RGB_IR=images256_rgb_ir
IMAGE_DIR_RGB=images256_rgb
IMAGE_DIR_IR_3CH=images256_ir_3ch

# DOMAIN に応じて使用ディレクトリを選択（未対応ドメインは従来の images_<domain> にフォールバック）
case "${DOMAIN}" in
  ir)  IMAGE_DIR_NAME="${IMAGE_DIR_IR}" ;;
  rgb) IMAGE_DIR_NAME="${IMAGE_DIR_RGB}" ;;
  ir_3ch) IMAGE_DIR_NAME="${IMAGE_DIR_IR_3CH}" ;;
  *)   IMAGE_DIR_NAME="images_${DOMAIN}" ;;
esac

DATE_TAG=$(date +"%Y_%m_%d")
SEG_MODEL_VERSION=2
CPT_DIR=checkpoints/ver${SEG_MODEL_VERSION}_${DOMAIN}_${MODEL_NAME}_${DATE_TAG}
LOG_DIR=logs/ver${SEG_MODEL_VERSION}_${DOMAIN}_${MODEL_NAME}_${DATE_TAG}

cd /home/usrs/taniuchi/workspace/projects/coloring_ir/segmentation
mkdir -p ${CPT_DIR}
cp scripts/ver2_ir_3ch_2025_11_27_train.sh ${CPT_DIR}/ver2_ir_3ch_2025_11_27_train.sh

python seg_train.py \
  --rootpath $ROOTPATH \
  --domain $DOMAIN \
  --gpu_id $GPU_ID \
  --color_mean $COLOR_MEAN \
  --color_std $COLOR_STD \
  --batch_size $BATCH_SIZE \
  --n_layers $N_LAYERS \
  --model_name $MODEL_NAME \
  --num_classes $NUM_CLASSES \
  --output_stride $OUTPUT_STRIDE \
  --pretrained_path $PRETRAINED_PATH \
  --num_epochs $NUM_EPOCHS \
  --lr $LR \
  --momentum $MOMENTUM \
  --weight_decay $WEIGHT_DECAY \
  --path_cpt $CPT_DIR \
  --path_logs $LOG_DIR \
  --input_size $INPUT_SIZE \
  --aux_weight $AUX_WEIGHT \
  --image_dir_name $IMAGE_DIR_NAME \
  --save_yaml


