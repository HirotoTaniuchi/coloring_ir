#!/bin/bash
GPU_ID=0
DATE=2023_5_20

#FLIR
DATASET_DIR=/home/usrs/ugawa/lab/work/datasets/FLIR_datasets_PearlGAN/testB
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/FLIR_datasets_PearlGAN/testB
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/FLIR_datasets_PearlGAN/testB
SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/resnet101/segmentation

DATASET_NAME="FLIR"

CHECKPOINT_DIR=6

MODEL="AAFSTNet2Seg"
STAGE="full"
OPTION="_RSGAN_RGB_seg0.75_concat"
GAN_TYPE="RSGAN"
RESULT_DIR="./results_p"
EPOCH="200"
CLAHE="" # --clahe
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="RGB"

SAL_MAP="None" # --saliency or --sod or None
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD=""    # --input_sod

USE_SEGMAP="--use_segmap"    # --use_segmap
USE_SEGMAP_PROB=0

LOADSIZE_W=360
LOADSIZE_H=288
FINESIZE_W=360
FINESIZE_H=288

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}

#mkdir -p ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_day_${EPOCH}
#cp test_FLIR_YOLO_seg.sh "${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_day_${EPOCH}/."

