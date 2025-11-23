#!/bin/bash
GPU_ID=0
DATE=2023_7_5

#Cityscapes
DATASET_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/day/AB/
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/Cityscapes/RGB/leftImg8bit_trainvaltest/leftImg8bit/val_all
DATASET_DIR_B=$DATASET_DIR_A
SEG_DIR=$DATASET_DIR_A

DATASET_NAME="KAIST"
TARGET_DATASET="Cityscapes512_256"

CHECKPOINT_DIR=8

MODEL="AAFSTNet2Seg"
STAGE="full"
OPTION="_Thermal"
GAN_TYPE="RSGAN"
RESULT_DIR="/home/usrs/ugawa/lab/work/datasets/Cityscapes/thermal/leftImg8bit/val_all_KAIST"
EPOCH="50"
CLAHE="" # --clahe
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="RGB"

SAL_MAP="None" # saliency or sod or None
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD=""    # --input_sod

USE_SEGMAP=""    # --use_segmap
USE_SEGMAP_PROB=0

LOADSIZE_W=512
LOADSIZE_H=512
FINESIZE_W=512
FINESIZE_H=512

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_${TARGET_DATASET} --phase test --which_epoch $EPOCH --serial_batches --how_many 30000 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}

