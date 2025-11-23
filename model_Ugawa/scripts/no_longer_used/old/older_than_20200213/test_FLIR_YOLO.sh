#!/bin/bash
GPU_ID=2
DATE=2023_5_10

#FLIR
DATASET_DIR=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/day/AB/
DATASET_DIR_A=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/A/day
DATASET_DIR_B=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day

DATASET_NAME="FLIR"

CHECKPOINT_DIR=6

MODEL="AAFSTNet2Sobel"
STAGE="full"
OPTION="_unet_ws5_wcon10_HSV"
GAN_TYPE="SGAN"
RESULT_DIR="./results_6/"
EPOCH="200"
CLAHE="" # --clahe
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="HSV"

SAL_MAP="sod" # --saliency or --sod
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD="--input_sod"    # --input_sod

LOADSIZE_W=256
LOADSIZE_H=256
FINESIZE_W=256
FINESIZE_H=256

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD

wait

# FLIR YOLO
DATASET_DIR=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/RGB
DATASET_DIR_A=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/thermal_8_bit
DATASET_DIR_B=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/night
DATASET_NAME="FLIR"

LOADSIZE_W=360
LOADSIZE_H=288
FINESIZE_W=360
FINESIZE_H=288

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace ${COLORSPACE} --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD

#mkdir -p ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/resized_fake_B
#python resize_images.py ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/fake_B ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/resized_fake_B
