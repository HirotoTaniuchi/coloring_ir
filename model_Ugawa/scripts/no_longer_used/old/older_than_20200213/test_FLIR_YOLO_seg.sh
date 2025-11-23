#!/bin/bash
GPU_ID=1
DATE=2023_6_13

#FLIR
DATASET_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/day/AB/
DATASET_DIR_B=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/A/day
DATASET_DIR_A=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day
SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/resnet101/segmentation

DATASET_NAME="FLIR"

CHECKPOINT_DIR=7

MODEL="AAFSTNet2Sobel"
STAGE="full"
OPTION="_RSGAN_Thermal"
GAN_TYPE="RSGAN"
RESULT_DIR="./results_7"
EPOCH="200"
CLAHE="" # --clahe
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="RGB"

SAL_MAP="sod" # saliency or sod or None
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD="--input_sod"    # --input_sod

USE_SEGMAP=""    # --use_segmap
USE_SEGMAP_PROB=1

LOADSIZE_W=256
LOADSIZE_H=256
FINESIZE_W=256
FINESIZE_H=256

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_resnet --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}

#mkdir -p ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_day_${EPOCH}
cp test_FLIR_YOLO_seg.sh "${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_day_${EPOCH}/."

wait

SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/empty/segmentation

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_empty --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}


# train val
DATASET_DIR=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/RGB
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train_val/A/day
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train_val/B/day
SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/empty/segmentation





# FLIR YOLO
DATASET_DIR=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/RGB
DATASET_DIR_A=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/thermal_8_bit
DATASET_DIR_B=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/night
SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/empty/segmentation
DATASET_NAME="FLIR"

LOADSIZE_W=360
LOADSIZE_H=288
FINESIZE_W=360
FINESIZE_H=288

#LOADSIZE_W=256
#LOADSIZE_H=256
#FINESIZE_W=256
#FINESIZE_H=256

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}


mkdir -p ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}
cp test_FLIR_YOLO_seg.sh "${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/."
#mkdir -p ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/resized_fake_B
#python resize_images.py ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/fake_B ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/resized_fake_B
