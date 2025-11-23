GPU_ID=3
DATE=2023_4_17

#FLIR
DATASET_DIR=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/day/AB/
DATASET_DIR_A=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/A/day
DATASET_DIR_B=${HOME}/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day
DATASET_NAME="FLIR"

CHECKPOINT_DIR=6

MODEL="AAFSTNet2Sobel"
STAGE="full"
OPTION="_unetD_ws5_sod_Lab"
GAN_TYPE="SGAN"
RESULT_DIR="./results_6/"
EPOCH="200"
CLAHE="" # --clahe
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="Lab"

SAL_MAP="sod" # --saliency or --sod
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD="--input_sod"    # --input_sod

LOADSIZE_W=256
LOADSIZE_H=256
FINESIZE_W=256
FINESIZE_H=256

#/home/usrs/ugawa/lab/work/PearlGAN/FLIR_testsets/test1

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD --stage full