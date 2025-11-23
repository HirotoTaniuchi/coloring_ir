GPU_ID=1
DATE=2023_4_10

#Kaist
DATASET_DIR=${HOME}/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/test_day/AB
DATASET_DIR_A=${HOME}/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/test_day/testA
DATASET_DIR_B=${HOME}/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/test_day/testB
DATASET_NAME="KAIST"

CHECKPOINT_DIR=6

MODEL="AAFSTNet2Sobel"
STAGE="full"
OPTION="_unetD_ws5_sodin"
GAN_TYPE="RSGAN"
RESULT_DIR="./results_6/"
EPOCH="latest"
CLAHE=""
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="RGB"

SAL_MAP="sod" # --saliency or --sod
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD="--input_sod"    # --input_sod

LOADSIZE_W=256
LOADSIZE_H=256
FINESIZE_W=256
FINESIZE_H=256

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD
#python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode aligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $A_INVERT $A_INVERT_PROB $CLAHE --colorspace $COLORSPACE
