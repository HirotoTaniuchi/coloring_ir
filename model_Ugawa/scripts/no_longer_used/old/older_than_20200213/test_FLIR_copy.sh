GPU_ID=0
DATE=2022_11_24

#FLIR
DATASET_DIR=/home/usrs/ugawa/lab/work/datasets/RoadScene/cropinfrared
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/RoadScene/cropinfrared
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/RoadScene/crop_HR_visible
DATASET_NAME="FLIR"

CHECKPOINT_DIR=6

MODEL="AAFSTNet2"
STAGE="full"
OPTION="_SGAN_unetD_wcon10_ws1"
GAN_TYPE="SGAN"
RESULT_DIR="./results_6/"
EPOCH="200"
CLAHE="" # --clahe
A_INVERT="" #--A_invert
A_INVERT_PROB="" # --A_invert_prob n
COLORSPACE="RGB"

LOADSIZE_W=320
LOADSIZE_H=256
FINESIZE_W=320
FINESIZE_H=256

#/home/usrs/ugawa/lab/work/PearlGAN/FLIR_testsets/test1

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE
