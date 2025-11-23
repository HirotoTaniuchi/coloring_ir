##### input segmentation map ######
GPU_ID=0
DATE=2023_2_7


#Freiburg
#DATASET_NAME="Freiburg"
#DATASET_DIR="None"
#DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/Freiburg/cropped/test/A/day
#DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/Freiburg/cropped/test/B/day
#SEG_DIR=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/results/FLIR/Thermal_to_Seg/scratch/resnet101/segmentation
#SEG_MODEL="resnet101"
#LOADSIZE_W=256
#LOADSIZE_H=256
#FINESIZE_W=256
#FINESIZE_H=256

#FLIR
DATASET_NAME="FLIR"
DATASET_DIR=/home/usrs/ugawa/lab/work/datasets/MS2/sync_data/_2021-08-13-16-31-10/thr/left_uint8
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/MS2/sync_data/_2021-08-13-16-31-10/thr/left_uint8
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/MS2/sync_data/_2021-08-13-16-31-10/thr/left_uint8
SEG_DIR=/home/usrs/ugawa/lab/work/datasets/MS2/sync_data/_2021-08-13-16-31-10/thr/seg_from_thermal/segmentation
SEG_MODEL="resnet101"
LOADSIZE_W=640
LOADSIZE_H=256
FINESIZE_W=640
FINESIZE_H=256

CHECKPOINT_DIR=6
MODEL="AAFSTNet2Sobel"
STAGE="full"
#OPTION="_RSGAN_RGB_seg0.25_concat"
OPTION="_ws1_wcd1_sodIn"
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
USE_SEGMAP_PROB=1    # テスト時は1

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir /home/usrs/ugawa/lab/work/datasets/MS2/sync_data/_2021-08-13-16-31-10/thr/left_colorized --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}

cp test_seg.sh /home/usrs/ugawa/lab/work/datasets/MS2/sync_data/_2021-08-13-16-31-10/thr/left_colorized/test_seg.sh
wait

##### input empty map ######
#SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/empty/segmentation
#python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_empty --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}


# train val
#DATASET_DIR=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/RGB
#DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train_val/A/day
#DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train_val/B/day
#SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_segmentation_real/empty/segmentation


##### for Yolo Image #####
DATASET_NAME="FLIR"
DATASET_DIR=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/RGB
DATASET_DIR_A=${HOME}/lab/work/datasets/FLIR_ADAS_1_3/val/thermal_8_bit
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/FLIR_ADAS_1_3/val/RGB
SEG_DIR=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/results/FLIR/Thermal_to_Seg/scratch/resnet101/all/segmentation

LOADSIZE_W=360
LOADSIZE_H=288
FINESIZE_W=360
FINESIZE_H=288

python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${WORK}/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_${SEG_MODEL} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB}

mkdir -p ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}
cp test_seg.sh "${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/."
#mkdir -p ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/resized_fake_B
#python resize_images.py ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/fake_B ${WORK}/TICCGAN/results_6/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_all_${EPOCH}/resized_fake_B
