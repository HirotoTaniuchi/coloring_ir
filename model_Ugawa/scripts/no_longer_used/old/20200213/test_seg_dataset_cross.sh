# 10月22日にAAFSTNet2andSegNetのSRBからOutconvを削除したので、それ以前のモデルをロードするときは注意

##### input segmentation map ######
GPU_ID=2
DATE=2023_10_23
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

DATASET_NAME="FLIR"
DATASET_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/day/AB/
#DATASET_DIR_A=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/A/day
#DATASET_DIR_B=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day
SEG_DIR=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/results/FLIR/Thermal_to_Seg/scratch/resnet101/day/segmentation
SEG_CKPT_PATH=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/checkpoints_scratch_Thermal/FLIR/best_deeplabv3plus_resnet101_cityscapes_os16.pth
SEG_MODEL="resnet101"

# KAIST
#DATASET_NAME="KAIST"
#DATASET_DIR=/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/test_night/AB/test
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/test_day/testA
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/test_day/testB
#SEG_DIR=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/results/KAIST/TIR_to_Seg/scratch/segmentation
#SEG_MODEL="resnet101"
#SEG_CKPT_PATH=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/checkpoints/KAIST_finetune_40000/latest_deeplabv3plus_resnet101_cityscapes_os16.pth

# FLIRandIRVI_Traffic
#DATASET_NAME=FLIRandIRVI_Traffic
#DATASET_DIR="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/day"
#DATASET_DIR_A="/home/usrs/ugawa/lab/work/datasets/FLIR_and_IRVI/train/trainB"
#DATASET_DIR_B="/home/usrs/ugawa/lab/work/datasets/FLIR_and_IRVI/train/trainA"
#SEG_DIR="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/trainB_seg/2023_7_25/segmentation"
#SEG_CKPT_PATH=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/checkpoints/KAIST_finetune_40000/latest_deeplabv3plus_resnet101_cityscapes_os16.pth


# IRVI
#DATASET_NAME=IRVI_Traffic
#DATASET_DIR="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/day"
#DATASET_DIR_A="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/testA"
#DATASET_DIR_B="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/testB"
#SEG_DIR="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/testA_seg/2023_7_25/scratch/segmentation"
#SEG_MODEL="resnet101"
#SEG_CKPT_PATH=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/checkpoints/FLIR_IRVI_Finetune_50000/latest_deeplabv3plus_resnet101_cityscapes_os16.pth
##SEG_CKPT_PATH=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/checkpoints_scratch_Thermal/FLIR/best_deeplabv3plus_resnet101_cityscapes_os16.pth
##SEG_CKPT_PATH=/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/checkpoints/KAIST_finetune_40000/latest_deeplabv3plus_resnet101_cityscapes_os16.pth

#DATASET_DIR_A="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/valA_1"
#DATASET_DIR_B="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/valB_1"

 
USE_SEG_MODEL="--use_seg_model"   # --use_seg_model
INPUT_TYPE_FOR_SEG_MODEL="real_A" # real_A or real_B

LOADSIZE_W=256
LOADSIZE_H=256
FINESIZE_W=256
FINESIZE_H=256

CHECKPOINT_DIR=8
MODEL="AAFSTNet2andSegNet2"
STAGE="full"    
OPTION="_RSGAN_LabelToFAB" 
GAN_TYPE="RSGAN"
RESULT_DIR="./results_cross"    
EPOCH="100"     
CLAHE=""    # --clahe
A_INVERT=""    #--A_invert
A_INVERT_PROB=""    # --A_invert_prob n
COLORSPACE="RGB"
                
SAL_MAP="None"    # saliency or sod or None
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD=""    # --input_sod
                
USE_SEGMAP=""    # --use_segmap
USE_SEGMAP_PROB=1    # テスト時は1
SEG_TO_FAB="--seg_to_FAB"    # --seg_to_FAB
SEG_MASK_SIZE=16   # テスト時は適当な値（どうせ使わないので）
SEG_MASK_RATIO=0   # テスト時は0
SEG_FILTER=""    # --seg_filter
SEG_FILTER_RADIUS=2
SEG_NUM_CLASSES=19  # --seg_num_classes (int)


python test.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${HOME}/lab/work/TICCGAN/checkpoints${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_${SEG_MODEL} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 --gan_type $GAN_TYPE $CLAHE $A_INVERT $A_INVERT_PROB --colorspace $COLORSPACE --sal_map $SAL_MAP $NORMALIZE_SAL $SAL_MASK $INPUT_SOD $USE_SEGMAP --stage ${STAGE} --use_segmap_prob ${USE_SEGMAP_PROB} --seg_mask_size $SEG_MASK_SIZE --seg_mask_ratio $SEG_MASK_RATIO $SEG_FILTER --seg_filter_radius $SEG_FILTER_RADIUS --input_type_for_seg_model $INPUT_TYPE_FOR_SEG_MODEL --seg_ckpt_path $SEG_CKPT_PATH $USE_SEG_MODEL $SEG_TO_FAB --seg_num_classes $SEG_NUM_CLASSES $CLAHE

cp test_seg_dataset_cross.sh $RESULT_DIR/${DATE}_${MODEL}_${DATASET_NAME}_${SEG_MODEL}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_seg.sh
