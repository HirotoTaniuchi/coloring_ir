# option
##############################################################
GPU_ID=0
DATE=2023_9_13
MODEL="AAFSTNet2"    # model name (str)
NET_D="basic"
OPTION=""    # additional information like 「wl_0.5」    _seg1_conv_segFilter10
STAGE="full"    # full or global
N_LAYERS_D=3 #--n_layers_D

#DATASET_NAME=FLIR
#DATASET_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train/day/AB/train
#DATASET_DIR_A=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train/A/day
#DATASET_DIR_B=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train/B/day
#SEG_DIR=$HOME/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train/B/day_segmentation/resnet101/segmentation

#DATASET_NAME=Freiburg
#DATASET_DIR=None
#DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/Freiburg/cropped/train/A/day
#DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/Freiburg/cropped/train/B/day
#SEG_DIR=/home/usrs/ugawa/lab/work/datasets/Freiburg/cropped/train/B/day_segmentation/gt

#DATASET_NAME=KAIST
#DATASET_DIR="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/AB/train"
#DATASET_DIR_A="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/trainA"
#DATASET_DIR_B="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/trainB"
#SEG_DIR="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/trainB_segmentation/resnet101/segmentation"

DATASET_NAME=IRVI_Traffic
DATASET_DIR="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/day"
DATASET_DIR_A="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/trainA"
DATASET_DIR_B="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/trainB"
SEG_DIR="/home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic/trainB_seg/2023_7_25/segmentation"

INPUT_NC=3
OUTPUT_NC=3

NTHREADS=2   # --nThreads default:2
LOADSIZE=256 # 384
FINESIZE=256
TRANSFORM="randomcrop"    # centercrop randomcrop

#重みなど
L1="charbonnier"    #charbonnier
W_L1=1    # --w_l1 (float) default:1
W_GAN=0.03    # --gan (float) default:0.03
W_TV=1    # --w_tv (float) default:1.0
W_VGG=1    # --w_vgg (flaot) default:1.0
W_EDGE=0    # --w_edge (float) default:0
W_L1_INST=0    # --w_l1_inst (float)
W_GAN_INST=0    # --w_gan_inst (float)
W_VGG_INST=0    # --w_vgg_inst (float)
W_CX=0    # --w_cx (float)
W_CX_INST=0    # --w_cx_inst
W_reg=0    # --w_reg
W_CON=0    # --w_con
W_SAL=0    # --w_sal
W_CD=0    # --w_CD
W_SEG=0    # --w_seg
BBOX=512    # --bbox_threshold (int)

#フラグ類
#NOLSGAN="--no_lsgan"    # store_true
GAN_TYPE="RSGAN"
D_LABEL="--D_label normal"    # [normal, noisy, noisy_soft]
D_INST_LABEL="--D_inst_label normal"    # [normal, noisy, noisy_soft]
NEED_BBOX=""    # store_true
MIRROR="--mirror"    # store_true --mirror
TENSORBOARD="--tensorboard"    # store_true
WANDB=""    # store_true
A_INVERT="" #--A_invert
A_INVERT_PROB="" #--A_invert_prob 0.5
OUT_GRAY="" #--out_gray
CLAHE=""
COLORSPACE="RGB"
L1_CHANNEL="all"
L1_BGONLY=""    # store_true --l1_bgOnly
USE_CONDITION="--use_condition 1" #--use_condition 1

SAL_MAP="None" # saliency or sod or None
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD=""    # --input_sod

USE_SEGMAP=""    # --use_segmap
USE_SEGMAP_PROB=0.5    # --use_segmap_prob (float)
SEG_MASK_SIZE=8    # --seg_mask_size (int)
SEG_MASK_RATIO=0  # --seg_mask_ratio (float)
SEG_TO_D=""  # --seg_to_D
SEG_FILTER=""    # --seg_filter
SEG_FILTER_RADIUS=2    # --seg_filter_radius (int)

DISPLAY_FREQ=100
VISDOM_ENV=${DATE}_${MODEL}_${DATASET_NAME}${OPTION}
VISDOM_PORT=-1

mkdir -p ${LOCALTMP}/checkpoints/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}

#EPOCH_COUNT="--epoch_count 51"    # --epoch_count (int)
#CONTINUE_TRAIN="--continue_train"    # store_trues --continue_train
#WHICH_EPOCH="--which_epoch latest"    # --which_epoch (int)
EPOCH_COUNT=""    # --epoch_count (int)
CONTINUE_TRAIN=""    # store_trues --continue_train
WHICH_EPOCH=""    # --which_epoch (int)

HALF_EPOCH=10


##############################################################
# command for trainnig
##############################################################

cp train_seg.sh ${LOCALTMP}/checkpoints/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/train_seg.sh
wait

python train.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_dir $SEG_DIR --dataset_mode unaligned --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --gpu_ids $GPU_ID --which_model_netG ${MODEL} --which_model_netD $NET_D --nThreads $NTHREADS --input_nc $INPUT_NC --output_nc $OUTPUT_NC --checkpoints_dir ${LOCALTMP}/checkpoints/${DATE}_${MODEL}_${DATASET_NAME} $EPOCH_COUNT $CONTINUE_TRAIN $WHICH_EPOCH $NEED_BBOX $D_LABEL $D_INST_LABEL --phase train --loadSize_W $LOADSIZE --loadSize_H $LOADSIZE --fineSize_W $FINESIZE --fineSize_H $FINESIZE --transform $TRANSFORM --l1 $L1 --w_l1 $W_L1 --w_gan $W_GAN --w_tv $W_TV --w_vgg $W_VGG --w_edge $W_EDGE --w_cx $W_CX --w_con $W_CON --w_l1_inst $W_L1_INST --w_gan_inst $W_GAN_INST --w_vgg_inst $W_VGG_INST --w_cx_inst $W_CX_INST --w_reg $W_reg --w_sal $W_SAL --w_cd $W_CD --w_seg $W_SEG --gan_type $GAN_TYPE --niter $HALF_EPOCH --niter_decay $HALF_EPOCH --bbox_threshold $BBOX --stage $STAGE  $TENSORBOARD $WANDB --display_env $VISDOM_ENV --n_layers_D $N_LAYERS_D --display_port $VISDOM_PORT $MIRROR $A_INVERT $A_INVERT_PROB $OUT_GRAY --sal_map $SAL_MAP $NORMALIZE_SAL $INPUT_SOD $SAL_MASK $USE_CONDITION --colorspace $COLORSPACE --l1_channel $L1_CHANNEL $L1_BGONLY --display_freq $DISPLAY_FREQ $USE_SEGMAP --use_segmap_prob $USE_SEGMAP_PROB --seg_mask_size $SEG_MASK_SIZE --seg_mask_ratio $SEG_MASK_RATIO $SEG_TO_D $SEG_FILTER --seg_filter_radius $SEG_FILTER_RADIUS
