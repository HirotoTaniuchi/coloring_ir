# option
##############################################################
GPU_ID=0
DATE=2023_4_10
MODEL="AAFSTNet2Sobel"    # model name (str)
NET_D="unet"
N_LAYERS_D=3 #--n_layers_D

DATASET_DIR="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/AB/train"
DATASET_DIR_A="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/trainA"
DATASET_DIR_B="/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy/train_day/trainB"
DATASET_NAME=KAIST

INPUT_NC=3
OUTPUT_NC=3

NTHREADS=2   # --nThreads default:2
LOADSIZE=512
FINESIZE=256
TRANSFORM="randomcrop"    # centercrop randomcrop

#重みなど
L1="charbonnier"    #charbonnier
W_L1=1    # --w_l1 (float) default:1
W_GAN=0.03    # --gan (float) default:0.03
W_TV=1    # --w_tv (float) default:1.0
W_VGG=1    # --w_vgg (flaot) default:1.0
W_EDGE=0    # --w_edge (flopipat) default:0
W_L1_INST=1    # --w_l1_inst (float)
W_GAN_INST=0.03    # --w_gan_inst (float)
W_VGG_INST=0    # --w_vgg_inst (float)
W_CX=0    # --w_cx (float)
W_CX_INST=0    # --w_cx_inst
W_reg=0    # --w_reg
W_CON=10    # --w_con
W_SAL=5    # --w_sal
W_CD=0    # --w_CD
BBOX=512    # --bbox_threshold (int)

#フラグ類
#NOLSGAN="--no_lsgan"    # store_true
GAN_TYPE="SGAN"
D_LABEL="--D_label normal"    # [normal, noisy, noisy_soft]
D_INST_LABEL="--D_inst_label normal"    # [normal, noisy, noisy_soft]
NEED_BBOX=""    # store_true
MIRROR="--mirror"    # store_true --mirror
TENSORBOARD="--tensorboard"    # store_true
WANDB=""    # store_true
A_INVERT="" #--A_invert
A_INVERT_PROB="" #--A_invert_prob 0.5
OUT_GRAY="" #--out_gray
SAL_MAP="sod" # --saliency or --sod
CLAHE=""
COLORSPACE="RGB"
USE_CONDITION="--use_condition 1" #--use_condition 1
NORMALIZE_SAL=""    # --normalize_sal
SAL_MASK=""     # --sal_mask
INPUT_SOD="--input_sod"    # --input_sod

VISDOM_ENV=${DATE}_${MODEL}_${DATASET_NAME}${OPTION}
VISDOM_PORT=-1
OPTION="_unetD_ws5_sodin"    # additional information like 「wl_0.5」

mkdir -p ${LOCALTMP}/checkpoints/${DATE}_${MODEL}_${DATASET_NAME}

EPOCH_COUNT="--epoch_count 9"    # --epoch_count (int)
CONTINUE_TRAIN="--continue_train"    # store_trues --continue_train
WHICH_EPOCH="--which_epoch latest"    # --which_epoch (int)
##############################################################


# command for trainnig
##############################################################
# full
python train.py --dataroot $DATASET_DIR --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --dataset_mode unaligned --name ${MODEL}_${DATASET_NAME}_full${OPTION} --gpu_ids $GPU_ID --which_model_netG ${MODEL} --which_model_netD $NET_D --nThreads $NTHREADS --input_nc $INPUT_NC --output_nc $OUTPUT_NC --checkpoints_dir ${LOCALTMP}/checkpoints/${DATE}_${MODEL}_${DATASET_NAME} $EPOCH_COUNT $CONTINUE_TRAIN $WHICH_EPOCH $NEED_BBOX $D_LABEL $D_INST_LABEL --phase train --loadSize_W $LOADSIZE --loadSize_H $LOADSIZE --fineSize_W $FINESIZE --fineSize_H $FINESIZE --transform $TRANSFORM --l1 $L1 --w_l1 $W_L1 --w_gan $W_GAN --w_tv $W_TV --w_vgg $W_VGG --w_edge $W_EDGE --w_cx $W_CX --w_con $W_CON --w_l1_inst $W_L1_INST --w_gan_inst $W_GAN_INST --w_vgg_inst $W_VGG_INST --w_cx_inst $W_CX_INST --w_reg $W_reg --w_sal $W_SAL --w_cd $W_CD --gan_type $GAN_TYPE --niter 50 --niter_decay 50 --bbox_threshold $BBOX --stage full  $TENSORBOARD $WANDB --display_env $VISDOM_ENV --n_layers_D $N_LAYERS_D --display_port $VISDOM_PORT $MIRROR $A_INVERT $A_INVERT_PROB $OUT_GRAY --sal_map $SAL_MAP $NORMALIZE_SAL $INPUT_SOD $SAL_MASK $USE_CONDITION --colorspace $COLORSPACE 
#--no_D_enc --use_condition 1

# 5/24以降はloss_G_gan_instがnonzero / 256*256で重みづけされている