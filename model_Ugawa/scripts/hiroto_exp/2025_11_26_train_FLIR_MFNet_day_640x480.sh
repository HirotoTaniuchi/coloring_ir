##############################################################
# option                                                     #
##############################################################
CHECKPOINT_DIR="/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa/checkpoints"    # チェックポイントを保存するディレクトリ
GPU_ID=0    # 使用するGPUのID

# データセットのパス
# DATASET_DIR_AにTIR画像のディレクトリを指定してください
# DATASET_DIR_Bに可視画像のディレクトリを指定してください
# SEG_CKPT_PATHにセグメンテーションモデルのチェックポイントのパスを指定してください
DATASET_NAME=FLIR_MFNet_day
DATASET_DIR_A=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images_ir_str/day/train_val
DATASET_DIR_B=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images_rgb_str/day/train_val
SEG_CKPT_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa/checkpoints_segmentation/FLIR_IRVI_Finetune/latest_deeplabv3plus_resnet101_cityscapes_os16.pth

# あとは実行して大丈夫なはずです

DATE=2025_11_26    # 実験を開始した日付：年_月_日
MODEL="AAFSTNet2andSegNet2"    # モデルの名前
NET_D="basic"   # Discriminatorの種類：論文ではbasicしか使用しません
OPTION="640x480"    # モデルに関する追加情報：チェックポイントを識別するために記載します
STAGE="full"    # 学習時はfull
EXP_NAME="ugawa_ni_tekiyou"

INPUT_NC=3     # 入力チャネル数
OUTPUT_NC=3    # 出力チャネル数

# LOADSIZE=384   # 画像の読み込みサイズ
# FINESIZE=256   # クロップサイズ
LOADSIZE_W=640
LOADSIZE_H=480
FINESIZE_W=640
FINESIZE_H=480
TRANSFORM="randomcrop"    # クロップの方式 [randomcrop, centercrop]

# 重みなど
BATCH_SIZE=1   # バッチサイズ
L1="l1"         # コンテンツ損失として使用するロスの種類：論文は'l1' ['l1', 'charbonnier', 'PatchNCE']
W_L1=1          # --w_l1 (float) default:1
W_GAN=0.03      # --gan (float) default:0.03
W_TV=1          # --w_tv (float) default:1.0
W_VGG=1         # --w_vgg (float) default:1.0
W_SEG=0         # --w_seg
W_GAN_SEG=0.03  # --w_gan_seg

#フラグ類
GAN_TYPE="RSGAN"                    # 敵対的損失の種類：論文ではRSGAN [RSGAN, SGAN, LSGAN]
MIRROR="--mirror"                   # 入力画像にランダム水平反転を用いるか [--mirror]
OUT_GRAY=""                         # 出力画像をグレースケール化するか [--out_gray]
CLAHE=""                            # claheを入力画像に適用するか [--clahe]
USE_CONDITION="--use_condition 1"   # Discriminatorを使用するか：1の場合使用

# セグメンテーションモデル関連
USE_SEG_MODEL="--use_seg_model"     # セグメンテーションモデルを使用するか [--use_seg_model]
INPUT_TYPE_FOR_SEG_MODEL="real_A"   # セグメンテーションモデルへ入力する画像の種類：論文ではreal_Aを使用し [real_A, real_B]
SEG_NUM_CLASSES=19  # ラベルマップのクラス数：論文では19(0~19の20クラス)
SEG_TO_D="--seg_to_D"               # ラベルマップを判別器に入力するか [--seg_to_D]
SEG_TO_D_INST="--seg_to_D_inst"     # ラベルマップをクラス画像判別器に入力するか [--seg_to_D_inst]
SEG_TO_FAB="--seg_to_FAB"           # 着色モジュールにラベルマップを入力するか[--seg_to_FAB]

# 可視化関係
DISPLAY_FREQ=100
TENSORBOARD="--tensorboard"         # tensorboardで学習状況を可視化するか [--tensorboard]
TENSORBOARD_DIR=./tensorboard_log   # tensorboardのログを保存するディレクトリ
VISDOM_ENV=${DATE}_${MODEL}_${DATASET_NAME}${OPTION}
VISDOM_PORT=-1  # visdomのポート番号：-1の場合使用しない


mkdir -p ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION}
mkdir -p ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION}/${EXP_NAME}

# 学習を途中から再開する時に使用するオプション
CONTINUE_TRAIN="--continue_train"    # 学習を途中から再開するか [--continue_train]
EPOCH_COUNT="--epoch_count 101"    # 学習を再開する際、開始時を何エポック目とするか [--epoch_count (int)]
WHICH_EPOCH="--which_epoch 100"    # 何エポック目のチェックポイントをロードするか [--which_epoch (int)]
HALF_EPOCH=100    # 総エポック数の1/2を記入してください

# 既存チェックポイントのコピー元
SRC_CKPT_DIR="/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa/checkpoints/2024_8_20_AAFSTNet2andSegNet2_FLIR/AAFSTNet2andSegNet2_FLIR_full_RSGAN"
# 必要なファイルのみコピー
cp "${SRC_CKPT_DIR}"/100_net*.pth "${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION}/${EXP_NAME}/" 2>/dev/null || true
cp "${SRC_CKPT_DIR}"/*.txt "${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION}/${EXP_NAME}/" 2>/dev/null || true
cp "${SRC_CKPT_DIR}"/*.sh "${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION}/${EXP_NAME}/" 2>/dev/null || true

##############################################################
# command for trainnig
##############################################################

# model_Ugawa ディレクトリへ移動してから実行
MODEL_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa
cd "$MODEL_DIR"
cp scripts/hiroto_exp/2025_11_26_train_FLIR_MFNet_day_640x480.sh ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION}/${EXP_NAME}/2025_11_26_train_FLIR_MFNet_day_640x480.sh
python train.py --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_ckpt_path $SEG_CKPT_PATH --dataset_mode unaligned --name ${EXP_NAME} --gpu_ids $GPU_ID --batchSize $BATCH_SIZE --which_model_netG ${MODEL} --which_model_netD $NET_D --input_nc $INPUT_NC --output_nc $OUTPUT_NC --checkpoints_dir ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}_${OPTION} $EPOCH_COUNT $CONTINUE_TRAIN $WHICH_EPOCH --phase train --loadSize_W $LOADSIZE_W --loadSize_H $LOADSIZE_H --fineSize_W $FINESIZE_W --fineSize_H $FINESIZE_H --transform $TRANSFORM --l1 $L1 --w_l1 $W_L1 --w_gan $W_GAN --w_tv $W_TV --w_vgg $W_VGG --w_seg $W_SEG --w_gan_seg $W_GAN_SEG --gan_type $GAN_TYPE --niter $HALF_EPOCH --niter_decay $HALF_EPOCH --stage $STAGE  $TENSORBOARD --tensorboard_dir $TENSORBOARD_DIR --display_env $VISDOM_ENV --display_port $VISDOM_PORT $MIRROR $OUT_GRAY $USE_CONDITION --display_freq $DISPLAY_FREQ  $SEG_TO_D $SEG_TO_D_INST --seg_num_classes $SEG_NUM_CLASSES $USE_SEG_MODEL --input_type_for_seg_model $INPUT_TYPE_FOR_SEG_MODEL $SEG_TO_FAB $CLAHE