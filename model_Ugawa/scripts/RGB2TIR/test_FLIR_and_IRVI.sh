##############################################################
# option                                                     #
##############################################################
# パスなど
CHECKPOINT_DIR="./checkpoints_RGB2TIR"    # チェックポイントを保存したディレクトリ
GPU_ID=0   # 使用するGPUのID
RESULT_DIR="./results_RGB2TIR"   # 結果を保存するディレクトリ 

# データセット関連
# CityscapesのRGBのTrain画像を一つのディレクトリにまとめ、それをDATASET_DIR_Aに指定する(DATA_DIR_Bは適当でよい)
# SEG_CKPT_PATHはセグメンテーションモデルのチェックポイントだが、ここでは使用しないため適当なもので良い
DATASET_NAME=FLIR_and_IRVI
DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/Cityscapes/RGB/leftImg8bit_trainvaltest/leftImg8bit/train_all
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/Cityscapes/RGB/leftImg8bit_trainvaltest/leftImg8bit/train_all
SEG_CKPT_PATH=./checkpoints_segmentation/FLIR_IRVI_Finetune/latest_deeplabv3plus_resnet101_cityscapes_os16.pth
SEG_MODEL="resnet101"

# モデルの情報：学習時と同じに設定してください
DATE=2024_8_18     # 実験を開始した日付：年_月_日
MODEL="AAFSTNet2andSegNet2_NoSeg"     # モデルの名前
OPTION="_RSGAN_RGB2TIR"     # モデルに関する追加情報：チェックポイントを識別するために記載
STAGE="full"    # fullに設定してください   
EPOCH="latest"     # ロードするエポック

# フラグ類
CLAHE=""    # claheを入力画像に適用するか [--clahe]

# 出力・クロップサイズ
LOADSIZE_W=1024
LOADSIZE_H=512
FINESIZE_W=1024
FINESIZE_H=512

# セグメンテーションモデル関連
USE_SEG_MODEL=""   # --use_seg_model
INPUT_TYPE_FOR_SEG_MODEL="real_A" # real_A or real_B
SEG_TO_FAB=""    # --seg_to_FAB
SEG_NUM_CLASSES=19  # --seg_num_classes (int)

python test.py --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_ckpt_path $SEG_CKPT_PATH --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 $CLAHE --stage ${STAGE} $USE_SEG_MODEL --input_type_for_seg_model $INPUT_TYPE_FOR_SEG_MODEL $SEG_TO_FAB --seg_num_classes $SEG_NUM_CLASSES $CLAHE

DATASET_DIR_A=/home/usrs/ugawa/lab/work/datasets/Cityscapes/RGB/leftImg8bit_trainvaltest/leftImg8bit/val_all
DATASET_DIR_B=/home/usrs/ugawa/lab/work/datasets/Cityscapes/RGB/leftImg8bit_trainvaltest/leftImg8bit/val_all

python test.py --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_ckpt_path $SEG_CKPT_PATH --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 $CLAHE --stage ${STAGE} $USE_SEG_MODEL --input_type_for_seg_model $INPUT_TYPE_FOR_SEG_MODEL $SEG_TO_FAB --seg_num_classes $SEG_NUM_CLASSES $CLAHE

cp scripts/RGB2TIR/test_FLIR_and_IRVI.sh ${RESULT_DIR}/${DATE}_${MODEL}_${DATASET_NAME}/${MODEL}_${DATASET_NAME}_${STAGE}${OPTION}/test_FLIR_and_IRVI.sh
