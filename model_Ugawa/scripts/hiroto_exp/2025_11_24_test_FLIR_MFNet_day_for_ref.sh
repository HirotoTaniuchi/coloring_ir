## 中間発表用サンプル画像作成のため、訓練済み宇川モデルにMFNetを入力する

##############################################################
# option                                                     #
##############################################################
# パスなど
CHECKPOINT_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa/checkpoints    # チェックポイントを保存したディレクトリ
GPU_ID=0    # 使用するGPUのID
## ⭐️ここを変更
RESULT_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/output_img/Ugawa_2025_11_24_AAFSTNet2andSegNet2_FLIR_MFNet_day_for_ref  # 結果を保存するディレクトリ 

# データセットのパス
# DATASET_DIR_AにTIR画像のディレクトリを指定してください
# DATASET_DIR_Bに可視画像のディレクトリを指定してください
# SEG_CKPT_PATHにセグメンテーションモデルのチェックポイントのパスを指定してください
DATASET_NAME=FLIR_MFNet_day
## ⭐️ここを変更 ## *A:TIR, B:RGB
DATASET_DIR_A=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images_ir_str/day/train_val
DATASET_DIR_B=/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images_rgb_str/day/train_val
SEG_CKPT_PATH=/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa/checkpoints_segmentation/FLIR_IRVI_Finetune/latest_deeplabv3plus_resnet101_cityscapes_os16.pth

# モデルの情報：学習時と同じに設定してください
DATE=2025_11_19     # 実験を開始した日付：年_月_日
MODEL="AAFSTNet2andSegNet2"     # モデルの名前
OPTION="_RSGAN"     # モデルに関する追加情報：チェックポイントを識別するために記載
STAGE="full"    # fullに設定してください   
EPOCH="latest"     # ロードするエポック

# フラグ類
CLAHE=""    # claheを入力画像に適用するか [--clahe]
SKIP_REALS="--skip_reals"  # real_A / real_B を保存しない (save_images skip_reals=True)
SKIP_4CH=""                # fake_4ch を生成しない場合 --skip_4ch

# 出力・クロップサイズ
## ⭐️ここを変更
LOADSIZE_W=640
LOADSIZE_H=480
FINESIZE_W=640
FINESIZE_H=480

# セグメンテーションモデル関連
USE_SEG_MODEL="--use_seg_model"   # --use_seg_model
INPUT_TYPE_FOR_SEG_MODEL="real_A" # real_A or real_B
SEG_TO_FAB="--seg_to_FAB"    # --seg_to_FAB
SEG_NUM_CLASSES=19  # --seg_num_classes (int)

# model_Ugawa ディレクトリへ移動してから実行
MODEL_DIR=/home/usrs/taniuchi/workspace/projects/coloring_ir/model_Ugawa
cd "$MODEL_DIR"

python test.py --dataroot_A $DATASET_DIR_A --dataroot_B $DATASET_DIR_B --seg_ckpt_path $SEG_CKPT_PATH --dataset_mode unaligned --loadSize_H $LOADSIZE_H --loadSize_W $LOADSIZE_W --fineSize_H $FINESIZE_H --fineSize_W $FINESIZE_W --which_model_netG $MODEL --gpu_ids $GPU_ID --checkpoints_dir ${CHECKPOINT_DIR}/${DATE}_${MODEL}_${DATASET_NAME} --name ${MODEL}_${DATASET_NAME}_${STAGE}${OPTION} --results_dir $RESULT_DIR --phase test --which_epoch $EPOCH --serial_batches --how_many 29178 $CLAHE --stage ${STAGE} $USE_SEG_MODEL --input_type_for_seg_model $INPUT_TYPE_FOR_SEG_MODEL $SEG_TO_FAB --seg_num_classes $SEG_NUM_CLASSES $SKIP_REALS $SKIP_4CH

cp scripts/hiroto_exp/2025_11_24_test_FLIR_MFNet_day_for_ref.sh ${RESULT_DIR}/2025_11_24_test_FLIR_MFNet_day_for_ref.sh