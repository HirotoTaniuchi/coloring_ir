    # PSNR、SSIM、LPIPS、FIDを計算し、画像ごとの結果をcsvに、平均をtxtに保存します

# FAKE_DIRには生成画像が入っているディレクトリのパスを指定してください
# REAL_DIRには実画像が入っているディレクトリのパスを指定してください
# FID_REAL_DIRにはFID計算用の実画像が入っているディレクトリのパスを指定してください("/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/val/B/day_and_day_from_notAlignedTrain"を256x256にリサイズしたもの)
# "/home/usrs/ugawa/lab/work/TICCGAN2/real_images/FLIR_256x256/day/real_B_day_and_day_from_notAlignedTrain_bicubic" に用意してあります
# SAVE_DIRには結果を保存するディレクトリのパスを指定してください
FAKE_DIR=../results_2/2024_2_15_AAFSTNet2andSegNet2_FLIR/AAFSTNet2andSegNet2_FLIR_full_RSGAN/test_night_latest/fake_B
REAL_DIR=../real_images/FLIR_256x256/day/real_B 
FID_REAL_DIR=../real_images/FLIR_256x256/day/real_B_day_and_day_from_notAlignedTrain_bicubic
SAVE_DIR=./exp2/FLIR_scores/night

# EXP_NAMEはSAVE_DIR以下に作成されるディレクトリとcsv,txtファイルの名前になります。モデル名+実験条件などが良いと思います
# GPU_IDは計算に使用するGPUのIDを指定してください
EXP_NAME=AAFSTNet2andSegNet2_FLIR_full_RSGAN-test_night_latest
GPU_ID=1

# FID_DIMにはFIDの計算に使用する次元数を指定してください(FLIR夜間の場合は2048)
FID_DIM=2048

mkdir -p $SAVE_DIR
python calculate_score.py --real_dir $REAL_DIR --fake_dir $FAKE_DIR --save_dir $SAVE_DIR --fid_real_dir $FID_REAL_DIR --exp_name $EXP_NAME --gpu_id $GPU_ID --fid_dim $FID_DIM