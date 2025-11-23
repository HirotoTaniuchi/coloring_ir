# PSNR、SSIM、LPIPS、FIDを計算し、画像ごとの結果をcsvに、平均をtxtに保存します

# FAKE_DIRには生成画像が入っているディレクトリのパスを指定してください
# REAL_DIRには実画像が入っているディレクトリのパスを指定してください
# FID_REAL_DIRにはFID計算用の実画像が入っているディレクトリのパスを指定してください(IRVIの場合はREAL_DIRと同じです)
# SAVE_DIRには結果を保存するディレクトリのパスを指定してください
FAKE_DIR=../results_1/2024_2_15_AAFSTNet2andSegNet2_IRVI_Traffic/AAFSTNet2andSegNet2_IRVI_Traffic_full_RSGAN/test_other_latest/fake_B
REAL_DIR=../real_images/IRVI_256x256/day/real_B
FID_REAL_DIR=../real_images/IRVI_256x256/day/real_B
SAVE_DIR=./exp1/IRVI_scores/day

# EXP_NAMEはSAVE_DIR以下に作成されるディレクトリとcsv,txtファイルの名前になります。モデル名+実験条件などが良いと思います
# GPU_IDは計算に使用するGPUのIDを指定してください
EXP_NAME=AAFSTNet2andSegNet2_IRVI_Traffic_full_RSGAN-test_other_latest
GPU_ID=1

# FID_DIMにはFIDの計算に使用する次元数を指定してください(IRVIの場合は768で固定です)
FID_DIM=768

mkdir -p $SAVE_DIR
python calculate_score.py --real_dir $REAL_DIR --fake_dir $FAKE_DIR --save_dir $SAVE_DIR --fid_real_dir $FID_REAL_DIR --exp_name $EXP_NAME --gpu_id $GPU_ID --fid_dim $FID_DIM --pixel_wise_metrics