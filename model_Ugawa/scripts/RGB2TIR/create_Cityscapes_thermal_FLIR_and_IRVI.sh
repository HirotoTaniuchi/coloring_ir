# 生成したCityscapesの擬似TIR画像が格納されたディレクトリをSOURCE_DIRに指定する(.../fake_Bという名前のディレクトリになると思います)
# Cityscapes Thermalデータセットを作るパスをTARGET_DIRに指定する
SOURCE_DIR=./results_RGB2TIR/2024_8_18_AAFSTNet2andSegNet2_NoSeg_FLIR_and_IRVI/AAFSTNet2andSegNet2_NoSeg_FLIR_and_IRVI_full_RSGAN_RGB2TIR/test_other_latest/fake_B
TARGET_DIR="./Cityscapes_Thermal/leftImg8bit/train" # trainデータセットのパスを指定する

mkdir -p ${TARGET_DIR}/aachen
mkdir -p ${TARGET_DIR}/bochum
mkdir -p ${TARGET_DIR}/bremen
mkdir -p ${TARGET_DIR}/cologne
mkdir -p ${TARGET_DIR}/darmstadt
mkdir -p ${TARGET_DIR}/dusseldorf
mkdir -p ${TARGET_DIR}/erfurt
mkdir -p ${TARGET_DIR}/hamburg
mkdir -p ${TARGET_DIR}/hanover
mkdir -p ${TARGET_DIR}/jena
mkdir -p ${TARGET_DIR}/krefeld
mkdir -p ${TARGET_DIR}/monchengladbach
mkdir -p ${TARGET_DIR}/strasbourg
mkdir -p ${TARGET_DIR}/stuttgart
mkdir -p ${TARGET_DIR}/tubingen
mkdir -p ${TARGET_DIR}/ulm
mkdir -p ${TARGET_DIR}/weimar
mkdir -p ${TARGET_DIR}/zurich

python map_to_original_structure.py --source_dir ${SOURCE_DIR} --target_dir ${TARGET_DIR}


# TARGET_DIR="/home/usrs/ugawa/lab/work/TICCGAN/Cityscapes_Thermal/leftImg8bit/val" # エラー訂正前
TARGET_DIR="./Cityscapes_Thermal/leftImg8bit/val" # valデータセットのパスを指定する # 訂正後

mkdir -p ${TARGET_DIR}/frankfurt
mkdir -p ${TARGET_DIR}/lindau
mkdir -p ${TARGET_DIR}/munster

python map_to_original_structure.py --source_dir ${SOURCE_DIR} --target_dir ${TARGET_DIR}
