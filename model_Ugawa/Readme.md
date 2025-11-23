# 使用方法・再現実験

※notionにあるマニュアルが最新版です。可能な限りそちらを参照してください。

# 熱赤外線(TIR)画像着色フレームワーク

## 環境

下記コマンドを実行して環境を作成します。

```bash
conda env create -n [env_name] -f TIR2Color_env.yaml
```

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url <https://download.pytorch.org/whl/cu116>
```

CUDAのバージョンは11.6です。

実験はほとんどdlbox01で実行しました。

## データセットの準備

"/home/usrs/ugawa/lab/work/datasets"に保存されているデータセットを使用すること想定しています。

一からデータセットを用意する場合は必要なデータセットをURLからダウンロードしてください。(FLIR Alignedはリンク切れのためURL無し)

- Cityscapes(セグメンテーションモジュールの学習に使用)
    - /home/usrs/ugawa/lab/work/datasets/Cityscapes/thermal
    - [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)
        - "gtFine_trainvaltest.zip"と"leftImg8bit_trainvaltest.zip"を使用
- KAIST Multispectral Pedestrian Benchmark
    - /home/usrs/ugawa/lab/work/datasets/kaist-cvpr15_TICCGAN/copy
    - [https://github.com/SoonminHwang/rgbt-ped-detection](https://github.com/SoonminHwang/rgbt-ped-detection)
- IRVI Traffic
    - /home/usrs/ugawa/lab/work/datasets/IRVI/single/traffic
    - [https://drive.google.com/file/d/1ZcJ0EfF5n_uqtsLc7-8hJgTcr2zHSXY3/view](https://drive.google.com/file/d/1ZcJ0EfF5n_uqtsLc7-8hJgTcr2zHSXY3/view)
- FLIR Aligned
    - /home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages

大抵、"A"と書かれたディレクトリにTIR画像が、"B"というディレクトリに可視光画像が入っています。

学習用TIR画像、学習用可視光画像、テスト用TIR画像、テスト用可視光画像が別々のディレクトリに分かれていればとりあえずOKです。

FLIRとKAISTは昼間に撮影したデータ(day)と夜間に撮影したデータ(night)があります。IRVI Trafficは昼間データのみです。

※FLIR AlignedはFLIR Thermal Dataset v1.3をTIR-可視光ペア画像になるように画素を調整したデータセットです。
FLIR Thermal Dataset v1.3は [ここ](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset) からダウンロード可能です。

## 大まかな使い方

各実験に合わせたshellスクリプトを実行すると、学習やテストが行われるようにしています。shellスクリプト内のパスをいくつか書き換える必要があるので注意してください。

### セグメンテーションモジュールの準備

セグメンテーションモジュールの準備は

1. 可視光→TIR変換モデルを学習
2. Cityscapesデータセットを擬似TIR画像化
3. 擬似TIR画像でDeepLab v3+を学習

という流れです。

なお、セグメンテーションモジュール用のチェックポイントはKAISTとFLIR兼IRVI用で2種類用意します。

1~3の手順でセグメンテーションモジュール用のチェックポイントを用意してください。(FLIRは単体で用意しても良いですが、IRVIは単体だとRGB→TIRがうまく変換できません)

1と2を省略する場合は"/home/usrs/ugawa/lab/work/datasets/Cityscapes/thermal"をセグメンテーションモジュール学習用のデータセットとして利用してください。".../Cityscapes/thermal/leftImg8bit/train"には現在FLIRとIRVIで学習した擬似TIR画像が格納されているので、KAIST用のセグメンテーションモデルを学習する場合はtrain_KAISTをtrainという名前にリネームしてください。

### 1. 可視光→TIR変換モデルを学習(FLIR兼IRVI用のセグメンテーションモデルの場合)

"scripts/RGB2TIR/train_FLIR_and_IRVI.sh"を実行し、RGB→TIRの変換を学習したモデルを用意します。

### 2. Cityscapesデータセットを擬似TIR画像化

"scripts/RGB2TIR/test_FLIR_and_IRVI.sh"を実行し、学習したモデルを用いてCityscapesデータセットのTrain画像(RGB)をTIR画像に変換します。

次に"scripts/RGB2TIR/create_Cityscapes_thermal_dir.sh"を実行し、Cityscapesデータセットのディレクトリ構造に合わせたデータセット(Cityscapes thermal)を作成します。

最後に作成したデータセットのディレクトリの"leftImg8bit"がある階層にRGB画像用のアノテーションが入ったディレクトリ("gtFine"と書かれたディレクトリ)をコピーします。("/home/usrs/ugawa/lab/work/datasets/Cityscapes/RGB/gtFine"をコピーすればOK)

"create_Cityscapes_thermal_FLIR_and_IRVI.sh"を実行すると、ファイル内でSOURCE_DIRに指定したディレクトリの画像がTARGET_DIRにCityscapesデータセットの構造に従って配置されます。デフォルトのパスだと”/home/usrs/ugawa/lab/work/TICCGAN/Cityscapes_Thermal”下に作成されるはずです。

### 3. 擬似TIR画像でDeepLab v3+を学習

DeepLab v3+はgithubからcloneできます([https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch))が、学習用のコマンドを書いたファイルが無い&ソースコードの細かい部分をいじっているので"/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch"を利用することをおすすめします。

宇川のディレクトリを利用する場合は"/home/usrs/ugawa/lab/work/DeepLabV3Plus-Pytorch/train_finetuning.sh"を実行してください。学習後、チェックポイントは"...DeepLabV3Plus-Pytorch/checkpoints"直下に保存されます。適宜別ディレクトリに移動したり、リネームするなどして上書きされないようにしてください。
"/home/usrs/ugawa/lab/work/TICCGAN/checkpoints_segmentation/FLIR_IRVI_Finetune"または"/home/usrs/ugawa/lab/work/TICCGAN/checkpoints_segmentation/KAIST_finetune"下に移動しておけば他のshファイルの実行時に後述の"SEG_CKPT_PATH"を書き換えなくても良い…はずです。
得られたチェックポイントを以後の着色モデルの学習に使用していきます。

宇川のディレクトリを利用しない場合は

```bash
DATAROOT="TIR化したCityscapesデータセットへのパス"
CKPT="RGB画像で学習済みのチェックポイントへのパス"

MODEL="deeplabv3plus_resnet101"
DATASET="cityscapes"
GPU_ID=0

python main.py --model $MODEL --dataset $DATASET --vis_port 28333 --gpu_id $GPU_ID  --lr 0.1 --lr_policy step  --crop_size 768 --batch_size 4 --output_stride 16 --data_root $DATAROOT --continue_training --ckpt $CKPT --total_itrs 40000
```

以上のような具合でモデルを学習してください。

### ※注意

"create_Cityscapes_thermal_FLIR_and_IRVI.sh"と"create_Cityscapes_thermal_KAIST.sh"は(TARGET_DIRを変更しない場合)実行時にお互いに画像を上書きし合う関係にあります。例えばKAIST用のモデルを学習したい場合はKAIST用の2→3の手順を**連続して**行なってください。

### 学習

学習は".../TICCGAN/scripts/train_〇〇.sh"(〇〇はデータセット名)を実行すると開始されます。

各種shファイルは"/home/usrs/ugawa/lab/work/TICCGAN"下での実行を想定してパスなどが入力されています。そのまま実行すれば"/home/usrs/ugawa/lab/work/TICCGAN"に結果などが保存されるようになっています。

もし、他の環境で実行する場合は実行前に以下の変数を必要に応じて合わせて書き換えてください。

- GPU_ID (使用するGPUの番号)
- DATASET_A (TIRを直接格納しているディレクトリのパス)
- DATASET_B (可視光画像を直接格納しているディレクトリのパス)
- SEG_CKPT_PATH (セグメンテーションモデルのチェックポイントのパス)

shellファイルの書き換えが終わったら、".../TICCGAN"下で以下を実行してください。(〇〇はデータセット名)

学習が開始されます。

```bash
sh scripts/expN/train_〇〇.sh
```

実行すると".../TICCGAN/checkpoints"にディレクトリが作成され、チェックポイントが保存されます。
論文ではFLIRは100エポック、KAISTとIRVIは20エポック学習しています。

### テスト

テストは".../TICCGAN/scripts/test_〇〇.sh"(〇〇はデータセット名)を実行すると開始されます。

実行する前に以下の変数を各自の環境に合わせて書き換えてください。

- GPU_ID (使用するGPUの番号)
- DATASET_A (TIRを直接格納しているディレクトリのパス)
- DATASET_B (可視光画像を直接格納しているディレクトリのパス)
- SEG_CKPT_PATH (セグメンテーションモデルのチェックポイントのパス)
- CHECKPOINT_DIR (チェックポイントを保存しているディレクトリのパス)
- RESULT_DIR (生成結果を保存するディレクトリのパス)

shellファイルの書き換えが終わったら、".../TICCGAN"下で以下を実行してください。(〇〇はデータセット名)

テストが開始されます。

```bash
sh scripts/expN/test_〇〇.sh
```

実行すると".../TICCGAN/results"内にディレクトリが作成され、出力結果が格納されます。"real_A"がTIR画像、"real_B"が実可視光画像、"fake_B"が生成画像です。

### 評価

定量評価用のプログラムは".../TICCGAN/evaluation"に格納されています。
".../TICCGAN/evaluation"下で以下を実行してください。(◯◯は評価したいデータセット名、△△は昼間TIR画像の実験では"day"、夜間TIR画像の実験では"night")

それぞれのshスクリプト内の"REAL_DIR"を可視光の実画像(256x256)のディレクトリのパスに、"FAKE_DIR”を生成画像のディレクトリのパスに設定してください。"SAVE_DIR"は評価結果を保存するディレクトリです。
評価用の画像は”/home/usrs/ugawa/lab/work/TICCGAN/real_images”に格納されています。
テスト実行後に"fake_B"と同じ階層に作成される"real_B"のディレクトリに格納されているものを使用することも可能です。

詳しくは該当shファイルに記載しています。

```bash
sh scripts/expN/〇〇_△△_calculate_score.sh
```

csvにはテスト画像1枚ずつの結果が、txtには平均値が出力されます。ただし、FIDは2つの画像群同士について1つの値しか出力しない指標であるので、FIDに関しては全ての画像で同じ値が入力されています。

## 再現実験について

論文で掲載した実験は以下のとおりです。

1. 昼間TIR画像を昼間可視光画像に変換する実験(英語論文Table 1, Fig.4)
2. 夜間TIR画像を昼間可視光画像に変換する実験(Table 2, Fig.6)
(Table 3, Fig.7, Fig.8はユーザースタディです。図表の元データに実験に使った画像と集計結果が入ってます)
3. 着色モジュールへ入力する特徴量とクラス敵対的損失の有無を変更する実験(Table 4, Fig.9)
4. 着色モジュールへ入力する特徴量の形式を変更する実験(Table 5, Fig.9)
5. クラス敵対的損失の重みを変更する実験(Table 6)
6. GANの種類を変更する実験(Table 7)
7. 学習データセットとテストデータセットの組み合わせを変更する実験(修士論文のみ、英語論文では掲載せず)

以下に各実験と対応するファイル名を記載します。

### 実験1:昼間TIR画像を昼間可視光画像に変換する実験

".../TICCGAN/scripts/exp1"以下

1. train_〇〇.sh
2. test_〇〇_day.sh

".../TICCGAN/evaluation/scripts/exp1"以下
評価

1. 〇〇_day_calculate_score.sh

### 実験2:夜間TIR画像を昼間可視光画像に変換する実験

実験2では実験1で学習したモデルを使用するので、新たにモデルの学習を行う必要はありません。

".../TICCGAN/scripts/exp2"以下

1. test_〇〇.sh

".../TICCGAN/evaluation/scripts/exp2"以下
評価

1. 〇〇_night_calculate_score.sh

### 実験3 & 4

".../TICCGAN/scripts/exp3-4"以下

学習

1. train_IRVI_NA_NA.sh
2. train_IRVI_NA_Lclass.sh
3. train_IRVI_label_NA.sh
4. train_IRVI_label_Lclass.sh
5. train_IRVI_feature_NA.sh

テスト

1. test_IRVI_NA_NA.sh
2. test_IRVI_NA_Lclass.sh
3. test_IRVI_label_NA.sh
4. test_IRVI_label_Lclass.sh
5. test_IRVI_feature_NA.sh

(ファイル名は ”[学習orテスト_[データセット]_[特徴量の形式]*_*[クラス敵対的損失の有無].sh” を意味しています)

".../TICCGAN/evaluation/scripts/exp3-4"以下
評価

1. IRVI_ablation_calculate_score.sh

### 実験5:クラス敵対的損失の重みを変更する実験

".../TICCGAN/scripts/exp5"以下

学習

1. train_IRVI_Lclass0.003.sh
2. train_IRVI_Lclass0.3.sh
3. train_IRVI_Lclass3.sh

テスト

1. test_IRVILclass0.003.sh
2. test_IRVILclass0.3.sh
3. test_IRVILclass3.sh

".../TICCGAN/evaluation/scripts/exp5"以下
評価

1. IRVI_LclassWeight_calculate_score.sh

w_Lclassが0の場合については実験3の"feature_NA"の結果を、w_Lclassが0.03(論文におけるハイパーパラメータ)の場合については実験1の結果を参照してください。

### 実験6:GANの種類を変更する実験

".../TICCGAN/scripts/exp6"以下

学習

1. train_IRVI_SGAN.sh
2. train_IRVI_LSGAN.sh

テスト

1. test_IRVI_SGAN.sh
2. test_IRVI_LSGAN.sh

".../TICCGAN/evaluation/scripts/exp6"以下
評価

1. IRVI_GANtype_calculate_score.sh

GANの種類がRSGAN(論文における計算方法)の場合については実験1の結果を参照してください。

### 実験7:学習データセットとテストデータセットの組み合わせを変更する実験

実験7では実験1で学習したモデルを使用するので、新たにモデルの学習を行う必要はありません。

".../TICCGAN/scripts/exp7"以下

テスト

1. test_FLIR_cross.sh
2. test_IRVI_cross.sh
3. test_KAIST_cross.sh

定量評価は無し