import argparse
from tqdm import tqdm
from skimage import metrics, io
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from lpips import LPIPS
from pytorch_fid.fid_score import calculate_fid_given_paths

def calculate_psnr(real_path:Path, fake_path:Path)->float:
    '''
    PSNRのスコアを計算する関数
    '''
    real_img = io.imread(real_path) / 255.0
    fake_img = io.imread(fake_path) / 255.0
    psnr_score = metrics.peak_signal_noise_ratio(real_img, fake_img, data_range=1.0)
    return psnr_score

def calculate_ssim(real_path:Path, fake_path:Path)->float:
    '''
    SSIMのスコアを計算する関数
    '''
    real_img = io.imread(real_path) / 255.0
    fake_img = io.imread(fake_path) / 255.0
    ssim_score = metrics.structural_similarity(real_img, fake_img, channel_axis=2, gaussian_weights=True, data_range=1.0)
    return ssim_score

def calculate_lpips(real_path:Path, fake_path:Path, loss_fn_lpips:LPIPS, device:torch.device)->float:
    '''
    LPIPSのスコアを計算する関数
    '''
    real_img = io.imread(real_path) / 255.0
    fake_img = io.imread(fake_path) / 255.0
    real_img = ((torch.from_numpy(np.transpose(real_img.copy(), (2, 0, 1))[None, ...]).to(torch.float32).clone() - 0.5) * 2).to(device)
    fake_img = ((torch.from_numpy(np.transpose(fake_img.copy(), (2, 0, 1))[None, ...]).to(torch.float32).clone() - 0.5) * 2).to(device)
    lpips_score = loss_fn_lpips(real_img, fake_img).item()
    return lpips_score

def get_commandline_args()->argparse.Namespace:
    '''
    コマンドライン引数を取得する関数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="実画像のディレクトリ")
    parser.add_argument("--fid_real_dir", type=str, default=None, help="FIDの実画像のディレクトリ")
    parser.add_argument("--fake_dir", type=str, required=True, help="生成画像のディレクトリ")
    parser.add_argument("--save_dir", type=str, required=True, help="保存先のディレクトリ")
    parser.add_argument("--exp_name", type=str, required=True, help="実験名")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU番号")
    parser.add_argument("--fid_dim", type=int, default=2048, help="FIDの特徴量の次元数")
    parser.add_argument("--pixel_wise_metrics", action='store_true', help="PSNR, SSIM, LPIPSを計算するかどうか")
    return parser.parse_args()

def main():
    # コマンドライン引数の設定
    args = get_commandline_args()

    # GPU,モデルの設定
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    loss_fn_lpips = LPIPS(net='alex').to(device)

    # ディレクトリの設定
    real_dir = Path(args.real_dir)
    fid_real_dir = Path(args.fid_real_dir)
    fake_dir = Path(args.fake_dir)
    save_dir = Path(args.save_dir)

    # 画像パス群と画像数の取得
    real_img_paths = sorted(list(real_dir.glob('*.png'))+list(real_dir.glob('*.jpg')))
    fake_img_paths = sorted(list(fake_dir.glob('*.png'))+list(fake_dir.glob('*.jpg')))
    if len(real_img_paths) != len(fake_img_paths) and args.pixel_wise_metrics:
        raise ValueError('実画像と生成画像の枚数が一致していません。pixel_wise_metricsがTrueの場合は枚数が一致している必要があります。')
    else:
        num_imgs = len(real_img_paths)
    
    # FIDの計算
    fid_score = calculate_fid_given_paths(paths=[fid_real_dir.as_posix(), fake_dir.as_posix()], batch_size=1, device=args.gpu_id, dims=args.fid_dim)

    save_dir = save_dir / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # PSNR, SSIM, LPIPSの計算
    if args.pixel_wise_metrics:
        # dataframeの作成
        df = pd.DataFrame(columns=['PSNR', 'SSIM', 'LPIPS', 'FID'])
        for i, real_path in tqdm(enumerate(real_img_paths), desc='Calculate Score', total=num_imgs):
            # 画像名の取得
            img_stem = real_path.stem.replace('_real_B', '')
            fake_path = fake_img_paths[i]
            # スコアの計算
            psnr_score = calculate_psnr(real_path, fake_path)
            ssim_score = calculate_ssim(real_path, fake_path)
            lpips_score = calculate_lpips(real_path, fake_path, loss_fn_lpips, device)
            # スコアをdataframeに追加
            df.loc[img_stem] = [psnr_score, ssim_score, lpips_score, fid_score]
        # csvファイルに画像ごとのスコアを保存
        df.to_csv(save_dir / f'{args.exp_name}.csv')

    # txtファイルに平均値を保存
    with open(save_dir / f'{args.exp_name}.txt', 'w') as f:
        if args.pixel_wise_metrics:
            f.write(f'PSNR: {df["PSNR"].mean()}\n')
            f.write(f'SSIM: {df["SSIM"].mean()}\n')
            f.write(f'LPIPS: {df["LPIPS"].mean()}\n')
            print(f'PSNR: {df["PSNR"].mean()}')
            print(f'SSIM: {df["SSIM"].mean()}')
            print(f'LPIPS: {df["LPIPS"].mean()}')
        f.write(f'FID: {fid_score}\n')
        print(f'FID: {fid_score}')
    
    # shファイルの設定をconfig.txtファイルに保存
    with open(save_dir / 'config.txt', 'w') as f:
        f.write(f'--real_dir {args.real_dir}\n')
        f.write(f'--fake_dir {args.fake_dir}\n')
        f.write(f'--fid_real_dir {args.fid_real_dir}\n')
        f.write(f'--save_dir {args.save_dir}\n')
        f.write(f'--exp_name {args.exp_name}\n')
        f.write(f'--gpu_id {args.gpu_id}\n')
        f.write(f'--fid_dim {args.fid_dim}\n')   
        f.write(f'--pixel_wise_metrics {args.pixel_wise_metrics}\n') 

if __name__ == '__main__':
    main() 