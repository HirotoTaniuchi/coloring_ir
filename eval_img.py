import numpy as np
from scipy.linalg import sqrtm
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim




def calculate_psnr(img1, img2):
    # print("calculateion started")
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    # print("calculated psnr: ", psnr)
    return psnr

def calculate_average_psnr(dir1, dir2):
    # print("avg_calculation started")
    psnr_values = []
    # print("os.listdir(dir1): ", os.listdir(dir1))
    for img_name in os.listdir(dir1):
        print("img_name: ", img_name)
        img1_path = os.path.join(dir1, img_name)
        img2_path = os.path.join(dir2, img_name)
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = np.array(Image.open(img1_path))
            img2 = np.array(Image.open(img2_path))
            psnr = calculate_psnr(img1, img2)
            psnr_values.append(psnr)
    average_psnr = np.mean(psnr_values)
    return average_psnr



def calculate_ssim(img1, img2):
    ssim_value, _ = ssim(img1, img2, full=True, multichannel=True)
    return ssim_value

def calculate_average_ssim(dir1, dir2):
    ssim_values = []
    for img_name in os.listdir(dir1):
        img1_path = os.path.join(dir1, img_name)
        img2_path = os.path.join(dir2, img_name)
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = np.array(Image.open(img1_path))
            img2 = np.array(Image.open(img2_path))
            ssim_value = calculate_ssim(img1, img2)
            ssim_values.append(ssim_value)
    average_ssim = np.mean(ssim_values)
    return average_ssim



# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = np.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrtm
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     # calculate score
#     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid

# def get_activations(images, model):
#     act = model.predict(images)
#     return act

# def calculate_fid_given_paths(path1, path2, model):
#     # load images
#     images1 = np.load(path1)
#     images2 = np.load(path2)
#     # calculate activations
#     act1 = get_activations(images1, model)
#     act2 = get_activations(images2, model)
#     # calculate FID
#     fid = calculate_fid(act1, act2)
#     return fid



if __name__ == '__main__':

    output_dir = "./output/ugawa_test_202502272000/AAFSTNet2andSegNet2_MFNet_full_RSGAN/test_other_latest/fake_B"
    gt_dir = "./output/ugawa_test_202502272000/AAFSTNet2andSegNet2_MFNet_full_RSGAN/test_other_latest/real_B"
    print(calculate_average_psnr(output_dir, gt_dir))
    print(calculate_average_ssim(output_dir, gt_dir))


    
