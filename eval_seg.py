import os
import numpy as np
from sklearn.metrics import jaccard_score
from seg_dataloader import make_testdatapath_list
import tqdm
from PIL import Image
import torch
import collections


def load_image(filepath):
    with Image.open(filepath) as img:
        # print(collections.Counter(np.array(img).flatten()))
        return np.array(img)

def calculate_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return ious


def calculate_miou(pred_dir, target_dir, file_list, num_classes=9):
    all_ious = []
    with open(file_list, 'r') as f:
        filenames = [line.strip() for line in f]
    
    for filename in tqdm.tqdm(filenames):
        pred_path = os.path.join(pred_dir, filename + ".png")
        target_path = os.path.join(target_dir, filename + ".png")
        
        pred = load_image(pred_path)
        target = load_image(target_path)
        
        ious = calculate_iou(pred, target, num_classes)
        all_ious.append(ious)
    
    all_ious = np.array(all_ious)
    miou = np.nanmean(all_ious, axis=0)  # Calculate mean IoU for each class
    return miou






def calculate_batch_miou(dataloader, num_classes=9):
    all_ious = []
    
    for batch in tqdm.tqdm(dataloader):
        preds, targets = batch  # Assuming dataloader returns a tuple of (predictions, targets)
        
        for pred, target in zip(preds, targets):
            pred = np.array(pred)
            target = np.array(target)
            
            ious = calculate_iou(pred, target, num_classes)
            all_ious.append(ious)
    
    all_ious = np.array(all_ious)
    miou = np.nanmean(all_ious, axis=0)  # Calculate mean IoU for each class
    return miou

def calculate_pixel_accuracy(pred, target, num_classes=9):
    """
    Calculate pixel accuracy for each class
    """
    accuracies = []
    pred = pred.cpu().numpy()  # Convert torch tensor to numpy array
    target = target.cpu().numpy()  # Convert torch tensor to numpy array
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        correct_pixels = np.logical_and(pred_inds, target_inds).sum()
        total_target_pixels = target_inds.sum()
        if total_target_pixels == 0:
            accuracies.append((-1,-1))  # If there are no target pixels for this class, do not include in evaluation
        else:
            accuracies.append((correct_pixels, total_target_pixels))
    return accuracies



def calculate_accuracy(pred_dir, target_dir, file_list, num_classes=9):
    all_correct_pixels = np.zeros(num_classes)
    all_total_pixels = np.zeros(num_classes)
    
    with open(file_list, 'r') as f:
        filenames = [line.strip() for line in f]
    
    for filename in tqdm.tqdm(filenames):
        pred_path = os.path.join(pred_dir, filename + ".png")
        target_path = os.path.join(target_dir, filename + ".png")
        
        # print(filename)
        pred = load_image(pred_path)

        target = load_image(target_path)
        
        accuracies = calculate_pixel_accuracy(torch.tensor(pred), torch.tensor(target), num_classes)
        
        # print(filename, "accuracies", accuracies[0:4])
        
        
        for cls in range(num_classes):
            correct_pixels, total_pixels = accuracies[cls]
            if total_pixels != -1:  # Only accumulate if there are target pixels for this class
                all_correct_pixels[cls] += correct_pixels
                all_total_pixels[cls] += total_pixels
    
    mean_accuracies = np.divide(all_correct_pixels, all_total_pixels, out=np.zeros_like(all_correct_pixels), where=all_total_pixels!=0)
    return mean_accuracies




if __name__ == "__main__":
    pred_dir = "/home/usrs/taniuchi/workspace/projects/coloring_ir/output_seg/deeplabv3plus_mobilenet_test_202505151144/predict"
    target_dir = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/labels"
    file_list = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/test_day.txt"
    
    miou = calculate_miou(pred_dir, target_dir, file_list)
    mean_iou = np.nanmean(miou)
    miou_lines = [f"Mean IoU: {mean_iou}\n"]
    for cls, iou in enumerate(miou):
        miou_lines.append(f"Class {cls} IoU: {iou}\n")
        print(f"Class {cls} IoU: {iou}")
    print(f"Mean IoU: {mean_iou}")

    mean_accuracies = calculate_accuracy(pred_dir, target_dir, file_list)
    mean_acc = np.nanmean(mean_accuracies)
    acc_lines = [f"Mean Accuracy: {mean_acc}\n"]
    for cls, accuracy in enumerate(mean_accuracies):
        acc_lines.append(f"Class {cls} Accuracy: {accuracy}\n")
        print(f"Class {cls} Accuracy: {accuracy}")
    print(f"Mean Accuracy: {mean_acc}")

    # readme.mdへの記録
    readme_path = os.path.join(os.path.dirname(pred_dir), "readme.md")
    with open(readme_path, "a") as f:
        f.write("# Segmentation Evaluation Results\n\n")
        f.write(f"**pred_dir:** `{pred_dir}`\n\n")
        f.write(f"**target_dir:** `{target_dir}`\n\n")
        f.write(f"**file_list:** `{file_list}`\n\n")
        f.write("## IoU\n")
        f.writelines(miou_lines)
        f.write("\n## Accuracy\n")
        f.writelines(acc_lines)
        f.write("\n---\n\n")