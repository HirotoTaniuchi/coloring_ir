from __future__ import print_function
import torch
import numpy as np
from PIL import Image, ImageCms
import os
from collections import OrderedDict

def rgb2lab(img_pil):
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    out = ImageCms.applyTransform(img_pil, rgb2lab_transform)
    return out

def lab2rgb(img_pil):
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    out = ImageCms.applyTransform(img_pil, lab2rgb_transform)
    return out
