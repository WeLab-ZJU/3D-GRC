import numpy as np
from skimage.metrics import structural_similarity
import os
def calc_psnr(gt: np.ndarray, predicted: np.ndarray, data_range):

    mse = np.mean(np.power(predicted / data_range - gt / data_range, 2))
    psnr = -10 * np.log10(mse)
    return psnr


def calc_ssim(gt: np.ndarray, predicted: np.ndarray, data_range):

    ssim = structural_similarity(gt, predicted, data_range=data_range)
    return ssim

def get_type_max(gt):
    if gt.dtype == np.uint8:
        return 255
    elif gt.dtype == np.uint16:
        return 65535
    elif gt.dtype == np.uint32:
        return 4294967295
    
def get_size_folder(folder):
    size = 0
    for path, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
        for d in dirs:
            size += get_size_folder(d)
    size = size / 1024
    return size


