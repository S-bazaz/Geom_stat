import os
import sys 
from skimage import io
import cupy as cp
import numpy as np
import histomicstk as htk
from tqdm import tqdm
import cv2

from PIL import Image
from pathlib import Path

root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

paths = list(root_path.glob("./Images/**/*_mask.tiff"))

def get_ROIs(img, mask):
    ROIs = []
    nrows, ncols = img.shape[:2]
    for i in range(nrows//512):
        for j in range(ncols//512):
            ROI = img[256+512*i:256+512*(i+1),256+512*j:256+512*(j+1)]
            mask_ROI = mask[256+512*i:256+512*(i+1),256+512*j:256+512*(j+1)]
            cell_type = np.bincount(mask_ROI.flatten()).argmax()
            ROI = cv2.GaussianBlur(ROI, (7,7), 0)
            if cell_type not in [0,1,2]:
                ROIs.append([ROI,cell_type])
    return ROIs

def deconvolution(ROIs, W):
    ROIs_gray = []
    for ROI,c in ROIs:
        ROI_gray = htk.preprocessing.color_deconvolution.color_deconvolution(ROI, W).Stains[:,:,0]
        if np.count_nonzero(ROI_gray==255)<32768: # Less than 12.5% background
            ROIs_gray.append([ROI_gray,c])
    return ROIs_gray

indices = {3:1, 4:1, 5:1}
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
stains = ["hematoxylin", "eosin","null"]
W = np.array([stain_color_map[st] for st in stains]).T

for mask_path in tqdm(paths):
    img_path = Path(str(mask_path)[:-10] + ".tiff")
    img = io.imread(img_path)
    mask = io.imread(mask_path)[:,:,0]
    ROIs = get_ROIs(img, mask)
    ROIs_gray = deconvolution(ROIs,W)
    if ROIs_gray == []:
        os.remove(f"{mask_path}")
        os.remove(f"{img_path}")
    for ROI, cell_type in ROIs_gray:
        index = indices[cell_type]
        Image.fromarray(ROI).save(root_path / f"raw_images/Gleason {cell_type}/{index}.png")
        indices[cell_type] += 1