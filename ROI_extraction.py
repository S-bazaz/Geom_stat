import os
import sys 
from skimage import io
import cupy as cp
import numpy as np
import histomicstk as htk
from tqdm import tqdm

from PIL import Image
from pathlib import Path

root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

paths = list(root_path.glob("**/*.tiff"))


def get_ROIs(img):
    ROIs = []
    nrows, ncols = img.shape[:2]
    for i in range(nrows//512):
        for j in range(ncols//512):
            ROI = img[256+512*i:256+512*(i+1),256+512*j:256+512*(j+1)]
            ROIs.append(ROI)
    return ROIs

def deconvolution(ROIs, W):
    ROIs_gray = []
    for ROI in ROIs:
        ROI_gray = htk.preprocessing.color_deconvolution.color_deconvolution(ROI, W).Stains[:,:,0]
        if np.count_nonzero(ROI_gray==255)<32768: # Less than 12.5% background
            ROIs_gray.append(ROI_gray)
    return ROIs_gray

indices = {"Gleason 3":0, "Gleason 4":0, "Gleason 5":0,}
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
stains = ["hematoxylin", "eosin","null"]
W = np.array([stain_color_map[st] for st in stains]).T

for path in tqdm(paths):
    cell_type = path.parent.stem
    img = io.imread(path)
    ROIs = get_ROIs(img)
    ROIs_gray = deconvolution(ROIs,W)
    for ROI in ROIs_gray:
        index = indices[cell_type]
        Image.fromarray(ROI).save(root_path / f"ROIs/{cell_type}/{index}.png")
        indices[cell_type] += 1