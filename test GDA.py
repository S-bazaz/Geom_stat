##############
#  Packages  #
##############
import os
import sys 
from skimage import io
import matplotlib.pyplot as plt 
import cupy as cp
import numpy as np
import seaborn as sns 
import histomicstk as htk

from PIL import Image
from pathlib import Path
##################
#      Imports   #
##################
#from utils import *
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from utils import (
    compress_image,
    color_deconvolution
    
    )

img_path = root_path.joinpath("raw_images")

img_name = "1.tiff"

test_img_path = str(img_path.joinpath(img_name))
#img = io.imread("5.tiff")[:, :, :3]

img = io.imread(test_img_path)[:,:,:3]
#######################
#   image procesing   #
#######################



img = compress_image(img)

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

stains = ["hematoxylin", "eosin","null"]

W = cp.array([stain_color_map[st] for st in stains]).T

img = color_deconvolution(img, W).Stains[:,:,0] # Grayscale image

def tau_activation(img,tau):
    return cp.greater(img,tau)

img = tau_activation(img,200)
mid0 = img.shape[0]//2
mid1 = img.shape[1]//2
x = 400
y = -100
img = img[mid0+y:mid0+y+512, mid1+x:mid1+x+512].get() +0
sns.heatmap(img, cmap = "gray")
# plt.imshow(img, cmap="gray")
# plt.axis("off")
plt.show()
