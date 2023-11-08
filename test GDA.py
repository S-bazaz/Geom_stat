##############
#  Packages  #
##############
import os
import sys 

from skimage import io
import matplotlib.pyplot as plt 
# os.add_dll_directory(r"C:\Windows\SysWOW64")
# os.add_dll_directory(r"C:\Windows\System32")
import cupy as cp
import numpy as np

import histomicstk as htk

from PIL import Image
from pathlib import Path
##################
#      Imports   #
##################
#from utils import *
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from Geom_stat.utils import (
    compress_image,
    color_deconvolution
    
    )

img_path = root_path.joinpath("images")

img_name = "1.tiff"
#img_name = "5.tiff"

test_img_path = str(img_path.joinpath(img_name))
#img = io.imread("5.tiff")[:, :, :3]

img = io.imread(test_img_path)[:, :, :3]
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

img = tau_activation(img, 100)

plt.imshow(img.get(), cmap="gray")
plt.axis("off")
plt.show()