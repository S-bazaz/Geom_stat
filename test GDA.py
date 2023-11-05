from skimage import io
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from utils import *
from PIL import Image

import histomicstk as htk

img = io.imread("5.tiff")[:, :, :3]
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