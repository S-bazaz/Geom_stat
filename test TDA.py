# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys
from skimage import io
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
from ripser import lower_star_img, ripser
from plotly.offline import plot
import plotly.io as pio
import cv2

pio.renderers = "browser"

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from utils_tda import (
    get_h1_diagrams_from_dct_bin,get_h_fig_from_df, get_dataframes_from_h0_h1_mats
)


########################
#   bin img creation   #
########################

img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")


img_name = "test_binary.png"
img_name2 = "Gleason 3/147.png"

test_img_path = str(img_path.joinpath(img_name))
test_img_path2 = str(img_path.joinpath(img_name2))

img = io.imread(test_img_path)



img2 = io.imread(test_img_path2)
img2 = cv2.GaussianBlur(img2, (9,9), 0)

# binarizationand
# bin_mat = (img[:, :, 0] != 0) + 0
# bin_mat2 = (img2[:, :, 0] > 30) + 0

for tau in range(254,0,-1):
    bin_mat2 = (img2 > tau) + 0
    plt.imshow(bin_mat2, cmap = "gray")
    plt.savefig(root_path / f"binary/{tau}.png")
    plt.close()
# # sns.heatmap(1 - bin_mat)
# # plt.show()
#     sns.heatmap(bin_mat2)
#     plt.show()
    
    
import numpy as np
from scipy import sparse
def lower_star_img2(img):
    """
    Construct a lower star filtration on an image

    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data

    Returns
    -------
    I: ndarray (K, 2)
        A 0-dimensional persistence diagram corresponding to the sublevelset filtration
    """
    m, n = img.shape

    idxs = np.arange(m * n).reshape((m, n))

    I = idxs.flatten()
    J = idxs.flatten()
    V = img.flatten()

    # Connect 8 spatial neighbors
    tidxs = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tidxs[1:-1, 1:-1] = idxs

    tD = np.ones_like(tidxs) * np.nan
    tD[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            thisJ = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
            thisD = np.roll(np.roll(tD, di, axis=0), dj, axis=1)
            thisD = np.maximum(thisD, tD)

            # Deal with boundaries
            boundary = ~np.isnan(thisD)
            thisI = tidxs[boundary]
            thisJ = thisJ[boundary]
            thisD = thisD[boundary]

            I = np.concatenate((I, thisI.flatten()))
            J = np.concatenate((J, thisJ.flatten()))
            V = np.concatenate((V, thisD.flatten()))

    sparseDM = sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))

    return ripser(sparseDM, distance_matrix=True, maxdim=1)["dgms"]

dgm = lower_star_img2(img2[:,:])

df0,df1 = get_dataframes_from_h0_h1_mats(dgm[0], dgm[1])

# print(dgm)
plot(get_h_fig_from_df(df1))
############################
#   pipelines components   #
############################

# h1_diagram = get_h1_mat_from_binary_mat(bin_mat)
# df = get_dataframe_from_h1_mat(h1_diagram)
# fig = get_h1_fig_from_df(df, title = f"<b>H1</b>cerveau")
# plot(fig)

#################
#   pipelines   #
#################

# dct_bin_mat = {"cerveau": bin_mat, "cellules": bin_mat2}

# dct_df, dct_figs = get_h1_diagrams_from_dct_bin(
#     dct_bin_mat,
#     save_df=True,
#     save_png=True,
#     save_html=True,
#     parquet_name="test",
#     saving_path=saving_path,
# )

# plot(dct_figs["cerveau"])
