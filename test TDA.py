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

from utils_tda import load_img_and_save_homology_to_parquet


########################
#   bin img creation   #
########################

# img_path = root_path.joinpath("raw_images")
# saving_path = root_path.joinpath("outputs")


# img_name = "test_binary.png"
# #img_name2 = "Gleason 3/147.png"
# img_name2 = "cellules.jfif"


# test_img_path = str(img_path.joinpath(img_name))
# test_img_path2 = str(img_path.joinpath(img_name2))

# img = io.imread(test_img_path)
# img = cv2.GaussianBlur(img, (9,9), 0)

# img2 = io.imread(test_img_path2)
# img2 = cv2.GaussianBlur(img2, (9,9), 0)

load_img_and_save_homology_to_parquet(root_path)


#################
#   pipelines   #
#################

# dct_bin_mat = {"cerveau": img, "cellules": img2}

# dct_df, dct_figs = get_h1_diagrams_from_dct_mat(
#     dct_bin_mat,
#     save_df=True,
#     save_png=True,
#     save_html=True,
#     parquet_name="test",
#     saving_path=saving_path,
# )

# plot(dct_figs["cerveau"]["h0"])
