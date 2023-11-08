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

from plotly.offline import plot
import plotly.io as pio

pio.renderers = "browser"

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
print(root_path)
sys.path.insert(0, str(root_path))

from utils_tda import (
    get_h1_mat_from_binary_mat,
    get_dataframe_from_h1_mat,
    get_h1_fig_from_df,
    get_h1_diagrams_from_dct_bin,
)


########################
#   bin img creation   #
########################

img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")


img_name = "test_binary.png"
img_name2 = "cellules.jfif"

test_img_path = str(img_path.joinpath(img_name))
test_img_path2 = str(img_path.joinpath(img_name2))

img = io.imread(test_img_path)
img2 = io.imread(test_img_path2)
# binarizationand
bin_mat = (img[:, :, 0] != 0) + 0
bin_mat2 = (img2[:, :, 0] > 30) + 0


sns.heatmap(1 - bin_mat)
plt.show()
sns.heatmap(1 - bin_mat2)
plt.show()

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

dct_bin_mat = {"cerveau": bin_mat, "cellules": bin_mat2}

dct_df, dct_figs = get_h1_diagrams_from_dct_bin(
    dct_bin_mat,
    save_df=True,
    save_png=True,
    save_html=True,
    parquet_name="h1_test",
    saving_path=saving_path,
)

# plot(dct_figs["cerveau"])
