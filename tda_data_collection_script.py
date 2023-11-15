# -*- coding: utf-8 -*-
##############
#  Packages  #
##############

import os
import sys
from pathlib import Path

import plotly.io as pio

pio.renderers = "browser"

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from utils_tda_and_clustering import load_img_and_save_homology_to_parquet

#################
#   pipelines   #
#################

# modifiez : raw_data_selection.yaml pour décider des images à importer

base_name = "b1"

###### save png ######
# load_img_and_save_homology_to_parquet(
#     root_path,  parquet_name=base_name, save_df=False, save_png=True, batch_size=20)

###### save vectors ######
load_img_and_save_homology_to_parquet(
    root_path,  parquet_name=base_name, save_df=True, save_png=False, batch_size=20)


###### Voir clustering.py pour l'étude des bases sauvegardés ######

