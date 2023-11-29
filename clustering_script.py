# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys 


from pathlib import Path
import seaborn as sns 
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from umap import UMAP

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt


pio.renderers = "browser"

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from utils_tda_and_clustering import (
    homology_parquet_to_matrix_bootstraps,
    make_pca_bootstraps,
    get_clusters,
    transform_gleason,
    meta_clustering,
    choose_representative_bootstrap,
    get_meta_bootstraps
)

img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")

#############
#    Test   #
#############



# importation and vectorization of diagrams
base_name = "b1"
base_path = str(saving_path.joinpath(f"{base_name}.parquet"))
df_ident, bootstraps, original = homology_parquet_to_matrix_bootstraps(base_path)
print("bootstrap ok")

# get images ids to identify similarities with Gleason classification
img_ids = df_ident["img_id"]

# PCA

reduced_bootstraps, reduced_original, vars_explained = make_pca_bootstraps(bootstraps, original)

# Ward clustering for all bootstraps
clustered_bootstraps, clustered_original = get_clusters(reduced_bootstraps, reduced_original)

# Adding the gleason space coordinates to the bootstraps
gleason_bootstraps = transform_gleason(bootstraps)

# Extracting only clusters and corrsponding Gleason coordinates and reshaping the matrix
clusters = []
for b in gleason_bootstraps:
    clusters = clusters + b["clusters"]
gleason_points = np.array([b["gleason_coords"] for b in gleason_bootstraps]).reshape(-1,3)

# Meta-clustering to get the 6 meta-clusters
meta_clusters_indices = meta_clustering(gleason_points)
meta_clusters = [[clusters[k] for k in c] for c in meta_clusters_indices]

# Selection of a representative bootstrap for visualization
meta_clusters_gleason_coords = [[gleason_points[k] for k in c] for c in meta_clusters_indices]
representative_bootstrap = choose_representative_bootstrap(gleason_bootstraps, meta_clusters_gleason_coords)

# Bootstrap of meta-clusters to assess stability
meta_bootstraps = get_meta_bootstraps(meta_clusters)

print(meta_clusters, meta_bootstraps)

# sns.heatmap(linked, yticklabels=img_ids)

# fig = plt.figure(figsize=(15, 15))
# dn = dendrogram(linked, orientation='right', show_leaf_counts=False, no_plot=True)
# temp = {dn["leaves"][ii]: img_ids[ii] for ii in range(len(dn["leaves"]))}
# def llf(xx):
#     return "{} - custom label!".format(temp[xx])

# dendrogram(
#             linked,
#             # truncate_mode='lastp',  # show only the last p merged clusters
#             # p=p,  # show only the last p merged clusters
#             leaf_label_func=llf,
#             leaf_rotation=60.,
#             leaf_font_size=12.,
#             show_contracted=True,  # to get a distribution impression in truncated branches
#             )

# plt.show()


# t-SNE clustering
embedding_mat3 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=47).fit_transform(representative_bootstrap["reduced"])
fig = px.scatter(x = embedding_mat3[:,0], y = embedding_mat3[:,1],template="plotly_dark", hover_name = img_ids, title = "T-SNE après ACP")
plot(fig)