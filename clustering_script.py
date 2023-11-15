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

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


pio.renderers = "browser"

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from utils_tda_and_clustering import (
    homology_parquet_to_matrix,
    pca_before_clustering,
)

img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")

#############
#    Test   #
#############



# importation and vectorization of diagrams
base_name = "b1"
base_path = str(saving_path.joinpath(f"{base_name}.parquet"))
df_ident, embedding_mat = homology_parquet_to_matrix(base_path)

# get images ids to identify similarities with Gleason classification
img_ids = df_ident["img_id"]

# PCA
embedding_mat2 = pca_before_clustering(embedding_mat, n_components = 5, standard = False)


# Ward clustering
linked = linkage(embedding_mat2, 'ward')
sns.heatmap(linked, yticklabels=img_ids)

fig = plt.figure(figsize=(15, 15))
dn = dendrogram(linked, orientation='right', show_leaf_counts=False, no_plot=True)
temp = {dn["leaves"][ii]: img_ids[ii] for ii in range(len(dn["leaves"]))}
def llf(xx):
    return "{} - custom label!".format(temp[xx])

dendrogram(
            linked,
            # truncate_mode='lastp',  # show only the last p merged clusters
            # p=p,  # show only the last p merged clusters
            leaf_label_func=llf,
            leaf_rotation=60.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
            )

plt.show()


# t-SNE clustering
embedding_mat3 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(embedding_mat2)
fig = px.scatter(x = embedding_mat3[:,0], y = embedding_mat3[:,1],template="plotly_dark", hover_name = img_ids, title = "T-SNE après ACP")
plot(fig )
