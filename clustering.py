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

from utils_tda import (
    homology_parquet_to_matrix,
    pca_before_clustering,
    
)
img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")

#############
#    Test   #
#############

base_name = "test"
df_ident, embedding_mat = homology_parquet_to_matrix(str(saving_path.joinpath(f"{base_name}.parquet")))

embedding_mat2 = pca_before_clustering(embedding_mat, n_components = 4 ,standard = False)
Z = linkage(embedding_mat2, 'ward')
sns.heatmap(Z)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
print(embedding_mat2)
embedding_mat3 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embedding_mat2)
print(embedding_mat3)

# embedding_mat3_bis = UMAP().fit_transform(embedding_mat2)
# print(embedding_mat3_bis)

fig = px.scatter(x = embedding_mat3[:,0], y = embedding_mat3[:,1],template="plotly_dark")
plot(fig)
# fig = px.scatter(x = embedding_mat3_bis[:,0], y = embedding_mat3_bis[:,1],template="plotly_dark")
# plot(fig)