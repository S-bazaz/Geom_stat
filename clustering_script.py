# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys 

import pickle

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
import pandas as pd
import matplotlib.pyplot as plt


pio.renderers = "browser"

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

from utils_tda_and_clustering import (
    save_tsne,
    homology_parquet_to_matrix_bootstraps,
    make_pca_bootstraps,
    get_clusters,
    transform_gleason,
    meta_clustering,
    choose_representative_bootstrap,
    get_meta_bootstraps,
    get_stability,
    save_homology_densities,
    get_repr_imgs_and_clustering,
    pca_before_clustering,
    get_all_diagram_as_list,
    get_dtw_mat,
    get_dtw_mat_fast,
    get_sub_dist_mat,
    update_clustering_from_mat_dist

)

img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")

######################
#    vectorization   #
######################

# importation and vectorization of diagrams
base_name = "b1"
base_path = str(saving_path.joinpath(f"{base_name}.parquet"))

# Si les bootstraps sont pas encore créés
# df_ident, bootstraps, original = homology_parquet_to_matrix_bootstraps(base_path)

# Uncomment si les bootstraps sont déjà sauvegardés

with open(str(saving_path.joinpath("df_ident.pkl")), "rb") as f:
    df_ident = pickle.load(f)
# with open(str(saving_path.joinpath("bootstraps.pkl")), "rb") as f:
#     bootstraps = pickle.load(f)
with open(str(saving_path.joinpath("original.pkl")), "rb") as f:
    original = pickle.load(f)
# print("bootstrap ok")

# Uncomment if you want to save the current bootstraps
# with open(str(saving_path.joinpath("df_ident.pkl")), "wb") as f:
#     pickle.dump(df_ident, f)
# with open(str(saving_path.joinpath("bootstraps.pkl")), "wb") as f:
#     pickle.dump(bootstraps, f)
# with open(str(saving_path.joinpath("original.pkl")), "wb") as f:
#     pickle.dump(original, f)

####################
#    Clustering    #
####################


# get images ids to identify similarities with Gleason classification
# img_ids = df_ident["img_id"]

# PCA
# reduced_bootstraps, reduced_original, vars_explained = make_pca_bootstraps(bootstraps, original)

# # # Ward clustering for all bootstraps
# clustered_bootstraps, clustered_original = get_clusters(reduced_bootstraps, reduced_original)

# # # Adding the gleason space coordinates to the bootstraps
# gleason_bootstraps = transform_gleason(bootstraps)

# # # Extracting only clusters and corrsponding Gleason coordinates and reshaping the matrix
# clusters = []
# for b in gleason_bootstraps:
#     clusters = clusters + b["clusters"]
# gleason_points = np.array([b["gleason_coords"] for b in gleason_bootstraps]).reshape(-1,3)


# with open(str(saving_path.joinpath("clustered_bootstraps.pkl")), "wb") as f:
#     pickle.dump(clustered_bootstraps, f)
# with open(str(saving_path.joinpath("gleason_points.pkl")), "wb") as f:
#     pickle.dump(gleason_points, f)

# with open(str(saving_path.joinpath("clustered_bootstraps.pkl")), "rb") as f:
#     clustered_bootstraps = pickle.load(f)
# with open(str(saving_path.joinpath("gleason_points.pkl")), "rb") as f:
#     gleason_points = pickle.load(f)

########################
#   Meta Clustering    #
########################

# Meta-clustering to get the 6 meta-clusters
# meta_clusters_indices = meta_clustering(gleason_points)
# meta_clusters = [[clusters[k] for k in c] for c in meta_clusters_indices]

# # # Selection of a representative bootstrap for visualization
# meta_clusters_gleason_coords = [[gleason_points[k] for k in c] for c in meta_clusters_indices]

# representative_bootstrap = choose_representative_bootstrap(gleason_bootstraps, meta_clusters_gleason_coords)

# # # Bootstrap of meta-clusters to assess stability
# meta_bootstraps = get_meta_bootstraps(meta_clusters)

# meta_clusters_fused = []
# for c in meta_clusters:
#     for cluster in c:
#         meta_clusters_fused.append(cluster)

# stabilities = []
# for b in tqdm(meta_bootstraps):
#     stability = get_stability(b, meta_clusters_fused)
#     stabilities.append(stability)

# print(np.mean(stabilities), np.std(stabilities))

# # Save and load representative boostrap
# with open(str(saving_path.joinpath("representative_bootstrap.pkl")), "wb") as f:
#     pickle.dump(representative_bootstrap, f)

with open(str(saving_path.joinpath("representative_bootstrap.pkl")), "rb") as f:
    representative_bootstrap = pickle.load(f)


#############
#   T-SNE   #
#############

# t-SNE clustering
# print("T-SNE")
# embedding_mat3 = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=47).fit_transform(representative_bootstrap["reduced"])

# with open(str(saving_path.joinpath("representative_tsne.pkl")), "wb") as f:
#     pickle.dump(embedding_mat3, f)

# with open(str(saving_path.joinpath("representative_tsne.pkl")), "rb") as f:
#     embedding_mat3 = pickle.load(f)


#######################
#   Visualizations    #
#######################


##### Dendrogram

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

print("Plotting")


##### T-SNE
#save_tsne(representative_bootstrap, saving_path, lst_perplexity=[100], save_pair_plot = True)


##### Densities
#save_homology_densities(base_path, representative_bootstrap, saving_path)



##### PCA point cloud
# import pandas as pd
# df_reduced= pd.DataFrame(representative_bootstrap["reduced"],columns = np.arange(6))
# fig = sns.pairplot(df_reduced)
# plt.savefig(str(saving_path.joinpath(f"representativ_pair_plot.png")))

##### PCA point cloud
# #print(representative_bootstrap["gleason_coords"]})
# from sklearn.decomposition import PCA
# pca = PCA(n_components=6)
# pca.fit(representative_bootstrap["embedding_mat"])
# print(pca.explained_variance_ratio_)
# main_component = np.round(pca.components_[0], 3)
# fig = px.line(np.abs(main_component), template = "plotly_dark", title = "<b>Absolute value of principal component coordinates</b>")
# fig.update_traces(line=dict(color="palegreen", width=3))
# fig.show()


##### Gleason barplot
# gleason_coord = representative_bootstrap["gleason_coords"]

# print(gleason_coord)

# df_gleason = pd.DataFrame({
#     "Number": gleason_coord.reshape(18),
#     "Gleason class":np.tile([f"Gleason {k}" for k in range(3,6)], 6),
#     "Cluster":np.repeat(np.arange(1,7), 3)
# })

# print(df_gleason)
# fig = px.bar(df_gleason, 
#              x="Cluster", 
#              y="Number",
#              color="Gleason class",
#              title="<b>Gleason proportion for each cluster</b>", 
#              template = "plotly_dark"
#              )
# fig.write_html(str(saving_path.joinpath(f"repr_gleason_count.html")))

##################
#   DTW tests    #
##################


##### matrice de distances

import time as t

# lst_st, lst_gleason_class = get_all_diagram_as_list(base_path)
# mat_dist = get_dtw_mat(lst_st)


# t1 = t.time()
# mat_dist2 = get_dtw_mat_fast(base_path)
# t2 = t.time()

# print(f"mat dist done! t={t2-t1}")

# with open(str(saving_path.joinpath("mat_dist.pkl")), "wb") as f:
#     pickle.dump(mat_dist, f)

with open(str(saving_path.joinpath("mat_dist.pkl")), "rb") as f:
    mat_dist = pickle.load(f)


#### test original
reduced_original,_ = pca_before_clustering(original["embedding_mat"], n_components = 6, standard = False) # We will need the original data to compute the Gap Statistic
original["reduced"] = reduced_original


##### Analyse
nclust = 3
update_clustering_from_mat_dist(mat_dist, original, nb_clusters=nclust)

save_tsne(original, saving_path, lst_perplexity=[50,100,200], save_pair_plot = False, test_other_clust = False, metric = f"dtw{nclust}")

#save_homology_densities(base_path, original, saving_path, metric = f"dtw{nclust}")

#### Gleason barplot

# gleason_coord = original["gleason_coords"]

# df_gleason = pd.DataFrame({
#     "Number": gleason_coord.reshape(3*nclust),
#     "Gleason class":np.tile([f"Gleason {k}" for k in range(3,6)], nclust),
#     "Cluster":np.repeat(np.arange(1,nclust+1), 3)
# })

# fig = px.bar(df_gleason, 
#              x="Cluster", 
#              y="Number",
#              color="Gleason class",
#              title="<b>DTW Gleason proportion for each cluster</b> ", 
#              template = "plotly_dark"
#              )
# fig.write_html(str(saving_path.joinpath(f"repr_gleason_countdtw{nclust}.html")))

