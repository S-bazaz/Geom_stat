# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys
import yaml

import numpy as np
import pandas as pd
from skimage import io

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm

from pathlib import Path
from ripser import ripser, Rips
from scipy import sparse
from fastparquet import write

from typing import Dict, Optional, Union, BinaryIO, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN



from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

from plotly.offline import plot

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[0]
sys.path.insert(0, str(root_path))

img_path = root_path.joinpath("raw_images")
saving_path = root_path.joinpath("outputs")

################
#   fetcher    #
################


def load_yaml(path: str) -> Dict:
    """
    Charge et retourne la configuration à partir d'un fichier YAML.

    Args:
        path (str): Chemin du fichier de configuration YAML.

    Returns:
        dict: Configuration chargée à partir du fichier YAML.
    """
    return yaml.safe_load(open(path, "r"))


def add_path_to_nodes(dct: Dict[str, Union[Dict, None]], accu_path: Path) -> None:
    """
    Ajoute un chemin aux nœuds d'un dictionnaire récursif.

    Args:
        dct (Dict[str, Union[Dict, None]]): Le dictionnaire à modifier.
        accu_path (Path): Le chemin accumulé à ajouter aux nœuds.

    Returns:
        None
    """
    if dct is not None:
        for k in dct:
            if dct[k] is not None:
                add_path_to_nodes(dct[k], accu_path.joinpath(k))
                dct[k]["path"] = str(accu_path.joinpath(k))
            else:
                dct[k] = {"path": str(accu_path.joinpath(k))}

        if "path" not in dct:
            dct["path"] = str(accu_path)


def get_leaves_with_path(
    dct: Dict[str, Union[Dict, None]], accu: List[str] = []
) -> None:
    """
    Récupère les chemins des feuilles d'un dictionnaire récursif.

    Args:
        dct (Dict[str, Union[Dict, None]]): Le dictionnaire à parcourir.
        accu (List[str], optionnel): La liste accumulatrice pour stocker les chemins.
        Par défaut, une liste vide.

    Returns:
        None
    """
    if dct is not None:
        child = list(dct)
        child.remove("path")
        if len(child) > 0:
            for k in child:
                get_leaves_with_path(dct[k], accu)
        else:
            accu.append(dct["path"])


def import_where_right_format(
    file_path: str, formats: List[str], dct_accu: Dict[str, str], prefix=""
) -> None:
    """
    Vérifie si le format du fichier correspond à ceux spécifiés dans la liste `formats`,
    puis concerve le chemin

    Args:
        file_path (str): Le chemin du fichier.
        formats (List[str]): Une liste de formats valides.
        dct_accu (Dict[str, str]): Le dictionnaire pour stocker les chemins des fichiers au format correct.

    Returns:
        None
    """
    pathobj = Path(file_path)
    ext = pathobj.suffix[1:]

    if ext in formats:
        # print(file_path)
        # dct_accu[f"{prefix}{pathobj.stem}"] = io.imread(file_path)
        dct_accu[f"{prefix}{pathobj.stem}"] = file_path


def get_all_data_from_path(
    path: str,
    formats: List[str],
    dct_accu: Dict[str, str],
) -> None:
    """
    Parcourt récursivement un répertoire,
    et sauvegarde les chemins des fichiers correspondant aux formats spécifiés dans le dictionnaire `dct_accu`.

    Args:
        path (str): Le chemin du répertoire à parcourir.
        formats (List[str]): Une liste de formats valides.
        dct_accu (Dict[str, str]): Le dictionnaire pour stocker les chemins des fichiers au format correct.
            Par défaut, None.

    Returns:
        None
    """
    for root, dirs, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            import_where_right_format(
                file_path, formats, dct_accu, prefix=Path(path).stem + "_")


def get_all_data_from_paths(
    paths: List[str],
    formats: List[str],
    dct_accu: Dict[str, str],
) -> None:
    """
    Parcourt récursivement plusieurs répertoires
     et sauvegarde les chemins des fichiers correspondant
     aux formats spécifiés dans le dictionnaire `dct_accu`.

    Args:
        paths (List[str]):
            Une liste de chemins de répertoires à parcourir.
        formats (List[str]):
            Une liste de formats valides.
        dct_accu (Dict[str, str]):
            Le dictionnaire pour stocker les chemins des fichiers au format correct.

    Returns:
        None
    """
    for path in paths:
        get_all_data_from_path(path, formats, dct_accu)


def get_selected_data_paths(doc: Dict) -> Dict[str, Dict[str, str]]:
    """
    Récupère tous les chemins des fichiers de données sélectionnés

    Args:
        mode (str, optionnel): Le mode spécifique. Par défaut, "aus".

    Returns:
        Dict[str, Dict[str, str]]: Un dictionnaire contenant les fichiers de données sélectionnés,
        organisés par type ("local" et "s3").
    """

    formats = doc["raw_data_formats"]
    paths = []
    get_leaves_with_path(doc["raw_images"], accu=paths)

    data_files = {}
    get_all_data_from_paths(paths, formats, data_files)
    return data_files


def get_selected_img_paths(root_path):
    yaml_path = root_path.joinpath("raw_data_selection.yaml")
    doc = load_yaml(str(yaml_path))
    add_path_to_nodes(doc["raw_images"], root_path.joinpath("raw_images"))
    return get_selected_data_paths(doc)


##########################
#   images to parquet    #
##########################

# def get_h0_h1_mats_from_binary_mat(bin_mat: np.ndarray) -> np.ndarray:

#     rips = Rips(maxdim=1, coeff=2, verbose=False)
#     bin_mat2 = np.array(bin_mat)
#     if bin_mat2.shape[1] > bin_mat2.shape[0]:
#         bin_mat2 = bin_mat2.T
#     diagrams = rips.fit_transform(
#         bin_mat2, distance_matrix=False, metric="euclidean")
#     return np.round(diagrams[0], 3), np.round(diagrams[1], 3)


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


def get_h0_h1_mats_from_img(img: np.ndarray) -> np.ndarray:
    diagrams = lower_star_img2(img)
    return np.round(diagrams[0], 3), np.round(diagrams[1], 3)


def get_dataframes_from_h0_h1_mats(h0_diagram: np.ndarray, h1_diagram: np.ndarray) -> tuple:
    """
    Convert H0 and H1 diagrams to pandas DataFrames.

    Args:
        h0_diagram (np.ndarray): H0 diagram.
        h1_diagram (np.ndarray): H1 diagram.

    Returns:
        tuple: Two pandas DataFrames containing information about birth, death, and persistence.
    """
    accu = []
    for h_diag in [h0_diagram, h1_diagram]:
        df = pd.DataFrame({"Birth": h_diag[:, 0], "Death": h_diag[:, 1]})
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df["Persistence"] = (df["Death"] - df["Birth"])
        accu.append(df[["Birth", "Death", "Persistence"]].copy())
    return tuple(accu)


def get_h_fig_from_df(df: pd.DataFrame, title: str = "<b>H0 or H1</b>") -> go.Figure:
    """
    Create a 2D scatter plot figure from an H0 or H1 homology DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing columns for Birth, Death, and Persistence.
        title (str): The title for the scatter plot.

    Returns:
        go.Figure: A Plotly figure representing the H1 homology scatter plot.
    """

    fig = px.scatter(
        df,
        x="Birth",
        y="Death",
        title=title,
        color="Persistence",
        size="Persistence",
        color_continuous_scale=px.colors.sequential.YlOrBr,  # YlOrBr Plasma_r
        opacity=0.6,
        template="plotly_dark",
    )
    fig.add_shape(type="line", x0=0, y0=0,
                  x1=df["Death"].max(), y1=df["Death"].max(), opacity=0.1)
    return fig


def get_h1_diagrams_from_dct_paths(
    img_paths,
    save_df: bool = True,
    save_png: bool = True,
    save_html: bool = False,
    parquet_name: str = "base_test",
    saving_path: str = saving_path,
) -> Tuple[dict, dict]:
    """
    Compute homology diagrams and optionally save them as DataFrame in a parquet, PNG images, and HTML files.

    Args:
        dct_img_paths : chemins des images en gris scale
        save_df (bool): Whether to save H1 diagrams as DataFrames.
        save_png (bool): Whether to save H1 diagrams as PNG images.
        save_html (bool): Whether to save H1 diagrams as HTML files.
        parquet_name (str): The name for the saved Parquet DataFrame (if saved).
        saving_path (str): The directory path for saving outputs.

    Returns:
        Tuple[dict, dict]: A tuple containing dictionaries with H1 diagrams and associated Plotly figures.
    """
    
    if isinstance(img_paths, dict):
        iterator = img_paths.items()
        n_img = len(iterator)
    else:
        # img_paths peut être un array issue d'un découpage en batch
        iterator = img_paths
        n_img = img_paths.shape[0]
        
    
    if save_df:

        parquet_path = str(saving_path.joinpath(f"{parquet_name}.parquet"))
        lst_col = [
            "img_id",
            "h0__Births",
            "h0__Deaths",
            "h0__Persistences",
            "h1__Births",
            "h1__Deaths",
            "h1__Persistences",
        ]
        df_to_save = pd.DataFrame(columns=lst_col, data=np.full(
            [n_img, len(lst_col)], ""))

        def columns_as_strings(df): return df.astype(
            str).apply("\n".join, axis=0).values

        def update_df_to_save(i, img_id, df_h0, df_h1):
            df_to_save.iloc[i, 0] = img_id
            df_to_save.iloc[i, 1:4] = columns_as_strings(df_h0)
            df_to_save.iloc[i, 4:] = columns_as_strings(df_h1)

    i = 0
    
    for img_id, img_path in iterator:

        df_h0, df_h1 = get_dataframes_from_h0_h1_mats(
            *get_h0_h1_mats_from_img(
                io.imread(img_path)
                )
        )

        if save_png or save_html:
            fig_h0 = get_h_fig_from_df(
                df_h0, title=f"<b>0 dimensional holes</b> {img_id}")

            fig_h1 = get_h_fig_from_df(
                df_h1, title=f"<b>1 dimensional holes</b> {img_id}")

        if save_df:
            update_df_to_save(i, img_id, df_h0, df_h1)

        i += 1
        if save_df:
            fig_core_name = f"{parquet_name}_{img_id}"
        else:
            fig_core_name = f"{img_id}"

        if save_png:
            h0_png_path = str(saving_path.joinpath(
                "png", f"{fig_core_name}_h0.png"))
            h1_png_path = str(saving_path.joinpath(
                "png", f"{fig_core_name}_h1.png"))

            fig_h0.write_image(
                h0_png_path,
                format="png",
                engine="kaleido",
            )
            fig_h1.write_image(
                h1_png_path,
                format="png",
                engine="kaleido",
            )

        if save_html:

            html_path_h0 = str(
                saving_path.joinpath("html", f"{fig_core_name}_h0.html")
            )
            html_path_h1 = str(
                saving_path.joinpath("html", f"{fig_core_name}_h1.html")
            )
            fig_h0.write_html(html_path_h0)
            fig_h1.write_html(html_path_h1)

    if save_df:
        write(
            filename=parquet_path,
            data=df_to_save,
            append=os.path.exists(parquet_path),
            write_index=False,
        )

#### Final function######

def load_img_and_save_homology_to_parquet(root_path: Path, parquet_name: str = "basetest", save_df: bool = True, save_png: bool = False, batch_size: int = 10):
    """
    Load image paths, process them in batches, and save homology information to Parquet files.

    Args:
        root_path (Path): The root path where images are located.
        parquet_name (str): The base name for the Parquet files.
        save_df (bool): Whether to save homology information to Parquet files.
        save_png (bool): Whether to save homology plots as PNG files.
        batch_size (int): The size of each batch for processing images.
    """
    
    dct_img_paths = get_selected_img_paths(root_path)
    img_path_and_ids = np.array(list(dct_img_paths.items()))

    n_img = img_path_and_ids.shape[0]
    n_batch = n_img // batch_size

    for img_paths_batch in tqdm(np.array_split(img_path_and_ids, n_batch)):
        get_h1_diagrams_from_dct_paths(
            img_paths_batch,
            save_df=save_df,
            save_png=save_png,
            save_html=False,
            parquet_name=parquet_name,
            saving_path=root_path.joinpath("outputs")
        )


###############################
#   parquet and clustering    #
###############################

def homology_parquet_to_matrix_bootstraps(base_path: Path):
    """
    Read homology information from a Parquet file and create an embedding matrix.

    Args:
        base_path (Path): Path to the Parquet file.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing a DataFrame with identification information
        and an embedding matrix.
    """
    df = pd.read_parquet(base_path)
    # print(f"loading {base_path}")
    # print("##columns## \n", list(df))

    df_ident = pd.DataFrame({"mat_index": np.arange(
        len(df.index)), "df_index": df.index, "img_id": df["img_id"]})
    
    indices_gleason3 = np.array([i for i in np.arange(len(df.index)) if df.iloc[i]["img_id"][:9]=="Gleason 3"])
    indices_gleason4 = np.array([i for i in np.arange(len(df.index)) if df.iloc[i]["img_id"][:9]=="Gleason 4"])
    indices_gleason5 = np.array([i for i in np.arange(len(df.index)) if df.iloc[i]["img_id"][:9]=="Gleason 5"])

    nb_gleason3 = len(indices_gleason3)
    nb_gleason4 = len(indices_gleason4)
    nb_gleason5 = len(indices_gleason5)

    bootstrap_len = min([nb_gleason3, nb_gleason4, nb_gleason5]) # To have the same proportion of each Gleason type

    # Bootstraps creation

    bootstraps = []

    for _ in tqdm(range(100)): # We create 100 bootstraps
        indices_gleason3_bootstrap = np.random.choice(np.arange(len(indices_gleason3)),bootstrap_len,replace=False)
        indices_gleason4_bootstrap = np.random.choice(np.arange(len(indices_gleason4)),bootstrap_len,replace=False)
        indices_gleason5_bootstrap = np.random.choice(np.arange(len(indices_gleason5)),bootstrap_len,replace=False)

        gleason3_bootstrap = indices_gleason3[indices_gleason3_bootstrap]
        gleason4_bootstrap = indices_gleason4[indices_gleason4_bootstrap]
        gleason5_bootstrap = indices_gleason5[indices_gleason5_bootstrap]

        indices_bootstrap = np.concatenate([gleason3_bootstrap, gleason4_bootstrap, gleason5_bootstrap])

        bootstrap = df.iloc[indices_bootstrap]
        bootstrap = bootstrap.reset_index(drop=True)

        len_h0 = bootstrap['h0__Deaths'].str.split("\n").apply(len)
        len_h1 = bootstrap['h1__Deaths'].str.split("\n").apply(len)
        min_len_h0 = len_h0.min()
        min_len_h1 = len_h1.min()
    # mean_len_h0 = len_h0.mean()
    # mean_len_h1 = len_h1.mean()

    # print("\n ##h0 length##")
    # print(len_h0.describe())
    # print("\n ##h1 length##")
    # print(len_h1.describe())

        def f_sort_cut(s, cut):
            lst = s.split("\n")
            lst = list(map(float, lst))
            lst.sort(reverse=True)
            lst = lst[:cut]
            return lst

        def f_h0(s): return f_sort_cut(s, min_len_h0)
        def f_h1(s): return f_sort_cut(s, min_len_h1)

        series_h0 = bootstrap['h0__Persistences'].apply(f_h0)
        series_h1 = bootstrap['h1__Persistences'].apply(f_h1)
        embedding_mat = np.zeros((len(bootstrap.index), min_len_h0+min_len_h1))

        for i in range(len(bootstrap.index)):
            embedding_mat[i, :] = np.array(series_h0[i]+series_h1[i])
        b = {"embedding_mat" : embedding_mat, "images" : df_ident["img_id"].iloc[indices_bootstrap].to_numpy()}
        bootstraps.append(b)

    # Original data processing
    len_h0 = df['h0__Deaths'].str.split("\n").apply(len)
    len_h1 = df['h1__Deaths'].str.split("\n").apply(len)
    min_len_h0 = len_h0.min()
    min_len_h1 = len_h1.min()
    series_h0 = df['h0__Persistences'].apply(f_h0)
    series_h1 = df['h1__Persistences'].apply(f_h1)
    original_embedding_mat = np.zeros((len(df.index), min_len_h0+min_len_h1))

    for i in range(len(df.index)):
        original_embedding_mat[i, :] = np.array(series_h0[i]+series_h1[i])
    original = {"embedding_mat": original_embedding_mat, "images" : df_ident["img_id"].to_numpy()}
    
    return df_ident, bootstraps, original


def pca_before_clustering(data, n_components=6, standard=True):
    """
    Apply Principal Component Analysis (PCA) to the input data.

    Args:
        data (array-like or pd.DataFrame): The input data.
        n_components (int, optional): Number of components to keep. Defaults to 6.
        standard (bool, optional): Whether to standardize the data. Defaults to True.

    Returns:
        array-like: Transformed data after PCA.
    """
    # standardization
    if standard:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    res = pca.fit_transform(data)
    # print("variance ratio", pca.explained_variance_ratio_)
    return res,np.sum(pca.explained_variance_ratio_)

def make_pca_bootstraps(bootstraps, original, n_components = 6):
    vars_explained = []
    for b in bootstraps:
        embedding_mat = b["embedding_mat"]
        pca, var_explained = pca_before_clustering(embedding_mat, n_components = n_components, standard = False)
        b["reduced"] = pca
        vars_explained.append(var_explained)
    reduced_original,_ = pca_before_clustering(original["embedding_mat"], n_components = n_components, standard = False) # We will need the original data to compute the Gap Statistic
    original["reduced"] = reduced_original
    return bootstraps, original, vars_explained

def get_clusters(bootstraps, original,  nb_clusters=6):
    for b in bootstraps:
        linked = linkage(b["reduced"], "ward")
        clusters_indices = fcluster(linked, nb_clusters, criterion="maxclust")
        clusters = [[j for j in range(len(clusters_indices)) if clusters_indices[j]==i] for i in range(1,nb_clusters+1)]
        b["clusters"] = clusters
    linked = linkage(original["reduced"], "ward")
    clusters_indices = fcluster(linked, nb_clusters, criterion="maxclust")
    clusters = [[j for j in range(len(clusters_indices)) if clusters_indices[j]==i] for i in range(1,nb_clusters+1)]
    original["clusters"] = clusters
    return bootstraps, original

def transform_gleason(bootstraps):
    for b in bootstraps:
        images = b["images"]
        gleason_coords = []
        for c in b["clusters"]:
            cluster_images = images[c]
            nb_gleason3 = np.sum([(s[:9]=="Gleason 3") for s in cluster_images])
            nb_gleason4 = np.sum([(s[:9]=="Gleason 4") for s in cluster_images])
            nb_gleason5 = np.sum([(s[:9]=="Gleason 5") for s in cluster_images])
            
            gleason_coords.append([nb_gleason3, nb_gleason4, nb_gleason5])
        b["gleason_coords"] = np.array(gleason_coords)
    return bootstraps

def meta_clustering(gleason_points, nb_clusters=6):
    linked = linkage(gleason_points, method='ward')
    meta_clusters_indices = fcluster(linked, nb_clusters, criterion="maxclust")
    meta_clusters = [[j for j in range(len(meta_clusters_indices)) if meta_clusters_indices[j]==i] for i in range(1,nb_clusters+1)]
    return meta_clusters

def choose_representative_bootstrap(gleason_bootstraps, meta_clusters):
    min_distance = np.infty
    meta_centroids = [np.mean(c) for c in meta_clusters]
    for b in gleason_bootstraps:
        dist = 0
        gleason_coords = b["gleason_coords"]
        for c in gleason_coords:
            for k in range(len(meta_clusters)):
                if np.equal(meta_clusters[k],c).all(1).any():
                    dist += np.linalg.norm(meta_centroids[k]-c)
        if dist<min_distance:
            min_distance=dist
            representative_bootstrap = b
    return representative_bootstrap

def get_meta_bootstraps(meta_clusters, n_bootstraps=1000):
    bootstraps = []
    for _ in range(n_bootstraps):
        bootstrap = []
        for c in meta_clusters: # Bootstrapping each meta-cluster
            indices = np.random.choice(len(c), len(c), replace=True)
            bootstrap += [c[i] for i in indices]
        bootstraps.append(bootstrap)
    return bootstraps

def get_stability(bootstrap, meta_clusters):
    visited = []
    max_similarities = np.zeros(len(bootstrap))
    for i in range(len(bootstrap)):
        for j in range(len(bootstrap)):
            if j not in visited:
                c1 = np.array(bootstrap[i])
                c2 = np.array(meta_clusters[j])
                intersection = len(np.intersect1d(c1,c2))
                jaccard = intersection/(len(c1)+len(c2)-intersection)
                if jaccard>max_similarities[i]:
                    max_similarities[i] = jaccard
                    best = j
        visited.append(best)

    return np.mean(max_similarities)

######################
#   Visualization    #
######################

def get_repr_imgs_and_clustering(representative_bootstrap):
    repr_img = representative_bootstrap["images"]
    repr_clustering_6 = np.empty(repr_img.shape[0], dtype='object')
    repr_clustering_gleason = np.empty(repr_img.shape[0], dtype='object')

    for i,img in enumerate(repr_img):
        for k,c in enumerate(representative_bootstrap["clusters"]):
            if i in c:
                repr_clustering_6[i]=f"c{k+1}"
                
        repr_clustering_gleason[i] = img[:9]

    return repr_img, repr_clustering_6, repr_clustering_gleason


def save_tsne(representative_bootstrap, saving_path, lst_perplexity=[47], save_pair_plot = False, test_other_clust = True, metric = ""):
    repr_img, repr_clustering_6, repr_clustering_gleason = get_repr_imgs_and_clustering(representative_bootstrap)
    repr_reduced = representative_bootstrap["reduced"]
    if test_other_clust:
        dct_clust = {
            "Ward": repr_clustering_6,
            "GMM":GaussianMixture(n_components=6, random_state=0).fit_predict(repr_reduced).astype(str),
            "HDBSCAN": DBSCAN(eps = 0.01).fit_predict(repr_reduced).astype(str),

        }
    else:
        dct_clust = {
            "Ward": repr_clustering_6,
        }

    pair_plot = save_pair_plot
    if pair_plot:
        df_reduced = pd.DataFrame(repr_reduced, columns = np.arange(6))
        fig = sns.pairplot(df_reduced)
        plt.savefig(str(saving_path.joinpath(f"representativ_pair_plot.png")))

        for algo_name, clust in dct_clust.items():
            df_reduced[algo_name] = clust

    for perplexity in lst_perplexity:
        print(f"perplexity:{perplexity}")
        repr_tsne = TSNE(n_components=2, 
                                learning_rate='auto', 
                                init='random', 
                                perplexity=perplexity
                                ).fit_transform(repr_reduced)
        


        for algo_name, clust in dct_clust.items():
            fig = px.scatter(
                x = repr_tsne[:,0], 
                y = repr_tsne[:,1], 
                color = clust, 
                color_discrete_sequence = px.colors.qualitative.T10,
                symbol = repr_clustering_gleason,
                template="plotly_dark", 
                hover_name = repr_img, 
                title = f"<b>T-SNE representative bootstrap</b> Clustering {metric} {algo_name} <br> perplexity = {perplexity}"
            )
            fig.write_html(str(saving_path.joinpath(f"repr_tsne_perplexity_{metric}{algo_name}-{perplexity}.html")))
            if pair_plot:
                fig = sns.pairplot(df_reduced[np.arange(6).tolist()+[algo_name]], 
                        hue=algo_name
                                )
                plt.savefig(str(saving_path.joinpath(f"representativ_pair_plot_clustered_{metric}{algo_name}.png")))
        pair_plot = False 

def split_imgs_by_clusters(representative_bootstrap, clutering):
    repr_img = representative_bootstrap["images"]
    clustered_imgs = {c:[] for c in np.unique(clutering)}
    for i,img in enumerate(repr_img):
        clustered_imgs[clutering[i]].append(img)
    return clustered_imgs

def get_concatenated_homogies_df(df):
    df2 = df.copy().drop(columns=['img_id'])
    df2 = df2.apply("\n".join , axis=0)
    f_convert = lambda lst: list(map(float, lst))
    df2 = df2.str.split("\n").apply(f_convert)
    lst_col = ["Births", "Deaths", "Persistences"]
    df_h0 = pd.DataFrame({col:df2.loc[f"h0__{col}"] for col in lst_col})
    df_h1 = pd.DataFrame({col:df2.loc[f"h1__{col}"] for col in lst_col})
    return df_h0, df_h1

def get_df_by_cluster_and_h(base_path, representative_bootstrap, clutering):
    df = pd.read_parquet(base_path)
    clustered_imgs = split_imgs_by_clusters(representative_bootstrap, clutering)
    dct_df = {}
    for clust_name, imgs in  clustered_imgs.items():
        df_clust = df.loc[df["img_id"].isin(imgs)]
        df_h0, df_h1 = get_concatenated_homogies_df(df_clust)
        dct_df[f"{clust_name}_h0"] = df_h0
        dct_df[f"{clust_name}_h1"] = df_h1
    return dct_df


def save_homology_densities_one_cluster(df,  cluster_name="c1", saving_path=None, show=False):
    plt.clf()
    plt.style.use('dark_background')
    plt.figure()
    print(cluster_name)

    fig =  sns.kdeplot(data=df, x=f"Births", y= f"Deaths",#hue=f"Persistences",
        # fill=True, palette="crest",
        # cmap="mako",
        thresh=0.1,
        levels=40,
        fill=True,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        alpha=.8,
        warn_singular=False
        )
    plt.title(cluster_name)
    if show:
        fig.show()
    if saving_path:
        figpath = saving_path.joinpath(f"{cluster_name}.png")
        plt.savefig(figpath) 


def save_homology_densities(base_path, representative_bootstrap, saving_path, metric = "" ):
    repr_img, repr_clustering_6, repr_clustering_gleason = get_repr_imgs_and_clustering(representative_bootstrap)

    dct_clust = {
            "Ward": repr_clustering_6,
            # "GMM":GaussianMixture(n_components=6, random_state=0).fit_predict(repr_reduced).astype(str),
        }
    for algo_name, clust in dct_clust.items():
        dct_df = get_df_by_cluster_and_h(base_path, representative_bootstrap, clust)
        for clust_name, df in dct_df.items():
            save_homology_densities_one_cluster(df,  f"{metric}{algo_name}_{clust_name}", saving_path=saving_path, show=False)


########################
#   dtw exploration    #
########################

from tslearn.metrics import dtw as ts_dtw

def get_all_diagram_as_list(base_path):
    df = pd.read_parquet(base_path)
    lst_st = []
    lst_gleason_class = []
    f_convert = lambda lst: list(map(float, lst))
    for index, row in df.iterrows():
        arr_h1 = f_convert(row["h0__Persistences"].split("\n"))
        arr_h0 = f_convert(row["h1__Persistences"].split("\n"))
        lst_st.append([arr_h0, arr_h1])
        lst_gleason_class.append(row["img_id"][:9])
    return lst_st, lst_gleason_class


#Complètement brute force => long! (20-48h)
# à parallèliser avec kleops
def get_dtw_mat(lst_st):
    mat_dist = np.zeros((len(lst_st), len(lst_st)))
    for i in tqdm(range(len(lst_st))):
        for j in range(i-1):
            h0_i = lst_st[i][0]
            h1_i = lst_st[i][1]
            h0_j = lst_st[j][0]
            h1_j = lst_st[j][1]

            h0_dtw = ts_dtw(h0_i, h0_j)
            h1_dtw = ts_dtw(h1_i, h1_j)

            mat_dist[i,j] = np.linalg.norm([h0_dtw, h1_dtw])
            mat_dist[j,i] = np.linalg.norm([h0_dtw, h1_dtw])
    return mat_dist

def update_clustering_from_mat_dist(mat_dist, representative_bootstrap, nb_clusters=6):
    condensed_dist_mat = ssd.squareform(mat_dist) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
    linked = linkage(condensed_dist_mat, 'ward')
    clusters_indices = fcluster(linked, nb_clusters, criterion="maxclust")
    clusters = [[j for j in range(len(clusters_indices)) if clusters_indices[j]==i] for i in range(1,nb_clusters+1)]
    representative_bootstrap["clusters"] = clusters
    gleason_coords = []
    for c in representative_bootstrap["clusters"]:
        cluster_images = representative_bootstrap["images"][c] #images.iloc[c]
        nb_gleason3 = np.sum([(s[:9]=="Gleason 3") for s in cluster_images])
        nb_gleason4 = np.sum([(s[:9]=="Gleason 4") for s in cluster_images])
        nb_gleason5 = np.sum([(s[:9]=="Gleason 5") for s in cluster_images])
        
        gleason_coords.append([nb_gleason3, nb_gleason4, nb_gleason5])
    representative_bootstrap["gleason_coords"] = np.array(gleason_coords)

import dtaidistance as dtai

def get_dtw_mat_fast(base_path):
    df = pd.read_parquet(base_path)
    lst_st_h0 = []
    lst_st_h1 = []
    f_convert = lambda lst: list(map(float, lst))
    for index, row in df.iterrows():
        arr_h1 = np.array(f_convert(row["h0__Persistences"].split("\n")))
        arr_h0 = np.array(f_convert(row["h1__Persistences"].split("\n")))
        lst_st_h0.append(arr_h0)
        lst_st_h0.append(arr_h1)
    mat_h0 =  dtai.dtw.distance_matrix_fast(lst_st_h0)
    mat_h1 =  dtai.dtw.distance_matrix_fast(lst_st_h1)
    return np.sqrt(mat_h0**2+mat_h1**2)

def get_sub_dist_mat(mat_dist, representative_bootstrap, df_ident):
    imgs = representative_bootstrap["images"]
    index_repr = df_ident["img_id"].isin(imgs)
    return mat_dist[index_repr, :][:, index_repr]