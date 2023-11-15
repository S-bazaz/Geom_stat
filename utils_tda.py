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


from pathlib import Path
from ripser import ripser, Rips
from scipy import sparse
from fastparquet import write

from typing import Dict, Optional, Union, BinaryIO, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from scipy.spatial.distance import pdist
# from scipy.cluster.hierarchy import ward

from scipy.cluster.hierarchy import dendrogram, linkage
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
    puis importe le fichier

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
        dct_accu[f"{prefix}{pathobj.stem}"] = io.imread(file_path)


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


def get_selected_data_files(doc: Dict) -> Dict[str, Dict[str, str]]:
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


def get_selected_img(root_path):
    yaml_path = root_path.joinpath("raw_data_selection.yaml")
    doc = load_yaml(str(yaml_path))
    add_path_to_nodes(doc["raw_images"], root_path.joinpath("raw_images"))

    return get_selected_data_files(doc)


##########################
#   images to parquet    #
##########################

def get_h0_h1_mats_from_binary_mat(bin_mat: np.ndarray) -> np.ndarray:

    rips = Rips(maxdim=1, coeff=2, verbose=False)
    bin_mat2 = np.array(bin_mat)
    if bin_mat2.shape[1] > bin_mat2.shape[0]:
        bin_mat2 = bin_mat2.T
    diagrams = rips.fit_transform(
        bin_mat2, distance_matrix=False, metric="euclidean")
    return np.round(diagrams[0], 3), np.round(diagrams[1], 3)


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


def get_dataframes_from_h0_h1_mats(h0_diagram: np.ndarray, h1_diagram: np.ndarray) -> pd.DataFrame:

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


def get_h1_diagrams_from_dct_mat(
    dct_img: dict,
    save_df: bool = True,
    save_png: bool = True,
    save_html: bool = False,
    parquet_name: str = "base_test",
    saving_path: str = saving_path,
) -> Tuple[dict, dict]:
    """
    Compute H1 homology diagrams and optionally save them as DataFrames, PNG images, and HTML files.

    Args:
        dct_img
        save_df (bool): Whether to save H1 diagrams as DataFrames.
        save_png (bool): Whether to save H1 diagrams as PNG images.
        save_html (bool): Whether to save H1 diagrams as HTML files.
        parquet_name (str): The name for the saved Parquet DataFrame (if saved).
        saving_path (str): The directory path for saving outputs.

    Returns:
        Tuple[dict, dict]: A tuple containing dictionaries with H1 diagrams and associated Plotly figures.
    """
    
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
        df_to_save = pd.DataFrame(columns=lst_col, data = np.full([len(dct_img),len(lst_col)], ""))
        
        columns_as_strings = lambda df : df.astype(str).apply("\n".join, axis=0).values
        
        def update_df_to_save(i, img_id, df_h0, df_h1):
            df_to_save.iloc[i,0] = img_id
            df_to_save.iloc[i,1:4] = columns_as_strings(df_h0)
            df_to_save.iloc[i,4:] = columns_as_strings(df_h1)
        
        
    
    i=0
    for img_id, img in dct_img.items():

        df_h0, df_h1 = get_dataframes_from_h0_h1_mats(
            *get_h0_h1_mats_from_img(img)
            )
        
        if save_png or save_html:
            fig_h0 = get_h_fig_from_df(
                df_h0, title=f"<b>0 dimensional holes</b> {img_id}")
    
            fig_h1 = get_h_fig_from_df(
                df_h1, title=f"<b>1 dimensional holes</b> {img_id}")

        if save_df:
            update_df_to_save(i, img_id, df_h0, df_h1)

        i+=1



        fig_core_name = f"{parquet_name}_{img_id}"

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

def load_img_and_save_homology_to_parquet(root_path, parquet_name="basetest"):
    dct_img = get_selected_img(root_path)
    get_h1_diagrams_from_dct_mat(
        dct_img,
        save_df=True,
        save_png=True,
        save_html=False,
        parquet_name=parquet_name,
        saving_path=root_path.joinpath("outputs")
    )


###############################
#   parquet and clustering    #
###############################

def homology_parquet_to_matrix(base_path):
    df = pd.read_parquet(base_path)
    print(f"loading {base_path}")
    print("##columns## \n", list(df))

    df_ident = pd.DataFrame({"mat_inex": np.arange(
        len(df.index)), "df_index": df.index, "img_id": df["img_id"]})
    len_h0 = df['h0__Deaths'].str.split("\n").apply(len)
    len_h1 = df['h1__Deaths'].str.split("\n").apply(len)
    min_len_h0 = len_h0.min()
    min_len_h1 = len_h1.min()
    # mean_len_h0 = len_h0.mean()
    # mean_len_h1 = len_h1.mean()

    print("\n ##h0 length##")
    print(len_h0.describe())
    print("\n ##h1 length##")
    print(len_h1.describe())

    def f_sort_cut(s, cut):
        lst = s.split("\n")
        lst = list(map(float, lst))
        lst.sort(reverse=True)
        lst = lst[:cut]
        return lst

    def f_h0(s): return f_sort_cut(s, min_len_h0)
    def f_h1(s): return f_sort_cut(s, min_len_h1)

    series_h0 = df['h0__Persistences'].apply(f_h0)
    series_h1 = df['h1__Persistences'].apply(f_h1)
    embedding_mat = np.zeros((len(df.index), min_len_h0+min_len_h1))

    for i in range(len(df.index)):
        embedding_mat[i, :] = np.array(series_h0[i]+series_h1[i])
    return df_ident, embedding_mat


def pca_before_clustering(data, n_components=6, standard=True):
    # standardization
    if standard:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    res = pca.fit_transform(data)
    print("variance ratio", pca.explained_variance_ratio_)
    return res
