# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from ripser import ripser, Rips
from fastparquet import write

from typing import Tuple

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

##################
#   functions    #
##################


def get_h0_h1_mats_from_binary_mat(bin_mat: np.ndarray) -> np.ndarray:

    rips = Rips(maxdim=1, coeff=2, verbose=False)
    bin_mat2 = np.array(bin_mat)
    if bin_mat2.shape[1] > bin_mat2.shape[0]:
        bin_mat2 = bin_mat2.T
    diagrams = rips.fit_transform(
        bin_mat2, distance_matrix=False, metric="euclidean")
    return np.round(diagrams[0],3), np.round(diagrams[1],3)


def get_dataframes_from_h0_h1_mats(h0_diagram: np.ndarray, h1_diagram: np.ndarray) -> pd.DataFrame:

    accu = []
    for h_diag in [h0_diagram, h1_diagram]:

        df = pd.DataFrame({"Birth": h_diag[:, 0], "Death": h_diag[:, 1]})
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df["rayon_moyen"] = (df["Birth"] + df["Death"]) / 2
        df["Hole scale"] = np.sqrt(2) * (df["Death"] - df["rayon_moyen"])
        accu.append(df[["Birth", "Death", "Hole scale"]].copy())
    return tuple(accu)


def get_h_fig_from_df(df: pd.DataFrame, title: str = "<b>H0 or H1</b>") -> go.Figure:
    """
    Create a 2D scatter plot figure from an H0 or H1 homology DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing columns for Birth, Death, and Hole scale.
        title (str): The title for the scatter plot.

    Returns:
        go.Figure: A Plotly figure representing the H1 homology scatter plot.
    """

    fig = px.scatter(
        df,
        x="Birth",
        y="Death",
        title=title,
        color="Hole scale",
        size="Hole scale",
        color_continuous_scale=px.colors.sequential.YlOrBr,  # YlOrBr Plasma_r
        opacity=0.6,
        template="plotly_dark",
    )
    fig.add_shape(type="line", x0=0, y0=0,
                  x1=df["Death"].max(), y1=df["Death"].max(), opacity=0.1)
    return fig


def get_h1_diagrams_from_dct_bin(
    dct_bin_mat: dict,
    save_df: bool = True,
    save_png: bool = True,
    save_html: bool = False,
    parquet_name: str = "H1_test",
    saving_path: str = saving_path,
) -> Tuple[dict, dict]:
    """
    Compute H1 homology diagrams and optionally save them as DataFrames, PNG images, and HTML files.

    Args:
        dct_bin_mat (dict): A dictionary containing image IDs as keys and binary matrices as values.
        save_df (bool): Whether to save H1 diagrams as DataFrames.
        save_png (bool): Whether to save H1 diagrams as PNG images.
        save_html (bool): Whether to save H1 diagrams as HTML files.
        parquet_name (str): The name for the saved Parquet DataFrame (if saved).
        saving_path (str): The directory path for saving outputs.

    Returns:
        Tuple[dict, dict]: A tuple containing dictionaries with H1 diagrams and associated Plotly figures.
    """
    dct_df = {}
    dct_figs = {}
   
    if save_df:
        parquet_path = str(saving_path.joinpath(f"{parquet_name}.parquet"))
        col_diag = ["Birth", "Death", "Hole scale"]
        lst_col = ["img_id"] + [f"h0__{col}s"for col in col_diag] + [f"h1__{col}s"for col in col_diag]
        df_to_save = pd.DataFrame(columns=lst_col)
        
    for img_id, bin_mat in dct_bin_mat.items():

        h0_diagram, h1_diagram = get_h0_h1_mats_from_binary_mat(bin_mat)
        df_h0, df_h1 = get_dataframes_from_h0_h1_mats(h0_diagram, h1_diagram)
        dct_df[img_id] = {"h0": df_h0.copy, "h1": df_h1.copy}
        if save_df:
            
            dct_row = {"img_id": img_id}
            dct_row.update(
                {f"h0__{col}s": df_h0[col].to_string(
                    index=False, header=False) for col in df_h0}
            )
            dct_row.update(
                {f"h1__{col}s": df_h1[col].to_string(
                    index=False, header=False,) for col in df_h1}
            )

            df_to_save.loc[len(df_to_save)] = dct_row.copy()

        fig_h0 = get_h_fig_from_df(
            df_h0, title=f"<b>0 dimensional holes</b> {img_id}")
        fig_h1 = get_h_fig_from_df(
            df_h1, title=f"<b>1 dimensional holes</b> {img_id}")

        dct_figs[img_id] = {"h0": fig_h0, "h1": fig_h1}

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
                h0_png_path,
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
    return dct_df, dct_figs

def homology_parquet_to_matrix(base_path):
    df = pd.read_parquet(base_path)
    print(f"loading {base_path}")
    print("##columns## \n",list(df))
    
    df_ident = pd.DataFrame({"mat_inex":np.arange(len(df.index)), "df_index":df.index, "img_id":df["img_id"]})
    
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
    
    f_h0 = lambda s: f_sort_cut(s, min_len_h0)
    f_h1 = lambda s: f_sort_cut(s, min_len_h1)
    
    series_h0 = df['h0__Hole scales'].apply(f_h0)
    series_h1 = df['h1__Hole scales'].apply(f_h1)
    embedding_mat = np.zeros((len(df.index), min_len_h0+min_len_h1))

    for i in range(len(df.index)):
        embedding_mat[i,:] = np.array(series_h0[i]+series_h1[i])
    return df_ident, embedding_mat
        

def pca_before_clustering(data, n_components = 6 ,standard = True):
    #standardization
    if standard:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    res = pca.fit_transform(data)
    print("variance ratio", pca.explained_variance_ratio_)
    return res



