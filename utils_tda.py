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

def get_h1_mat_from_binary_mat(bin_mat: np.ndarray) -> np.ndarray:
    """
    Calculate the H1 homology matrix from a binary matrix.

    Args:
        bin_mat (np.ndarray): A binary matrix where rows represent data points and 
                            columns represent binary features.

    Returns:
        np.ndarray: The H1 homology matrix obtained from the binary matrix.
    """
    rips = Rips(maxdim=1, coeff=2, verbose=False)
    bin_mat2 = np.array(bin_mat)
    if bin_mat2.shape[1] > bin_mat2.shape[0]:
        bin_mat2 = bin_mat2.T
    diagrams = rips.fit_transform(bin_mat2, distance_matrix=False, metric="euclidean")
    return diagrams[1]


def get_dataframe_from_h1_mat(h1_diagram: np.ndarray) -> pd.DataFrame:
    """
    Create a Pandas DataFrame from an H1 homology matrix.

    Args:
        h1_diagram (np.ndarray): The H1 homology matrix with columns representing 
                               birth and death values for each topological feature.

    Returns:
        pd.DataFrame: A DataFrame containing columns for Birth, Death, and Hole scale.
    """
    df = pd.DataFrame({"Birth": h1_diagram[:, 0], "Death": h1_diagram[:, 1]})
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df["rayon_moyen"] = (df["Birth"] + df["Death"]) / 2
    df["Hole scale"] = np.sqrt(2) * (df["Death"] - df["rayon_moyen"])
    return df[["Birth", "Death", "Hole scale"]]


def get_h1_fig_from_df(df: pd.DataFrame, title: str = "<b>H1</b>") -> go.Figure:
    """
    Create a 2D scatter plot figure from an H1 homology DataFrame.

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
        color_continuous_scale=px.colors.sequential.Viridis,
        opacity=0.6,
        template="plotly_dark",
    )

    fig.add_shape(
        type="line", x0=0, y0=0, x1=df["Death"].max(), y1=df["Death"].max(), opacity=0.1
    )
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
        df_to_save = pd.DataFrame(
            {"img_id": [], "Births": [], "Deaths": [], "Hole scales": []}
        )
    for img_id, bin_mat in dct_bin_mat.items():

        h1_diagram = get_h1_mat_from_binary_mat(bin_mat)

        dct_df[img_id] = h1_diagram.copy()

        df = get_dataframe_from_h1_mat(h1_diagram)
        if save_df:
            dct_row = {
                f"{col}s": df[col].to_string(index=False, header=False,) for col in df
            }
            dct_row["img_id"] = img_id
            df_to_save.loc[len(df_to_save)] = dct_row.copy()
        fig = get_h1_fig_from_df(df, title=f"<b>H1</b>{img_id}")

        dct_figs[img_id] = fig

        if save_png:
            png_path = str(saving_path.joinpath("png", f"{parquet_name}_{img_id}.png"))
            fig.write_image(
                png_path,
                format="png",
                engine="kaleido",
                # width = 1980,
                # height = 1080,
            )
        if save_html:
            html_path = str(
                saving_path.joinpath("html", f"{parquet_name}_{img_id}.html")
            )
            fig.write_html(html_path)
    if save_df:
        write(
            filename=parquet_path,
            data=df,
            append=os.path.exists(parquet_path),
            write_index=False,
        )
    return dct_df, dct_figs
