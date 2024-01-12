#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:04:08 2018

@author: antony
"""

import pandas as pd
import phenograph
import collections
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import lib10x
import libplot
from libsparse import SparseDataFrame


# As per 10x
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
TSNE_LEARNING = 200  # 200
TSNE_RANDOM_STATE = 0  # 42
TSNE_METRIC = "correlation"  #'euclidean' #'correlation'
TSNE_INIT = "pca"


def get_dim_file(dir, name, mode="tsne"):
    if dir.endswith("/"):
        dir = dir[:-1]

    return "{}/{}_data_{}.txt".format(dir, mode, name)


def get_tsne_file(dir, name, tpmmode=True, logmode=True):
    return get_dim_file(dir, name, mode="tsne")


def read_tsne(file):
    print("Reading clusters from {}...".format(file))

    return pd.read_csv(file, sep="\t", header=0, index_col=0)


def new_tsne():
    return TSNE(
        n_components=2,
        verbose=1,
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LEARNING,
        n_iter=TSNE_MAX_ITER,
        method="barnes_hut",
        random_state=TSNE_RANDOM_STATE,
        init=TSNE_INIT,
        metric=TSNE_METRIC,
    )


def load_tsne(data, name, n=50, tpmmode=True, logmode=True, exclude=[]):
    """
    Run t-sne

    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    name : str
        Name of run.

    Returns
    -------
    tsne :  array, shape (n_samples, 2)
        The tsne coordinates for each sample

    """

    file = get_tsne_file(name, tpmmode=tpmmode, logmode=logmode)

    if not os.path.isfile(file):
        print("{} was not found, creating it with n={}...".format(file, n))

        _, pca = lib10x.pca(data, n=n, exclude=exclude)

        tsne = new_tsne()

        tsne_results = tsne.fit_transform(pca)

        data = pd.DataFrame(
            {
                "Barcode": data.index,
                "TSNE-1": tsne_results[:, 0],
                "TSNE-2": tsne_results[:, 1],
            }
        )
        data = data[["Barcode", "TSNE-1", "TSNE-2"]]
        data = data.set_index("Barcode")

        data.to_csv(file, sep="\t", header=True, index=True)

    return read_tsne(file)


def load_pca_tsne(
    pca, name, tpmmode=True, logmode=True, exclude=[], cache=True, dir="."
):
    """
    Run t-sne using pca result

    Parameters
    ----------
    pca : array, shape (n_samples, n_pca)
        pca matrix.

    name: str
        name of pca results

    Returns
    -------
    tsne :  array, shape (n_samples, 2)
        The tsne coordinates for each sample
    """

    file = get_tsne_file(dir, name, tpmmode=tpmmode, logmode=logmode)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it...".format(file))

        # perplexity = 5, n_iter = 5000, learning = 10

        tsne = new_tsne()

        if isinstance(pca, SparseDataFrame):
            tsne_results = SparseDataFrame(
                tsne.fit_transform(pca.data), pca.index, pca.columns
            )
        else:
            tsne_results = tsne.fit_transform(pca)

        data = pd.DataFrame(
            {
                "Barcode": pca.index,
                "TSNE-1": tsne_results[:, 0],
                "TSNE-2": tsne_results[:, 1],
            }
        )
        data = data[["Barcode", "TSNE-1", "TSNE-2"]]
        data = data.set_index("Barcode")

        data.to_csv(file, sep="\t", header=True)

    return read_tsne(file)


def lz_dz_diversity_plot(diversity, x, y, lz_indices, dz_indices, ax, cmap, norm):
    # How many labels to cycle through (it cannot exceed the number of colors)

    # cmap = plt.cm.plasma
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=max(1, round(max(diversity))))

    ret = None

    if len(lz_indices) > 0:
        div = np.take(diversity, lz_indices)
        x1 = np.take(x, lz_indices)
        y1 = np.take(y, lz_indices)
        indices = np.argsort(div)
        div = np.take(div, indices)
        x1 = np.take(x1, indices)
        y1 = np.take(y1, indices)
        ret = ax.scatter(
            x1,
            y1,
            c=div,
            cmap=cmap,
            norm=norm,
            s=lib10x.MARKER_SIZE,
            alpha=0.8,
            marker="^",
        )

    if len(dz_indices) > 0:
        div = np.take(diversity, dz_indices)
        x1 = np.take(x, dz_indices)
        y1 = np.take(y, dz_indices)
        indices = np.argsort(div)
        div = np.take(div, indices)
        x1 = np.take(x1, indices)
        y1 = np.take(y1, indices)
        ret = ax.scatter(
            x1,
            y1,
            c=div,
            cmap=cmap,
            norm=norm,
            alpha=0.8,
            s=lib10x.MARKER_SIZE,
            marker="o",
        )

    return ret


def diversity_plot(
    pca_results, diversity, lz_indices, dz_indices, prefix, diversity_type
):
    fig, ax = lib10x.make_figure()

    cmap = plt.cm.plasma
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(1, round(max(diversity))))

    x = pca_results["tsne-1"].tolist()
    y = pca_results["tsne-2"].tolist()

    lz_dz_diversity_plot(diversity, x, y, lz_indices, dz_indices, ax, cmap, norm)

    libplot.format_simple_axes(ax, title="t-SNE")

    cax = fig.add_axes([0.8, 0.05, 0.15, 0.02])
    # d = np.array([[0,1]])
    # im = cax.imshow(d, interpolation='nearest', cmap=cmap, norm=norm)
    # fig.colorbar(im, cax=cax, orientation='horizontal')

    matplotlib.colorbar.ColorbarBase(
        cax, cmap=cmap, norm=norm, ticks=[0.0, 1.0], orientation="horizontal"
    )

    ax.set_title("Shannon Diversity Index")
    libplot.save_plot(fig, "tsne_{}_{}_diversity.pdf".format(prefix, diversity_type))


def tsne_legend(ax, labels, colors):
    labeln = min(len(colors), np.max(labels) + 1)

    for i in range(0, labeln):
        color = colors[i]

        size = len(np.where(labels == i)[0])

        ax.scatter(
            [],
            [],
            marker="o",
            color=color,
            alpha=0.9,
            s=libplot.MARKER_SIZE,
            label="Cluster {} ({})".format(i + 1, size),
        )


def get_tsne_plot_name(name, t1=1, t2=2):
    return "tsne_{}.pdf".format(name)  # , t1, t2)


def format_simple_axes(ax, title="t-SNE", dim1=1, dim2=2, subtitle1="", subtitle2=""):
    libplot.invisible_axes(ax)

    ax.annotate(
        "",
        xy=(40, 0),  # theta, radius
        xytext=(-2, 0),
        xycoords="axes pixels",
        textcoords="axes pixels",
        arrowprops=dict(arrowstyle="->", facecolor="red"),
    )

    ax.annotate(
        "",
        xy=(0, 40),  # theta, radius
        xytext=(0, -2),
        xycoords="axes pixels",
        textcoords="axes pixels",
        arrowprops=dict(arrowstyle="->", facecolor="black"),
    )

    if subtitle1 != "":
        ax.text(
            0,
            -0.04,
            "{} {} ({})".format(title, dim1, subtitle1),
            transform=ax.transAxes,
        )
    else:
        ax.text(0, -0.04, "{} {}".format(title, dim1), transform=ax.transAxes)

    if subtitle2 != "":
        ax.text(
            -0.04,
            0,
            "{} {} ({})".format(title, dim2, subtitle2),
            va="bottom",
            transform=ax.transAxes,
            rotation=90,
        )
    else:
        ax.text(
            -0.04,
            0,
            "{} {}".format(title, dim2),
            va="bottom",
            transform=ax.transAxes,
            rotation=90,
        )
