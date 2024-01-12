import pandas as pd
import numpy as np
import libplot
import collections
import scipy.stats as stats
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import phenograph
from sklearn.cluster import KMeans

# import randomly
import os
import umap
from libsparse import SparseDataFrame
from pydiffmap import diffusion_map
import lib10x

DMAP_PARAMS = {"n_jobs": -1, "algorithm": "ball_tree"}


def load_monocle_state_clusters(file="monocle_ptime.txt"):
    df = pd.read_csv(file, sep="\t", header=0)

    ret = df[["Barcode", "State"]]
    ret = ret.rename(columns={"State": "Cluster"})
    ret = ret.set_index("Barcode")

    return ret


def sort_clusters_by_exp(df, d, cuttree, colors=None):
    """
    Resort clusters so that expression moves in a gradient left to right
    high to low.

    Parameters
    ----------
    df : DataFrame
        Data to be reordered
    d : Dendrogram
        Dendrogram
    cuttree : cuttree object
        Cut tree object

    Returns
    -------
    numpy.ndarray
        Array of cells now ordered

    """

    lidx = np.array(d["leaves"])

    if colors is None:
        colors = get_colors()

    max_c = np.max(cuttree)

    clust_ord_map = collections.defaultdict(lambda: collections.defaultdict(list))
    within_clust_ord_map = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )

    for c in range(1, max_c + 1):
        cidx = np.where(cuttree == c)[0]

        # data related to cluster
        dc = df.iloc[cuttree == c, :].values

        # Find the max value and its col index from the mean expression
        # profile of the cluster
        m = dc.mean(axis=0)
        cx = m.max()
        cix = m.argmax()

        # if max value occurs on right side, invert x so that we sort from
        # high expression in the middle to high expression at the end to
        # maintain the gradient
        if cix < 50:
            cx = -cx

        # sort by position and then expression level
        clust_ord_map[cix][cx].append(c)

        # order within cluster

        xs = dc.max(axis=1)
        ixs = dc.argmax(axis=1)

        for i in range(0, xs.size):
            x = xs[i]
            ix = ixs[i]

            if ix < 50:
                x = -x

            within_clust_ord_map[c][ix][x].append(cidx[i])

    # New order of clusters
    ordered_cluster_ids = []

    for ix in sorted(clust_ord_map):
        # sort within block from highest to lowest
        for x in sorted(clust_ord_map[ix]):
            ordered_cluster_ids.extend(clust_ord_map[ix][x])

    idx = []

    for c in ordered_cluster_ids:
        # For each cluster, look for the indices for that cluster in cuttree
        # so we know which samples are in the cluster. Next, look up those
        # indices in the sorted leaves so that we can preserve within cluster
        # ordering
        # idx.extend(lidx[np.isin(lidx, np.where(cuttree == c)[0])])

        wcom = within_clust_ord_map[c]

        for ix in sorted(wcom):
            # sort within block from highest to lowest
            for x in sorted(wcom[ix]):
                idx.extend(wcom[ix][x])

    idx = np.array(idx)

    labels_idx = []

    ordered_clusters = cuttree[idx]

    for c in ordered_cluster_ids:
        cidx = np.where(ordered_clusters == c)[0]
        labels_idx.append(cidx[cidx.size // 2])

    labels = ["Cluster {}".format(c) for c in ordered_cluster_ids]
    label_colors = [colors[c - 1] for c in ordered_cluster_ids]
    row_colors = [colors[c - 1] for c in ordered_clusters]

    return idx, labels_idx, labels, label_colors, row_colors


def pseudo_trace(
    df, clusters, name, colors=None, cluster_order=None, labels=None, sort=True
):
    """
    Create histogram like time plots from monocle for each cluster
    """

    if colors is None:
        colors = lib10x.get_colors()

    fig, ax = libplot.new_fig(w=10, h=8, direction="out")
    # fig = libplot.new_base_fig(w=12, h=8)

    max_x = np.ceil(df["Pseudotime"].max())

    x = np.linspace(0, max_x, 500)

    y = 0

    if cluster_order is None:
        cluster_order = np.array(sorted(set(clusters["Cluster"])))

    if labels is None:
        labels = ["Cluster {}".format(c) for c in cluster_order]

    label_colors = []
    densities = []
    for c in cluster_order:
        d = df["Pseudotime"][df["Cluster"] == c].values

        density = stats.gaussian_kde(d)

        densities.append(density(x))  # np.clip(density(x), 0, 1)

        if isinstance(colors, dict):
            color = colors[c]
        else:
            # assume array
            color = colors[c - 1]

        label_colors.append(color)

        # axc = libplot.new_ax(fig, root_ax=ax, direction='out')

        # libplot.plot(x, p, c=colors[c - 1], label='Cluster {}'.format(c), ax=axc, alpha=0.5)

    cluster_order = np.array(cluster_order)
    densities = np.array(densities)
    label_colors = np.array(label_colors)
    labels = np.array(labels)

    if sort:
        peaks = [np.argmax(density) for density in densities]

        print(peaks)

        idx = np.argsort(peaks)

        print(idx)

        cluster_order = cluster_order[idx]
        densities = densities[idx]
        label_colors = label_colors[idx]
        labels = labels[idx]

    y = len(cluster_order) - 1

    for i in range(0, cluster_order.size):
        c = cluster_order[i]
        p = y + densities[i]
        color = label_colors[i]

        ax.fill_between(x, p, y, edgecolor="none", facecolor=color, alpha=0.5)

        # segments = [((x[i], p[i]), (x[i + 1], p[i + 1])) for i in range(0, x.size - 1)]

        # cmap = LinearSegmentedColormap.from_list('c{}'.format(c), [colors[c - 1], libplot.get_tint(colors[c - 1], 0.5)])

        # lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(x.min(), x.max()), linewidth=3)
        # Used by cmap to determine color combined with the norm function
        # lc.set_array(x)
        # axc.add_collection(lc)

        # break

        y -= 1

    # invert because we plot on negative y so that plots are rendered
    # top to bottom
    label_colors = label_colors[::-1]

    # libplot.format_legend(ax)
    ax.set_xlabel("Pseudotime")
    # ax.set_ylabel('Cluster')
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim([0, max_x])
    ax.set_xticks([0, max_x])
    ax.set_xticklabels([0, 1])
    ax.set_ylim([-0.5, cluster_order.size])
    ax2 = ax.twinx()
    ax2.set_ylim([-0.5, cluster_order.size])
    ax2.set_yticks([i + 0.5 for i in range(0, cluster_order.size)])
    ax2.set_yticklabels([label for label in labels[::-1]])
    ax2.tick_params(axis="y", colors="white")
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Color the labels
    libplot.color_labels(
        ax2, label_colors, label_colors.size
    )  # , color_indices=(cluster_order[::-1]))

    libplot.savefig(fig, "pseudotime_trace_{}.pdf".format(name))


def remove_bad_cells(data, genes, gene_index=None):
    """
    Find cells that expression certain genes and flag them for removal

    Parameters
    ----------
    data : DataFrame
        count matrix
    genes : list
        genes to check.
    gene_index : optional, list
        list of genes for each row in data.

    """

    if gene_index is None:
        _, gene_index = data.index.str.split(";").str

    idx = np.where(gene_index.isin(genes))[0]

    gp = collections.defaultdict(set)

    for g in genes:
        idx = np.where(gene_index == g)[0]

        exp = data.iloc[idx, :].sum(axis=0)

        idx2 = np.where(exp > 0)[0]

        for i in idx2:
            gp[i].add(g)

    removed_cols = list(sorted(gp))

    # remove the bad cells
    ret = data.iloc[:, np.setxor1d(range(0, data.shape[1]), removed_cols)]

    # log what we remove and why

    genes = []

    for i in removed_cols:
        genes.append(";".join(sorted(gp[i])))

    df = pd.DataFrame({"Barcode": data.columns[removed_cols].values, "Genes": genes})
    df = df.set_index("Barcode")
    df.to_csv("removed_cells.txt", sep="\t", header=True, index=True)

    return ret


# def randomly_filter(counts, name, fdr=0.01):
#     """
#     Use the Randomly package to remove genes that just look like noise

#     Parameters
#     ----------
#     counts: df: Pandas dataframe, shape (n_genes, n_cells)
#         Counts matrix
#     name: str,
#         Name to call output files
#     fdr: float, optional
#         fdr threshold for filtering genes
#     """

#     # Need to transpose for randomly
#     cr = counts.T

#     model = randomly.Rm()
#     model.preprocess(
#         cr, min_tp=0, min_genes_per_cell=0, min_cells_per_gene=0, refined=True
#     )
#     model.refining(min_trans_per_gene=7)
#     model.fit()
#     model.plot_mp(path="{}_mp.pdf".format(name))
#     model.plot_statistics(path="{}_mp_stats.pdf".format(name))

#     # get the genes of interest
#     cr2 = model.return_cleaned(fdr=fdr).T

#     # filter counts to include just the genes of interest
#     cr3 = counts.loc[counts.index.isin(cr2.index), :]
#     print("Randomly filtered size", cr3.shape)

#     cr3.to_csv(
#         "{}_umi.txt.gz".format(name),
#         sep="\t",
#         header=True,
#         index=True,
#         compression="gzip",
#     )

#     # Also create TPM

#     s = cr3.sum(axis=0)

#     s = 1000000 / s

#     cr3 = cr2 * s
#     cr3 = (cr3 + 1).apply(np.log2)
#     cr3.round(3).to_csv(
#         "{}_tpm_log2.txt.gz".format(name),
#         sep="\t",
#         header=True,
#         index=True,
#         compression="gzip",
#     )


def get_dim_file(dir, name, mode="tsne"):
    print(dir, name, mode)
    if dir.endswith("/"):
        dir = dir[:-1]

    return "{}/{}_data_{}.txt".format(dir, mode, name)


def get_umap_file(dir, name):
    return get_dim_file(dir, name, mode="umap")


def get_centroid_file(dir, name, method="tsne"):
    return get_dim_file(dir, name, mode="{}_cluster_centroid".format(method))


def get_dmap_file(dir, name, tpmmode=True, logmode=True):
    return get_dim_file(dir, name, mode="dmap")


def read_dim_reduct(file):
    print("Reading clusters from {}...".format(file))

    return pd.read_csv(file, sep="\t", header=0, index_col=0)


def write_clusters(headers, labels, name, tpmmode=True, logmode=True, dir="."):
    file = get_cluster_file(dir, name, tpmmode=tpmmode, logmode=logmode)

    print("Writing clusters to {}...".format(file))

    if type(headers) is pd.core.frame.DataFrame:
        headers = headers.iloc[:, 0].tolist()

    # one based is a convience for users so that they don't have to use
    # zero based ids
    df = pd.DataFrame(
        {"Barcode": headers, "Cluster": labels, "cluster_one_based": (labels + 1)}
    )

    df = df[["Barcode", "Cluster"]]
    df = df.set_index("Barcode")

    df.to_csv(file, sep="\t", header=True, index=True)


def write_kmeans_clusters(name, clusters, headers, labels, dir="."):
    file = get_kmeans_file(dir, name, clusters)

    print("Writing k-means clusters to {}...".format(file))

    if type(headers) is pd.core.frame.DataFrame:
        headers = headers.iloc[:, 0].tolist()

    # one based is a convience for users so that they don't have to use
    # zero based ids
    df = pd.DataFrame({"Barcode": headers, "Cluster": labels})

    df = df[["Barcode", "Cluster"]]
    df = df.set_index("Barcode")

    df.to_csv(file, sep="\t", header=True, index=True)


def read_clusters(file):
    print("Reading clusters from {}...".format(file))

    data = pd.read_csv(file, sep="\t", header=0, index_col=0)

    cluster_map = collections.defaultdict(int)

    for i in range(0, data.shape[0]):
        cluster_map[data.index[i]] = data["Cluster"][i]

    return cluster_map, data


def get_kmeans_file(dir, name, clusters):
    if dir.endswith("/"):
        dir = dir[:-1]

    file = "{}/clusters_kmeans_{}_{}.txt".format(dir, clusters, name)

    return file


def get_cluster_file(dir, name, tpmmode=True, logmode=True):
    if dir.endswith("/"):
        dir = dir[:-1]

    file = "{}/clusters_{}.txt".format(dir, name)

    # if tpmmode:
    #  file += '_tpm'

    # if logmode:
    #  file += '_log2'

    # file += '.txt'

    return file


def load_phenograph_clusters(pca, name, cache=True, neighbors=20, dir="."):
    """
    Given a pca matrix, cluster on it

    Parameters
    ----------
    pca : array, shape (n_samples, n_pca)

    name : str
        Name of run.

    cache : bool, optional
        Create a file of the clusters for reloading. Default is True.

    Returns
    -------
    tsne :  array, shape (n_samples, 2)
        The tsne coordinates for each sample
    """

    file = get_cluster_file(dir, name)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it with...".format(file))

        k = min(pca.shape[0] - 2, neighbors)

        # Find the interesting clusters
        labels, graph, Q = phenograph.cluster(pca, k=k)

        if min(labels) == -1:
            new_label = 100
            labels[np.where(labels == -1)] = new_label

        labels += 1

        write_clusters(pca.index.tolist(), labels, name)

    cluster_map, data = read_clusters(file)

    labels = data
    # return cluster_map, labels
    return labels


def load_kmeans_clusters(pca, name, clusters=10, cache=True, dir="."):
    """
    Given a pca matrix, cluster on it

    Parameters
    ----------
    pca : array, shape (n_samples, n_pca)

    name : str
        Name of run.

    cache : bool, optional
        Create a file of the clusters for reloading. Default is True.

    Returns
    -------
    tsne :  array, shape (n_samples, 2)
        The tsne coordinates for each sample
    """

    file = get_kmeans_file(dir, name, clusters)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it with...".format(file))

        labels = KMeans(n_clusters=clusters).fit_predict(pca)

        if min(labels) == -1:
            new_label = 100
            labels[np.where(labels == -1)] = new_label

        labels += 1

        write_kmeans_clusters(name, clusters, pca.index.tolist(), labels, dir=dir)

    cluster_map, data = read_clusters(file)

    labels = data

    return labels


def get_pca_file(dir, name, tpmmode=True, logmode=True):
    if dir.endswith("/"):
        dir = dir[:-1]

    return "{}/pca_data_{}.txt".format(dir, name)


def get_pca_var_file(name, tpmmode=True, logmode=True):
    return "pca_var_{}.txt".format(name)


def read_pca(file, dir="."):
    if not os.path.isfile(file):
        file = get_pca_file(dir, file)

    print("Reading pca from {}...".format(file))

    return pd.read_csv(file, sep="\t", header=0, index_col=0)


def load_pca(
    data,
    name,
    n=50,
    mode="random",
    tpmmode=True,
    logmode=True,
    exclude=[],
    cache=True,
    dir=".",
):
    file = get_pca_file(dir, name, tpmmode=tpmmode, logmode=logmode)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it with n={}...".format(file, n))

        p, pca = lib10x.pca(data, n=n, mode=mode, exclude=exclude)

        labels = ["PC-{}".format(x + 1) for x in range(0, pca.shape[1])]

        var_file = get_pca_var_file(name, tpmmode=tpmmode, logmode=logmode)
        df = pd.DataFrame(
            p.explained_variance_ratio_, index=labels, columns=["Variance"]
        )
        df.to_csv(var_file, sep="\t", header=True, index=True)

        df = pd.DataFrame(pca, index=data.columns, columns=labels)
        df.to_csv(file, sep="\t", header=True, index=True)

    return read_pca(file)


def load_umap(
    pca,
    name,
    tpmmode=True,
    logmode=True,
    exclude=[],
    cache=True,
    mode="original",
    spread=1,
    dir=".",
):
    """
    Run umap on a data set

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

    print("what", dir, name)
    file = get_umap_file(dir, name)

    print("umap file", file)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it...".format(file))

        # perplexity = 5, n_iter = 5000, learning = 10

        # The original parameters
        #

        if mode == "compressed":
            print("Compressed UMAP mode")
            u = umap.UMAP(
                n_neighbors=20,
                min_dist=0.2,
                metric="correlation",
                spread=spread,
                n_components=2,
            )
        else:
            print("Original UMAP mode")
            u = umap.UMAP(
                n_neighbors=10,
                min_dist=0.1,
                spread=spread,
                metric="correlation",
                n_components=2,
            )

        if isinstance(pca, SparseDataFrame):
            embedding = SparseDataFrame(
                u.fit_transform(pca.data), pca.index, pca.columns
            )
        else:
            embedding = u.fit_transform(pca)

        data = pd.DataFrame(index=pca.index)
        data.index.name = "Barcode"
        data["UMAP-1"] = embedding[:, 0]
        data["UMAP-2"] = embedding[:, 1]
        # data['UMAP-3'] = embedding[:, 2]

        data.to_csv(file, sep="\t", header=True)

    return read_dim_reduct(file)


def load_cluster_centroids(tsne, clusters, name, method="tsne", cache=True, dir="."):
    file = get_centroid_file(dir, name, method=method)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it...".format(file))

        # perplexity = 5, n_iter = 5000, learning = 10

        # The original parameters

        cluster_ids = list(sorted(np.unique(clusters["Cluster"])))

        d = np.zeros((len(cluster_ids), 2))

        mx = tsne.iloc[:, 0].min()
        my = tsne.iloc[:, 1].min()
        rx = tsne.iloc[:, 0].max() - mx
        ry = tsne.iloc[:, 1].max() - my

        print("aha")

        for i in range(0, len(cluster_ids)):
            c = cluster_ids[i]
            idx = np.where(clusters["Cluster"] == c)[0]

            xs = tsne.iloc[idx, 0].values
            ys = tsne.iloc[idx, 1].values

            print(i, c)

            if c == 12:
                print(c)
                print(tsne.iloc[idx, :])

            centroid = [xs.mean(), ys.mean()]

            # remove outliers
            nx = abs(xs - centroid[0])
            ny = abs(ys - centroid[1])

            nx /= nx.max()
            ny /= ny.max()

            idx1 = np.where(nx < 0.1)[0]
            idx2 = np.where(ny < 0.1)[0]

            idx = np.intersect1d(idx1, idx2)

            print(c, xs.size, idx.size)

            xs = xs[idx]
            ys = ys[idx]

            centroid = [xs.mean(), ys.mean()]

            d[i, 0] = (centroid[0] - mx) / rx
            d[i, 1] = (centroid[1] - my) / ry

        df = pd.DataFrame(d, columns=["x", "y"])
        df = df.set_index(np.array(cluster_ids))
        df.round(3).to_csv(file, sep="\t", header=True, index=True)

    return read_dim_reduct(file)


def load_dmap(pca, name, cache=True, dir="."):
    """
    Run diffusion map on a data set

    Parameters
    ----------
    pca : array, shape (n_samples, n_pca)
        PCA matrix.
    name: str
        name of pca results

    Returns
    -------
    tsne :  array, shape (n_samples, 2)
        The tsne coordinates for each sample
    """

    file = get_dmap_file(dir, name)

    # dt = d.T

    print("dmap file", file)

    if not os.path.isfile(file) or not cache:
        print("{} was not found, creating it...".format(file))

        dmap = diffusion_map.DiffusionMap.from_sklearn(
            n_evecs=4, k=64, epsilon="bgh", alpha=1.0, neighbor_params=DMAP_PARAMS
        )

        embedding = dmap.fit_transform(pca)

        data = pd.DataFrame(
            {
                "Barcode": pca.index,
                "DMAP-1": embedding[:, 0],
                "DMAP-2": embedding[:, 1],
                "DMAP-3": embedding[:, 2],
                "DMAP-4": embedding[:, 3],
            }
        )
        data = data[["Barcode", "DMAP-1", "DMAP-2", "DMAP-3", "DMAP-4"]]
        data = data.set_index("Barcode")

        data.to_csv(file, sep="\t", header=True)

    return read_dim_reduct(file)
