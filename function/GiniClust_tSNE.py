import pandas
import argparse
import os
from matplotlib import pyplot
from sklearn import manifold
from function.GiniClust_parameters import GiniClust_Parameters


def giniClust_tSNE(cell_cell_distance: pandas.DataFrame, c_membership: pandas.Series, main_args: argparse.Namespace, expr_params: GiniClust_Parameters):
    n_clusters = c_membership.nunique()
    mycols = ["#D9D9D9"] #grey85
    assert n_clusters >= 2
    rainbow = [pyplot.cm.hsv(i / (n_clusters - 1)) for i in range(n_clusters - 1)]
    for c in rainbow:
        mycols.append(tuple(c))
    assert main_args.type in ["RNA-seq", "qPCR"]
    if main_args.type == "RNA-seq":
        seed = 10
        tsne = manifold.TSNE(n_components=2, metric="precomputed", perplexity=expr_params.perplexity,
                             max_iter=2000, random_state=seed, init="random", learning_rate="auto")
        embedding = tsne.fit_transform(cell_cell_distance.values)
        levels = list(pandas.Categorical(c_membership).categories)
        if "Singleton" in levels:
            levels = ["Singleton"] + [lev for lev in levels if lev != "Singleton"]
        color_map = {level: mycols[i] for i, level in enumerate(levels)}
        colors = c_membership.map(color_map)
        pyplot.figure(figsize=(8,8))
        pyplot.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=20, marker="o", edgecolor="none")
        pyplot.xlabel("Dimension_1")
        pyplot.ylabel("Dimension_2")
        pyplot.title("")
        for level, color in color_map.items():
            pyplot.scatter([],[],c=[color],label=level)
        pyplot.legend(loc="upper left", frameon=False, fontsize=12)
        pyplot.savefig(os.path.join(main_args.out, "figures", f"{expr_params.experiment_id}_tsne_plot.pdf"))
        pyplot.close()

        # save results
        tsne_coord2 = pandas.DataFrame(embedding, index=cell_cell_distance.index, columns=["dim1","dim2"])
        tsne_coord2.to_csv(os.path.join(main_args.out, f"{expr_params.experiment_id}_Rtnse_coord2.csv"), sep=",", index=True, header=True)

    elif main_args.type == "qPCR":
        seed = 7
        tsne = manifold.TSNE(n_components=2, metric="precomputed",perplexity=expr_params.perplexity,
                             max_iter=1000, random_state=seed, init="random", learning_rate="auto")
        embedding = tsne.fit_transform(cell_cell_distance.values)
        levels = list(pandas.Categorical(c_membership).categories)
        if "Singleton" in levels:
            levels = ["Singleton"] + [lev for lev in levels if lev != "Singleton"]
        color_map = {level: mycols[i] for i, level in enumerate(levels)}
        colors = c_membership.map(color_map)
        pyplot.figure(figsize=(8,8))
        pyplot.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=20, marker="o", edgecolor="none")
        pyplot.xlabel("tSNE_1")
        pyplot.ylabel("tSNE_2")
        for level, color in color_map.items():
            pyplot.scatter([],[],c=[color],label=level)
        pyplot.legend(loc="upper left", frameon=False, fontsize=12)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(main_args.out, "figures", f"{expr_params.experiment_id}_tsne_plot.pdf"))
        pyplot.close()
        tsne_coord2 = pandas.DataFrame(embedding, index=cell_cell_distance.index, columns=["dim1", "dim2"])
        tsne_coord2.to_csv(os.path.join(main_args.out, f"{expr_params.experiment_id}_Rtnse_coord2.csv"), sep=",", index=True, header=True)
    else:
        raise ValueError("Undefined type")