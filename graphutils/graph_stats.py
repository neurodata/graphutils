#%%
"""graph_stats : functionality for computing statistics on ndmg directories.
"""
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from graspy.utils import pass_to_ranks

from .graph_io import NdmgGraphs
from .utils import replace_doc, discr_stat


class NdmgStats(NdmgGraphs):
    """Compute statistics from a ndmg directory.

    Parameters
    ----------
    X : np.ndarray, shape (n, v*v), 2D
        numpy array, created by vectorizing each adjacency matrix and stacking.

    Methods
    -------
    pass_to_ranks : returns None 
        change state of object.
        calls pass to ranks on `self.graphs`, `self.X`, or both.
    save_X_and_Y : returns None
        Saves `self.X` and `self.Y` into a directory.
    discriminability : return float
        discriminability statistic for this dataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = self._X()
        self.Y = self.subjects

    def __repr__(self):
        return f"NdmgStats : {str(self.directory)}"

    def _X(self, graphs=None):
        """
        this will be a single matrix,
        created by vectorizing each array in `self.graphs`,
        and then appending that array as a row to X.

        Parameters
        ----------
        graphs : None or np.ndarray
            if None, graphs will be `self.graphs`.

        Returns
        -------
        X : np.ndarray, shape (n, v*v), 2D
            numpy array, created by vectorizing each adjacency matrix and stacking.
        """
        if graphs is None:
            graphs = self.graphs
        if graphs.ndim == 3:
            n, v1, v2 = np.shape(graphs)
            return np.reshape(graphs, (n, v1 * v2))
        elif len(self.files) == 1:
            warnings.warn("Only one graph in directory.")
            return graphs
        else:
            raise ValueError("Dimensionality of input must be 3.")

    def save_X_and_Y(self, output_directory="cwd", output_name=""):
        """
        Save `self.X` and `self.subjects` into an output directory.

        Parameters
        ----------
        output_directory : str, default current working directory
            Directory in which to save the output.

        Returns
        -------
        namedtuple with str
            namedtuple of `name.X, name.Y`. Paths to X and Y.
        """
        if not output_name:
            output_name = self.name

        if output_directory == "cwd":
            output_directory = Path.cwd()
        p = Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)

        X_name = f"{str(p)}/{output_name}_X.csv"
        Y_name = f"{str(p)}/{output_name}_Y.csv"

        np.savetxt(X_name, self.X, fmt="%f", delimiter=",")
        np.savetxt(Y_name, self.subjects, fmt="%s")

        name = namedtuple("name", ["X", "Y"])
        return name(X_name, Y_name)

    # @replace_doc(pass_to_ranks.__doc__)
    # def pass_to_ranks(self, graph):
    #     """
    #     pass-to-ranks method, generally to call on `self.graphs` or `self.X`.

    #     Assigns ranks to all non-zero edges, settling ties using 
    #     the average. Ranks are then scaled by 
    #     :math:`\frac{rank(\text{non-zero edges})}{\text{total non-zero edges} + 1}`.

    #     Parameters
    #     ----------
    #     graphs : np.ndarray
    #         if X, call pass to ranks on `self.X`
    #         if graphs, call pass to ranks on `self.graphs`.
    #     """

    #     graphs = np.copy(graph)
    #     if not isinstance(graph, np.ndarray):
    #         raise ValueError("input to pass_to_ranks must be a numpy array.")

    #     # TODO : broken. Needs to be called on an nxn array.
    #     # if input is a single graph, just call PTR.
    #     # if input is a 3D array, call PTR along subject columns.
    #     # if input is an nx(d*d) array, reshape each row, call PTR on it, then un-reshape, and return the PTRd nx(d*d) array.
    #     return pass_to_ranks(graph)

    @replace_doc(discr_stat.__doc__)
    def discriminability(self, PTR=True, **kwargs):
        """
        Attach discriminability functionality to the object.
        See `discr_stat` for full documentation.
        
        Returns
        -------
        stat : float
            Discriminability statistic.
        """
        if PTR:
            graphs = np.copy(self.graphs)
            graphs = np.array([pass_to_ranks(graph) for graph in graphs])
            X = self._X(graphs)
            return discr_stat(X, self.Y, **kwargs)

        return discr_stat(self.X, self.Y, **kwargs)


def url_to_ndmg_dir(urls):
    """
    take a list of urls or filepaths,
    get a dict of NdmgGraphs objects
    
    Parameters
    ----------
    urls : list
        list of urls or filepaths. 
        Each element should be of the same form as the input to a `NdmgGraphs` object.
    
    Returns
    -------
    dict
        dict of {dataset:NdmgGraphs} objects.
    
    Raises
    ------
    TypeError
        Raises error if input is not a list.
    """

    # checks for type
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list):
        raise TypeError("urls must be a list of URLs.")

    # appends each object
    return_value = {}
    for url in urls:
        try:
            val = NdmgStats(url)
            key = val.name
            return_value[key] = val
        except ValueError:
            warnings.warn(f"Graphs for {url} not found. Skipping ...")
            continue

    return return_value
