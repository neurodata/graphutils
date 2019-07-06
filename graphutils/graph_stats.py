#%%
"""graph_stats : functionality for computing statistics on ndmg directories.
"""
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from graspy.utils import pass_to_ranks as PTR

from .graph_io import NdmgGraphs
from .utils import add_doc, discr_stat


class NdmgDiscrim(NdmgGraphs):
    """Compute discriminability from a ndmg directory.

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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = self._X()
        self.Y = self.subjects

    def __repr__(self):
        return f"NdmgDiscrim obj at {str(self.directory)}"

    def _X(self):
        """
        this will be a single matrix,
        created by vectorizing each array in `self.graphs`,
        and then appending that array as a row to X.

        Parameters
        ----------
        graphs : None or np.ndarray
            if None, graphs will be `self.graphs`.
            if not None, 
        PTR : bool, default False
            if True, call pass_to_ranks on X.

        Returns
        -------
        X : np.ndarray, shape (n, v*v), 2D
            numpy array, created by vectorizing each adjacency matrix and stacking.
        """
        if self.graphs.ndim == 3:
            n, v1, v2 = np.shape(self.graphs)
            return np.reshape(self.graphs, (n, v1 * v2))
        elif len(self.files) == 1:
            warnings.warn("Only one graph in directory.")
            return self.graphs
        else:
            raise ValueError("Dimensionality of input must be 3.")

    def pass_to_ranks(self, on="all"):
        """
        pass-to-ranks method, to call on `self.graphs` or `self.X`.
        When called, modifies one or both of these properties,
        depending on parameters.

        Assigns ranks to all non-zero edges, settling ties using 
        the average. Ranks are then scaled by 
        :math:`\frac{rank(\text{non-zero edges})}{\text{total non-zero edges} + 1}`.

        Parameters
        ----------
        on : str, "all", "X", or "graphs"
            if all, call pass to ranks on `self.X` and `self.graphs`
            if X, call pass to ranks on `self.X`
            if graphs, call pass to ranks on `self.graphs`.
        """

        def PTR_functionality(graphs):
            non_zeros = graphs[graphs != 0]
            rank = rankdata(non_zeros)
            normalizer = rank.shape[0]
            rank = rank / (normalizer + 1)
            graphs[graphs != 0] = rank
            return graphs

        if on == "X":
            self.X = PTR_functionality(self.X)
        elif on == "graphs":
            self.graphs = PTR_functionality(self.X)
        elif on == "all":
            self.X = PTR_functionality(self.X)
            self.graphs = PTR_functionality(self.graphs)
        else:
            raise ValueError("`on` must be all, X, or graphs.")

    def save_X_and_Y(self, output_directory="cwd"):
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
        if output_directory == "cwd":
            output_directory = Path.cwd()
        p = Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)

        X_name = f"{str(p)}/{self.name}_X.csv"
        Y_name = f"{str(p)}/{self.name}_Y.csv"

        np.savetxt(X_name, self.X, fmt="%f", delimiter=",")
        np.savetxt(Y_name, self.subjects, fmt="%s")

        name = namedtuple("name", ["X", "Y"])
        return name(X_name, Y_name)

    @add_doc(discr_stat.__doc__)
    def discriminability(self, **kwargs):
        """
        Attach discriminability functionality to the object.
        See `discr_stat` for full documentation.
        
        Returns
        -------
        stat : float
            Discriminability statistic.
        """
        return discr_stat(self.X, self.Y, **kwargs)

