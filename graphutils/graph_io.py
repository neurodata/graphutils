"""graph_io: input/output utilities for graphs.
"""
#%%
from pathlib import Path
import os
import re
from typing import List

import networkx as nx
import numpy as np
from scipy.stats import rankdata
from graspy.utils import pass_to_ranks as PTR
from graspy.utils import import_edgelist

#%%
class NdmgDirectory:
    """Class that contains utility methods for use on a `ndmg` output directory.
       Contains derived properties, and useful methods.

    Attributes
    ----------
    dir : Path
        Path object to the directory containing edgelists.
    files : list, sorted
        List of path objects corresponding to each edgelist.
    graphs : np.ndarray, shape (n, v, v), 3D
        Volumetric numpy array, n vxv adjacency matrices corresponding to each edgelist.
        graphs[0, :, :] corresponds to files[0].
    X : np.ndarray, shape (n, v*v), 2D
        numpy array, created by vectorizing each adjacency matrix and stacking.
    Y : np.ndarray, shape n, 1D
        subject IDs, sorted set of all subject IDs in `dir`.
        Y[0] corresponds to files[0].

    Methods
    -------
    _X : returns np.ndarray, shape (n, v*v), 2D
        helper function to create `self.X.`
        Allows `self.X` to remain a property,
        but gives functionality for specifying pass-to-ranks on a particular set of graphs.
    _pass_to_ranks : returns np.ndarray
        returns an adjacency matrix with pass-to-ranks called on it.
    save_X_and_Y : returns None
        Saves `self.X` and `self.Y` into a directory.
    """

    # TODO : tests to make sure that the order of files, graphs, X, and Y all correspond to each other.

    def __init__(self, directory):
        if not isinstance(directory, (str, Path)):
            # TODO : {type(directory)} isn't returning output in `message`
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        else:
            self.dir = Path(directory)
            self.name = self.dir.name
            self.X = self.X()

    def __repr__(self):
        return str(self.dir)

    @property
    def files(self):
        """
        From a directory containing edgelist files, 
        return a list of edgelist files, 
        sorted alphabetically.

        This property is ground truth for how the n's should be sorted.
        
        Parameters
        ----------
        path : directory of edgelist files.
        
        Returns
        -------
        output : list, sorted
            Sorted list of Paths to files in `path`.
        """
        # TODO : check if all files are edgelists with the correct stuff in them,
        #        and if they are not,
        #        raise an exception
        correct_suffixes = [".ssv", ".csv"]
        output = sorted(
            [
                filepath
                for filepath in self.dir.iterdir()
                if filepath.suffix in correct_suffixes
            ]
        )
        return output

    @property
    def graphs(self):
        """
        volumetric numpy array, shape (n, v, v),
        accounting for isolate nodes by unioning the vertices of all component edgelists,
        sorted in the same order as `self.files`.

        Returns
        -------
        graphs : np.ndarray, shape (n, v, v), 3D
            Volumetric numpy array, n vxv adjacency matrices corresponding to each edgelist.
            graphs[0, :, :] corresponds to files[0].D

        """
        # TODO : test to make sure sort order is the same as `self.files`
        list_of_arrays = import_edgelist(self.files)
        return np.array(list_of_arrays)

    @property
    def Y(self):
        """
        Get subject IDs
        
        Returns
        -------
        out : np.ndarray 
            Array of strings. Each element is a subject ID.
        """
        # TODO : test to make sure sort order is the same as `self.files`
        pattern = r"(?<=sub-)(\w*)(?=_ses)"
        names = [re.findall(pattern, str(edgelist))[0] for edgelist in self.files]
        return np.array(names)

    def _X(self, PTR=False):
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
        # TODO : test to make sure order of rows matches order of `self.files`.
        # TODO : test to make sure order within rows matches order within arrays of `self.files`.
        graphs = self.graphs.copy()
        if graphs.ndim == 3:
            if PTR:
                graphs = self._pass_to_ranks(graphs=graphs)
            n, v1, v2 = np.shape(graphs)
            return np.reshape(graphs, (n, v1 * v2))
        else:
            raise ValueError("Dimensionality of input must be 3.")

    def _pass_to_ranks(self, method="simple-nonzero", on="all"):
        """
        pass-to-ranks method, to call on `self.graphs` or `self.X`.
        When called, modifies one or both of these properties,
        depending on parameters.

        Parameters
        ----------
        method : str
            how to pass to ranks
        on : str, "all", "X", or "graphs"
            if all, call pass to ranks on `self.X` and `self.graphs`
            if X, call pass to ranks on `self.X`
            if graphs, call pass to ranks on `self.graphs`.
        """
        # TODO : make this functionality change state of self.X and self.graphs.
        def functionality(graphs):
            non_zeros = graphs[graphs != 0]
            rank = rankdata(non_zeros)
            normalizer = rank.shape[0]
            rank = rank / (normalizer + 1)
            graphs[graphs != 0] = rank
            return graphs

        if on == "X":
            graphs = self.X
        elif on == "graphs":
            graphs = self.graphs.copy()
        else:
            raise ValueError("`on` must be all, X, or graphs.")

        return functionality(graphs)

    def save_X_and_Y(self, output_directory="."):
        # TODO : test this method
        """
        Save `self.X` and `self.Y` into an output directory.

        Parameters

        Returns
        -------
        X : np.ndarray, shape (n, v*v), 2D
            numpy array, created by vectorizing each adjacency matrix and stacking.
        """
        if output_directory == ".":
            output_directory = os.getcwd()
        np.savetxt(
            f"{output_directory}/{self.name}_X.csv", self.X, fmt="%f", delimiter=","
        )
        np.savetxt(f"{output_directory}/{self.name}_Y.csv", self.Y, fmt="%s")


# TODO : test this
