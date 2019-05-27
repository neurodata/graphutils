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


def foo():
    pass


class NdmgDirectory:
    """Class that contains utility methods for use on a `ndmg` output directory.
       Contains derived properties, and useful methods.

    Attributes
    ----------
    dir : Path
        Path object to the directory containing edgelists.
    name : str
        Base name of directory.
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
    _pass_to_ranks : returns None
        change state of object.
        calls pass to ranks on `self.graphs`, `self.X`, or both.
    save_X_and_Y : returns None
        Saves `self.X` and `self.Y` into a directory.
    """

    # TODO : tests to make sure that the order of files, graphs, X, and Y all correspond to each other.
    # TODO : automatically find delimiter
    # TODO : functionality for calculating discriminability

    def __init__(self, directory):
        if not isinstance(directory, (str, Path)):
            # TODO : {type(directory)} isn't returning output in `message`
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        else:
            self.dir = Path(directory)
            self.name = self.dir.name
            self.files = self._files()
            self.graphs = self._graphs()
            self.X = self._X()
            self.Y = self._Y()

    def __repr__(self):
        return f"NdmgDirectory object at {str(self.dir)}"

    def _files(self):
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
        correct_suffixes = {".ssv", ".csv"}
        output = sorted(
            [
                filepath
                for filepath in self.dir.iterdir()
                if filepath.suffix in correct_suffixes
            ]
        )
        return output

    def _graphs(self):
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

    def _Y(self):
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
        # TODO : test to make sure order of rows matches order of `self.files`.
        # TODO : test to make sure order within rows matches order within arrays of `self.files`.
        if self.graphs.ndim == 3:
            n, v1, v2 = np.shape(self.graphs)
            return np.reshape(self.graphs, (n, v1 * v2))
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
