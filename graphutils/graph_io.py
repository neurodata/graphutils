#%%
"""graph_io: input/output utilities for graphs.
"""
from pathlib import Path
import os
import re
from typing import List
from functools import reduce
from collections import namedtuple

import networkx as nx
import numpy as np
from scipy.stats import rankdata
from graspy.utils import pass_to_ranks as PTR
from graspy.utils import import_edgelist


class NdmgDirectory:
    """Class that contains utility methods for use on a `ndmg` output directory.
       Contains derived properties, and useful methods.

    Attributes
    ----------
    dir : Path
        Path object to the directory containing edgelists.
    name : str
        Base name of directory.
    delimiter : str
        delimiter in edgelist files.
    files : list, sorted
        List of path objects corresponding to each edgelist.
    vertices : np.ndarray
        sorted union of all nodes across edgelists.
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

    # TODO : functionality for calculating discriminability
    # TODO : add delimiter everywhere

    def __init__(self, directory, delimiter=" "):
        if not isinstance(directory, (str, Path)):
            # TODO : {type(directory)} isn't returning output in `message`
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        else:
            self.dir = Path(directory)
            self.name = self.dir.name
            self.delimiter = delimiter
            self.files = self._files()
            self.vertices = self._vertices()
            self.graphs = self._graphs()
            self.X = self._X()
            self.Y = self._Y()

        # works because of falsiness of 0
        if not len(self.files):
            print("Warning : There are no edgelists in this directory.")

        print("initiated")

    def __repr__(self):
        return f"NdmgDirectory obj at {str(self.dir)}"

    def _files(self):
        """
        From a directory containing edgelist files, 
        return a list of edgelist files, 
        sorted.

        This property is ground truth for how the scans should be sorted.
        
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

    @property
    def _nx_graphs(self):
        """
        List of networkx graph objects. Hidden property, mainly for use to calculate vertices.

        Returns
        -------
        nx_graphs : List[nx.Graph]
            List of networkX graphs corresponding to subjects.
        """
        nx_graphs = [
            nx.read_weighted_edgelist(f, nodetype=int, delimiter=self.delimiter)
            for f in self.files
        ]
        return nx_graphs

    def _vertices(self):
        return np.sort(reduce(np.union1d, [G.nodes for G in self._nx_graphs]))

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
        list_of_arrays = import_edgelist(self.files, delimiter=self.delimiter)
        return np.array(list_of_arrays)

    def _Y(self):
        """
        Get subject IDs
        
        Returns
        -------
        out : np.ndarray 
            Array of strings. Each element is a subject ID.
        """
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
        if self.graphs.ndim == 3:
            n, v1, v2 = np.shape(self.graphs)
            return np.reshape(self.graphs, (n, v1 * v2))
        else:
            raise ValueError("Dimensionality of input must be 3.")

    def _pass_to_ranks(self, on="all"):
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

        # TODO : this could likely be done more pythonically
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
        # TODO : test this method
        """
        Save `self.X` and `self.Y` into an output directory.

        Parameters
        ----------
        output_directory : str, default current working directory
            Directory in which to save the output.

        Returns
        -------
        namedtuple
            namedtuple of `name.X, name.Y`.
        """
        if output_directory == "cwd":
            output_directory = Path.cwd()
        p = Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)

        X_name = f"{str(p)}/{self.name}_X.csv"
        Y_name = f"{str(p)}/{self.name}_Y.csv"

        np.savetxt(X_name, self.X, fmt="%f", delimiter=",")
        np.savetxt(Y_name, self.Y, fmt="%s")

        name = namedtuple("name", ["X", "Y"])
        print("updated")
        return name(X_name, Y_name)
