#%%
"""graph_io: input/output utilities for graphs.
"""
from pathlib import Path
import shutil
import os
import re
import glob
from functools import reduce, partial
import warnings

import networkx as nx
import numpy as np
from graspy.utils import import_edgelist


class NdmgDirectory:
    """Class that contains utility methods for use on a `ndmg` output directory.
       Contains derived properties, and useful methods.

    Attributes
    ----------

    delimiter : str
        The delimiter used in edgelists.
    directory : Path
        Path object to the directory passed to NdmgDirectory.
    name : str
        Base name of directory.
    files : list, sorted
        List of path objects corresponding to each edgelist.
    vertices : np.ndarray
        sorted union of all nodes across edgelists.
    graphs : np.ndarray, shape (n, v, v), 3D
        Volumetric numpy array, n vxv adjacency matrices corresponding to each edgelist.
        graphs[0, :, :] corresponds to files[0].
    subjects : np.ndarray, shape n, 1D
        subject IDs, sorted set of all subject IDs in `dir`.
        Y[0] corresponds to files[0].
    """

    def __init__(self, directory, delimiter=" "):
        if not isinstance(directory, (str, Path)):
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        else:
            self.delimiter = delimiter
            self.directory = Path(directory)
            self.name = self.directory.name
            self.files = self._files(directory)
            self.vertices = self._vertices()
            self.graphs = self._graphs()
            self.subjects = self._subjects()

        # works because of falsiness of 0
        if not len(self.files):
            print("Warning : There are no edgelists in this directory.")

    def __repr__(self):
        return f"NdmgDirectory obj at {str(self.directory)}"

    def _files(self, directory):
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
        output = []
        correct_suffixes = [".ssv", ".csv"]
        keywords = ["sub-", "ds_adj"]
        correct_suffixes = {".ssv", ".csv"}
        for dirname, _, files in os.walk(directory):
            for filename in files:
                right_suffix = Path(filename).suffix in correct_suffixes
                right_filename = all(i in filename for i in keywords)
                if right_suffix and right_filename:
                    output.append(Path(dirname) / Path(filename))
        return sorted(output)

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
        if not isinstance(list_of_arrays, list):
            list_of_arrays = [list_of_arrays]
        return np.atleast_3d(list_of_arrays)
    
    def _subjects(self):
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

    def to_directory(self, dst=None):
        """
        Send all `self.files`to `directory`.

        Parameters
        ----------
        directory : str or Path
            directory to send files to.
        """
        # TODO : test
        if dst is None:
            dst = self.directory
        p = Path(dst).resolve()
        p.mkdir(parents=True, exist_ok=True)
        for filename in self.files:
            shutil.copy(filename, p)