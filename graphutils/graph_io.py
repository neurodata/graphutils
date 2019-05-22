"""graph_io: input/output utilities for graphs.
"""
#%%
from pathlib import Path
import os
import re
from typing import List

import networkx as nx
import numpy as np
from graspy.utils import pass_to_ranks
from graspy.utils import import_edgelist as PTR

# TODO : functionality to turn multiple graphs into a single X matrix.
# TODO : functionality for getting output vector
# TODO : functionalitiy for pulling in graphs from directories
# TODO : Class for returning stuff from filepaths
# TODO : Class for manipulating graphs once they're created

#%%
class NdmgDirectory:
    """Class that contains utility methods for use on a `ndmg` output directory.
       Contains derived properties, and useful methods.

    Attributes
    ----------
    directory :
        TODO
    files : 
        TODO
    graphs : 
        TODO
    X : 
        TODO
    names : 
        TODO
    
    """

    def __init__(self, directory):
        if not isinstance(directory, (str, Path)):
            # TODO : {type(directory)} isn't returning output in `message`
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        else:
            self.dir = Path(directory)

    def __repr__(self):
        return str(self.dir)

    @property
    def files(self):
        """
        From a directory containing edgelist files, 
        return a list of edgelist files, 
        sorted alphabetically.
        
        Parameters
        ----------
        path : directory of edgelist files.
        
        Returns
        -------
        Sorted list of Paths to files in `path`.
        """
        # TODO : check if all files are edgelists with the correct stuff in them,
        #        and if they are not,
        #        raise an exception
        correct_suffixes = [".ssv", ".csv"]
        return sorted(
            [
                filepath
                for filepath in self.dir.iterdir()
                if filepath.suffix in correct_suffixes
            ]
        )

    @property
    def graphs(self):
        """
        List of numpy arrays,
        accounting for isolate nodes by unioning the vertices of all component edgelists,
        sorted in the same order as `self.files`.
        """
        # TODO : test to make sure sort order is the same as `self.files`
        return import_edgelist(self.files)

    @property
    def X(self):
        # TODO : this will be a single matrix,
        #        created by vectorizing each array in `self.graphs`,
        #        and then appending that array as a row to X.
        # TODO : test to make sure order of rows matches order of `self.files`.
        # TODO : maybe rebuild this functionality
        return np.stack(list(map(np.ravel, self.graphs)), axis=0)

    @property
    def Y(self):
        # TODO : this will be a list of subject ids,
        #        sorted the same way as `self.files`.
        pass

    def _pass_to_ranks(self, graph, method="simple-nonzero"):
        # Just to have this as a utility method
        return PTR(graph, method)

    def save_X_and_Y(self, output_directory):
        # TODO : this method will save `X` and `Y` as csvs into `output_directory`.
        pass


f = NdmgDirectory(
    "/Users/alex/Dropbox/NeuroData/ndmg-paper/data/graphs/native_graphs_NKI"
)
# f = NdmgDirectory(5)
np.shape(f.X)
#%%

# ---------- Depracated code below ---------- #


def return_sorted_graph(edgelist: str, n_nodes: int, delimiter=" ", nodetype=int):
    """From a graph file with n_nodes, return a sorted adjacency matrix.

    Parameters
    ----------
    edgelist : str or Path
        filepath to edgelist file.
    n_nodes : int
        number of nodes the numpy array should have.

    Returns
    -------
    out : np.ndarray
        numpy array with shape n_nodes x n_nodes. Adjacency matrix.
    """
    # TODO : Should be good. Needs tests.
    graph = nx.read_weighted_edgelist(edgelist, nodetype=nodetype, delimiter=delimiter)
    vertices = np.arange(1, n_nodes + 1)
    out = nx.to_numpy_array(graph, nodelist=vertices, dtype=np.float)
    return out


def path_files_list(path: str):
    """
    From a directory containing edgelist files, 
    return a list of adjacency matrices, 
    sorted alphabetically.
    
    Parameters
    ----------
    path : directory of edgelist files.
    
    Returns
    -------
    Sorted list of paths to files in `path`.
    """
    # TODO : try/except to make sure `path` is a directory
    path = Path(path).absolute()
    if path.is_dir():
        return sorted([str(fl) for fl in path.iterdir()])
    else:
        raise NotADirectoryError("`path` must be a directory.")


def get_X(arrays: List, PTR=True):
    """
    Ravel every array in arrays, 
    return single array with rows corresponding to each raveled array.

    Parameters
    ----------
    arrays : List
        list of arrays, each the same size.
    
    PTR : bool
        if True, call pass_to_ranks.


    Returns
    -------
    X : Concatenated X matrix.
    """
    # TODO: check to make sure all of the arrays are the same length
    # TODO: make sure nodes aren't being sorted differently for each array

    # pass to ranks first
    if PTR:
        try:
            arrays = [pass_to_ranks(array) for array in arrays]
        except ValueError:
            # Arrays shouldn't have to be square, and pass_to_ranks only works if square.
            print("Pass to Ranks not done -- nonsquare matrix")
            pass

    # stack 'em up
    return np.stack(list(map(np.ravel, arrays)), axis=0)


def get_Y(list_of_ndmg_paths):
    """
    From a list of paths to graph files, return a list of the subjects those graphs belong to.

    Parameters
    ----------
    list_of_ndmg_paths : list of paths to individual graphs.

    Returns
    -------
    Y : 1d array, shape (n_samples), type(str)
    """
    # TODO : get target vector from ndmg graphs directory
    # TODO : turn this into general form.
    subrgx = re.compile(r"(sub-)([0-9]*)(_)")

    # used to be a list comprehension, but this is more clear
    subjects_list = []
    for path in list_of_ndmg_paths:
        # to make sure there aren't any random files
        if Path(path).suffix == ".ssv":
            subject_re = re.search(subrgx, path)
            subject = subject_re.group(2)
            subjects_list.append(subject)

    return np.array(subjects_list)
