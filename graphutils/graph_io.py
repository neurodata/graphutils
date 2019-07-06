#%%
"""graph_io: input/output utilities for graphs.
"""
from pathlib import Path
import shutil
import os
import re
from functools import reduce
import warnings

import networkx as nx
import numpy as np
from graspy.utils import import_edgelist

from graphutils.utils import is_graph
from graphutils.utils import filter_graph_files
from graphutils.s3_utils import (
    get_matching_s3_objects,
    get_credentials,
    s3_download_graph,
    parse_path,
)


class NdmgDirectory:
    """
    Contains methods for use on a `ndmg` output directory.
    Top-level object of this package.

    Parameters
    ----------
    directory : str
        filepath or s3 url to the directory containing graph outputs.
        if s3, input should be `s3://bucket-prefix/`.
        if filepath, input should be the absolute path.
    directory : Path
        Path object to the directory passed to NdmgGraphs.
        Takes either an s3 bucket or a local directory string as input.
    atlas : str
        atlas to get graph files of.
    delimiter : str
        delimiter in graph files.

    Attributes
    ----------
    files : list, sorted
        List of path objects corresponding to each edgelist.
    name : str
        Base name of directory.
    """

    def __init__(self, directory, atlas="", suffix="ssv", delimiter=" "):
        if not isinstance(directory, (str, Path)):
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        self.s3 = False
        self.directory = Path(directory)
        self.delimiter = delimiter
        if str(directory).startswith("s3:"):
            self.s3 = True
        self.atlas = atlas
        self.suffix = suffix
        self.files = self._files(directory)
        if not len(self.files):
            warnings.warn("warning : no edgelists found.")
        self.name = self.directory.parent.name

    def __repr__(self):
        return f"NdmgGraphs obj at {str(self.directory)}"

    def _files(self, directory):
        """
        From a directory or s3 bucket containing edgelist files, 
        return a list of edgelist files, 
        sorted.

        This property is ground truth for how the scans should be sorted.
        
        Parameters
        ----------
        path : directory of edgelist files or s3 bucket
        
        Returns
        -------
        output : list, sorted
            Sorted list of Paths to files in `path`.
        """
        output = []

        # grab files from s3 instead of locally
        if self.s3:
            output = self._get_s3(directory, atlas=self.atlas, suffix=self.suffix)

        else:
            for dirname, _, files in os.walk(directory):
                file_ends = list(
                    filter_graph_files(files, suffix=self.suffix, atlas=self.atlas)
                )
                graphnames = [
                    Path(dirname) / Path(graphname) for graphname in file_ends
                ]
                if all(graphname.exists for graphname in graphnames):
                    output.extend(graphnames)

        return sorted(output)

    def _get_s3(self, path, **kwargs):
        output = []
        # parse bucket and path from self.directory
        # TODO: this breaks if the s3 directory structure changes
        bucket, prefix = parse_path(path)
        dataset = prefix.split("/")[0]
        local_dir = Path.home() / Path(f".ndmg_s3_dir/{dataset}")
        if self.atlas:
            local_dir = local_dir / Path(self.atlas)
        else:
            local_dir = local_dir / Path("no_atlas")
        self.directory = local_dir

        # if our local_dir already has graph files in it, just use that
        is_dir = local_dir.is_dir()
        has_graphs = False
        if is_dir:
            has_graphs = filter_graph_files(
                local_dir.iterdir(), return_bool=True, **kwargs

        # Check if has_graphs just got toggled.
        if has_graphs:
            print(f"Local path {local_dir} found. Using that.")
            graphs = filter_graph_files(local_dir.iterdir(), **kwargs)
            return list(graphs)

        print(f"Downloading objects from s3 into {local_dir}...")

        # get generator of object names
        unfiltered_objs = get_matching_s3_objects(
            bucket, prefix=prefix, suffix=self.suffix
        )
        objs = filter_graph_files(unfiltered_objs, **kwargs)

        # download each s3 graph and append local filepath to output
        for obj in objs:
            name = Path(obj).name
            local = str(local_dir / Path(name))
            print(f"Downloading {name} ...")
            s3_download_graph(bucket, obj, local)
            output.append(local)

        # update self.directory
        self.directory = local_dir

        # return
        return output

    def to_directory(self, dst=None):
        """
        Send all `self.files`to `directory`.

        Parameters
        ----------
        directory : str or Path
            directory to send files to.
        """
        if dst is None:
            dst = self.directory / "graph_outputs"
        p = Path(dst).resolve()
        p.mkdir(parents=True, exist_ok=True)
        for filename in self.files:
            shutil.copy(filename, p)


class NdmgGraphs(NdmgDirectory):
    """
    NdmgDirectory which contains graph objects.

    Parameters
    ----------
    delimiter : str
        The delimiter used in edgelists    


    Attributes
    ----------
    delimiter : str
        The delimiter used in edgelists    
    vertices : np.ndarray
        sorted union of all nodes across edgelists.
    graphs : np.ndarray, shape (n, v, v), 3D
        Volumetric numpy array, n vxv adjacency matrices corresponding to each edgelist.
        graphs[0, :, :] corresponds to files[0].
    subjects : np.ndarray, shape n, 1D
        subject IDs, sorted set of all subject IDs in `dir`.
        Y[0] corresponds to files[0].
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.vertices = self._vertices()
        self.graphs = self._graphs()
        self.subjects = self._subjects()

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
        nx_graphs = self._nx_graphs()
        return np.sort(reduce(np.union1d, [G.nodes for G in nx_graphs]))

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
        val = NdmgGraphs(url)
        key = val.name
        return_value[key] = val

    return return_value

