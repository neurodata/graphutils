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
        name of dataset.
    to_directory : func
        Send all graph files to a directory of your choosing
    """

    def __init__(self, directory, atlas="", suffix="csv", delimiter=" "):
        if not isinstance(directory, (str, Path)):
            message = f"Directory must be type str or Path. Instead, it is type {type(directory)}."
            raise TypeError(message)
        self.s3 = str(directory).startswith("s3:")
        self.directory = directory
        self.delimiter = delimiter
        self.atlas = atlas
        self.suffix = suffix
        self.files = self._files(directory)
        self.name = self._get_name()
        if not len(self.files):
            raise ValueError(f"No graphs found in {str(self.directory)}.")

    def __repr__(self):
        return f"NdmgDirectory : {str(self.directory)}"

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
            self.directory = Path(self.directory)
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
        local_dir = Path.home() / Path(f".ndmg_s3_dir/{prefix}")
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
            )

        # If has_graphs just got toggled, return all the graphs.
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

        # return
        if not output:
            raise ValueError("No graphs found in the directory given.")
        return output

    def _get_name(self):
        """
        return directory beneath ".ndmg_s3_dir".
        
        Returns
        -------
        str
            name of dataset.
        """
        if not self.s3:
            return self.directory.name

        parts = Path(self.directory).parts
        dataset_index = parts.index(".ndmg_s3_dir") + 1
        return parts[dataset_index]

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
        subject IDs, sorted array of all subject IDs in `dir`.
        subjects[0] corresponds to files[0].
    sessions : np.ndarray, shape n, 1D
        session IDs, sorted array of all sessions.
        sessions[0] corresponds to files[0].

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.nx_graphs = self._nx_graphs()
        self.vertices = self._vertices()
        self.sort_nx_graphs()  # get vertices, sort nx graphs
        self.graphs = self._graphs()
        self.subjects = self._parse()[0]
        self.sessions = self._parse()[1]

    def __repr__(self):
        return f"NdmgGraphs : {str(self.directory)}"

    def _nx_graphs(self):
        """
        List of networkx graph objects. Hidden property, mainly for use to calculate vertices.

        Returns
        -------
        nx_graphs : List[nx.Graph]
            List of networkX graphs corresponding to subjects.
        """
        files_ = [str(name) for name in self.files]
        nx_graphs = [
            nx.read_weighted_edgelist(f, nodetype=int, delimiter=self.delimiter)
            for f in files_
        ]
        return nx_graphs

    def _vertices(self):
        """
        Calculate the unioned number of nodes across all graph files.
        
        Returns
        -------
        np.array
            Sorted array of unioned nodes.
        """
        return np.sort(reduce(np.union1d, [G.nodes for G in self.nx_graphs]))

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

    def _parse(self):
        """
        Get subject IDs
        
        Returns
        -------
        out : np.ndarray 
            Array of strings. Each element is a subject ID.
        """
        pattern = r"(?<=sub-|ses-)(\w*)(?=_ses|_dwi)"
        subjects = [re.findall(pattern, str(edgelist))[0] for edgelist in self.files]
        sessions = [re.findall(pattern, str(edgelist))[1] for edgelist in self.files]
        return np.array(subjects), np.array(sessions)

    def sort_nx_graphs(self):
        """
        Ensure that all networkx graphs have the same number of nodes.

        Returns
        -------
        None
        """
        for graph in self.nx_graphs:
            graph.add_nodes_from(self.vertices)

