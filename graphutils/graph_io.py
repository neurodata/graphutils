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

from graphutils.utils import CORRECT_SUFFIXES
from graphutils.utils import is_graph
from graphutils.utils import filter_graph_files
from graphutils.s3_utils import get_matching_s3_objects, get_credentials, s3_download_graph, parse_path


class NdmgDirectory:

    """
    Contains methods for use on a `ndmg` output directory.
    Central object of this package.

    Parameters
    ----------
    directory : str
        filepath or s3 url to the directory containing graph outputs.
        if s3, input should be `s3://bucket-prefix/`.
        if filepath, input should be the absolute path.
    delimiter : str
        The delimiter within each edgelist output file.

    Attributes
    ----------
    delimiter : str
        The delimiter used in edgelists.
    directory : Path
        Path object to the directory passed to NdmgDirectory.
        Takes either an s3 bucket or a local directory string as input.
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
        self.s3 = False
        if str(directory).startswith("s3:"):
            self.s3 = True
        self.delimiter = delimiter
        self.directory = Path(directory)
        self.files = self._files(directory)
        self.name = self.directory.name
        if not len(self.files):
            warnings.warn("warning : no edgelists found.")
        self.vertices = self._vertices()
        self.graphs = self._graphs()
        self.subjects = self._subjects()
                
    def __repr__(self):
        return f"NdmgDirectory obj at {str(self.directory)}"

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
            # parse bucket and path from self.directory
            # TODO: this breaks if the s3 directory structure changes
            bucket, prefix = parse_path(directory)
            dataset = prefix.split("/")[0]
            local_dir = Path.home() / Path(f".ndmg_s3_dir/{dataset}")
            self.directory = local_dir
            
            # if our local_dir already has graph files in it, just use that
            if local_dir.is_dir() and any(is_graph(x) for x in local_dir.iterdir()):
                print(f"Local directory {local_dir} found. Getting graphs from there instead of s3.")
                return list(filter_graph_files(local_dir.iterdir()))

            print(f"Downloading objects from s3 into {local_dir}...")

            # get generator of object names
            unfiltered_objs = get_matching_s3_objects(bucket, prefix=prefix, suffix=CORRECT_SUFFIXES)
            objs = filter_graph_files(unfiltered_objs) 
            
            # download each s3 graph and append local filepath to output
            for obj in objs:
                name = Path(obj).name
                local = str(local_dir / Path(name))
                print(f"Downloading {name} ...")
                s3_download_graph(bucket, obj, local)
                output.append(local)
            
            # update self.directory
            self.directory = local_dir

        else:
            for dirname, _, files in os.walk(directory):
                file_ends = list(filter_graph_files(files))
                graphnames = [Path(dirname) / Path(graphname) for graphname in file_ends]
                if all(graphname.exists for graphname in graphnames):
                    output.extend(graphnames)
                
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
        if dst is None:
            dst = self.directory / "graph_outputs"
        p = Path(dst).resolve()
        p.mkdir(parents=True, exist_ok=True)
        for filename in self.files:
            shutil.copy(filename, p)


def url_to_ndmg_dir(urls):
    """
    take a list of urls or filepaths,
    get a dict of NdmgDirectory objects
    
    Parameters
    ----------
    urls : list
        list of urls or filepaths. 
        Each element should be of the same form as the input to a `NdmgDirectory` object.
    
    Returns
    -------
    dict
        dict of {dataset:NdmgDirectory} objects.
    
    Raises
    ------
    TypeError
        Raises error if input is not a list.
    """

    if not isinstance(urls, list):
        raise TypeError("urls must be a list of URLs.")
    
    return_value = {}
    for url in urls:
        val = NdmgDirectory(url)
        key = val.name
        return_value[key] = val
    
    return return_value
