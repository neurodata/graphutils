import os
import shutil
from pathlib import Path
import re

import pytest
import numpy as np
import networkx as nx
from graphutils.graph_io import NdmgDirectory


class TestNdmgDirectory:
    def test_object_has_attributes(self, ND):
        assert all(
            hasattr(ND, attr)
            for attr in [
                "delimiter",
                "directory",
                "name",
                "files",
                "vertices",
                "graphs",
                "subjects",
            ]
        )

        assert ND.directory, "directory doesn't exist"
        assert ND.files, "no files found"
        assert isinstance(ND.subjects, np.ndarray), "subjects doesn't exist"
        assert isinstance(ND.graphs, np.ndarray), "graphs doesn't exist"
        assert isinstance(ND.vertices, np.ndarray), "graphs doesn't exist"

    def test_files_has_data(self, ND):
        # check if there are files
        assert len(ND.files) != 0, f"{dir(ND)}"

        # check if those files exist locally
        assert all(
            [filename.exists() for filename in ND.files]
        ), "Filenames do not exist."

        # check if all files have data in them
        for filename in ND.files:
            array = np.genfromtxt(str(filename), delimiter=" ")
            assert array.shape[1] == 3

    def test_ordering(self, ND):
        # test if ordering of all properties correspond
        for i, _ in enumerate(ND.files):
            graphi_file = ND.files[i]
            graphi_subject = ND.subjects[i]
            graphi_graph = ND.graphs[i]

            # subject/files correspond to same scan
            pattern = r"(?<=sub-)(\w*)(?=_ses)"
            assert re.findall(pattern, str(graphi_file))[0] == graphi_subject

            # subject/files and graphs/X-rows correspond to the same scan
            current_NX = nx.read_weighted_edgelist(
                graphi_file, nodetype=int, delimiter=ND.delimiter
            )
            graph_from_file_i = nx.to_numpy_array(
                current_NX, nodelist=ND.vertices, dtype=np.float
            )
            assert np.array_equal(graph_from_file_i, graphi_graph)

    def test_to_directory(self, ND):
        # TODO: use tmp_path_factory
        p = "/tmp/testdir"
        if Path(p).exists():
            shutil.rmtree(p)
        ND.to_directory(p)

        # test that original directory still exists unchanged
        for new, old in zip(sorted(Path(p).iterdir()), ND.files):
            assert new.name == old.name

    # TODO : test to check atlas pulls from the right thing.
    # TODO : test to check s3 directory pulling.
