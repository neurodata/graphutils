import os
import shutil
from pathlib import Path
import re

import pytest
import numpy as np
import networkx as nx
from graphutils.graph_io import NdmgDirectory
from graphutils.graph_stats import NdmgDiscrim

# class TestNdmgDiscrim:
#     # TODO: test X ordering
#         # graphi_X = ND.X[i].reshape(
#         #     int(np.sqrt(ND.X.shape[1])), int(np.sqrt(ND.X.shape[1]))
#         # )  # TODO

#         # graphs/X-rows correspond to same scan
#         # assert np.array_equal(graphi_graph, graphi_X)  # TODO
#     def test_dir(self, shared_datadir):
#         assert isinstance(shared_datadir, Path)

#     def test_PTR(self, NDD):
#         # TODO : make sure I can recreate X and Y from the files in `save_X_and_Y`
#         # TODO : make sure I refresh the NdmgDirectory object after testing _pass_to_ranks
#         pass

#     def test_save_X_and_Y(self, NDD, tmp_path_factory):
#         # assert we can recreate X and Y from the csv's
#         tmp = tmp_path_factory.mktemp("savedir")
#         saveloc = NDD.save_X_and_Y(tmp)

#         X = np.loadtxt(saveloc.X, delimiter=",")
#         Y = np.loadtxt(saveloc.Y, dtype=str)

#         assert np.array_equal(NDD.X, X)
#         assert np.array_equal(NDD.subjects, Y)