import os
import shutil
from pathlib import Path

from graphutils import graph_io
import pytest
import numpy as np
from graphutils.graph_io import NdmgDirectory


DATAPATH = "/Users/alex/Dropbox/NeuroData/graphutils/tests/test_data"


@pytest.fixture
def basedir(tmp_path):
    # populate tmpdir with file
    # TODO : figure out how to not have to instantiate this every time
    #        a test is run
    for filename in Path(DATAPATH).iterdir():
        shutil.copy(filename, tmp_path)
    return tmp_path


@pytest.fixture
def ND(basedir):
    # so that I don't have to instantiate all the time
    return NdmgDirectory(basedir)


class TestNdmgDirectory:
    def test_dir(self, basedir):
        assert isinstance(basedir, Path)

    def test_has_attributes(self, ND):
        assert all(
            hasattr(ND, attr) for attr in ["dir", "name", "files", "graphs", "X", "Y"]
        )

    def test_has_data(self, ND):
        # check if there are files
        assert len(ND.files) != 0, f"{dir(ND)}"

        # check if all files have data in them
        for filename in ND.files:
            # TODO : dynamic delimiter, should be ND.delimiter
            array = np.genfromtxt(str(filename), delimiter=" ")
            assert array.shape[1] == 3


# test files
# TODO : check if all files are edgelists with the correct stuff in them,
#        and if they are not,
#        raise an exception

# TODO : tests to make sure that the order of files, graphs, X, and Y all correspond to each other.

# test graphs
# TODO : test to make sure sort order is the same as `self.files`

# test Y
# TODO : test to make sure sort order is the same as `self.files`

# test _X
# TODO : test to make sure order of rows matches order of `self.files`.
# TODO : test to make sure order within rows matches order within arrays of `self.files`.

# Test _pass_to_ranks

# test save_X_and_Y
