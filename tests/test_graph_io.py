from graphutils import graph_io
import pytest
import numpy as np
from graphutils.graph_io import get_X


@pytest.fixture(scope="module")
def testmat():
    return np.full(5, 0)


def test_return_sorted_graph():
    pass


def test_path_files_list():
    pass


def test_get_X():
    a = np.arange(1, 10).reshape(3, 3)
    b = np.arange(9, 0, -1).reshape(3, 3)
    array_list = [a, b]
    # assert True == True
    assert np.all(get_X(array_list, PTR=False) == np.array([a.flatten(), b.flatten()]))


def test_get_Y():
    pass
