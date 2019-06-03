
from pathlib import Path
import pytest

from graphutils.graph_io import NdmgDirectory
from graphutils.graph_stats import NdmgDiscrim

@pytest.fixture(params=["simple_graphs", "full_directory"])
def ND(shared_datadir, request):
    p = Path(request.param)
    return NdmgDirectory(shared_datadir / p)

@pytest.fixture(params=["simple_graphs", "full_directory"])
def NDD(shared_datadir, request):
    p = Path(request.param)
    return NdmgDiscrim(shared_datadir / p)