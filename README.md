https://travis-ci.org/neurodata/graphutils.svg?branch=master

Collection of utility functions for working with folders of edgelists;
Generally outputs of `ndmg`.

Usage:

```
from graphutils.graph_stats import NdmgStats

n = NdmgStats('s3://ndmg-data/NKI1/ndmg_0-1-2/')  # downloads every edgelist file on s3 into a local temp directory using `boto3`
m = NdmgStats('local/ndmg/path')  # grabs every edgelist file in a local ndmg output directory

for either `n` or `m`, you can immediately call:
n.files : list of full paths to every graph file
n.directory : local directory files are saved into.
n.to_directory(dir) : save all edgelists into a particular directory
n.graphs: 3d numpy array of all 2d adjacency matrices in the dataset
n.subjects: 1d numpy array of subject numbers, sorted such that the order corresponds to n.graphs
n.discriminability : discriminability statistic for this dataset. Code needs to be verified, but it does give me numbers for every dataset. Passes PTR as default behavior.
```
