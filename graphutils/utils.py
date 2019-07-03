#%%
from pathlib import Path

KEYWORDS = ("sub-", "ds_adj")
CORRECT_SUFFIXES = (".ssv", ".csv")

def is_graph(filename):

    correct_suffix = Path(filename).suffix in CORRECT_SUFFIXES
    correct_filename = all(i in str(filename) for i in KEYWORDS)  
    return correct_suffix and correct_filename

def filter_graph_files(file_list):
    for filename in file_list:
        if is_graph(filename):
            yield(filename)
