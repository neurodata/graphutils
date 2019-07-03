#%%
from pathlib import Path

KEYWORDS = ("sub-", "ds_adj")
CORRECT_SUFFIXES = (".ssv", ".csv")

def is_graph(filename):
    """
    Check if `filename` is a ndmg graph file.
    
    Parameters
    ----------
    filename : str or Path
        location of the file.
    
    Returns
    -------
    bool
        True if the file has the ndmg naming convention, else False.
    """

    correct_suffix = Path(filename).suffix in CORRECT_SUFFIXES
    correct_filename = all(i in str(filename) for i in KEYWORDS)  
    return correct_suffix and correct_filename

def filter_graph_files(file_list):
    """
    Generator. 
    Checks if each file in `file_list` is a ndmg edgelist,
    yields it if it is.
    
    Parameters
    ----------
    file_list : iterator
        iterator of inputs to the `is_graph` function.
    """
    for filename in file_list:
        if is_graph(filename):
            yield(filename)
