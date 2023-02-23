"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: 

Utility/Auxiliar code

Holds utility code that can be reused by several modules.

"""


import sys, ctypes, os, glob
from timeit import default_timer as timer

'''class added by us students'''
class Timer:
    def __init__(self):
        self.start_ts = None
        self.stop_ts = None

    def start(self):
        self.start_ts = timer()
        return self.start_ts

    def stop(self):
        self.stop_ts = timer()
        return self.stop_ts - self.start_ts

class Block:
    def __init__(self, token, postings):
        self.token = token
        self.postings = postings # {token : # {pmid1: {'w': norm_w1, 'positions': [pos1,pos2]}, pmid2: {'w': norm_w2, 'positions': [pos1,pos2]}}}

    def __repr__(self):
        return f"token={self.token} | postings={self.postings} | df={self.df}"

def dynamically_init_class(module_name, **kwargs):
    """Dynamically initializes a python object based
    on the given class name that resides inside module
    specified by the `module_name`.
    
    The `class` name must be specified as an additional argument,
    this argument will be caught under kwargs variable.
    
    The reason for not directly specifying the class as argument is 
    because `class` is a reserved keyword in python, which may be
    confusing if it is seen as an argument of a function. 
    Additionally, this way the function integrates nicely with the
    `.get_kwargs()` method from the `Param` object.

    Parameters
    ----------
    module_name : str
        the name of the module where the class resides
    kwargs : Dict[str, object]
        python dictionary that holds the variables and their values
        that are used as arguments during the class initialization.
        Note that the variable `class` must be here and that it will
        not be passed as an initialization argument since it is removed
        from this dict.
    
    Returns
        ----------
        object
            python instance
    """

    class_name = kwargs.pop("class")
    return getattr(sys.modules[module_name], class_name)(**kwargs)

def malloc_trim():
    if os.name == "posix": # if Linux OS
        ctypes.CDLL('libc.so.6').malloc_trim(0) # force free malloc

def are_postings_split(index_folder):
    '''returns true if postings are split across more than one file'''
    postings_paths = glob.glob(f"{index_folder}/postings*.pkl")

    if len(postings_paths) == 1:
        return False
    return True

def extract_data_from_index(token, index, index_folder):
    ''' returns document frequency and file pointer from index'''
    if are_postings_split(index_folder): # several postings files, need to read file pointer from index
        return index[token][0], index[token][1] # index {token : [df, filepointer]}
        
    return index[token], 0 # index {token : df} (fp=0)
