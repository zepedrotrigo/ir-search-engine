"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: 

Core functionality of the system, its the
main system entry point and holds the top-level
logic. It starts by spliting the execution path 
based on the operation mode.

"""
import time

from tokenizers import dynamically_init_tokenizer, PubMedTokenizer
from reader import dynamically_init_reader
from index import dynamically_init_indexer, BaseIndex
from searcher import dynamically_init_searcher

def add_more_options_to_indexer(indexer_parser, indexer_settings_parser, indexer_doc_parser):
    """Add more options to the main program argparser.
    This function receives three argparser as arguments,
    and each of them can be used to add parsing options
    for the indexer mode. Check the details below to
    understand the difference between them.

    Parameters
    ----------
    indexer_parser : ArgumentParser
        This is the base argparser used during the indexer
        mode.
    indexer_settings_parser : ArgumentParser
        Derives from the ìndexer_parser` and specifies a group setting
        for the indexer. This should be used if we aim to add options to 
        the indexer.
    indexer_doc_parser : ArgumentParser
        Derives from the ìndexer_parser` and specifies a group setting
        for the document processing classes. This should be used if we 
        aim to add options to tokenizer, reader or any other document
        processing class.

    """
    indexer_doc_parser.add_argument('--tk.case_folding', 
                                type=bool, 
                                default=True,
                                help='Decides whether to use normalization to lowercase or not. (default=true).')

    indexer_doc_parser.add_argument('--tk.allow_numbers', 
                                type=bool, 
                                default=False,
                                help='Decides whether to allow numbers as valid tokens or not. (default=false).')

    indexer_doc_parser.add_argument('--indexer.ranking_schema', 
                            type=str, 
                            default="tfidf",
                            help='Choose indexer ranking schema. (default=tfidf).')

def engine_logic(args):
    """
    Entrypoint for the main engine logic. Here we split
    the current two modes of execution. The indexer mode 
    and the searcher mode. Read the Readme.md to better
    understand this methodology.
    
    Parameters
    ----------
    args : argparse.Namespace
        Output of the ArgumentParser::parse_args method
        it holds every parameter that was specified during the
        cli and its default values.
    
    """
    
    
    if args.mode == "indexer":
        
        indexer_logic(args.path_to_collection,
                      args.index_output_folder,
                      args.indexer,
                      args.reader,
                      args.tk)
        
    elif args.mode == "searcher":
        ## TO BE DONE
        # searcher_logic [This will be probably the name of the function still in revision]
        searcher_logic(args.index_folder,
                       args.path_to_questions,
                       args.output_file,
                       args.top_k,
                       args.reader,
                       args.tk,
                       args.ranking)
        
    else:
        # this should be ensured by the argparser
        raise RuntimeError("Enter the else condition on the main.py, which should never happen!")
    

def indexer_logic(path_to_collection, 
                  index_output_folder, 
                  indexer_args, 
                  reader_args, 
                  tk_args):
    """
    Entrypoint for the main indexer logic. Here we start by
    dynamically loading the main modules (reader, tokenizer,
    indexer). Then we build 
    
    Parameters
    ----------
    args : argparse.Namespace
        Output of the ArgumentParser::parse_args method
        it holds every parameter that was specified during the
        cli and its default values.
    
    """
    # Students can change if they want
    #index_output_folder = f"{index_output_folder}/{time.time()}"
    
    # init reader
    reader = dynamically_init_reader(path_to_collection=path_to_collection,
                                   **reader_args.get_kwargs())
    
    # init tokenizer
    tokenizer = dynamically_init_tokenizer(**tk_args.get_kwargs())
    
    # init indexer
    indexer = dynamically_init_indexer(**indexer_args.get_kwargs())
    
    # execute the indexer logic
    indexer.build_index(reader, tokenizer, index_output_folder)
    
    # get the final index
    index = indexer.get_index()
    
    # print some statistics about the produced index
    indexer.print_statistics(index_output_folder)
    
    
    
def searcher_logic(index_folder,
                   path_to_questions,
                   output_file,
                   top_k,
                   reader_args,
                   tk_args,
                   ranking_args):


    reader = dynamically_init_reader(path_to_questions=path_to_questions,
                                    **reader_args.get_kwargs())
    
    

    ranker = dynamically_init_searcher(**ranking_args.get_kwargs())

    # load the index from disk
    index = BaseIndex.load_from_disk(index_folder)

    stored_tokenizer_kwargs = index.get_tokenizer_kwargs()
    if stored_tokenizer_kwargs:
        # if new tk parameters are specified we override the arguments loaded from the index
        # this enables initialize a tokenizer that differs from the one used during indexation
        stored_tokenizer_kwargs.update(tk_args.get_kwargs_without_defaults())
        tk_kwargs = stored_tokenizer_kwargs
    else:
        # the tokenizer was not saved in the index so lets use the one defined in the CLI (and use the default values if not defined)
        tk_kwargs = tk_args.get_kwargs()

    # o esqueleto está a dar problemas com isto, por isso vou comentar
    #tokenizer = dynamically_init_tokenizer(**tk_kwargs)
    tokenizer = PubMedTokenizer(minL=3, stopwords_path="stopw.txt", stemmer=None, case_folding=True, allow_numbers=False)

    #ranker.batch_search(index, reader, tokenizer, output_file, top_k=top_k)
    ranker.search(tokenizer, index_folder, top_k, reader)
