# RI-2022 practical assignment (IR-system)

This repository contains the started code to build a fully functional IR system, it was projected to serve as guidelines to the RI students during their class assignments. Here, the students will find a definition of a generic enough search engine API, that they should complete with the most adequate IR methods learned during the classes.

## Authors <!-- omit in toc -->

[José Trigo - 98597](https://github.com/zepedrotrigo)\
[Simão Arrais - 85132](https://github.com/simaoarrais)

## Table-of-Contents


* [Program Overview](#program-overview)
	* [Life cycle flow](#life-cycle-flow)
	* [Argparse preprocessing](#argparse-options)
* [How to run](#how-to-run)
* [High-level API overview](#high-level-api-overview)
	* [Reader API](#reader-api)
	* [Tokenizer API](#tokenizer-api)
	* [Indexer API](#indexer-api)
* [Code Overview and Tips](#code-overview-and-tips)
	* [How to add more options to the program](#more-options)
	* [Dynamic loading of classes](#class-loading)
* [Results](#results)
    * [Indexer](#assignment-1-delivery-indexing)
    * [Searcher](#assignment-2-delivery-tf-idf-bm25-searcher)

## Program Overview

The file `main.py` corresponds to the main entry point of the system that has two modes of operation the **indexer mode** and the **searcher mode**

- **indexer** mode: Responsible for the creation of indexes for a specific document collection. The index is a special data structure that enables fast searching over an enormous amount of data (text).
- **searcher** mode: Responsible for searching and ranking the documents (text) given a specific question. The searcher presupposes that an index was previously built.

The `main.py` also contains the CLI (command line interface) that exposes the correct options/ways to run the IR system, further extension is possible by the students, however, they should not change the main structure of the CLI. Furthermore, the students **should not** change the `main.py` file, and in the case that they want to specify additional options, they should implement the functions `add_more_options_to_*` in the file `core.py`, which exposes the argparser that is being used in the `main.py` file.

The `core.py` corresponds to the first student entry point, meaning that here is where the students can start to make changes to the code to solve the proposed assignments. As a recommendation, we already offer a well-defined high-level API that the students can follow to implement the missing IR functionalities.

### Life cycle flow

As mentioned, the `main.py` is the entry point of the system. The first code to run is the construction of a CLI interpreter through the `argparse` lib. Here, the students have the option of adding new terminal arguments by extending the current `argparse` functionality, for that, the students **should** use the `add_more_options_to_*` functions at the beginning of the `core.py` file (see [this](#more-options) for more details). After this, the terminal arguments are parsed and preprocessed (see [this](#argparse-options) for more details) creating a well-defined `argparse.Namespace` that holds every configuration about the current execution. The execution is then passed to the [`engine_logic`](https://github.com/detiuaveiro/RI-2022/blob/master/core.py#L31) function that depending on the specified mode of execution will split the execution path between the `indexer_logic` and `searcher_logic` functions. We consider that the previous functions are good starting points for the student to start to develop the remainder of the search engine system. 

At the time of writing, we also specified a minimalistic definition of the `indexer_logic` main body that the students may or may not change.

<h3 name="argparse-options">
Argparse preprocessing
</h3>

Before explaining this step is recommended that the student get familiar with the CLI defined on the `main.py` file. 

As observable in the file, every optional argument should be formatted according to the following format `--<group name>.<variable name> <variable name>` or `--<variable name> <variable name>`. Here, we recommend the use of the first option (which is the option used in the `main.py`) since we are following an Objecting Oriented (OO) approach and each variable must be specified within an object context. So, in this case, the `group name` corresponds to that object that should hold its variable. By following this methodology, the `.grouping_args` function will automatically group every option that follows this format into a `Params` object that can be easily accessible. 

As an example consider the [tokenizer options](https://github.com/detiuaveiro/RI-2022/blob/master/main.py#L159-L181) defined in the `main.py`. Here, we specified the `tk` as the group name followed by four different variables and their respective values (class, minL, stopwords_path and stemmer). Then the preprocessing will create a tk entry on the `argparse.Namespace` associated with a Params object that holds all four variables. To access its values we just need to do ``` args.tk.class ```, or ``` args.tk.minL ```, etc... Furthermore, we can also use ``` args.tk.get_kwargs()``` to get a python dictionary with all the variable and its values stored inside the `Params` object.

Note that the `.class` is a special variable used for the specification of the class that should be initialized at runtime, we use this to dynamically initialise the modules that we can be specified through the CLI (see [this](#class-loading) for more details). 


## How to run

Here are some examples of how to run the `main.py` file that we encourage you to try out.

Example of command to run the indexer:
```bash
python main.py indexer collections/pubmed_tiny.jsonl pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer potterNLTK
```

Here, the program should index the documents present on `pubmed_tiny.jsonl` file and save the resulting index to the folder pubmedSPIMIindex, we also specified special options that are sent to the tokenizer (tk), for instance, we said to use the stopword file `stopw.txt`.

Example of command to run the searcher:
```
It will be updated before the second assignment
```

The program also has a built-in help menu for each of the execution modes, try:
```bash
python main.py -h
```

```bash
python main.py indexer -h
```

```bash
python main.py searcher -h
```

## High-level API overview

Our high-level API follows the main modules of an IR system (indexer, tokenizer, reader, searcher), and uses a plugin-like architecture to initialize each module, which offers a high degree of modularity and eases the addition of future functionality. The name of the classes that should be initialized are specified as CLI arguments (see the `main.py` file for more detail) 

The remainder of the files can be freely adapted by the students, however, we recommend sticking with this already defined API.

### Reader API

High-level abstraction on how the documents are read from an input stream (like a disk) to a continuous stream of text. The code resides in the `reader.py` file, the base class is the `Reader` class that only holds the path to the collection that the program should read. The students should complete this API by implementing the missing functionality to read the documents stored on disk over a specific format. Besides the base class, we also created a PubMedReader that extends the previous one and should be responsible to read the pubmed.jsonl files that are required for the first assignment. 

### Tokenizer API

High-level abstraction on how the text should be processed and split into individual tokens. The code resides in the `tokenizer.py` file, the base class is the `Tokenizer` class that exposes the high-level method `_.tokenize(document)_` that should be implemented by its sub-classes. The students should complete this API by implementing the missing functionality. Besides the base class, we also created a `PubMedTokenizer` that extends the previous one and should be responsible for the tokenization of PubMed articles.

**Note that here `PubMedTokenizer` may not be the best name to give, since the tokenizer may be generic enough to be used with other types of documents, so the student should consider this aspect of the code as well.**

### Indexer API

High-level abstraction on how the tokens (from documents) should be indexed by the engine. The code resides in the `indexer.py` file, the base class is the `Indexer` class that exposes the high-level method `_.build_index(reader, tokenizer, index_output_folder)_` that should be implemented by its base classes. The students should complete this API by implementing the missing functionality. Besides the base class, we also created a `SPIMIIndexer` that extends the previous one and should be responsible for the implementation of the SPIMI algorithm. Here, we also specified a basic index abstraction called BaseIndex, which holds some high-level functionality that an index must have.

Tip: May be worth it to think of the BaseIndex not as the final index but as an abstract (index manager) so that it eases the coordination of a group of index pieces that holds part of the overall index, recall that it will be imposible to fully load the index for the bigger collections of the pubmed.

## Code Overview and Tips

<h3 name="more-options">
How to add more options to the program (correct way according to the CLI)
</h3>

In order to expand the CLI options without changing the `main.py` file, we expose the argparser object in the [_add_more_options_to_indexer_](https://github.com/detiuaveiro/RI-2022/blob/master/core.py#L16) function. Here, the students can then add additional arguments that will be parsed at runtime. When looking at the signature of the _add_more_options_to_indexer_ function it is observable that it receives three instances of `argparse.ArgumentParser` objects, which may be confusing when looking for the first time. However, the reason behind this is just for a matter of organization and better looking build-in helping menus. Any of these arguments can be used to add options to the program and the code will just work! But for sake of completeness the first argunment (`indexer_parser`) corresponds to the base `argparse.ArgumentParser` that was used to add the positional indexer arguments (path_to_collection and index_output_folder). Then, `indexer_settings_parser` was used to specify the optional settings for the indexer, while the `indexer_doc_parser` was used to specify the optional settings for the document processing classes like the `Tokenizer` and `Reader`.

As an example consider that now:
 - We want to add an option to the indexer for enabling index-compressing
 - We want to add an option to perform the tokenization with multi processes
 - We want to create a new options group to specify special options for DEBUG
     - adds an option that changes the logging level.
     
All of this new ideas can be trivially achievable by extending the [_add_more_options_to_indexer_](https://github.com/detiuaveiro/RI-2022/blob/master/core.py#L16) function, like so:

```python

def add_more_options_to_indexer(indexer_parser, indexer_settings_parser, indexer_doc_parser):
    
    # adding option to the indexer_settings_parser that was set up under the "Indexer settings" group
    indexer_settings_parser.add_argument('--indexer.index_compression', 
                                         action="store_true",
                                         help='If this flag is added the variable index_compression becomes True else it will be False')

    # adding option to the indexer_doc_parser that was set up under the "Indexer document processing settings"
    # Side note, as you probably know python has a global interpreter lock, which prevents the utilization of 
    # threading for high computation tasks. So that's the reason to use multi-process instead. 
    indexer_doc_parser.add_argument('--tk.multiprocess', 
                                    type=int, 
                                    default=1,
                                    help='Number of process to be used during the tokenization. (default=1).')
    
    # creating a new group of settings
    debug_settings_parser = indexer_parser.add_argument_group('DEBUG settings', 'This settings corresponds optional DEBUG configurations')
    debug_settings_parser.add_argument('--config.logging_level', 
                                        type=int, 
                                        default=1,
                                        help='Changes the logging level(default=1).')
```

We suggest that the students add this code to the [_add_more_options_to_indexer_](https://github.com/detiuaveiro/RI-2022/blob/master/core.py#L16) function and see the differences when running this system. (RUN: `python main.py indexer -h` and `python main.py indexer collections/pubmed_tiny.jsonl pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer a`).

```
$ python main.py indexer collections/pubmed_tiny.jsonl pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer a
init PubMedReader| self.path_to_collection='collections/pubmed_tiny.jsonl'
init PubMedTokenizer| minL=2, stopwords_path='stopw.txt', stemmer='a'
PubMedTokenizer also caught the following additional arguments {'multiprocess': 1}
init SPIMIIndexer| posting_threshold=None, memory_threshold=None
SPIMIIndexer also caught the following additional arguments {'index_compression': False}
Indexing some documents...
Print some stats about this index.. This should be implemented by the base classes
```

As observable, when running the program, now two new information lines appear (`PubMedTokenizer also caught the following additional arguments {'multiprocess': 1}` and `SPIMIIndexer also caught the following additional arguments {'index_compression': False}`), this happened to show that the newly added arguments are being passed to the respective classes. However, they are not being caugh by their constructors. So, this mechanism enables the addition of any arbitrary number of terminal arguments, but only the arguments that are defined in the __init__ method of each class will be caught! For instance, consider making the following change to the `PubMedTokenizer` in the `tokenizer.py` file:

```python
class PubMedTokenizer(Tokenizer):
    
    def __init__(self, 
                 minL, 
                 stopwords_path, 
                 stemmer,
                 multiprocess,
                 *args, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.minL = minL
        self.stopwords_path = stopwords_path
        self.stemmer = stemmer
        self.multiprocess = multiprocess
        print("init PubMedTokenizer|", f"{minL=}, {stopwords_path=}, {stemmer=}, {multiprocess=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
```

This change will result in this new output:

```
init PubMedReader| self.path_to_collection='collections/pubmed_tiny.jsonl'
init PubMedTokenizer| minL=2, stopwords_path='stopw.txt', stemmer='a', multiprocess=1
init SPIMIIndexer| posting_threshold=None, memory_threshold=None
SPIMIIndexer also caught the following additional arguments {'index_compression': False}
Indexing some documents...
Print some stats about this index.. This should be implemented by the base classes
```

As observable, now the PubMedTokenizer is catching the new added argument `multiprocess`.

<h3 name="class-loading">
Dynamic loading of classes
</h3>

Following the plugin-like architecture, each of the main modules is dynamically initialized by specifying its class name as a string for each main module. For that, each of the main modules has a _dynamically_init_* _ function that automatically initializes any class that belongs to each module. 

For instance, consider the Reader module (reader.py), at the beginning of the file we can see the _dynamically_init_reader_ function that should be used to dynamically initialize any class that resides inside that file, which means that we can initialize the class PubMedReader at runtime by calling the previous function with the class name (_dynamically_init_reader("PubMedReader")_). Note that by default the program sets a CLI argument (_reader.class_) as "PubMedReader", which turns up to be the default reader that will be loaded by the program.

At this point, it should be clear that for adding new functionality the only thing that it is required to do is to implement a new class in the reader.py file that holds the code for a new type of reader. For instance, if we want to read XML files, we can build an XMLReader that extends Reader and when running the program we specified that this class should be used, like so:

```python 
class XMLReader(Reader):
    
    def __init__(self, 
                 path_to_collection:str,
		 xml_in_memory_limit: float,
                 **kwargs): # VERY IMPORTANT TO HAVE THIS ARGUMENT, in order to catch non matching arguments with the constructor signature
        super().__init__(path_to_collection, **kwargs)
	self.xml_in_memory_limit=xml_in_memory_limit
        print("init XMLReader|", f"{self.path_to_collection=}, f"{self.xml_in_memory_limit=}")
```

```bash
python main.py indexer collections/pubmed_tiny.xml pubmedSPIMIindex --reader.class XMLReader
```

**Important note: For all of the modules class that are dynamically loaded the last argument of the constructor must be the `**kwargs` or else the program may crash if any option from the CLI that is not specified on the constructor is used to initialize that class.**

If your class needs extra arguments this is also easily achievable, just add the extra arguments to the CLI by extending the argparser (exposed by the [add_more_options_to_indexer](https://github.com/detiuaveiro/RI-2022/blob/master/core.py#L16) function). Note that the added arguments must be optional and under the _reader._ namespace, as an example, consider:

```python
def add_more_options_to_indexer(indexer_parser, indexer_settings_parser, indexer_doc_parser):
    
    # adding option to the indexer_doc_parser that was set up under the "Indexer document processing settings" group
    # however, this will also work if it was added to the other exposed parsers.
    indexer_doc_parser.add_argument('--reader.xml_in_memory_limit', 
                                    type=float, 
                                    default=0.5,
                                    help='Fraction of the available RAM that the XMLReader will use. (default=0.5).')
```

After changing this function in the `core.py` file the argument _xml_in_memory_limit_ is automatically passed to the reader class, that in this case will be the XMLReader. For more information on how this mechanism works check [this](#more-options).
<!-- 
#### Code TIP

Consider the implementation of a manager class that can handle multiple types of files, for instance, a ReaderManager that instantiates other types of readers like PubMedReader, JsonReader, XMLReader, etc... Although during this assignment a reader that can read a jsonl file is enough, its consider a good programming practice to build modular and easily expandible solutions.
-->

## Results
## Assignment 1 delivery: Indexing
### pubmed_2022_tiny.jsonl.gz dataset:
```
python3 main.py indexer collections/pubmed_2022_tiny.jsonl.gz pubmedSPIMIindex --tk.minL 3 --tk.stopwords stopw.txt
```
#### No Stemming:
```
Total indexing time: 30.52s
Merging time: 0.00s
Number of temporary index segments written to disk: 0
Total index size on disk: 129.1 MB
Vocabulary size: 252104
```
#### Using Porter Stemmer:
```
Total indexing time: 330.79s
Merging time: 0.00s
Number of temporary index segments written to disk: 0
Total index size on disk: 120.2 MB
Vocabulary size: 203450
```
#### Using Snowball Stemmer:
```
Total indexing time: 219.19s
Merging time: 0.00s
Number of temporary index segments written to disk: 0
Total index size on disk: 120.1 MB
Vocabulary size: 205597
```
#### Using Snowball Stemmer on a (simulated) 10 MB memory machine:
```
Total indexing time: 239.18s
Merging time: 7.55s
Number of temporary index segments written to disk: 5
Total index size on disk: 120.5 MB
Vocabulary size: 205597
```
---
### pubmed_2022_small.jsonl.gz dataset:
```
python3 main.py indexer collections/pubmed_2022_small.jsonl.gz pubmedSPIMIindex --tk.minL 3 --tk.stopwords stopw.txt
```
### Using Snowball Stemmer:
```
Total indexing time: 2296.68s
Merging time: 0.00s
Number of temporary index segments written to disk: 0
Total index size on disk: 1218.2 MB
Vocabulary size: 851528
```
#### Using Snowball Stemmer on a (simulated) 4 GB memory machine:
```
Total indexing time: 2867.44s
Merging time: 113.13s
Number of temporary index segments written to disk: 9
Total index size on disk: 1212.8 MB
Vocabulary size: 831017
```
---
## Assignment 2 delivery: tf-idf / bm25 searcher
Note: The provided searcher class was not fully used / was subject to some modifications
### pubmed_2022_tiny.jsonl.gz dataset:
#### [TF-IDF] using Snowball Stemmer on a (simulated) 10 MB memory machine:
```
python3 main.py indexer collections/pubmed_2022_tiny.jsonl.gz pubmedSPIMIindex --tk.minL 3 --tk.stopwords stopw.txt --tk.stemmer snowball --indexer.ranking_schema tfidf --indexer.tfidf.smart lnc.ltc --indexer.memory_threshold 10000000

Total indexing time: 250.77s
Merging time: 4.89s
Number of temporary index segments written to disk: 5
Total index size on disk: 191.5 MB
Vocabulary size: 205597
```
```
python3 main.py searcher pubmedSPIMIindex questions_path output.txt ranking.tfidf

-> Search Results [lnc.ltc] for 'Are gut microbiota profiles altered by irradiation?'

 1. PMID 31807366 (score: 0.1280908066035698)
 2. PMID 33553216 (score: 0.12155517532790486)
 3. PMID 31531264 (score: 0.10624792198701505)
 4. PMID 19692992 (score: 0.09710719527511845)
 5. PMID 2221304 (score: 0.09559623250467161)
 6. PMID 31435014 (score: 0.09308591060550439)
 7. PMID 33183623 (score: 0.09293085833944309)
 8. PMID 22482724 (score: 0.09290208551634566)
 9. PMID 23320832 (score: 0.09233760328310095)
10. PMID 34267042 (score: 0.0915114014982219)

-> Search Results [lnc.npc] for 'Are gut microbiota profiles altered by irradiation?'

 1. PMID 31807366 (score: 0.13918268890991972)
 2. PMID 33553216 (score: 0.1333560228927687)
 3. PMID 20498668 (score: 0.12633861973282695)
 4. PMID 30140751 (score: 0.12041036383308512)
 5. PMID 31531264 (score: 0.1165627072528011)
 6. PMID 27767329 (score: 0.1158325672457819)
 7. PMID 19692992 (score: 0.1073260143604947)
 8. PMID 33183623 (score: 0.10476233294731835)
 9. PMID 2221304 (score: 0.10387428295956451)
10. PMID 31435014 (score: 0.10212289844685982)
```
#### [BM25] using Snowball Stemmer on a (simulated) 10 MB memory machine:
```
python3 main.py indexer collections/pubmed_2022_tiny.jsonl.gz pubmedSPIMIindex --tk.minL 3 --tk.stopwords stopw.txt --tk.stemmer snowball --indexer.ranking_schema bm25 --indexer.memory_threshold 10000000

Total indexing time: 235.95s
Merging time: 4.26s
Number of temporary index segments written to disk: 5
Total index size on disk: 191.5 MB
Vocabulary size: 205597
```
```
python3 main.py searcher pubmedSPIMIindex questions_path output.txt ranking.bm25

-> Search Results [k=1.2 b=0.75] for 'Are gut microbiota profiles altered by irradiation?'

 1. PMID 34107468 (score: 15.131277071447716)
 2. PMID 30353179 (score: 15.027911989940797)
 3. PMID 34220497 (score: 14.925685111023675)
 4. PMID 31660868 (score: 14.780973550603884)
 5. PMID 22617141 (score: 14.64548952498341)
 6. PMID 32605083 (score: 14.51634862871743)
 7. PMID 34422677 (score: 14.396395827062861)
 8. PMID 34054805 (score: 14.395533771048932)
 9. PMID 31436705 (score: 14.347797831449121)
10. PMID 32305646 (score: 14.322504522042081)
```
---
## Assignment 3 delivery
Example of command to run the Searcher for BM25:
```
python3 main.py searcher pubm collections/questions_with_gs/question_E8B1_gs.jsonl output.txt ranking.bm25
```

Results from the first question of the file question_E8B1_gs.jsonl with BM25

```
Question -> {'query_id': '5e48e0e0f8b2df0d49000001', 'query_text': 'Are gut microbiota profiles altered by irradiation?', 'documents_pmid': ['30430918', '30343431', '30459840']}

Precision -> 0.3
Recall -> 1.0
Avg-Precision -> 0.9722222222222223
F-Measure -> 0.4615384615384615

1. PMID 30430918 (score: {'score': 23.851756392383344, 'num_search_terms': 5, 'token_positions': [[5, 230], [3, 105, 221, 228], [4, 106, 229], [107], [8, 85, 146, 157]]})
2. PMID 30343431 (score: {'score': 22.651207515018037, 'num_search_terms': 4, 'token_positions': [[2, 22, 71, 117, 130], [3, 23, 118, 131], [119, 132], [127]]})
3. PMID 22993202 (score: {'score': 20.952361789562342, 'num_search_terms': 4, 'token_positions': [[6, 12, 199, 209], [13, 66, 77, 161, 183], [75, 89], [196]]})
4. PMID 30459840 (score: {'score': 20.77857858912166, 'num_search_terms': 3, 'token_positions': [[47, 67, 78, 116, 134, 157, 185], [79, 117, 135], [9, 73]]})
5. PMID 34422677 (score: {'score': 20.549175501568648, 'num_search_terms': 4, 'token_positions': [[241], [4, 36, 73, 90, 155, 180, 198, 267, 289], [5, 37, 91, 156, 181, 199, 268, 290], [179, 196, 229]]})
6. PMID 32418127 (score: {'score': 20.314317840243152, 'num_search_terms': 4, 'token_positions': [[47], [8, 42, 98, 181, 219], [9, 43, 99, 189, 220], [104, 141]]})
7. PMID 22964750 (score: {'score': 20.294057249452447, 'num_search_terms': 3, 'token_positions': [[10, 27, 69, 91, 107, 140, 167, 188], [11, 28, 70, 92, 108, 112, 141, 168, 189], [12, 29, 93]]})
8. PMID 30828805 (score: {'score': 20.116509990988863, 'num_search_terms': 4, 'token_positions': [[32], [1, 17, 70, 79, 164, 193, 218, 257], [2, 80, 159, 194, 219, 247], [161, 223]]})
9. PMID 33599103 (score: {'score': 19.948232499040664, 'num_search_terms': 4, 'token_positions': [[73], [33, 42, 59, 280, 314], [34, 43, 60, 69, 281, 315], [49, 137, 248]]})
10. PMID 33654382 (score: {'score': 19.8997764071322, 'num_search_terms': 4, 'token_positions': [[27], [13, 45, 78, 135, 208, 259, 269], [14, 46, 79, 136, 209, 260, 268], [74]]})
```
Example of command to run the Searcher with tf-idf:
```
python3 main.py searcher pubm collections/questions_with_gs/question_E8B1_gs.jsonl output.txt ranking.tfidf
```

Results from the first question of the file question_E8B1_gs.jsonl with BM25
```
{'query_id': '5e48e0e0f8b2df0d49000001', 'query_text': 'Are gut microbiota profiles altered by irradiation?', 'documents_pmid': ['30430918', '30343431', '30459840']}

Precision -> 0.3
Recall -> 1.0
Avg-Precision -> 0.9722222222222223
F-Measure -> 0.4615384615384615

 1. PMID 30430918 (score: {'score': 4.718125827630548, 'num_search_terms': 5, 'token_positions': [[5, 230], [3, 105, 221, 228], [4, 106, 229], [107], [8, 85, 146, 157]]})
 2. PMID 30343431 (score: {'score': 4.480644756113361, 'num_search_terms': 4, 'token_positions': [[2, 22, 71, 117, 130], [3, 23, 118, 131], [119, 132], [127]]})
 3. PMID 22993202 (score: {'score': 4.144595378341256, 'num_search_terms': 4, 'token_positions': [[6, 12, 199, 209], [13, 66, 77, 161, 183], [75, 89], [196]]})
 4. PMID 30459840 (score: {'score': 4.110219251362643, 'num_search_terms': 3, 'token_positions': [[47, 67, 78, 116, 134, 157, 185], [79, 117, 135], [9, 73]]})
 5. PMID 34422677 (score: {'score': 4.064840931438678, 'num_search_terms': 4, 'token_positions': [[241], [4, 36, 73, 90, 155, 180, 198, 267, 289], [5, 37, 91, 156, 181, 199, 268, 290], [179, 196, 229]]})
 6. PMID 32418127 (score: {'score': 4.018383640013776, 'num_search_terms': 4, 'token_positions': [[47], [8, 42, 98, 181, 219], [9, 43, 99, 189, 220], [104, 141]]})
 7. PMID 22964750 (score: {'score': 4.014375884143722, 'num_search_terms': 3, 'token_positions': [[10, 27, 69, 91, 107, 140, 167, 188], [11, 28, 70, 92, 108, 112, 141, 168, 189], [12, 29, 93]]})
 8. PMID 30828805 (score: {'score': 3.979255187285962, 'num_search_terms': 4, 'token_positions': [[32], [1, 17, 70, 79, 164, 193, 218, 257], [2, 80, 159, 194, 219, 247], [161, 223]]})
 9. PMID 33599103 (score: {'score': 3.9459681467884655, 'num_search_terms': 4, 'token_positions': [[73], [33, 42, 59, 280, 314], [34, 43, 60, 69, 281, 315], [49, 137, 248]]})
10. PMID 33654382 (score: {'score': 3.9363830271445144, 'num_search_terms': 4, 'token_positions': [[27], [13, 45, 78, 135, 208, 259, 269], [14, 46, 79, 136, 209, 260, 268], [74]]})
```

#### Notes

The above listed results were tested on an i7-8750h laptop with 16GB of memory

Some results have tiny discrepancies (<0.0001%) in vocabulary size due to special characters/encoding issues.  

The program utilizes up to 80% of available (or simulated) memory before dumping contents do disk.  

If the postings list is split among multiple files, file pointers will be included in the final index.  
