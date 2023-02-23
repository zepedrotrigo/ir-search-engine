"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: 

Indexer module

Holds the code/logic addressing the Indexer class
and the index managment.

"""

import pickle, os, glob, time, sys, shutil
from math import log10, sqrt
from utils import dynamically_init_class, Timer, Block, malloc_trim, extract_data_from_index


def dynamically_init_indexer(**kwargs):
    """Dynamically initializes a Indexer object from this
    module.

    Parameters
    ----------
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
    return dynamically_init_class(__name__, **kwargs)


class Indexer:
    """
    Top-level Indexer class
    
    This loosly defines a class over the concept of 
    an index.

    """
    
    def __init__(self, 
                 index_instance,
                 **kwargs):
        super().__init__()
        self._index = index_instance
    
    def get_index(self):
        return self._index
    
    def build_index(self, reader, tokenizer, index_output_folder):
        """
        Holds the logic for the indexing algorithm.
        
        This method should be implemented by more specific sub-classes
        
        Parameters
        ----------
        reader : Reader
            a reader object that knows how to read the collection
        tokenizer: Tokenizer
            a tokenizer object that knows how to convert text into
            tokens
        index_output_folder: str
            the folder where the resulting index or indexes should
            be stored, with some additional information.
            
        """
        raise NotImplementedError()
    

class SPIMIIndexer(Indexer):
    """
    The SPIMIIndexer represents an indexer that
    holds the logic to build an index according to the
    spimi algorithm.

    """
    def __init__(self, 
                 posting_threshold, 
                 memory_threshold,
                 tfidf,
                 ranking_schema,
                 **kwargs):
        # lets suppose that the SPIMIIindex uses the inverted index, so
        # it initializes this type of index
        super().__init__(InvertedIndex(), **kwargs)
        self.posting_threshold = posting_threshold
        self.tfidf = tfidf
        self.ranking_schema = ranking_schema
        self.k1 = kwargs.get("bm25")["k1"]
        self.b = kwargs.get("bm25")["b"]
        self.statistics = {"merging_time": 0, "temp_index_segments_n": 0}
        self.logarithm = {} # store pre-calculated logarithms to fetch them later
        self.timer = Timer()

# ------------------------ determines memory threshold ----------------------- #
        available_mem = int(os.popen('free -m').readlines()[1].split()[-1])
        if memory_threshold == None:
            memory_threshold = 2**64

        self.memory_threshold = min(memory_threshold*0.7, available_mem*1e6*0.7)
# ---------------------------------------------------------------------------- #

        print("init SPIMIIndexer|", f"{posting_threshold=}, {memory_threshold=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")


    def build_index(self, reader, tokenizer, index_output_folder): 
        print("Indexing some documents...")
        self.timer.start() 
        block_n = dl_sum = 0
        index =  {} # {token : df}
        postings = {} # {token : # {pmid1: {'w': norm_w1, 'positions': [pos1,pos2]}, pmid2: {'w': norm_w2, 'positions': [pos1,pos2]}}}
        dl_lens = {} # used to store document lengths for bm25

        if os.path.exists(index_output_folder): # make a new dir to save temporary blocks as well as final index
            shutil.rmtree(index_output_folder)
        os.makedirs(index_output_folder)

        i = 0
        reader_gen = reader.read()
        for doc in reader_gen:
            i+=1
            text = doc["title"]+" "+doc["abstract"]
            tokens = tokenizer.tokenize(text)

            for count, t in enumerate(tokens):
                pmid = int(doc["pmid"])

                if t not in index:
                    index[t] = 1

                if t not in postings:
                    postings[t] = {pmid: {'w': 1, 'positions': [count]}}
                else:
                    if pmid in postings[t]:
                        postings[t][pmid]['w'] += 1 # increment tf
                        postings[t][pmid]['positions'].append(count)
                    else:
                        postings[t][pmid] = {'w': 1, 'positions': [count]}

                        if t in index:
                            index[t] += 1 # increment df

            if self.ranking_schema == "bm25":
                l = len(tokens)
                dl_sum += l
                dl_lens[pmid] = l
            else: # if the chosen ranking schema is tf-idf (default schema)
                postings = self.calc_norm_tfidf_weights(pmid, tokens, postings) # now that we have the tf of each token, we can calculate the tfidf weights for each token in this doc

            postings, i, block_n = self.dump_if_threshold_reached(index, postings, i, block_n, index_output_folder)


        # ---------------------- Save index and postings to disk --------------------- #

        sorted_postings = dict(sorted(postings.items(), key=lambda x: x[0]))
        self.statistics["total_indexing_time"] = self.timer.stop()

        merge = block_n # False if block_n==0 else True
        if merge: # if postings were dumped because of memory constraints, we first need to merge the postings
            self.dump_block(sorted_postings, block_n, index_output_folder) # dump current/last block
            self.timer.start()
            index = self.merge_blocks(index, index_output_folder)
            self.statistics["merging_time"] = self.timer.stop()
        
        sorted_index = dict(sorted(index.items(), key=lambda x: x[0]))
        self.write_to_disk(sorted_index, "index", "", index_output_folder) # save index to disk
        self.statistics["vocabulary_size"] = len(index)

        if not merge:
            self.write_to_disk(sorted_postings, "postings", 0, index_output_folder) # save postings to disk

        if self.ranking_schema == "bm25": # if bm25 schema is selected, calc bm25 weights
            avdl = dl_sum / i
            self.calc_bm25_weights(i, avdl, dl_lens, index, index_output_folder)

        with open("document_N.txt", "w") as f:
            f.write(str(i))

        self._index = index

    def merge_blocks(self, index, index_output_folder):
        '''The priority token is the next token to be inserted in the merged index (chosen by its alphabetic order)'''
        postings = {} # {token : # {pmid1: {'w': norm_w1, 'positions': [pos1,pos2]}, pmid2: {'w': norm_w2, 'positions': [pos1,pos2]}}}
        ptr = 0
        block_paths = glob.glob(f"./{index_output_folder}/block*.pkl")
        blocks_reader = [open(b, "rb") for b in block_paths] # read all blocks simultaneously
        blocks = [pickle.load(b) for b in blocks_reader] # advance read buffer by one for each block and store the read elements
        tokens = [b.token for b in blocks] # this list is used to perform a min() comparison between the elements of the blocks to choose the next element to be merged

        while True:
            # dump postings to disk if memory threshold is met. The second condition is explained in the function definition
            while sys.getsizeof(postings)*2 < self.memory_threshold or self.check_duplicates_in_list(tokens):
                if len(tokens) == 0:
                    self.write_to_disk(postings, "postings", ptr, index_output_folder)
                    self.delete_temp_index_blocks(index_output_folder)
                    return index

                # Get block number where priority token is located through the min() function (min() has O(n) complexity)
                min_block = tokens.index(min(tokens))
                priority_token = tokens[min_block] # get priority token from block
                index = self.update_index_fp(index, priority_token, ptr)

                if priority_token in postings:
                    postings[priority_token] = {**postings[priority_token], **blocks[min_block].postings} # merge the two postings lists
                else:
                    postings[priority_token] = blocks[min_block].postings

                # iterate/read next dictionary from the min_block
                new_block = self.read_next_dict_from_block(blocks_reader, min_block)

                #update blocks and tokens lists
                if new_block != None:
                    blocks[min_block] = new_block
                    tokens[min_block] = new_block.token
                else:
                    del blocks[min_block]
                    del tokens[min_block]

            # after reaching memory limit, dump postings to disk
            # we cannot dump current priority token's postings yet since there is still one left in the list
            temp_postings = postings[priority_token]
            del postings[priority_token] # delete it before dumping so it doesn't get dumped twice

            self.write_to_disk(postings, "postings", ptr, index_output_folder)

            postings =  {priority_token:temp_postings} # delete all except current priority token
            ptr+=1
            index[priority_token][1] = ptr # update current priority token's file pointer
            malloc_trim() # free memory on linux

    def print_statistics(self, index_output_folder):
        files = glob.glob(f"./{index_output_folder}/*")
        total_size = sum([os.path.getsize(f) for f in files])

        print("\n\nSTATISTICS:")
        print(f'Total indexing time: {self.statistics["total_indexing_time"]:.2f}s')
        print(f'Merging time: {self.statistics["merging_time"]:.2f}s')
        print(f'Number of temporary index segments written to disk: {self.statistics["temp_index_segments_n"]}')
        print(f'Total index size on disk: {(total_size*1e-6):.1f} MB')
        print(f'Vocabulary size: {self.statistics["vocabulary_size"]}')

    def dump_if_threshold_reached(self, index, postings, i, block_n, index_output_folder):
        ''' dump data to a temporary block.pkl file in disk if memory threshold or postings threshold is reached'''
        used_mem = sys.getsizeof(postings)# in bytes
        if i == self.posting_threshold or used_mem*2 > self.memory_threshold:
            sorted_postings = dict(sorted(postings.items(), key=lambda x: x[0]))
            self.dump_block(sorted_postings, block_n, index_output_folder)

            block_n += 1
            postings = {}
            i = 0
            malloc_trim() # free memory on linux

        return postings, i, block_n

    def dump_block(self, postings, ptr, index_output_folder):
        ''' dump Blocks to a temporary block.pkl file in disk'''
        with open(f"./{index_output_folder}/block{ptr}.pkl", "wb") as f:
            for k,v in postings.items():
                block = Block(token=k, postings=v)
                pickle.dump(block, f)

    def read_next_dict_from_block(self, blocks_reader, n):
        '''iterates the blocks_reader buffer by one block'''
        try:
            return pickle.load(blocks_reader[n]) # advance read buffer by one for selected block
        except EOFError:
            return None
    
    def write_to_disk(self, data, type, filepointer, index_output_folder):
        '''writes index.pkl or posting(s).pkl files to disk'''
        with open(f"./{index_output_folder}/{type}{filepointer}.pkl", "wb") as f:
            pickle.dump(data, f)

    def check_duplicates_in_list(self, lst):
        '''We cannot dump index before we merge the postings for the same token from all the blocks'''
        '''Therefore, first we need to verify if all postings for a token were merged'''
        '''(which means there can't be duplicate tokens in the list of tokens)'''
        return len(lst) != len(set(lst))

    def delete_temp_index_blocks(self, index_output_folder):
        '''deletes all temporary block.pkl files'''
        block_paths = glob.glob(f"./{index_output_folder}/block*.pkl")
        self.statistics["temp_index_segments_n"] = len(block_paths)

        for f in block_paths:
            os.remove(f)

    def update_index_fp(self, index, t, ptr):
        if isinstance(index[t], int):
            df = index[t]
            index[t] = [df, ptr] # add pointer to index
        else:
            index[t][1] = ptr # update pointer

        return index

    def log(self, n):
        if n not in self.logarithm:
            self.logarithm[n] = log10(n)
        return self.logarithm[n]

    def calc_tfidf_weight(self, tf):
        if self.tfidf["smart"].startswith("lnc"):
            l = 1 + self.log(tf)
            w = l # w = l because w = l*n = l*1 = l
        
        return w

    def calc_norm_tfidf_weights(self, pmid, tokens, postings):
        '''Calculate normalized token weights'''
        w_sum = 0
        for t in tokens:
            w = self.calc_tfidf_weight(tf=postings[t][pmid]["w"])
            postings[t][pmid]["w"] = w
            w_sum += w**2

        denominator = sqrt(w_sum)

        for t in tokens: # after calculating sqrt(w_sum) we can store the normalized weight
            postings[t][pmid]["w"] /= denominator

        return postings

    def calc_bm25_weights(self, N, avdl, dl_lens, index, index_output_folder):
        postings_paths = glob.glob(f"./{index_output_folder}/postings*.pkl")

        for f in postings_paths:
            with open(f, "rb") as p: # open
                postings = pickle.load(p)

            for token, v in postings.items(): # calc and store score
                df, _ = extract_data_from_index(token, index, index_output_folder)
                for pmid, dictionary in v.items():
                    tf = dictionary["w"]
                    dl = dl_lens[pmid]
                    postings[token][pmid]["w"] = log10(N/df) * ((self.k1+1)*tf) / (self.k1*((1-self.b)+self.b*dl/avdl)+tf)

            with open(f, "wb") as f: # save
                pickle.dump(postings, f)


class BaseIndex:
    """
    Top-level Index class
    
    This loosly defines a class over the concept of 
    an index.

    """

    def get_tokenizer_kwargs(self):
        """
        Index should store the arguments used to initialize the index as aditional metadata
        """
        return {}

    def add_term(self, term, doc_id, *args, **kwargs):
        raise NotImplementedError()
    
    def print_statistics(self):
        raise NotImplementedError()
    
    @classmethod
    def load_from_disk(cls, path_to_folder:str):
        """
        Loads the index from disk, note that this
        the process may be complex, especially if your index
        cannot be fully loaded. Think of ways to coordinate
        this job and have a top-level abstraction that can
        represent the entire index even without being fully load
        in memory.
        
        Tip: The most important thing is to always know where your
        data is on disk and how to easily access it. Recall that the
        disk access are the slowest operation in a computation device, 
        so they should be minimized.
        
        Parameters
        ----------
        path_to_folder: str
            the folder where the index or indexes are stored.
            
        """
        #with open(f'{path_to_folder}/index.pkl', 'rb') as f:
        #    index = pickle.load(f)
        return cls()

class InvertedIndex(BaseIndex):
    
    # make an efficient implementation of an inverted index
        
    @classmethod
    def load_from_disk(cls, path_to_folder:str):
        raise NotImplementedError()
    
    def print_statistics(self):
        print("Print some stats about this index.. This should be implemented by the base classes")
    
    