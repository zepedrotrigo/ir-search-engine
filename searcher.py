import pickle, os, itertools, math
from utils import dynamically_init_class, extract_data_from_index
from math import sqrt, log10


def dynamically_init_searcher(**kwargs):
    """Dynamically initializes a Tokenizer object from this
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

class BaseSearcher:

    def search(self, index, query_tokens, top_k):
        pass

    def batch_search(self, index, reader, tokenizer, output_file, top_k=1000):
        print("searching...")
        # loop that reads the questions

            # aplies the tokenization to get the query_tokens
        query_tokens = []
        results = self.search(index, query_tokens, top_k)

        # write results to disk

class TFIDFRanking(BaseSearcher):

    def __init__(self, smart, **kwargs) -> None:
        super().__init__(**kwargs)
        self.smart = smart
        self.logarithm = {} # store pre-calculated logarithms to fetch them later

        print("init TFIDFRanking|", f"{smart=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def search(self, tokenizer, index_folder, top_k, reader):
        with open("document_N.txt", "r") as f:
            N = int(f.readline())
        index = load_from_disk(index_folder, filename="index.pkl")

        for question in reader.read():
            print(question)
            tokens = [t for t in tokenizer.tokenize(question["query_text"]) if t in index]
            query_weights = self.calc_normalized_weights(N, tokens, index, index_folder)
            ranked_results = ranked_retrieval(index, tokens, query_weights, top_k, index_folder)
            precision, recall, average_precision = calculate_precision_and_recall(ranked_results, question["documents_pmid"], k = 10)
            f_measure = calculate_fmeasure(precision, recall)
            print(f'\nPrecision -> {precision}')
            print(f'Recall -> {recall}')
            print(f'Avg-Precision -> {average_precision}')
            print(f'F-Measure -> {f_measure}\n')
            display_results(ranked_results)


    def log(self, n):
        if n not in self.logarithm:
            self.logarithm[n] = log10(n)
        return self.logarithm[n]

    def calc_tfidf_weight(self, tf, N, df):
        if self.smart == "lnc.ltc":
            l = 1 + self.log(tf)
            t = self.log(N/df)
        elif self.smart == "lnc.npc":
            l = tf
            x = (N-df)/df

            if x >= 1:
                t = self.log(x)
            else:
                t = 0.0000000001 # avoid division by zero error

        w = l*t
        return w

    def calc_normalized_weights(self, N, tokens, index, index_folder):
        '''Calculate normalized token weights'''
        weights = {}
        w_sum = 0
        for t in tokens:
            df, _ = extract_data_from_index(t, index, index_folder)
            tf = tokens.count(t)
            w = self.calc_tfidf_weight(tf, N, df)
            weights[t] = w
            w_sum += w**2

        denominator = sqrt(w_sum)

        for k in weights: # after calculating sqrt(w_sum) we can store the normalized weight
            weights[k] /= denominator

        return weights


class BM25Ranking(BaseSearcher):

    def __init__(self, k1, b, **kwargs) -> None:
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        print("init BM25Ranking|", f"{k1=}", f"{b=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def search(self, tokenizer, index_folder, top_k, reader):
        with open("document_N.txt", "r") as f:
            N = int(f.readline())
        index = load_from_disk(index_folder, filename="index.pkl")
        
        for question in reader.read():
            print(question)
            tokens = [t for t in tokenizer.tokenize(question["query_text"]) if t in index]
            query_weights = self.calc_weights(N, tokens, index, index_folder)
            ranked_results = ranked_retrieval(index, tokens, query_weights, top_k, index_folder)
            precision, recall, average_precision = calculate_precision_and_recall(ranked_results, question["documents_pmid"], k = 10)
            f_measure = calculate_fmeasure(precision, recall)
            print(f'\nPrecision -> {precision}')
            print(f'Recall -> {recall}')
            print(f'Avg-Precision -> {average_precision}')
            print(f'F-Measure -> {f_measure}\n')
            display_results(ranked_results)
            

    
    def calc_weights(self, N, tokens, index, index_folder):
        weights = {}

        for t in tokens:
            tf = tokens.count(t)
            df, _ = extract_data_from_index(t, index, index_folder)
            w = log10(N/df) * (((self.k1+1)*tf) / (self.k1*tf))
            weights[t] = w

        return weights

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #

def load_from_disk(index_folder, filename):
    '''loads index/postings files from disk'''
    with open(f'{index_folder}/{filename}', 'rb') as f:
        index = pickle.load(f)
    
    return index


def clear():
    '''clears terminal'''
    os.system('cls' if os.name == 'nt' else 'clear')


def ranked_retrieval(index, search_tokens, query_weights, top_k, index_folder):
    documents = {}
    prev_fp = None

    for t in search_tokens:
        _, fp = extract_data_from_index(t, index, index_folder)

        if prev_fp != fp:
            postings = load_from_disk(index_folder, filename=f"postings{fp}.pkl")
            prev_fp = fp

        for pmid, dictionary in postings[t].items():
            wt = dictionary["w"]
            score = query_weights[t] * wt
            if pmid in documents:
                documents[pmid]["score"] += score
                documents[pmid]["num_search_terms"] += 1
                documents[pmid]["token_positions"].append(postings[t][pmid]["positions"])
            else:
                documents[pmid] = {
                    "score": score,
                    "num_search_terms": 1,
                    "token_positions": [postings[t][pmid]["positions"]]
                    }


    # -------- boost the scores of documents using the minimum window size ------- #
    num_distinct_terms = len(set(search_tokens))
    for pmid, doc_data in documents.items():
        if num_distinct_terms == len(doc_data["token_positions"]):
            min_window_size = find_min_window_size(doc_data["token_positions"])
            boost = boost_factor(min_window_size, num_distinct_terms)
        else:
            boost = 1

    sorted_top_k_scores = sorted(documents.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]
    return sorted_top_k_scores


def display_results(results):
    results_per_page = 10
    for i in range(0,len(results),results_per_page):
        for j in range(results_per_page):
            print(f"{j+1:>2}. PMID {results[i+j][0]} (score: {results[i+j][1]})")

        cmd = input(f"\n\nPress [ENTER] to Show more results\nWrite 'n' for New query\n\n-> ")
        clear()

        if cmd != "":
            break


def exponential_decay(x, A, lambd):
    return A * math.exp(-lambd * x)


def boost_factor(min_window_size, num_distinct_terms, max_boost=2.0, lambd=0.007):
    if min_window_size == num_distinct_terms:
        return max_boost
    else:
        return max(1, exponential_decay(min_window_size, max_boost, lambd))


def find_min_window_size(token_positions):
    min_window_size = float("inf")

    # Iterate over all combinations of occurrences of the tokens
    for indices in itertools.product(*token_positions):
        window_size = max(indices) - min(indices)
        min_window_size = min(min_window_size, window_size)
    
    return min_window_size

def calculate_precision_and_recall(ranked_results, relevant_results, k):
    # Precision and Recal vars
    tp = 0
    fp = 0
    fn = 0

    # Average Precision vars
    average_sum = 0
    current_precision = 1

    for i in range(k):
        result = ranked_results[i]
        if str(result[0]) in relevant_results:
            tp += 1
            if current_precision != 1:
                current_precision += 1/(i+1)
            average_sum += current_precision
        else:
            fp += 1
            current_precision -= 1/(i+1)
    
    if tp != len(relevant_results):
        fn = len(relevant_results) - tp
    
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    average_precision = average_sum/len(relevant_results)
    
    return precision, recall, average_precision

def calculate_fmeasure(precision, recall):
    return (2*recall*precision)/(recall + precision)