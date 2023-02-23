"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: 

Tokenizer module

Holds the code/logic addressing the Tokenizer class
and implemetns logic in how to process text into
tokens.

"""

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import re
from utils import dynamically_init_class


def dynamically_init_tokenizer(**kwargs):
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
    

class Tokenizer:
    """
    Top-level Tokenizer class
    
    This loosly defines a class over the concept of 
    an index.

    """
    def __init__(self, **kwargs):
        super().__init__()
    
    def tokenize(self, text):
        """
        Tokenizes a piece of text, this should be
        implemented by specific Tokenizer sub-classes.
        
        Parameters
        ----------
        text : str
            Sequence of text to be tokenized
            
        Returns
        ----------
        object
            An object that represent the output of the
            tokenization, yet to be defined by the students
        """
        raise NotImplementedError()

        
class PubMedTokenizer(Tokenizer):
    """
    An example of subclass that represents
    a special tokenizer responsible for the
    tokenization of articles from the PubMed.

    """
    def __init__(self, minL, stopwords_path, stemmer, case_folding, allow_numbers, *args, **kwargs):
        super().__init__(**kwargs)
        self.minL = minL
        self.stopwords_path = stopwords_path
        self.case_folding = case_folding
        self.allow_numbers = allow_numbers
        self.regex_pattern = re.compile(r'\W')

        if stemmer == None:
            self.stemmer = None
        elif ("snow") in stemmer.lower():
            self.stemmer = SnowballStemmer(language='english')
        else:
            self.stemmer = PorterStemmer()


        with open(self.stopwords_path, 'r') as f:
            stop_words = f.readlines()

        self.stop_words = set(stop_words)


        print("init PubMedTokenizer|", f"{minL=}, {stopwords_path=}, {stemmer=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def tokenize(self, text: str):
        tokens = []

        for word in re.split(self.regex_pattern, text):
            if self.minL and len(word) < self.minL:
                continue

            if word.isnumeric() and not self.allow_numbers:
                continue

            if self.case_folding:
                word = word.lower()

            if word in self.stop_words:
                continue
            
            if self.stemmer:
                word = self.stemmer.stem(word)

            tokens.append(word)

        return tokens
