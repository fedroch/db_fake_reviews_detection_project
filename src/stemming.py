from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
def word_stem(word: str):
    '''стемминг отдельных слов'''
    stemmer = PorterStemmer()
    return stemmer.stem(word)