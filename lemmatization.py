from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def lemmatize_text_no_POS(word: str):
    '''лемматизация текста(по словам) без указания части речи'''
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)