import os
from pathlib import Path
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction import text
from string import punctuation, digits
from src.lemmatization import lemmatize_text_no_POS, lemmatize_text_with_POS_nltk, lemmatize_text_with_POS_spacy
from src.stemming import word_stem

def print_base_info(df):
    """вывод общей информации о датасете"""
    print("head:")
    print(df.head())
    print("info:")
    df.info()
    print("describe:")
    df.describe()
    print('первая строка:')
    print(df.head(1))

def print_col(cnt: int, df, rows: list):
    '''вывод первых cnt строк датафрейма из столбцов rows'''
    print(df[rows].head(cnt))   

script_dir = Path(__file__).parent.parent
raw_data = pd.read_csv(script_dir / 'data/raw/fake reviews dataset.csv')

clear_data = raw_data.copy()
stop_words = text.ENGLISH_STOP_WORDS
added_stopwords = set([word.rstrip() for word in open(script_dir / 'stopwords/added_stopwords.txt', 'r').readlines()]) # менять можно в файлике
excluded_stopwords = set([word.rstrip() for word in open(script_dir / 'stopwords/excluded_stopwords.txt', 'r').readlines()]) # менять можно в файлике
stop_words = stop_words.union(added_stopwords).difference(excluded_stopwords)
# clear_data['text_'] = (
#     clear_data['text_']
#     .str.lower() # почему-то без приведения к нижнему регистру скор модели больше
#     .replace(rf'[{punctuation}{digits}\n]', ' ', regex=True)
#     .replace('  ',' ')
#     .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
#     # .apply(lambda x: ' '.join([lemmatize_text_no_POS(word) for word in word_tokenize(x) if word not in stop_words])) #с лемматизацией по отдельным словам
#     # .apply(lambda x: ' '.join([word_stem(word) for word in word_tokenize(x) if word not in stop_words])) стемминг вместо лемматизации
# )
clear_data['text_'] = clear_data['text_'].apply(lemmatize_text_with_POS_nltk) # лемматизация текста с учетом контекста через nltk (пока пока проц, но скор самы лучший)
# clear_data['text_'] = lemmatize_text_with_POS_spacy(clear_data['text_'].tolist()) # лемматизация текста с учетом контекста через spacy (пока пока проц и скор хуже чем с nltk ещё и кучу оперативы жрет, но я хотя бы попробовал + оно чут чут быстрее)
