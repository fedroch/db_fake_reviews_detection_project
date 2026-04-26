import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction import text
from string import punctuation, digits
from src.lemmatization import lemmatize_text_no_POS # потом нужно будет сделать функцию с POS и протестировать, будуи ли отличия
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

raw_data = pd.read_csv('db_fake_reviews_detection_project/data/raw/fake_reviews_dataset.csv')

clear_data = raw_data
stop_words = text.ENGLISH_STOP_WORDS
added_stopwords = set(open('db_fake_reviews_detection_project/stopwords/added_stopwords.txt', 'r').readlines()) # менять можно в файлике
excluded_stopwords = set(open('db_fake_reviews_detection_project/stopwords/excluded_stopwords.txt', 'r').readlines()) # менять можно в файлике
stop_words = stop_words.union(added_stopwords).difference(excluded_stopwords)
clear_data['text_'] = (
    clear_data['text_']
    .str.lower()
    .replace(rf'[{punctuation}{digits}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([lemmatize_text_no_POS(word) for word in word_tokenize(x) if word not in stop_words])) #с лемматизацией по отдельным словам
)


# print_base_info(clear_data)

# print('\n//////////////////////////////////////////////////////////\n')

# print_col(5, clear_data, ['text_'])