import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction import text
from string import punctuation

def print_base_info(df):
    """вывод общей информации о датасете"""
    print("head:")
    print(df.head())
    print("info:")
    df.info()
    print("describe:")
    df.describe()
    print('кол-во пропусков в каждом столбце:')
    print(df.isnull().sum())
    print('первая строка:')
    print(df.head(1))

def print_col(cnt: int, df, rows: list):
    '''вывод первых cnt строк датафрейма из столбцов rows'''
    print(df[rows].head(cnt))

pd.set_option('display.max_columns', None) # добавлено только потому что у меня на экран все не влезает надо не забыть убрать
raw_data = pd.read_csv('db_fake_job_postings_project/fake_job_postings.csv')

clear_data = raw_data
# print(clear_data.head(1))
stop_words = text.ENGLISH_STOP_WORDS
added_stopwords = set(open('db_fake_job_postings_project/stopwords/added_stopwords.txt', 'r').readlines()) # менять можно в файлике
excluded_stopwords = set(open('db_fake_job_postings_project/stopwords/excluded_stopwords.txt', 'r').readlines()) # менять можно в файлике
stop_words = stop_words.union(added_stopwords).difference(excluded_stopwords)
clear_data['title'] = (
    clear_data['title']
    .fillna('')
    .str.lower()
    .replace('C++', 'cpp').replace('C#', 'Csharp').replace('.NET', 'dotnet')
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)

clear_data['company_profile'] = (
    clear_data['company_profile']
    .fillna('')
    .str.lower()
    .replace('C++', 'cpp').replace('C#', 'Csharp').replace('.NET', 'dotnet')
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)

clear_data['description'] = (
    clear_data['description']
    .fillna('')
    .str.lower()
    .replace('C++', 'cpp').replace('C#', 'Csharp').replace('.NET', 'dotnet')
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)

clear_data['requirements'] = (
    clear_data['requirements']
    .fillna('')
    .str.lower()
    .replace('C++', 'cpp').replace('C#', 'Csharp').replace('.NET', 'dotnet')
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)

clear_data['benefits'] = (
    clear_data['benefits']
    .fillna('')
    .str.lower()
    .replace('C++', 'cpp').replace('C#', 'Csharp').replace('.NET', 'dotnet')
    .replace(rf'[{punctuation}\n]',' ', regex=True)
    .replace('  ',' ')
    .apply( lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)

# print_base_info(clear_data)

# print('//////////////////////////////////////////////////////////')

# print_col(5, clear_data, ['title', 'company_profile', 'description', 'requirements', 'benefits'])