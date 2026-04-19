import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
raw_data = pd.read_csv('db_fake_job_postings_project/fake_job_postings.csv')
#вывод общеё информации о датасете
# raw_data.head()
# raw_data.info()
# raw_data.describe()
# raw_data.isnull().sum()
# print(raw_data.head(1))
clear_data = raw_data
# print(clear_data.head(1))
stop_words = stopwords.words('english')
clear_data['title'] = (
    clear_data['title']
    .fillna('')
    .str.lower()
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)
print(clear_data['title'].head(10))
clear_data['company_profile'] = (
    clear_data['company_profile']
    .fillna('')
    .str.lower()
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)
# print(clear_data['company_profile'].head(5))
clear_data['description'] = (
    clear_data['description']
    .fillna('')
    .str.lower()
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)
# print(clear_data['description'].head(5))
clear_data['requirements'] = (
    clear_data['requirements']
    .fillna('')
    .str.lower()
    .replace(rf'[{punctuation}\n]', ' ', regex=True)
    .replace('  ',' ')
    .apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)
# print(clear_data['requirements'].head(5))
clear_data['benefits'] = (
    clear_data['benefits']
    .fillna('')
    .str.lower()
    .replace(rf'[{punctuation}\n]',' ', regex=True)
    .replace('  ',' ')
    .apply( lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
)
# print(clear_data['benefits'].head(5))