from src.data_parce import clear_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # по сравнению с bag_of_words поменялось только это (на удивление работает хуже)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(
    lowercase=False, # текст уже распарщен, так что все итак в нижнем регистре
    token_pattern=r'\b\w+\b', # по дефолту удаляет одиночные символы (тут они остаются)
    # min_df=5, # убираем слова встреченные <5 раз
    sublinear_tf=True, # по сравнению с bag_of_words поменялось только это
    ngram_range=(1, 2), # ищем пары слов и одиночные
    max_features=5000 # оставляем 5000 самых частотных слов (ало видюха, иди нахуй)
)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    clear_data['text_'], 
    clear_data['label'], 
    test_size=0.2, 
    random_state=42
)

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

# Предсказываем на тех данных, которые модель еще не видела
predictions = logistic_model.predict(X_test)

print(f"Общая точность (Accuracy): {accuracy_score(y_test, predictions):.2%}")
print("\nПодробный отчет:")
print(classification_report(y_test, predictions))