from src.data_parce import clear_data
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

vectorizer = CountVectorizer(
    lowercase=False, # текст уже распарщен, так что все итак в нижнем регистре
    token_pattern=r'\b\w+\b', # хз чё это но надо
    # min_df=5, # убираем слова встреченные <5 раз
    ngram_range=(1, 2), # ищем пары слов и одиночные
    max_features=5000 # оставляем 5000 самых частотных слов (ало проц ...(цензура))
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

predictions = logistic_model.predict(X_test)

joblib.dump({'model': logistic_model, 'vectorizer': vectorizer},
            MODELS_DIR / 'bag_of_words.pkl')
print(f"Модель сохранена в {MODELS_DIR / 'bag_of_words.pkl'}")

print(f"Общая точность (Accuracy): {accuracy_score(y_test, predictions):.2%}")
print("\nПодробный отчет:")
print(classification_report(y_test, predictions))


from src.graphics import (
    plot_confusion_matrix,
    plot_top_words,
    plot_metrics_by_class,
    plot_proba_histogram,
    plot_roc_curve,
)

plot_confusion_matrix(y_test, predictions)
plot_top_words(vectorizer, logistic_model, n=15)
plot_metrics_by_class(y_test, predictions)
plot_proba_histogram(logistic_model, X_test, y_test)
plot_roc_curve(logistic_model, X_test, y_test)

plt.show()