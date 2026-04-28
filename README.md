<div align="center">

# 🔍 FRDD (fake_reviews_detection_project)

### Автоматическое выявление поддельных отзывов методами NLP и машинного обучения

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![sklearn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-154F5B?style=for-the-badge&logo=python&logoColor=white)](https://nltk.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)

[![Status](https://img.shields.io/badge/Статус-В%20разработке-yellow?style=flat-square)]()
[![License](https://img.shields.io/badge/Лицензия-MIT-green?style=flat-square)]()

</div>

---

##  О проекте

Поддельные отзывы искажают рейтинги товаров и услуг, вводят покупателей в заблуждение и подрывают доверие к платформам. Данный проект решает задачу **бинарной классификации отзывов** — автоматически отличает настоящие отзывы от сгенерированных/поддельных.


## Структура проекта

---

---

##  Данные

**Источник:** [Fake Reviews Dataset — Kaggle](https://www.kaggle.com)

| Параметр | Значение |
|---|---|
| Метки | `OR` — оригинальный · `CG` — сгенерированный |
| Тип данных | Текст отзыва + рейтинг (1–5 ★) |
| Формат | CSV |
| Язык | Английский |

---

## Пайплайн

```
Сырые данные
    │
    ▼
┌─────────────────────────┐
│  EDA & Анализ           │  Распределение классов, длины текстов,
│                         │  стоп-слова, n-gram анализ
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Предобработка текста   │  Лемматизация (nltk), нижний регистр,
│                         │  удаление стоп-слов и пунктуации
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Feature Engineering    │  TF-IDF (uni/bigrams) + структурные
│                         │  признаки (длина, !, прилагательные)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Обучение моделей       │  Logistic Regression · Naive Bayes
│                         │  Random Forest · (опц.) LSTM / BERT
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Оценка и интерпретация │  F1-score · ROC-AUC · Feature Importance
│                         │  Топ-слова фейковых отзывов
└─────────────────────────┘
```

---

##  Модели

### Классические методы

| Модель | Описание |
|---|---|
| **Logistic Regression** | Быстрый интерпретируемый baseline |
| **Naive Bayes** | Эффективен на TF-IDF признаках |
| **Random Forest** | Устойчивость к шуму, feature importance |

### Deep Learning (возможно будет)

| Модель | Описание |
|---|---|
| **LSTM** | Учёт последовательности слов |
| **BERT (fine-tune)** | SOTA для текстовой классификации |

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/fedroch/db_fake_reviews_detection_project/.git
cd db_fake_reviews_detection_project
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

## Стек технологий

```python
# Данные
pandas, numpy

# NLP
nltk                  # Токенизация, лемматизация, стоп-слова
scikit-learn          # TF-IDF, CountVectorizer

# Моделирование
scikit-learn          # LogReg, NaiveBayes, RandomForest
imbalanced-learn      # SMOTE, class_weight

# Deep Learning (опц.)
tensorflow / pytorch  # LSTM, BERT fine-tuning

# Визуализация
matplotlib, seaborn
wordcloud
```

---

## Команда

| Участник | Роль |
|---|---|
| Алексей Василев | ML-инженер · обучение моделей, метрики |
| Федор Чечик | NLP / EDA · предобработка, TF-IDF |

---

## Roadmap

- [x] Постановка задачи и выбор датасета
- [x] Описание пайплайна и признаков
- [ ] EDA + анализ дисбаланса классов
- [x] Предобработка текста
- [ ] Feature engineering (TF-IDF + структурные)
- [ ] Baseline-модели (LogReg, NaiveBayes)
- [ ] Random Forest + оценка метрик
- [ ] Интерпретация: топ-слова фейков
- [ ] (опц.) LSTM / BERT fine-tuning
- [ ] Итоговый отчёт и выводы

---
