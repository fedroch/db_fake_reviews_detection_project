from src.data_parce import clear_data, raw_data
from wordcloud import WordCloud
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from joblib import load

FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def wordcloud_visualization(data):
    '''визуализация облака слов'''
    text = ' '.join(data['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Облако слов')
    fig.savefig(FIGURES_DIR / 'wordcloud.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'wordcloud.png'}")
    plt.close(fig)


def plot_histogram(data):
    '''визуализация гистограммы для числовых данных'''
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data['rating'], bins=30, color='blue', edgecolor='black')
    ax.set_title('Гистограмма для rating')
    ax.set_xlabel('rating')
    ax.set_ylabel('Частота')
    ax.grid(axis='y', alpha=0.75)
    fig.savefig(FIGURES_DIR / 'histogram_rating.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'histogram_rating.png'}")
    plt.close(fig)


def plot_bar_chart(data, column):
    '''визуализация гистограммы для категориальных данных'''
    fig, ax = plt.subplots(figsize=(10, 5))
    data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black', ax=ax)
    ax.set_title(f'Гистограмма для {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Частота')
    ax.grid(axis='y', alpha=0.75)
    fig.savefig(FIGURES_DIR / f'bar_{column}.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / f'bar_{column}.png'}")
    plt.close(fig)


def frequency_distribution(data, column):
    '''визуализация распределения частот для текстовых данных'''
    from nltk import FreqDist
    all_words = ' '.join(data[column].dropna()).split()
    freq_dist = FreqDist(all_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    freq_dist.plot(30, cumulative=False,show=False)  # show=False отключает авто-plt.show() внутри nltk
    ax.set_title(f'Распределение частот для {column}')
    ax.set_xlabel('Слова')
    ax.set_ylabel('Частота')
    fig.savefig(FIGURES_DIR / f'freq_dist_{column}.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / f'freq_dist_{column}.png'}")
    plt.close(fig)


def plot_text_length_distribution(data, column='text'):
    '''распределение длин текстов по классам'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    data['text_len'] = data[column].str.split().str.len()

    for ax, cls, color in zip(axes, data['label'].unique(), ['steelblue', 'tomato']):
        subset = data[data['label'] == cls]['text_len']
        ax.hist(subset, bins=40, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlim(0, 500)
        cls_b = {0: 'Fake (CG)', 1: 'Real (OR)'}
        ax.set_title(f'Длина текста — класс «{cls_b[cls]}»')
        ax.set_xlabel('Количество слов')
        ax.set_ylabel('Частота')
        ax.grid(axis='y', alpha=0.4)
        ax.axvline(subset.mean(), color='black', linestyle='--', linewidth=1,
                   label=f'среднее: {subset.mean():.0f}')
        ax.legend()

    fig.suptitle('Распределение длин текстов по классам', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'text_length_distribution.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'text_length_distribution.png'}")
    plt.close(fig)


#  Оценка модели

def plot_confusion_matrix(y_test, predictions, class_names=None):
    '''матрица ошибок модели'''
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_test, predictions)
    labels = class_names if class_names else sorted(set(y_test))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Реально')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'confusion_matrix.png'}")
    plt.close(fig)


def plot_top_words_per_category(vectorizer, model, category_texts, category_name, n=15, model_name='LogisticRegression'):
    '''топ слов, толкающих к каждому классу с учетом специфики категории'''
    feature_names = vectorizer.get_feature_names_out()
    global_coef = model.coef_[0]
    X_cat = vectorizer.transform(category_texts)
    mean_tfidf = np.asarray(X_cat.mean(axis=0)).flatten()
    category_importance = global_coef * mean_tfidf

    # 4. Выбираем топ-n слов (отрицательные -> оригинал, положительные -> фейк)
    top_pos_idx = np.argsort(category_importance)[-n:]
    top_neg_idx = np.argsort(category_importance)[:n]
    top_idx = np.concatenate([top_neg_idx, top_pos_idx])
    words = feature_names[top_idx]
    values = category_importance[top_idx]
    colors = ['steelblue' if v < 0 else 'tomato' for v in values]
    # Отрисовка
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(words, values, color=colors, edgecolor='white')
    ax.axvline(0, color='gray', linewidth=0.8)
    # Обновляем заголовки, чтобы было видно категорию
    ax.set_title(f'Топ-{n} слов для категории: {category_name}')
    ax.set_xlabel(f'Вклад слова (Вес {model_name} × Частота в категории)')
    
    # Немного сдвинем текст, чтобы он не наезжал на бары
    ax.text(values.min() * 0.9, n - 0.5, '← оригинал (OR)',
            fontsize=10, color='steelblue', va='center')
    ax.text(values.max() * 0.1, n + 0.5, '→ фейк (CG)',
            fontsize=10, color='tomato', va='center')
            
    fig.tight_layout()
    
    # Сохраняем с именем категории
    save_dir = FIGURES_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True) # На всякий случай создаем папку
    save_path = save_dir / f'top_words_{category_name}.png'
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"График для {category_name} сохранён: {save_path}")
    plt.close(fig)


def plot_metrics_by_class(y_test, predictions):
    '''precision / recall / f1 по каждому классу'''
    from sklearn.metrics import classification_report

    report = classification_report(y_test, predictions, output_dict=True)
    classes = [k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind='bar', colormap='tab10', edgecolor='white', rot=0, ax=ax)
    ax.set_ylim(0, 1.1)
    ax.set_title('Метрики по классам')
    ax.set_xlabel('Класс')
    ax.set_ylabel('Значение')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.4)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'metrics_by_class.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'metrics_by_class.png'}")
    plt.close(fig)


def plot_proba_histogram(model, X_test, y_test):
    '''распределение вероятностей предсказаний по классам'''
    proba = model.predict_proba(X_test)
    classes = model.classes_
    colors = ['steelblue', 'tomato', '#7B5EA7', '#2A9D8F']

    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 4), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    for i, (cls, ax) in enumerate(zip(classes, axes)):
        ax.hist(proba[:, i], bins=40, color=colors[i % len(colors)],
                alpha=0.75, edgecolor='white', linewidth=0.4)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2, label='порог 0.5')
        ax.axvspan(0.4, 0.6, alpha=0.08, color='red', label='зона сомнений')
        ax.set_title(f'Класс «{cls}»')
        ax.set_xlabel('P(класс)')
        ax.set_xlim(0, 1)
        ax.grid(axis='y', alpha=0.4)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Количество примеров')
    fig.suptitle('Уверенность модели по классам', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'proba_histogram.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'proba_histogram.png'}")
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test):
    '''ROC-кривая с AUC'''
    from sklearn.metrics import roc_curve, auc

    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba, pos_label=model.classes_[1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-кривая')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'roc_curve.png'}")
    plt.close(fig)

def punctuation_distribution(data_raw):
    '''Функция для анализа плотности знаков препинания по классам'''
    from string import punctuation
    import seaborn as sns
    import re
    import matplotlib.pyplot as plt
    df = data_raw.copy()
    label_map = {0: 'Fake (CG)', 1: 'Real (OR)'}
    df['label_name'] = df['label'].map(label_map)
    punc_pattern = f"[{re.escape(punctuation)}]"
    df['text_punc'] = df['text'].str.count(punc_pattern).fillna(0)
    df['text_len'] = df['text'].str.len().replace(0, 1)
    df['punc_density'] = (df['text_punc'] / df['text_len']) * 100
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # Левый график
    sns.barplot(
        data=df, x='label_name', y='punc_density', 
        palette=['#E74C3C', '#3498DB'], ax=ax[0], capsize=.1
    )
    ax[0].set_title('Средняя плотность знаков (%)', fontsize=14)
    ax[0].set_ylabel('Процент пунктуации в тексте')
    ax[0].set_xlabel('Тип отзыва')
    # Правый график:(распределение)
    sns.kdeplot(
        data=df, x='punc_density', hue='label_name', 
        fill=True, palette=['#E74C3C', '#3498DB'], ax=ax[1], common_norm=False
    )
    ax[1].set_title('Распределение плотности знаков', fontsize=14)
    ax[1].set_xlabel('Плотность (%)')
    ax[1].set_ylabel('Плотность вероятности')
    upper_limit = df['punc_density'].quantile(0.99)
    ax[1].set_xlim(0, upper_limit)
    fig.tight_layout()
    save_path = FIGURES_DIR / 'text_punctuation_distribution_v2.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"график сохранён: {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    wordcloud_visualization(clear_data.sample(500000, random_state=42))
    frequency_distribution(raw_data, 'text')
    plot_histogram(clear_data)
    plot_bar_chart(clear_data, 'label')
    plot_text_length_distribution(clear_data)
    punctuation_distribution(raw_data)
    categories = raw_data['category'].dropna().unique()
    from sklearn.linear_model import LogisticRegression # свинина на pickle требует импортировать все классы, которые были в модели при сохранении
    model_data = load(Path(__file__).parent.parent / 'models' / 'bag_of_words.pkl')
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    for cat in categories:
        cat_texts = clear_data[clear_data['category'] == cat]['text'].sample(min(20000, len(clear_data[clear_data['category'] == cat])), random_state=42).tolist()
        plot_top_words_per_category(vectorizer, model, cat_texts, category_name=cat)
