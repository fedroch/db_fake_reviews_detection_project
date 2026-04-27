from data_parce import clear_data
from wordcloud import WordCloud
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def wordcloud_visualization(data):
    '''визуализация облака слов'''
    text = ' '.join(data['text_'])
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
    all_words = ' '.join(data[column]).split()
    freq_dist = FreqDist(all_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    freq_dist.plot(30, cumulative=False, ax=ax, show=False)  # show=False отключает авто-plt.show() внутри nltk
    ax.set_title(f'Распределение частот для {column}')
    ax.set_xlabel('Слова')
    ax.set_ylabel('Частота')
    fig.savefig(FIGURES_DIR / f'freq_dist_{column}.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / f'freq_dist_{column}.png'}")
    plt.close(fig)


def plot_text_length_distribution(data, column='text_'):
    '''распределение длин текстов по классам'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    data['text_len'] = data[column].str.split().str.len()

    for ax, cls, color in zip(axes, data['label'].unique(), ['steelblue', 'tomato']):
        subset = data[data['label'] == cls]['text_len']
        ax.hist(subset, bins=40, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(f'Длина текста — класс «{cls}»')
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


def plot_top_words(vectorizer, model, n=15):
    '''топ слов, толкающих к каждому классу (по коэффициентам логрeg)'''
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    top_pos_idx = np.argsort(coef)[-n:]
    top_neg_idx = np.argsort(coef)[:n]
    top_idx = np.concatenate([top_neg_idx, top_pos_idx])

    words = feature_names[top_idx]
    values = coef[top_idx]
    colors = ['steelblue' if v < 0 else 'tomato' for v in values]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(words, values, color=colors, edgecolor='white')
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_title(f'Топ-{n} слов по весу модели')
    ax.set_xlabel('Коэффициент LogisticRegression')
    ax.text(values.min() * 0.5, n - 0.5, '← оригинал (OR)',
            fontsize=9, color='steelblue', va='center')
    ax.text(values.max() * 0.1, n + 0.5, '→ фейк (CG)',
            fontsize=9, color='tomato', va='center')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'top_words.png', dpi=150, bbox_inches='tight')
    print(f"График сохранён: {FIGURES_DIR / 'top_words.png'}")
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

if __name__ == '__main__':
    wordcloud_visualization(clear_data)
    frequency_distribution(clear_data, 'text_')
    plot_histogram(clear_data)
    plot_bar_chart(clear_data, 'label')
    plot_text_length_distribution(clear_data)