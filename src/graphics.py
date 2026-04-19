from src.data_parce import clear_data
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def wordcloud_visualization(data):
    text = ' '.join(data['text_'])
    '''визуализация облака слов'''
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

def plot_histogram(data):
    '''визуализация гистограммы для числовых данных'''
    plt.figure(figsize=(10, 5))
    plt.hist(data['rating'], bins=30, color='blue', edgecolor='black')
    plt.title('Гистограмма для rating')
    plt.xlabel('rating')
    plt.ylabel('Частота')
    plt.grid(axis='y', alpha=0.75)

def plot_bar_chart(data, column):
    '''визуализация гистограммы для категориальных данных'''
    plt.figure(figsize=(10, 5))
    data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black')
    plt.title(f'Гистограмма для {column}')
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.grid(axis='y', alpha=0.75)

def frequency_distribution(data, column):
    '''визуализация распределения частот для текстовых данных'''
    from nltk import FreqDist
    from nltk.tokenize import word_tokenize
    all_words = ' '.join(data[column]).split()
    freq_dist = FreqDist(all_words)
    plt.figure(figsize=(10, 5))
    freq_dist.plot(30, cumulative=False)
    plt.title(f'Распределение частот для {column}')
    plt.xlabel('Слова')
    plt.ylabel('Частота')
wordcloud_visualization(clear_data)
frequency_distribution(clear_data, 'text_')
plot_histogram(clear_data)
plot_bar_chart(clear_data, 'label')

plt.show()