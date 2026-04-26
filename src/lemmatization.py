import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import spacy

def lemmatize_text_no_POS(word: str):
    '''лемматизация текста(по словам) без указания части речи'''
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def get_wordnet_pos(nltk_tag: str):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text_with_POS_nltk(text: str):
    '''лемматизация текста с указанием части речи через nltk'''
    lemmatizer = WordNetLemmatizer()
    words_tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(words_tokens)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in pos_tags])
    return lemmatized_text

def lemmatize_text_with_POS_spacy(texts: list):
    '''лемматизация текста с указанием части речи с помощью spacy'''
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmatized_texts = []
    for doc in nlp.pipe(texts, batch_size=500, n_process=-1):
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        lemmatized_texts.append(" ".join(lemmas))
        
    return lemmatized_texts
