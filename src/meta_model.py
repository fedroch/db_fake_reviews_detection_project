import torch
import numpy as np
import gc
import joblib
from pathlib import Path
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from src import bert_classifier
from torch.utils.data import DataLoader
from string import punctuation
import re

MODELS_DIR = Path(__file__).parent.parent / 'models'
model_path = MODELS_DIR / 'bert_finetuned'

def get_bert_emb(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Расчет BERT эмбеддингов"):
            input_ids = batch['input_ids'].to(bert_classifier.DEVICE)
            attention_mask = batch['attention_mask'].to(bert_classifier.DEVICE)
            
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_state = outputs.last_hidden_state[:, 0, :] 
            
            embeddings.append(cls_state.cpu().numpy().astype(np.float32))
            labels.extend(batch['labels'].cpu().numpy())
    return np.vstack(embeddings), np.array(labels)


def load_bert_embeddings(model_path):
    """Загружает сохранённые эмбеддинки из модели с использованием mmap для экономии RAM"""
    train_emb_path = model_path / 'bert_embeddings' /'train_embeddings.npy'
    train_labels_path = model_path / 'bert_embeddings' / 'train_labels.npy'
    test_emb_path = model_path / 'bert_embeddings' / 'test_embeddings.npy'
    test_labels_path = model_path / 'bert_embeddings' / 'test_labels.npy'
    
    if train_emb_path.exists() and test_emb_path.exists():
        print("Загрузка сохранённых эмбеддингов")
        # mmap_mode='r' не грузит файл в RAM сразу
        X_train_bert = np.load(train_emb_path, mmap_mode='r')
        y_train_labels = np.load(train_labels_path)
        X_test_bert = np.load(test_emb_path, mmap_mode='r')
        y_test_labels = np.load(test_labels_path)
        print(f"  Train эмбеддинки: {X_train_bert.shape}")
        print(f"  Test эмбеддинки: {X_test_bert.shape}")
        return X_train_bert, y_train_labels, X_test_bert, y_test_labels
    else:
        return None

def get_custom_features(texts):
    """Докидываем модели экстра признаки (в данном случае длина и кол-во пунктуации)"""
    features = np.zeros((len(texts), 3), dtype=np.float32)
    punc_set = set(punctuation)
    
    for i, text in enumerate(tqdm(texts, desc="Извлечение кастомных признаков")):
        if not isinstance(text, str):
            text = str(text)
        text_len = len(text)
        punctuation_count = sum(1 for char in text if char in punc_set)
        
        features[i, 0] = text_len
        features[i, 1] = punctuation_count
        features[i, 2] = 0 if not text_len else punctuation_count / text_len
    return features

if __name__ == '__main__':
    # Попытка загрузить сохранённые эмбеддинки
    embeddings_result = load_bert_embeddings(model_path)
    if embeddings_result is not None:
        X_train_bert, y_train_labels, X_test_bert, y_test_labels = embeddings_result
    else:
        # Если нет, пересчитываем через BERT
        print("Сохранённые эмбеддинки не найдены, пересчитываем...")
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(bert_classifier.DEVICE)
        model.eval()

        train_dataset = bert_classifier.ReviewDataset(bert_classifier.X_train_raw, bert_classifier.y_train)
        test_dataset  = bert_classifier.ReviewDataset(bert_classifier.X_test_raw,  bert_classifier.y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=bert_classifier.BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            collate_fn=bert_classifier.collate_batch
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=bert_classifier.BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            collate_fn=bert_classifier.collate_batch
        )
        
        X_train_bert, y_train_labels = get_bert_emb(model, train_loader)
        X_test_bert, y_test_labels = get_bert_emb(model, test_loader)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("Создание кастомных признаков")
    X_train_custom = get_custom_features(bert_classifier.X_train_raw)
    X_test_custom = get_custom_features(bert_classifier.X_test_raw)

    del bert_classifier.raw_data
    del bert_classifier.X_train_raw
    del bert_classifier.X_test_raw
    gc.collect()

    print("Объединение признаков")
    # тута создается копия, и оператива бим бим бам бом
    X_train_mixed = np.hstack((X_train_bert, X_train_custom)).astype(np.float32)
    
    X_test_mixed = np.hstack((X_test_bert, X_test_custom)).astype(np.float32)

    print("Масштабирование признаков")
    scaler = StandardScaler(copy=False)
    # чё широкий что-ли
    X_train_scaled = scaler.fit_transform(X_train_mixed)
    
    # Больше не нужно
    del X_train_mixed
    gc.collect()
    
    X_test_scaled = scaler.transform(X_test_mixed)
    del X_test_mixed
    gc.collect()

    print("Обучение мета-модели")
    
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_train_scaled, y_train_labels)

    print("\n─── Результаты Мета-модели ───")
    predictions = meta_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test_labels, predictions):.2%}")
    print(classification_report(y_test_labels, predictions, target_names=['CG', 'OR']))

    joblib.dump({
        'meta_model': meta_model,
        'scaler': scaler
    }, MODELS_DIR / 'meta_model.pkl')
    print(f"Мета-модель (линейная часть и скейлер) сохранена в {MODELS_DIR / 'meta_model.pkl'}")
