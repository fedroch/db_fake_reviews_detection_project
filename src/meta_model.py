import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from src.bert_classifier import (
    ReviewDataset,
    MAX_LEN,
    BATCH_SIZE,
    DEVICE,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    label2id,
    id2label,
    collate_batch
)
from torch.utils.data import DataLoader
from string import punctuation
import re

MODELS_DIR = Path(__file__).parent.parent / 'models'
model_path = MODELS_DIR / 'bert_finetuned'

def get_bert_emb(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_state = outputs.last_hidden_state[:, 0, :] 
            
            embeddings.append(cls_state.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    return np.vstack(embeddings), np.array(labels)


def load_bert_embeddings(model_path):
    """Загружает сохранённые эмбеддинки из модели"""
    train_emb_path = model_path / 'train_embeddings.npy'
    train_labels_path = model_path / 'train_labels.npy'
    test_emb_path = model_path / 'test_embeddings.npy'
    test_labels_path = model_path / 'test_labels.npy'
    
    if train_emb_path.exists() and test_emb_path.exists():
        print("Загрузка сохранённых эмбеддингов...")
        X_train_bert = np.load(train_emb_path)
        y_train_labels = np.load(train_labels_path)
        X_test_bert = np.load(test_emb_path)
        y_test_labels = np.load(test_labels_path)
        print(f"  Train эмбеддинки: {X_train_bert.shape}")
        print(f"  Test эмбеддинки: {X_test_bert.shape}")
        return X_train_bert, y_train_labels, X_test_bert, y_test_labels
    else:
        return None

def get_custom_features(texts):
    """Докидываем модели экстра признаки (в данном случае длина и кол-во пунктуации)"""
    features =[]
    for text in texts:
        text_len = len(text)
        punctuation_count = sum(text.count(p) for p in punctuation)
        punc_desity = punctuation_count / text_len
        features.append([text_len, punctuation_count, punc_desity])
    return np.array(features)

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
        model.to(DEVICE)
        model.eval()

        train_dataset = ReviewDataset(X_train_raw, y_train)
        test_dataset  = ReviewDataset(X_test_raw,  y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            collate_fn=collate_batch
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            collate_fn=collate_batch
        )
        print("Расчет кастомных признаков...")
        X_train_bert, y_train_labels = get_bert_emb(model, train_loader)
        X_test_bert, y_test_labels = get_bert_emb(model, test_loader)

    X_train_custom = get_custom_features(X_train_raw)
    X_test_custom = get_custom_features(X_test_raw)

    X_train_mixed = np.hstack((X_train_bert, X_train_custom))
    X_test_mixed = np.hstack((X_test_bert, X_test_custom))

    # Чё сильно широкий чтоли?
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_mixed)
    X_test_scaled = scaler.transform(X_test_mixed)

    # мета-модель
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_train_scaled, y_train_labels)

    print("\n─── Результаты Мета-модели ───")
    predictions = meta_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test_labels, predictions):.2%}")
    print(classification_report(y_test_labels, predictions, target_names=['CG', 'OR']))
