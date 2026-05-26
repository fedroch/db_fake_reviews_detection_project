from src import meta_model, bert_classifier
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from torch.utils.data import DataLoader

def predict_meta(texts, bert_model, logistic_model, scaler):
    bert_model.eval()
    with torch.no_grad():
        # Лейблы не роляют если что
        inference_dataset = bert_classifier.ReviewDataset(texts, [0] * len(texts))
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=bert_classifier.BATCH_SIZE,
            shuffle=False,
            pin_memory=False, # важно, потому шо в collate_batch все уже отправлено на девайс
            num_workers=0,
            collate_fn=bert_classifier.collate_batch
        )
        bert_embeddings, _ = meta_model.get_bert_emb(bert_model, inference_loader)
        custom_features = meta_model.get_custom_features(texts)
        X_meta = np.hstack((bert_embeddings, custom_features))
        X_meta_scaled = scaler.transform(X_meta)
        predictions = logistic_model.predict(X_meta_scaled)
    return predictions

def predict_bert(texts, bert_model):
    bert_model.eval()
    with torch.no_grad():
        inference_dataset = bert_classifier.ReviewDataset(texts, [0] * len(texts))
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=bert_classifier.BATCH_SIZE,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=bert_classifier.collate_batch
        )
        predictions = []
        for batch in inference_loader:
            input_ids = batch['input_ids'].to(bert_classifier.DEVICE)
            attention_mask = batch['attention_mask'].to(bert_classifier.DEVICE)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            predictions.extend(preds.tolist())
    return predictions

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / 'data/raw/llm_reviews.csv'
    with open(file_path, 'r', encoding='utf-8') as f:
        reviews = [line.strip() for line in f if line.strip()]
    data = pd.DataFrame({'review': reviews})
    tokenizer = BertTokenizerFast.from_pretrained(meta_model.model_path)
    bert_model = BertForSequenceClassification.from_pretrained(meta_model.model_path)
    bert_model.to(bert_classifier.DEVICE)
    imported_model = joblib.load(bert_classifier.MODELS_DIR / 'meta_model.pkl')
    logistic_model = imported_model['meta_model']
    scaler = imported_model['scaler']
    predictions = predict_meta(data['review'].tolist(), bert_model, logistic_model, scaler)
    print(predictions)
    print("\n---\n")
    bert_predictions = predict_bert(data['review'].tolist(), bert_model)
    print(bert_predictions)
    print("\n---\n веса:")
    print(logistic_model.coef_[0]) 
