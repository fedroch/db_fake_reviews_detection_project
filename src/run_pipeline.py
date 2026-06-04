import random

from src import meta_model, bert_classifier
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def predict_meta(df, bert_model, logistic_model, scaler):
    # df должен содержать колонки 'text' и 'rating' (нужны get_custom_features)
    bert_model.eval()
    texts = df["text"].astype(str).tolist()
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
        custom_features = meta_model.get_custom_features(df)
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

KEEP_COLS = ["text", "label", "rating", "category"]


def _normalize(df):
    """Приводит датасет к единым колонкам text/label/rating/category."""
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = 0 if col in ("label", "rating") else ""
    df = df[KEEP_COLS].copy()
    df["text"]   = df["text"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df = df[df["label"].notna()]
    return df.reset_index(drop=True)


def load_test_datasets(data_dir):
    """Грузит 3 тестовых датасета, нормализует их к общему виду."""
    random.seed(42)

    df_pseudo = pd.read_csv(
        data_dir / 'pseudo_labeled_amazon_reviews.csv',
        skiprows=lambda i: (i > 0) and (random.random() > 0.1)
    ).fillna("")

    df_llm = pd.read_csv(data_dir / 'amazon_reviews_llm_annotated.csv').fillna("")
    df_llm = df_llm.rename(columns={"review": "text"})
    df_llm["label"] = 0

    df_fake = pd.read_csv(data_dir / 'fake_reviews_dataset.csv').fillna("")
    df_fake = df_fake.rename(columns={"text_": "text"})
    df_fake["label"] = df_fake["label"].map({"CG": 0, "OR": 1})
    # у категорий _5 вконце
    df_fake["category"] = df_fake["category"].astype(str).str.replace(r"_\d+$", "", regex=True)
    return {
        "pseudo_labeled": _normalize(df_pseudo),
        "llm_annotated":  _normalize(df_llm),
        "fake_reviews":   _normalize(df_fake),
    }

def evaluate(name, df, mode, bert_model, logistic_model=None, scaler=None):
    """Прогоняет модель по одному датасету и считает метрики."""
    y_true = df["label"].to_numpy()
    if mode == "meta":
        y_pred = predict_meta(df, bert_model, logistic_model, scaler)
    else:
        y_pred = predict_bert(df["text"].tolist(), bert_model)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"dataset": name, "n": len(df), "accuracy": acc,
            "precision": p, "recall": r, "f1": f1}

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data/raw'
    args = sys.argv[1:]

    if args[:1] == ["run"]:
        if len(args) < 2:
            print("использование: run_pipeline.py run <путь к файлу с отзывами>")
            sys.exit(1)
        with open(args[1], 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        data = pd.DataFrame({'text': reviews})
        data['rating'] = 0
        tokenizer = BertTokenizerFast.from_pretrained(meta_model.model_path)
        bert_model = BertForSequenceClassification.from_pretrained(meta_model.model_path)
        bert_model.to(bert_classifier.DEVICE)
        imported_model = joblib.load(bert_classifier.MODELS_DIR / 'meta_model.pkl')
        logistic_model = imported_model['meta_model']
        scaler = imported_model['scaler']

        predictions = predict_meta(data, bert_model, logistic_model, scaler)
        print(predictions)
        print("\n---\n")
        bert_predictions = predict_bert(data['text'].tolist(), bert_model)
        print(bert_predictions)
        print("\n---\n веса:")
        print(logistic_model.coef_[0])

    elif args[:1] == ["test"]:
        mode = args[1] if len(args) > 1 else "meta"
        if mode not in ("meta", "bert"):
            print("использование: run_pipeline.py test [meta|bert]")
            sys.exit(1)

        datasets = load_test_datasets(data_dir)

        tokenizer = BertTokenizerFast.from_pretrained(meta_model.model_path)
        bert_model = BertForSequenceClassification.from_pretrained(meta_model.model_path)
        bert_model.to(bert_classifier.DEVICE)

        logistic_model = scaler = None
        if mode == "meta":
            imported_model = joblib.load(bert_classifier.MODELS_DIR / 'meta_model.pkl')
            logistic_model = imported_model['meta_model']
            scaler = imported_model['scaler']

        rows = [
            evaluate(name, df, mode, bert_model, logistic_model, scaler)
            for name, df in datasets.items()
        ]
        table = pd.DataFrame(rows).set_index("dataset")
        print(f"\n=== Результаты ({mode}) ===")
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    else:
        print("нужен флаг: run | test")
