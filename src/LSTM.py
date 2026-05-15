import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from transformers import get_scheduler
from src.data_parce import clear_data as data
import json

# data = pd.read_csv(Path(__file__).parent.parent / 'data/raw/fake reviews dataset.csv') # маленький датасет
# data = pd.read_csv(Path(__file__).parent.parent / 'data/raw/pseudo_labeled_amazon_reviews.csv') # жирненький датасет

def load_lstm_pretrained(save_dir, device):
    save_dir = Path(save_dir)
    with open(save_dir / 'LSTM_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(save_dir / 'LSTM_vocab.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    tokenizer = custom_tokenizer(max_vocab_size=config["vocab_size"])
    tokenizer.word2id = word2id
    tokenizer.id2word = {int(v): k for k, v in word2id.items()}
    model = LSTM(
        vocab_size=config["vocab_size"], 
        emb_dim=config["emb_dim"], 
        hidden_dim=config["hidden_dim"]
    )
    model.load_state_dict(torch.load(save_dir / 'pytorch_LSTM_model.bin', map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def load_glove_embeddings(word2id, glove_path, emb_dim=100):
    vocab_size = len(word2id)
    # Инициализируем матрицу небольшим шумом
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, emb_dim))
    hits = 0
    misses = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            values = line.split()
            # Пропускаем, если строка пустая или битая
            if len(values) < emb_dim + 1:
                continue 
            word = values[0]
            if word in word2id:
                try:
                    # Пытаемся взять только первые emb_dim чисел после слова
                    vector = np.array(values[1:emb_dim+1], dtype='float32')
                    embedding_matrix[word2id[word]] = vector
                    hits += 1
                except ValueError:
                    misses += 1
                    continue           
    print(f"Загрузка окончена. Успешно: {hits}, Пропущено: {misses}")      
    return torch.from_numpy(embedding_matrix).float()

class custom_tokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.id2word = {0: '<PAD>', 1: '<UNK>'}
    def build_vocab(self, texts):
        counter = Counter(word for text in texts for word in str(text).split())
        most_common = counter.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word
    def encode(self, text, max_len=50):
        ids = [self.word2id.get(word, 1) for word in str(text).split()]
        return ids[:max_len] + [0] * (max_len - len(ids))

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=250):
        self.data = [torch.tensor(tokenizer.encode(text, max_len)) for text in texts]
        self.lengths = [min(len(str(text).split()), max_len) for text in texts]
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.data[idx]
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        label = self.labels[idx]
        return text, length, label
class LSTM(nn.Module):
    def __init__(self, vocab_size=10000, emb_dim=128, hidden_dim=128, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim*2, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        out = out.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        pooled, _ = torch.max(out, dim=1)
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        return self.fc(self.dropout(pooled))

def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    losses, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for texts, lengths, labels in data_loader:
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            outputs = model(texts, lengths)
            if criterion is not None:
                losses.append(criterion(outputs, labels).item())
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    loss = float(np.mean(losses)) if losses else None
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return loss, acc, all_preds, all_labels

def train_model(model, train_loader, val_loader, device, epochs=15, patience=5):
    optimizer = optim.AdamW(model.parameters(),weight_decay=0.001, lr=1e-3)
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = num_training_steps // 10
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds, train_true = [], []
        for texts, lengths, labels in train_loader:
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            train_true.extend(labels.detach().cpu().tolist())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
        )

        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: нет улучшений val_acc {patience} эпох(и) подряд.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Загружена лучшая модель: Val Acc {best_val_acc:.2%}, Val Loss {best_val_loss:.4f}")

def run_pipeline(X_train, y_train, X_val, y_val):
    # импорт ембедингов
    emb_dim = 100
    glove_path = Path(__file__).parent.parent / 'data/embeddings/glove.6B.100d.txt'
    # Токенизация
    tokenizer = custom_tokenizer(max_vocab_size=10000)
    tokenizer.build_vocab(X_train)
    if glove_path.exists():
        weights = load_glove_embeddings(tokenizer.word2id, glove_path, emb_dim=emb_dim)
    else:
        print(f"Файл с эмбеддингами не найден по пути {glove_path}. модель обучается с нуля")
        weights = None
    # Загрузчики данных
    train_ds = ReviewDataset(X_train, y_train, tokenizer, max_len=128)
    val_ds = ReviewDataset(X_val, y_val, tokenizer, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    # Инициализация модели
    model = LSTM(vocab_size=len(tokenizer.word2id), emb_dim=emb_dim, hidden_dim=128, pretrained_weights=weights)
    model.to(device)
    # Обучение
    print("Рецепт иишницы на видеокарте: возьмите 1 модель LSTM, добавьте капельку слез и приправьте 25 эпохами обучения. Подавайте горячим!")
    train_model(model, train_loader, val_loader, device, epochs=15, patience=5)
    
    return model, tokenizer

def save_lstm_pretrained(model, tokenizer, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 1. Сохраняем веса модели
    weights_path = save_dir / 'pytorch_LSTM_model.bin'
    torch.save(model.state_dict(), weights_path)
    # 2. Сохраняем конфиг архитектуры (чтобы знать размеры слоев при загрузке)
    config = {
        "vocab_size": model.embedding.num_embeddings,
        "emb_dim": model.embedding.embedding_dim,
        "hidden_dim": model.lstm.hidden_size,
        # "num_layers": model.lstm.num_layers,
        "bidirectional": model.lstm.bidirectional
    }
    with open(save_dir / 'LSTM_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    with open(save_dir / 'LSTM_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(tokenizer.word2id, f, ensure_ascii=False, indent=4)
    print(f"\nМодель и токенизатор сохранены в {save_dir}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Ползем на GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Увы, всё еще на процессоре...")

    # data['label'] = data['label'].map({'OR': 0, 'CG': 1}) # для маленького датасета

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], 
        data['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=data['label']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        random_state=42,
        stratify=y_train
    )

    model, tokenizer = run_pipeline(X_train, y_train, X_val, y_val)

    tests_loader = DataLoader(ReviewDataset(X_test, y_test, tokenizer, max_len=128), batch_size=128, shuffle=False)
    _, _, preds, true = evaluate_model(model, tests_loader, device)
    print(f"Общая точность (Accuracy): {accuracy_score(true, preds):.2%}")
    print("\nПодробный отчет:")
    print(classification_report(true, preds))
    MODELS_DIR = Path(__file__).parent.parent / 'models'
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = MODELS_DIR / 'LSTM'
    save_lstm_pretrained(model, tokenizer, save_path)
