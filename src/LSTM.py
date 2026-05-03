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
# from src.data_parce import clear_data as data
import json

# data = pd.read_csv(Path(__file__).parent.parent / 'data/raw/fake reviews dataset.csv')

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
    # Инициализируем матрицу небольшим шумом (это важно для обучения)
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
    def __init__(self, max_vocab_size=5000):
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
        self.labels = torch.tensor(labels.values,dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        return text, label
class LSTM(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=128, hidden_dim=128, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim*2, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        pooled, _ = torch.max(out, dim=1)
        return self.fc(self.dropout(pooled))

def train_model(model, train_loader, device, epochs=20):
    optimizer = optim.Adam(model.parameters(),weight_decay=1e-5, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss =0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def run_pipeline(X_train, y_train):
    # импорт ембедингов
    emb_dim = 100
    glove_path = Path(__file__).parent.parent / 'data/embeddings/glove.6B.100d.txt'
    # Токенизация
    tokenizer = custom_tokenizer(max_vocab_size=5000)
    tokenizer.build_vocab(X_train)
    if glove_path.exists():
        weights = load_glove_embeddings(tokenizer.word2id, glove_path, emb_dim=emb_dim)
    else:
        print(f"Файл с эмбеддингами не найден по пути {glove_path}. модель обучается с нуля")
        weights = None
    # Загрузчики данных
    train_ds = ReviewDataset(X_train, y_train, tokenizer, max_len=250)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    # Инициализация модели
    model = LSTM(vocab_size=5000, emb_dim=emb_dim, hidden_dim=256, pretrained_weights=weights)
    model.to(device)
    # Обучение
    print("Рецепт иишницы на видеокарте: возьмите 1 модель LSTM, добавьте капельку слез и приправьте 10 эпохами обучения. Подавайте горячим!")
    train_model(model, train_loader, device, epochs=15)
    
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

    data['label'] = data['label'].map({'OR': 0, 'CG': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        data['text_'], 
        data['label'], 
        test_size=0.2, 
        random_state=42
    )
    model, tokenizer = run_pipeline(X_train, y_train)

    tests_loader = DataLoader(ReviewDataset(X_test, y_test, tokenizer, max_len=250), batch_size=128, shuffle=False)
    preds, true = [], []
    model.eval()
    with torch.no_grad():
        for texts, labels in tests_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true.extend(labels.cpu().numpy())
    print(f"Общая точность (Accuracy): {accuracy_score(true, preds):.2%}")
    print("\nПодробный отчет:")
    print(classification_report(true, preds))
    MODELS_DIR = Path(__file__).parent.parent / 'models'
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = MODELS_DIR / 'LSTM'
    save_lstm_pretrained(model, tokenizer, save_path)