from src.data_parce import raw_data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import numpy as np


#  Настройки — подобраны под RTX 2060 (6GB)


MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'bert-base-uncased'
MAX_LEN    = 128   # ключевой параметр: 128 влезает в 6GB, 256 скорее всего нет
BATCH_SIZE = 32     # для 6GB безопасный размер, попробуй 8 если будет OOM
EPOCHS     = 3
LR         = 2e-5

print(f"Устройство: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"Видеокарта:  {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:        {total_mem:.1f} GB")
else:
    print("CUDA не найдена — обучение на CPU будет очень долгим!")


#  Маппинг меток


label2id = {'CG': 0, 'OR': 1}
id2label = {0: 'CG', 1: 'OR'}


#  Датасет
raw_data = raw_data.fillna('')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = torch.tensor(
            [label2id[l] if isinstance(l, str) and l in label2id else int(l) for l in labels],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text':   self.texts[idx],
            'labels': self.labels[idx]
        }


def collate_batch(batch):
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])

    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    encodings['labels'] = labels
    return encodings


#  Подготовка данных


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    raw_data['text'],
    raw_data['label'],
    test_size=0.2,
    random_state=42
)

print(f"\nTrain: {len(X_train_raw)} | Test: {len(X_test_raw)}")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

print("Токенизация train...")
train_dataset = ReviewDataset(X_train_raw, y_train)
print("Токенизация test...")
test_dataset  = ReviewDataset(X_test_raw,  y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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

#  Обучение


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader):
        input_ids       = batch['input_ids'].to(DEVICE)
        attention_mask  = batch['attention_mask'].to(DEVICE)
        token_type_ids  = batch['token_type_ids'].to(DEVICE)
        labels          = batch['labels'].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            if DEVICE.type == 'cuda':
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  шаг {i+1}/{len(loader)} | loss: {total_loss/(i+1):.4f} | VRAM: {vram:.2f} GB")
            else:
                print(f"  шаг {i+1}/{len(loader)} | loss: {total_loss/(i+1):.4f}")

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def extract_and_save_embeddings(model, loader, save_path, name):
    """Извлекает CLS-эмбеддинги и сохраняет их вместе с метками"""
    model.eval()
    embeddings = []
    labels = []
    
    print(f"Извлечение эмбеддинков для {name}...")
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch   = batch['labels'].cpu().numpy()
            
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(cls_embeddings)
            labels.extend(labels_batch)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Сохранение
    embeddings_path = save_path / f'{name}_embeddings.npy'
    labels_path = save_path / f'{name}_labels.npy'
    
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    
    print(f"  Эмбеддинки сохранены: {embeddings_path} ({embeddings.shape})")
    print(f"  Метки сохранены: {labels_path} ({labels.shape})")
    
    return embeddings, labels

if __name__ == "__main__":
    #  Модель
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    model.to(DEVICE)

    # покажем сколько памяти заняла модель
    if DEVICE.type == 'cuda':
        used = torch.cuda.memory_allocated() / 1e9
        print(f"\nПамять после загрузки модели: {used:.2f} GB")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    print("\n─── Начало обучения ───")
    for epoch in range(EPOCHS):
        print(f"\nЭпоха {epoch + 1} / {EPOCHS}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"  Средний loss: {avg_loss:.4f}")

        labels_true, labels_pred = evaluate(model, test_loader)
        acc = accuracy_score(labels_true, labels_pred)
        print(f"  Accuracy на тесте: {acc:.2%}")

        if DEVICE.type == 'cuda':
            print(f"  Пик VRAM за эпоху: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    #  Финальный отчёт

    print("\n─── Финальные метрики ───")
    labels_true, labels_pred = evaluate(model, test_loader)
    print(f"Accuracy: {accuracy_score(labels_true, labels_pred):.2%}")
    print(classification_report(labels_true, labels_pred, target_names=['CG', 'OR']))

    #  Сохранение

    save_path = MODELS_DIR / 'bert_finetuned'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nМодель сохранена в {save_path}")

    #  Сохранение эмбеддинков

    print("\n─── Сохранение эмбеддинков ───")
    train_embeddings, train_labels = extract_and_save_embeddings(model, train_loader, save_path, 'train')
    test_embeddings, test_labels = extract_and_save_embeddings(model, test_loader, save_path, 'test')
    print("\nЭмбеддинки сохранены и готовы для meta_model!")
