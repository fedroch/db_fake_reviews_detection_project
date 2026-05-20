import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast

from src.data_parce import raw_data


class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'labels': self.labels[idx]
        }


def collate_batch(batch, tokenizer, max_len):
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    encodings['labels'] = labels
    return encodings


def extract_embeddings(model, dataloader, output_dir, split_name):
    model.eval()
    embeddings = []
    labels = []

    print(f"Извлечение эмбеддингов для {split_name}...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
            labels.append(batch['labels'].cpu().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{split_name}_embeddings.npy"
    labels_path = output_dir / f"{split_name}_labels.npy"
    np.save(emb_path, embeddings)
    np.save(labels_path, labels)

    print(f"  Сохранено: {emb_path} ({embeddings.shape})")
    print(f"  Сохранено: {labels_path} ({labels.shape})")
    return embeddings, labels

def parse_args():
    parser = argparse.ArgumentParser(description='Extract BERT embeddings and save them as .npy files.')
    parser.add_argument('--model-dir', type=Path, default=Path('models/bert_finetuned'),
                        help='Путь к готовой модели BERT. Если не задан, будет загружен bert-base-uncased.')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='Имя модели из Hugging Face, если модель из model-dir не найдена.')
    parser.add_argument('--output-dir', type=Path, default=Path('models/bert_finetuned/bert_embeddings'),
                        help='Папка для сохранения эмбеддингов и меток.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Размер батча для извлечения эмбеддингов.')
    parser.add_argument('--max-len', type=int, default=128,
                        help='Максимальная длина токенов.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Число потоков DataLoader.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Случайное состояние для train/test split.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_dir.exists() and (args.model_dir / 'config.json').exists():
        model_source = args.model_dir
        print(f"Загружаем модель из {model_source}")
    else:
        model_source = args.model_name
        print(f"Папка модели не найдена, загружаем модель из Hugging Face: {model_source}")

    tokenizer = BertTokenizerFast.from_pretrained(model_source)
    model = BertForSequenceClassification.from_pretrained(model_source)
    model.to(device)

    text = raw_data['text'].fillna('').astype(str)
    label = raw_data['label'].astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        text,
        label,
        test_size=0.2,
        random_state=args.seed
    )

    train_dataset = ReviewDataset(X_train_raw, y_train)
    test_dataset = ReviewDataset(X_test_raw, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, args.max_len)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, args.max_len)
    )

    extract_embeddings(model, train_loader, args.output_dir, 'train')
    extract_embeddings(model, test_loader, args.output_dir, 'test')

    print('\nГотово. Эмбеддинги сохранены в', args.output_dir)


if __name__ == '__main__':
    main()
