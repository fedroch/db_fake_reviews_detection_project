import torch
import pandas as pd
import tqdm
from pathlib import Path
from datasets import load_dataset
from src.LSTM import load_lstm_pretrained

def generate_pseudo_labels(categories, sample_pct=0.01, threshold=0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).parent.parent / 'models/LSTM'
    model, tokenizer = load_lstm_pretrained(model_path, device)
    model.eval()
    all_dfs = [] # Будем собирать список DataFrame
    for cat in categories:
        print(f"\nОбработка категории: {cat}")
        # Загружаем 1% данных
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            name=f"raw_review_{cat}", 
            split=f"full[:{int(sample_pct*100)}%]", 
            streaming=False,
            trust_remote_code=True
        )
        labeled_texts, labeled_scores, labeled_ratings = [], [], []
        with torch.no_grad():
            # Шаг 128 для батчей
            for i in tqdm.tqdm(range(0, len(dataset), 128)):
                batch = dataset[i : i + 128]
                texts_raw = batch['text']
                texts_raw = [str(t) if t is not None else "" for t in batch['text']]
                ratings_raw = batch['rating']
                # Токенизация и инференс
                encoded = [tokenizer.encode(t, max_len=250) for t in texts_raw]
                input_ids = torch.tensor(encoded).to(device)
                logits = model(input_ids)
                probs = torch.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)      
                # Фильтруем только уверенные предсказания
                for j, prob in enumerate(max_probs):
                    if prob >= threshold:
                        labeled_texts.append(texts_raw[j])
                        labeled_scores.append(preds[j].item())
                        labeled_ratings.append(ratings_raw[j])
        print(f"Сохранено уверенных предсказаний: {len(labeled_texts)}")
        if labeled_texts:
            cat_df = pd.DataFrame({
                'category': cat,
                'rating': labeled_ratings,
                'label': labeled_scores,
                'text': labeled_texts
            })
            all_dfs.append(cat_df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

if __name__ == "__main__":
    categories = ['Subscription_Boxes',
                'All_Beauty',
                'Amazon_Fashion',
                'Appliances',
                'Arts_Crafts_and_Sewing',
                'Automotive',
                'Baby_Products',
                'Beauty_and_Personal_Care',
                'Books',
                'CDs_and_Vinyl',
                'Cell_Phones_and_Accessories',
                'Clothing_Shoes_and_Jewelry',
                'Digital_Music',
                'Electronics',
                'Gift_Cards',
                'Grocery_and_Gourmet_Food',
                'Handmade_Products',
                'Health_and_Household',
                'Health_and_Personal_Care',
                'Home_and_Kitchen',
                'Industrial_and_Scientific',
                'Kindle_Store',
                'Magazine_Subscriptions',
                'Movies_and_TV',
                'Office_Products',
                'Patio_Lawn_and_Garden',
                'Pet_Supplies',
                'Software',
                'Sports_and_Outdoors',
                'Tools_and_Home_Improvement',
                'Toys_and_Games',
                'Video_Games'] 
    df = generate_pseudo_labels(categories)
    if not df.empty:
        save_path = Path(__file__).parent.parent / 'data/raw/pseudo_labeled_amazon_reviews.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nИтог: {len(df)} строк сохранено в {save_path}")
    else:
        print("Уверенных предсказаний не найдено.")