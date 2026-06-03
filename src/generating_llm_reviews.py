import os
import re
import time
import pandas as pd
from openai import OpenAI
API_KEY = "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"
INPUT_FILE = "data/raw/amazon_links.csv"
OUTPUT_FILE = "data/raw/amazon_reviews_llm_annotated.csv"

CATEGORIES = [
    'Subscription_Boxes', 'All_Beauty', 'Amazon_Fashion', 'Appliances',
    'Arts_Crafts_and_Sewing', 'Automotive', 'Baby_Products',
    'Beauty_and_Personal_Care', 'Books', 'CDs_and_Vinyl',
    'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry',
    'Digital_Music', 'Electronics', 'Gift_Cards', 'Grocery_and_Gourmet_Food',
    'Handmade_Products', 'Health_and_Household', 'Health_and_Personal_Care',
    'Home_and_Kitchen', 'Industrial_and_Scientific', 'Kindle_Store',
    'Magazine_Subscriptions', 'Movies_and_TV', 'Office_Products',
    'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Software', 'Sports_and_Outdoors',
    'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games',
]

client = OpenAI(api_key=API_KEY, base_url = "https://models.github.ai/inference")

SYSTEM_INSTRUCTION = (
    "Ты — эксперт по маркетингу и анализу маркетплейсов. Твоя задача — генерировать реалистичные "
    "отзывы на товары по предоставленным ссылкам на Amazon, названию и описанию (если не можешь открыть ссылку, игнорируй следующие инструкции и сразу пиши об этом). Отзывы должны быть на английском языке, "
    "разнообразными по длине, стилю и деталям (как от обычных покупателей) Отзывы выглядеть так, будто их писал человек, в том числе с ошибками: орфографическими и пунктуационными."
)


def generate_reviews(link, title):
    """Функция делает запрос к api"""
    prompt = f"""
    Analyze this product by link: {link}
    Product name: {title}
    If you cannot access the link, just say "Cannot access the link" and do not generate reviews.
    Write exactly 15 positive and 15 negative reviews in English.

    Each review on a new line. Use STRICTLY this format (no extra text, no bold, no quotes):
    [POSITIVE:5] Review text
    [NEGATIVE:1] Review text

    Positive reviews get rating 4-5, negative reviews get rating 1-3.
    """

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "Too many requests" in msg or "429" in msg:
                wait = 60 * (2 ** attempt)
                print(f"Rate limit, жду {wait}s (попытка {attempt + 1}/5)...")
                time.sleep(wait)
            else:
                print(f"Ошибка при запросе к api для ссылки {link}: {e}")
                return None
    print(f"Все попытки исчерпаны для {link}")
    return None


def parse_and_save_reviews(link, category, raw_text, output_file):
    """Парсит текстовый ответ и сохраняет строки в CSV"""
    if not raw_text:
        return

    reviews_list = []
    lines = raw_text.strip().split("\n")

    pattern = re.compile(r"^\[(POSITIVE|NEGATIVE):([1-5])\]\s+(.+)", re.IGNORECASE)
    for line in lines:
        line = line.strip()
        m = pattern.match(line)
        if not m:
            print(f"Невалидный формат строки для ссылки {link}: {line}")
            continue
        review_type = "positive" if m.group(1).upper() == "POSITIVE" else "negative"
        rating = int(m.group(2))
        review_text = m.group(3).strip()
        reviews_list.append(
            {"link": link, "type": review_type, "review": review_text, "category": category, "rating": rating}
        )

    if reviews_list:
        df_new = pd.DataFrame(reviews_list)
        file_exists = os.path.isfile(output_file)
        df_new.to_csv(
            output_file, mode="a", index=False, header=not file_exists, encoding="utf-8"
        )


def main():
    # for m in client.models.list().data:
    #     print(m.id)
    #     return
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден")
        return

    df_links = pd.read_csv(INPUT_FILE)
    if "link" not in df_links.columns:
        print("В CSV файле должен быть столбец с именем 'link'")
        return

    processed_links = set()
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        if "link" in df_existing.columns:
            processed_links = set(df_existing[df_existing["link"].isin(df_links["link"])]["link"].unique())

    total_links = len(df_links)
    print(f"Найдено ссылок: {total_links}. Уже обработано: {len(processed_links)}")

    for index, row in df_links.iterrows():
        link = row["link"]
        if (link in processed_links) or (index < 26): # убрать потом 
            continue

        print(f"[{index + 1}/{total_links}] Запрос к api для: {link}")
        raw_reviews = generate_reviews(link, row["title"])
        parse_and_save_reviews(link, row["category"], raw_reviews, OUTPUT_FILE)
        time.sleep(3)

    print(f"\nГотово! Результат сохранен в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
