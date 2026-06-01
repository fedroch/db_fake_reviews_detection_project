import os
import re
import time
import pandas as pd
from openai import OpenAI
API_KEY = "неа"
INPUT_FILE = "data/raw/amazon_links.csv"
OUTPUT_FILE = "data/raw/amazon_reviews_llm_result.csv"
ANNOTATED_FILE = "data/raw/amazon_reviews_llm_annotated.csv"

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

client = OpenAI(api_key=API_KEY, base_url = "https://openrouter.ai/api/v1")

SYSTEM_INSTRUCTION = (
    "Ты — эксперт по маркетингу и анализу маркетплейсов. Твоя задача — генерировать реалистичные "
    "отзывы на товары по предоставленным ссылкам на Amazon. Отзывы должны быть на английском языке, "
    "разнообразными по длине, стилю и деталям (как от обычных покупателей)."
)


def generate_reviews(link):
    """Функция делает запрос к Openrute"""
    prompt = f"""
    Проанализируй этот товар по ссылке: {link}
    Придумай и напиши для него ровно 10 положительных отзывов и ровно 10 отрицательных отзывов на английском языке.
    
    Каждый отзыв пиши с новой строки. 
    Используй строго следующий формат ответа и ничего лишнего (без вводных слов, без форматирования жирным):
    [ПОЛОЖИТЕЛЬНЫЙ] Текст отзыва...
    [ОТРИЦАТЕЛЬНЫЙ] Текст отзыва...
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при запросе к DeepSeek для ссылки {link}: {e}")
        return None


def parse_and_save_reviews(link, raw_text, output_file):
    """Парсит текстовый ответ и сохраняет строки в CSV"""
    if not raw_text:
        return

    reviews_list = []
    lines = raw_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("[ПОЛОЖИТЕЛЬНЫЙ]"):
            review_text = line.replace("[ПОЛОЖИТЕЛЬНЫЙ]", "").strip()
            reviews_list.append(
                {"link": link, "type": "positive", "review": review_text}
            )
        elif line.startswith("[ОТРИЦАТЕЛЬНЫЙ]"):
            review_text = line.replace("[ОТРИЦАТЕЛЬНЫЙ]", "").strip()
            reviews_list.append(
                {"link": link, "type": "negative", "review": review_text}
            )

    if reviews_list:
        df_new = pd.DataFrame(reviews_list)
        file_exists = os.path.isfile(output_file)
        df_new.to_csv(
            output_file, mode="a", index=False, header=not file_exists, encoding="utf-8"
        )


def annotate_reviews_for_link(link, reviews):
    """Запрашивает у модели категорию товара и оценки для уже готовых отзывов."""
    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)
    reviews_str = "\n".join(
        f"[{i + 1}] ({r['type']}) {r['review']}"
        for i, r in enumerate(reviews)
    )
    prompt = f"""Product link: {link}

Reviews:
{reviews_str}

Task:
1. Determine the product category — choose EXACTLY ONE from the list below:
{categories_str}

2. Assign a rating from 1 to 5 for each review (positive reviews are usually 4-5, negative usually 1-2).

Reply strictly in this format (no extra text):
[CATEGORY] category_name
[1] rating
[2] rating
...
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка аннотирования для {link}: {e}")
        return None


def parse_annotation(raw_text, reviews):
    """Извлекает категорию и оценки из ответа модели."""
    if not raw_text:
        return None, [None] * len(reviews)

    category = None
    ratings = [None] * len(reviews)
    cat_lookup = {c.lower(): c for c in CATEGORIES}

    for line in raw_text.strip().split("\n"):
        line = line.strip()

        # Оценка: [N] digit
        rating_match = re.match(r"\[(\d+)\]\s*(\d)", line)
        if rating_match:
            idx = int(rating_match.group(1)) - 1
            rating = int(rating_match.group(2))
            if 0 <= idx < len(reviews):
                ratings[idx] = rating
            continue

        # Явный формат [CATEGORY]/[КАТЕГОРИЯ] category_name
        explicit_match = re.match(r"\[(?:CATEGORY|КАТЕГОРИЯ)\]\s*(\S+)", line, re.IGNORECASE)
        if explicit_match:
            val = explicit_match.group(1).strip()
            category = cat_lookup.get(val.lower(), val)
            continue

        # Модель использует категорию как тег: [Electronics], [BOOKS], etc.
        direct_match = re.match(r"^\[([^\d\]]+)\]", line)
        if direct_match:
            val = direct_match.group(1).strip()
            matched = cat_lookup.get(val.lower())
            if matched:
                category = matched

    return category, ratings


def annotate_existing_reviews():
    """Читает OUTPUT_FILE, аннотирует категорией и оценками, сохраняет в ANNOTATED_FILE."""
    if not os.path.exists(OUTPUT_FILE):
        print(f"Файл {OUTPUT_FILE} не найден")
        return

    df = pd.read_csv(OUTPUT_FILE)
    if "link" not in df.columns or "review" not in df.columns:
        print("Ожидаются столбцы 'link' и 'review'")
        return

    df_done = pd.DataFrame()
    if os.path.exists(ANNOTATED_FILE):
        df_done = pd.read_csv(ANNOTATED_FILE)

    already_done = set()
    if not df_done.empty and "link" in df_done.columns and "category" in df_done.columns:
        already_done = set(df_done[df_done["category"].notna()]["link"].unique())

    links = df["link"].unique()
    print(f"Всего товаров: {len(links)}, уже аннотировано: {len(already_done)}")

    for link in links:
        if link in already_done:
            continue

        group = df[df["link"] == link].to_dict("records")
        print(f"Аннотирую: {link} ({len(group)} отзывов)")

        raw = annotate_reviews_for_link(link, group)
        category, ratings = parse_annotation(raw, group)

        if category is None:
            print(f"  Не удалось извлечь категорию, пропускаю. Ответ модели:\n{raw}\n")
            continue

        rows = []
        for i, review in enumerate(group):
            rows.append({
                "link": review["link"],
                "type": review["type"],
                "review": review["review"],
                "category": category,
                "rating": ratings[i],
            })

        if not df_done.empty:
            df_done = df_done[df_done["link"] != link]
            df_done = pd.concat([df_done, pd.DataFrame(rows)], ignore_index=True)
            df_done.to_csv(ANNOTATED_FILE, index=False, encoding="utf-8")
        else:
            df_new = pd.DataFrame(rows)
            df_new.to_csv(ANNOTATED_FILE, index=False, encoding="utf-8")
            df_done = df_new
        time.sleep(2)

    print(f"Готово! Аннотированный файл: {ANNOTATED_FILE}")


def main():
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
            processed_links = set(df_existing["link"].unique())

    total_links = len(df_links)
    print(f"Найдено ссылок: {total_links}. Уже обработано: {len(processed_links)}")

    for index, row in df_links.iterrows():
        link = row["link"]
        if link in processed_links:
            continue

        print(f"[{index + 1}/{total_links}] Запрос к api для: {link}")
        raw_reviews = generate_reviews(link)
        parse_and_save_reviews(link, raw_reviews, OUTPUT_FILE)
        time.sleep(3)

    print(f"\nГотово! Результат сохранен в {OUTPUT_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "annotate":
        annotate_existing_reviews()
    else:
        main()