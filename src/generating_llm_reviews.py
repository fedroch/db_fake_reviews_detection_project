import os
import time
import pandas as pd
from openai import OpenAI
API_KEY = "хе хе, гитхаб оказывается выкладывать секретну информацию в репозиторий"
INPUT_FILE = "data/raw/amazon_links.csv"
OUTPUT_FILE = "data/raw/amazon_reviews_llm_result.csv"

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
    main()