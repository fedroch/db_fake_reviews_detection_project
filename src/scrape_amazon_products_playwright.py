import asyncio
import pandas as pd
import random
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Маппинг категорий на URL-параметры Amazon
CATEGORY_URLS = {
    'Subscription_Boxes': 'subscription-boxes',
    'All_Beauty': 'beauty',
    'Amazon_Fashion': 'fashion',
    'Appliances': 'appliances',
    'Arts_Crafts_and_Sewing': 'arts-crafts-sewing',
    'Automotive': 'automotive',
    'Baby_Products': 'baby-products',
    'Beauty_and_Personal_Care': 'beauty',
    'Books': 'books',
    'CDs_and_Vinyl': 'music',
    'Cell_Phones_and_Accessories': 'electronics',
    'Clothing_Shoes_and_Jewelry': 'fashion',
    'Digital_Music': 'digital-music',
    'Electronics': 'electronics',
    'Gift_Cards': 'gift-cards',
    'Grocery_and_Gourmet_Food': 'grocery',
    'Handmade_Products': 'handmade',
    'Health_and_Household': 'health-household',
    'Health_and_Personal_Care': 'health-personal-care',
    'Home_and_Kitchen': 'home-kitchen',
    'Industrial_and_Scientific': 'industrial-scientific',
    'Kindle_Store': 'kindle-store',
    'Magazine_Subscriptions': 'magazines',
    'Movies_and_TV': 'movies-tv',
    'Office_Products': 'office-products',
    'Patio_Lawn_and_Garden': 'patio-lawn-garden',
    'Pet_Supplies': 'pet-supplies',
    'Software': 'software',
    'Sports_and_Outdoors': 'sports-outdoors',
    'Tools_and_Home_Improvement': 'tools-home-improvement',
    'Toys_and_Games': 'toys-games',
    'Video_Games': 'video-games'
}

async def is_captcha(page):
    """Проверяет, появилась ли капча."""
    content = await page.content()
    captcha_markers = [
        "api-services-support@amazon.com",
        "Type the characters you see in this image",
        "Enter the characters you see below",
        "Robot Check"
    ]
    return any(marker in content for marker in captcha_markers)

async def navigate_with_retry(page, url, max_retries=3):
    """Навигация с повторными попытками при ошибках или капче."""
    for attempt in range(max_retries):
        try:
            logger.info(f"  Попытка {attempt + 1}: Переход на {url}")
            response = await page.goto(url, wait_until='networkidle', timeout=60000)
            
            if response and response.status == 200:
                if await is_captcha(page):
                    logger.warning("  ⚠ Обнаружена капча! Ожидание перед повтором...")
                    await asyncio.sleep(random.uniform(5, 10))
                    continue
                return response
            
            logger.warning(f"  ⚠ Неверный статус: {response.status if response else 'No response'}")
            
        except PlaywrightTimeoutError:
            logger.warning(f"  ⚠ Таймаут при загрузке {url}")
        except Exception as e:
            logger.error(f"  ❌ Ошибка навигации: {e}")
        
        await asyncio.sleep(random.uniform(2, 5))
    return None

async def scrape_amazon_products(num_products_per_category=50):
    """
    Скрейпит популярные товары с Amazon для каждой категории.
    """
    
    all_products = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
            ]
        )
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            locale='en-US',
            viewport={'width': 1920, 'height': 1080},
        )
        
        await context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            """
        )
        
        page = await context.new_page()
        page.set_default_timeout(30000)
        
        try:
            # Перемешиваем категории, чтобы не идти по списку всегда одинаково
            categories = list(CATEGORY_URLS.items())
            random.shuffle(categories)
            
            for category_name, category_url in categories:
                logger.info(f"\n📦 Обработка категории: {category_name}")
                
                url = f"https://www.amazon.com/s?k={category_url}&ref=nb_sb_noss&sort=popularity-rank"
                response = await navigate_with_retry(page, url)
                
                if not response:
                    logger.error(f"  ❌ Не удалось загрузить категорию {category_name} после всех попыток")
                    continue
                
                # Небольшой скролл вниз для подгрузки контента
                await page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(random.uniform(1, 2))
                
                try:
                    await page.wait_for_selector("div.s-result-item", timeout=15000)
                except Exception:
                    logger.warning("  ⚠ Элементы поиска не найдены вовремя")
                
                product_elements = await page.query_selector_all("div.s-result-item[data-asin]")
                if not product_elements:
                    # Пробуем более общий селектор если с ASIN не нашлось
                    product_elements = await page.query_selector_all("div[data-component-type='s-search-result']")
                
                logger.info(f"  ✓ Найдено {len(product_elements)} потенциальных товаров")
                
                products_collected = 0
                for product in product_elements:
                    if products_collected >= num_products_per_category:
                        break
                        
                    try:
                        asin = await product.get_attribute('data-asin')
                        if not asin: continue

                        link_elem = await product.query_selector("h2 a")
                        if not link_elem:
                            continue
                            
                        link = await link_elem.get_attribute('href')
                        if not link:
                            continue
                            
                        if link.startswith('/'): 
                            link = f"https://www.amazon.com{link}"
                        
                        # Очищаем ссылку от лишних параметров для чистоты
                        if '?' in link:
                            link = link.split('?')[0]
                        
                        # Для логов все же вытянем название, чтобы видеть прогресс
                        title_elem = await product.query_selector("h2 a span, h2 span")
                        title = (await title_elem.text_content() or 'Unknown').strip() if title_elem else 'Unknown'
                        
                        all_products.append({
                            'asin': asin,
                            'link': link
                        })
                        
                        products_collected += 1
                        logger.info(f"    {products_collected}. [{asin}] {title[:50]}...")
                        
                    except Exception as e:
                        logger.debug(f"    ✗ Ошибка парсинга товара: {e}")
                        continue
                
                logger.info(f"  ✅ Собрано: {products_collected}")
                # Пауза между категориями
                await asyncio.sleep(random.uniform(3, 7))
                
        finally:
            await context.close()
            await browser.close()
    
    return all_products

async def main():
    logger.info("🚀 Начинаем сбор ссылок с Amazon...")
    
    products = await scrape_amazon_products(num_products_per_category=50)
    
    if products:
        df = pd.DataFrame(products)
        # Убираем дубликаты по ASIN (самый надежный способ)
        df = df.drop_duplicates(subset=['asin'])
        
        # Оставляем только колонку с ссылками
        links_df = df[['link']]
        
        save_path = Path(__file__).parent.parent / 'data/raw/amazon_links.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        links_df.to_csv(save_path, index=False, encoding='utf-8')
        
        logger.info(f"\n✓ Итого собрано {len(links_df)} уникальных ссылок")
        logger.info(f"✓ Сохранено в: {save_path}")
    else:
        logger.error("❌ Не удалось собрать ссылки")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Сбор данных прерван пользователем")
