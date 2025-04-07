import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Target URL (EUR/USD News Page)
base_url = "https://www.dailyforex.com/currencies/eur/usd"
driver.get(base_url)

# List to store scraped data
data = []

# Page counter
page_count = 1
max_pages = 30

while page_count <= max_pages:
    print(f"Scraping Page {page_count}...")

    # Wait for articles to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.article-info"))
    )

    # Extract articles
    articles = driver.find_elements(By.CSS_SELECTOR, "div.article-info")

    for article in articles:
        try:
            # Title & Link
            title_elem = article.find_element(By.CSS_SELECTOR, "a.article-title")
            title = title_elem.text.strip()
            link = title_elem.get_attribute('href')

            # Description
            try:
                desc_elem = article.find_element(By.CSS_SELECTOR, "p.article-description")
                description = desc_elem.text.strip()
            except:
                description = "N/A"

            # # Category
            # try:
            #     category_elem = article.find_element(By.CSS_SELECTOR, "span.article-type")
            #     category = category_elem.text.strip()
            # except:
            #     category = "N/A"

            # Published Date
            try:
                date_elem = article.find_element(By.CSS_SELECTOR, "time")
                timestamp_str = date_elem.get_attribute('datetime') if date_elem else "N/A"
            except:
                timestamp_str = "N/A"

            # Store data
            data.append([timestamp_str, title, link, description, ])

            print(f"Title: {title}\nLink: {link}\nPublished: {timestamp_str}\n{'-'*50}")

        except Exception as e:
            print(f"Error extracting article: {e}")

    # Attempt to go to next page
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, "li.btn-next a")

        # Check if "Next" button is disabled
        if "disabled" in next_button.get_attribute("class"):
            print("Next button is disabled. Ending scrape.")
            break

        # Click using JavaScript to bypass restrictions
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(3)  # Wait for next page to load
        page_count += 1

    except:
        print("Next button not found. Stopping.")
        break

# Close WebDriver
driver.quit()

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data, columns=["date", "title", "link", "description" ])
csv_filename = "dailyforex_eurusd_news.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"Data successfully saved to {csv_filename}")
