from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

options = Options()
options.add_argument("--headless")  # Optional: Run in headless mode
options.add_argument("--disable-gpu")
options.add_argument("--disable-software-rasterizer")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--allow-running-insecure-content")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

try:
    driver.get("https://www.investing.com/economic-calendar/")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "economicCalendarFilters"))
    )
    print("Page loaded successfully.")

    # Close cookie popup
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        ).click()
    except:
        print("Cookie consent popup not found.")

    # Set the date range
    date_button = driver.find_element(By.ID, "datePickerToggleBtn")
    date_button.click()

    start_input = driver.find_element(By.ID, "startDate")
    start_input.clear()
    start_input.send_keys("08/23/2024")

    end_input = driver.find_element(By.ID, "endDate")
    end_input.clear()
    end_input.send_keys("01/23/2025")

    apply_button = driver.find_element(By.ID, "applyBtn")
    apply_button.click()

    time.sleep(3)

    # Scroll to load data
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Simulate human behavior

        # Check if new data loaded
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

        # Optional: Add logging
        print("Scrolled down...")

    data_list = []
    # Extract data
    rows = driver.find_elements(By.CSS_SELECTOR, "tr.js-event-item")
    for row in rows:
        event_data = {
            "date": row.get_attribute("data-event-datetime"),
            "country": row.find_element(By.CSS_SELECTOR, ".flagCur").text.strip(),
            "event": row.find_element(By.CSS_SELECTOR, ".event").text.strip(),
            "impact": row.find_element(By.CSS_SELECTOR, ".sentiment").get_attribute("title").strip(),
            "actual": row.find_element(By.CSS_SELECTOR, ".act").text.strip() if row.find_elements(By.CSS_SELECTOR, ".act") else "",
            "forecast": row.find_element(By.CSS_SELECTOR, ".fore").text.strip() if row.find_elements(By.CSS_SELECTOR, ".fore") else "",
            "previous": row.find_element(By.CSS_SELECTOR, ".prev").text.strip() if row.find_elements(By.CSS_SELECTOR, ".prev") else ""
        }
        print(event_data)
        data_list.append(event_data)

    csv_filename = "economic_calendar_data.csv"
    df = pd.DataFrame(data_list)
    df.to_csv("economic_calendar_data.csv", index=False)
    print(" Data saved to economic_calendar_data.csv")
finally:
    driver.quit()
