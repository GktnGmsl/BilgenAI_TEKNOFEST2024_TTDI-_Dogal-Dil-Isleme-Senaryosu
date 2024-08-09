import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

# ChromeDriver'ın yolunu belirler
chrome_driver_path = "C:/Users/gktng/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"

# ChromeDriverManager'ın son sürümünü yükler
options = Options()
options.headless = False

# ChromeDriver'ı başlat
driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

def complaint_scrapper(url, page_max):
    start_time = time.time()

    out_arr = []

    for page_num in range(1, page_max + 1):
        try:
            cur_url = f"{url}{page_num}"
            driver.get(cur_url)

            # Çerez uyarısını kapatır (varsa)
            try:
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll'))
                ).click()
                time.sleep(2)  # İçeriğin yüklenmesi için bekleme süresini belirtir
            except Exception as e:
                print("Çerez uyarısı bulunamadı veya kapatılamadı:", e)

            # Sayfa kaydırma işlemi yapılır.
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)  # İçeriğin yüklenmesi için bekleme süresini belirtir

                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # Şikayet URL'lerini alır
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".complaint-layer.card-v3-container"))
                )
            except Exception as e:
                print("Şikayet URL'leri yüklenemedi:", e)
                continue

            complaint_urls = driver.find_elements(By.CSS_SELECTOR, ".complaint-layer.card-v3-container")
            urls = [el.get_attribute('href') for el in complaint_urls if el.get_attribute('href')]

            for complaint_url in urls:
                try:
                    driver.get(complaint_url)
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".complaint-detail-description"))
                    )
                    time.sleep(3)  # İçeriğin yüklenmesi için bekleme süresini belirtir

                    # Şikayet bilgilerini al
                    complaint_title = driver.find_element(By.CSS_SELECTOR, ".complaint-title").text
                    complaint_desc = driver.find_element(By.CSS_SELECTOR, ".complaint-detail-description p").text

                    out_arr.append([complaint_title, complaint_desc])
                except Exception as e:
                    print(f"Detaylı metin alınırken hata oluştu: {e}")
        except Exception as e:
            print(f"Sayfa {page_num} işlenirken hata oluştu: {e}")

    driver.quit()

    end_time = time.time()
    print(f"Web scraping ended in {(end_time - start_time) * 1000:.0f}ms")

    return out_arr


def arr_to_csv(filename, data):
    df = pd.DataFrame(data, columns=["title", "description"])
    df.to_csv(filename, index=False, encoding='utf-8-sig')


# Şikayet verilerini çeker
result = complaint_scrapper(
    "https://www.sikayetvar.com/turkcell=",
    350
)

# Verileri CSV dosyasına yazar
arr_to_csv('sikayet-var-full.csv', result)

print("Yorumlar CSV dosyasına kaydedildi.")