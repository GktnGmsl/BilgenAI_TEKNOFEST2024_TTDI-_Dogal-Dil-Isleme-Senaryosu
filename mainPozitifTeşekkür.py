import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ChromeDriver'ı ayarlar
options = Options()
options.headless = False
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def complaint_scrapper(url, page_max):
    start_time = time.time()

    out_arr = []

    for page_num in range(1, page_max + 1):
        cur_url = f"{url}&page={page_num}"
        driver.get(cur_url)

        # Çerez uyarısını kapatır (varsa)
        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll'))
            ).click()
            time.sleep(2)  # İçeriğin yüklenmesi için bekleme süresi
        except Exception as e:
            print("Çerez uyarısı bulunamadı veya kapatılamadı:", e)

        # Sayfa kaydırma işlemini gerçekleştirir
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)  # İçeriğin yüklenmesi için bekleme süresini belirtmek için kullanılır

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Çözülmüş şikayetleri alır.
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".complaint-layer.card-v3-container"))
            )
        except Exception as e:
            print(f"Şikayet kartları yüklenemedi: {e}")
            continue

        complaint_cards = driver.find_elements(By.CSS_SELECTOR, ".complaint-layer.card-v3-container")
        complaint_links = [card.get_attribute('href') for card in complaint_cards if card.find_element(By.CSS_SELECTOR, ".solved-badge")]

        for complaint_link in complaint_links:
            try:
                driver.get(complaint_link)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".solution-content-body"))
                )
                time.sleep(3)  # İçeriğin yüklenmesi için bekleme süresi belirtilir

                # Şikayet bilgilerini al
                complaint_title = driver.find_element(By.CSS_SELECTOR, ".complaint-title").text
                complaint_desc = driver.find_element(By.CSS_SELECTOR, ".solution-content-body").text

                out_arr.append([complaint_title, complaint_desc])
            except Exception as e:
                print(f"Detaylı metin alınırken hata oluştu: {e}")

        print(f"Sayfa {page_num} tamamlandı.")

    driver.quit()

    end_time = time.time()
    print(f"Web scraping ended in {(end_time - start_time) * 1000:.0f}ms")

    return out_arr

def arr_to_csv(filename, data):
    df = pd.DataFrame(data, columns=["title", "description"])
    df.to_csv(filename, index=False, encoding='utf-8-sig')

# Verileri çeker
result = complaint_scrapper(
    "https://www.sikayetvar.com/turkcell?resolved=true",
    267
)

# Verileri CSV dosyasına yazar
arr_to_csv('Pozitifsikayet-var-full.csv', result)

print("Yorumlar CSV dosyasına kaydedildi.")