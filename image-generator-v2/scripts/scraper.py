from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import os
import csv
from pathlib import Path
import requests
import time
# Set up the Chrome options for headless mode
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Set the path to your ChromeDriver
service = Service(executable_path='/usr/local/bin/chromedriver')

# Initialize the browser
driver = webdriver.Chrome()

# Set headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Directory to save images
image_dir = 'nsfw_images'


# Initialize CSV
csv_path = Path("nsfw_dir.csv")
if not csv_path.exists():
    with open("nsfw_dir.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "image_name", "caption"])

# List of categories (you can add more as needed)
categories = [
    "/pussy",
    "/schoolgirl",
    "/teen",
    "/latina",
    "/petite",
    "/cumshot",
    "/anal",
    "/homemade",
    "/cum-in-mouth",
    "/blowjob",
    "/lesbian",
    "/upskirt",
    "/hardcore",
    "/pov",
    "/blonde",
    "/brunette",
    "/missionary",
    "/pussy-licking",
    "/selfie",
    "/shower"
]

for category in categories:
    os.makedirs(image_dir + category, exist_ok=True)
    


# Loop through each category
for category in categories:
    print(f"\nStarting to scrape images from {category}...")

    count = 0
    scroll_count = 0
    max_images = 140
    seen_links = set()
    driver.get(f"https://www.pornpics.com{category}/")

    while count < max_images:
        # Wait for the images to load
        driver.implicitly_wait(2)

        # Parse the HTML
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = soup.find_all('a', class_='rel-link')

        for link in links:
            img = link.find('img')
            caption_url = ""
            if img:
                img_url = img.get('data-src')
                caption_url = img.get('alt')
                
                # Skip ignored URLs
                ignored_urls = [
                    'https://static.pornpics.com/style/img/1px.png',
                    'https://static.pornpics.com/style/img/asacp.svg',
                    'https://static.pornpics.com/style/img/logo.svg',
                    'https://static.pornpics.com/style/img/google-icon.svg',
                    'https://static.pornpics.com/style/img/logo-dark.svg'
                ]
                if img_url in ignored_urls or img_url in seen_links:
                    continue
                seen_links.add(img_url)
                # Save the image
                img_name = category[1:] + " " + f" {count + 1}.jpg"
                img_path = os.path.join(image_dir + category, img_name)

                # Use requests to fetch the image
                with open(img_path, 'wb') as img_file:
                    img_file.write(requests.get(img_url, headers=headers).content)
                count += 1
                print(f"Saved {img_name} from category: {category}")
                if count >= max_images:
                    break

                # save caption and catagory
                with open("nsfw_dir.csv", "a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)

                    
                    writer.writerow([category, f"image_{count}.jpg", f"caption {caption_url}"])
            scroll_count += 1
            if (scroll_count >= 10):
                # Scroll down to load more images
                driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
                driver.implicitly_wait(1)
                scroll_count = 0

    print(f"Finished! Saved {count} images from {category}.")

# Close the browser
driver.quit()
