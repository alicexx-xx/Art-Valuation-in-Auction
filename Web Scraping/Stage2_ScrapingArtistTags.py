"""
Script Summary — Stage 2: Artist Biography + Auction Entry Scraper

This script automates the second stage of a three-step scraping pipeline for Artsy.net.
It enriches a filtered list of artist URLs (from Stage 1) by extracting artist metadata
and collecting all auction entry tags associated with each artist.

What it does:
1. **Logs into Artsy.net** using undetected_chromedriver and stored credentials.
2. **Loads filtered artist URLs** from `results/artist_url_{letter}.pickle`.
3. For each artist:
   - Navigates to their artist page and extracts:
     • Name
     • Country and birth/death year
     • Biography
   - Navigates to the auction results page and:
     • Scrapes all `<a>` tags pointing to auction result entries (e.g. `/auction-result/...`)
     • Follows pagination to collect all such entries
4. **Saves results** in `results/auction_entries_{letter}.pickle` as a list of dictionaries,
   one per artist.

Output format per artist:
{
    'artist_id': "k1",
    'artist_name': "John Doe",
    'artist_country_year': "United States, b. 1950",
    'artist_biography': "...",
    'auction_tags': ["<a href='/auction-result/...'>...</a>", ...],
    'artist_url': "https://www.artsy.net/artist/..."
}

Notes:
- Supports multiple letters (e.g., `'k'`, `'l'`, etc.).
- Includes optional limit for testing on a subset of artists.
- Uses BeautifulSoup for parsing HTML and Selenium for interaction.
"""


import time
from bs4 import BeautifulSoup
import re
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from scraper_utils import append_list_of_dicts_to_pickle, read_list_of_dicts_from_appended_pickle, artsy_login


def create_uc_driver():
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=options, use_subprocess=True)


def close_popup_if_present(driver):
    try:
        popup_close_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[aria-label="Close"]'))
        )
        driver.execute_script("arguments[0].click();", popup_close_button)
        print("Popup closed.")
    except:
        pass

def get_artist_description(driver):
    try:
        close_popup_if_present(driver)

        # Try clicking the "Read more" span inside a button
        try:
            read_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Read more']]"))
            )
            driver.execute_script("arguments[0].click();", read_more_button)
            print("'Read more' clicked.")
            time.sleep(2)
        except Exception as e:
            print("No 'Read more' button found or clickable.")

        soup = BeautifulSoup(driver.page_source, "html.parser")

        artist_name = soup.select_one("h1.huMlRx")
        artist_name = artist_name.text.strip() if artist_name else "N/A"

        artist_country = soup.select_one("h2.irDwAE")
        artist_country = artist_country.text.strip() if artist_country else "N/A"

        bio = soup.select_one("div[class*='ArtistHeader__Bio']")
        artist_bio = bio.get_text(strip=True) if bio else "N/A"

        return artist_name, artist_country, artist_bio

    except Exception as e:
        print(f"Error extracting artist description: {e}")
        return "N/A", "N/A", "N/A"


def scrape_auction_tags(driver, artist_url):
    try:
        driver.get(
            artist_url + "/auction-results?categories%5B0%5D=Painting&categories%5B1%5D=Work%20on%20Paper&hide_upcoming=true&allow_empty_created_dates=true&currency=&include_estimate_range=false&include_unknown_prices=true&allow_unspecified_sale_dates=true"
        )
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/auction-result/"]'))
        )

        all_work = []
        soup = BeautifulSoup(driver.page_source, "html.parser")
        all_work.extend(soup.find_all("a", href=re.compile("/auction-result/")))

        while True:
            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-testid='next'][style*='opacity: 1']"))
                )
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(5)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                all_work.extend(soup.find_all("a", href=re.compile("/auction-result/")))
            except:
                break

        return [str(tag) for tag in all_work]

    except Exception as e:
        print(f"Failed to collect auction links for {artist_url}: {e}")
        return []


# --------- Main Stage 2 Runner ---------
if __name__ == "__main__":
    letters = ['k', 'l']
    email = "artauctionproject.57@gmail.com"
    password = "Artauctionproject2025!"
 
    driver = create_uc_driver()
    print("Launching browser...")
    driver.get("https://www.artsy.net/")
    print("Logging in...")
    artsy_login(driver, email, password)
    time.sleep(8)

    for letter in letters:
        print(f"\nStarting letter '{letter}'...\n")

        artist_urls = read_list_of_dicts_from_appended_pickle(f"results/artist_url_{letter}.pickle")

        ####### Optional limit for testing ########
        # artist_urls = artist_urls[5:15]
        # print(f"Limiting to {len(artist_urls)} artists for Stage 2")

        # Clear output for current letter
        output_path = f"results/auction_entries_{letter}.pickle"
        open(output_path, 'wb').close()

        for i, item in enumerate(artist_urls):
            url = item['url']
            artist_id = f"{letter}{i + 1}"

            print(f"\nScraping: {url} ({artist_id})")
            start_time = time.time()

            # Go to artist page first
            driver.get(url)
            time.sleep(3)

            artist_name, country_year, bio = get_artist_description(driver)
            auction_tags = scrape_auction_tags(driver, url)

            duration = time.time() - start_time
            print(f"Done: {artist_name} ({len(auction_tags)} results) in {duration:.2f} sec")

            append_list_of_dicts_to_pickle(
                [{
                    'artist_id': artist_id,
                    'artist_name': artist_name,
                    'artist_country_year': country_year,
                    'artist_biography': bio,
                    'auction_tags': auction_tags,
                    'artist_url': url
                }],
                output_path
            )

        print(f"\nFinished letter '{letter}' — results saved to {output_path}")

    driver.quit()
    print("\nStage 2 complete — all auction links saved!")