"""
This script performs a two-stage web scraping pipeline to collect artist URLs and filter them based on auction result availability from Artsy.net.

 **Step 1: Collect Artist URLs**
    - Visits pages listing artists by first letter (e.g., A-Z).
    - Extracts artist profile URLs using Selenium and multithreading.
    - Stores these URLs in a queue for further processing.

🔹 **Step 2: Filter by Auction Result Count**
    - For each artist, navigates to their auction results page.
    - Checks whether they have ≥10 past/upcoming auction results.
    - Accepts or skips URLs based on this threshold.
    - Saves results to separate pickle files (`accepted`, `skipped`, `failures`) per letter.

🔹 **Key Features:**
    - Uses `undetected_chromedriver` to bypass bot detection.
    - Loads cookies for authenticated sessions.
    - Handles Cloudflare challenges and implements retry logic.
    - Thread-safe file writing with shared counters and logs.
    - Includes optional retry mechanism for failed artist URLs.

🔹 **Usage:**
    - Run the script as-is for a specific letter (currently `'w'`).
    - Customize `run_scraper(letter)` or `run_scraper_limited(letter, max_pages)` as needed.
    - Outputs are stored in the `results/` directory.

🔹 **Dependencies:**
    - Selenium, undetected-chromedriver, BeautifulSoup, tqdm, and standard Python libraries.
"""


import os
import time
import re
import random
import pickle
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from scraper_utils import append_list_of_dicts_to_pickle
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
import traceback
import pickle
import undetected_chromedriver as uc
from selenium.common.exceptions import NoSuchElementException


MAX_RETRIES = 3
TOTAL_MAX_RETRIES = 3
MIN_AUCTION_RESULTS = 10
COLLECTOR_THREADS = 3  
STAGE2_WORKERS = 1 
PATIENCE_EMPTY_PAGES = 5

write_lock = Lock()
print_lock = Lock()
artist_queue = Queue()

os.makedirs("results", exist_ok=True)


def create_driver():
    options = uc.ChromeOptions()
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(options=options, use_subprocess=True)

    # Load cookies
    try:
        cookies = pickle.load(open("cookies.pkl", "rb"))
        driver.get("https://www.artsy.net/")  # must navigate to domain before setting cookies
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                print(f"Skipping cookie (not settable yet): {cookie.get('name')} - {e}")
        driver.refresh()
    except Exception as e:
        print(f"Failed to load cookies: {e}")

    return driver

def is_cloudflare_challenge(driver):
    try:
        # Detect the hidden Cloudflare CAPTCHA input
        driver.find_element(By.CSS_SELECTOR, "input[name='cf-turnstile-response']")
        return True
    except NoSuchElementException:
        return False

def write_pickle_threadsafe(data, filepath):
    with write_lock:
        append_list_of_dicts_to_pickle(data, filepath)

def get_auction_result_count(driver, artist_url):
    try:
        driver.set_page_load_timeout(15)
        auction_url = artist_url + "/auction-results?categories%5B0%5D=Painting&categories%5B1%5D=Work%20on%20Paper&hide_upcoming=false&allow_empty_created_dates=true&currency=&include_estimate_range=false&include_unknown_prices=true&allow_unspecified_sale_dates=true"
        driver.get(auction_url)
        time.sleep(0.5)
        page_source = driver.page_source.lower()

        # Step 1: Skip if artist has no auction results
        no_results_patterns = [
            "there are currently no auction results for this artist",
            "there aren’t any works available that meet the following criteria",
            "there aren't any works available that meet the following criteria",
            "change your filter criteria to view more works",
            "clear all filters"
        ]
        for phrase in no_results_patterns:
            if phrase in page_source:
                return 0

        # Step 2: Find auction result sections
        result = driver.execute_script("""
            let data = { past: -1, upcoming: -1 };
            let divs = document.querySelectorAll('div');

            for (let i = 0; i < divs.length; i++) {
                let label = divs[i].innerText.trim().toLowerCase();
                
                if (label === 'past auctions') {
                    let next = divs[i + 1];
                    if (next && next.innerText) {
                        let cleaned = next.innerText.replace(/\\u00a0/g, ' ').trim();
                        let match = cleaned.match(/(\\d+)\\s+results?/i);
                        if (match) data.past = parseInt(match[1]);
                    }
                } else if (label === 'upcoming auctions') {
                    let next = divs[i + 1];
                    if (next && next.innerText) {
                        let cleaned = next.innerText.replace(/\\u00a0/g, ' ').trim();
                        let match = cleaned.match(/(\\d+)\\s+results?/i);
                        if (match) data.upcoming = parseInt(match[1]);
                    }
                }
            }

            return data;
        """)

        past = result.get("past", -1)
        upcoming = result.get("upcoming", -1)

        # Step 3: Apply acceptance thresholds
        if past >= 0 and upcoming >= 0:
            return past + upcoming if past + upcoming >= 10 else 0
        elif past >= 0:
            return past if past >= 10 else 0
        elif upcoming >= 0:
            return upcoming if upcoming >= 10 else 0
        else:
            return 0  # No sections found

    except Exception as e:
        # Step 4: Log error for retry
        print(f"Error in get_auction_result_count: {e}")
        with open("debug_failures.log", "a", encoding="utf-8") as f:
            f.write(f"Error in get_auction_result_count for {artist_url}: {str(e)}\n")
        return None

# Shared counters for tracking saved URLs
accepted_counter = defaultdict(int)
skipped_counter = defaultdict(int)

def check_artist(driver, url, letter, attempt=1):
    try:
        time.sleep(random.uniform(0.2, 0.5))  # anti-bot avoidance
        
        if is_cloudflare_challenge(driver):
            print(f"Cloudflare CAPTCHA detected. Skipping: {url}")
            skipped_counter[letter] += 1
            write_pickle_threadsafe([{"url": url}], f'results/artist_url_{letter}_skipped.pickle')
            return "SKIPPED"

        count = get_auction_result_count(driver, url)

        if count is None:
            raise RuntimeError("Auction count could not be determined (returned None)")

        if count >= MIN_AUCTION_RESULTS:
            accepted_counter[letter] += 1
            write_pickle_threadsafe([{"url": url}], f'results/artist_url_{letter}.pickle')
            if accepted_counter[letter] % 2 == 0:
                print(f"[ACCEPTED] {accepted_counter[letter]}: {url}")
            return "ACCEPTED"
        else:
            skipped_counter[letter] += 1
            write_pickle_threadsafe([{"url": url}], f'results/artist_url_{letter}_skipped.pickle')
            if skipped_counter[letter] % 5 == 0:
                print(f"[SKIPPED] {skipped_counter[letter]}: {url}")
            return "SKIPPED"

    except Exception as e:
        print(f"[{letter}] {url} (Attempt {attempt}) {type(e).__name__}: {e}")
        with open("debug_failures.log", "a", encoding="utf-8") as f:
            f.write(f"[{letter}] {url} (Attempt {attempt}) {str(e)}\n")

        time.sleep(2 ** attempt + random.uniform(0, 0.5))

        if attempt < MAX_RETRIES:
            return check_artist(driver, url, letter, attempt + 1)

        # Final attempt failed
        try:
            write_pickle_threadsafe([{"url": url}], f'results/artist_url_{letter}_failure.pickle')
            print(f"Final failure logged: {url}")
        except Exception as write_error:
            print(f"Failed to write to failure.pickle: {write_error}")

        return "FAILED"

def collect_artist_urls(driver, letter, page):
    try:
        driver.get(f'https://www.artsy.net/artists/artists-starting-with-{letter}?page={page}')
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[class*="ArtistsByLetter__Name"]'))
        )
        elements = driver.find_elements(By.CSS_SELECTOR, '[class*="ArtistsByLetter__Name"]')
        urls = list(set(e.get_attribute("href") for e in elements if e.get_attribute("href")))
        with print_lock:
            # print(f"[STAGE 1] Page {page} -> {len(urls)} artists")
            pass
        for url in urls:
            artist_queue.put((letter, url))
        return len(urls)
    except Exception as e:
        print(f"[ERROR] Failed to collect URLs on page {page}: {e}")
        return -1

# Shared counters
progress = defaultdict(int)

def stage2_worker(tqdm_bar):
    driver = create_driver()

    while True:
        try:
            letter, url = artist_queue.get(timeout=10)
        except:
            break  # Queue empty

        try:
            result = check_artist(driver, url, letter)
        except Exception as e:
            print(f"check_artist() crashed for {url}: {e}")
            result = "FAILED"

        if result == "ACCEPTED":
            progress['stored'] += 1
        elif result == "SKIPPED":
            progress['skipped'] += 1
        elif result == "FAILED":
            progress['failed'] += 1
        else:
            print(f"Unexpected result from check_artist(): {result}")

        artist_queue.task_done()

        with tqdm_bar.get_lock():
            tqdm_bar.update(1)
            tqdm_bar.set_description(
                f"Stage 2 ✅{progress['stored']} 🔴{progress['skipped']} ❌{progress['failed']}"
            )

    driver.quit()
    
def retry_failed_artists(letter):
    filepath = f"results/artist_url_{letter}_failure.pickle"
    if not os.path.exists(filepath):
        return

    try:
        with open(filepath, "rb") as f:
            failed_urls = []
            while True:
                try:
                    failed_urls.extend(pickle.load(f))
                except EOFError:
                    break
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return

    print(f"\nRetrying {len(failed_urls)} previously failed URLs for letter '{letter}'")

    driver = create_driver()
    for url in failed_urls:
        check_artist(driver, url, letter)
    driver.quit()

def run_scraper(letter):
    # Clearing Pickle Files
    open(f'results/artists_url_{letter}.pickle', 'wb').close()
    open(f'results/artist_url_{letter}_failure.pickle', 'wb').close()
    open(f'results/artist_url_{letter}_skipped.pickle', 'wb').close()

    processed_pages = set()
    failed_pages = set()
    empty_counter = 0
    page = 1
    last_page_detected = False

    drivers = [create_driver() for _ in range(COLLECTOR_THREADS)]  # Create a pool of drivers
    try:
        while not last_page_detected:
            with ThreadPoolExecutor(max_workers=COLLECTOR_THREADS) as executor:
                futures = {}
                for driver in drivers:
                    if page not in processed_pages and page not in failed_pages:
                        futures[executor.submit(collect_artist_urls, driver, letter, page)] = page
                        page += 1

                for future in as_completed(list(futures.keys())):
                    result = future.result()
                    page_number = futures[future]

                    if result == 0:
                        empty_counter += 1
                    elif result == -1:
                        failed_pages.add(page_number)
                    else:
                        empty_counter = 0
                        processed_pages.add(page_number)

                        # Detect the last page if fewer than 100 artists are found
                        if result < 100:
                            last_page_detected = True
                            print(f"Last page detected: Page {page_number} with {result} artists.")

                    del futures[future]

                if empty_counter >= PATIENCE_EMPTY_PAGES:
                    print("Hit 5 empty pages. Breaking...")
                    break
    finally:
        for driver in drivers:
            driver.quit()  # Ensure all drivers are closed after Stage 1

    # Retry failed pages with a limit
    retry_attempts = 0
    while failed_pages and retry_attempts < MAX_RETRIES:
        print(f"Retrying {len(failed_pages)} failed pages (Attempt {retry_attempts + 1}/{MAX_RETRIES})...")
        retry_attempts += 1
        drivers = [create_driver() for _ in range(COLLECTOR_THREADS)]  # Recreate drivers for retries
        try:
            with ThreadPoolExecutor(max_workers=COLLECTOR_THREADS) as executor:
                futures = {executor.submit(collect_artist_urls, driver, letter, page): page for driver, page in zip(drivers, failed_pages)}
                failed_pages.clear()

                for future in as_completed(list(futures.keys())):
                    result = future.result()
                    page_number = futures[future]

                    if result == 0:
                        empty_counter += 1
                    elif result == -1:
                        failed_pages.add(page_number)
                    else:
                        processed_pages.add(page_number)

                        # Detect the last page if fewer than 100 artists are found
                        if result < 100:
                            last_page_detected = True
                            print(f"Last page detected during retry: Page {page_number} with {result} artists.")

                    del futures[future]
        finally:
            for driver in drivers:
                driver.quit()

    print(f"\nStage 1 complete. Launching Stage 2 with {STAGE2_WORKERS} workers...")
    total_items = artist_queue.qsize()
    tqdm_bar = tqdm(total=total_items, desc="Stage 2", position=0)

    threads = [Thread(target=stage2_worker, args=(tqdm_bar,)) for _ in range(STAGE2_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    tqdm_bar.close()
    print(f"Done with letter '{letter}'\n{'-'*40}")

def run_scraper_limited(letter, max_pages=10):
    print(f"\n[LIMITED] Starting Stage 1 for '{letter}' (Max {max_pages} pages)")
    open(f'results/artist_url_{letter}.pickle', 'wb').close()

    processed_pages = set()
    failed_pages = set()
    empty_counter = 0
    page = 1

    drivers = [create_driver() for _ in range(COLLECTOR_THREADS)]  # Create a pool of drivers
    try:
        while page <= max_pages:
            with ThreadPoolExecutor(max_workers=COLLECTOR_THREADS) as executor:
                futures = {}
                for driver in drivers:
                    if page > max_pages:
                        break
                    if page not in processed_pages and page not in failed_pages:
                        futures[executor.submit(collect_artist_urls, driver, letter, page)] = page
                        page += 1

                for future in as_completed(list(futures.keys())):
                    result = future.result()
                    page_number = futures[future]

                    if result == 0:
                        empty_counter += 1
                    elif result == -1:
                        failed_pages.add(page_number)
                    else:
                        empty_counter = 0
                        processed_pages.add(page_number)

                    del futures[future]

                if empty_counter >= PATIENCE_EMPTY_PAGES:
                    print("Hit 5 empty pages. Breaking...")
                    break
    finally:
        for driver in drivers:
            driver.quit()  # Ensure all drivers are closed after Stage 1

    print(f"\n[LIMITED] Stage 1 complete. Launching Stage 2 with {STAGE2_WORKERS} workers...")
    total_items = artist_queue.qsize()
    tqdm_bar = tqdm(total=total_items, desc="Stage 2", position=0)

    threads = [Thread(target=stage2_worker, args=(tqdm_bar,)) for _ in range(STAGE2_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    tqdm_bar.close()
    print(f"[LIMITED] Done with letter '{letter}'\n{'-'*40}")


if __name__ == "__main__":
    for letter in ['w ']:
        print(f"\nStarting Stage 1 for '{letter}'")
        run_scraper(letter)
        # run_scraper_limited(letter, max_pages=20)
        retry_failed_artists(letter)
        print(f"Retried failures for letter '{letter}'")
