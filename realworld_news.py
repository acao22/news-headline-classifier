## Extract External Validation Data from the Real World

import pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import time # Recommended for ethical scraping

# Configuration
MAX_WORKERS = 32         # Max number of simultaneous requests (threads)
MAX_URLS_PER_PAGE = 300  # Max links to extract from each section page
TOTAL_URLS_TARGET = 3000 # Stop collecting once this limit is passed

# Define Section URLs for multi-page scraping (MUST BE EXTENDED TO REACH 1000)
FOX_SECTION_URLS = [
    "https://www.foxnews.com/politics",
    "https://www.foxnews.com/politics/senate",
    "https://www.foxnews.com/politics/house"
    "https://www.foxnews.com/sports",
    "https://www.foxnews.com/us",
     "https://www.foxnews.com/us/crime",
    "https://www.foxnews.com/world",
    "https://www.foxnews.com/opinion",
    "https://www.foxnews.com/media",
    "https://www.foxnews.com/lifestyle",
    "https://www.foxnews.com/health",
    "https://www.foxnews.com/entertainment",
    "https://www.foxnews.com/category/tech/artificial-intelligence",

]

NBC_SECTION_URLS = [
    "https://www.nbcnews.com/politics",
    "https://www.nbcnews.com/politics/congress",
    "https://www.nbcnews.com/politics/white-house",
    "https://www.nbcnews.com/meet-the-press",
    "https://www.nbcnews.com/business",
    "https://www.nbcnews.com/sports",
    "https://www.nbcnews.com/us-news",
    "https://www.nbcnews.com/news/crime-courts",
    "https://www.nbcnews.com/culture-matters",
    "https://www.nbcnews.com/world",
    "https://www.nbcnews.com/health",
    "https://www.nbcnews.com/new-york",
    "https://www.nbcnews.com/los-angeles",
]

#  Existing URL Filtering Functions 

def is_fox_article_url(url: str) -> bool:
    # ... (Your existing Fox URL filtering logic) ...
    if "foxnews.com" not in url:
        return False
    url = url.split("?", 1)[0].split("#", 1)[0]
    blacklist_parts = ["/video/", "/videos/", "/category/", "/live/", "/shows/", "/podcasts/", "/media/", "/weather/", "/about/", "/contact/"]
    if any(part in url for part in blacklist_parts):
        return False
    if not url.startswith("https://www.foxnews.com"):
        return False
    path = url.replace("https://www.foxnews.com", "")
    if path.count("/") < 2:
        return False
    last_segment = path.rstrip("/").split("/")[-1]
    if "-" not in last_segment:
        return False
    return True

def is_nbc_article_url(url: str) -> bool:
    # ... (Your existing NBC URL filtering logic) ...
    if "nbcnews.com" not in url:
        return False
    url = url.split("?", 1)[0].split("#", 1)[0]
    blacklist_parts = ["/video/", "/meet-the-press/", "/live/", "/weather/", "/careers/", "/about/"]
    if any(part in url for part in blacklist_parts):
        return False
    if not url.startswith("https://www.nbcnews.com"):
        return False
    path = url.replace("https://www.nbcnews.com", "")
    if path.count("/") < 2:
        return False
    last_segment = path.rstrip("/").split("/")[-1]
    if "-" not in last_segment:
        return False
    return True

# URL Collection Functions

def get_fox_article_urls_from_page(page_url, max_links):
    # Added random delay for ethical scraping
    time.sleep(1) 
    resp = requests.get(page_url, timeout=8)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.foxnews.com" + href
        if is_fox_article_url(href):
            urls.add(href)
        if len(urls) >= max_links:
            break
    return list(urls)

def get_nbc_article_urls_from_page(page_url, max_links):
    time.sleep(1) # Added random delay for ethical scraping
    resp = requests.get(page_url, timeout=8)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.nbcnews.com" + href
        if is_nbc_article_url(href):
            urls.add(href)
        if len(urls) >= max_links:
            break
    return list(urls)

# Concurrent URL Collection Function

def get_urls_concurrently(page_urls, url_extractor_func):
    all_urls = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the URL extractor function to all section pages
        futures = [executor.submit(url_extractor_func, url, MAX_URLS_PER_PAGE) for url in page_urls]
        
        for future in concurrent.futures.as_completed(futures):
            # Check if we have enough URLs before processing next result
            if len(all_urls) >= TOTAL_URLS_TARGET:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                urls_from_page = future.result()
                initial_count = len(all_urls)
                all_urls.update(urls_from_page)
                newly_added = len(all_urls) - initial_count
                print(f"Collected {newly_added} NEW URLs from a page. Total collected: {len(all_urls)}")
            except Exception as exc:
                print(f'URL collection generated an exception: {exc}')
    return list(all_urls)[:TOTAL_URLS_TARGET]


# Headline Scraping Function

def normalize_url(url):
    if url.endswith(".print"):
        return url[:-6]
    return url

def scrape_headline(raw_url):
    # ... (Your existing headline scraping logic) ...
    url = normalize_url(raw_url)
    try:
        # Added random delay for ethical scraping
        time.sleep(0.5) 
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        fox_like = soup.find("h1", class_="headline")
        if fox_like and fox_like.get_text(strip=True):
            return fox_like.get_text(strip=True)
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)
        return ""
    except:
        return ""

# Concurrent Headline Scraping Function

def scrape_headlines_concurrently(url_list):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(scrape_headline, url): url for url in url_list}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                headline = future.result()
                if headline:
                    results.append({"url": url, "headline": headline})
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
    return results


# Execute URL Collection

print(f"--- Starting Concurrent URL Collection (Target: {TOTAL_URLS_TARGET}) ---")
fox_urls_new = get_urls_concurrently(FOX_SECTION_URLS, get_fox_article_urls_from_page)
nbc_urls_new = get_urls_concurrently(NBC_SECTION_URLS, get_nbc_article_urls_from_page)

final_urls_list = fox_urls_new + nbc_urls_new
final_urls_list = list(set(final_urls_list))[:TOTAL_URLS_TARGET] # Ensure uniqueness and final limit

print(f"--- Total UNIQUE URLs collected: {len(final_urls_list)} ---")
print(f"--- Total UNIQUE FOX URLs collected: {len(fox_urls_new)} ---")
print(f"--- Total UNIQUE NBC URLs collected: {len(nbc_urls_new)} ---")



# Export Only URLs to CSV

# Create the final DataFrame with just the URLs
urls_df = pd.DataFrame({'url': final_urls_list})

# Export the URLs DataFrame to CSV
print("Exporting URL list to external_urls_only.csv")
urls_df.to_csv("external_urls_only.csv", index=False)

# Optional: Scrape headlines for full data set (using the second concurrent step)
# Uncomment the block below if you also need the headlines, not just the URLs.
"""
print("\n Concurrent Headline Scraping")
scraped_results = scrape_headlines_concurrently(final_urls_list)

web_rows = []
for row in scraped_results:
    source = 'fox' if 'foxnews.com' in row['url'] else 'nbc'
    web_rows.append({
        "url": row["url"], 
        "source": source, 
        "headline": row["headline"]
    })

web_df = pd.DataFrame(web_rows)
web_df = web_df[web_df["headline"].str.len() > 0].reset_index(drop=True)

# Export the full DataFrame
print(f"Exporting full dataset ({len(web_df)} rows) to external_val_dataset.csv")
web_df.to_csv("external_val_dataset.csv", index=False)
"""