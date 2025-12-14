"""
Preprocessing module for News Headline Classifier.
Exposes prepare_data() function as required by backend.
Handles both training data (with headlines) and test data (URL-only).
For URL-only data, extracts pseudo-headlines from URL slugs (no scraping allowed).
"""
import pandas as pd
import numpy as np
import re
import string
from urllib.parse import urlparse


def clean_text(text):
    """
    Clean and normalize text for better model performance.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # convert to lowercase
    text = text.lower()
    
    # rm URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # rm extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def normalize_url(url):
    """remove .print suffix if present"""
    if isinstance(url, str) and url.endswith(".print"):
        return url[:-6]
    return url


def extract_headline_from_url(url):
    """
    Extract pseudo-headline from URL slug.
    Converts URL path segments into readable text.
    Example: 'trump-calls-out-biden' -> 'trump calls out biden'
    
    This is used when scraping is not allowed - we extract meaningful
    text from the URL structure itself.
    """
    if pd.isna(url) or not isinstance(url, str):
        return ""
    
    url = normalize_url(url)
    
    try:
        # parse URL to get path
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        # split into segments and filter out empty/invalid ones
        segments = [s for s in path.split('/') if s and len(s) > 1]
        
        # skip common non-content segments
        skip_segments = {'www', 'http', 'https', 'com', 'net', 'org', 
                        'foxnews', 'nbcnews', 'news', 'select', 'shows',
                        'video', 'videos', 'category', 'live', 'podcasts',
                        'media', 'weather', 'about', 'contact', 'politics',
                        'entertainment', 'sports', 'lifestyle', 'health',
                        'world', 'us', 'business', 'tech', 'science'}
        
        # get meaningful segments (usually the last few contain the article slug)
        meaningful_segments = [s for s in segments if s not in skip_segments]
        
        if not meaningful_segments:
            return ""
        
        # take the last segment(s) which usually contain the article title
        # for fox: usually just last segment
        # for nbc: sometimes last segment has ID, so take last 2-3 segments
        if 'nbcnews.com' in url.lower():
            # nbc URLs often have format: /section/subsection/article-title-rcna123456
            # take last 2-3 segments, skip the ID segment
            slug_segments = meaningful_segments[-3:] if len(meaningful_segments) >= 3 else meaningful_segments
            # filter out segments that look like IDs (rcna123456, etc)
            slug_segments = [s for s in slug_segments if not re.match(r'^rcna\d+$', s, re.I)]
        else:
            # fox news: usually just the last segment
            slug_segments = meaningful_segments[-1:]
        
        if not slug_segments:
            return ""
        
        # combine segments and convert hyphens to spaces
        combined = ' '.join(slug_segments)
        headline = combined.replace('-', ' ').replace('_', ' ')
        
        # clean up: remove extra spaces, numbers at end (like rcna IDs)
        headline = re.sub(r'\s+', ' ', headline)
        headline = re.sub(r'\s+rcna\d+\s*$', '', headline, flags=re.I)
        headline = headline.strip()
        
        return headline if len(headline) > 5 else ""  # minimum length check
        
    except Exception:
        return ""


def infer_source_from_url(url):
    """
    Infer news source from URL domain.
    Returns 'fox' or 'nbc' based on domain.
    """
    if pd.isna(url) or not isinstance(url, str):
        return None
    url_lower = url.lower()
    if 'foxnews.com' in url_lower:
        return 'fox'
    elif 'nbcnews.com' in url_lower:
        return 'nbc'
    return None


def prepare_data(csv_path: str):
    """
    Backend-aligned prepare_data():
    - X: URL-derived text (url slug parsing, not headline)
    - y: derived from URL domain (fox=0, nbc=1)
    - No scraping, no reliance on 'headline' column. (for explatory discussion only)
    """
    df = pd.read_csv(csv_path)

    if "url" not in df.columns:
        raise ValueError(f"CSV must contain a 'url' column. Found columns: {df.columns.tolist()}")

    # normalize urls
    urls = df["url"].astype(str).apply(normalize_url)

    # labels from domain (drop non-fox/nbc)
    sources = urls.apply(infer_source_from_url)  # 'fox'/'nbc'/None
    keep = sources.notna()
    urls = urls[keep].reset_index(drop=True)
    sources = sources[keep].reset_index(drop=True)

    # X from URL slug text
    X = urls.apply(extract_headline_from_url).tolist()
    X = [clean_text(x) for x in X]

    # drop empties after cleaning
    keep2 = [len(x) > 0 for x in X]
    X = [x for x, k in zip(X, keep2) if k]
    sources = sources[[i for i, k in enumerate(keep2) if k]].reset_index(drop=True)

    y = (sources != "fox").astype(int).tolist()

    return X, y
