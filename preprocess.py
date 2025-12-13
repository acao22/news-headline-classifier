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
    Prepare data from CSV file for model training/inference.
    
    Handles two cases:
    1. Training data: CSV has 'headline' and 'source' columns
    2. Test data: CSV has only 'url' column (will extract pseudo-headlines from URL slugs)
    
    Args:
        csv_path: Path to CSV file with either:
                  - columns: url, source, headline (training data)
                  - column: url (test data, will extract headlines from URL structure)
        
    Returns:
        X: List/array of preprocessed headline strings
        y: List/array of labels (0 for 'fox', 1 for 'nbc')
           If source can't be determined, returns list of 0s
    """
    # load CSV
    df = pd.read_csv(csv_path)
    
    # check if we have headlines or need to extract from URLs
    if 'headline' in df.columns:
        # training data format - headlines already provided
        df = df.dropna(subset=['headline'])
        headlines = df['headline'].tolist()
    elif 'url' in df.columns:
        # test data format - extract pseudo-headlines from URL slugs
        # (scraping not allowed, so we parse the URL structure)
        print(f"extracting headlines from {len(df)} URLs...")
        headlines = []
        valid_indices = []
        for idx, url in enumerate(df['url']):
            headline = extract_headline_from_url(url)
            if headline:  # only keep if we successfully extracted
                headlines.append(headline)
                valid_indices.append(idx)
        # filter df to only rows where we got headlines
        df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"successfully extracted {len(headlines)} headlines from URLs")
    else:
        raise ValueError("CSV must have either 'headline' or 'url' column")
    
    # clean headlines
    headlines = [clean_text(h) for h in headlines]
    
    # remove empty headlines after cleaning
    valid_mask = [len(h) > 0 for h in headlines]
    headlines = [h for h, valid in zip(headlines, valid_mask) if valid]
    df = df[valid_mask].reset_index(drop=True)
    
    # get labels - either from 'source' column or infer from URL
    if 'source' in df.columns:
        # use provided source labels
        y = (df['source'] != 'fox').astype(int).tolist()
    elif 'url' in df.columns:
        # infer source from URL domain
        sources = df['url'].apply(infer_source_from_url)
        y = (sources != 'fox').astype(int).tolist()
    else:
        # can't determine source, return zeros (will be wrong but allows model to run)
        y = [0] * len(headlines)
    
    return headlines, y
