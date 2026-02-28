import re
from urllib.parse import urlparse

def extract_features(url):
    """
    Takes a URL string and returns a list of numbers (features).
    Each number represents something suspicious (or not) about the URL.
    """
    features = {}

    # 1. Length of the full URL
    # Phishing URLs tend to be very long to hide the real domain
    features['url_length'] = len(url)

    # 2. Does it use HTTPS?
    # Legitimate sites almost always use HTTPS (secure)
    features['has_https'] = 1 if url.startswith("https") else 0

    # 3. Does it have an IP address instead of a domain name?
    # e.g., http://192.168.1.1/login is very suspicious
    features['has_ip'] = 1 if re.search(
        r'(\d{1,3}\.){3}\d{1,3}', url) else 0

    # 4. Does it contain the '@' symbol?
    # Browsers ignore everything before @ in a URL — used to trick people
    # e.g., http://amazon.com@evil.com — you actually go to evil.com
    features['has_at_symbol'] = 1 if "@" in url else 0

    # 5. How many dots are in the URL?
    # Too many dots = suspicious subdomains
    # e.g., login.secure.amazon.com.evil.xyz
    features['dot_count'] = url.count(".")

    # 6. How many hyphens are in the domain?
    # Phishing domains often look like: amazon-secure-login.xyz
    try:
        domain = urlparse(url).netloc
        features['hyphen_count'] = domain.count("-")
    except:
        features['hyphen_count'] = 0

    # 7. How long is just the domain part?
    # Legitimate domains are usually short (amazon.com, google.com)
    try:
        features['domain_length'] = len(urlparse(url).netloc)
    except:
        features['domain_length'] = 0

    # 8. How many subdomains are there?
    # www.login.secure.amazon.xyz → 3 subdomains = suspicious
    try:
        parts = urlparse(url).netloc.split(".")
        features['subdomain_count'] = len(parts) - 2 if len(parts) > 2 else 0
    except:
        features['subdomain_count'] = 0

    # 9. Does the URL contain suspicious keywords?
    suspicious_words = ['login', 'secure', 'account', 'update', 
                        'banking', 'confirm', 'password', 'verify',
                        'signin', 'ebayisapi', 'webscr', 'paypal']
    features['has_suspicious_word'] = 1 if any(
        word in url.lower() for word in suspicious_words) else 0

    # 10. Does the URL have a port number?
    # e.g., http://amazon.com:8080/login — unusual for legit sites
    features['has_port'] = 1 if re.search(r':\d+', urlparse(url).netloc) else 0

    # 11. How many slashes in the URL path?
    # Very deep paths can indicate obfuscation
    try:
        features['slash_count'] = urlparse(url).path.count("/")
    except:
        features['slash_count'] = 0

    # 12. Does the domain have numbers in it?
    # e.g., amaz0n.com — phishers replace letters with numbers
    try:
        features['digits_in_domain'] = sum(
            c.isdigit() for c in urlparse(url).netloc)
    except:
        features['digits_in_domain'] = 0

    return list(features.values())


# Column names matching the order above — needed for the ML model
FEATURE_NAMES = [
    'url_length',
    'has_https',
    'has_ip',
    'has_at_symbol',
    'dot_count',
    'hyphen_count',
    'domain_length',
    'subdomain_count',
    'has_suspicious_word',
    'has_port',
    'slash_count',
    'digits_in_domain'
]


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.amazon.com/products/shoes",
        "http://amaz0n-login-secure.xyz/account/verify?user=123"
    ]
    for url in test_urls:
        print(f"\nURL: {url}")
        feats = extract_features(url)
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"  {name:30s} → {val}")
