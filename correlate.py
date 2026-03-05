"""
correlate.py — Threat Correlation Engine
─────────────────────────────────────────
Implements BOTH:
  • Cosine Similarity  — on feature vectors
  • Jaccard Similarity — on URL token sets
Combined score detects coordinated phishing campaigns.
"""

import re
import numpy as np
from urllib.parse             import urlparse
from difflib                  import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from database                 import (get_all_phishing_urls,
                                      create_campaign,
                                      update_campaign_last_seen)


# ── Jaccard Similarity ────────────────────────────────────────────────────────
def jaccard_similarity(url1: str, url2: str) -> float:
    """
    Jaccard similarity between two URLs.
    Tokenises by splitting on non-alphanumeric characters.
    J(A,B) = |A ∩ B| / |A ∪ B|

    Example:
        url1 = "http://amaz0n-login.xyz/account/verify"
        url2 = "http://amazon-secure.xyz/account/confirm"
        Shared tokens: {http, xyz, account} → J = 3/10 = 0.30
    """
    def tokenise(url):
        return set(re.split(r'[^a-z0-9]', url.lower())) - {'', 'http', 'https', 'www'}

    a = tokenise(url1)
    b = tokenise(url2)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union        = len(a | b)
    return intersection / union


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def cosine_feature_similarity(f1: list, f2: list) -> float:
    """
    Cosine similarity between two 12-dim feature vectors.
    Measures how similar the URL structures are numerically.
    """
    try:
        a = np.array(f1, dtype=float).reshape(1, -1)
        b = np.array(f2, dtype=float).reshape(1, -1)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(cosine_similarity(a, b)[0][0])
    except Exception:
        return 0.0


# ── Domain String Similarity ──────────────────────────────────────────────────
def domain_similarity(d1: str, d2: str) -> float:
    """Levenshtein-based string similarity between domain names."""
    return SequenceMatcher(None, d1.lower(), d2.lower()).ratio()


# ── Lexical Pattern Matching ──────────────────────────────────────────────────
PHISHING_PATTERNS = [
    r"secure[-.]",    r"login[-.]",   r"verify[-.]",
    r"account[-.]",   r"update[-.]",  r"banking[-.]",
    r"confirm[-.]",   r"paypal",      r"ebay",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # raw IP
    r"[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\.",      # triple-hyphen domain
]

def shared_pattern_score(url1: str, url2: str) -> float:
    """Fraction of phishing patterns shared by both URLs."""
    matches = sum(
        1 for p in PHISHING_PATTERNS
        if re.search(p, url1, re.I) and re.search(p, url2, re.I)
    )
    return matches / len(PHISHING_PATTERNS)


# ── Combined Similarity Score ─────────────────────────────────────────────────
def combined_similarity(new_url, new_features,
                         old_url, old_features) -> dict:
    """
    Compute all similarity signals and return weighted combined score.

    Weights (chosen to reflect research literature):
        Cosine   (feature vectors)  — 35%
        Jaccard  (URL tokens)       — 35%
        Domain   (string edit)      — 20%
        Patterns (lexical)          — 10%
    """
    new_domain = urlparse(new_url).netloc
    old_domain = urlparse(old_url).netloc

    cosine  = cosine_feature_similarity(new_features, old_features)
    jaccard = jaccard_similarity(new_url, old_url)
    domain  = domain_similarity(new_domain, old_domain)
    pattern = shared_pattern_score(new_url, old_url)

    combined = (cosine  * 0.35 +
                jaccard * 0.35 +
                domain  * 0.20 +
                pattern * 0.10)

    return {
        "combined": round(combined, 4),
        "cosine":   round(cosine,   4),
        "jaccard":  round(jaccard,  4),
        "domain":   round(domain,   4),
        "pattern":  round(pattern,  4),
    }


# ── Main Correlation Function ─────────────────────────────────────────────────
def correlate_url(new_url: str,
                  new_features: list,
                  threshold: float = 0.65) -> dict:
    """
    Compare a new URL against all stored phishing URLs.

    Returns:
        similar_urls      — top matching URLs with scores
        max_similarity    — highest combined score found
        similarity_detail — breakdown of signals for top match
        campaign_id       — if campaign detected, its ID
        correlation_label — human-readable verdict
    """
    stored = get_all_phishing_urls(limit=500)

    if not stored:
        return {
            "similar_urls":      [],
            "max_similarity":    0.0,
            "similarity_detail": {},
            "campaign_id":       None,
            "correlation_label": "NO HISTORY — first scan in database"
        }

    similar = []

    for row in stored:
        old_url      = row["url"]
        old_features = [
            row.get("url_length", 0),      row.get("has_https", 0),
            row.get("has_ip", 0),           row.get("has_at_symbol", 0),
            row.get("dot_count", 0),        row.get("hyphen_count", 0),
            row.get("domain_length", 0),    row.get("subdomain_count", 0),
            row.get("has_suspicious_word",0),row.get("has_port", 0),
            row.get("slash_count", 0),      row.get("digits_in_domain", 0),
        ]

        sim = combined_similarity(new_url, new_features,
                                   old_url, old_features)

        if sim["combined"] >= threshold:
            similar.append({
                "id":         row["id"],
                "url":        old_url,
                "risk_score": row["risk_score"],
                **sim
            })

    # Sort by combined score
    similar.sort(key=lambda x: x["combined"], reverse=True)
    similar = similar[:10]

    max_sim = similar[0]["combined"] if similar else 0.0
    top_detail = {
        k: similar[0][k]
        for k in ("cosine","jaccard","domain","pattern")
    } if similar else {}

    # ── Campaign detection ────────────────────────────────────────────────────
    campaign_id = None
    if len(similar) >= 3:
        ids      = [s["id"] for s in similar]
        avg_risk = sum(s["risk_score"] for s in similar) / len(similar)
        sig      = _extract_signature(new_url, [s["url"] for s in similar])

        # Check if URLs already belong to a campaign
        existing_campaigns = [s.get("campaign_id") for s in similar
                               if s.get("campaign_id")]
        if existing_campaigns:
            campaign_id = existing_campaigns[0]
            update_campaign_last_seen(campaign_id)
        else:
            campaign_id = create_campaign(ids, sig, avg_risk)

    # ── Label ─────────────────────────────────────────────────────────────────
    if max_sim >= 0.90:
        label = "🚨 HIGHLY CORRELATED — Almost certainly same campaign"
    elif max_sim >= 0.75:
        label = "⚠️ STRONGLY CORRELATED — Same phishing pattern"
    elif max_sim >= 0.65:
        label = "🟡 CORRELATED — Shared phishing indicators"
    elif max_sim >= 0.40:
        label = "🔵 WEAKLY CORRELATED — Some shared tokens"
    else:
        label = "✅ ISOLATED — No strong correlation found"

    return {
        "similar_urls":      similar,
        "max_similarity":    round(max_sim, 4),
        "similarity_detail": top_detail,
        "campaign_id":       campaign_id,
        "correlation_label": label,
    }


def _extract_signature(new_url: str, old_urls: list) -> str:
    """Find the most common pattern across similar URLs."""
    all_urls = [new_url] + old_urls
    for pat in PHISHING_PATTERNS:
        hits = [u for u in all_urls if re.search(pat, u, re.I)]
        if len(hits) >= len(all_urls) * 0.6:
            return pat
    return "mixed-pattern"