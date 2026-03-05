"""
correlate.py — Threat Correlation Module
─────────────────────────────────────────
Compares a new URL against all stored phishing URLs.
Finds similar domains, patterns and coordinated campaigns.
"""

import json
import re
import numpy as np
from urllib.parse    import urlparse
from difflib         import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from database        import get_all_phishing_urls, create_campaign


# ── 1. Domain similarity ──────────────────────────────────────────────────────
def domain_similarity(d1: str, d2: str) -> float:
    """String similarity between two domain names (0.0 – 1.0)."""
    return SequenceMatcher(None, d1.lower(), d2.lower()).ratio()


# ── 2. Feature vector similarity ─────────────────────────────────────────────
def feature_similarity(f1: list, f2: list) -> float:
    """Cosine similarity between two feature vectors (0.0 – 1.0)."""
    try:
        a = np.array(f1, dtype=float).reshape(1, -1)
        b = np.array(f2, dtype=float).reshape(1, -1)
        # Avoid division by zero
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(cosine_similarity(a, b)[0][0])
    except Exception:
        return 0.0


# ── 3. Lexical pattern matching ───────────────────────────────────────────────
PHISHING_PATTERNS = [
    r"secure[-.]",
    r"login[-.]",
    r"verify[-.]",
    r"account[-.]",
    r"update[-.]",
    r"banking[-.]",
    r"confirm[-.]",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",   # IP address
    r"[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\.",        # triple-hyphen domain
]

def shared_pattern_count(url1: str, url2: str) -> int:
    """Count how many phishing patterns both URLs share."""
    count = 0
    for pat in PHISHING_PATTERNS:
        if re.search(pat, url1, re.I) and re.search(pat, url2, re.I):
            count += 1
    return count


# ── 4. Main correlation function ──────────────────────────────────────────────
def correlate_url(new_url: str,
                  new_features: list,
                  threshold: float = 0.75) -> dict:
    """
    Compare a new URL against all stored phishing URLs.

    Returns:
        {
          similar_urls   : list of (id, url, score) tuples
          max_similarity : float
          campaign_id    : int or None
          correlation_label : str
        }
    """
    stored = get_all_phishing_urls(limit=500)

    if not stored:
        return {
            "similar_urls":        [],
            "max_similarity":      0.0,
            "campaign_id":         None,
            "correlation_label":   "NO HISTORY"
        }

    new_domain  = urlparse(new_url).netloc
    similar     = []

    for row in stored:
        old_url     = row["url"]
        old_domain  = row["domain"] or urlparse(old_url).netloc

        try:
            old_features = json.loads(row["features_json"])
        except Exception:
            old_features = []

        # Three similarity signals
        dom_sim  = domain_similarity(new_domain, old_domain)
        feat_sim = feature_similarity(new_features, old_features) if old_features else 0.0
        pat_sim  = shared_pattern_count(new_url, old_url) / len(PHISHING_PATTERNS)

        # Weighted combined score
        combined = (dom_sim * 0.40) + (feat_sim * 0.40) + (pat_sim * 0.20)

        if combined >= threshold:
            similar.append({
                "id":         row["id"],
                "url":        old_url,
                "score":      round(combined, 3),
                "risk_score": row["risk_score"]
            })

    # Sort by similarity descending
    similar.sort(key=lambda x: x["score"], reverse=True)
    similar = similar[:10]   # keep top 10

    max_sim = similar[0]["score"] if similar else 0.0

    # ── Campaign detection ────────────────────────────────────────────────────
    campaign_id = None
    if len(similar) >= 3:
        # 3+ similar URLs = likely coordinated campaign
        ids      = [s["id"] for s in similar]
        avg_risk = sum(s["risk_score"] for s in similar) / len(similar)
        pattern  = _extract_common_pattern(new_url, [s["url"] for s in similar])
        campaign_id = create_campaign(ids, pattern, avg_risk)

    # Label
    if max_sim >= 0.90:
        label = "HIGHLY CORRELATED — Likely same campaign"
    elif max_sim >= 0.75:
        label = "CORRELATED — Similar phishing pattern detected"
    elif max_sim >= 0.50:
        label = "WEAKLY CORRELATED — Some shared indicators"
    else:
        label = "ISOLATED — No strong correlation found"

    return {
        "similar_urls":      similar,
        "max_similarity":    round(max_sim, 3),
        "campaign_id":       campaign_id,
        "correlation_label": label
    }


def _extract_common_pattern(new_url: str, old_urls: list) -> str:
    """Extract the most common lexical pattern across similar URLs."""
    all_urls = [new_url] + old_urls
    for pat in PHISHING_PATTERNS:
        matches = [u for u in all_urls if re.search(pat, u, re.I)]
        if len(matches) >= len(all_urls) * 0.6:
            return pat
    return "mixed-pattern"