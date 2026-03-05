"""
risk_score.py — Dynamic Risk Scoring Module
────────────────────────────────────────────
Combines ML confidence + VirusTotal results +
similarity score into a 0–100 risk score with grade.
"""


# ── Weight configuration ──────────────────────────────────────────────────────
WEIGHTS = {
    "ml_confidence":   0.35,   # ML model probability
    "vt_score":        0.40,   # VirusTotal engine ratio
    "similarity":      0.15,   # Correlation with known threats
    "feature_penalty": 0.10,   # High-risk URL features
}

# ── Grade thresholds ──────────────────────────────────────────────────────────
GRADES = [
    (85, "CRITICAL", "#ff0000"),
    (65, "HIGH",     "#ff3d3d"),
    (40, "MEDIUM",   "#ffaa00"),
    (20, "LOW",      "#aaff00"),
    ( 0, "SAFE",     "#00e676"),
]


def compute_risk_score(ml_prediction: int,
                       ml_confidence: float,
                       vt_data:       dict  = None,
                       similarity:    float = 0.0,
                       features:      list  = None) -> dict:
    """
    Compute a 0–100 dynamic risk score.

    Parameters:
        ml_prediction  : 0 = legit, 1 = phishing
        ml_confidence  : probability from model (0.0–1.0)
        vt_data        : dict returned by vt_scan_url() or None
        similarity     : max correlation score (0.0–1.0)
        features       : list of 12 extracted URL features

    Returns:
        {
          score        : float  (0–100)
          grade        : str    (SAFE / LOW / MEDIUM / HIGH / CRITICAL)
          color        : str    (hex colour)
          breakdown    : dict   (contribution of each signal)
          explanation  : list   (human-readable reasons)
        }
    """

    explanation = []
    breakdown   = {}

    # ── Signal 1: ML confidence ───────────────────────────────────────────────
    if ml_prediction == 1:
        ml_signal = ml_confidence          # 0.0–1.0
    else:
        ml_signal = 1.0 - ml_confidence    # invert — low risk if legit

    ml_contribution = ml_signal * 100 * WEIGHTS["ml_confidence"]
    breakdown["ML Model"] = round(ml_contribution, 1)

    if ml_prediction == 1 and ml_confidence > 0.8:
        explanation.append(f"ML model flagged as phishing with {ml_confidence*100:.1f}% confidence")
    elif ml_prediction == 1:
        explanation.append(f"ML model flagged as phishing ({ml_confidence*100:.1f}% confidence)")

    # ── Signal 2: VirusTotal score ────────────────────────────────────────────
    vt_signal = 0.0
    if vt_data and not vt_data.get("error") and vt_data.get("total", 0) > 0:
        flagged   = vt_data.get("malicious", 0) + vt_data.get("suspicious", 0)
        total     = vt_data["total"]
        vt_signal = flagged / total

        if flagged > 10:
            explanation.append(f"VirusTotal: {flagged}/{total} engines flagged as malicious")
        elif flagged > 3:
            explanation.append(f"VirusTotal: {flagged} engines flagged as suspicious")
        elif flagged > 0:
            explanation.append(f"VirusTotal: {flagged} engine(s) raised concern")

    vt_contribution = vt_signal * 100 * WEIGHTS["vt_score"]
    breakdown["VirusTotal"] = round(vt_contribution, 1)

    # ── Signal 3: Similarity / correlation ───────────────────────────────────
    sim_contribution = similarity * 100 * WEIGHTS["similarity"]
    breakdown["Threat Correlation"] = round(sim_contribution, 1)

    if similarity >= 0.90:
        explanation.append("Highly similar to known phishing URLs in database")
    elif similarity >= 0.75:
        explanation.append("Correlated with previously detected phishing patterns")
    elif similarity >= 0.50:
        explanation.append("Shares some indicators with known threats")

    # ── Signal 4: Feature-based penalty ──────────────────────────────────────
    feat_penalty = 0.0
    if features and len(features) >= 12:
        (url_len, has_https, has_ip, has_at,
         dots, hyphens, dom_len, subdomains,
         susp_word, has_port, slashes, digits) = features[:12]

        penalties = []
        if has_ip     == 1: penalties.append(("IP address as domain", 0.30))
        if has_at     == 1: penalties.append(("@ symbol in URL",      0.25))
        if has_https  == 0: penalties.append(("No HTTPS",             0.20))
        if susp_word  == 1: penalties.append(("Suspicious keywords",  0.15))
        if url_len    > 100:penalties.append(("Extremely long URL",   0.10))
        if hyphens    > 3:  penalties.append(("Many hyphens",         0.10))
        if subdomains > 3:  penalties.append(("Excessive subdomains", 0.10))
        if digits     > 3:  penalties.append(("Many digits in domain",0.10))
        if has_port   == 1: penalties.append(("Non-standard port",    0.10))

        if penalties:
            feat_penalty = min(sum(p for _, p in penalties), 1.0)
            top_reasons  = [r for r, _ in penalties[:3]]
            explanation.append("Suspicious features: " + ", ".join(top_reasons))

    feat_contribution = feat_penalty * 100 * WEIGHTS["feature_penalty"]
    breakdown["URL Features"] = round(feat_contribution, 1)

    # ── Final score ───────────────────────────────────────────────────────────
    raw_score = (ml_contribution +
                 vt_contribution +
                 sim_contribution +
                 feat_contribution)

    # If ML says legit AND VT says clean → cap score at 30
    if ml_prediction == 0 and vt_signal == 0.0:
        raw_score = min(raw_score, 30.0)

    score = round(min(max(raw_score, 0), 100), 1)

    # ── Grade ─────────────────────────────────────────────────────────────────
    grade, color = "SAFE", "#00e676"
    for threshold, g, c in GRADES:
        if score >= threshold:
            grade, color = g, c
            break

    if not explanation:
        explanation.append("No significant threat indicators detected")

    return {
        "score":       score,
        "grade":       grade,
        "color":       color,
        "breakdown":   breakdown,
        "explanation": explanation,
    }


def grade_color(grade: str) -> str:
    """Return hex colour for a given grade string."""
    mapping = {
        "CRITICAL": "#ff0000",
        "HIGH":     "#ff3d3d",
        "MEDIUM":   "#ffaa00",
        "LOW":      "#aaff00",
        "SAFE":     "#00e676",
        "UNKNOWN":  "#7a8aaa",
    }
    return mapping.get(grade, "#7a8aaa")