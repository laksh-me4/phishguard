"""
database.py — Research-Grade Phishing Intelligence Database
────────────────────────────────────────────────────────────
Proper relational structure:
  urls       — one row per scanned URL
  features   — one row per URL with all 12 features as columns
  campaigns  — grouped phishing campaigns
  daily_stats— aggregated daily metrics
"""

import sqlite3
import json
import os
from datetime import datetime

DB_FILE = "phishing_intelligence.db"


def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all research-grade tables."""
    conn = get_connection()
    c    = conn.cursor()

    # ── 1. URLs table ─────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS urls (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        url             TEXT NOT NULL,
        domain          TEXT,
        scan_time       TEXT NOT NULL,

        -- ML results
        ml_prediction   INTEGER,        -- 0=legit, 1=phishing
        ml_confidence   REAL,           -- 0.0–1.0

        -- VirusTotal
        vt_malicious    INTEGER DEFAULT 0,
        vt_suspicious   INTEGER DEFAULT 0,
        vt_harmless     INTEGER DEFAULT 0,
        vt_total        INTEGER DEFAULT 0,

        -- Dynamic risk
        risk_score      REAL    DEFAULT 0,
        risk_grade      TEXT    DEFAULT 'UNKNOWN',

        -- Correlation
        max_similarity  REAL    DEFAULT 0,
        campaign_id     INTEGER DEFAULT NULL,

        FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
    )
    """)

    # ── 2. Features table — proper relational, one row per URL ───────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS features (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        url_id              INTEGER NOT NULL,

        url_length          INTEGER DEFAULT 0,
        has_https           INTEGER DEFAULT 0,
        has_ip              INTEGER DEFAULT 0,
        has_at_symbol       INTEGER DEFAULT 0,
        dot_count           INTEGER DEFAULT 0,
        hyphen_count        INTEGER DEFAULT 0,
        domain_length       INTEGER DEFAULT 0,
        subdomain_count     INTEGER DEFAULT 0,
        has_suspicious_word INTEGER DEFAULT 0,
        has_port            INTEGER DEFAULT 0,
        slash_count         INTEGER DEFAULT 0,
        digits_in_domain    INTEGER DEFAULT 0,

        FOREIGN KEY (url_id) REFERENCES urls(id)
    )
    """)

    # ── 3. Campaigns table — research-grade ───────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS campaigns (
        campaign_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_signature  TEXT,           -- common pattern/regex
        first_seen          TEXT NOT NULL,
        last_seen           TEXT NOT NULL,
        num_urls            INTEGER DEFAULT 1,
        avg_risk_score      REAL    DEFAULT 0,
        status              TEXT    DEFAULT 'ACTIVE'
    )
    """)

    # ── 4. Daily stats table ──────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS daily_stats (
        date                TEXT PRIMARY KEY,
        total_scanned       INTEGER DEFAULT 0,
        phishing_found      INTEGER DEFAULT 0,
        safe_found          INTEGER DEFAULT 0,
        avg_risk_score      REAL    DEFAULT 0,
        new_campaigns       INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Research-grade database initialised →", DB_FILE)


def insert_scan(url, domain, ml_prediction, ml_confidence,
                vt_malicious=0, vt_suspicious=0,
                vt_harmless=0, vt_total=0,
                risk_score=0, risk_grade="UNKNOWN",
                max_similarity=0.0, campaign_id=None,
                features_list=None):
    """
    Insert a full scan record.
    Writes to both `urls` and `features` tables.
    Returns the new url row id.
    """
    conn = get_connection()
    c    = conn.cursor()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert into urls
    c.execute("""
    INSERT INTO urls
        (url, domain, scan_time,
         ml_prediction, ml_confidence,
         vt_malicious, vt_suspicious, vt_harmless, vt_total,
         risk_score, risk_grade, max_similarity, campaign_id)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (url, domain, now,
          int(ml_prediction), float(ml_confidence or 0),
          int(vt_malicious), int(vt_suspicious),
          int(vt_harmless), int(vt_total),
          float(risk_score), risk_grade,
          float(max_similarity), campaign_id))

    url_id = c.lastrowid

    # Insert into features
    if features_list and len(features_list) >= 12:
        f = features_list
        c.execute("""
        INSERT INTO features
            (url_id, url_length, has_https, has_ip, has_at_symbol,
             dot_count, hyphen_count, domain_length, subdomain_count,
             has_suspicious_word, has_port, slash_count, digits_in_domain)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (url_id, f[0], f[1], f[2], f[3],
              f[4], f[5], f[6], f[7],
              f[8], f[9], f[10], f[11]))

    # Update daily stats
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("""
    INSERT INTO daily_stats
        (date, total_scanned, phishing_found, safe_found, avg_risk_score)
    VALUES (?,1,?,?,?)
    ON CONFLICT(date) DO UPDATE SET
        total_scanned  = total_scanned + 1,
        phishing_found = phishing_found + ?,
        safe_found     = safe_found + ?,
        avg_risk_score = (avg_risk_score + ?) / 2.0
    """, (today,
          int(ml_prediction), int(ml_prediction == 0), float(risk_score),
          int(ml_prediction), int(ml_prediction == 0), float(risk_score)))

    conn.commit()
    conn.close()
    return url_id


def create_campaign(url_ids, signature, avg_risk):
    """Create or update a campaign record."""
    conn = get_connection()
    c    = conn.cursor()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("""
    INSERT INTO campaigns
        (campaign_signature, first_seen, last_seen, num_urls, avg_risk_score)
    VALUES (?,?,?,?,?)
    """, (signature, now, now, len(url_ids), float(avg_risk)))
    campaign_id = c.lastrowid

    # Link URLs to campaign
    c.executemany(
        "UPDATE urls SET campaign_id=? WHERE id=?",
        [(campaign_id, uid) for uid in url_ids]
    )

    # Update daily stats
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("""
    INSERT INTO daily_stats (date, total_scanned, new_campaigns)
    VALUES (?,0,1)
    ON CONFLICT(date) DO UPDATE SET new_campaigns = new_campaigns + 1
    """, (today,))

    conn.commit()
    conn.close()
    return campaign_id


def update_campaign_last_seen(campaign_id):
    """Update last_seen and num_urls for existing campaign."""
    conn = get_connection()
    c    = conn.cursor()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
    UPDATE campaigns
    SET last_seen = ?, num_urls = num_urls + 1
    WHERE campaign_id = ?
    """, (now, campaign_id))
    conn.commit()
    conn.close()


def get_all_phishing_urls(limit=500):
    """Return phishing URLs joined with their features."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT u.id, u.url, u.domain, u.risk_score,
           f.url_length, f.has_https, f.has_ip, f.has_at_symbol,
           f.dot_count, f.hyphen_count, f.domain_length, f.subdomain_count,
           f.has_suspicious_word, f.has_port, f.slash_count, f.digits_in_domain
    FROM urls u
    LEFT JOIN features f ON f.url_id = u.id
    WHERE u.ml_prediction = 1
    ORDER BY u.scan_time DESC
    LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_recent_scans(limit=20):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT url, scan_time, ml_prediction, ml_confidence,
           risk_score, risk_grade, vt_malicious, vt_total,
           domain, max_similarity, campaign_id
    FROM urls
    ORDER BY scan_time DESC
    LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_dashboard_stats():
    conn = get_connection()
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM urls")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM urls WHERE ml_prediction=1")
    phishing = c.fetchone()[0]

    c.execute("SELECT AVG(risk_score) FROM urls WHERE ml_prediction=1")
    avg_risk = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM campaigns WHERE status='ACTIVE'")
    campaigns = c.fetchone()[0]

    c.execute("SELECT risk_grade, COUNT(*) cnt FROM urls GROUP BY risk_grade")
    grade_dist = {r["risk_grade"]: r["cnt"] for r in c.fetchall()}

    conn.close()
    return {"total": total, "phishing": phishing,
            "safe": total - phishing,
            "avg_risk": round(avg_risk, 1),
            "campaigns": campaigns,
            "grade_dist": grade_dist}


def get_daily_stats(days=14):
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT date, total_scanned, phishing_found, safe_found,
           avg_risk_score, new_campaigns
    FROM daily_stats
    ORDER BY date DESC
    LIMIT ?
    """, (days,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return list(reversed(rows))


def get_top_targeted_domains(limit=10):
    """Most frequently seen phishing domains."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT domain, COUNT(*) cnt, AVG(risk_score) avg_risk
    FROM urls
    WHERE ml_prediction=1 AND domain IS NOT NULL AND domain != ''
    GROUP BY domain
    ORDER BY cnt DESC
    LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_campaign_growth():
    """Campaign count over time for chart."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT DATE(first_seen) date, COUNT(*) new_campaigns
    FROM campaigns
    GROUP BY DATE(first_seen)
    ORDER BY date
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_feature_stats():
    """Average feature values for phishing vs legit — for research analysis."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT u.ml_prediction,
           AVG(f.url_length)          avg_url_length,
           AVG(f.dot_count)           avg_dot_count,
           AVG(f.hyphen_count)        avg_hyphen_count,
           AVG(f.subdomain_count)     avg_subdomain_count,
           AVG(f.domain_length)       avg_domain_length,
           AVG(f.has_https)           pct_https,
           AVG(f.has_suspicious_word) pct_suspicious_word,
           AVG(f.digits_in_domain)    avg_digits
    FROM urls u
    JOIN features f ON f.url_id = u.id
    GROUP BY u.ml_prediction
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Init on import ────────────────────────────────────────────────────────────
init_db()