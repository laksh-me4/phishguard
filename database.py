"""
database.py — Phishing Intelligence Database
─────────────────────────────────────────────
Stores every scanned URL with features, ML score,
VT results, risk score and timestamp.
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
    """Create all tables if they don't exist."""
    conn = get_connection()
    c    = conn.cursor()

    # ── Main URL intelligence table ──────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS url_intelligence (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        url             TEXT NOT NULL,
        scanned_at      TEXT NOT NULL,

        -- ML results
        ml_prediction   INTEGER,        -- 0=legit, 1=phishing
        ml_confidence   REAL,           -- 0.0 - 1.0

        -- VirusTotal results
        vt_malicious    INTEGER DEFAULT 0,
        vt_suspicious   INTEGER DEFAULT 0,
        vt_harmless     INTEGER DEFAULT 0,
        vt_total        INTEGER DEFAULT 0,

        -- Dynamic risk score
        risk_score      REAL DEFAULT 0,     -- 0-100
        risk_grade      TEXT DEFAULT 'UNKNOWN',  -- LOW/MEDIUM/HIGH/CRITICAL

        -- URL features (stored as JSON)
        features_json   TEXT,

        -- Domain extracted from URL
        domain          TEXT,

        -- Correlation results
        similar_urls    TEXT,   -- JSON list of similar URL IDs
        campaign_id     INTEGER DEFAULT NULL
    )
    """)

    # ── Campaign table — groups related phishing URLs ────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS campaigns (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at     TEXT NOT NULL,
        url_count       INTEGER DEFAULT 1,
        avg_risk_score  REAL DEFAULT 0,
        common_pattern  TEXT,
        status          TEXT DEFAULT 'ACTIVE'  -- ACTIVE / RESOLVED
    )
    """)

    # ── Daily stats table ────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS daily_stats (
        date            TEXT PRIMARY KEY,
        total_scanned   INTEGER DEFAULT 0,
        phishing_found  INTEGER DEFAULT 0,
        avg_risk_score  REAL DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialised →", DB_FILE)


def insert_url(url, ml_prediction, ml_confidence,
               vt_malicious=0, vt_suspicious=0,
               vt_harmless=0, vt_total=0,
               risk_score=0, risk_grade="UNKNOWN",
               features=None, domain="",
               similar_urls=None, campaign_id=None):
    """Insert a scanned URL record into the database."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    INSERT INTO url_intelligence
        (url, scanned_at, ml_prediction, ml_confidence,
         vt_malicious, vt_suspicious, vt_harmless, vt_total,
         risk_score, risk_grade, features_json,
         domain, similar_urls, campaign_id)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        url,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        int(ml_prediction),
        float(ml_confidence or 0),
        int(vt_malicious),
        int(vt_suspicious),
        int(vt_harmless),
        int(vt_total),
        float(risk_score),
        risk_grade,
        json.dumps(features or []),
        domain,
        json.dumps(similar_urls or []),
        campaign_id
    ))

    # Update daily stats
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("""
    INSERT INTO daily_stats (date, total_scanned, phishing_found, avg_risk_score)
    VALUES (?, 1, ?, ?)
    ON CONFLICT(date) DO UPDATE SET
        total_scanned  = total_scanned + 1,
        phishing_found = phishing_found + ?,
        avg_risk_score = (avg_risk_score + ?) / 2
    """, (today,
          int(ml_prediction), float(risk_score),
          int(ml_prediction), float(risk_score)))

    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def get_all_phishing_urls(limit=500):
    """Return all URLs flagged as phishing — used for correlation."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT id, url, domain, features_json, risk_score
    FROM url_intelligence
    WHERE ml_prediction = 1
    ORDER BY scanned_at DESC
    LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_recent_scans(limit=20):
    """Return the most recent scans for the dashboard."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT url, scanned_at, ml_prediction, ml_confidence,
           risk_score, risk_grade, vt_malicious, vt_total, domain
    FROM url_intelligence
    ORDER BY scanned_at DESC
    LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_dashboard_stats():
    """Return summary statistics for the intelligence dashboard."""
    conn = get_connection()
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM url_intelligence")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM url_intelligence WHERE ml_prediction=1")
    phishing = c.fetchone()[0]

    c.execute("SELECT AVG(risk_score) FROM url_intelligence WHERE ml_prediction=1")
    avg_risk = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM campaigns WHERE status='ACTIVE'")
    campaigns = c.fetchone()[0]

    c.execute("""
    SELECT risk_grade, COUNT(*) as cnt
    FROM url_intelligence
    GROUP BY risk_grade
    """)
    grade_dist = {r["risk_grade"]: r["cnt"] for r in c.fetchall()}

    conn.close()
    return {
        "total":      total,
        "phishing":   phishing,
        "safe":       total - phishing,
        "avg_risk":   round(avg_risk, 1),
        "campaigns":  campaigns,
        "grade_dist": grade_dist
    }


def get_daily_stats(days=7):
    """Return daily scan stats for the last N days."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    SELECT date, total_scanned, phishing_found, avg_risk_score
    FROM daily_stats
    ORDER BY date DESC
    LIMIT ?
    """, (days,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def create_campaign(urls_ids, common_pattern, avg_risk):
    """Create a new phishing campaign record."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
    INSERT INTO campaigns (detected_at, url_count, avg_risk_score, common_pattern)
    VALUES (?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          len(urls_ids), avg_risk, common_pattern))
    campaign_id = c.lastrowid

    # Link URLs to this campaign
    c.executemany("""
    UPDATE url_intelligence SET campaign_id=? WHERE id=?
    """, [(campaign_id, uid) for uid in urls_ids])

    conn.commit()
    conn.close()
    return campaign_id


# ── Init on import ────────────────────────────────────────────────────────────
init_db()