"""
app.py  —  PhishGuard  |  VirusTotal + ML Detection
Run with:  streamlit run app.py
"""

import streamlit as st
import pickle
import numpy as np
import os
import time
import base64
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from features import extract_features, FEATURE_NAMES

load_dotenv()
VT_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")

st.set_page_config(
    page_title="PhishGuard — URL Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050810 !important;
    font-family: 'Syne', sans-serif;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid #1a2040;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: #2a3a6a; border-radius: 4px; }

.hero { text-align: center; padding: 3rem 1rem 2rem; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0d1f4a, #1a3a7a);
    border: 1px solid #2a5aaa; color: #5a9aff;
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 3px; padding: 6px 18px; border-radius: 20px;
    margin-bottom: 1.5rem; text-transform: uppercase;
}
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem); font-weight: 800;
    line-height: 1.05; letter-spacing: -2px;
    background: linear-gradient(135deg, #ffffff 0%, #7ab3ff 50%, #4a7fe8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1rem;
}
.hero-sub {
    color: #5a6a9a; font-size: 1rem; letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}

.stTextInput > div > div > input {
    background: #0a0f20 !important; border: 1.5px solid #1a2a5a !important;
    border-radius: 12px !important; color: #e8eaf0 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.95rem !important;
    padding: 14px 18px !important; transition: border-color 0.3s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #3a6aff !important;
    box-shadow: 0 0 0 3px rgba(58,106,255,0.15) !important;
}
.stTextInput > label {
    color: #5a7aaa !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important; letter-spacing: 1px;
}

.stButton > button {
    background: linear-gradient(135deg, #1a3a8a, #2a5adf) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important; font-weight: 700 !important;
    letter-spacing: 1px !important; padding: 12px 20px !important;
    transition: all 0.25s ease !important; text-transform: uppercase;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a4aaa, #3a6aff) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(58,106,255,0.35) !important;
}

.card-safe {
    background: linear-gradient(135deg, #041a0e, #062510);
    border: 1px solid #1a6a3a; border-left: 5px solid #00e676;
    border-radius: 16px; padding: 28px 32px; margin: 16px 0;
}
.card-phishing {
    background: linear-gradient(135deg, #1a0404, #250606);
    border: 1px solid #6a1a1a; border-left: 5px solid #ff3d3d;
    border-radius: 16px; padding: 28px 32px; margin: 16px 0;
}
.card-title-safe     { font-size: 2rem; font-weight: 800; color: #00e676; letter-spacing: -1px; }
.card-title-phishing { font-size: 2rem; font-weight: 800; color: #ff3d3d; letter-spacing: -1px; }
.card-sub { color: #7a8aaa; font-family: 'Space Mono', monospace; font-size: 0.85rem; margin-top: 8px; line-height: 1.6; }

/* VirusTotal card */
.vt-card {
    background: #080c18; border: 1px solid #1a2a5a;
    border-radius: 16px; padding: 24px; margin: 16px 0;
}
.vt-title {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 3px; color: #3a6aff; margin-bottom: 16px;
    text-transform: uppercase;
}
.vt-engine-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px; margin-top: 12px;
}
.vt-engine-clean {
    background: #041a0e; border: 1px solid #1a4a2a;
    border-radius: 8px; padding: 8px 12px;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00e676;
}
.vt-engine-malicious {
    background: #1a0404; border: 1px solid #4a1a1a;
    border-radius: 8px; padding: 8px 12px;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #ff3d3d;
}
.vt-score-big {
    font-size: 3.5rem; font-weight: 800; line-height: 1;
    font-family: 'Space Mono', monospace;
}
.vt-bar-wrap {
    background: #0a0f20; border-radius: 8px; height: 12px;
    margin: 12px 0 6px; overflow: hidden; border: 1px solid #1a2040;
}
.vt-bar-fill-danger { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #cc0000, #ff3d3d); }
.vt-bar-fill-safe   { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #00b050, #00e676); }

.conf-wrap {
    background: #0a0f20; border-radius: 8px; height: 10px;
    margin: 12px 0 4px; overflow: hidden; border: 1px solid #1a2040;
}
.conf-fill-safe     { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #00b050, #00e676); }
.conf-fill-phishing { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #cc0000, #ff3d3d); }
.conf-label { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #5a6a9a; }

.feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
.feat-card { background: #080c18; border: 1px solid #121830; border-radius: 12px; padding: 14px 16px; }
.feat-card-bad  { border-left: 3px solid #ff3d3d !important; }
.feat-card-good { border-left: 3px solid #00e676 !important; }
.feat-name  { font-weight: 700; font-size: 0.85rem; color: #c8d0e8; }
.feat-val   { font-family: 'Space Mono', monospace; font-size: 0.8rem; float: right; color: #7a8aaa; }
.feat-desc  { font-size: 0.72rem; color: #3a4a6a; margin-top: 4px; line-height: 1.4; }

.stat-row { display: flex; gap: 10px; margin-bottom: 20px; }
.stat-tile { flex: 1; background: #080c18; border: 1px solid #121830; border-radius: 12px; padding: 16px; text-align: center; }
.stat-num  { font-size: 2rem; font-weight: 800; line-height: 1; }
.stat-lbl  { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #3a4a6a; letter-spacing: 2px; margin-top: 4px; }

.hist-item {
    background: #080c18; border: 1px solid #121830; border-radius: 8px;
    padding: 10px 12px; margin-bottom: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #5a6a9a;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.stTabs [data-baseweb="tab-list"] {
    background: #080c18 !important; border-radius: 12px;
    padding: 4px; border: 1px solid #121830; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #5a6a9a !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    font-weight: 700 !important; letter-spacing: 1px; padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a8a, #2a5adf) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a2a5a, transparent);
    margin: 24px 0;
}
.how-card { background: #080c18; border: 1px solid #121830; border-radius: 14px; padding: 24px; margin-bottom: 16px; }
.how-num  { font-size: 3rem; font-weight: 800; color: #1a3a7a; line-height: 1; font-family: 'Space Mono', monospace; }
.how-title { font-size: 1.1rem; font-weight: 700; color: #c8d0e8; margin: 8px 0 6px; }
.how-desc  { font-size: 0.85rem; color: #5a6a9a; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── VirusTotal functions ───────────────────────────────────────────────────────
def vt_scan_url(url: str, api_key: str) -> dict | None:
    """Submit URL to VirusTotal and return analysis results."""
    if not api_key:
        return None
    headers = {"x-apikey": api_key, "Accept": "application/json"}

    # Step 1 — submit URL for scanning
    try:
        resp = requests.post(
            "https://www.virustotal.com/api/v3/urls",
            headers=headers,
            data={"url": url},
            timeout=15
        )
        if resp.status_code != 200:
            return {"error": f"Submission failed ({resp.status_code})"}

        analysis_id = resp.json()["data"]["id"]

        # Step 2 — poll for results (max 20s)
        for _ in range(10):
            time.sleep(2)
            report = requests.get(
                f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                headers=headers,
                timeout=15
            )
            data = report.json().get("data", {})
            if data.get("attributes", {}).get("status") == "completed":
                stats   = data["attributes"]["stats"]
                results = data["attributes"]["results"]
                return {
                    "malicious":   stats.get("malicious", 0),
                    "suspicious":  stats.get("suspicious", 0),
                    "harmless":    stats.get("harmless", 0),
                    "undetected":  stats.get("undetected", 0),
                    "total":       sum(stats.values()),
                    "engines":     results,
                    "error":       None
                }
        return {"error": "Scan timed out — try again"}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def vt_verdict(vt: dict) -> tuple[str, str]:
    """Return (verdict_label, colour) based on VirusTotal results."""
    if not vt or vt.get("error"):
        return "UNKNOWN", "#7a8aaa"
    flagged = vt["malicious"] + vt["suspicious"]
    if flagged == 0:
        return "CLEAN", "#00e676"
    elif flagged <= 3:
        return "SUSPICIOUS", "#ffaa00"
    else:
        return "CONFIRMED THREAT", "#ff3d3d"


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 16px'>
        <div style='font-family:"Space Mono",monospace;font-size:0.65rem;
                    letter-spacing:3px;color:#3a4a6a;margin-bottom:8px'>PHISHGUARD</div>
        <div style='font-size:1.4rem;font-weight:800;color:#e8eaf0;letter-spacing:-0.5px'>
            Threat Intelligence
        </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    safe  = sum(1 for h in st.session_state.history if h["result"] == "Safe")
    phish = total - safe

    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-tile'><div class='stat-num' style='color:#e8eaf0'>{total}</div><div class='stat-lbl'>SCANNED</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#00e676'>{safe}</div><div class='stat-lbl'>SAFE</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#ff3d3d'>{phish}</div><div class='stat-lbl'>THREATS</div></div>
    </div>
    <div class='divider'></div>
    <div style='font-family:"Space Mono",monospace;font-size:0.65rem;letter-spacing:2px;color:#3a4a6a;margin-bottom:12px'>
        RECENT SCANS
    </div>""", unsafe_allow_html=True)

    for h in reversed(st.session_state.history[-6:]):
        icon  = "✅" if h["result"] == "Safe" else "⚠️"
        short = h["url"][:32] + "..." if len(h["url"]) > 32 else h["url"]
        st.markdown(f"<div class='hist-item'>{icon} {short}</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("<div style='color:#3a4a6a;font-size:0.8rem;font-family:\"Space Mono\",monospace'>No scans yet</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # API key status indicator
    if VT_API_KEY:
        st.markdown("""
        <div style='background:#041a0e;border:1px solid #1a4a2a;border-radius:10px;
                    padding:12px;font-family:"Space Mono",monospace;font-size:0.7rem;
                    color:#00e676;text-align:center'>
            ✅ VirusTotal API Connected
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#1a0e04;border:1px solid #4a2a1a;border-radius:10px;
                    padding:12px;font-family:"Space Mono",monospace;font-size:0.7rem;
                    color:#ffaa00;text-align:center'>
            ⚠️ No API Key — add to .env file
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("""
    <div style='margin-top:20px;font-family:"Space Mono",monospace;font-size:0.65rem;
                color:#2a3a5a;text-align:center;line-height:1.8'>
        ML Model + VirusTotal API<br>Trained on 11,000+ real URLs
    </div>""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model not found! Run `python train_model.py` first.")
    st.stop()

st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🛡️ &nbsp; ML + VirusTotal Threat Intelligence</div>
    <div class='hero-title'>PhishGuard</div>
    <div class='hero-sub'>Dual-layer phishing detection · ML model + 90 antivirus engines</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚡  SCAN URL", "📋  BULK SCAN", "📖  HOW IT WORKS"])

# ══════════════════════════════════════════════════════════
# TAB 1 — Single URL
# ══════════════════════════════════════════════════════════
with tab1:
    url_input = st.text_input("", placeholder="https://www.example.com/page",
                               label_visibility="collapsed")

    use_vt = st.checkbox(
        "🔬 Also scan with VirusTotal (90+ antivirus engines) — takes ~20 seconds",
        value=bool(VT_API_KEY)
    )

    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    check_btn = c1.button("⚡  SCAN URL",         use_container_width=True)
    safe_ex   = c2.button("✅ Safe demo",          use_container_width=True)
    phish_ex  = c3.button("⚠️ Phishing demo",     use_container_width=True)

    if safe_ex:  url_input = "https://www.amazon.com/products/shoes"
    if phish_ex: url_input = "http://amaz0n-login-secure.xyz/account/verify?user=123456"

    if (check_btn or safe_ex or phish_ex) and url_input.strip():
        url = url_input.strip()
        if not url.startswith("http"):
            url = "http://" + url

        # ── ML prediction ──────────────────────────────────────────────────
        features   = extract_features(url)
        feat_array = np.array(features).reshape(1, -1)
        prediction = model.predict(feat_array)[0]
        try:
            proba      = model.predict_proba(feat_array)[0]
            confidence = proba[prediction] * 100
        except:
            confidence = None

        # ── VirusTotal scan ────────────────────────────────────────────────
        vt_data = None
        if use_vt and VT_API_KEY:
            with st.spinner("🔬 Sending to VirusTotal — scanning with 90+ engines..."):
                vt_data = vt_scan_url(url, VT_API_KEY)

        # Save history
        st.session_state.history.append({
            "url":    url,
            "result": "Safe" if prediction == 0 else "Phishing",
            "conf":   confidence,
            "time":   datetime.now().strftime("%H:%M:%S")
        })

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Two column layout: ML result | VT result ───────────────────────
        col_ml, col_vt = st.columns(2)

        with col_ml:
            st.markdown("""
            <div style='font-family:"Space Mono",monospace;font-size:0.65rem;
                        letter-spacing:3px;color:#3a6aff;margin-bottom:12px'>
                ML MODEL RESULT
            </div>""", unsafe_allow_html=True)

            if prediction == 0:
                st.markdown(f"""
                <div class='card-safe'>
                    <div class='card-title-safe'>✅ SAFE</div>
                    <div class='card-sub'>
                        Appears legitimate.<br>
                        {"Confidence: <b style='color:#00e676'>%.1f%%</b>" % confidence if confidence else ""}
                    </div>
                    {"<div class='conf-wrap'><div class='conf-fill-safe' style='width:%.1f%%'></div></div>" % confidence if confidence else ""}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='card-phishing'>
                    <div class='card-title-phishing'>⚠️ PHISHING</div>
                    <div class='card-sub'>
                        Shows signs of being malicious.<br>
                        {"Confidence: <b style='color:#ff3d3d'>%.1f%%</b>" % confidence if confidence else ""}
                    </div>
                    {"<div class='conf-wrap'><div class='conf-fill-phishing' style='width:%.1f%%'></div></div>" % confidence if confidence else ""}
                </div>""", unsafe_allow_html=True)

        with col_vt:
            st.markdown("""
            <div style='font-family:"Space Mono",monospace;font-size:0.65rem;
                        letter-spacing:3px;color:#3a6aff;margin-bottom:12px'>
                VIRUSTOTAL RESULT
            </div>""", unsafe_allow_html=True)

            if not use_vt or not VT_API_KEY:
                st.markdown("""
                <div class='vt-card' style='text-align:center;padding:40px 24px'>
                    <div style='font-size:2rem'>🔬</div>
                    <div style='font-family:"Space Mono",monospace;font-size:0.75rem;
                                color:#3a4a6a;margin-top:12px'>
                        Enable checkbox above<br>to scan with 90+ engines
                    </div>
                </div>""", unsafe_allow_html=True)

            elif vt_data and vt_data.get("error"):
                st.markdown(f"""
                <div class='vt-card'>
                    <div style='color:#ffaa00;font-family:"Space Mono",monospace;font-size:0.8rem'>
                        ⚠️ VT Error: {vt_data['error']}
                    </div>
                </div>""", unsafe_allow_html=True)

            elif vt_data:
                verdict, v_color = vt_verdict(vt_data)
                flagged = vt_data["malicious"] + vt_data["suspicious"]
                total_e = vt_data["total"]
                pct     = (flagged / total_e * 100) if total_e > 0 else 0
                bar_cls = "vt-bar-fill-danger" if flagged > 0 else "vt-bar-fill-safe"

                st.markdown(f"""
                <div class='vt-card'>
                    <div class='vt-title'>VIRUSTOTAL — {total_e} ENGINES</div>
                    <div style='display:flex;align-items:baseline;gap:12px'>
                        <div class='vt-score-big' style='color:{v_color}'>{flagged}</div>
                        <div style='color:#5a6a9a;font-family:"Space Mono",monospace;font-size:0.8rem'>
                            / {total_e} engines flagged
                        </div>
                    </div>
                    <div class='vt-bar-wrap'>
                        <div class='{bar_cls}' style='width:{pct:.1f}%'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;
                                font-family:"Space Mono",monospace;font-size:0.7rem;margin-bottom:16px'>
                        <span style='color:#ff3d3d'>🔴 Malicious: {vt_data['malicious']}</span>
                        <span style='color:#ffaa00'>🟡 Suspicious: {vt_data['suspicious']}</span>
                        <span style='color:#00e676'>🟢 Clean: {vt_data['harmless']}</span>
                    </div>
                    <div style='background:{"#1a0404" if flagged > 0 else "#041a0e"};
                                border:1px solid {"#4a1a1a" if flagged > 0 else "#1a4a2a"};
                                border-radius:10px;padding:12px;text-align:center;
                                font-weight:800;font-size:1rem;color:{v_color}'>
                        {verdict}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Show which engines flagged it
                if flagged > 0:
                    st.markdown("""
                    <div style='font-family:"Space Mono",monospace;font-size:0.65rem;
                                letter-spacing:2px;color:#3a4a6a;margin:16px 0 8px'>
                        ENGINES THAT FLAGGED THIS URL
                    </div>""", unsafe_allow_html=True)

                    html_engines = "<div class='vt-engine-grid'>"
                    shown = 0
                    for eng_name, eng_data in vt_data["engines"].items():
                        if eng_data.get("category") in ("malicious", "suspicious"):
                            cat = eng_data.get("category", "").upper()
                            res = eng_data.get("result", "")
                            html_engines += f"""
                            <div class='vt-engine-malicious'>
                                🔴 <b>{eng_name}</b><br>
                                <span style='color:#7a4a4a'>{res or cat}</span>
                            </div>"""
                            shown += 1
                            if shown >= 12:
                                break
                    html_engines += "</div>"
                    st.markdown(html_engines, unsafe_allow_html=True)

        # ── Combined verdict ───────────────────────────────────────────────
        if vt_data and not vt_data.get("error"):
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            flagged = vt_data["malicious"] + vt_data["suspicious"]
            both_safe     = prediction == 0 and flagged == 0
            both_phishing = prediction == 1 and flagged > 3
            ml_safe_vt_bad = prediction == 0 and flagged > 3
            ml_bad_vt_safe = prediction == 1 and flagged == 0

            if both_safe:
                verdict_html = "<div style='background:#041a0e;border:1px solid #1a6a3a;border-radius:12px;padding:20px;text-align:center'><span style='font-size:1.5rem;font-weight:800;color:#00e676'>✅ VERIFIED SAFE — Both ML model and VirusTotal agree this URL is safe</span></div>"
            elif both_phishing:
                verdict_html = "<div style='background:#1a0404;border:1px solid #6a1a1a;border-radius:12px;padding:20px;text-align:center'><span style='font-size:1.5rem;font-weight:800;color:#ff3d3d'>🚨 CONFIRMED THREAT — Both ML model and VirusTotal flagged this URL</span></div>"
            elif ml_safe_vt_bad:
                verdict_html = "<div style='background:#1a0e04;border:1px solid #6a4a1a;border-radius:12px;padding:20px;text-align:center'><span style='font-size:1.5rem;font-weight:800;color:#ffaa00'>⚠️ CAUTION — ML says safe but VirusTotal flagged it. Avoid this URL.</span></div>"
            elif ml_bad_vt_safe:
                verdict_html = "<div style='background:#0e1a04;border:1px solid #4a6a1a;border-radius:12px;padding:20px;text-align:center'><span style='font-size:1.5rem;font-weight:800;color:#aaff00'>🟡 LIKELY SAFE — ML flagged it but VirusTotal found no threats.</span></div>"
            else:
                verdict_html = "<div style='background:#0a0f20;border:1px solid #1a2a5a;border-radius:12px;padding:20px;text-align:center'><span style='font-size:1.5rem;font-weight:800;color:#ffaa00'>⚠️ INCONCLUSIVE — Exercise caution</span></div>"

            st.markdown(verdict_html, unsafe_allow_html=True)

        # ── Feature breakdown ──────────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"Space Mono",monospace;font-size:0.7rem;
                    letter-spacing:2px;color:#3a4a6a;margin-bottom:14px'>
            ML FEATURE ANALYSIS
        </div>""", unsafe_allow_html=True)

        feature_info = {
            "url_length":           ("URL Length",           "Phishing URLs are often very long",            lambda v: v > 75),
            "has_https":            ("Uses HTTPS",           "Secure sites always use HTTPS",                lambda v: v == 0),
            "has_ip":               ("IP as Domain",         "Using raw IP instead of domain is suspicious", lambda v: v == 1),
            "has_at_symbol":        ("@ Symbol",             "Used to hide real destination domain",         lambda v: v == 1),
            "dot_count":            ("Dot Count",            "Too many dots = suspicious subdomains",        lambda v: v > 3),
            "hyphen_count":         ("Hyphen Count",         "amaz0n-secure-login.xyz style domains",        lambda v: v > 1),
            "domain_length":        ("Domain Length",        "Legitimate domains are usually short",         lambda v: v > 20),
            "subdomain_count":      ("Subdomain Count",      "Excessive subdomains are suspicious",          lambda v: v > 2),
            "has_suspicious_word":  ("Suspicious Keywords",  "login, verify, account, confirm etc.",         lambda v: v == 1),
            "has_port":             ("Non-Standard Port",    "Legit sites don't use :8080 in URLs",          lambda v: v == 1),
            "slash_count":          ("Path Depth",           "Very deep URL paths = obfuscation",            lambda v: v > 5),
            "digits_in_domain":     ("Digits in Domain",     "amaz0n — replacing letters with numbers",      lambda v: v > 2),
        }

        html_feats = "<div class='feat-grid'>"
        for fname, fval in zip(FEATURE_NAMES, features):
            label, desc, is_bad = feature_info.get(fname, (fname, "", lambda v: False))
            bad       = is_bad(fval)
            cls       = "feat-card-bad" if bad else "feat-card-good"
            dot_color = "#ff3d3d" if bad else "#00e676"
            html_feats += f"""
            <div class='feat-card {cls}'>
                <span class='feat-name'><span style='color:{dot_color}'>●</span> &nbsp;{label}</span>
                <span class='feat-val'>{fval}</span>
                <div class='feat-desc'>{desc}</div>
            </div>"""
        html_feats += "</div>"
        st.markdown(html_feats, unsafe_allow_html=True)

    elif check_btn and not url_input.strip():
        st.warning("Please enter a URL.")

# ══════════════════════════════════════════════════════════
# TAB 2 — Bulk scan
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='font-family:"Space Mono",monospace;font-size:0.7rem;
                letter-spacing:2px;color:#3a4a6a;margin-bottom:14px'>
        BULK URL SCANNER — paste one URL per line (ML only, no VT for bulk)
    </div>""", unsafe_allow_html=True)

    bulk_input = st.text_area("", height=160,
        placeholder="https://www.google.com\nhttp://evil-login.xyz/verify\nhttps://github.com",
        label_visibility="collapsed")
    bulk_btn = st.button("⚡  SCAN ALL URLS", use_container_width=True)

    if bulk_btn and bulk_input.strip():
        urls = [u.strip() for u in bulk_input.strip().split("\n") if u.strip()]
        rows = []
        bar  = st.progress(0, text="Scanning...")

        for i, url in enumerate(urls):
            u = url if url.startswith("http") else "http://" + url
            f = extract_features(u)
            p = model.predict(np.array(f).reshape(1, -1))[0]
            try:
                c = model.predict_proba(np.array(f).reshape(1, -1))[0][p] * 100
            except:
                c = None
            rows.append({
                "URL":        url,
                "Result":     "✅ Safe" if p == 0 else "⚠️ Phishing",
                "Confidence": f"{c:.1f}%" if c else "N/A",
                "Risk Level": "LOW" if p == 0 else "HIGH"
            })
            bar.progress((i+1)/len(urls), text=f"Scanned {i+1}/{len(urls)}")

        bar.empty()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        s = sum(1 for r in rows if "Safe" in r["Result"])
        p = len(rows) - s
        st.markdown(f"""
        <div style='display:flex;gap:16px;margin-top:16px'>
            <div class='stat-tile' style='flex:1'><div class='stat-num' style='color:#e8eaf0'>{len(rows)}</div><div class='stat-lbl'>TOTAL</div></div>
            <div class='stat-tile' style='flex:1'><div class='stat-num' style='color:#00e676'>{s}</div><div class='stat-lbl'>SAFE</div></div>
            <div class='stat-tile' style='flex:1'><div class='stat-num' style='color:#ff3d3d'>{p}</div><div class='stat-lbl'>THREATS</div></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — How it works
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='how-card'>
        <div class='how-num'>01</div>
        <div class='how-title'>You enter a URL</div>
        <div class='how-desc'>No website is opened. The URL is analysed as text — completely safe.</div>
    </div>
    <div class='how-card'>
        <div class='how-num'>02</div>
        <div class='how-title'>ML model analyses 12 features</div>
        <div class='how-desc'>URL length · HTTPS · IP as domain · @ symbol · dots · hyphens · domain length · subdomains · suspicious keywords · port · path depth · digits in domain</div>
    </div>
    <div class='how-card'>
        <div class='how-num'>03</div>
        <div class='how-title'>VirusTotal scans with 90+ engines</div>
        <div class='how-desc'>Your URL is checked against 90+ antivirus and threat intelligence engines simultaneously — the same technology used by enterprise security teams.</div>
    </div>
    <div class='how-card'>
        <div class='how-num'>04</div>
        <div class='how-title'>Combined verdict</div>
        <div class='how-desc'>Both results are compared to give a final VERIFIED SAFE, CONFIRMED THREAT, or CAUTION verdict — dual-layer protection.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='divider'></div>
<div style='text-align:center;font-family:"Space Mono",monospace;font-size:0.65rem;
            color:#2a3a5a;padding-bottom:20px'>
    PhishGuard · Python · XGBoost · scikit-learn · VirusTotal API · Streamlit
</div>
""", unsafe_allow_html=True)