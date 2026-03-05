"""
app.py  —  PhishGuard Research Edition
Run with:  streamlit run app.py
"""

import streamlit as st
import pickle, numpy as np, os, time, requests, pandas as pd
from datetime  import datetime
from dotenv    import load_dotenv
from urllib.parse import urlparse

from features    import extract_features, FEATURE_NAMES
from database    import insert_url, get_recent_scans, get_dashboard_stats, get_daily_stats
from correlate   import correlate_url
from risk_score  import compute_risk_score, grade_color

load_dotenv()
VT_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")

st.set_page_config(page_title="PhishGuard Intelligence", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
html,body,[data-testid="stAppViewContainer"]{background:#050810!important;font-family:'Syne',sans-serif;color:#e8eaf0;}
[data-testid="stSidebar"]{background:#080c18!important;border-right:1px solid #1a2040;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:#2a3a6a;border-radius:4px;}

.hero{text-align:center;padding:2.5rem 1rem 1.5rem;}
.hero-badge{display:inline-block;background:linear-gradient(135deg,#0d1f4a,#1a3a7a);border:1px solid #2a5aaa;color:#5a9aff;font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:3px;padding:6px 18px;border-radius:20px;margin-bottom:1.2rem;text-transform:uppercase;}
.hero-title{font-size:clamp(2rem,5vw,3.8rem);font-weight:800;line-height:1.05;letter-spacing:-2px;background:linear-gradient(135deg,#ffffff 0%,#7ab3ff 50%,#4a7fe8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.8rem;}
.hero-sub{color:#5a6a9a;font-size:.9rem;letter-spacing:1px;font-family:'Space Mono',monospace;}

.stTextInput>div>div>input{background:#0a0f20!important;border:1.5px solid #1a2a5a!important;border-radius:12px!important;color:#e8eaf0!important;font-family:'Space Mono',monospace!important;font-size:.95rem!important;padding:14px 18px!important;}
.stTextInput>label{color:#5a7aaa!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;letter-spacing:1px;}

.stButton>button{background:linear-gradient(135deg,#1a3a8a,#2a5adf)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;font-weight:700!important;letter-spacing:1px!important;padding:12px 20px!important;transition:all .25s ease!important;text-transform:uppercase;}
.stButton>button:hover{background:linear-gradient(135deg,#2a4aaa,#3a6aff)!important;transform:translateY(-2px)!important;box-shadow:0 8px 25px rgba(58,106,255,.35)!important;}

.card-safe{background:linear-gradient(135deg,#041a0e,#062510);border:1px solid #1a6a3a;border-left:5px solid #00e676;border-radius:16px;padding:24px 28px;margin:12px 0;}
.card-phishing{background:linear-gradient(135deg,#1a0404,#250606);border:1px solid #6a1a1a;border-left:5px solid #ff3d3d;border-radius:16px;padding:24px 28px;margin:12px 0;}

.risk-dial{background:#080c18;border:1px solid #1a2a5a;border-radius:16px;padding:24px;text-align:center;margin:12px 0;}
.risk-num{font-size:4rem;font-weight:800;font-family:'Space Mono',monospace;line-height:1;}
.risk-grade{font-size:1.2rem;font-weight:800;letter-spacing:2px;margin-top:8px;}
.risk-bar-wrap{background:#0a0f20;border-radius:8px;height:12px;margin:14px 0 6px;overflow:hidden;border:1px solid #1a2040;}

.corr-card{background:#080c18;border:1px solid #1a2a5a;border-radius:16px;padding:20px;margin:12px 0;}
.breakdown-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #121830;}
.breakdown-row:last-child{border-bottom:none;}

.stat-tile{background:#080c18;border:1px solid #121830;border-radius:12px;padding:16px;text-align:center;}
.stat-num{font-size:2rem;font-weight:800;line-height:1;}
.stat-lbl{font-family:'Space Mono',monospace;font-size:.65rem;color:#3a4a6a;letter-spacing:2px;margin-top:4px;}

.feat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;}
.feat-card{background:#080c18;border:1px solid #121830;border-radius:10px;padding:12px 14px;}
.feat-card-bad{border-left:3px solid #ff3d3d!important;}
.feat-card-good{border-left:3px solid #00e676!important;}

.hist-item{background:#080c18;border:1px solid #121830;border-radius:8px;padding:10px 12px;margin-bottom:6px;font-family:'Space Mono',monospace;font-size:.7rem;color:#5a6a9a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

.stTabs [data-baseweb="tab-list"]{background:#080c18!important;border-radius:12px;padding:4px;border:1px solid #121830;gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:#5a6a9a!important;font-family:'Space Mono',monospace!important;font-size:.75rem!important;font-weight:700!important;letter-spacing:1px;padding:10px 16px!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1a3a8a,#2a5adf)!important;color:white!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:20px!important;}

.divider{height:1px;background:linear-gradient(90deg,transparent,#1a2a5a,transparent);margin:20px 0;}
.mono-label{font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:3px;color:#3a4a6a;margin-bottom:10px;text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)


# ── VirusTotal ────────────────────────────────────────────────────────────────
def vt_scan_url(url, api_key):
    if not api_key: return None
    headers = {"x-apikey": api_key, "Accept": "application/json"}
    try:
        r = requests.post("https://www.virustotal.com/api/v3/urls",
                          headers=headers, data={"url": url}, timeout=15)
        if r.status_code != 200: return {"error": f"HTTP {r.status_code}"}
        aid = r.json()["data"]["id"]
        for _ in range(10):
            time.sleep(2)
            rep  = requests.get(f"https://www.virustotal.com/api/v3/analyses/{aid}",
                                headers=headers, timeout=15).json()
            attr = rep.get("data", {}).get("attributes", {})
            if attr.get("status") == "completed":
                s = attr["stats"]
                return {"malicious": s.get("malicious",0),
                        "suspicious": s.get("suspicious",0),
                        "harmless": s.get("harmless",0),
                        "undetected": s.get("undetected",0),
                        "total": sum(s.values()),
                        "engines": attr["results"], "error": None}
        return {"error": "Timed out"}
    except Exception as e:
        return {"error": str(e)}


@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"): return None
    with open("model.pkl","rb") as f: return pickle.load(f)

model = load_model()
if "history" not in st.session_state: st.session_state.history = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 12px'>
        <div style='font-family:"Space Mono",monospace;font-size:.65rem;letter-spacing:3px;color:#3a4a6a;margin-bottom:6px'>PHISHGUARD</div>
        <div style='font-size:1.3rem;font-weight:800;color:#e8eaf0'>Intelligence Platform</div>
    </div><div class='divider'></div>""", unsafe_allow_html=True)

    stats = get_dashboard_stats()
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px'>
        <div class='stat-tile'><div class='stat-num' style='color:#e8eaf0'>{stats['total']}</div><div class='stat-lbl'>SCANNED</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#ff3d3d'>{stats['phishing']}</div><div class='stat-lbl'>THREATS</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#00e676'>{stats['safe']}</div><div class='stat-lbl'>SAFE</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#ffaa00'>{stats['campaigns']}</div><div class='stat-lbl'>CAMPAIGNS</div></div>
    </div>
    <div class='divider'></div>
    <div class='mono-label'>RECENT SCANS</div>""", unsafe_allow_html=True)

    recent = get_recent_scans(6)
    for r in recent:
        icon  = "✅" if r["ml_prediction"] == 0 else "⚠️"
        short = r["url"][:30] + "..." if len(r["url"]) > 30 else r["url"]
        grade = r.get("risk_grade","?")
        gc    = grade_color(grade)
        st.markdown(f"""
        <div class='hist-item'>{icon} {short}
        <span style='float:right;color:{gc};font-weight:700'>{grade}</span></div>""",
        unsafe_allow_html=True)

    if not recent:
        st.markdown("<div style='color:#3a4a6a;font-size:.8rem;font-family:\"Space Mono\",monospace'>No scans yet</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    vt_ok = bool(VT_API_KEY)
    st.markdown(f"""
    <div style='background:{"#041a0e" if vt_ok else "#1a0e04"};border:1px solid {"#1a4a2a" if vt_ok else "#4a2a1a"};
                border-radius:10px;padding:10px;font-family:"Space Mono",monospace;font-size:.7rem;
                color:{"#00e676" if vt_ok else "#ffaa00"};text-align:center'>
        {"✅ VirusTotal Connected" if vt_ok else "⚠️ No VT API Key"}
    </div>""", unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Run `python train_model.py` first."); st.stop()

st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🛡️ &nbsp; Database-Driven Threat Intelligence Platform</div>
    <div class='hero-title'>PhishGuard</div>
    <div class='hero-sub'>ML Detection · VirusTotal API · Threat Correlation · Dynamic Risk Scoring</div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["⚡ SCAN URL", "📋 BULK SCAN",
                                   "📊 INTELLIGENCE DASHBOARD", "📖 HOW IT WORKS"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SCAN
# ══════════════════════════════════════════════════════════
with tab1:
    url_input = st.text_input("", placeholder="https://www.example.com",
                               label_visibility="collapsed")
    use_vt = st.checkbox("🔬 Scan with VirusTotal (90+ engines) — ~20 seconds",
                         value=bool(VT_API_KEY))

    c1,c2,c3 = st.columns([2,1.2,1.2])
    check_btn = c1.button("⚡ SCAN URL",        use_container_width=True)
    safe_ex   = c2.button("✅ Safe demo",        use_container_width=True)
    phish_ex  = c3.button("⚠️ Phishing demo",   use_container_width=True)

    if safe_ex:  url_input = "https://www.amazon.com/products/shoes"
    if phish_ex: url_input = "http://amaz0n-login-secure.xyz/account/verify?user=123456"

    if (check_btn or safe_ex or phish_ex) and url_input.strip():
        url = url_input.strip()
        if not url.startswith("http"): url = "http://" + url

        # 1️⃣ ML
        features   = extract_features(url)
        feat_array = np.array(features).reshape(1,-1)
        prediction = model.predict(feat_array)[0]
        try:
            proba      = model.predict_proba(feat_array)[0]
            confidence = float(proba[prediction])
        except:
            confidence = 0.5

        # 2️⃣ VirusTotal
        vt_data = None
        if use_vt and VT_API_KEY:
            with st.spinner("🔬 Scanning with VirusTotal..."):
                vt_data = vt_scan_url(url, VT_API_KEY)

        # 3️⃣ Threat Correlation
        with st.spinner("🔗 Correlating with threat database..."):
            domain  = urlparse(url).netloc
            corr    = correlate_url(url, features)

        # 4️⃣ Dynamic Risk Score
        risk = compute_risk_score(
            ml_prediction = int(prediction),
            ml_confidence = confidence,
            vt_data       = vt_data,
            similarity    = corr["max_similarity"],
            features      = features
        )

        # 5️⃣ Save to DB
        insert_url(
            url           = url,
            ml_prediction = int(prediction),
            ml_confidence = confidence,
            vt_malicious  = vt_data.get("malicious",0)  if vt_data and not vt_data.get("error") else 0,
            vt_suspicious = vt_data.get("suspicious",0) if vt_data and not vt_data.get("error") else 0,
            vt_harmless   = vt_data.get("harmless",0)   if vt_data and not vt_data.get("error") else 0,
            vt_total      = vt_data.get("total",0)      if vt_data and not vt_data.get("error") else 0,
            risk_score    = risk["score"],
            risk_grade    = risk["grade"],
            features      = features,
            domain        = domain,
            similar_urls  = [s["id"] for s in corr["similar_urls"]],
            campaign_id   = corr["campaign_id"]
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Layout: ML | Risk Score | VT ──────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='mono-label'>ML PREDICTION</div>", unsafe_allow_html=True)
            if prediction == 0:
                st.markdown(f"""
                <div class='card-safe'>
                    <div style='font-size:1.8rem;font-weight:800;color:#00e676'>✅ SAFE</div>
                    <div style='color:#7a8aaa;font-family:"Space Mono",monospace;font-size:.8rem;margin-top:8px'>
                        Confidence: <b style='color:#00e676'>{confidence*100:.1f}%</b>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='card-phishing'>
                    <div style='font-size:1.8rem;font-weight:800;color:#ff3d3d'>⚠️ PHISHING</div>
                    <div style='color:#7a8aaa;font-family:"Space Mono",monospace;font-size:.8rem;margin-top:8px'>
                        Confidence: <b style='color:#ff3d3d'>{confidence*100:.1f}%</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='mono-label'>DYNAMIC RISK SCORE</div>", unsafe_allow_html=True)
            rc = risk["color"]
            st.markdown(f"""
            <div class='risk-dial'>
                <div class='risk-num' style='color:{rc}'>{risk['score']}</div>
                <div style='color:#3a4a6a;font-family:"Space Mono",monospace;font-size:.7rem;margin-top:4px'>OUT OF 100</div>
                <div class='risk-bar-wrap'>
                    <div style='width:{risk["score"]}%;height:100%;border-radius:8px;
                                background:linear-gradient(90deg,#1a3a8a,{rc})'></div>
                </div>
                <div class='risk-grade' style='color:{rc}'>{risk["grade"]}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='mono-label'>VIRUSTOTAL</div>", unsafe_allow_html=True)
            if not use_vt or not VT_API_KEY:
                st.markdown("""
                <div class='corr-card' style='text-align:center;padding:40px 20px'>
                    <div style='color:#3a4a6a;font-family:"Space Mono",monospace;font-size:.75rem'>
                        Enable checkbox<br>to scan with VT
                    </div>
                </div>""", unsafe_allow_html=True)
            elif vt_data and vt_data.get("error"):
                st.markdown(f"<div class='corr-card'><span style='color:#ffaa00;font-size:.8rem'>⚠️ {vt_data['error']}</span></div>",
                            unsafe_allow_html=True)
            elif vt_data:
                flagged = vt_data["malicious"] + vt_data["suspicious"]
                total_e = vt_data["total"]
                fc      = "#ff3d3d" if flagged > 0 else "#00e676"
                pct     = (flagged/total_e*100) if total_e else 0
                st.markdown(f"""
                <div class='corr-card'>
                    <div style='font-size:2.5rem;font-weight:800;color:{fc};font-family:"Space Mono",monospace'>{flagged}</div>
                    <div style='color:#5a6a9a;font-family:"Space Mono",monospace;font-size:.75rem'>/ {total_e} engines flagged</div>
                    <div class='risk-bar-wrap'>
                        <div style='width:{pct:.1f}%;height:100%;border-radius:8px;background:{"linear-gradient(90deg,#cc0000,#ff3d3d)" if flagged>0 else "linear-gradient(90deg,#00b050,#00e676)"}'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;font-family:"Space Mono",monospace;font-size:.65rem;margin-top:8px'>
                        <span style='color:#ff3d3d'>🔴 {vt_data['malicious']}</span>
                        <span style='color:#ffaa00'>🟡 {vt_data['suspicious']}</span>
                        <span style='color:#00e676'>🟢 {vt_data['harmless']}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Risk Score Breakdown ───────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='mono-label'>RISK SCORE BREAKDOWN</div>", unsafe_allow_html=True)
            st.markdown("<div class='corr-card'>", unsafe_allow_html=True)
            for signal, contribution in risk["breakdown"].items():
                pct = min(contribution / 35 * 100, 100)
                st.markdown(f"""
                <div class='breakdown-row'>
                    <span style='font-size:.85rem;color:#c8d0e8'>{signal}</span>
                    <div style='display:flex;align-items:center;gap:10px'>
                        <div style='width:80px;height:6px;background:#0a0f20;border-radius:4px;overflow:hidden'>
                            <div style='width:{pct}%;height:100%;background:{risk["color"]};border-radius:4px'></div>
                        </div>
                        <span style='font-family:"Space Mono",monospace;font-size:.75rem;color:{risk["color"]}'>{contribution}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Explanation
            st.markdown("<div class='mono-label' style='margin-top:14px'>WHY THIS SCORE?</div>",
                        unsafe_allow_html=True)
            for exp in risk["explanation"]:
                st.markdown(f"""
                <div style='background:#080c18;border-left:3px solid {risk["color"]};
                            border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:6px;
                            font-size:.82rem;color:#c8d0e8'>
                    {exp}
                </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='mono-label'>THREAT CORRELATION</div>", unsafe_allow_html=True)
            sim_color = "#ff3d3d" if corr["max_similarity"] > 0.75 else \
                        "#ffaa00" if corr["max_similarity"] > 0.50 else "#00e676"
            st.markdown(f"""
            <div class='corr-card'>
                <div style='font-size:2rem;font-weight:800;font-family:"Space Mono",monospace;color:{sim_color}'>
                    {corr['max_similarity']*100:.1f}%
                </div>
                <div style='color:#5a6a9a;font-size:.8rem;margin:4px 0 12px'>max similarity to known threats</div>
                <div style='background:{"#1a0404" if corr["max_similarity"]>0.75 else "#080c18"};
                            border:1px solid {"#4a1a1a" if corr["max_similarity"]>0.75 else "#1a2040"};
                            border-radius:8px;padding:10px;font-family:"Space Mono",monospace;
                            font-size:.75rem;color:{sim_color}'>
                    {corr["correlation_label"]}
                </div>
                {"<div style='margin-top:12px;font-family:\"Space Mono\",monospace;font-size:.65rem;color:#ffaa00'>🚨 CAMPAIGN #" + str(corr['campaign_id']) + " DETECTED</div>" if corr["campaign_id"] else ""}
            </div>""", unsafe_allow_html=True)

            if corr["similar_urls"]:
                st.markdown("<div class='mono-label' style='margin-top:14px'>SIMILAR THREATS IN DB</div>",
                            unsafe_allow_html=True)
                for s in corr["similar_urls"][:4]:
                    short = s["url"][:45] + "..." if len(s["url"]) > 45 else s["url"]
                    st.markdown(f"""
                    <div style='background:#080c18;border:1px solid #1a0a0a;border-radius:8px;
                                padding:8px 12px;margin-bottom:5px;font-family:"Space Mono",monospace;
                                font-size:.68rem;color:#7a4a4a'>
                        🔴 {short}
                        <span style='float:right;color:#ff3d3d'>{s['score']*100:.0f}%</span>
                    </div>""", unsafe_allow_html=True)

        # ── Feature Breakdown ──────────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='mono-label'>URL FEATURE ANALYSIS</div>", unsafe_allow_html=True)
        feature_info = {
            "url_length":          ("URL Length",          "Phishing URLs are often very long",             lambda v: v>75),
            "has_https":           ("Uses HTTPS",          "Secure sites use HTTPS",                        lambda v: v==0),
            "has_ip":              ("IP as Domain",        "Raw IP instead of domain name",                 lambda v: v==1),
            "has_at_symbol":       ("@ Symbol",            "Used to hide the real destination",             lambda v: v==1),
            "dot_count":           ("Dot Count",           "Too many dots = suspicious subdomains",         lambda v: v>3),
            "hyphen_count":        ("Hyphens",             "amaz0n-secure-login.xyz style",                 lambda v: v>1),
            "domain_length":       ("Domain Length",       "Legit domains are usually short",               lambda v: v>20),
            "subdomain_count":     ("Subdomains",          "Excessive subdomains suspicious",               lambda v: v>2),
            "has_suspicious_word": ("Suspicious Keywords", "login, verify, account, confirm…",              lambda v: v==1),
            "has_port":            ("Non-Standard Port",   "Legit sites don't use :8080",                   lambda v: v==1),
            "slash_count":         ("Path Depth",          "Very deep paths = obfuscation",                 lambda v: v>5),
            "digits_in_domain":    ("Digits in Domain",    "amaz0n → letter-to-digit substitution",         lambda v: v>2),
        }
        html_feats = "<div class='feat-grid'>"
        for fname, fval in zip(FEATURE_NAMES, features):
            label, desc, is_bad = feature_info.get(fname,(fname,"",lambda v:False))
            bad = is_bad(fval)
            html_feats += f"""
            <div class='feat-card {"feat-card-bad" if bad else "feat-card-good"}'>
                <span style='font-weight:700;font-size:.85rem;color:#c8d0e8'>
                    <span style='color:{"#ff3d3d" if bad else "#00e676"}'>●</span> &nbsp;{label}
                </span>
                <span style='font-family:"Space Mono",monospace;font-size:.8rem;float:right;color:#7a8aaa'>{fval}</span>
                <div style='font-size:.72rem;color:#3a4a6a;margin-top:4px;line-height:1.4'>{desc}</div>
            </div>"""
        html_feats += "</div>"
        st.markdown(html_feats, unsafe_allow_html=True)

    elif check_btn and not url_input.strip():
        st.warning("Please enter a URL.")

# ══════════════════════════════════════════════════════════
# TAB 2 — BULK SCAN
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='mono-label'>BULK URL SCANNER — one URL per line</div>",
                unsafe_allow_html=True)
    bulk_input = st.text_area("", height=150,
        placeholder="https://www.google.com\nhttp://evil-login.xyz/verify",
        label_visibility="collapsed")
    bulk_btn = st.button("⚡ SCAN ALL", use_container_width=True)

    if bulk_btn and bulk_input.strip():
        urls = [u.strip() for u in bulk_input.split("\n") if u.strip()]
        rows = []
        bar  = st.progress(0, text="Scanning...")
        for i, url in enumerate(urls):
            u = url if url.startswith("http") else "http://"+url
            f = extract_features(u)
            p = model.predict(np.array(f).reshape(1,-1))[0]
            try: conf = float(model.predict_proba(np.array(f).reshape(1,-1))[0][p])
            except: conf = 0.5
            risk = compute_risk_score(int(p), conf, features=f)
            insert_url(url=u, ml_prediction=int(p), ml_confidence=conf,
                       risk_score=risk["score"], risk_grade=risk["grade"],
                       features=f, domain=urlparse(u).netloc)
            rows.append({"URL": url,
                         "Result":     "✅ Safe" if p==0 else "⚠️ Phishing",
                         "Risk Score": risk["score"],
                         "Grade":      risk["grade"],
                         "Confidence": f"{conf*100:.1f}%"})
            bar.progress((i+1)/len(urls), text=f"Scanned {i+1}/{len(urls)}")
        bar.empty()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        s = sum(1 for r in rows if "Safe" in r["Result"])
        st.markdown(f"""
        <div style='display:flex;gap:12px;margin-top:14px'>
            <div class='stat-tile' style='flex:1'><div class='stat-num' style='color:#e8eaf0'>{len(rows)}</div><div class='stat-lbl'>TOTAL</div></div>
            <div class='stat-tile' style='flex:1'><div class='stat-num' style='color:#00e676'>{s}</div><div class='stat-lbl'>SAFE</div></div>
            <div class='stat-tile' style='flex:1'><div class='stat-num' style='color:#ff3d3d'>{len(rows)-s}</div><div class='stat-lbl'>THREATS</div></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — INTELLIGENCE DASHBOARD
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='mono-label'>THREAT INTELLIGENCE OVERVIEW</div>",
                unsafe_allow_html=True)
    stats = get_dashboard_stats()
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px'>
        <div class='stat-tile'><div class='stat-num' style='color:#e8eaf0'>{stats['total']}</div><div class='stat-lbl'>TOTAL SCANNED</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#ff3d3d'>{stats['phishing']}</div><div class='stat-lbl'>THREATS</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#00e676'>{stats['safe']}</div><div class='stat-lbl'>SAFE</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#ffaa00'>{stats['avg_risk']}</div><div class='stat-lbl'>AVG RISK</div></div>
        <div class='stat-tile'><div class='stat-num' style='color:#ff3d3d'>{stats['campaigns']}</div><div class='stat-lbl'>CAMPAIGNS</div></div>
    </div>""", unsafe_allow_html=True)

    # Grade distribution
    st.markdown("<div class='mono-label'>RISK GRADE DISTRIBUTION</div>", unsafe_allow_html=True)
    grade_order = ["CRITICAL","HIGH","MEDIUM","LOW","SAFE","UNKNOWN"]
    html_grades = "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
    for g in grade_order:
        cnt = stats["grade_dist"].get(g, 0)
        gc  = grade_color(g)
        html_grades += f"""
        <div class='stat-tile' style='min-width:100px'>
            <div class='stat-num' style='color:{gc}'>{cnt}</div>
            <div class='stat-lbl'>{g}</div>
        </div>"""
    html_grades += "</div>"
    st.markdown(html_grades, unsafe_allow_html=True)

    # Recent scans table
    st.markdown("<div class='mono-label' style='margin-top:20px'>RECENT INTELLIGENCE RECORDS</div>",
                unsafe_allow_html=True)
    recent = get_recent_scans(15)
    if recent:
        df = pd.DataFrame(recent)
        df["ml_prediction"] = df["ml_prediction"].map({0:"✅ Safe", 1:"⚠️ Phishing"})
        df["ml_confidence"] = df["ml_confidence"].apply(lambda x: f"{x*100:.1f}%")
        df["risk_score"]    = df["risk_score"].apply(lambda x: f"{x:.1f}")
        st.dataframe(df[["url","scanned_at","ml_prediction","risk_score","risk_grade","vt_malicious"]],
                     use_container_width=True, hide_index=True)
    else:
        st.markdown("<div style='color:#3a4a6a;font-family:\"Space Mono\",monospace;font-size:.8rem'>No records yet — scan some URLs first!</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════
with tab4:
    for num, title, desc in [
        ("01", "URL Feature Extraction",     "12 numeric features extracted from the URL text — length, HTTPS, IP address, @ symbol, dots, hyphens, domain length, subdomains, suspicious keywords, port, path depth, digits in domain."),
        ("02", "XGBoost ML Classification",  "Features fed into XGBoost model trained on 11,000+ real URLs. Outputs binary prediction (Safe/Phishing) with confidence probability."),
        ("03", "VirusTotal API Scan",         "URL submitted to VirusTotal and checked against 90+ antivirus engines simultaneously — the same technology used by enterprise security teams."),
        ("04", "Threat Correlation Engine",   "New URL compared against all stored phishing URLs using domain similarity, cosine feature similarity and shared lexical patterns. Groups coordinated attacks into campaigns."),
        ("05", "Dynamic Risk Scoring",        "Combines ML confidence (35%), VirusTotal ratio (40%), threat similarity (15%) and feature penalties (10%) into a 0-100 risk score with SAFE/LOW/MEDIUM/HIGH/CRITICAL grade."),
        ("06", "Intelligence Database",       "Every scan stored in SQLite with features, scores, VT results and correlation data — enabling longitudinal threat analysis and campaign tracking."),
    ]:
        st.markdown(f"""
        <div style='background:#080c18;border:1px solid #121830;border-radius:14px;padding:20px;margin-bottom:12px'>
            <div style='font-size:2.5rem;font-weight:800;color:#1a3a7a;font-family:"Space Mono",monospace'>{num}</div>
            <div style='font-size:1rem;font-weight:700;color:#c8d0e8;margin:6px 0'>{title}</div>
            <div style='font-size:.85rem;color:#5a6a9a;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class='divider'></div>
<div style='text-align:center;font-family:"Space Mono",monospace;font-size:.65rem;color:#2a3a5a;padding-bottom:20px'>
    PhishGuard Research Edition · Python · XGBoost · SQLite · VirusTotal API · Streamlit
</div>""", unsafe_allow_html=True)