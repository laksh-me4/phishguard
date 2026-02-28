# 🛡️ Phishing URL Detection System

A Machine Learning project that analyses a URL and predicts whether it is
**Safe** or **Phishing** — without ever opening the website.

---

## 📁 Project Structure

```
phishing-detector/
│
├── features.py           ← Extracts 12 numeric features from any URL
├── train_model.py        ← Trains + compares ML models, saves best one
├── app.py                ← Streamlit web interface (run this to use the app)
├── model.pkl             ← Saved trained model (created after training)
├── phishing_site_urls.csv← Dataset (add your Kaggle CSV here)
├── model_evaluation.png  ← Confusion matrix + accuracy chart
├── feature_importance.png← Which features matter most (Random Forest)
└── README.md             ← This file
```

---

## 🚀 How to Run (Step by Step)

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

### 2. Add your dataset
Place your Kaggle CSV file named `phishing_site_urls.csv` in this folder.
It needs two columns: `url` and `label` (0 = Legit, 1 = Phishing).

If you skip this step, a small demo dataset is auto-generated.

### 3. Train the model
```bash
python train_model.py
```
This will:
- Extract features from all URLs
- Train Logistic Regression + Random Forest
- Print accuracy of both
- Save the best model as `model.pkl`
- Show evaluation charts

### 4. Launch the web app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 🧠 Features Extracted from Each URL

| Feature               | What it checks                             |
|-----------------------|--------------------------------------------|
| URL Length            | Long URLs are suspicious                   |
| Has HTTPS             | Legit sites use HTTPS                      |
| Has IP address        | IP instead of domain name = suspicious     |
| Has @ symbol          | Used to hide real domain                   |
| Dot count             | Too many dots = suspicious subdomains      |
| Hyphen count          | amaz0n-secure-login.xyz style domains      |
| Domain length         | Short domains are usually legitimate       |
| Subdomain count       | Excessive subdomains are suspicious        |
| Suspicious keywords   | "login", "verify", "account", "password"  |
| Non-standard port     | :8080 in URL is suspicious                 |
| Slash count           | Very deep URL paths                        |
| Digits in domain      | amaz0n → replacing letters with numbers    |

---

## 📊 Models Used

| Model               | Type          | Notes                         |
|---------------------|---------------|-------------------------------|
| Logistic Regression | Linear        | Simple, fast, interpretable   |
| Random Forest       | Ensemble tree | More accurate, shows importance|

The best-performing model is automatically saved.

---

## 🌐 Deployment (Optional — to share online)

1. Push your project to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Click Deploy — free!

---

## 🎓 Technologies

- **Python 3.8+**
- **pandas / numpy** — data handling
- **scikit-learn** — machine learning
- **Streamlit** — web interface
- **matplotlib / seaborn** — charts
- **re / urllib** — URL parsing

---

## 💡 Ideas to Extend

- Add WHOIS lookup (domain age — new domains are suspicious)
- Add VirusTotal API integration for live scanning
- Add a browser extension
- Deploy to Render.com for free hosting
- Try XGBoost for even higher accuracy
