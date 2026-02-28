"""
train_model.py  —  UPGRADED VERSION
────────────────────────────────────
Trains 3 models: Logistic Regression, Random Forest, XGBoost
Compares all 3, saves the best one.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model         import LogisticRegression
from sklearn.ensemble             import RandomForestClassifier
from sklearn.model_selection      import train_test_split, cross_val_score
from sklearn.metrics              import (accuracy_score, classification_report,
                                          confusion_matrix, roc_curve, auc)
from xgboost                      import XGBClassifier
from features                     import extract_features, FEATURE_NAMES

# ── 1. LOAD DATASET ──────────────────────────────────────────────────────────
print("📂 Loading dataset...")

CSV_FILE = "phishing_site_urls.csv"

if not os.path.exists(CSV_FILE):
    print(f"❌  '{CSV_FILE}' not found. Creating demo dataset...\n")
    import random, string

    def random_legit():
        domains = ["amazon","google","facebook","github","microsoft",
                   "netflix","apple","twitter","linkedin","wikipedia"]
        d    = random.choice(domains)
        path = "".join(random.choices(string.ascii_lowercase, k=random.randint(3,8)))
        return f"https://www.{d}.com/{path}"

    def random_phishing():
        words  = ["secure","login","verify","account","update",
                  "banking","paypal","confirm","password"]
        domain = f"amaz0n-{''.join(random.choices(string.ascii_lowercase,k=6))}"
        kw     = random.choice(words)
        tld    = random.choice([".xyz",".tk",".ml",".ga",".cf"])
        return (f"http://{domain}{tld}/{kw}"
                f"?user={''.join(random.choices(string.digits,k=8))}")

    rows = (
        [{"url": random_legit(),    "label": 0} for _ in range(500)] +
        [{"url": random_phishing(), "label": 1} for _ in range(500)]
    )
    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
    print("✅  Demo CSV created.\n")

df = pd.read_csv(CSV_FILE)
print(f"✅  Loaded {len(df)} rows.  Columns: {list(df.columns)}")

# Normalise column names
df.columns = [c.strip().lower() for c in df.columns]
url_col = next((c for c in df.columns if "url"   in c), None)
lbl_col = next((c for c in df.columns
                if c in ("label","status","result","phishing")), None)

if url_col is None or lbl_col is None:
    raise ValueError(
        f"Cannot find URL/label columns. Found: {list(df.columns)}\n"
        "Rename your columns to 'url' and 'label'.")

df = df[[url_col, lbl_col]].rename(columns={url_col:"url", lbl_col:"label"})

if df["label"].dtype == object:
    mapping = {"good":0,"bad":1,"legitimate":0,"phishing":1,
               "legit":0,"benign":0,"malicious":1}
    df["label"] = df["label"].str.lower().map(mapping)

df = df.dropna()
df["label"] = df["label"].astype(int)

print(f"\n📊 Class distribution:")
print(df["label"].value_counts().rename({0:"Legit (0)", 1:"Phishing (1)"}))

# ── 2. FEATURE EXTRACTION ────────────────────────────────────────────────────
print("\n⚙️  Extracting features (this may take a minute)...")
X = np.array([extract_features(url) for url in df["url"]])
y = df["label"].values
print(f"✅  Feature matrix: {X.shape}")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📚 Train: {len(X_train)}  |  🧪 Test: {len(X_test)}")

# ── 4. DEFINE ALL 3 MODELS ───────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200,
                                                  random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200,
                                         learning_rate=0.1,
                                         max_depth=6,
                                         eval_metric="logloss",
                                         random_state=42,
                                         verbosity=0),
}

results = {}

for name, model in models.items():
    print(f"\n🤖 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    # 5-fold cross validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    results[name] = {
        "model":    model,
        "preds":    preds,
        "accuracy": acc,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }

    print(f"   Test Accuracy  : {acc*100:.2f}%")
    print(f"   CV Accuracy    : {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")
    print(classification_report(y_test, preds,
                                target_names=["Legit","Phishing"]))

# ── 5. PICK BEST MODEL ───────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best model : {best_name}  "
      f"({results[best_name]['accuracy']*100:.2f}% accuracy)")

# ── 6. SAVE BEST MODEL ───────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("💾  Saved to model.pkl")

# ── 7. CHARTS ────────────────────────────────────────────────────────────────
sns.set_style("darkgrid")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Phishing URL Detector — Full Model Evaluation",
             fontsize=16, fontweight="bold", y=0.98)

# -- A) Confusion matrix (best model)
ax1 = fig.add_subplot(2, 3, 1)
cm  = confusion_matrix(y_test, results[best_name]["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["Legit","Phishing"],
            yticklabels=["Legit","Phishing"])
ax1.set_title(f"Confusion Matrix\n({best_name})")
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

# -- B) Accuracy comparison bar chart
ax2   = fig.add_subplot(2, 3, 2)
names = list(results.keys())
accs  = [results[n]["accuracy"]*100 for n in names]
colors= ["#4C72B0","#DD8452","#55A868"]
bars  = ax2.bar(names, accs, color=colors, width=0.5)
ax2.set_ylim(0, 115)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{acc:.1f}%", ha="center", fontweight="bold")

# -- C) Cross-validation scores
ax3      = fig.add_subplot(2, 3, 3)
cv_means = [results[n]["cv_mean"]*100 for n in names]
cv_stds  = [results[n]["cv_std"]*100  for n in names]
ax3.bar(names, cv_means, yerr=cv_stds, color=colors,
        width=0.5, capsize=8, alpha=0.85)
ax3.set_ylim(0, 115)
ax3.set_ylabel("CV Accuracy (%)")
ax3.set_title("5-Fold Cross Validation\n(mean ± std)")
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax3.text(i, m + s + 1.5, f"{m:.1f}%", ha="center", fontweight="bold")

# -- D) ROC curves (all 3 models)
ax4 = fig.add_subplot(2, 3, 4)
for name, color in zip(names, colors):
    model = results[name]["model"]
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc     = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {roc_auc:.3f})")
ax4.plot([0,1],[0,1],"k--", lw=1)
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curves — All Models")
ax4.legend(loc="lower right", fontsize=9)

# -- E) Feature importance (XGBoost or Random Forest)
ax5 = fig.add_subplot(2, 1, 2)
fi_model_name = "XGBoost" if "XGBoost" in results else "Random Forest"
fi_model      = results[fi_model_name]["model"]
importances   = fi_model.feature_importances_
sorted_idx    = np.argsort(importances)[::-1]
ax5.bar(range(len(FEATURE_NAMES)),
        importances[sorted_idx],
        color="#4C72B0", alpha=0.85)
ax5.set_xticks(range(len(FEATURE_NAMES)))
ax5.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx],
                    rotation=40, ha="right", fontsize=9)
ax5.set_ylabel("Importance Score")
ax5.set_title(f"Feature Importance ({fi_model_name})")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊  Chart saved → model_evaluation.png")

print("\n✅  All done!  Run:  streamlit run app.py")