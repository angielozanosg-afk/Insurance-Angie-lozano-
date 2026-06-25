# ⚖️ Insurance Claims Bias Audit Dashboard

A professional Streamlit dashboard for auditing bias in insurance claim settlement processes.

## 📋 Features

| Module | Description |
|--------|-------------|
| **Descriptive Analytics** | Cross-tabulations, distributions, heatmaps against `POLICY_STATUS` |
| **Diagnostic / Bias Analysis** | Chi-Square, Mann-Whitney U, ANOVA, Cramér's V across age, income, zone, gender |
| **ML Models** | KNN, Decision Tree, Random Forest, Gradient Boosting with full feature engineering |
| **Model Performance** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrices, Overfitting check |
| **Findings & Scorecard** | Executive bias scorecard + regulatory recommendations |

## 🚀 Deploy on Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Insurance bias audit dashboard"
git remote add origin https://github.com/YOUR_USERNAME/insurance-bias-dashboard.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy**

Your app will be live at:  
`https://YOUR_USERNAME-insurance-bias-dashboard-app-XXXXX.streamlit.app`

## 🖥️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/insurance-bias-dashboard.git
cd insurance-bias-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 📂 File Structure

```
insurance-bias-dashboard/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 📊 Dataset

Upload `Insurance.csv` through the sidebar. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `POLICY_NO` | int | Policy identifier |
| `PI_GENDER` | str | M / F |
| `PI_AGE` | int | Policyholder age |
| `PI_ANNUAL_INCOME` | str | Annual income (comma-formatted) |
| `SUM_ASSURED` | str | Coverage amount (comma-formatted) |
| `ZONE` | str | Geographic/team zone |
| `PAYMENT_MODE` | str | Annual/Monthly/etc |
| `EARLY_NON` | str | EARLY / NON EARLY |
| `MEDICAL_NONMED` | str | MEDICAL / NON MEDICAL |
| `PI_OCCUPATION` | str | Occupation category |
| `REASON_FOR_CLAIM` | str | Cause of death |
| `POLICY_STATUS` | str | **Target** — Approved Death Claim / Repudiate Death |

## ⚖️ Legal / Regulatory Framework

- **4/5ths Rule** (EEOC Uniform Guidelines, 1978)
- **Disparate Impact Doctrine** (Griggs v. Duke Power Co., 1971)
- **IRDAI Claim Settlement Guidelines** (Circular IRDAI/Life/Cir/GV/120/06/2022)
- **EU Algorithmic Accountability Principles**
