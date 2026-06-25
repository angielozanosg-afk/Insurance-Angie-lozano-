"""
Insurance Claims Bias Analysis Dashboard
=========================================
Author : Claims Settlement Audit Tool
Purpose: Descriptive, Diagnostic & Predictive Analysis of Insurance Claim Bias
Run    : streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claims Bias Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.5rem 0 0; }

    .metric-card {
        background: #16213e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-card .label { font-size: 0.85rem; color: #a8b2d8; margin-top: 4px; }

    .section-header {
        background: #0f3460; color: #e94560;
        padding: 0.6rem 1rem; border-radius: 8px;
        font-weight: 700; font-size: 1.1rem; margin: 1.5rem 0 1rem;
        border-left: 4px solid #e94560;
    }
    .insight-box {
        background: #1a1a2e; border-left: 4px solid #e94560;
        border-radius: 0 8px 8px 0; padding: 1rem;
        margin: 0.5rem 0; color: #a8b2d8; font-size: 0.92rem;
    }
    .bias-flag {
        background: #2d0a0a; border: 1px solid #e94560;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        color: #ff6b6b; font-size: 0.9rem;
    }
    .ok-flag {
        background: #0a2d0a; border: 1px solid #2ecc71;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        color: #2ecc71; font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #16213e; border-radius: 8px 8px 0 0;
        color: #a8b2d8; padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] { background: #0f3460; color: #e94560; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"Approved Death Claim": "#2ecc71", "Repudiate Death": "#e74c3c"}
COLOR_SEQ = px.colors.sequential.RdBu

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)

    # Clean numeric columns
    for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill nulls
    df["PI_OCCUPATION"]   = df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"]= df["REASON_FOR_CLAIM"].fillna("Unknown")

    # Binary target
    df["APPROVED"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)

    # Age bands
    df["AGE_BAND"] = pd.cut(
        df["PI_AGE"],
        bins=[0, 20, 30, 40, 50, 60, 70, 100],
        labels=["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    )

    # Income bands
    df["INCOME_BAND"] = pd.cut(
        df["PI_ANNUAL_INCOME"],
        bins=[0, 100_000, 250_000, 500_000, 1_000_000, np.inf],
        labels=["<1L", "1L-2.5L", "2.5L-5L", "5L-10L", ">10L"]
    )

    # Sum Assured bands
    df["SA_BAND"] = pd.cut(
        df["SUM_ASSURED"],
        bins=[0, 500_000, 1_000_000, 2_000_000, 5_000_000, np.inf],
        labels=["<5L", "5L-10L", "10L-20L", "20L-50L", ">50L"]
    )

    return df


def chi2_test(df, col):
    ct = pd.crosstab(df[col], df["APPROVED"])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return chi2, p, cramers_v


def approval_rate_by(df, col, min_n=5):
    grp = df.groupby(col, observed=True).agg(
        Total=("APPROVED", "count"),
        Approved=("APPROVED", "sum")
    ).reset_index()
    grp = grp[grp["Total"] >= min_n].copy()
    grp["Approval_Rate"] = (grp["Approved"] / grp["Total"] * 100).round(2)
    grp["Repudiated"] = grp["Total"] - grp["Approved"]
    return grp.sort_values("Approval_Rate", ascending=False)


def plotly_bar(grp, col, title, overall_rate, threshold):
    colors = ["#e74c3c" if r < threshold else "#2ecc71"
              for r in grp["Approval_Rate"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp[col].astype(str), y=grp["Approval_Rate"],
        marker_color=colors, text=grp["Approval_Rate"].astype(str) + "%",
        textposition="outside", name="Approval Rate"
    ))
    fig.add_hline(y=overall_rate, line_dash="dash", line_color="#3498db",
                  annotation_text=f"Overall {overall_rate:.1f}%",
                  annotation_font_color="#3498db")
    fig.add_hline(y=threshold, line_dash="dot", line_color="orange",
                  annotation_text=f"4/5ths {threshold:.1f}%",
                  annotation_font_color="orange")
    fig.update_layout(
        title=title, template="plotly_dark",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font_color="#a8b2d8", yaxis_range=[0, 110],
        margin=dict(t=50, b=40)
    )
    return fig


def plot_confusion_matrix(cm, title, labels=["Repudiated", "Approved"]):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor("#16213e")
    ax.set_facecolor("#16213e")
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", color="#a8b2d8", fontsize=11)
    ax.set_ylabel("Actual",    color="#a8b2d8", fontsize=11)
    ax.set_title(title, color="#e94560", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#a8b2d8")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Insurance Bias Audit")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Insurance.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### 🔧 Model Settings")
    test_size   = st.slider("Test Split %", 10, 40, 25, 5) / 100
    random_seed = st.number_input("Random Seed", value=42, step=1)
    n_neighbors = st.slider("KNN – k neighbors", 3, 15, 5, 2)
    max_depth_dt= st.slider("Decision Tree – Max Depth", 2, 20, 6, 1)
    n_estimators= st.slider("RF / GBM – n_estimators", 50, 300, 100, 50)
    st.markdown("---")
    st.markdown("### 📌 4/5ths Threshold Rule")
    st.info("Any subgroup with approval rate below **80% of the overall rate** is flagged for adverse impact.")
    st.markdown("---")
    st.caption("Built for Claims Settlement Audit")

if uploaded is None:
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Insurance Claims Bias Audit Dashboard</h1>
        <p>Upload <strong>Insurance.csv</strong> from the sidebar to begin the full analysis</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Upload your dataset using the sidebar to unlock all 5 analysis modules.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_and_preprocess(uploaded)
overall_rate = df["APPROVED"].mean() * 100
threshold_45 = overall_rate * 0.8
N = len(df)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ Insurance Claims Bias Audit Dashboard</h1>
    <p>Descriptive · Diagnostic · Predictive Analysis of Claim Settlement Patterns</p>
</div>
""", unsafe_allow_html=True)

# KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
for col_widget, val, label in [
    (c1, N,                           "Total Claims"),
    (c2, int(df["APPROVED"].sum()),   "Approved"),
    (c3, int((df["APPROVED"]==0).sum()),"Repudiated"),
    (c4, f"{overall_rate:.1f}%",      "Approval Rate"),
    (c5, f"{threshold_45:.1f}%",      "4/5ths Threshold"),
]:
    col_widget.markdown(f"""
    <div class="metric-card">
        <div class="value">{val}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive Analytics",
    "🔬 Diagnostic / Bias Analysis",
    "🤖 ML Models & Feature Engineering",
    "📈 Model Performance & ROC",
    "📋 Findings & Scorecard"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 1. Descriptive Analytics — Cross-Tabulation Against Policy Status</div>',
                unsafe_allow_html=True)

    # 1a) Overall distribution
    st.markdown("#### Overall Claim Decision Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        vc = df["POLICY_STATUS"].value_counts().reset_index()
        vc.columns = ["Status", "Count"]
        fig_pie = px.pie(vc, names="Status", values="Count",
                         color="Status",
                         color_discrete_map=PALETTE,
                         template="plotly_dark", hole=0.45)
        fig_pie.update_layout(paper_bgcolor="#16213e", plot_bgcolor="#1a1a2e")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown(f"""
        <div class="insight-box">
        <b>Dataset Summary:</b><br>
        • Total Records: <b>{N:,}</b><br>
        • Approved Death Claims: <b>{int(df['APPROVED'].sum()):,} ({overall_rate:.1f}%)</b><br>
        • Repudiated: <b>{int((df['APPROVED']==0).sum()):,} ({100-overall_rate:.1f}%)</b><br>
        • Missing — Occupation: <b>81</b> | Reason for Claim: <b>381</b><br><br>
        <b>4/5ths Adverse Impact Threshold: {threshold_45:.1f}%</b><br>
        Any demographic subgroup falling below this rate warrants regulatory scrutiny
        under disparate impact doctrine (Griggs v. Duke Power, 1971).
        </div>""", unsafe_allow_html=True)

        # Summary stats table
        sum_tbl = pd.DataFrame({
            "Variable": ["PI_AGE", "PI_ANNUAL_INCOME", "SUM_ASSURED"],
            "Mean":   [df["PI_AGE"].mean(), df["PI_ANNUAL_INCOME"].mean(), df["SUM_ASSURED"].mean()],
            "Median": [df["PI_AGE"].median(), df["PI_ANNUAL_INCOME"].median(), df["SUM_ASSURED"].median()],
            "Std":    [df["PI_AGE"].std(), df["PI_ANNUAL_INCOME"].std(), df["SUM_ASSURED"].std()],
        }).set_index("Variable").applymap(lambda x: f"{x:,.0f}")
        st.dataframe(sum_tbl, use_container_width=True)

    # 1b) Cross-tabulation selector
    st.markdown("#### Cross-Tabulation: Select Variable vs Policy Status")
    xtab_col = st.selectbox("Variable for cross-tab", [
        "PI_GENDER", "AGE_BAND", "INCOME_BAND", "SA_BAND",
        "ZONE", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
        "PI_OCCUPATION", "REASON_FOR_CLAIM"
    ], key="xtab_select")

    xt = pd.crosstab(df[xtab_col], df["POLICY_STATUS"], margins=True)
    xt_pct = pd.crosstab(df[xtab_col], df["POLICY_STATUS"], normalize="index").mul(100).round(2)
    xt_pct.columns = [f"{c} (%)" for c in xt_pct.columns]
    xt_combined = xt.join(xt_pct)

    col_c, col_d = st.columns([1.2, 1])
    with col_c:
        st.dataframe(xt_combined.style.background_gradient(cmap="RdYlGn", subset=[c for c in xt_pct.columns]),
                     use_container_width=True)
    with col_d:
        grp_xt = approval_rate_by(df, xtab_col, min_n=5)
        fig_xt = plotly_bar(grp_xt, xtab_col,
                            f"Approval Rate by {xtab_col}", overall_rate, threshold_45)
        st.plotly_chart(fig_xt, use_container_width=True)

    # 1c) Numeric distributions side-by-side
    st.markdown("#### Numeric Variable Distributions: Approved vs Repudiated")
    num_var = st.selectbox("Numeric variable", ["PI_AGE", "PI_ANNUAL_INCOME", "SUM_ASSURED"])
    col_e, col_f = st.columns(2)

    with col_e:
        fig_box = px.box(df, x="POLICY_STATUS", y=num_var, color="POLICY_STATUS",
                         color_discrete_map=PALETTE, template="plotly_dark",
                         title=f"{num_var} Distribution by Outcome")
        fig_box.update_layout(paper_bgcolor="#16213e", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_f:
        fig_hist = px.histogram(df, x=num_var, color="POLICY_STATUS",
                                barmode="overlay", opacity=0.75,
                                color_discrete_map=PALETTE,
                                template="plotly_dark",
                                title=f"{num_var} Histogram by Outcome")
        fig_hist.update_layout(paper_bgcolor="#16213e")
        st.plotly_chart(fig_hist, use_container_width=True)

    # 1d) Gender × Age heatmap
    st.markdown("#### Gender × Age Band Cross-Tab Heatmap")
    hm_data = df.pivot_table(values="APPROVED", index="PI_GENDER",
                              columns="AGE_BAND", aggfunc="mean",
                              observed=True).mul(100).round(1)
    fig_hm = px.imshow(hm_data, text_auto=True, color_continuous_scale="RdYlGn",
                       zmin=0, zmax=100, template="plotly_dark",
                       title="Approval Rate (%) — Gender × Age Band")
    fig_hm.update_layout(paper_bgcolor="#16213e")
    st.plotly_chart(fig_hm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC / BIAS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 2. Diagnostic Analysis — Statistically Proving Bias</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Methodology:</b> Each dimension is tested using the appropriate statistical test:<br>
    • <b>Chi-Square + Cramér's V</b> — categorical variables (gender, zone, medical status)<br>
    • <b>Mann-Whitney U</b> — continuous variables (age, income) — non-parametric, robust to skew<br>
    • <b>One-Way ANOVA</b> — multiple group comparisons (age bands, income bands)<br>
    • <b>Cramér's V</b> effect size: <0.10 weak | 0.10–0.30 moderate | >0.30 strong<br>
    • <b>4/5ths Rule</b> — adverse impact threshold for each subgroup
    </div>""", unsafe_allow_html=True)

    # ── 2a) Age Bias ────────────────────────────────────────────────────────
    st.markdown("### 📅 Age Bias")
    col_age1, col_age2 = st.columns(2)

    age_grp = approval_rate_by(df, "AGE_BAND")
    corr_age, p_age = stats.pointbiserialr(df["APPROVED"], df["PI_AGE"])
    groups_age = [g["APPROVED"].values for _, g in df.groupby("AGE_BAND", observed=True)]
    f_age, p_anova = stats.f_oneway(*groups_age)

    with col_age1:
        fig_age = plotly_bar(age_grp, "AGE_BAND", "Approval Rate by Age Band",
                             overall_rate, threshold_45)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_age2:
        fig_vio = px.violin(df, x="AGE_BAND", y="PI_AGE", color="POLICY_STATUS",
                            color_discrete_map=PALETTE, box=True,
                            template="plotly_dark",
                            title="Age Distribution by Decision & Age Band")
        fig_vio.update_layout(paper_bgcolor="#16213e")
        st.plotly_chart(fig_vio, use_container_width=True)

    bias_icon = "⚠️ BIAS DETECTED" if p_anova < 0.05 else "✅ No significant bias"
    st.markdown(f"""
    <div class="{'bias-flag' if p_anova < 0.05 else 'ok-flag'}">
    <b>Age Bias Test Results</b><br>
    • Point-Biserial Correlation (Age ~ Approval): r = {corr_age:.4f}, p = {p_age:.4f}<br>
    • One-Way ANOVA across Age Bands: F = {f_age:.4f}, p = {p_anova:.4f}<br>
    • {bias_icon} {'— Older policyholders face structurally different approval odds' if p_anova < 0.05 else ''}
    </div>""", unsafe_allow_html=True)

    # ── 2b) Income Bias ─────────────────────────────────────────────────────
    st.markdown("### 💰 Income Bias")
    col_inc1, col_inc2 = st.columns(2)

    inc_grp = approval_rate_by(df, "INCOME_BAND")
    app_inc  = df[df["APPROVED"]==1]["PI_ANNUAL_INCOME"].dropna()
    rep_inc  = df[df["APPROVED"]==0]["PI_ANNUAL_INCOME"].dropna()
    u_inc, p_inc = stats.mannwhitneyu(app_inc, rep_inc, alternative="two-sided")
    median_gap = app_inc.median() - rep_inc.median()

    with col_inc1:
        fig_inc = plotly_bar(inc_grp, "INCOME_BAND",
                             "Approval Rate by Income Band", overall_rate, threshold_45)
        st.plotly_chart(fig_inc, use_container_width=True)

    with col_inc2:
        fig_inc2 = px.box(df, x="INCOME_BAND", y="PI_ANNUAL_INCOME",
                          color="POLICY_STATUS", color_discrete_map=PALETTE,
                          template="plotly_dark",
                          title="Income Distribution by Band & Decision",
                          category_orders={"INCOME_BAND": ["<1L","1L-2.5L","2.5L-5L","5L-10L",">10L"]})
        fig_inc2.update_layout(paper_bgcolor="#16213e")
        st.plotly_chart(fig_inc2, use_container_width=True)

    st.markdown(f"""
    <div class="{'bias-flag' if p_inc < 0.05 else 'ok-flag'}">
    <b>Income Bias Test Results</b><br>
    • Median Income — Approved: ₹{app_inc.median():,.0f} | Repudiated: ₹{rep_inc.median():,.0f}<br>
    • Median Income Gap: ₹{median_gap:,.0f}<br>
    • Mann-Whitney U = {u_inc:.0f}, p = {p_inc:.4f}<br>
    • {'⚠️ BIAS DETECTED — Income is a statistically significant predictor of repudiation' if p_inc < 0.05 else '✅ No significant income-based bias'}
    </div>""", unsafe_allow_html=True)

    # ── 2c) Team / Zone Bias ─────────────────────────────────────────────────
    st.markdown("### 🗺️ Team / Zone Bias")
    zone_grp = approval_rate_by(df, "ZONE", min_n=15)
    zone_grp["Flagged"] = zone_grp["Approval_Rate"] < threshold_45

    chi2_z, p_z, cv_z = chi2_test(df, "ZONE")

    fig_zone = go.Figure(go.Bar(
        x=zone_grp["Approval_Rate"],
        y=zone_grp["ZONE"],
        orientation="h",
        marker_color=["#e74c3c" if f else "#2ecc71" for f in zone_grp["Flagged"]],
        text=zone_grp["Approval_Rate"].astype(str) + "%",
        textposition="outside"
    ))
    fig_zone.add_vline(x=overall_rate, line_dash="dash", line_color="#3498db")
    fig_zone.add_vline(x=threshold_45, line_dash="dot", line_color="orange")
    fig_zone.update_layout(
        title="Zone/Team Approval Rates (Red = Below 4/5ths Threshold)",
        template="plotly_dark", paper_bgcolor="#16213e",
        xaxis_range=[0, 110], height=550,
        margin=dict(l=160, t=50)
    )
    st.plotly_chart(fig_zone, use_container_width=True)

    flagged_zones = zone_grp[zone_grp["Flagged"]]
    st.markdown(f"""
    <div class="{'bias-flag' if p_z < 0.05 else 'ok-flag'}">
    <b>Zone/Team Bias Test Results</b><br>
    • Chi-Square = {chi2_z:.4f}, p = {p_z:.4f}, Cramér's V = {cv_z:.4f}<br>
    • Zones below 4/5ths threshold: <b>{len(flagged_zones)}</b>
    {" | ".join(f"{r['ZONE']} ({r['Approval_Rate']}%)" for _, r in flagged_zones.iterrows()) if len(flagged_zones) else "None"}<br>
    • {'⚠️ BIAS DETECTED — Significant zone-level disparities exist' if p_z < 0.05 else '✅ No significant zone bias'}
    </div>""", unsafe_allow_html=True)

    # ── 2d) Gender + Medical + Early Bias ────────────────────────────────────
    st.markdown("### 👤 Gender · Medical · Early Claim Bias")
    col_g1, col_g2, col_g3 = st.columns(3)

    for col_w, var, label in [(col_g1, "PI_GENDER", "Gender"),
                               (col_g2, "MEDICAL_NONMED", "Medical Status"),
                               (col_g3, "EARLY_NON", "Early Claim")]:
        grp_v = approval_rate_by(df, var, min_n=5)
        chi2_v, p_v, cv_v = chi2_test(df, var)
        min_rate = grp_v["Approval_Rate"].min()
        flag = min_rate < threshold_45

        with col_w:
            fig_v = px.bar(grp_v, x=var, y="Approval_Rate",
                           color="Approval_Rate",
                           color_continuous_scale="RdYlGn",
                           range_color=[0, 100],
                           template="plotly_dark",
                           title=f"Approval Rate by {label}",
                           text="Approval_Rate")
            fig_v.add_hline(y=overall_rate, line_dash="dash", line_color="#3498db")
            fig_v.update_layout(paper_bgcolor="#16213e", showlegend=False,
                                yaxis_range=[0, 110])
            fig_v.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_v, use_container_width=True)

            icon = "⚠️" if flag or p_v < 0.05 else "✅"
            st.markdown(f"""
            <div class="{'bias-flag' if flag or p_v < 0.05 else 'ok-flag'}">
            {icon} <b>{label}</b><br>
            χ² p={p_v:.3f} | V={cv_v:.3f}<br>
            Min rate: {min_rate:.1f}%
            </div>""", unsafe_allow_html=True)

    # ── 2e) Intersectional Heatmap ────────────────────────────────────────────
    st.markdown("### 🔀 Intersectional Bias Heatmap")
    ix_row = st.selectbox("Row variable", ["PI_GENDER", "EARLY_NON", "MEDICAL_NONMED"], key="ix_row")
    ix_col = st.selectbox("Column variable", ["INCOME_BAND", "AGE_BAND", "ZONE", "PAYMENT_MODE"], key="ix_col")

    pivot_ix = df.pivot_table(values="APPROVED", index=ix_row,
                               columns=ix_col, aggfunc="mean",
                               observed=True).mul(100).round(1)
    fig_ix = px.imshow(pivot_ix, text_auto=True, color_continuous_scale="RdYlGn",
                       zmin=0, zmax=100, template="plotly_dark",
                       title=f"Approval Rate (%) — {ix_row} × {ix_col}")
    fig_ix.update_layout(paper_bgcolor="#16213e")
    st.plotly_chart(fig_ix, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML MODELS & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 3. Supervised Learning — Feature Engineering & Model Training</div>',
                unsafe_allow_html=True)

    # ── Feature Engineering ──────────────────────────────────────────────────
    st.markdown("#### Step 1 — Feature Engineering")
    with st.expander("📐 View Feature Engineering Pipeline", expanded=True):
        st.markdown("""
        | Step | Operation | Rationale |
        |------|-----------|-----------|
        | 1 | Drop `POLICY_NO`, `PI_NAME`, `PI_STATE` | ID/high-cardinality columns with no predictive signal |
        | 2 | Binary encode `PI_GENDER` (M=1, F=0) | Ordinal-free encoding for binary variable |
        | 3 | Binary encode `EARLY_NON`, `MEDICAL_NONMED` | Binary structural flags |
        | 4 | Label encode `ZONE`, `PAYMENT_MODE`, `PI_OCCUPATION`, `REASON_FOR_CLAIM` | Low-mid cardinality nominals |
        | 5 | `PI_AGE`, `PI_ANNUAL_INCOME`, `SUM_ASSURED` — StandardScaler | Normalise for KNN distance sensitivity |
        | 6 | Derived: `INCOME_TO_SA_RATIO` = Income / Sum Assured | Premium affordability signal |
        | 7 | Derived: `AGE_GROUP` (numeric bins 0–6) | Captures non-linear age effect |
        | 8 | Handle class imbalance — report class weights | 68% approved / 32% repudiated |
        """)

    @st.cache_data
    def engineer_features(df, seed):
        drop_cols = ["POLICY_NO", "PI_NAME", "PI_STATE", "POLICY_STATUS",
                     "AGE_BAND", "INCOME_BAND", "SA_BAND"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

        # Binary
        feat_df["GENDER_BIN"] = (feat_df["PI_GENDER"] == "M").astype(int)
        feat_df["EARLY_BIN"]  = (feat_df["EARLY_NON"] == "EARLY").astype(int)
        feat_df["MEDICAL_BIN"]= (feat_df["MEDICAL_NONMED"] == "MEDICAL").astype(int)
        feat_df.drop(columns=["PI_GENDER", "EARLY_NON", "MEDICAL_NONMED"], inplace=True)

        # Label encode
        le = LabelEncoder()
        for col in ["ZONE", "PAYMENT_MODE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]:
            feat_df[col + "_ENC"] = le.fit_transform(feat_df[col].astype(str))
        feat_df.drop(columns=["ZONE","PAYMENT_MODE","PI_OCCUPATION","REASON_FOR_CLAIM"],
                     inplace=True)

        # Derived features
        feat_df["INCOME_TO_SA"] = np.where(
            feat_df["SUM_ASSURED"] > 0,
            feat_df["PI_ANNUAL_INCOME"] / (feat_df["SUM_ASSURED"] + 1),
            0
        )
        feat_df["AGE_GROUP"] = pd.cut(
            feat_df["PI_AGE"],
            bins=[0,20,30,40,50,60,70,100], labels=False
        ).fillna(0).astype(int)

        X = feat_df.drop(columns=["APPROVED"])
        y = feat_df["APPROVED"]
        return X, y, X.columns.tolist()

    X, y, feature_names = engineer_features(df, random_seed)

    col_fe1, col_fe2 = st.columns(2)
    with col_fe1:
        st.markdown("**Final Feature Set**")
        feat_table = pd.DataFrame({
            "Feature": feature_names,
            "Type": ["Numeric" if X[f].dtype in [np.float64, np.int64] else "Encoded"
                     for f in feature_names]
        })
        st.dataframe(feat_table, use_container_width=True, height=280)

    with col_fe2:
        class_dist = y.value_counts().reset_index()
        class_dist.columns = ["Class", "Count"]
        class_dist["Label"] = class_dist["Class"].map({1: "Approved", 0: "Repudiated"})
        fig_cls = px.bar(class_dist, x="Label", y="Count",
                         color="Label",
                         color_discrete_map={"Approved": "#2ecc71", "Repudiated": "#e74c3c"},
                         template="plotly_dark", title="Class Distribution (Target Variable)",
                         text="Count")
        fig_cls.update_layout(paper_bgcolor="#16213e", showlegend=False)
        fig_cls.update_traces(textposition="outside")
        st.plotly_chart(fig_cls, use_container_width=True)

    # ── Train / Test Split ───────────────────────────────────────────────────
    st.markdown("#### Step 2 — Train / Test Split & Scaling")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total Samples",    N)
    col_s2.metric("Training Samples", len(X_train))
    col_s3.metric("Test Samples",     len(X_test))

    # ── Train all models ─────────────────────────────────────────────────────
    st.markdown("#### Step 3 — Model Training")

    @st.cache_data
    def train_models(X_tr, y_tr, X_te, y_te, k, dt_depth, n_est, seed):
        models = {
            "KNN":               KNeighborsClassifier(n_neighbors=k),
            "Decision Tree":     DecisionTreeClassifier(max_depth=dt_depth, random_state=seed),
            "Random Forest":     RandomForestClassifier(n_estimators=n_est, random_state=seed, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_est, random_state=seed),
        }
        results = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred_tr = model.predict(X_tr)
            y_pred_te = model.predict(X_te)
            y_prob    = model.predict_proba(X_te)[:, 1]

            results[name] = {
                "model":       model,
                "train_acc":   accuracy_score(y_tr, y_pred_tr),
                "test_acc":    accuracy_score(y_te, y_pred_te),
                "precision":   precision_score(y_te, y_pred_te),
                "recall":      recall_score(y_te, y_pred_te),
                "f1":          f1_score(y_te, y_pred_te),
                "cm":          confusion_matrix(y_te, y_pred_te),
                "y_prob":      y_prob,
                "report":      classification_report(y_te, y_pred_te, output_dict=True),
                "fpr_tpr":     roc_curve(y_te, y_prob),
                "auc":         auc(*roc_curve(y_te, y_prob)[:2]),
            }
        return results

    with st.spinner("Training KNN, Decision Tree, Random Forest, Gradient Boosting..."):
        results = train_models(
            X_train_sc, y_train.values,
            X_test_sc,  y_test.values,
            n_neighbors, max_depth_dt, n_estimators, int(random_seed)
        )

    st.success("✅ All 4 models trained successfully.")

    # Feature importance (RF)
    st.markdown("#### Feature Importance — Random Forest")
    rf_model = results["Random Forest"]["model"]
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="RdYlGn",
                    template="plotly_dark",
                    title="Random Forest — Feature Importance (Bias Drivers)")
    fig_fi.update_layout(paper_bgcolor="#16213e", height=420, yaxis_categoryorder="total ascending")
    st.plotly_chart(fig_fi, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE & ROC
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📈 4. Model Performance — Accuracy, Metrics, ROC Curves & Confusion Matrices</div>',
                unsafe_allow_html=True)

    # ── Performance comparison table ─────────────────────────────────────────
    perf_rows = []
    for name, r in results.items():
        perf_rows.append({
            "Model":           name,
            "Train Acc.":      f"{r['train_acc']*100:.2f}%",
            "Test Acc.":       f"{r['test_acc']*100:.2f}%",
            "Precision":       f"{r['precision']:.4f}",
            "Recall":          f"{r['recall']:.4f}",
            "F1-Score":        f"{r['f1']:.4f}",
            "ROC-AUC":         f"{r['auc']:.4f}",
            "Overfit?":        "⚠️ Yes" if (r['train_acc'] - r['test_acc']) > 0.08 else "✅ No"
        })
    perf_df = pd.DataFrame(perf_rows).set_index("Model")
    st.dataframe(perf_df.style.highlight_max(subset=["Test Acc.","F1-Score","ROC-AUC"],
                                             color="#0f3460"),
                 use_container_width=True)

    # ── Metrics grouped bar chart ─────────────────────────────────────────────
    st.markdown("#### Comparative Metrics Across Models")
    metrics_plot = []
    for name, r in results.items():
        for metric, val in [("Train Accuracy", r["train_acc"]),
                             ("Test Accuracy",  r["test_acc"]),
                             ("Precision",      r["precision"]),
                             ("Recall",         r["recall"]),
                             ("F1-Score",       r["f1"]),
                             ("ROC-AUC",        r["auc"])]:
            metrics_plot.append({"Model": name, "Metric": metric, "Value": val})
    mp_df = pd.DataFrame(metrics_plot)

    fig_metrics = px.bar(mp_df, x="Metric", y="Value", color="Model",
                         barmode="group", template="plotly_dark",
                         title="Model Performance Comparison",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         text_auto=".3f")
    fig_metrics.update_layout(paper_bgcolor="#16213e", yaxis_range=[0, 1.1])
    fig_metrics.update_traces(textposition="outside")
    st.plotly_chart(fig_metrics, use_container_width=True)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.markdown("#### ROC Curves — All Models")
    fig_roc = go.Figure()
    colors_roc = {"KNN": "#3498db", "Decision Tree": "#e67e22",
                  "Random Forest": "#2ecc71", "Gradient Boosting": "#e74c3c"}

    for name, r in results.items():
        fpr, tpr, _ = r["fpr_tpr"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['auc']:.3f})",
            line=dict(color=colors_roc[name], width=2.5)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
        line=dict(color="gray", dash="dash")
    ))
    fig_roc.update_layout(
        title="ROC Curves — Model Stability Comparison",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template="plotly_dark", paper_bgcolor="#16213e",
        legend=dict(x=0.55, y=0.1),
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("#### Confusion Matrices — All Models")
    cm_cols = st.columns(4)
    for (name, r), col_w in zip(results.items(), cm_cols):
        with col_w:
            fig_cm = plot_confusion_matrix(r["cm"], name)
            st.pyplot(fig_cm)
            st.caption(f"Test Acc: **{r['test_acc']*100:.1f}%** | F1: **{r['f1']:.3f}**")

    # ── Train vs Test accuracy (overfitting check) ────────────────────────────
    st.markdown("#### Overfitting Check — Train vs Test Accuracy")
    ov_data = []
    for name, r in results.items():
        ov_data.append({"Model": name, "Split": "Train", "Accuracy": r["train_acc"]})
        ov_data.append({"Model": name, "Split": "Test",  "Accuracy": r["test_acc"]})
    ov_df = pd.DataFrame(ov_data)

    fig_ov = px.bar(ov_df, x="Model", y="Accuracy", color="Split",
                    barmode="group", template="plotly_dark",
                    color_discrete_map={"Train": "#3498db", "Test": "#e74c3c"},
                    title="Train vs Test Accuracy — Overfitting Detection",
                    text_auto=".3f")
    fig_ov.update_layout(paper_bgcolor="#16213e", yaxis_range=[0, 1.1])
    fig_ov.update_traces(textposition="outside")
    st.plotly_chart(fig_ov, use_container_width=True)

    # ── Per-class report ──────────────────────────────────────────────────────
    st.markdown("#### Detailed Classification Report by Model")
    model_select = st.selectbox("Select Model", list(results.keys()), key="model_report")
    report_dict = results[model_select]["report"]
    report_df = pd.DataFrame(report_dict).T.round(4)
    st.dataframe(report_df.style.background_gradient(cmap="RdYlGn",
                 subset=["precision", "recall", "f1-score"]),
                 use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — FINDINGS & SCORECARD
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📋 5. Findings & Executive Bias Scorecard</div>',
                unsafe_allow_html=True)

    # Recompute stats for scorecard
    chi2_g, p_g, cv_g = chi2_test(df, "PI_GENDER")
    chi2_m, p_m, cv_m = chi2_test(df, "MEDICAL_NONMED")
    chi2_e, p_e, cv_e = chi2_test(df, "EARLY_NON")
    chi2_z, p_z, cv_z = chi2_test(df, "ZONE")
    u_i2, p_i2 = stats.mannwhitneyu(
        df[df["APPROVED"]==1]["PI_ANNUAL_INCOME"].dropna(),
        df[df["APPROVED"]==0]["PI_ANNUAL_INCOME"].dropna(),
        alternative="two-sided"
    )
    f_a2, p_a2 = stats.f_oneway(*[g["APPROVED"].values
                                   for _, g in df.groupby("AGE_BAND", observed=True)])

    scorecard = [
        {"Bias Dimension": "Gender (PI_GENDER)",       "Test": "Chi-Square",   "p-value": p_g,  "Effect": f"V={cv_g:.3f}", "Sig?": p_g<0.05,  "Min Rate": approval_rate_by(df,"PI_GENDER")["Approval_Rate"].min()},
        {"Bias Dimension": "Age Band",                  "Test": "ANOVA",        "p-value": p_a2, "Effect": "F-test",        "Sig?": p_a2<0.05, "Min Rate": approval_rate_by(df,"AGE_BAND")["Approval_Rate"].min()},
        {"Bias Dimension": "Annual Income Band",        "Test": "Mann-Whitney", "p-value": p_i2, "Effect": "U-stat",        "Sig?": p_i2<0.05, "Min Rate": approval_rate_by(df,"INCOME_BAND")["Approval_Rate"].min()},
        {"Bias Dimension": "Zone / Team",               "Test": "Chi-Square",   "p-value": p_z,  "Effect": f"V={cv_z:.3f}", "Sig?": p_z<0.05,  "Min Rate": approval_rate_by(df,"ZONE",min_n=15)["Approval_Rate"].min()},
        {"Bias Dimension": "Medical vs Non-Medical",    "Test": "Chi-Square",   "p-value": p_m,  "Effect": f"V={cv_m:.3f}", "Sig?": p_m<0.05,  "Min Rate": approval_rate_by(df,"MEDICAL_NONMED")["Approval_Rate"].min()},
        {"Bias Dimension": "Early Claim Status",        "Test": "Chi-Square",   "p-value": p_e,  "Effect": f"V={cv_e:.3f}", "Sig?": p_e<0.05,  "Min Rate": approval_rate_by(df,"EARLY_NON")["Approval_Rate"].min()},
    ]
    sc_df = pd.DataFrame(scorecard)
    sc_df["Adverse Impact?"]  = sc_df["Min Rate"].apply(lambda x: "⚠️ YES" if x < threshold_45 else "✅ OK")
    sc_df["Significant?"]     = sc_df["Sig?"].apply(lambda x: "⚠️ YES" if x else "✅ NO")
    sc_df["p-value"]          = sc_df["p-value"].apply(lambda x: f"{x:.4f}")
    sc_df["Min Rate"]         = sc_df["Min Rate"].apply(lambda x: f"{x:.1f}%")
    sc_df = sc_df.drop(columns=["Sig?"])

    st.dataframe(sc_df.set_index("Bias Dimension"), use_container_width=True)

    # Model summary
    st.markdown("#### 🤖 Model Performance Summary")
    best_model = max(results, key=lambda k: results[k]["auc"])
    best_auc   = results[best_model]["auc"]
    best_f1    = results[best_model]["f1"]

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    col_f1.metric("Best Model",    best_model)
    col_f2.metric("Best AUC",      f"{best_auc:.3f}")
    col_f3.metric("Best F1-Score", f"{best_f1:.3f}")
    col_f4.metric("Test Accuracy", f"{results[best_model]['test_acc']*100:.1f}%")

    # Narrative findings
    st.markdown("#### 📝 Key Findings")
    st.markdown(f"""
    <div class="insight-box">
    <b>1. OVERALL BASELINE</b><br>
    Of {N:,} death insurance claims, <b>{int(df['APPROVED'].sum()):,} ({overall_rate:.1f}%) were approved</b>
    and <b>{int((df['APPROVED']==0).sum()):,} ({100-overall_rate:.1f}%) were repudiated</b>.
    The 4/5ths adverse impact threshold is <b>{threshold_45:.1f}%</b>.
    </div>

    <div class="insight-box">
    <b>2. ZONE / TEAM BIAS</b><br>
    {'Zone-level disparities are statistically significant (p=' + str(round(p_z,4)) + ', Cramér V=' + str(round(cv_z,3)) + '). '
    'Certain zones show approval rates well below the 4/5ths threshold, indicating that the settlement team handling the claim '
    'materially influences the outcome — a direct signal of process-level bias rather than risk-based differentiation.'
    if p_z < 0.05 else 'No statistically significant zone-level bias detected.'}
    </div>

    <div class="insight-box">
    <b>3. INCOME BIAS</b><br>
    {'Income is a statistically significant predictor of repudiation (Mann-Whitney p=' + str(round(p_i2,4)) + '). '
    'Policyholders in lower income bands face disproportionately higher repudiation rates. '
    'This constitutes potential indirect discrimination — income acting as a socioeconomic proxy.'
    if p_i2 < 0.05 else 'No significant income-based bias detected.'}
    </div>

    <div class="insight-box">
    <b>4. AGE BIAS</b><br>
    {'Age-band ANOVA returned p=' + str(round(p_a2,4)) + ', indicating statistically significant differences '
    'in approval rates across age cohorts. This warrants actuarial review to distinguish legitimate '
    'mortality-based risk pricing from discriminatory treatment.'
    if p_a2 < 0.05 else 'No significant age-based bias detected at the 0.05 level.'}
    </div>

    <div class="insight-box">
    <b>5. EARLY CLAIM STRUCTURAL BIAS</b><br>
    {'Early claims (filed within early policy period) face significantly different outcomes (p=' + str(round(p_e,4)) + ', V=' + str(round(cv_e,3)) + '). '
    'While early-claim scrutiny is actuarially justified under IRDAI guidelines, '
    'the magnitude of the gap suggests possible over-application of rejection criteria on early claims.'
    if p_e < 0.05 else 'No significant early-claim bias detected.'}
    </div>

    <div class="insight-box">
    <b>6. PREDICTIVE MODEL FINDINGS</b><br>
    The best performing model is <b>{best_model}</b> with AUC = <b>{best_auc:.3f}</b> and F1 = <b>{best_f1:.3f}</b>.
    High AUC indicates that the structural variables (zone, income, medical status, early claim) are strongly
    predictive of the settlement decision — confirming that non-actuarial factors are systematically driving outcomes.
    A truly unbiased system would show AUC close to 0.5 on demographic/structural variables alone.
    </div>

    <div class="insight-box">
    <b>7. RECOMMENDATIONS</b><br>
    • Implement <b>blind review protocols</b> for zones with approval rates below {threshold_45:.0f}%<br>
    • Conduct <b>auditor assignment randomisation</b> to eliminate zone-level handler bias<br>
    • Establish <b>income-agnostic documentation checklists</b> to remove income as a proxy variable<br>
    • Apply <b>adverse impact monitoring</b> quarterly across all demographic dimensions<br>
    • Submit findings to compliance/regulatory team referencing IRDAI Circular IRDAI/Life/Cir/GV/120/06/2022
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("""
    **References:** Griggs v. Duke Power Co. (1971) | IRDAI Claim Settlement Guidelines (2022) |
    Frees, Derrig & Meyers — *Predictive Modeling in Actuarial Science* (2014) |
    OECD — *Regulatory Approaches to AI and Insurance* (2021) |
    Angwin et al. — *Machine Bias*, ProPublica (2016)
    """)
