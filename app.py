import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VC Funding Predictor",
    page_icon="💰",
    layout="centered"
)

# ── Load model & column list ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

try:
    model, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💰 VC Funding Amount Predictor")
st.markdown(
    "Estimate the **total funding a startup is likely to receive** "
    "based on its funding structure and history."
)
st.divider()

if not model_loaded:
    st.error(
        "⚠️ Model files (`model.pkl` / `columns.pkl`) not found. "
        "Please run the training notebook first."
    )
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Startup Details")

col1, col2 = st.columns(2)

with col1:
    funding_rounds   = st.number_input("Number of Funding Rounds",    min_value=0, max_value=20,  value=3, step=1)
    seed             = st.number_input("Seed Funding (USD)",           min_value=0, value=500_000, step=50_000, format="%d")
    venture          = st.number_input("Venture Funding (USD)",        min_value=0, value=2_000_000, step=100_000, format="%d")
    angel            = st.number_input("Angel Investment (USD)",       min_value=0, value=100_000, step=10_000, format="%d")
    debt_financing   = st.number_input("Debt Financing (USD)",         min_value=0, value=0, step=10_000, format="%d")
    private_equity   = st.number_input("Private Equity (USD)",         min_value=0, value=0, step=100_000, format="%d")
    funding_gap_days = st.number_input("Days Between First & Last Round", min_value=0, value=365, step=30)

with col2:
    round_A = st.number_input("Series A (USD)", min_value=0, value=1_000_000, step=100_000, format="%d")
    round_B = st.number_input("Series B (USD)", min_value=0, value=0, step=100_000, format="%d")
    round_C = st.number_input("Series C (USD)", min_value=0, value=0, step=100_000, format="%d")
    round_D = st.number_input("Series D (USD)", min_value=0, value=0, step=100_000, format="%d")
    round_E = st.number_input("Series E (USD)", min_value=0, value=0, step=100_000, format="%d")
    round_F = st.number_input("Series F (USD)", min_value=0, value=0, step=100_000, format="%d")
    round_G = st.number_input("Series G (USD)", min_value=0, value=0, step=100_000, format="%d")
    round_H = st.number_input("Series H (USD)", min_value=0, value=0, step=100_000, format="%d")

founded_year = st.slider("Year Founded", min_value=1990, max_value=2025, value=2015)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Funding Amount", use_container_width=True, type="primary"):

    raw = {
        'funding_rounds':   funding_rounds,
        'seed':             seed,
        'venture':          venture,
        'angel':            angel,
        'debt_financing':   debt_financing,
        'private_equity':   private_equity,
        'round_A':          round_A,
        'round_B':          round_B,
        'round_C':          round_C,
        'round_D':          round_D,
        'round_E':          round_E,
        'round_F':          round_F,
        'round_G':          round_G,
        'round_H':          round_H,
        'funding_gap_days': funding_gap_days,
        'founded_year':     founded_year,
    }

    # Apply the same log1p transform on skewed feature columns used during training
    skew_features = ['seed','venture','angel','debt_financing','private_equity',
                     'round_A','round_B','round_C','round_D','round_E',
                     'round_F','round_G','round_H']
    for col in skew_features:
        if raw[col] >= 0:
            raw[col] = np.log1p(raw[col])

    # Build dataframe aligned to training columns
    input_df = pd.DataFrame([raw])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols].fillna(0)

    # Predict (model returns log_funding → expm1 → actual USD)
    log_pred    = model.predict(input_df)[0]
    actual_pred = np.expm1(log_pred)

    # ── Display result ────────────────────────────────────────────────────────
    st.success("✅ Prediction Complete!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Funding",  f"${actual_pred:,.0f}")
    c2.metric("Log-scale value",    f"{log_pred:.3f}")

    # Funding tier
    if actual_pred < 500_000:
        tier, color = "🌱 Seed Stage", "#4CAF50"
    elif actual_pred < 5_000_000:
        tier, color = "🚀 Early Stage (Series A)", "#2196F3"
    elif actual_pred < 50_000_000:
        tier, color = "📈 Growth Stage (Series B/C)", "#FF9800"
    else:
        tier, color = "🦄 Late Stage / Unicorn Territory", "#9C27B0"

    c3.metric("Funding Tier", tier)

    st.info(
        f"**Interpretation:** Based on the inputs provided, this startup is likely to raise "
        f"approximately **${actual_pred:,.0f}** in total funding. "
        f"This places it in the **{tier}** category."
    )

    # Feature importance bar chart
    st.subheader("🔍 Feature Importance (Random Forest)")
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=False).head(10)
    st.bar_chart(top10)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Random Forest Regressor · Target: log1p(funding_total_usd) · "
    "Trained on Crunchbase VC dataset"
)
