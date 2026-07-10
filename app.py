import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="ShopSense | Purchase Intention Predictor",
    page_icon="🛍️",
    layout="wide"
)

# ==============================================================================
# STYLING
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Poppins', sans-serif !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #134E5E 100%);
    padding: 2.2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 {
    color: #ffffff;
    font-size: 2rem;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #B8C6CC;
    font-size: 1rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.35);
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.7rem;
}

/* Buttons */
.stButton>button, .stFormSubmitButton>button {
    background: linear-gradient(90deg, #10B981, #059669);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    width: 100%;
    transition: 0.2s;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    background: linear-gradient(90deg, #059669, #047857);
    box-shadow: 0 4px 14px rgba(16,185,129,0.35);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛍️ ShopSense — Purchase Intention Predictor</h1>
    <p>Predict whether an online shopper will complete a purchase this session, based on their real-time browsing behaviour.</p>
    <span class="badge">🏆 Powered by Random Forest — 89.6% test accuracy</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD MODEL + SCALER
# ==============================================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `model.pkl` and/or `scaler.pkl`. Make sure both files "
        "are in the same folder as this app."
    )
    st.stop()

DATA_PATH = "online_shoppers_intention.csv"

@st.cache_data
def load_raw_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

raw_df = load_raw_data()

# ==============================================================================
# CONSTANTS
# ==============================================================================
# LabelEncoder sorts unique values alphabetically -> these mappings match the
# encoders fitted in the training notebook.
MONTH_MAP = {
    "Aug": 0, "Dec": 1, "Feb": 2, "Jul": 3, "June": 4,
    "Mar": 5, "May": 6, "Nov": 7, "Oct": 8, "Sep": 9
}
VISITOR_MAP = {"New_Visitor": 0, "Other": 1, "Returning_Visitor": 2}

FEATURE_ORDER = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend"
]

# Results from the notebook's model comparison (train_test_split random_state=42)
MODEL_RESULTS = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Logistic Regression", "KNN", "Decision Tree"],
    "Accuracy": [0.896188, 0.880373, 0.869019, 0.865369, 0.857259]
})

OS_OPTIONS = {f"OS {i}": i for i in range(1, 9)}
BROWSER_OPTIONS = {f"Browser {i}": i for i in range(1, 14)}
REGION_OPTIONS = {f"Region {i}": i for i in range(1, 10)}
TRAFFIC_OPTIONS = {f"Traffic Type {i}": i for i in range(1, 21)}

tab_predict, tab_compare, tab_insights = st.tabs(
    ["🔮 Predict", "🏆 Model Comparison", "📊 Data Insights"]
)

# ==============================================================================
# TAB 1: PREDICTION
# ==============================================================================
with tab_predict:
    left, right = st.columns([1.3, 1])

    with left:
        with st.form("prediction_form"):
            st.markdown("##### 📄 Pages Viewed & Time Spent")
            c1, c2, c3 = st.columns(3)
            with c1:
                administrative = st.number_input("Administrative pages", min_value=0, value=0, step=1)
                administrative_duration = st.number_input("↳ time spent (sec)", min_value=0.0, value=0.0, key="ad")
            with c2:
                informational = st.number_input("Informational pages", min_value=0, value=0, step=1)
                informational_duration = st.number_input("↳ time spent (sec)", min_value=0.0, value=0.0, key="in")
            with c3:
                product_related = st.number_input("Product pages", min_value=0, value=1, step=1)
                product_related_duration = st.number_input("↳ time spent (sec)", min_value=0.0, value=0.0, key="pr")

            st.markdown("##### 📈 Engagement Quality")
            c4, c5, c6 = st.columns(3)
            with c4:
                bounce_rates = st.slider("Bounce rate", 0.0, 1.0, 0.02, 0.01)
            with c5:
                exit_rates = st.slider("Exit rate", 0.0, 1.0, 0.05, 0.01)
            with c6:
                page_values = st.number_input("Page value ($)", min_value=0.0, value=0.0)

            st.markdown("##### 👤 Visitor Details")
            c7, c8, c9 = st.columns(3)
            with c7:
                month = st.selectbox("Month", list(MONTH_MAP.keys()))
                weekend = st.selectbox("Weekend session?", ["No", "Yes"])
            with c8:
                visitor_type = st.selectbox("Visitor type", list(VISITOR_MAP.keys()))
                operating_systems = st.selectbox("Operating system", list(OS_OPTIONS.keys()))
            with c9:
                browser = st.selectbox("Browser", list(BROWSER_OPTIONS.keys()))
                region = st.selectbox("Region", list(REGION_OPTIONS.keys()))

            traffic_type = st.selectbox("Traffic source", list(TRAFFIC_OPTIONS.keys()))

            submitted = st.form_submit_button("🔍 Predict Purchase Intention")

    with right:
        if not submitted:
            st.markdown("##### ℹ️ How it works")
            st.write(
                "Fill in a visitor's session behaviour on the left — how many pages "
                "they viewed, how long they stayed, and how engaged they were — and "
                "the model will estimate their likelihood of completing a purchase."
            )
            st.markdown(
                "- 🛒 **High pages + long duration + repeat product views** → likely to buy\n"
                "- 🚪 **Few pages + short duration + high bounce/exit rate** → unlikely to buy"
            )
        else:
            input_dict = {
                "Administrative": administrative,
                "Administrative_Duration": administrative_duration,
                "Informational": informational,
                "Informational_Duration": informational_duration,
                "ProductRelated": product_related,
                "ProductRelated_Duration": product_related_duration,
                "BounceRates": bounce_rates,
                "ExitRates": exit_rates,
                "PageValues": page_values,
                "SpecialDay": 0.0,  # not surfaced in the UI; defaulted to "not near a special day"
                "Month": MONTH_MAP[month],
                "OperatingSystems": OS_OPTIONS[operating_systems],
                "Browser": BROWSER_OPTIONS[browser],
                "Region": REGION_OPTIONS[region],
                "TrafficType": TRAFFIC_OPTIONS[traffic_type],
                "VisitorType": VISITOR_MAP[visitor_type],
                "Weekend": 1 if weekend == "Yes" else 0
            }

            input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            prob_no, prob_yes = proba[0], proba[1]

            st.markdown("##### 🎯 Result")
            if prediction == 1:
                st.success("✅ **Likely to PURCHASE**")
            else:
                st.warning("❌ **Unlikely to purchase**")

            m1, m2 = st.columns(2)
            m1.metric("Chance of Purchase", f"{prob_yes:.1%}")
            m2.metric("Chance of No Purchase", f"{prob_no:.1%}")

            prob_df = pd.DataFrame({
                "Outcome": ["Purchase", "No Purchase"],
                "Probability (%)": [prob_yes * 100, prob_no * 100]
            }).set_index("Outcome")
            st.bar_chart(prob_df)

            st.markdown("---")
            if prediction == 1:
                st.info("🎉 Already converting — no discount needed. Keep checkout frictionless.")
            else:
                st.markdown("##### 💡 Suggested Retention Actions")
                st.markdown(
                    "- 🏷️ Show a limited-time discount\n"
                    "- 🔁 Recommend related products\n"
                    "- 🎟️ Send a coupon code\n"
                    "- 🚚 Offer free delivery\n"
                    "- 📢 Show a personalized retargeting ad"
                )

            with st.expander("See raw input passed to the model"):
                st.dataframe(input_df)

# ==============================================================================
# TAB 2: MODEL COMPARISON
# ==============================================================================
with tab_compare:
    st.markdown("##### Why Random Forest?")
    st.write(
        "Five algorithms were trained and evaluated on the same 80/20 train-test "
        "split. Random Forest gave the best accuracy and was selected for this app."
    )

    ranked = MODEL_RESULTS.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    cols = st.columns(len(ranked))
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, col in enumerate(cols):
        with col:
            is_best = i == 0
            st.metric(
                f"{medals[i]} {ranked.loc[i, 'Model']}",
                f"{ranked.loc[i, 'Accuracy']:.2%}",
                "Deployed" if is_best else None
            )

    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#10B981" if m == "Random Forest" else "#4C72B0" for m in ranked["Model"]]
    bars = ax.barh(ranked["Model"], ranked["Accuracy"], color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0.75, 0.95)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Set Accuracy")
    for bar, acc in zip(bars, ranked["Accuracy"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2%}", va="center", fontsize=9)
    st.pyplot(fig)

    st.markdown("##### Random Forest — Classification Report (test set)")
    report_df = pd.DataFrame({
        "Class": ["No Purchase (0)", "Purchase (1)"],
        "Precision": [0.91, 0.76],
        "Recall": [0.97, 0.55],
        "F1-score": [0.94, 0.64],
        "Support": [2055, 411]
    })
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.caption(
        "Overall accuracy: 90%. The model is very reliable at spotting non-buyers "
        "(97% recall) and reasonably good at spotting buyers (55% recall) — a "
        "typical trade-off on imbalanced purchase data like this."
    )

# ==============================================================================
# TAB 3: DATA INSIGHTS / EDA
# ==============================================================================
with tab_insights:
    st.markdown("##### Feature Importance")
    st.write("Which browsing-behaviour signals matter most to the Random Forest model.")

    try:
        importance_df = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance - Random Forest")
        st.pyplot(fig)
    except AttributeError:
        st.info("This model type doesn't expose `feature_importances_`.")

    st.divider()

    if raw_df is None:
        st.warning(
            f"To see the full EDA dashboard (purchase distribution, visitor type, "
            f"monthly trend, correlation heatmap, etc.), add your dataset as "
            f"`{DATA_PATH}` in the same folder as `app.py`, then redeploy."
        )
    else:
        df = raw_df.copy()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Purchase Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            df["Revenue"].value_counts().plot(kind="bar", ax=ax, color="#55A868")
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        with col_b:
            st.markdown("##### Visitor Type Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            df["VisitorType"].value_counts().plot(kind="bar", ax=ax, color=["green", "orange", "blue"])
            ax.set_xlabel("Visitor Type")
            ax.set_ylabel("Number of Visitors")
            st.pyplot(fig)

        st.markdown("##### Purchases by Month")
        fig, ax = plt.subplots(figsize=(9, 4))
        df.groupby("Month")["Revenue"].sum().plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Purchases")
        st.pyplot(fig)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("##### Weekend vs Purchase")
            fig, ax = plt.subplots(figsize=(5, 4))
            pd.crosstab(df["Weekend"], df["Revenue"]).plot(kind="bar", ax=ax)
            ax.set_xlabel("Weekend")
            ax.set_ylabel("Number of Customers")
            st.pyplot(fig)
        with col_d:
            st.markdown("##### Bounce Rate Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df["BounceRates"], bins=30, color="#8172B2")
            ax.set_xlabel("Bounce Rate")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.markdown("##### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
