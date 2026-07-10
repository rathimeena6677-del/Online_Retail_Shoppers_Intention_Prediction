import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Online Shopper Purchase Intention Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Online Shopper Purchase Intention Predictor")
st.write(
    "Predict whether a visitor will **make a purchase (Revenue = Yes/No)** "
    "based on their browsing behaviour, using a Random Forest model trained "
    "on the *Online Shoppers Purchasing Intention* dataset."
)

# ----------------------------------------------------------------------------
# Load model + scaler
# ----------------------------------------------------------------------------
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
        "are in the same folder as this app (see `train_model.py` to generate them)."
    )
    st.stop()

# Optional: raw dataset for EDA graphs (only needed for the "Data Insights" tab)
DATA_PATH = "online_shoppers_intention.csv"

@st.cache_data
def load_raw_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

raw_df = load_raw_data()

# ----------------------------------------------------------------------------
# Encodings used during training (must match the notebook's LabelEncoder output)
# LabelEncoder sorts the unique values alphabetically, so:
# Month order:   Aug, Dec, Feb, Jul, June, Mar, May, Nov, Oct, Sep
# VisitorType:   New_Visitor, Other, Returning_Visitor
# ----------------------------------------------------------------------------
MONTH_MAP = {
    "Aug": 0, "Dec": 1, "Feb": 2, "Jul": 3, "June": 4,
    "Mar": 5, "May": 6, "Nov": 7, "Oct": 8, "Sep": 9
}
VISITOR_MAP = {
    "New_Visitor": 0, "Other": 1, "Returning_Visitor": 2
}

FEATURE_ORDER = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend"
]

tab_predict, tab_insights = st.tabs(["🔮 Predict", "📊 Data Insights"])

# ==============================================================================
# TAB 1: PREDICTION
# ==============================================================================
with tab_predict:
    with st.form("prediction_form"):
        st.subheader("Page visit behaviour")
        col1, col2 = st.columns(2)
        with col1:
            administrative = st.number_input("Administrative pages visited", min_value=0, value=0, step=1)
            informational = st.number_input("Informational pages visited", min_value=0, value=0, step=1)
            product_related = st.number_input("Product-related pages visited", min_value=0, value=1, step=1)
        with col2:
            administrative_duration = st.number_input("Administrative duration (seconds)", min_value=0.0, value=0.0)
            informational_duration = st.number_input("Informational duration (seconds)", min_value=0.0, value=0.0)
            product_related_duration = st.number_input("Product-related duration (seconds)", min_value=0.0, value=0.0)

        st.subheader("Site engagement metrics")
        col3, col4, col5 = st.columns(3)
        with col3:
            bounce_rates = st.number_input("Bounce rate", min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
        with col4:
            exit_rates = st.number_input("Exit rate", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
        with col5:
            page_values = st.number_input("Page value", min_value=0.0, value=0.0)

        special_day = st.slider("Closeness to a special day (0 = far, 1 = very close)", 0.0, 1.0, 0.0, 0.1)

        st.subheader("Visitor & session info")
        col6, col7 = st.columns(2)
        with col6:
            month = st.selectbox("Month", list(MONTH_MAP.keys()))
            visitor_type = st.selectbox("Visitor type", list(VISITOR_MAP.keys()))
            weekend = st.selectbox("Weekend session?", ["No", "Yes"])
        with col7:
            operating_systems = st.number_input("Operating system (code)", min_value=1, value=1, step=1)
            browser = st.number_input("Browser (code)", min_value=1, value=1, step=1)
            region = st.number_input("Region (code)", min_value=1, value=1, step=1)

        traffic_type = st.number_input("Traffic type (code)", min_value=1, value=1, step=1)

        submitted = st.form_submit_button("Predict Purchase Intention")

    if submitted:
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
            "SpecialDay": special_day,
            "Month": MONTH_MAP[month],
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": VISITOR_MAP[visitor_type],
            "Weekend": 1 if weekend == "Yes" else 0
        }

        input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        prob_no, prob_yes = proba[0], proba[1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Likely to **PURCHASE** (Revenue = Yes)")
        else:
            st.warning("❌ Likely **NOT** to purchase (Revenue = No)")

        # ---- Probability breakdown (chart) ----
        prob_df = pd.DataFrame({
            "Outcome": ["Will Purchase (Yes)", "Will Not Purchase (No)"],
            "Probability (%)": [prob_yes * 100, prob_no * 100]
        })
        st.write("**Prediction Confidence**")
        st.bar_chart(prob_df.set_index("Outcome"))

        col_a, col_b = st.columns(2)
        col_a.metric("Chance of Purchase (Yes)", f"{prob_yes:.2%}")
        col_b.metric("Chance of No Purchase (No)", f"{prob_no:.2%}")

        st.divider()

        # ---- Business action based on prediction ----
        if prediction == 1:
            st.info(
                "🎉 This visitor is already likely to buy — **no discount or offer needed**. "
                "Focus on a smooth, fast checkout experience instead."
            )
        else:
            st.subheader("💡 Suggested Retention Actions")
            st.write(
                "This visitor is unlikely to purchase. Consider triggering one or more "
                "of the following to improve conversion:"
            )
            st.markdown(
                """
- 🏷️ **Show a limited-time discount**
- 🔁 **Recommend related/popular products**
- 🎟️ **Send a coupon code via email/pop-up**
- 🚚 **Offer free delivery**
- 📢 **Display personalized retargeting ads**
                """
            )

        with st.expander("See raw input passed to the model"):
            st.dataframe(input_df)

# ==============================================================================
# TAB 2: DATA INSIGHTS / EDA (mirrors the notebook's exploratory graphs)
# ==============================================================================
with tab_insights:
    st.subheader("Model: Feature Importance")
    st.write("Which browsing-behaviour features matter most to the Random Forest model.")

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

        st.subheader("Purchase Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["Revenue"].value_counts().plot(kind="bar", ax=ax, color="#55A868")
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Visitor Type Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["VisitorType"].value_counts().plot(kind="bar", ax=ax, color=["green", "orange", "blue"])
        ax.set_xlabel("Visitor Type")
        ax.set_ylabel("Number of Visitors")
        st.pyplot(fig)

        st.subheader("Purchases by Month")
        fig, ax = plt.subplots(figsize=(9, 4))
        df.groupby("Month")["Revenue"].sum().plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Purchases")
        st.pyplot(fig)

        st.subheader("Weekend vs Purchase")
        fig, ax = plt.subplots(figsize=(5, 4))
        pd.crosstab(df["Weekend"], df["Revenue"]).plot(kind="bar", ax=ax)
        ax.set_xlabel("Weekend")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("Bounce Rate Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df["BounceRates"], bins=30, color="#8172B2")
            ax.set_xlabel("Bounce Rate")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        with col_h2:
            st.subheader("Exit Rate Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df["ExitRates"], bins=30, color="#CCB974")
            ax.set_xlabel("Exit Rate")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
