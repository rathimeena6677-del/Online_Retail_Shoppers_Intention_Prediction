import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==============================================================================
# STEP 1: BASIC PAGE SETTINGS
# ==============================================================================
# This just sets the browser tab title, the little icon, and makes the page wide.
st.set_page_config(
    page_title="ShopSense | Purchase Intention Predictor",
    page_icon="🛍️",
    layout="wide"
)

# A simple title and description at the top of the page.
st.title("🛍️ ShopSense — Purchase Intention Predictor")
st.write(
    "Predict whether an online shopper will complete a purchase this session, "
    "based on their browsing behaviour."
)
st.success("🏆 Powered by Random Forest — 89.6% test accuracy")


# ==============================================================================
# STEP 2: LOAD THE TRAINED MODEL AND SCALER FROM DISK
# ==============================================================================
# model.pkl = the trained Random Forest model
# scaler.pkl = the tool that scales/normalizes numbers the same way as training
#
# @st.cache_resource means: only load these files ONCE, not every time
# the page reruns (Streamlit reruns the whole script a LOT).

@st.cache_resource
def load_model_and_scaler():
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        loaded_scaler = pickle.load(f)
    return loaded_model, loaded_scaler

# Try to load them. If the files are missing, stop the app with a clear message.
try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error(
        "Could not find model.pkl and/or scaler.pkl. "
        "Make sure both files are in the same folder as this app."
    )
    st.stop()  # stops the script here so nothing below runs


# ==============================================================================
# STEP 3: SOME FIXED VALUES WE WILL REUSE (LOOKUP TABLES)
# ==============================================================================
# The model was trained on NUMBERS, not text. So when the training notebook
# encoded "Dec" as a number, it always came out as a specific number.
# These dictionaries let us turn the text the user picks back into that number.

MONTH_TO_NUMBER = {
    "Aug": 0, "Dec": 1, "Feb": 2, "Jul": 3, "June": 4,
    "Mar": 5, "May": 6, "Nov": 7, "Oct": 8, "Sep": 9
}

VISITOR_TYPE_TO_NUMBER = {
    "New_Visitor": 0,
    "Other": 1,
    "Returning_Visitor": 2
}

# The model expects the columns in this EXACT order.
FEATURE_ORDER = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend"
]

# These are just simple dropdown choices, numbered 1 to N.
OPERATING_SYSTEM_OPTIONS = ["OS 1", "OS 2", "OS 3", "OS 4", "OS 5", "OS 6", "OS 7", "OS 8"]
BROWSER_OPTIONS = [f"Browser {i}" for i in range(1, 14)]      # Browser 1 ... Browser 13
REGION_OPTIONS = [f"Region {i}" for i in range(1, 10)]        # Region 1 ... Region 9
TRAFFIC_OPTIONS = [f"Traffic Type {i}" for i in range(1, 21)] # Traffic Type 1 ... 20

# Results from testing 5 different models in the training notebook.
MODEL_RESULTS = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Logistic Regression", "KNN", "Decision Tree"],
    "Accuracy": [0.896188, 0.880373, 0.869019, 0.865369, 0.857259]
})

DATA_PATH = "online_shoppers_intention.csv"


# ==============================================================================
# STEP 4: THE THREE TABS
# ==============================================================================
tab_predict, tab_compare, tab_insights = st.tabs(
    ["🔮 Predict", "🏆 Model Comparison", "📊 Data Insights"]
)


# ------------------------------------------------------------------------------
# TAB 1: PREDICT
# ------------------------------------------------------------------------------
with tab_predict:

    # Let the user choose how they want to give us data:
    # one visitor by hand, or many visitors from a CSV file.
    input_mode = st.radio(
        "Choose input method",
        ["Manual Entry", "Upload CSV (batch predict)"],
        horizontal=True
    )

    # ==========================================================================
    # MODE A: MANUAL ENTRY — predict for ONE visitor
    # ==========================================================================
    if input_mode == "Manual Entry":

        # Split the page into two side-by-side sections:
        # - left  = all the input fields (a bit wider, ratio 1.3)
        # - right = the result, once the user clicks Predict (ratio 1)
        # This way the result is visible right away, no scrolling needed.
        left, right = st.columns([1.3, 1])

        with left:
            st.subheader("Enter visitor details")
            st.caption("The default numbers below are a real purchasing session from the dataset.")

            # --- Pages viewed and time spent ---
            st.markdown("**Pages Viewed & Time Spent**")
            col1, col2, col3 = st.columns(3)
            with col1:
                administrative = st.number_input("Administrative pages", min_value=0, value=3)
                administrative_duration = st.number_input("Admin time spent (sec)", min_value=0.0, value=67.5)
            with col2:
                informational = st.number_input("Informational pages", min_value=0, value=0)
                informational_duration = st.number_input("Info time spent (sec)", min_value=0.0, value=0.0)
            with col3:
                product_related = st.number_input("Product pages", min_value=0, value=30)
                product_related_duration = st.number_input("Product time spent (sec)", min_value=0.0, value=761.75)

            # --- Engagement quality ---
            st.markdown("**Engagement Quality**")
            col4, col5, col6 = st.columns(3)
            with col4:
                bounce_rates = st.slider("Bounce rate", 0.0, 1.0, 0.0, 0.01)
            with col5:
                exit_rates = st.slider("Exit rate", 0.0, 1.0, 0.015, 0.01)
            with col6:
                page_values = st.number_input("Page value ($)", min_value=0.0, value=19.31)

            # --- Visitor details ---
            st.markdown("**Visitor Details**")
            col7, col8, col9 = st.columns(3)
            with col7:
                month = st.selectbox("Month", list(MONTH_TO_NUMBER.keys()), index=1)  # default "Dec"
                weekend = st.selectbox("Weekend session?", ["No", "Yes"], index=1)
            with col8:
                visitor_type = st.selectbox("Visitor type", list(VISITOR_TYPE_TO_NUMBER.keys()), index=2)
                operating_system = st.selectbox("Operating system", OPERATING_SYSTEM_OPTIONS, index=1)
            with col9:
                browser = st.selectbox("Browser", BROWSER_OPTIONS, index=3)
                region = st.selectbox("Region", REGION_OPTIONS, index=0)

            traffic_type = st.selectbox("Traffic source", TRAFFIC_OPTIONS, index=0)

            # A normal button, not a form. Simpler to follow for a beginner.
            predict_clicked = st.button("🔍 Predict Purchase Intention")

        # This "with right:" block is at the SAME indentation level as "with left:"
        # above (both are directly inside "if input_mode == 'Manual Entry':").
        # Anything placed inside it shows up in the right-hand column.
        with right:
            if not predict_clicked:
                # Nothing predicted yet — show a friendly hint instead of empty space.
                st.subheader("Result")
                st.info(
                    "Fill in the visitor details on the left, then click "
                    "**Predict Purchase Intention** to see the result here."
                )
            else:

                # Step A: turn the text choices into the numbers the model expects.
                month_number = MONTH_TO_NUMBER[month]
                visitor_type_number = VISITOR_TYPE_TO_NUMBER[visitor_type]
                weekend_number = 1 if weekend == "Yes" else 0

                # These dropdowns are just "OS 2" -> 2, "Browser 4" -> 4, etc.
                # We can get the number by taking the last part of the text.
                operating_system_number = int(operating_system.split(" ")[-1])
                browser_number = int(browser.split(" ")[-1])
                region_number = int(region.split(" ")[-1])
                traffic_type_number = int(traffic_type.split(" ")[-1])

                # Step B: put everything into one row, in the exact column order
                # the model was trained on.
                one_row = pd.DataFrame([{
                    "Administrative": administrative,
                    "Administrative_Duration": administrative_duration,
                    "Informational": informational,
                    "Informational_Duration": informational_duration,
                    "ProductRelated": product_related,
                    "ProductRelated_Duration": product_related_duration,
                    "BounceRates": bounce_rates,
                    "ExitRates": exit_rates,
                    "PageValues": page_values,
                    "SpecialDay": 0.0,  # not shown in the UI, assume "not a special day"
                    "Month": month_number,
                    "OperatingSystems": operating_system_number,
                    "Browser": browser_number,
                    "Region": region_number,
                    "TrafficType": traffic_type_number,
                    "VisitorType": visitor_type_number,
                    "Weekend": weekend_number
                }])[FEATURE_ORDER]

                # Step C: scale the numbers the same way the training data was scaled.
                scaled_row = scaler.transform(one_row)

                # Step D: ask the model to predict.
                prediction = model.predict(scaled_row)[0]          # 0 = no purchase, 1 = purchase
                probabilities = model.predict_proba(scaled_row)[0]  # [prob_no, prob_yes]
                probability_no = probabilities[0]
                probability_yes = probabilities[1]

                # Step E: show the result.
                st.subheader("Result")
                if prediction == 1:
                    st.success("✅ Likely to PURCHASE")
                else:
                    st.warning("❌ Unlikely to purchase")

                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Chance of Purchase", f"{probability_yes:.1%}")
                metric_col2.metric("Chance of No Purchase", f"{probability_no:.1%}")

                # A simple bar chart comparing the two probabilities.
                chart_data = pd.DataFrame({
                    "Outcome": ["Purchase", "No Purchase"],
                    "Probability (%)": [probability_yes * 100, probability_no * 100]
                }).set_index("Outcome")
                st.bar_chart(chart_data)

                # Some friendly suggestions depending on the result.
                st.markdown("---")
                if prediction == 1:
                    st.markdown("**Business Insights**")
                    st.markdown(
                        "- 🎉 This visitor is already converting — no discount needed\n"
                        "- 🧾 Keep checkout frictionless (avoid extra steps or popups)\n"
                        "- ⬆️ Try upselling or cross-selling related/premium products\n"
                        "- ⭐ Prompt for a review or ask them to join a loyalty program\n"
                        "- 📦 Highlight fast/free shipping to reinforce the decision"
                    )
                else:
                    st.markdown("**Business Insights**")
                    st.markdown(
                        "- 🏷️ Show a limited-time discount\n"
                        "- 🔁 Recommend related products\n"
                        "- 🎟️ Send a coupon code\n"
                        "- 🚚 Offer free delivery\n"
                        "- 📢 Show a personalized retargeting ad"
                    )

                # Show the exact numbers we sent into the model.
                # NOTE: this is placed INSIDE the "if predict_clicked" block
                # on purpose. "one_row" only exists after the button is clicked,
                # so this must never be shown/run before that happens.
                with st.expander("See raw input passed to the model"):
                    st.dataframe(one_row)

    # ==========================================================================
    # MODE B: UPLOAD CSV — predict for MANY visitors at once
    # ==========================================================================
    else:
        st.subheader("Upload a CSV to predict many visitors at once")
        st.caption(
            "Expected columns: Administrative, Administrative_Duration, Informational, "
            "Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, "
            "ExitRates, PageValues, SpecialDay (optional), Month, OperatingSystems, Browser, "
            "Region, TrafficType, VisitorType, Weekend"
        )

        uploaded_csv = st.file_uploader("Upload visitor sessions CSV", type=["csv"])

        if uploaded_csv is not None:
            try:
                # Read the file the user uploaded.
                original_data = pd.read_csv(uploaded_csv)

                # Make a COPY to edit, so we keep the original safe for later.
                data_to_predict = original_data.copy()

                # --- Turn "Month" text into numbers (Dec -> 1, Aug -> 0, etc.) ---
                # If it's already a number, this leaves it alone.
                def convert_month(value):
                    if pd.isna(value):
                        return np.nan
                    if isinstance(value, str):
                        return MONTH_TO_NUMBER.get(value.strip(), np.nan)
                    return value

                data_to_predict["Month"] = data_to_predict["Month"].apply(convert_month)

                # --- Turn "VisitorType" text into numbers ---
                def convert_visitor_type(value):
                    if pd.isna(value):
                        return np.nan
                    if isinstance(value, str):
                        return VISITOR_TYPE_TO_NUMBER.get(value.strip(), np.nan)
                    return value

                data_to_predict["VisitorType"] = data_to_predict["VisitorType"].apply(convert_visitor_type)

                # --- Turn "Weekend" into 0 / 1 ---
                def convert_weekend(value):
                    if pd.isna(value):
                        return np.nan
                    if isinstance(value, bool):
                        return int(value)
                    if isinstance(value, (int, float)):
                        return int(value)
                    text_value = str(value).strip().lower()
                    weekend_lookup = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
                    return weekend_lookup.get(text_value, np.nan)

                data_to_predict["Weekend"] = data_to_predict["Weekend"].apply(convert_weekend)

                # If SpecialDay wasn't included, just assume 0.0 for everyone.
                if "SpecialDay" not in data_to_predict.columns:
                    data_to_predict["SpecialDay"] = 0.0

                # Check that every column the model needs is actually present.
                missing_columns = [col for col in FEATURE_ORDER if col not in data_to_predict.columns]

                if missing_columns:
                    st.error(f"Missing required column(s): {', '.join(missing_columns)}")

                elif data_to_predict[FEATURE_ORDER].isnull().any().any():
                    st.error(
                        "Some values couldn't be understood (check Month/VisitorType/Weekend "
                        "spelling). Please review the uploaded file."
                    )

                else:
                    # Put columns in the right order, then scale + predict.
                    features_only = data_to_predict[FEATURE_ORDER]
                    scaled_features = scaler.transform(features_only)

                    predictions = model.predict(scaled_features)
                    purchase_probabilities = model.predict_proba(scaled_features)[:, 1]

                    # Build a results table: original data + new prediction columns.
                    results = original_data.copy()
                    results["Prediction"] = np.where(predictions == 1, "Purchase", "No Purchase")
                    results["Purchase Probability"] = (purchase_probabilities * 100).round(2)

                    st.success(f"✅ Predicted {len(results)} sessions.")

                    result_col1, result_col2 = st.columns(2)
                    result_col1.metric("Predicted to Purchase", int((predictions == 1).sum()))
                    result_col2.metric("Predicted No Purchase", int((predictions == 0).sum()))

                    st.dataframe(results, use_container_width=True)

                    # Let the user download the results as a CSV file.
                    csv_bytes = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=csv_bytes,
                        file_name="purchase_intention_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as error:
                st.error(f"Couldn't process that file: {error}")

        else:
            st.info("Upload a CSV above to get predictions for multiple visitors at once.")


# ------------------------------------------------------------------------------
# TAB 2: MODEL COMPARISON
# ------------------------------------------------------------------------------
with tab_compare:
    st.subheader("Why Random Forest?")
    st.write(
        "Five algorithms were trained and evaluated on the same 80/20 train-test "
        "split. Random Forest gave the best accuracy and was selected for this app."
    )

    # Sort so the best model is first.
    ranked_models = MODEL_RESULTS.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Show one metric box per model.
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    columns = st.columns(len(ranked_models))
    for i in range(len(ranked_models)):
        model_name = ranked_models.loc[i, "Model"]
        model_accuracy = ranked_models.loc[i, "Accuracy"]
        with columns[i]:
            label = "Deployed" if i == 0 else None
            st.metric(f"{medals[i]} {model_name}", f"{model_accuracy:.2%}", label)

    st.write("")  # a little spacing

    # A horizontal bar chart comparing all 5 models.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_colors = []
    for model_name in ranked_models["Model"]:
        if model_name == "Random Forest":
            bar_colors.append("#10B981")  # highlight the winner in green
        else:
            bar_colors.append("#4C72B0")

    bars = ax.barh(ranked_models["Model"], ranked_models["Accuracy"], color=bar_colors)
    ax.invert_yaxis()
    ax.set_xlim(0.75, 0.95)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Set Accuracy")

    # Write the accuracy percentage at the end of each bar.
    for bar, accuracy in zip(bars, ranked_models["Accuracy"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{accuracy:.2%}", va="center", fontsize=9)

    st.pyplot(fig)

    # A simple table showing precision/recall/f1 for the winning model.
    st.subheader("Random Forest — Classification Report (test set)")
    report_table = pd.DataFrame({
        "Class": ["No Purchase (0)", "Purchase (1)"],
        "Precision": [0.91, 0.76],
        "Recall": [0.97, 0.55],
        "F1-score": [0.94, 0.64],
        "Support": [2055, 411]
    })
    st.dataframe(report_table, use_container_width=True, hide_index=True)
    st.caption(
        "Overall accuracy: 90%. The model is very reliable at spotting non-buyers "
        "(97% recall) and reasonably good at spotting buyers (55% recall) — a "
        "typical trade-off on imbalanced purchase data like this."
    )


# ------------------------------------------------------------------------------
# TAB 3: DATA INSIGHTS
# ------------------------------------------------------------------------------
with tab_insights:

    # --- Feature importance chart ---
    st.subheader("Feature Importance")
    st.write("Which browsing-behaviour signals matter most to the Random Forest model.")

    # Not every model type has "feature_importances_", so we guard against that.
    if hasattr(model, "feature_importances_"):
        importance_table = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance_table["Feature"], importance_table["Importance"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance - Random Forest")
        st.pyplot(fig)
    else:
        st.info("This model type doesn't expose feature importances.")

    st.divider()

    # --- Load a dataset to explore (either uploaded, or the bundled default) ---
    st.subheader("Dataset")

    uploaded_data_file = st.file_uploader(
        "Upload a CSV to explore your own data (optional)",
        type=["csv"],
        help="Must have the same columns as the online_shoppers_intention dataset.",
        key="insights_uploader"  # different key from the one in Tab 1
    )

    dataset = None
    dataset_source_message = None

    if uploaded_data_file is not None:
        try:
            dataset = pd.read_csv(uploaded_data_file)
            dataset_source_message = f"📤 Using uploaded file: {uploaded_data_file.name}"
        except Exception as error:
            st.error(f"Couldn't read that file: {error}")

    # If nothing was uploaded, fall back to the bundled CSV file if it exists.
    if dataset is None and os.path.exists(DATA_PATH):
        dataset = pd.read_csv(DATA_PATH)
        dataset_source_message = f"📁 Using bundled default dataset: {DATA_PATH}"

    if dataset_source_message:
        st.caption(dataset_source_message)

    if dataset is None:
        st.warning(
            f"No dataset available. Either upload a CSV above, or add "
            f"{DATA_PATH} to the same folder as app.py and redeploy."
        )
    else:
        # --- Purchase distribution & visitor type distribution, side by side ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Purchase Distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            dataset["Revenue"].value_counts().plot(kind="bar", ax=ax, color="#55A868")
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with chart_col2:
            st.markdown("**Visitor Type Distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            dataset["VisitorType"].value_counts().plot(kind="bar", ax=ax, color=["green", "orange", "blue"])
            ax.set_xlabel("Visitor Type")
            ax.set_ylabel("Number of Visitors")
            st.pyplot(fig)

        # --- Purchases by month ---
        st.markdown("**Purchases by Month**")
        fig, ax = plt.subplots(figsize=(9, 4))
        dataset.groupby("Month")["Revenue"].sum().plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Purchases")
        st.pyplot(fig)

        # --- Weekend vs purchase, and bounce rate distribution, side by side ---
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("**Weekend vs Purchase**")
            fig, ax = plt.subplots(figsize=(5, 4))
            pd.crosstab(dataset["Weekend"], dataset["Revenue"]).plot(kind="bar", ax=ax)
            ax.set_xlabel("Weekend")
            ax.set_ylabel("Number of Customers")
            st.pyplot(fig)

        with chart_col4:
            st.markdown("**Bounce Rate Distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(dataset["BounceRates"], bins=30, color="#8172B2")
            ax.set_xlabel("Bounce Rate")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # --- Correlation heatmap ---
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
