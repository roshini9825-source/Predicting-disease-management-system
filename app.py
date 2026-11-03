import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predicting Disease Management System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Predicting Disease Management System")
st.subheader("Multi-Disease Case Prediction")
st.write(
    "Machine Learning system for analysing disease case patterns "
    "and estimating expected cases."
)


# =========================================================
# DISEASE FILES
# FIX: key was "Hepatitis " (trailing space) while the
# management page used "Hepatitis A" -> KeyError. Now consistent.
# =========================================================

FILES = {
    "Hepatitis A": "hepatitis.csv",
    "Measles": "measles.csv",
    "Mumps": "mumps.csv",
    "Pertussis": "pertussis.csv",
    "Polio": "polio.csv",
    "Rubella": "rubella.csv",
    "Smallpox": "smallpox.csv"
}

# NOTE: incidence_per_capita is computed FROM cases (cases / population),
# so using it as a feature leaks the answer into the model and inflates R².
# It is deliberately excluded from the model features.
FEATURES = [
    "year", "week_number", "month_estimate", "quarter", "state_encoded",
    "previous_cases", "previous_cases_2", "previous_cases_3", "previous_cases_4",
    "rolling_average_3", "rolling_average_5", "rolling_std_5",
    "same_week_last_year",
]


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_all_data():
    frames = []
    for disease_name, file_name in FILES.items():
        try:
            d = pd.read_csv(file_name)
            d["disease_display"] = disease_name
            frames.append(d)
        except FileNotFoundError:
            pass

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# =========================================================
# BUILD FEATURES
# =========================================================

def build_features(d):
    d = d.drop_duplicates().copy()

    for col in ["cases", "week", "incidence_per_capita"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Rows without a valid case count or week are unusable for training.
    d = d.dropna(subset=["cases", "week"])
    d["incidence_per_capita"] = d["incidence_per_capita"].fillna(
        d["incidence_per_capita"].median()
    )
    d["state"] = d["state"].fillna("Unknown").astype(str)

    d["year"] = (d["week"] // 100).astype(int)
    d["week_number"] = (d["week"] % 100).astype(int)
    d["month_estimate"] = ((d["week_number"] - 1) // 4) + 1
    d["quarter"] = ((d["week_number"] - 1) // 13) + 1

    d = d.sort_values(["state", "week"]).reset_index(drop=True)

    g = d.groupby("state")["cases"]
    for k in range(1, 5):
        name = "previous_cases" if k == 1 else f"previous_cases_{k}"
        d[name] = g.shift(k)

    d["rolling_average_3"] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    d["rolling_average_5"] = g.transform(lambda x: x.shift(1).rolling(5).mean())
    d["rolling_std_5"] = g.transform(lambda x: x.shift(1).rolling(5).std())
    d["same_week_last_year"] = g.shift(52)

    return d


# =========================================================
# TRAIN SEPARATE MODEL FOR EACH DISEASE
# =========================================================

@st.cache_resource
def train_models(data):
    models, metrics, processed_data, feature_importance = {}, {}, {}, {}

    for disease_name in FILES.keys():
        d = data[data["disease_display"] == disease_name]
        if d.empty:
            continue

        d = build_features(d)

        # Fill missing lag values with the state's own median, then global.
        lag_cols = [
            "previous_cases", "previous_cases_2", "previous_cases_3",
            "previous_cases_4", "rolling_average_3", "rolling_average_5",
            "rolling_std_5", "same_week_last_year",
        ]
        state_median = d.groupby("state")["cases"].transform("median")
        for c in lag_cols:
            d[c] = d[c].fillna(state_median).fillna(d["cases"].median())

        state_encoder = LabelEncoder()
        d["state_encoded"] = state_encoder.fit_transform(d["state"])

        # Chronological 80/20 split (a random split lets the model see
        # neighbouring weeks of the test rows through the lag features).
        d = d.sort_values("week").reset_index(drop=True)
        split = int(len(d) * 0.80)
        train, test = d.iloc[:split], d.iloc[split:]

        X_train, X_test = train[FEATURES], test[FEATURES]
        y_train, y_test = train["cases"], test["cases"]

        # Train on log1p(cases): case counts are very skewed.
        if disease_name == "Hepatitis A":
            model = ExtraTreesRegressor(
                n_estimators=150, min_samples_leaf=1,
                max_features=1.0, random_state=42, n_jobs=2,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=150, min_samples_leaf=1,
                max_features="sqrt", bootstrap=True,
                random_state=42, n_jobs=2,
            )

        model.fit(X_train, np.log1p(y_train))
        predictions = np.clip(np.expm1(model.predict(X_test)), 0, None)

        mse = mean_squared_error(y_test, predictions)
        metrics[disease_name] = {
            "r2": r2_score(y_test, predictions),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_test, predictions),
            "y_test": y_test,
            "predictions": predictions,
        }

        models[disease_name] = {
            "model": model,
            "encoder": state_encoder,
            "features": FEATURES,
        }

        feature_importance[disease_name] = (
            pd.DataFrame({
                "Feature": FEATURES,
                "Importance": model.feature_importances_,
            })
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        processed_data[disease_name] = d

    return models, metrics, processed_data, feature_importance


# =========================================================
# LOAD + TRAIN
# =========================================================

data = load_all_data()

if data.empty:
    st.error(
        "No disease CSV files were found. "
        "Upload all CSV files to the same GitHub repository as app.py."
    )
    st.stop()

models, metrics, disease_data_dict, feature_importance = train_models(data)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 Disease Management")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard", "Dataset Overview", "Disease Trends",
        "State-wise Analysis", "Prediction", "Model Performance",
        "Feature Importance", "Disease Management",
    ],
)

available = [x for x in FILES if x in disease_data_dict]
if not available:
    st.error("No disease dataset could be processed.")
    st.stop()

selected = st.sidebar.selectbox("Select Disease", available)

disease_df = disease_data_dict[selected].copy()
selected_metrics = metrics[selected]
model = models[selected]["model"]
state_encoder = models[selected]["encoder"]
features = models[selected]["features"]


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":
    st.header(f"📊 {selected} Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(disease_df):,}")
    c2.metric("Total Cases", f"{disease_df['cases'].sum():,.0f}")
    c3.metric("Average Cases", f"{disease_df['cases'].mean():.2f}")
    c4.metric("R² Score", f"{selected_metrics['r2']:.4f}")

    st.markdown("---")

    r2 = selected_metrics["r2"]
    if r2 >= 0.80:
        st.success(f"Excellent model performance! R² Score = {r2:.4f}")
    elif r2 >= 0.60:
        st.info(f"Good model performance. R² Score = {r2:.4f}")
    else:
        st.warning(f"Model performance can be improved. Current R² Score = {r2:.4f}")


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":
    st.header(f"📋 {selected} Dataset Overview")
    st.write(f"Selected disease: **{selected}**")
    st.write("Dataset shape:", disease_df.shape)
    st.dataframe(disease_df.head(100), use_container_width=True)

    st.subheader("Missing Values")
    st.dataframe(
        disease_df.isnull().sum().rename("Missing Values"),
        use_container_width=True,
    )

    st.subheader("Statistical Summary")
    st.dataframe(disease_df.describe(), use_container_width=True)


# =========================================================
# DISEASE TRENDS
# =========================================================

elif page == "Disease Trends":
    st.header(f"📈 {selected} Disease Trends")

    yearly = disease_df.groupby("year")["cases"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.set_title(f"{selected} Yearly Cases Trend")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")
    cols = [
        "cases", "incidence_per_capita", "previous_cases", "previous_cases_2",
        "previous_cases_3", "rolling_average_3", "year", "week_number",
    ]
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(disease_df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title(f"{selected} Correlation Heatmap")
    st.pyplot(fig2)
    plt.close(fig2)


# =========================================================
# STATE-WISE ANALYSIS
# =========================================================

elif page == "State-wise Analysis":
    st.header(f"🗺️ {selected} State-wise Analysis")

    state_cases = (
        disease_df.groupby("state")["cases"].sum().sort_values(ascending=False)
    )

    st.dataframe(state_cases.rename("Total Cases").to_frame(), use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    state_cases.head(15).plot(kind="bar", ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Total Cases")
    ax.set_title(f"Top States - {selected}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":
    st.header("🔮 Disease Case Prediction")
    st.write(f"Prediction for **{selected}**")

    states = sorted(disease_df["state"].astype(str).unique())
    state = st.selectbox("State", states)

    week = st.number_input(
        "Week (YYYYWW)",
        min_value=190001, max_value=210053,
        value=int(disease_df["week"].median()),
    )

    def med(col):
        return float(disease_df[col].median())

    previous = st.number_input("Previous Cases", min_value=0.0, value=med("previous_cases"))
    previous2 = st.number_input("Previous 2 Cases", min_value=0.0, value=med("previous_cases_2"))
    previous3 = st.number_input("Previous 3 Cases", min_value=0.0, value=med("previous_cases_3"))
    previous4 = st.number_input("Previous 4 Cases", min_value=0.0, value=med("previous_cases_4"))
    rolling3 = st.number_input("3-Week Rolling Average", min_value=0.0, value=med("rolling_average_3"))
    rolling5 = st.number_input("5-Week Rolling Average", min_value=0.0, value=med("rolling_average_5"))
    rolling_std = st.number_input("5-Week Rolling Std Dev", min_value=0.0, value=med("rolling_std_5"))
    last_year = st.number_input("Cases Same Week Last Year", min_value=0.0, value=med("same_week_last_year"))

    if st.button("🔮 Predict Cases"):
        week = int(week)
        week_number = week % 100

        if week_number < 1 or week_number > 53:
            st.error("Week must be in YYYYWW format with week number 01-53.")
            st.stop()

        row = pd.DataFrame([{
            "year": week // 100,
            "week_number": week_number,
            "month_estimate": ((week_number - 1) // 4) + 1,
            "quarter": ((week_number - 1) // 13) + 1,
            "state_encoded": state_encoder.transform([state])[0],
            "previous_cases": previous,
            "previous_cases_2": previous2,
            "previous_cases_3": previous3,
            "previous_cases_4": previous4,
            "rolling_average_3": rolling3,
            "rolling_average_5": rolling5,
            "rolling_std_5": rolling_std,
            "same_week_last_year": last_year,
        }])

        prediction = max(0.0, float(np.expm1(model.predict(row[features])[0])))
        st.success(f"Estimated expected cases: **{prediction:.2f}**")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":
    st.header(f"🤖 {selected} Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{selected_metrics['r2']:.4f}")
    c2.metric("MSE", f"{selected_metrics['mse']:.4f}")
    c3.metric("RMSE", f"{selected_metrics['rmse']:.4f}")
    c4.metric("MAE", f"{selected_metrics['mae']:.4f}")

    st.markdown("---")
    st.subheader("All Disease Model Performance")

    performance_df = pd.DataFrame([
        {
            "Disease": name,
            "R² Score": m["r2"],
            "MSE": m["mse"],
            "RMSE": m["rmse"],
            "MAE": m["mae"],
        }
        for name, m in metrics.items()
    ])

    st.dataframe(
        performance_df.style.format({
            "R² Score": "{:.4f}", "MSE": "{:.2f}",
            "RMSE": "{:.2f}", "MAE": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.subheader("Actual vs Predicted Cases")

    y_test = selected_metrics["y_test"]
    y_pred = selected_metrics["predictions"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    lim = [0, max(float(np.max(y_test)), float(np.max(y_pred)))]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Cases")
    ax.set_ylabel("Predicted Cases")
    ax.set_title(f"{selected} - Actual vs Predicted")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":
    st.header(f"⭐ {selected} Feature Importance")

    imp = feature_importance[selected]
    st.dataframe(imp, use_container_width=True)

    model_name = "Extra Trees" if selected == "Hepatitis A" else "Random Forest"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name} Feature Importance - {selected}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# DISEASE MANAGEMENT
# =========================================================

elif page == "Disease Management":
    st.header(f"🛡️ {selected} Management & Preventive Actions")

    st.info(
        "These are general preventive/management suggestions, "
        "not a medical diagnosis."
    )

    actions = {
        "Hepatitis A": [
            "Maintain hand hygiene",
            "Use safe drinking water",
            "Follow food hygiene practices",
            "Seek medical advice when symptoms occur",
        ],
        "Measles": [
            "Maintain vaccination awareness",
            "Avoid close contact during suspected infection",
            "Monitor outbreaks",
            "Seek medical advice when symptoms occur",
        ],
        "Mumps": [
            "Maintain vaccination awareness",
            "Practise good respiratory hygiene",
            "Avoid close contact when ill",
            "Seek medical advice when symptoms occur",
        ],
        "Pertussis": [
            "Maintain vaccination awareness",
            "Practise respiratory hygiene",
            "Avoid close contact when ill",
            "Seek medical advice when symptoms occur",
        ],
        "Polio": [
            "Maintain vaccination awareness",
            "Use safe water and sanitation",
            "Practise good hygiene",
            "Seek medical advice when symptoms occur",
        ],
        "Rubella": [
            "Maintain vaccination awareness",
            "Follow public-health guidance",
            "Avoid exposure during suspected infection",
            "Seek medical advice when symptoms occur",
        ],
        "Smallpox": [
            "Follow official public-health guidance",
            "Report suspected cases to health authorities",
            "Avoid close contact with suspected cases",
            "Seek urgent professional medical advice",
        ],
    }

    for i, action in enumerate(actions.get(selected, []), 1):
        st.write(f"**{i}.** {action}")


# =========================================================
# SIDEBAR FOOTER
# =========================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "Random Forest / Extra Trees Regression • Separate Models for Each Disease"
)
