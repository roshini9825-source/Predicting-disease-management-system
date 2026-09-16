import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
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
# =========================================================

FILES = {
    "Hepatitis ": "hepatitis.csv",
    "Measles": "measles.csv",
    "Mumps": "mumps.csv",
    "Pertussis": "pertussis.csv",
    "Polio": "polio.csv",
    "Rubella": "rubella.csv",
    "Smallpox": "smallpox.csv"
}


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
# TRAIN SEPARATE MODEL FOR EACH DISEASE
# =========================================================

@st.cache_resource
def train_models(data):

    models = {}
    metrics = {}
    processed_data = {}
    feature_importance = {}

    for disease_name in FILES.keys():

        d = data[
            data["disease_display"] == disease_name
        ].copy()

        if d.empty:
            continue

        # -------------------------------------------------
        # DATA CLEANING
        # -------------------------------------------------

        d = d.drop_duplicates()

        d["cases"] = pd.to_numeric(
            d["cases"],
            errors="coerce"
        )

        d["week"] = pd.to_numeric(
            d["week"],
            errors="coerce"
        )

        d["incidence_per_capita"] = pd.to_numeric(
            d["incidence_per_capita"],
            errors="coerce"
        )

        d["cases"] = d["cases"].fillna(
            d["cases"].median()
        )

        d["week"] = d["week"].fillna(
            d["week"].median()
        )

        d["incidence_per_capita"] = d[
            "incidence_per_capita"
        ].fillna(
            d["incidence_per_capita"].median()
        )

        d["state"] = (
            d["state"]
            .fillna("Unknown")
            .astype(str)
        )

        # -------------------------------------------------
        # TIME FEATURES
        # -------------------------------------------------

        d["year"] = (
            d["week"] // 100
        ).astype(int)

        d["week_number"] = (
            d["week"] % 100
        ).astype(int)

        d["month_estimate"] = (
            ((d["week_number"] - 1) // 4) + 1
        )

        d["quarter"] = (
            ((d["week_number"] - 1) // 13) + 1
        )

        # -------------------------------------------------
        # SORT DATA
        # -------------------------------------------------

        d = d.sort_values(
            ["state", "week"]
        ).reset_index(drop=True)

        # -------------------------------------------------
        # PREVIOUS CASE FEATURES
        # -------------------------------------------------

        d["previous_cases"] = (
            d.groupby("state")["cases"]
            .shift(1)
        )

        d["previous_cases_2"] = (
            d.groupby("state")["cases"]
            .shift(2)
        )

        d["previous_cases_3"] = (
            d.groupby("state")["cases"]
            .shift(3)
        )

        d["previous_cases_4"] = (
            d.groupby("state")["cases"]
            .shift(4)
        )

        d["rolling_average_3"] = (
            d.groupby("state")["cases"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(3)
                .mean()
            )
        )

        d["rolling_average_5"] = (
            d.groupby("state")["cases"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(5)
                .mean()
            )
        )

        # -------------------------------------------------
        # FILL FEATURE MISSING VALUES
        # -------------------------------------------------

        median_cases = d["cases"].median()

        lag_columns = [
            "previous_cases",
            "previous_cases_2",
            "previous_cases_3",
            "previous_cases_4",
            "rolling_average_3",
            "rolling_average_5"
        ]

        for column in lag_columns:

            d[column] = d[column].fillna(
                median_cases
            )

        # -------------------------------------------------
        # ENCODE STATE
        # -------------------------------------------------

        state_encoder = LabelEncoder()

        d["state_encoded"] = (
            state_encoder.fit_transform(
                d["state"]
            )
        )

        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------

        features = [

            "year",

            "week_number",

            "month_estimate",

            "quarter",

            "state_encoded",

            "incidence_per_capita",

            "previous_cases",

            "previous_cases_2",

            "previous_cases_3",

            "previous_cases_4",

            "rolling_average_3",

            "rolling_average_5"

        ]

        X = d[features]

        y = d["cases"]

        # -------------------------------------------------
        # RANDOM 80/20 SPLIT
        # -------------------------------------------------

        np.random.seed(42)

        indices = np.arange(len(d))

        np.random.shuffle(indices)

        split = int(
            len(indices) * 0.80
        )

        train_indices = indices[:split]

        test_indices = indices[split:]

        X_train = X.iloc[train_indices]

        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]

        y_test = y.iloc[test_indices]

        # -------------------------------------------------
        # IMPROVED RANDOM FOREST
        # -------------------------------------------------
        if disease_name == "Hepatitis ":

             model = ExtraTreesRegressor(
                n_estimators =150,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1.0,
                random_state=42,
                n_jobs=2
            )

        else:

            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
                n_jobs=2
           )
       
        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        r2 = r2_score(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        # -------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------

        models[disease_name] = {

            "model": model,

            "encoder": state_encoder,

            "features": features

        }

        # -------------------------------------------------
        # SAVE METRICS
        # -------------------------------------------------

        metrics[disease_name] = {

            "r2": r2,

            "mse": mse,

            "rmse": rmse,

            "mae": mae,

            "y_test": y_test,

            "predictions": predictions

        }

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        feature_importance[disease_name] = (

            pd.DataFrame({

                "Feature": features,

                "Importance":
                model.feature_importances_

            })

            .sort_values(
                "Importance",
                ascending=False
            )

        )

        processed_data[disease_name] = d

    return (
        models,
        metrics,
        processed_data,
        feature_importance
    )


# =========================================================
# LOAD DATA
# =========================================================

data = load_all_data()


if data.empty:

    st.error(
        "No disease CSV files were found. "
        "Upload all CSV files to the same GitHub repository as app.py."
    )

    st.stop()


# =========================================================
# TRAIN MODELS
# =========================================================

(
    models,
    metrics,
    disease_data_dict,
    feature_importance
) = train_models(data)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 Disease Management")

page = st.sidebar.radio(

    "Select Page",

    [
        "Dashboard",
        "Dataset Overview",
        "Disease Trends",
        "State-wise Analysis",
        "Prediction",
        "Model Performance",
        "Feature Importance",
        "Disease Management"
    ]

)


diseases = list(FILES.keys())


selected = st.sidebar.selectbox(
    "Select Disease",
    diseases
)


# =========================================================
# SELECTED DISEASE DATA
# =========================================================

if selected not in disease_data_dict:

    st.error(
        f"{selected} dataset/model is not available."
    )

    st.stop()


disease_df = disease_data_dict[
    selected
].copy()


selected_model = models[
    selected
]


selected_metrics = metrics[
    selected
]


model = selected_model[
    "model"
]

state_encoder = selected_model[
    "encoder"
]

features = selected_model[
    "features"
]


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header(
        f"📊 {selected} Dashboard"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{len(disease_df):,}"
    )

    c2.metric(
        "Total Cases",
        f"{disease_df['cases'].sum():,.0f}"
    )

    c3.metric(
        "Average Cases",
        f"{disease_df['cases'].mean():.2f}"
    )

    c4.metric(
        "R² Score",
        f"{selected_metrics['r2']:.4f}"
    )

    st.markdown("---")

    if selected_metrics["r2"] >= 0.80:

        st.success(
            f"Excellent model performance! "
            f"R² Score = {selected_metrics['r2']:.4f}"
        )

    elif selected_metrics["r2"] >= 0.60:

        st.info(
            f"Good model performance. "
            f"R² Score = {selected_metrics['r2']:.4f}"
        )

    else:

        st.warning(
            f"Model performance can be improved. "
            f"Current R² Score = {selected_metrics['r2']:.4f}"
        )


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    st.header(
        f"📋 {selected} Dataset Overview"
    )

    st.write(
        f"Selected disease: **{selected}**"
    )

    st.write(
        "Dataset shape:",
        disease_df.shape
    )

    st.dataframe(
        disease_df.head(100),
        use_container_width=True
    )

    st.subheader(
        "Missing Values"
    )

    st.dataframe(
        disease_df.isnull()
        .sum()
        .rename("Missing Values"),
        use_container_width=True
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        disease_df.describe(),
        use_container_width=True
    )


# =========================================================
# DISEASE TRENDS
# =========================================================

elif page == "Disease Trends":

    st.header(
        f"📈 {selected} Disease Trends"
    )

    yearly = (
        disease_df
        .groupby("year")["cases"]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        yearly.index,
        yearly.values,
        marker="o"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Total Cases"
    )

    ax.set_title(
        f"{selected} Yearly Cases Trend"
    )

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)

    st.subheader(
        "Correlation Heatmap"
    )

    cols = [

        "cases",

        "incidence_per_capita",

        "previous_cases",

        "previous_cases_2",

        "previous_cases_3",

        "rolling_average_3",

        "year",

        "week_number"

    ]

    fig2, ax2 = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(

        disease_df[cols]
        .corr(),

        annot=True,

        fmt=".2f",

        cmap="coolwarm",

        ax=ax2

    )

    ax2.set_title(
        f"{selected} Correlation Heatmap"
    )

    st.pyplot(fig2)

    plt.close(fig2)


# =========================================================
# STATE-WISE ANALYSIS
# =========================================================

elif page == "State-wise Analysis":

    st.header(
        f"🗺️ {selected} State-wise Analysis"
    )

    state_cases = (

        disease_df
        .groupby("state")["cases"]
        .sum()
        .sort_values(
            ascending=False
        )

    )

    st.dataframe(

        state_cases
        .rename("Total Cases")
        .to_frame(),

        use_container_width=True

    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    state_cases.head(15).plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "State"
    )

    ax.set_ylabel(
        "Total Cases"
    )

    ax.set_title(
        f"Top States - {selected}"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.header(
        "🔮 Disease Case Prediction"
    )

    st.write(
        f"Prediction for **{selected}**"
    )

    states = sorted(
        disease_df[
            "state"
        ]
        .astype(str)
        .unique()
    )

    state = st.selectbox(
        "State",
        states
    )

    week = st.number_input(

        "Week (YYYYWW)",

        min_value=190001,

        max_value=210053,

        value=int(
            disease_df["week"]
            .median()
        )

    )

    incidence = st.number_input(

        "Incidence per Capita",

        min_value=0.0,

        value=float(
            disease_df[
                "incidence_per_capita"
            ].median()
        )

    )

    previous = st.number_input(

        "Previous Cases",

        min_value=0.0,

        value=float(
            disease_df[
                "previous_cases"
            ].median()
        )

    )

    previous2 = st.number_input(

        "Previous 2 Cases",

        min_value=0.0,

        value=float(
            disease_df[
                "previous_cases_2"
            ].median()
        )

    )

    previous3 = st.number_input(

        "Previous 3 Cases",

        min_value=0.0,

        value=float(
            disease_df[
                "previous_cases_3"
            ].median()
        )

    )

    previous4 = st.number_input(

        "Previous 4 Cases",

        min_value=0.0,

        value=float(
            disease_df[
                "previous_cases_4"
            ].median()
        )

    )

    rolling3 = st.number_input(

        "3-Week Rolling Average",

        min_value=0.0,

        value=float(
            disease_df[
                "rolling_average_3"
            ].median()
        )

    )

    rolling5 = st.number_input(

        "5-Week Rolling Average",

        min_value=0.0,

        value=float(
            disease_df[
                "rolling_average_5"
            ].median()
        )

    )

    if st.button(
        "🔮 Predict Cases"
    ):

        state_encoded = (
            state_encoder
            .transform([state])[0]
        )

        year = int(
            week // 100
        )

        week_number = int(
            week % 100
        )

        month_estimate = int(
            ((week_number - 1) // 4) + 1
        )

        quarter = int(
            ((week_number - 1) // 13) + 1
        )

        row = pd.DataFrame([{

            "year": year,

            "week_number": week_number,

            "month_estimate":
            month_estimate,

            "quarter":
            quarter,

            "state_encoded":
            state_encoded,

            "incidence_per_capita":
            incidence,

            "previous_cases":
            previous,

            "previous_cases_2":
            previous2,

            "previous_cases_3":
            previous3,

            "previous_cases_4":
            previous4,

            "rolling_average_3":
            rolling3,

            "rolling_average_5":
            rolling5

        }])

        prediction = model.predict(
            row[features]
        )[0]

        prediction = max(
            0,
            prediction
        )

        st.success(
            f"Estimated expected cases: "
            f"**{prediction:.2f}**"
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header(
        f"🤖 {selected} Model Performance"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "R² Score",
        f"{selected_metrics['r2']:.4f}"
    )

    c2.metric(
        "MSE",
        f"{selected_metrics['mse']:.4f}"
    )

    c3.metric(
        "RMSE",
        f"{selected_metrics['rmse']:.4f}"
    )

    c4.metric(
        "MAE",
        f"{selected_metrics['mae']:.4f}"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # ALL DISEASE PERFORMANCE
    # -----------------------------------------------------

    st.subheader(
        "All Disease Model Performance"
    )

    performance_rows = []

    for disease_name in metrics:

        performance_rows.append({

            "Disease":
            disease_name,

            "R² Score":
            metrics[disease_name]["r2"],

            "MSE":
            metrics[disease_name]["mse"],

            "RMSE":
            metrics[disease_name]["rmse"],

            "MAE":
            metrics[disease_name]["mae"]

        })

    performance_df = pd.DataFrame(
        performance_rows
    )

    st.dataframe(

        performance_df.style.format({

            "R² Score":
            "{:.4f}",

            "MSE":
            "{:.2f}",

            "RMSE":
            "{:.2f}",

            "MAE":
            "{:.2f}"

        }),

        use_container_width=True

    )

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader(
        "Actual vs Predicted Cases"
    )

    y_test = selected_metrics[
        "y_test"
    ]

    y_pred = selected_metrics[
        "predictions"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    ax.set_xlabel(
        "Actual Cases"
    )

    ax.set_ylabel(
        "Predicted Cases"
    )

    ax.set_title(
        f"{selected} - Actual vs Predicted"
    )

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.header(
        f"⭐ {selected} Feature Importance"
    )

    imp = feature_importance[
        selected
    ]

    st.dataframe(
        imp,
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(

        imp["Feature"][::-1],

        imp["Importance"][::-1]

    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"Random Forest Feature Importance - {selected}"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# DISEASE MANAGEMENT
# =========================================================

elif page == "Disease Management":

    st.header(
        f"🛡️ {selected} Management & Preventive Actions"
    )

    st.info(
        "These are general preventive/management suggestions, "
        "not a medical diagnosis."
    )

    actions = {

        "Hepatitis A": [

            "Maintain hand hygiene",

            "Use safe drinking water",

            "Follow food hygiene practices",

            "Seek medical advice when symptoms occur"

        ],

        "Measles": [

            "Maintain vaccination awareness",

            "Avoid close contact during suspected infection",

            "Monitor outbreaks",

            "Seek medical advice when symptoms occur"

        ],

        "Mumps": [

            "Maintain vaccination awareness",

            "Practise good respiratory hygiene",

            "Avoid close contact when ill",

            "Seek medical advice when symptoms occur"

        ],

        "Pertussis": [

            "Maintain vaccination awareness",

            "Practise respiratory hygiene",

            "Avoid close contact when ill",

            "Seek medical advice when symptoms occur"

        ],

        "Polio": [

            "Maintain vaccination awareness",

            "Use safe water and sanitation",

            "Practise good hygiene",

            "Seek medical advice when symptoms occur"

        ],

        "Rubella": [

            "Maintain vaccination awareness",

            "Follow public-health guidance",

            "Avoid exposure during suspected infection",

            "Seek medical advice when symptoms occur"

        ],

        "Smallpox": [

            "Follow official public-health guidance",

            "Report suspected cases to health authorities",

            "Avoid close contact with suspected cases",

            "Seek urgent professional medical advice"

        ]

    }

    for i, action in enumerate(
        actions[selected],
        1
    ):

        st.write(
            f"**{i}.** {action}"
        )


# =========================================================
# SIDEBAR FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Random Forest Regression • "
    "Separate Models for Each Disease"
)