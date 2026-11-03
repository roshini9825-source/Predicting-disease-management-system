import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
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
# DISEASE FILES  (key "Hepatitis A" now matches the management page)
# =========================================================

FILES = {
    "Hepatitis A": "hepatitis.csv",
    "Measles": "measles.csv",
    "Mumps": "mumps.csv",
    "Pertussis": "pertussis.csv",
    "Rubella": "rubella.csv",
    "Smallpox": "smallpox.csv",
}

# incidence_per_capita is derived from cases, so it is NOT used as a feature (target leakage).


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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================================================
# FEATURE ENGINEERING (only past information is used)
# =========================================================

FEATURES = [
    "year","week_number","month_estimate","quarter","week_sin","week_cos","time_index","state_encoded",
    "previous_cases","previous_cases_2","previous_cases_3","previous_cases_4",
    "previous_cases_5","previous_cases_6","previous_cases_7","previous_cases_8",
    "rolling_average_3","rolling_average_5","rolling_average_8","rolling_average_13","rolling_average_26",
    "rolling_std_5","rolling_max_8","ewm_4",
    "same_week_last_year","last_year_window",
    "national_prev_1","national_prev_2","national_roll_4",
]

def build_features(d):
    d = d.drop_duplicates(subset=["week","state"]).copy()
    for c in ["cases","week","incidence_per_capita"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["cases","week"])
    d["state"] = d["state"].fillna("Unknown").astype(str)
    d["week"] = d["week"].astype(int)
    d["year"] = d["week"]//100
    d["week_number"] = d["week"]%100
    d["month_estimate"] = ((d["week_number"]-1)//4)+1
    d["quarter"] = ((d["week_number"]-1)//13)+1
    d["week_sin"] = np.sin(2*np.pi*d["week_number"]/52.0)
    d["week_cos"] = np.cos(2*np.pi*d["week_number"]/52.0)
    ranks = {w:i for i,w in enumerate(sorted(d["week"].unique()))}
    d["time_index"] = d["week"].map(ranks)
    d = d.sort_values(["state","week"]).reset_index(drop=True)

    g = d.groupby("state")["cases"]
    for k in range(1,9):
        d["previous_cases" if k==1 else f"previous_cases_{k}"] = g.shift(k)
    sh = g.shift(1)
    gs = sh.groupby(d["state"])
    for w in (3,5,8,13,26):
        d[f"rolling_average_{w}"] = gs.transform(lambda x,w=w: x.rolling(w, min_periods=1).mean())
    d["rolling_std_5"] = gs.transform(lambda x: x.rolling(5, min_periods=2).std())
    d["rolling_max_8"] = gs.transform(lambda x: x.rolling(8, min_periods=1).max())
    d["ewm_4"] = gs.transform(lambda x: x.ewm(span=4, min_periods=1).mean())

    # same week last year (+-1 week window), matched by calendar, not by row offset
    key = d[["state","year","week_number","cases"]]
    def lookup(delta):
        k = key.copy(); k["year"] = k["year"]+1; k["week_number"] = k["week_number"]-delta
        k = k.rename(columns={"cases":f"ly{delta}"})
        return d[["state","year","week_number"]].merge(k[["state","year","week_number",f"ly{delta}"]],on=["state","year","week_number"],how="left")[f"ly{delta}"].values
    d["same_week_last_year"] = lookup(0)
    d["last_year_window"] = np.nanmean(np.vstack([lookup(-1),lookup(0),lookup(1)]),axis=0)

    # national totals from PREVIOUS weeks only
    nat = d.groupby("time_index")["cases"].sum()
    nat = nat.reindex(range(int(nat.index.max())+1)).fillna(0)
    d["national_prev_1"] = d["time_index"].map(nat.shift(1))
    d["national_prev_2"] = d["time_index"].map(nat.shift(2))
    d["national_roll_4"] = d["time_index"].map(nat.shift(1).rolling(4,min_periods=1).mean())

    lag_cols = FEATURES[8:]
    sm = d.groupby("state")["cases"].transform("median")
    for c in lag_cols:
        d[c] = d[c].fillna(sm if c not in ("national_prev_1","national_prev_2","national_roll_4") else 0).fillna(0)
    d["state_encoded"] = LabelEncoder().fit_transform(d["state"])
    return d


def new_model():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42,
    )


# =========================================================
# TRAIN SEPARATE MODEL FOR EACH DISEASE
# R² is the mean of 5-fold cross-validation (stable; a single split is
# dominated by a few outlier weeks). The final model is fit on all rows.
# =========================================================

@st.cache_resource
def train_models(data):
    models, metrics, processed, importance = {}, {}, {}, {}

    for disease_name in FILES:
        d = data[data["disease_display"] == disease_name]
        if d.empty:
            continue

        d = build_features(d)
        X, y = d[FEATURES], d["cases"]

        oof = np.zeros(len(d))
        fold_r2 = []
        for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
            m = new_model().fit(X.iloc[tr], y.iloc[tr])
            oof[te] = np.clip(m.predict(X.iloc[te]), 0, None)
            fold_r2.append(r2_score(y.iloc[te], oof[te]))

        model = new_model().fit(X, y)

        mse = mean_squared_error(y, oof)
        metrics[disease_name] = {
            "r2": float(np.mean(fold_r2)),
            "fold_r2": fold_r2,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y, oof),
            "y_test": y,
            "predictions": oof,
        }

        sample = d.sample(min(3000, len(d)), random_state=42)
        pi = permutation_importance(
            model, sample[FEATURES], sample["cases"],
            n_repeats=2, random_state=42, n_jobs=1,
        )
        importance[disease_name] = (
            pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        models[disease_name] = {"model": model, "features": FEATURES}
        processed[disease_name] = d

    return models, metrics, processed, importance


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
# PREDICTION  (features are built automatically from the history)
# =========================================================

elif page == "Prediction":
    st.header("🔮 Disease Case Prediction")
    st.write(f"Prediction for **{selected}**")

    states = sorted(disease_df["state"].astype(str).unique())
    state = st.selectbox("State", states)

    sdf = disease_df[disease_df["state"] == state].sort_values("week", ascending=False)
    week = st.selectbox("Week (YYYYWW)", sdf["week"].head(300).tolist())
    row = sdf[sdf["week"] == week].iloc[0]

    with st.expander("Adjust inputs manually (what-if)"):
        def num(label, col):
            return st.number_input(label, min_value=0.0, value=float(row[col]))
        overrides = {
            "previous_cases": num("Previous Cases", "previous_cases"),
            "previous_cases_2": num("Previous 2 Cases", "previous_cases_2"),
            "previous_cases_3": num("Previous 3 Cases", "previous_cases_3"),
            "previous_cases_4": num("Previous 4 Cases", "previous_cases_4"),
            "rolling_average_3": num("3-Week Rolling Average", "rolling_average_3"),
            "rolling_average_5": num("5-Week Rolling Average", "rolling_average_5"),
        }

    if st.button("🔮 Predict Cases"):
        x = row[features].astype(float).to_frame().T
        for k, v in overrides.items():
            x[k] = v
        prediction = max(0.0, float(model.predict(x[features])[0]))
        st.success(f"Estimated expected cases: **{prediction:.2f}**")
        st.caption(f"Actual reported cases for this week: {row['cases']:.0f}")


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
    st.caption("R² = mean of 5-fold cross-validation; folds: " + ", ".join(f"{v:.3f}" for v in selected_metrics["fold_r2"]))

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

    model_name = "Gradient Boosting (permutation)"

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
    "Gradient Boosting Regression • 5-fold CV R² • Separate Models for Each Disease"
)
