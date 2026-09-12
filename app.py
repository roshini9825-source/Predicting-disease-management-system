import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Predicting Disease Management System", page_icon="🩺", layout="wide")

st.title("🩺 Predicting Disease Management System")
st.subheader("Multi-Disease Case Prediction")
st.write("Machine Learning system for analysing disease case patterns and estimating expected cases.")

FILES = {
    "Hepatitis A": "hepatitis.csv",
    "Measles": "measles.csv",
    "Mumps": "mumps.csv",
    "Pertussis": "pertussis.csv",
    "Polio": "polio.csv",
    "Rubella": "rubella.csv",
    "Smallpox": "smallpox.csv",
}

@st.cache_data
def load_all_data():
    frames = []
    for name, file in FILES.items():
        try:
            d = pd.read_csv(file)
            d["disease_display"] = name
            frames.append(d)
        except FileNotFoundError:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_resource
def train_model(data):
    d = data.copy()
    d = d.drop_duplicates()
    d["cases"] = pd.to_numeric(d["cases"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d["incidence_per_capita"] = pd.to_numeric(d["incidence_per_capita"], errors="coerce")
    d["cases"] = d["cases"].fillna(d["cases"].mean())
    d["week"] = d["week"].fillna(d["week"].median())
    d["incidence_per_capita"] = d["incidence_per_capita"].fillna(d["incidence_per_capita"].median())
    d["state"] = d["state"].fillna("Unknown").astype(str)
    d["disease_display"] = d["disease_display"].fillna("Unknown").astype(str)
    d["year"] = (d["week"] // 100).astype(int)
    d["week_number"] = (d["week"] % 100).astype(int)
    d = d.sort_values(["disease_display", "state", "week"]).reset_index(drop=True)
    d["previous_cases"] = d.groupby(["disease_display", "state"])["cases"].shift(1).fillna(0)
    d["previous_cases_2"] = d.groupby(["disease_display", "state"])["cases"].shift(2).fillna(0)
    d["rolling_average"] = d.groupby(["disease_display", "state"])["cases"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    ).fillna(0)
    state_encoder = LabelEncoder()
    disease_encoder = LabelEncoder()
    d["state_encoded"] = state_encoder.fit_transform(d["state"])
    d["disease_encoded"] = disease_encoder.fit_transform(d["disease_display"])
    features = ["year", "week_number", "state_encoded", "disease_encoded",
                "incidence_per_capita", "previous_cases", "previous_cases_2", "rolling_average"]
    X = d[features]
    y = d["cases"]
    # Time-ordered 80/20 split for a more realistic forecasting evaluation.
    order = d["week"].sort_values().index
    split = int(len(order) * 0.8)
    train_idx, test_idx = order[:split], order[split:]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X.loc[train_idx], y.loc[train_idx])
    pred = model.predict(X.loc[test_idx])
    metrics = {
        "r2": r2_score(y.loc[test_idx], pred),
        "mse": mean_squared_error(y.loc[test_idx], pred),
        "rmse": np.sqrt(mean_squared_error(y.loc[test_idx], pred)),
        "mae": mean_absolute_error(y.loc[test_idx], pred),
    }
    return d, features, model, state_encoder, disease_encoder, metrics, y.loc[test_idx], pred

data = load_all_data()

if data.empty:
    st.error("No disease CSV files were found. Upload the CSV files to the same GitHub repository as app.py.")
    st.stop()

df, features, model, state_encoder, disease_encoder, metrics, y_test, y_pred = train_model(data)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Dataset Overview", "Disease Trends", "State-wise Analysis",
     "Prediction", "Model Performance", "Feature Importance", "Disease Management"]
)

diseases = list(FILES.keys())
selected = st.sidebar.selectbox("Select Disease", diseases)
disease_df = df[df["disease_display"] == selected].copy()

if page == "Dashboard":
    st.header(f"📊 {selected} Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(disease_df):,}")
    c2.metric("Total Cases", f"{disease_df['cases'].sum():,.0f}")
    c3.metric("Average Cases", f"{disease_df['cases'].mean():.2f}")
    c4.metric("Overall R² Score", f"{metrics['r2']:.4f}")
    st.markdown("---")
    st.write("Use the sidebar to explore disease trends, state-wise analysis, prediction, model performance, feature importance, and management actions.")

elif page == "Dataset Overview":
    st.header("📋 Dataset Overview")
    st.write(f"Selected disease: **{selected}**")
    st.dataframe(disease_df.head(100), use_container_width=True)
    st.write("Dataset shape:", disease_df.shape)
    st.write("Missing values:")
    st.dataframe(disease_df.isnull().sum().rename("Missing Values"), use_container_width=True)

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

    st.subheader("Correlation Heatmap")
    cols = ["cases", "incidence_per_capita", "previous_cases", "previous_cases_2", "rolling_average", "year", "week_number"]
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(disease_df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title(f"{selected} Correlation Heatmap")
    st.pyplot(fig2)

elif page == "State-wise Analysis":
    st.header(f"🗺️ {selected} State-wise Analysis")
    state_cases = disease_df.groupby("state")["cases"].sum().sort_values(ascending=False)
    st.dataframe(state_cases.rename("Total Cases").to_frame(), use_container_width=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    state_cases.head(15).plot(kind="bar", ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Total Cases")
    ax.set_title(f"Top States - {selected}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Prediction":
    st.header("🔮 Disease Case Prediction")
    st.write(f"Prediction for **{selected}**")
    states = sorted(disease_df["state"].astype(str).unique())
    state = st.selectbox("State", states)
    week = st.number_input("Week (YYYYWW)", min_value=190001, max_value=210053, value=int(disease_df["week"].median()))
    incidence = st.number_input("Incidence per Capita", min_value=0.0, value=float(disease_df["incidence_per_capita"].median()))
    previous = st.number_input("Previous Cases", min_value=0.0, value=float(disease_df["previous_cases"].median()))
    previous2 = st.number_input("Previous 2 Cases", min_value=0.0, value=float(disease_df["previous_cases_2"].median()))
    rolling = st.number_input("3-Week Rolling Average", min_value=0.0, value=float(disease_df["rolling_average"].median()))
    row = pd.DataFrame([{
        "year": int(week // 100),
        "week_number": int(week % 100),
        "state_encoded": state_encoder.transform([state])[0],
        "disease_encoded": disease_encoder.transform([selected])[0],
        "incidence_per_capita": incidence,
        "previous_cases": previous,
        "previous_cases_2": previous2,
        "rolling_average": rolling
    }])
    if st.button("🔮 Predict Cases"):
        prediction = model.predict(row[features])[0]
        st.success(f"Estimated expected cases: **{prediction:.2f}**")

elif page == "Model Performance":
    st.header("🤖 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['r2']:.4f}")
    c2.metric("MSE", f"{metrics['mse']:.4f}")
    c3.metric("RMSE", f"{metrics['rmse']:.4f}")
    c4.metric("MAE", f"{metrics['mae']:.4f}")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual Cases")
    ax.set_ylabel("Predicted Cases")
    ax.set_title("Actual vs Predicted Cases")
    ax.grid(True)
    st.pyplot(fig)

elif page == "Feature Importance":
    st.header("⭐ Feature Importance")
    imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.dataframe(imp, use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp["Feature"], imp["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

elif page == "Disease Management":
    st.header(f"🛡️ {selected} Management & Preventive Actions")
    st.info("These are general preventive/management suggestions, not a medical diagnosis.")
    actions = {
        "Hepatitis A": ["Maintain hand hygiene", "Use safe drinking water", "Follow food hygiene practices", "Seek medical advice when symptoms occur"],
        "Measles": ["Maintain vaccination awareness", "Avoid close contact during suspected infection", "Monitor outbreaks", "Seek medical advice when symptoms occur"],
        "Mumps": ["Maintain vaccination awareness", "Practise good respiratory hygiene", "Avoid close contact when ill", "Seek medical advice when symptoms occur"],
        "Pertussis": ["Maintain vaccination awareness", "Practise respiratory hygiene", "Avoid close contact when ill", "Seek medical advice when symptoms occur"],
        "Polio": ["Maintain vaccination awareness", "Use safe water and sanitation", "Practise good hygiene", "Seek medical advice when symptoms occur"],
        "Rubella": ["Maintain vaccination awareness", "Follow public-health guidance", "Avoid exposure during suspected infection", "Seek medical advice when symptoms occur"],
        "Smallpox": ["Follow official public-health guidance", "Report suspected cases to health authorities", "Avoid close contact with suspected cases", "Seek urgent professional medical advice"]
    }
    for i, action in enumerate(actions[selected], 1):
        st.write(f"**{i}.** {action}")

st.sidebar.markdown("---")
st.sidebar.caption("Random Forest Regression • Multi-Disease Analysis")
