
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Predicting Disease Management System", page_icon="🩺", layout="wide")

@st.cache_data
def prepare_data():
    df = pd.read_csv("hepatitis.csv").drop_duplicates()
    df["cases"] = df["cases"].fillna(df["cases"].mean())
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(df["week"].median())
    df["incidence_per_capita"] = df["incidence_per_capita"].fillna(df["incidence_per_capita"].median())
    df["disease"] = df["disease"].fillna("Unknown")
    df["year"] = (df["week"] // 100).astype(int)
    df["week_number"] = (df["week"] % 100).astype(int)
    df = df.sort_values(["state", "disease", "week"])
    df["previous_cases"] = df.groupby(["state","disease"])["cases"].shift(1).fillna(0)
    df["previous_cases_2"] = df.groupby(["state","disease"])["cases"].shift(2).fillna(0)
    df["rolling_average"] = df.groupby(["state","disease"])["cases"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    ).fillna(0)

    se = LabelEncoder()
    de = LabelEncoder()
    df["state_encoded"] = se.fit_transform(df["state"])
    df["disease_encoded"] = de.fit_transform(df["disease"])

    features = ["year","week_number","state_encoded","disease_encoded",
                "incidence_per_capita","previous_cases","previous_cases_2","rolling_average"]
    X, y = df[features], df["cases"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "R²": r2_score(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred)
    }

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return df, model, se, de, metrics, importance, y_test, pred

try:
    df, model, state_encoder, disease_encoder, metrics, importance, y_test, y_pred = prepare_data()
except FileNotFoundError:
    st.error("hepatitis.csv not found. Keep hepatitis.csv in the same folder as app.py.")
    st.stop()

st.sidebar.title("🩺 Disease Management")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard", "📊 Dataset Overview", "📈 Disease Trends",
    "🗺️ State-wise Analysis", "🤖 Prediction",
    "📌 Model Performance", "🔍 Feature Importance", "💡 Disease Management"
])

if page == "🏠 Dashboard":
    st.title("🩺 Predicting Disease Management System")
    st.subheader("Hepatitis A Case Prediction")
    st.write("Machine Learning system for analysing Hepatitis A case patterns and estimating expected cases.")

    a,b,c,d = st.columns(4)
    a.metric("Total Records", f"{len(df):,}")
    b.metric("Total Cases", f"{df.cases.sum():,.0f}")
    c.metric("Average Cases", f"{df.cases.mean():.2f}")
    d.metric("R² Score", f"{metrics['R²']:.4f}")

    yearly = df.groupby("year")["cases"].sum()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Cases")
    ax.set_title("Hepatitis A Cases Trend Over Years"); ax.grid(True)
    st.pyplot(fig)

elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    a,b = st.columns(2)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", df.shape[1])
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Statistics")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "📈 Disease Trends":
    st.title("📈 Hepatitis A Disease Trends")
    yearly = df.groupby("year")["cases"].sum()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Cases")
    ax.set_title("Hepatitis A Cases Trend Over Years"); ax.grid(True)
    st.pyplot(fig)

    import seaborn as sns
    corr = df[["year","week_number","incidence_per_capita","previous_cases",
               "previous_cases_2","rolling_average","cases"]].corr()
    fig2, ax2 = plt.subplots(figsize=(10,7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Heatmap - Hepatitis A")
    st.pyplot(fig2)

elif page == "🗺️ State-wise Analysis":
    st.title("🗺️ State-wise Hepatitis A Analysis")
    state_cases = df.groupby("state")["cases"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(state_cases.index.astype(str), state_cases.values)
    ax.set_xlabel("State"); ax.set_ylabel("Total Cases")
    ax.set_title("State-wise Hepatitis A Cases")
    plt.xticks(rotation=90); plt.tight_layout()
    st.pyplot(fig)
    a,b = st.columns(2)
    a.metric("Most Affected State", str(state_cases.idxmax()))
    a.write(f"Cases: {state_cases.max():,.0f}")
    b.metric("Least Affected State", str(state_cases.idxmin()))
    b.write(f"Cases: {state_cases.min():,.0f}")
    st.subheader("Top 5 States")
    st.dataframe(state_cases.head(5).to_frame("Total Cases"), use_container_width=True)

elif page == "🤖 Prediction":
    st.title("🤖 Hepatitis A Case Prediction")
    state = st.selectbox("Select State", sorted(df.state.unique()))
    week = st.number_input("Week (YYYYWW)", int(df.week.min()), int(df.week.max())+100, int(df.week.max()))
    incidence = st.number_input("Incidence per Capita", min_value=0.0, value=float(df.incidence_per_capita.median()))
    previous = st.number_input("Previous Cases", min_value=0.0, value=float(df.previous_cases.median()))
    previous2 = st.number_input("Previous 2nd Cases", min_value=0.0, value=float(df.previous_cases_2.median()))
    rolling = st.number_input("3-Week Rolling Average", min_value=0.0, value=float(df.rolling_average.median()))

    if st.button("🔮 Predict Cases", type="primary"):
        x = pd.DataFrame({
            "year":[int(week//100)], "week_number":[int(week%100)],
            "state_encoded":[int(state_encoder.transform([state])[0])],
            "disease_encoded":[int(disease_encoder.transform(["Hepatitis A"])[0])],
            "incidence_per_capita":[incidence], "previous_cases":[previous],
            "previous_cases_2":[previous2], "rolling_average":[rolling]
        })
        result = model.predict(x)[0]
        st.success("Prediction completed successfully!")
        st.metric("Predicted Hepatitis A Cases", f"{result:,.2f}")
        if result < 10: st.info("🟢 Low case level — continue routine monitoring.")
        elif result < 50: st.warning("🟡 Moderate case level — increase monitoring and preventive action.")
        else: st.error("🔴 High case level — review the situation and consult health authorities.")

elif page == "📌 Model Performance":
    st.title("📌 Model Performance")
    a,b,c,d = st.columns(4)
    a.metric("R² Score", f"{metrics['R²']:.4f}")
    b.metric("MSE", f"{metrics['MSE']:.4f}")
    c.metric("RMSE", f"{metrics['RMSE']:.4f}")
    d.metric("MAE", f"{metrics['MAE']:.4f}")

    comparison = pd.DataFrame({"Actual Cases": y_test.values[:100],
                               "Predicted Cases": np.round(y_pred[:100],2)})
    st.subheader("Actual vs Predicted")
    st.dataframe(comparison.head(20), use_container_width=True)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(comparison["Actual Cases"], label="Actual Cases")
    ax.plot(comparison["Predicted Cases"], label="Predicted Cases")
    ax.set_xlabel("Samples"); ax.set_ylabel("Cases")
    ax.set_title("Actual vs Predicted Hepatitis A Cases"); ax.legend()
    st.pyplot(fig)

elif page == "🔍 Feature Importance":
    st.title("🔍 Random Forest Feature Importance")
    st.dataframe(importance, use_container_width=True)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(importance.Feature, importance.Importance)
    ax.set_xlabel("Importance"); ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")
    ax.invert_yaxis(); plt.tight_layout()
    st.pyplot(fig)

elif page == "💡 Disease Management":
    st.title("💡 Disease Management & Preventive Actions")
    st.markdown("""
    ### 🧼 Hygiene
    - Maintain proper hand hygiene.
    - Use safe and clean drinking water.
    - Follow proper food hygiene practices.

    ### 📊 Monitoring
    - Monitor disease case trends regularly.
    - Identify states with increasing case levels.
    - Compare current cases with previous patterns.

    ### 🚨 High Case Levels
    - Increase public health monitoring.
    - Investigate possible outbreak patterns.
    - Take appropriate preventive measures.
    - Consult qualified health authorities for further action.

    ### 🤖 Machine Learning
    - Analyse historical disease case patterns.
    - Predict expected case levels.
    - Support data-driven monitoring and planning.
    """)

    st.warning("This system provides ML-based estimates for monitoring and planning. It is not a medical diagnosis system.")

st.sidebar.divider()
st.sidebar.caption("Random Forest Regression")
