import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Salary Predictor (Lite)", page_icon="💼", layout="centered")

DATASET_PATH = "Salary_Dataset_DataScienceLovers.csv"

# ---------- LOAD & CLEAN DATA ----------
@st.cache_data
def load_data():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset `{DATASET_PATH}` not found.")
        st.stop()

    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={
        "Job Title": "job_title",
        "Salary": "salary",
        "Location": "location",
        "Job Roles": "skills",
        "Company Name": "company",
    })
    df["salary"] = df["salary"] / 100000.0

    def estimate_exp(row):
        t = str(row["job_title"]).lower()
        s = str(row.get("Employment Status", "")).lower()
        if "intern" in s or "intern" in t or "trainee" in t:
            return 0.5
        if any(w in t for w in ["junior", "jr", "entry", "associate", "sde i", "sde 1"]):
            return 1.5
        if any(w in t for w in ["senior", "sr", "sde iii", "sde 3"]):
            return 6.0
        if any(w in t for w in ["lead", "staff"]):
            return 8.0
        if any(w in t for w in ["principal", "architect", "manager", "director", "head"]):
            return 12.0
        if any(w in t for w in ["sde ii", "sde 2"]):
            return 4.0
        return 3.0

    df["experience"] = df.apply(estimate_exp, axis=1)
    df = df.dropna(subset=["salary", "experience"])
    df = df[(df["salary"] >= 1.0) & (df["salary"] <= 75.0)]

    df["job_title"] = df["job_title"].astype(str).str.strip().str.title()
    df["location"] = df["location"].astype(str).str.strip()
    df["skill_count"] = df["skills"].fillna("").apply(lambda x: len(str(x).split(",")))

    return df


df = load_data()

# Label encode: convert text -> numbers for the model
df["job_title_cat"] = df["job_title"].astype("category")
df["location_cat"] = df["location"].astype("category")
df["role_code"] = df["job_title_cat"].cat.codes
df["loc_code"] = df["location_cat"].cat.codes

role_map_rev = {v: k for k, v in enumerate(df["job_title_cat"].cat.categories)}
loc_map_rev = {v: k for k, v in enumerate(df["location_cat"].cat.categories)}

top_roles = df["job_title"].value_counts().head(30).index.tolist()
top_locations = df["location"].value_counts().head(20).index.tolist()

# ---------- TRAIN MODEL ----------
@st.cache_resource
def train_model(_df):
    X = _df[["experience", "loc_code", "role_code", "skill_count"]]
    y = _df["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = round(r2_score(y_test, model.predict(X_test)) * 100, 1)
    return model, accuracy


model, accuracy = train_model(df)

# ---------- UI ----------
st.title("💼 Tech Salary Predictor (Lite)")
st.caption(f"Random Forest Model  |  Trained on {len(df):,} jobs  |  R² Score: {accuracy}%")
st.write("---")

col1, col2 = st.columns(2)
with col1:
    experience = st.slider("Years of Experience", 0, 25, 2)
    role = st.selectbox("Job Role / Title", sorted(top_roles))

with col2:
    location = st.selectbox("Location", sorted(top_locations))
    skills_input = st.text_input("Your Skills (comma-separated)", placeholder="e.g. Python, SQL, Docker, AWS")

# ---------- PREDICT ----------
if st.button("Predict Salary", type="primary", use_container_width=True):
    skill_list = [s.strip() for s in skills_input.split(",") if s.strip()] if skills_input else []
    skill_count = max(1, len(skill_list))

    loc_code = loc_map_rev.get(location, 0)
    role_code = role_map_rev.get(role, 0)

    input_data = pd.DataFrame(
        [[experience, loc_code, role_code, skill_count]],
        columns=["experience", "loc_code", "role_code", "skill_count"],
    )

    pred = float(model.predict(input_data)[0])
    pred = max(1.5, round(pred, 2))
    low = round(max(1.0, pred * 0.88), 2)
    high = round(pred * 1.15, 2)

    st.success(f"### 💰 Predicted Salary: ₹ {pred} LPA")
    st.info(f"**Expected Range:** ₹ {low} LPA – ₹ {high} LPA")