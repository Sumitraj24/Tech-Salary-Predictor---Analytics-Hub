import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="Tech Salary Predictor & Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .salary-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .stButton>button {
        background: linear-gradient(120deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(120deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. DATA LOADING & PREPROCESSING PIPELINE
# ==========================================================
DATASET_PATH = "Salary_Dataset_DataScienceLovers.csv"

@st.cache_data
def load_and_preprocess_data():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset `{DATASET_PATH}` was not found. Please ensure it is in the repository directory.")
        st.stop()

    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={
        'Job Title': 'job_title',
        'Salary': 'salary',
        'Location': 'location',
        'Job Roles': 'skills',
        'Company Name': 'company'
    })

    # Convert annual salary from INR to LPA
    df['salary'] = df['salary'] / 100000.0

    # Derive realistic experience (years) from job title and employment status
    def estimate_exp(row):
        t = str(row['job_title']).lower()
        s = str(row.get('Employment Status', '')).lower()
        if 'intern' in s or 'intern' in t or 'trainee' in t:
            return 0.5
        if any(w in t for w in ['junior', 'jr', 'entry', 'associate', 'sde i', 'sde 1']):
            return 1.5
        if any(w in t for w in ['senior', 'sr', 'sde iii', 'sde 3']):
            return 6.0
        if any(w in t for w in ['lead', 'staff']):
            return 8.0
        if any(w in t for w in ['principal', 'architect', 'manager', 'director', 'head']):
            return 12.0
        if any(w in t for w in ['sde ii', 'sde 2', 'developer ii']):
            return 4.0
        return 3.0

    df['experience'] = df.apply(estimate_exp, axis=1)

    # Standardize string fields
    for col in ['job_title', 'location', 'company']:
        df[col] = df[col].fillna('other').astype(str).str.lower().str.strip()

    df['location'] = df['location'].str.replace(r'hybrid\s*[-–]\s*', '', regex=True)
    df['location'] = df['location'].str.replace(r'\s*\(all areas\)', '', regex=True).str.strip()
    df['location'] = df['location'].replace({
        'bengaluru': 'bangalore',
        'new delhi': 'delhi',
        'gurgaon': 'delhi / ncr',
        'noida': 'delhi / ncr'
    })
    df['skills'] = df['skills'].fillna('').astype(str)

    # Remove outliers and invalid records
    df = df.dropna(subset=['salary', 'experience'])
    df = df[(df['salary'] >= 1.0) & (df['salary'] <= 75.0)]
    df = df[(df['experience'] >= 0.0) & (df['experience'] <= 30.0)]

    # Calculate skill density count
    df['skill_count'] = df['skills'].apply(
        lambda x: min(len(re.split(r'[,\s]+', str(x).strip())), 20)
    ).clip(1, 20)

    # Group rare categories into 'other'
    for col_name, keep_n in [('location', 25), ('job_title', 45), ('company', 80)]:
        top = df[col_name].value_counts().head(keep_n).index
        df[col_name] = df[col_name].where(df[col_name].isin(top), 'other')

    # Label Encoders
    le_loc = LabelEncoder()
    le_role = LabelEncoder()
    le_co = LabelEncoder()

    df['loc_code'] = le_loc.fit_transform(df['location'])
    df['role_code'] = le_role.fit_transform(df['job_title'])
    df['company_code'] = le_co.fit_transform(df['company'])

    # Normalized company tier (relative compensation benchmark)
    co_mean = df.groupby('company_code')['salary'].mean()
    df['company_tier'] = df['company_code'].map(co_mean)
    t_min, t_max = df['company_tier'].min(), df['company_tier'].max()
    df['company_tier'] = (df['company_tier'] - t_min) / (t_max - t_min + 1e-6)

    # Experience band (0 to 5)
    df['exp_band'] = pd.cut(
        df['experience'], bins=[-1, 1, 3, 6, 10, 15, 35], labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    return df, le_loc, le_role, le_co


df, le_loc, le_role, le_co = load_and_preprocess_data()

FEATURES = ['experience', 'exp_band', 'loc_code', 'role_code', 'company_tier', 'skill_count']

# ==========================================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================================
@st.cache_resource
def train_model(_df):
    X = _df[FEATURES]
    y = _df['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred) * 100, 2)

    feat_imp = pd.DataFrame({
        'Feature': ['Experience (Yrs)', 'Experience Band', 'Location', 'Job Role', 'Company Tier', 'Skill Count'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return model, mae, r2, feat_imp


model, mae, r2, feat_imp = train_model(df)

# ==========================================================
# 4. SIDEBAR - METRICS & INFO
# ==========================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=64)
    st.markdown("### 📈 Project Metrics")
    st.markdown(f"**Trained On:** `{len(df):,}` job records")
    st.markdown(f"**Model Algorithm:** `Random Forest`")
    st.markdown(f"**Accuracy (R²):** `{r2}%`")
    st.markdown(f"**Error Margin (MAE):** `±{mae} LPA`")
    st.markdown("---")
    st.markdown("### 🗂️ Dataset")
    st.markdown(f"- ✅ `{DATASET_PATH}`")
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("Input your years of experience, primary role, skills, and target location to get an estimated market salary range in Indian tech.")

# ==========================================================
# 5. MAIN HEADER
# ==========================================================
st.markdown('<div class="main-header">💼 Tech Salary Predictor & Analytics Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Mining & Machine Learning pipeline trained on Indian IT & Tech compensation benchmarks.</div>', unsafe_allow_html=True)

# Top Stat Cards
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown(f'<div class="metric-card"><h4>Trained Jobs</h4><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<div class="metric-card"><h4>Tech Hubs</h4><h2>{len(le_loc.classes_):,}</h2></div>', unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<div class="metric-card"><h4>Average Salary</h4><h2>₹ {df["salary"].mean():.1f} LPA</h2></div>', unsafe_allow_html=True)
with col_s4:
    st.markdown(f'<div class="metric-card"><h4>Model R²</h4><h2>{r2}%</h2></div>', unsafe_allow_html=True)

st.write("")

# ==========================================================
# 6. TABS NAVIGATION
# ==========================================================
tab_predict, tab_analytics, tab_model = st.tabs([
    "🔮 Predict Salary",
    "📊 Market Analytics & EDA",
    "🧠 Model Architecture & Insights"
])

# ----------------------------------------------------------
# TAB 1: SALARY PREDICTOR
# ----------------------------------------------------------
with tab_predict:
    st.markdown("### 🎯 Enter Candidate & Role Profile")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        experience = st.slider("Years of Total Tech Experience", 0, 25, 3, help="Total professional experience in years.")
        
        available_roles = sorted([r.title() for r in le_role.classes_ if r != 'other'])
        default_role_idx = available_roles.index("Software Development Engineer") if "Software Development Engineer" in available_roles else 0
        role = st.selectbox("Primary Job Role / Domain", available_roles, index=default_role_idx)

        education = st.selectbox(
            "Highest Education Level",
            ["Bachelor's (B.Tech / B.E / B.Sc)", "Master's (M.Tech / M.S / MCA)", "Diploma", "Doctorate (PhD)"],
            index=0
        )

    with form_col2:
        available_locations = sorted([l.title() for l in le_loc.classes_ if l != 'other'])
        default_loc_idx = available_locations.index("Bangalore") if "Bangalore" in available_locations else 0
        location = st.selectbox("Location / Base City", available_locations, index=default_loc_idx)

        company_input = st.text_input(
            "Target Company Name (Optional)",
            placeholder="e.g. TCS, Infosys, Amazon, Google, Microsoft, Wipro, Startup"
        )

        skill_input = st.text_input(
            "Core Technical Skills",
            value="Python, SQL, Machine Learning",
            placeholder="e.g. Python, SQL, Docker, AWS, React, PyTorch"
        )

    st.write("")
    predict_clicked = st.button("🚀 Calculate Expected Salary", use_container_width=True)

    if predict_clicked:
        # Preprocess inputs
        skills_clean = skill_input.strip() if skill_input else "general"
        skill_count = min(len(re.split(r'[,\s]+', skills_clean)), 20)
        
        # Experience band
        exp_band = int(pd.cut([experience], bins=[-1, 1, 3, 6, 10, 15, 35], labels=[0, 1, 2, 3, 4, 5])[0])
        
        # Safe Encoders with fallback
        loc_clean = location.strip().lower()
        if loc_clean in le_loc.classes_:
            loc_code = le_loc.transform([loc_clean])[0]
        elif 'other' in le_loc.classes_:
            loc_code = le_loc.transform(['other'])[0]
        else:
            loc_code = 0

        role_clean = role.strip().lower()
        if role_clean in le_role.classes_:
            role_code = le_role.transform([role_clean])[0]
        elif 'other' in le_role.classes_:
            role_code = le_role.transform(['other'])[0]
        else:
            role_code = 0

        # Company Tier calculation
        co_clean = company_input.strip().lower() if company_input else ""
        if co_clean and co_clean in le_co.classes_:
            co_code = le_co.transform([co_clean])[0]
            co_mean_map = df.groupby('company_code')['salary'].mean()
            raw_tier = co_mean_map.get(co_code, df['company_tier'].mean())
            t_min = df.groupby('company_code')['salary'].mean().min()
            t_max = df.groupby('company_code')['salary'].mean().max()
            company_tier = (raw_tier - t_min) / (t_max - t_min + 1e-6)
        else:
            company_tier = float(df['company_tier'].median())

        # Education Multiplier
        edu_mult = {
            "Diploma": 0.92,
            "Bachelor's (B.Tech / B.E / B.Sc)": 1.0,
            "Master's (M.Tech / M.S / MCA)": 1.08,
            "Doctorate (PhD)": 1.15
        }[education]

        # Predict
        input_row = pd.DataFrame(
            [[experience, exp_band, loc_code, role_code, company_tier, skill_count]],
            columns=FEATURES
        )
        base_pred = float(model.predict(input_row)[0])
        pred = round(max(1.5, base_pred * edu_mult), 2)
        low_bound = round(max(1.0, pred - mae), 2)
        high_bound = round(pred + mae, 2)
        monthly_in_hand = round((pred * 100000) / 12)

        st.markdown(f"""
        <div class="salary-badge">
            <h3 style="margin: 0; opacity: 0.9;">Estimated Market Compensation</h3>
            <h1 style="margin: 10px 0 5px 0; font-size: 3rem; font-weight: 800;">₹ {pred} LPA</h1>
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">
                Expected Salary Range: <strong>₹ {low_bound} – ₹ {high_bound} LPA</strong> &nbsp;|&nbsp; 
                Approx. <strong>₹ {monthly_in_hand:,} / month</strong> gross
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Factor Insights
        st.markdown("#### 🔍 Compensation Drivers Breakdown")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Experience Impact", f"{experience} Yrs", f"Band {exp_band + 1}/6")
        with col_f2:
            st.metric("Location Index", location, "Tier 1 Hub" if location in ['Bangalore', 'Hyderabad', 'Pune'] else "Standard")
        with col_f3:
            st.metric("Skill Depth Bonus", f"{skill_count} skills parsed", "+Market Competitive")

        # Future Projection
        next_exp = min(experience + 2, 25)
        next_band = int(pd.cut([next_exp], bins=[-1, 1, 3, 6, 10, 15, 35], labels=[0, 1, 2, 3, 4, 5])[0])
        next_input = pd.DataFrame([[next_exp, next_band, loc_code, role_code, company_tier, min(skill_count + 2, 20)]], columns=FEATURES)
        raw_next_pred = float(model.predict(next_input)[0]) * edu_mult
        next_pred = round(max(pred * 1.08, raw_next_pred), 2)
        growth_pct = round(((next_pred - pred) / pred) * 100, 1)

        st.info(f"📈 **Career Trajectory Forecast:** In 2 years (at **{next_exp} yrs experience**), your estimated market value could rise to **₹ {next_pred} LPA** (+{growth_pct}% growth).")

# ----------------------------------------------------------
# TAB 2: MARKET ANALYTICS & EDA
# ----------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Tech Compensation Insights & Distributions")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        # Salary Distribution
        fig_dist = px.histogram(
            df,
            x="salary",
            nbins=35,
            title="Overall Tech Salary Distribution (LPA)",
            labels={"salary": "Salary (Lakhs Per Annum - LPA)"},
            color_discrete_sequence=["#2563eb"],
            marginal="box"
        )
        fig_dist.update_layout(template="plotly_white", bargap=0.05)
        st.plotly_chart(fig_dist, width="stretch")

    with eda_col2:
        # Salary vs Experience
        exp_summary = df.groupby('experience')['salary'].mean().reset_index()
        fig_exp = px.line(
            exp_summary,
            x="experience",
            y="salary",
            markers=True,
            title="Average Salary Growth by Years of Experience",
            labels={"experience": "Years of Experience", "salary": "Average Salary (LPA)"},
            color_discrete_sequence=["#10b981"]
        )
        fig_exp.update_layout(template="plotly_white")
        st.plotly_chart(fig_exp, width="stretch")

    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        # Median Salary by Location
        top_locs = df[df['location'] != 'other']['location'].value_counts().head(8).index
        loc_df = df[df['location'].isin(top_locs)].groupby('location')['salary'].median().reset_index().sort_values(by='salary', ascending=True)
        loc_df['location'] = loc_df['location'].str.title()
        fig_loc = px.bar(
            loc_df,
            x="salary",
            y="location",
            orientation="h",
            title="Median Salary by Top Tech Cities",
            labels={"salary": "Median Salary (LPA)", "location": "City"},
            color="salary",
            color_continuous_scale="Viridis"
        )
        fig_loc.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_loc, width="stretch")

    with eda_col4:
        # Salary by Job Role
        top_roles = df[df['job_title'] != 'other']['job_title'].value_counts().head(8).index
        role_df = df[df['job_title'].isin(top_roles)].groupby('job_title')['salary'].median().reset_index().sort_values(by='salary', ascending=True)
        role_df['job_title'] = role_df['job_title'].str.title()
        fig_role = px.bar(
            role_df,
            x="salary",
            y="job_title",
            orientation="h",
            title="Median Salary by Top Tech Roles",
            labels={"salary": "Median Salary (LPA)", "job_title": "Role"},
            color="salary",
            color_continuous_scale="Blues"
        )
        fig_role.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_role, width="stretch")

# ----------------------------------------------------------
# TAB 3: MODEL ARCHITECTURE & DETAILS
# ----------------------------------------------------------
with tab_model:
    st.markdown("### 🧠 Machine Learning Model Specifications")

    col_m1, col_m2 = st.columns([1.2, 1])

    with col_m1:
        fig_imp = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance in Salary Determination",
            labels={"Importance": "Gini Importance Score", "Feature": "Feature"},
            color="Importance",
            color_continuous_scale="Purples"
        )
        fig_imp.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, width="stretch")

    with col_m2:
        st.markdown("#### Model Architecture")
        st.table(pd.DataFrame({
            "Hyperparameter": ["Estimator", "n_estimators", "max_depth", "min_samples_leaf", "Test Split", "R² Score", "MAE"],
            "Specification": ["RandomForestRegressor", "150", "12", "5", "20%", f"{r2}%", f"±{mae} LPA"]
        }))

        st.markdown("#### Data Mining Methodology")
        st.write("""
        1. **Data Cleaning & Normalization:** Parsed compensation data from tech compensation benchmarks, converted INR to LPA, and filtered outliers (1.0 to 75.0 LPA).
        2. **Seniority & Experience Mining:** Imputed experience profiles from job titles and seniority designations (Intern, Junior, Senior, Staff, Architect).
        3. **Categorical Encoding & Tiering:** Derived smoothed company compensation tiers and encoded locations.
        4. **Validation:** 80/20 train/test split with MAE and R² evaluation.
        """)