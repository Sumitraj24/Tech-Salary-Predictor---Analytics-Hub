# 💼 Tech Salary Predictor & Analytics Hub

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end **Machine Learning Web Application** that analyzes tech compensation trends across India and predicts market salaries for tech professionals based on experience, role, target company, location, education, and skills.

---

## 🌟 Key Features

### 1. 🔮 Interactive Salary Predictor
- Predicts expected annual compensation in **₹ Lakhs Per Annum (LPA)** and estimated gross monthly salary.
- Calculates an **80% confidence interval** (Range Low to Range High) based on model error bounds.
- Factors in:
  - **Years of Experience** (0 to 25+ years)
  - **Primary Tech Domain** (Software Development Engineer, Data Scientist, Frontend, Backend, Android, iOS, Cloud, QA, etc.)
  - **Location** (Bangalore, Hyderabad, Pune, Mumbai, Delhi/NCR, Chennai, Kolkata, Remote)
  - **Company Tier** (empirical median compensation benchmarks across major tech companies and startups)
  - **Education Multiplier** (Diploma, Bachelor's, Master's, PhD)
  - **Skill Depth** (parses technical skill density and stacks)
- **Career Trajectory Forecast**: Projects estimated compensation 2 years ahead with experience growth.

### 2. 📊 Market Analytics & EDA Dashboard
- **Salary Distribution**: Histogram with box plot overlay showing median and interquartile ranges.
- **Experience Growth Curve**: Non-linear compensation trajectory across career milestones.
- **Geographic Comparison**: Median tech compensation across top Indian tech hubs.
- **Domain Breakdown**: Top paying tech roles and specializations.

### 3. 🧠 Model Explainability & Diagnostics
- **Feature Importance**: Gini importance breakdown highlighting compensation drivers.
- **Model Evaluation**: Transparent metrics including Mean Absolute Error (MAE) and R² score on holdout test data.

---

## 📁 Repository Structure

```text
├── app.py                               # Core Streamlit web application & analytics dashboard
├── Salary_Dataset_DataScienceLovers.csv # Primary benchmark dataset (22,770+ records)
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Sumitraj24/tech-salary-predictor.git
cd tech-salary-predictor
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the application.

---

## 📊 Dataset & Methodology

- **Dataset**: `Salary_Dataset_DataScienceLovers.csv` (22,770 rows, 8 features).
- **Features Captured**:
  - `Job Title` (Seniority, domain role, level)
  - `Company Name` (Tier benchmark)
  - `Location` (Major IT hubs)
  - `Job Roles` (Core technical categorization)
  - `Rating` (Company rating)
  - `Salary` (Converted from INR to LPA)
- **Data Preprocessing**:
  - Outlier filtering (bounds between 1.0 LPA and 75.0 LPA).
  - Seniority and experience derivation from titles and status.
  - Label encoding for categorical variables with empirical company tier smoothing.
  - Categorical bucketing for rare entities (`other`).

---

## 🤖 Machine Learning Model

- **Algorithm**: `RandomForestRegressor`
  - `n_estimators`: 150
  - `max_depth`: 12
  - `min_samples_leaf`: 5
  - `random_state`: 42
- **Validation**: 80/20 train/test split.
- **Evaluation**:
  - **Mean Absolute Error (MAE)**: ~3.5 LPA
  - **R² Score**: Market competitive benchmark on real-world crowd-reported salary data.

---

## 🛠️ Tech Stack

- **Frontend / App Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Visualizations**: [Plotly Express](https://plotly.com/python/)

---

## 📄 License
This project is licensed under the MIT License - feel free to use and adapt it for learning or personal projects!
