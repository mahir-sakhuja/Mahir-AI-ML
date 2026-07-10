import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Set page config for a premium dashboard feel
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="💼",
    layout="wide"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f1f5f9;
    }
    h1, h2, h3 {
        color: #818cf8 !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Employee Retention Predictor")
st.markdown("Predict the probability of an employee leaving the company using Logistic Regression.")

# Load data
csv_path = os.path.join(os.path.dirname(__file__), "HR_comma_sep.csv")
if not os.path.exists(csv_path):
    st.error("Dataset `HR_comma_sep.csv` not found!")
    st.stop()

@st.cache_data
def load_and_train():
    df = pd.read_csv(csv_path)
    
    # Features & Target
    X = df.drop(columns=['left'])
    y = df['left']
    
    # Pipelines for numerical and categorical attributes
    num_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
    cat_features = ['Department', 'salary']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X, y)
    return df, pipeline

df, pipeline = load_and_train()

# Sidebar for employee data inputs
st.sidebar.header("👤 Employee Details")
satisfaction = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.05)
evaluation = st.sidebar.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.05)
projects = st.sidebar.number_input("Number of Projects", 1, 10, 3)
hours = st.sidebar.slider("Average Monthly Hours", 50, 350, 200, 10)
tenure = st.sidebar.slider("Time Spent in Company (years)", 1, 10, 3)
accident = st.sidebar.checkbox("Had Work Accident?")
promotion = st.sidebar.checkbox("Promoted in last 5 years?")
dept = st.sidebar.selectbox("Department", sorted(df['Department'].unique()))
salary = st.sidebar.selectbox("Salary Level", ["low", "medium", "high"])

# Create prediction dataframe
input_data = pd.DataFrame([{
    'satisfaction_level': satisfaction,
    'last_evaluation': evaluation,
    'number_project': projects,
    'average_montly_hours': hours,
    'time_spend_company': tenure,
    'Work_accident': 1 if accident else 0,
    'promotion_last_5years': 1 if promotion else 0,
    'Department': dept,
    'salary': salary
}])

pred_prob = pipeline.predict_proba(input_data)[0][1]
pred_class = pipeline.predict(input_data)[0]

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(df):,}</div>
        <div class="metric-lbl">Total Dataset Size</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    retention_rate = (1 - df['left'].mean()) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{retention_rate:.1f}%</div>
        <div class="metric-lbl">Average Retention Rate</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    color = "#f43f5e" if pred_class == 1 else "#10b981"
    status_text = "LIKELY TO LEAVE" if pred_class == 1 else "LIKELY TO STAY"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: {color};">{status_text}</div>
        <div class="metric-lbl">Model Prediction</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Main output layout
col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader("Retention Probability Analysis")
    st.write(f"The model estimates a **{pred_prob*100:.1f}%** chance of this employee leaving the company.")
    
    # Display gauge/bar chart for probability
    st.progress(pred_prob)
    
    st.write("### Prediction Interpretation")
    if pred_prob > 0.7:
        st.error("⚠️ **High Risk:** The satisfaction level, workload, or salary structure points to a very high risk of employee turnover. Corrective action/review is recommended.")
    elif pred_prob > 0.4:
        st.warning("⚡ **Moderate Risk:** Monitor satisfaction and workload balance closely.")
    else:
        st.success("✅ **Low Risk:** The employee shows standard patterns consistent with those who stay with the company.")

with col_stats:
    st.subheader("Historical Correlations")
    st.write("Average stats of employees who **stayed** vs **left**:")
    grouped_stats = df.groupby('left')[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years']].mean()
    grouped_stats.index = ['Stayed (0)', 'Left (1)']
    st.dataframe(grouped_stats)
