import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import os

# Set page config for a premium dashboard feel
st.set_page_config(
    page_title="Canada Income Predictor",
    page_icon="🍁",
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

st.title("🍁 Canada Per Capita Income Predictor")
st.markdown("Simple Linear Regression application predicting Canada's per capita income based on historical data.")

# Load data
csv_path = os.path.join(os.path.dirname(__file__), "canada_per_capita_income.csv")
if not os.path.exists(csv_path):
    st.error("Dataset `canada_per_capita_income.csv` not found!")
    st.stop()

df = pd.read_csv(csv_path)

# Prepare model
X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

# Sidebar for predictions
st.sidebar.header("🔮 Prediction Panel")
predict_year = st.sidebar.slider("Select Year to Predict:", min_value=2017, max_value=2050, value=2020)

predicted_val = model.predict([[predict_year]])[0]

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{df['year'].min()} - {df['year'].max()}</div>
        <div class="metric-lbl">Historical Range</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">${df['per capita income (US$)'].max():,.2f}</div>
        <div class="metric-lbl">Max Historical Income ({df.loc[df['per capita income (US$)'].idxmax(), 'year']})</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: #4ade80;">${predicted_val:,.2f}</div>
        <div class="metric-lbl">Predicted Income for {predict_year}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

col_plot, col_coef = st.columns([2, 1])

with col_plot:
    st.subheader("Historical Data vs. Regression Line")
    
    # Generate regression line coordinates
    years_range = np.arange(df['year'].min(), predict_year + 2).reshape(-1, 1)
    predicted_line = model.predict(years_range)
    
    fig = go.Figure()
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df['year'], y=df['per capita income (US$)'],
        mode='markers', name='Actual Data',
        marker=dict(color='#38bdf8', size=8)
    ))
    # Line plot
    fig.add_trace(go.Scatter(
        x=years_range.flatten(), y=predicted_line,
        mode='lines', name='Regression Line',
        line=dict(color='#818cf8', width=2)
    ))
    # Prediction point
    fig.add_trace(go.Scatter(
        x=[predict_year], y=[predicted_val],
        mode='markers', name=f'Prediction ({predict_year})',
        marker=dict(color='#4ade80', size=12, symbol='star')
    ))
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Per Capita Income (USD)", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_coef:
    st.subheader("Model Coefficients")
    st.markdown(f"""
    * **Intercept (c):** `{model.intercept_:.4f}`
    * **Coefficient (m):** `{model.coef_[0]:.4f}`
    * **Regression Equation:**  
      `Income = {model.coef_[0]:.2f} * Year + ({model.intercept_:.2f})`
    """)
    st.write("### Data Table")
    st.dataframe(df.sort_values(by="year", ascending=False), height=250)
