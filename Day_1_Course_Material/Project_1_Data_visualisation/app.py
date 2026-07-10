import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config for a premium dashboard feel
st.set_page_config(
    page_title="Google Play Store Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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

# App Title & Header
st.title("🤖 Google Play Store Analytics Dashboard")
st.markdown("An interactive case study demonstrating data cleaning, outlier analysis, and advanced visualizations.")

# Data Loading & Cleaning function with caching
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Cleaning Rating (drop nulls for analysis or fill with median)
    df = df.dropna(subset=['Rating'])
    
    # 2. Cleaning Size
    def clean_size(size):
        if pd.isna(size):
            return np.nan
        size = str(size).strip().lower()
        if 'varies with device' in size:
            return np.nan
        if 'm' in size:
            return float(size.replace('m', '')) * 1000
        if 'k' in size:
            return float(size.replace('k', ''))
        return float(size)
    df['Size'] = df['Size'].apply(clean_size)
    df['Size'] = df['Size'].fillna(df['Size'].median())
    
    # 3. Cleaning Installs
    def clean_installs(val):
        if pd.isna(val):
            return 0
        val = str(val).replace('+', '').replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return 0
    df['Installs'] = df['Installs'].apply(clean_installs)
    
    # 4. Cleaning Price
    def clean_price(val):
        if pd.isna(val):
            return 0.0
        val = str(val).replace('$', '').strip()
        try:
            return float(val)
        except ValueError:
            return 0.0
    df['Price'] = df['Price'].apply(clean_price)
    
    # Extract month/year/day for reviews if Last Updated is parsed
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    df['Updated Month'] = df['Last Updated'].dt.month
    df['Updated Year'] = df['Last Updated'].dt.year
    
    return df

csv_path = os.path.join(os.path.dirname(__file__), "googleplaystore_v2.csv")
if not os.path.exists(csv_path):
    st.error("Dataset `googleplaystore_v2.csv` not found!")
    st.stop()

df = load_and_clean_data(csv_path)

# Sidebar filters
st.sidebar.header("🔍 Filters")
category_filter = st.sidebar.multiselect(
    "Select App Category",
    options=sorted(df['Category'].unique()),
    default=[]
)
content_rating_filter = st.sidebar.multiselect(
    "Select Content Rating",
    options=sorted(df['Content Rating'].dropna().unique()),
    default=[]
)

# Apply filters
filtered_df = df.copy()
if category_filter:
    filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
if content_rating_filter:
    filtered_df = filtered_df[filtered_df['Content Rating'].isin(content_rating_filter)]

# KPIs / Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(filtered_df):,}</div>
        <div class="metric-lbl">Total Apps</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{filtered_df['Rating'].mean():.2f} ★</div>
        <div class="metric-lbl">Average Rating</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{filtered_df['Installs'].mean():,.0f}</div>
        <div class="metric-lbl">Avg Downloads</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(filtered_df[filtered_df['Price'] > 0]):,}</div>
        <div class="metric-lbl">Paid Apps</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Main tabs for structure
tab1, tab2, tab3 = st.tabs(["📊 Distribution & Outliers", "📈 Correlations & Scatter Plots", "🔍 Raw Data Explorer"])

with tab1:
    st.header("App Size & Rating Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("App Ratings Distribution (Plotly)")
        fig_rating = px.histogram(
            filtered_df, x="Rating", nbins=20,
            title="Histogram of Ratings",
            labels={"Rating": "App Rating"},
            color_discrete_sequence=["#818cf8"],
            marginal="box"
        )
        fig_rating.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9"
        )
        st.plotly_chart(fig_rating, use_container_width=True)
        
    with col2:
        st.subheader("Size vs. Category Boxplot")
        fig_box = px.box(
            filtered_df, x="Category", y="Size",
            title="Distribution of App Sizes (in KB) by Category",
            color="Type",
            color_discrete_map={"Free": "#38bdf8", "Paid": "#fb7185"}
        )
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.header("Correlation & Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating vs. Size Scatter Plot")
        fig_scatter = px.scatter(
            filtered_df, x="Size", y="Rating",
            color="Type", size="Installs",
            title="App Rating vs Size (Size of bubbles denotes Installs)",
            hover_name="App",
            color_discrete_map={"Free": "#38bdf8", "Paid": "#fb7185"},
            opacity=0.6
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        st.subheader("Correlation Heatmap")
        numeric_cols = filtered_df[['Rating', 'Size', 'Installs', 'Price', 'Reviews']]
        # convert Reviews to numeric
        numeric_cols['Reviews'] = pd.to_numeric(numeric_cols['Reviews'], errors='coerce')
        corr = numeric_cols.corr()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", fmt=".2f",
            ax=ax, cbar=False, annot_kws={"size": 10}
        )
        ax.tick_params(colors='#f1f5f9')
        for text in ax.texts:
            text.set_color('#f1f5f9')
            
        st.pyplot(fig)

with tab3:
    st.header("Raw Dataset Viewer")
    st.dataframe(
        filtered_df[['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']],
        use_container_width=True
    )
