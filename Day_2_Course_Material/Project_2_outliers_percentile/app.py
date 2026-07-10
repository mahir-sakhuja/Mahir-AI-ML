import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config for a premium analytics dashboard feel
st.set_page_config(
    page_title="Airbnb NYC Outliers Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #f3f4f6;
    }
    h1, h2, h3 {
        color: #818cf8 !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
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
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-val-outliers {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fb7185;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("🏠 Airbnb NYC Outlier Analysis & Filtering")
st.markdown("Explore and filter price outliers from the 2019 New York City Airbnb dataset using the **Percentile Method**.")

# Data Loading function with caching
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

csv_path = os.path.join(os.path.dirname(__file__), "AB_NYC_2019.csv")
if not os.path.exists(csv_path):
    st.error("Dataset `AB_NYC_2019.csv` not found!")
    st.stop()

df_raw = load_data(csv_path)

# Sidebar filters
st.sidebar.header("⚙️ Percentile Thresholds")
lower_pct = st.sidebar.slider("Lower Percentile Threshold", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.1f%%")
upper_pct = st.sidebar.slider("Upper Percentile Threshold", min_value=95.0, max_value=100.0, value=99.9, step=0.1, format="%.1f%%")

st.sidebar.write("---")
st.sidebar.header("🔍 Location Filter")
group_filter = st.sidebar.multiselect(
    "Select Neighbourhood Groups",
    options=sorted(df_raw['neighbourhood_group'].dropna().unique()),
    default=[]
)

# Apply location filter first
df = df_raw.copy()
if group_filter:
    df = df[df['neighbourhood_group'].isin(group_filter)]

# Calculate thresholds
min_threshold = df['price'].quantile(lower_pct / 100.0)
max_threshold = df['price'].quantile(upper_pct / 100.0)

# Filter datasets
outliers_lower = df[df['price'] < min_threshold]
outliers_upper = df[df['price'] > max_threshold]
total_outliers = len(outliers_lower) + len(outliers_upper)

df_cleaned = df[(df['price'] >= min_threshold) & (df['price'] <= max_threshold)]

# KPIs Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(df):,}</div>
        <div class="metric-lbl">Total Listings</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val-outliers">{total_outliers:,}</div>
        <div class="metric-lbl">Outliers Removed</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">${min_threshold:.1f}</div>
        <div class="metric-lbl">Min Price Limit ({lower_pct}%)</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">${max_threshold:,.1f}</div>
        <div class="metric-lbl">Max Price Limit ({upper_pct}%)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

tab1, tab2, tab3 = st.tabs(["📊 Price Distribution", "🗺️ Geographic Map", "🔍 Outliers Explorer"])

with tab1:
    st.header("Price Distributions Before & After Outlier Removal")
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.subheader("Distribution (Raw Prices)")
        fig_before = px.histogram(
            df, x="price", nbins=50,
            title="Raw Listings (Includes extreme outliers)",
            labels={"price": "Price ($)"},
            color_discrete_sequence=["#fb7185"],
            marginal="box"
        )
        fig_before.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f3f4f6"
        )
        st.plotly_chart(fig_before, use_container_width=True)
        
    with col_after:
        st.subheader("Distribution (Cleaned Prices)")
        fig_after = px.histogram(
            df_cleaned, x="price", nbins=50,
            title="Cleaned Listings (Outliers filtered)",
            labels={"price": "Price ($)"},
            color_discrete_sequence=["#38bdf8"],
            marginal="box"
        )
        fig_after.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f3f4f6"
        )
        st.plotly_chart(fig_after, use_container_width=True)

with tab2:
    st.header("Geographic Scatter Map of Airbnb Price Density")
    st.write("Displaying listings colored by price to show where high-cost locations are clustered.")
    
    # Map visualization using Scattermapbox or regular scatter chart
    # Let's use simple Scattermapbox / Mapbox or simple scatter plot of lat/lon if mapbox token isn't present
    # scatter_mapbox is built-in and works out-of-the-box!
    fig_map = px.scatter_mapbox(
        df_cleaned.sample(min(len(df_cleaned), 8000), random_state=42), 
        lat="latitude", lon="longitude",
        color="price", size="price",
        color_continuous_scale="Viridis",
        size_max=8, zoom=10,
        mapbox_style="carto-darkmatter",
        hover_name="name",
        hover_data=["neighbourhood_group", "price", "room_type"],
        title="Sampled Listings Map (Colored & Sized by Price)"
    )
    fig_map.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
        height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.header("Outlier Records Explorer")
    st.subheader(f"Listings below {lower_pct}% (${min_threshold:.1f}) and above {upper_pct}% (${max_threshold:,.1f})")
    
    outliers_df = pd.concat([outliers_lower, outliers_upper])
    if not outliers_df.empty:
        st.dataframe(
            outliers_df[['name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price', 'minimum_nights', 'number_of_reviews']],
            use_container_width=True
        )
    else:
        st.info("No outliers found with the current configuration.")
