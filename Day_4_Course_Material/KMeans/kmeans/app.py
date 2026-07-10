import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="K-Means Clustering", page_icon="🔵", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #0f2027 100%); color: #f3f4f6; }
    h1, h2, h3 { color: #a78bfa !important; font-weight: 700 !important; }
    .metric-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 20px; text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-4px); border-color: #7c3aed; }
    .metric-val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-lbl { font-size: 0.82rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.1em; }
</style>
""", unsafe_allow_html=True)

st.title("🔵 K-Means Clustering — Age vs. Income")
st.markdown("Discover natural customer segments by clustering **Age** and **Income** data. Toggle MinMax scaling to see how it improves cluster quality.")

# Load data
csv_path = os.path.join(os.path.dirname(__file__), "income.csv")
if not os.path.exists(csv_path):
    st.error("Dataset `income.csv` not found!")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(csv_path)

# Sidebar
st.sidebar.header("⚙️ Clustering Controls")
k = st.sidebar.slider("Number of Clusters (K)", min_value=1, max_value=8, value=3)
apply_scaling = st.sidebar.checkbox("Apply MinMax Scaling", value=False,
    help="Scaling normalizes Age and Income to [0,1], preventing Income from dominating the clusters.")

# Cluster data
def cluster_data(data, n_clusters, scale):
    features = data[['Age', 'Income($)']].copy()
    if scale:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(features_scaled)
        centers_scaled = km.cluster_centers_
        centers = scaler.inverse_transform(centers_scaled)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(features)
        centers = km.cluster_centers_
    return km.labels_, centers

labels, centers = cluster_data(df, k, apply_scaling)
df_plot = df.copy()
df_plot['Cluster'] = labels.astype(str)

# KPIs
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-lbl">Data Points</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-val">{k}</div><div class="metric-lbl">Clusters (K)</div></div>""", unsafe_allow_html=True)
with c3:
    status = "✅ Applied" if apply_scaling else "❌ Not Applied"
    st.markdown(f"""<div class="metric-card"><div class="metric-val" style="font-size:1.4rem">{status}</div><div class="metric-lbl">MinMax Scaling</div></div>""", unsafe_allow_html=True)

st.write("---")

tab1, tab2 = st.tabs(["🔵 Cluster Plot", "📈 Elbow Plot"])

with tab1:
    st.header("Clustered Age vs. Income")
    color_seq = ['#38bdf8', '#f472b6', '#34d399', '#fb923c', '#a78bfa', '#fbbf24', '#f87171', '#818cf8']
    
    fig = px.scatter(
        df_plot, x="Age", y="Income($)",
        color="Cluster",
        text="Name",
        color_discrete_sequence=color_seq,
        title=f"K-Means Clustering with K={k} {'(Scaled)' if apply_scaling else '(Unscaled)'}",
        labels={"Income($)": "Income ($)"},
    )
    
    # Add centroids
    for i, center in enumerate(centers):
        fig.add_trace(go.Scatter(
            x=[center[0]], y=[center[1]],
            mode='markers',
            marker=dict(symbol='star', size=20, color=color_seq[i % len(color_seq)], 
                       line=dict(width=2, color='white')),
            name=f"Centroid {i}",
            showlegend=True
        ))
    
    fig.update_traces(textposition='top center', marker_size=10)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Elbow Plot — Finding Optimal K")
    st.info("The **elbow point** is where adding more clusters stops significantly reducing error. That's your optimal K.")
    
    @st.cache_data
    def compute_elbow(data, scale):
        features = data[['Age', 'Income($)']].copy()
        if scale:
            scaler = MinMaxScaler()
            features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        sse = []
        k_range = range(1, 10)
        for ki in k_range:
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(features)
            sse.append(km.inertia_)
        return list(k_range), sse
    
    k_vals, sse_vals = compute_elbow(df, apply_scaling)
    
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_vals, y=sse_vals, mode='lines+markers',
        line=dict(color='#a78bfa', width=3),
        marker=dict(size=10, color='#38bdf8', line=dict(color='white', width=2)),
        name="SSE (Inertia)"
    ))
    fig_elbow.update_layout(
        title="Elbow Method: K vs Sum of Squared Errors",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Sum of Squared Errors (Inertia)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6", height=400
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

st.write("---")
st.subheader("📋 Data with Cluster Labels")
st.dataframe(df_plot[['Name', 'Age', 'Income($)', 'Cluster']], use_container_width=True)
