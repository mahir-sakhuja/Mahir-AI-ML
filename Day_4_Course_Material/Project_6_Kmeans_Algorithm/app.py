import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris K-Means Clustering", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #1a0533 100%); color: #f3f4f6; }
    h1, h2, h3 { color: #c084fc !important; font-weight: 700 !important; }
    .metric-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 20px; text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-4px); border-color: #9333ea; }
    .metric-val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-lbl { font-size: 0.82rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.1em; }
</style>
""", unsafe_allow_html=True)

st.title("🌸 Iris Flower Clustering — K-Means")
st.markdown("Cluster the classic **Iris dataset** using K-Means and compare clusters against true flower species labels.")

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]
df['species_id'] = iris.target

# Sidebar
st.sidebar.header("⚙️ Clustering Options")
all_features = iris.feature_names
feat_x = st.sidebar.selectbox("X-axis Feature", all_features, index=2)
feat_y = st.sidebar.selectbox("Y-axis Feature", all_features, index=3)
k = st.sidebar.slider("Number of Clusters (K)", 1, 8, 3)
apply_scaling = st.sidebar.checkbox("Apply MinMax Scaling", value=False)

# Cluster
features = df[[feat_x, feat_y]].copy()
if apply_scaling:
    scaler = MinMaxScaler()
    features_fit = scaler.fit_transform(features)
else:
    features_fit = features.values

km = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(features_fit).astype(str)
centers_raw = km.cluster_centers_
if apply_scaling:
    centers_raw = scaler.inverse_transform(centers_raw)

# KPIs
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip([c1, c2, c3, c4],
    [150, 3, k, "✅" if apply_scaling else "❌"],
    ["Total Samples", "True Species", "K Clusters", "Scaling"]):
    col.markdown(f"""<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

st.write("---")

tab1, tab2, tab3 = st.tabs(["🔵 K-Means Clusters", "🌿 True Species Labels", "📈 Elbow Plot"])

COLORS = ['#38bdf8', '#f472b6', '#34d399', '#fb923c', '#a78bfa', '#fbbf24', '#f87171', '#818cf8']

with tab1:
    st.header(f"K-Means Clusters (K={k})")
    fig = px.scatter(
        df, x=feat_x, y=feat_y, color="Cluster",
        color_discrete_sequence=COLORS,
        title=f"K-Means with K={k} | {feat_x} vs {feat_y}",
        hover_data=['species']
    )
    for i, c in enumerate(centers_raw):
        fig.add_trace(go.Scatter(
            x=[c[0]], y=[c[1]], mode='markers',
            marker=dict(symbol='star', size=22, color=COLORS[i % len(COLORS)],
                       line=dict(width=2, color='white')),
            name=f"Centroid {i}", showlegend=True
        ))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#f3f4f6", height=480)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("True Iris Species Labels")
    fig2 = px.scatter(
        df, x=feat_x, y=feat_y, color="species",
        color_discrete_sequence=['#38bdf8', '#f472b6', '#34d399'],
        title=f"True Species | {feat_x} vs {feat_y}",
        symbol='species'
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font_color="#f3f4f6", height=480)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Elbow Plot — Finding Optimal K")
    st.info("Look for the 'elbow' where adding more clusters stops significantly reducing error.")

    @st.cache_data
    def get_elbow(scale):
        feat_data = df[[feat_x, feat_y]].copy()
        if scale:
            s = MinMaxScaler()
            feat_data = pd.DataFrame(s.fit_transform(feat_data), columns=feat_data.columns)
        sse = []
        for ki in range(1, 10):
            km_e = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km_e.fit(feat_data)
            sse.append(km_e.inertia_)
        return sse

    sse = get_elbow(apply_scaling)
    fig3 = go.Figure(go.Scatter(
        x=list(range(1, 10)), y=sse,
        mode='lines+markers',
        line=dict(color='#c084fc', width=3),
        marker=dict(size=10, color='#38bdf8', line=dict(color='white', width=2))
    ))
    fig3.update_layout(
        title="Elbow Method: K vs Inertia",
        xaxis_title="K", yaxis_title="Inertia (SSE)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6", height=380
    )
    st.plotly_chart(fig3, use_container_width=True)
