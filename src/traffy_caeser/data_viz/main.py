import streamlit as st
from plotly import express as px
from traffy_caeser.data_viz.prepare_viz import prepare_data
import pandas as pd

st.set_page_config(page_title="Traffy Ceaser: Comments Analysis", layout="wide")

st.title("Welcome to Traffy Ceaser Analytics")

st.markdown(
    """
This dashboard contains:
- **Clustering Analysis** (See: `cluster_analysis`)
- **Geospatial Visualizations** (See: `geospatial`)
Use the sidebar to navigate between pages.
"""
)


@st.cache_data
def load_data_cached():
    return prepare_data()


with st.spinner("Loading PCA clustering plot..."):
    viz_data, clusters_counts, cluster_colors, color_map = load_data_cached()

    st.subheader("3D PCA Clustering Plot")
    fig = px.scatter_3d(
        viz_data,
        x="PC1",
        y="PC2",
        z="PC3",
        color="cluster",
        opacity=0.3,
        color_discrete_map=color_map,
        height=600,
        width=800,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="PC1", title_font=dict(size=16), tickfont=dict(size=12)),
            yaxis=dict(title="PC2", title_font=dict(size=16), tickfont=dict(size=12)),
            zaxis=dict(title="PC3", title_font=dict(size=16), tickfont=dict(size=12)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=dict(text="Cluster", font=dict(size=14)),
            font=dict(size=12),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
    )

    fig.update_traces(marker=dict(size=6, opacity=0.4))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Comments by Cluster (5 per cluster)")

    for cluster in sorted(viz_data["cluster"].unique()):
        cluster_comments = viz_data[viz_data["cluster"] == cluster]["comment"]
        n = min(5, len(cluster_comments))
        samples = cluster_comments.sample(n)
        st.markdown(f"**Cluster {cluster} samples:**")
        for c in samples:
            st.markdown(f"- {c}")
