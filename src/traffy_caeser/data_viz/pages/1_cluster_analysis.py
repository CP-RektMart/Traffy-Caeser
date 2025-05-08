import streamlit as st
import pandas as pd
from plotly import express as px
from traffy_caeser.data_viz.prepare_viz import prepare_data

st.title("Clustering Analysis")


@st.cache_data
def load_data_cached():
    return prepare_data()


with st.spinner("Loading Clustering analysis..."):
    viz_data, clusters_counts, cluster_colors, color_map, scraped_traffic = load_data_cached()
    try:
        # pie chart
        st.subheader("Cluster Statistics")
        fig = px.pie(
            clusters_counts,
            names="cluster",
            values="count",
            color="cluster",
            color_discrete_map={
                str(c): f"rgba({r},{g},{b},{a/255})"
                for c, (r, g, b, a) in cluster_colors.items()
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        # Sample comments
        st.subheader("Sample Comments by Cluster")
        html_blocks = """<div style="display: flex; align-items: center; font-weight: bold; background-color: #eee; padding: 8px 6px; border-bottom: 1px solid #ccc;">
            <div style="flex: 0 0 10%; text-align: center;">Cluster</div>
            <div style="flex: 0 0 10%; text-align: center;">Color</div>
            <div style="flex: 1;">Comment</div>
        </div>"""

        for _, row in viz_data.sample(50).iterrows():
            rgba = row["color"]
            rgba_str = f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3] / 255})"
            html_blocks += f"""
            <div style="display: flex; align-items: center; margin: 6px 0; padding: 6px;">
                <div style="flex: 0 0 10%; text-align: center;">{row["cluster"]}</div>
                <div style="flex: 0 0 10%; display: flex; justify-content: center;"><div style="width: 20px; height: 20px; background-color: {rgba_str}; border-radius: 4px;"></div></div>
                <div style="flex: 1;">{row["comment"]}</div>
            </div>"""

        st.markdown(
            f'<div style="max-height: 400px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">{html_blocks}</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error in clustering analysis: {e}")
