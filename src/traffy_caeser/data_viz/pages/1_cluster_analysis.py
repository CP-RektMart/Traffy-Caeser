import streamlit as st
import pandas as pd
from plotly import express as px
from traffy_caeser.data_viz.prepare_viz import prepare_data
import plotly.graph_objects as go
from traffy_caeser.data_viz.vertex_ai_utils import generate_cluster_summary

st.title("Clustering Analysis")


@st.cache_data
def load_data_cached():
    return prepare_data()


with st.spinner("Loading Clustering analysis..."):
    viz_data, clusters_counts, cluster_colors, color_map, scraped_traffic = (
        load_data_cached()
    )
    try:
        # bar chart sorted by number of members in cluster
        st.subheader("Cluster Statistics")
        sorted_counts = clusters_counts.sort_values("count", ascending=False)

        # Build color list for bars matching cluster_colors
        def rgb_to_hex(rgb):
            return "#%02x%02x%02x" % tuple(rgb[:3])

        bar_colors = [
            (
                rgb_to_hex(cluster_colors[int(cluster)])
                if int(cluster) in cluster_colors
                else "#888888"
            )
            for cluster in sorted_counts["cluster"]
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_counts["cluster"],
                    y=sorted_counts["count"],
                    marker_color=bar_colors,
                    text=sorted_counts["count"],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            xaxis_title="Cluster", yaxis_title="Number of Members", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3D PCA Clustering Plot
        st.subheader("3D PCA Clustering Plot")

        viz_data["cluster"] = viz_data["cluster"].astype(str)

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
                xaxis=dict(
                    title="PC1", title_font=dict(size=16), tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="PC2", title_font=dict(size=16), tickfont=dict(size=12)
                ),
                zaxis=dict(
                    title="PC3", title_font=dict(size=16), tickfont=dict(size=12)
                ),
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

        # Sample comments
        st.subheader("Sample Comments by Cluster")

        # Detect current theme
        theme = st.get_option("theme.base")
        if theme == "dark":
            header_bg_color = "#262730"  # Darker background for header in dark mode
            header_text_color = "#FAFAFA"
            row_bg_color = "#1E1E1E"  # Slightly lighter than header for rows
            row_text_color = "#FAFAFA"
            border_color = "#444"
            container_bg_color = "#0E1117"  # Streamlit's default dark background
        else:
            header_bg_color = "#eee"
            header_text_color = "#333"
            row_bg_color = "#f9f9f9"
            row_text_color = "#333"
            border_color = "#ccc"
            container_bg_color = "#f9f9f9"

        html_blocks = f"""<div style="display: flex; align-items: center; font-weight: bold; background-color: {header_bg_color}; color: {header_text_color}; padding: 8px 6px; border-bottom: 1px solid {border_color};">
            <div style="flex: 0 0 8%; text-align: center;">Cluster</div>
            <div style="flex: 0 0 8%; text-align: center;">Color</div>
            <div style="flex: 0 0 34%; text-align: center;">Cluster Summary</div>
            <div style="flex: 1;">Comment Samples (up to 3 per cluster)</div>
        </div>"""

        # Group by cluster and take a few samples from each
        for cluster_id, group_df in viz_data.groupby("cluster"):
            # Get the color for the current cluster
            if int(cluster_id) in cluster_colors:
                r, g, b, a_val = cluster_colors[int(cluster_id)]
                rgba_str = f"rgba({r}, {g}, {b}, {a_val / 255})"
            else:  # Fallback if color not in cluster_colors, though ideally it should be
                rgba_val = group_df["color"].iloc[0]
                rgba_str = f"rgba({rgba_val[0]}, {rgba_val[1]}, {rgba_val[2]}, {rgba_val[3] / 255})"

            # Get sample comments for this cluster
            comment_col = (
                "comment" if "comment" in group_df.columns else group_df.columns[0]
            )

            sample_size = min(len(group_df), 10)
            sample_comments = group_df.sample(sample_size, random_state=42)[
                comment_col
            ].tolist()

            # Generate summary for this cluster
            cluster_summary = generate_cluster_summary(sample_comments)

            # Display cluster header
            html_blocks += f"""
            <div style="display: flex; align-items: center; margin: 6px 0; padding: 6px; background-color: {row_bg_color}; color: {row_text_color}; border-bottom: 1px solid {border_color};">
            <div style="flex: 0 0 8%; text-align: center; font-weight: bold;">{cluster_id}</div>
            <div style="flex: 0 0 8%; display: flex; justify-content: center;"><div style="width: 20px; height: 20px; background-color: {rgba_str}; border-radius: 4px;"></div></div>
            <div style="flex: 0 0 34%; padding: 0 10px;">{cluster_summary}</div>
            <div style="flex: 1;">"""

            # Display up to 3 sample comments for this cluster
            for _, row in group_df.sample(
                min(len(group_df), 3), random_state=42
            ).iterrows():
                comment_text = row.get(comment_col, "(No comment available)")
                html_blocks += f"""<div style="padding: 2px 0; color: {row_text_color};">{"- " + comment_text}</div>"""

            html_blocks += """</div></div>"""

        st.markdown(
            f'<div style="max-height: 600px; overflow-y: auto; padding: 10px; border: 1px solid {border_color}; border-radius: 8px; background-color: {container_bg_color};">{html_blocks}</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error in clustering analysis: {e}")
