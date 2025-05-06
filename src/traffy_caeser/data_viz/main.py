import streamlit as st
import pandas as pd
import pydeck as pdk
from matplotlib import pyplot as plt
from plotly import express as px

from traffy_caeser.data_viz.prepare_viz import prepare_data


st.set_page_config(page_title="Traffy Ceaser: Comments analysis", layout="wide")
st.title("Comments Analysis on Traffy Fondue")


@st.cache_data
def load_data_cached():
    df = prepare_data()
    return df


MAP_STYLES = {
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Road": "mapbox://styles/mapbox/streets-v11",
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
}

df = load_data_cached()

if df.empty:
    st.error(
        "Failed to load data. Please check the console for errors from 'prepare_data'."
    )

filtered_data = df[["latitude", "longitude", "cluster", "comments"]]
try:
    # Perform DBSCAN clustering
    coords = filtered_data[["latitude", "longitude"]]
    eps_degrees = 0.002
    min_samples = 3

    # Analyze clusters
    clusters_count = filtered_data["cluster"].value_counts()
    clusters_count = clusters_count[clusters_count.index != -1]  # Exclude noise points
    top_clusters = clusters_count.nlargest(10)
    # Generate colors for clusters
    unique_clusters = filtered_data[filtered_data["cluster"].isin(top_clusters.index)][
        "cluster"
    ].unique()
    colormap = plt.get_cmap("hsv")
    cluster_colors = {
        cluster: [int(x * 255) for x in colormap(i / len(unique_clusters))[:3]] + [160]
        for i, cluster in enumerate(unique_clusters)
    }

    # Create visualization dataframe
    viz_data = filtered_data[filtered_data["cluster"].isin(top_clusters.index)].copy()
    viz_data["color"] = viz_data["cluster"].map(
        lambda c: (
            cluster_colors[c]
            if isinstance(cluster_colors[c], list)
            else list(cluster_colors[c])
        )
    )
    st.write(" sample:", viz_data.head())

    st.write("Draw a scatter map for clusters here", viz_data["color"])

    # Create cluster layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        viz_data,
        pickable=True,
        opacity=0.8,
        stroked=False,
        filled=True,
        radius_scale=50,
        line_width_min_pixels=1,
        get_position="[longitude, latitude]",
        get_fill_color="color",
    )
    # Create and display the map
    st.pydeck_chart(
        pdk.Deck(
            map_style=MAP_STYLES["Light"],
            initial_view_state=pdk.ViewState(
                latitude=viz_data["latitude"].mean(),
                longitude=viz_data["longitude"].mean(),
                zoom=9,
                pitch=0,
            ),
            layers=[layer],
        )
    )

    st.write("Draw a heatmap for clusters here")

    # Create heatmap layer
    layer = pdk.Layer(
        "HeatmapLayer",
        viz_data,
        pickable=True,
        opacity=0.8,
        stroked=False,
        filled=True,
        radius_scale=100,
        line_width_min_pixels=1,
        get_position="[longitude, latitude]",
    )
    # Create and display the map
    st.pydeck_chart(
        pdk.Deck(
            map_style=MAP_STYLES["Dark"],
            initial_view_state=pdk.ViewState(
                latitude=viz_data["latitude"].mean(),
                longitude=viz_data["longitude"].mean(),
                zoom=9,
                pitch=0,
            ),
            layers=[layer],
        )
    )

    st.write("Draw a hexagon map for clusters here")

    # Create hexagon layer
    layer = pdk.Layer(
        "HexagonLayer",
        viz_data,
        radius=1000,
        opacity=0.4,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        line_width_min_pixels=2,
    )
    # Create and display the map
    st.pydeck_chart(
        pdk.Deck(
            map_style=MAP_STYLES["Dark"],
            initial_view_state=pdk.ViewState(
                latitude=viz_data["latitude"].mean(),
                longitude=viz_data["longitude"].mean(),
                zoom=9,
                pitch=1,
            ),
            layers=[layer],
        )
    )

    # Cluster Analysis
    st.subheader("Cluster Statistics")

except Exception as e:
    st.error(f"Error in clustering analysis: {e}")


# # Price by neighborhood
# price_by_neighborhood = (
#     filtered_data.groupby("neighbourhood")["price"].agg(["mean", "count"]).reset_index()
# )
# price_by_neighborhood.columns = ["neighbourhood", "avg_price", "listings_count"]

# fig_scatter = px.scatter(
#     price_by_neighborhood,
#     x="listings_count",
#     y="avg_price",
#     text="neighbourhood",
#     title="Average Price vs Number of Listings by Neighborhood",
#     labels={"listings_count": "Number of Listings", "avg_price": "Average Price (THB)"},
# )
# fig_scatter.update_traces(textposition="top center")
# st.plotly_chart(fig_scatter)
