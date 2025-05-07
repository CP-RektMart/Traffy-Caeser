import streamlit as st
import pandas as pd
import pydeck as pdk
from matplotlib import pyplot as plt
from traffy_caeser.data_viz.prepare_viz import prepare_data

st.title("Geospatial Visualization")


@st.cache_data
def load_data_cached():
    return prepare_data()


MAP_STYLES = {
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Road": "mapbox://styles/mapbox/streets-v11",
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
    "Satellite_Streets": "mapbox://styles/mapbox/satellite-streets-v12",
    "Standard": "mapbox://styles/mapbox/standard",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
}

viz_data, clusters_counts, cluster_colors, color_map = load_data_cached()

try:
    # Scatter map
    st.subheader("Problems labeled by clusters")
    layer1 = pdk.Layer(
        "ScatterplotLayer",
        viz_data,
        pickable=True,
        opacity=0.92,
        stroked=False,
        filled=True,
        radius_scale=200,
        get_position="[longitude, latitude]",
        get_fill_color="color",
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style=MAP_STYLES["Road"],
            initial_view_state=pdk.ViewState(
                latitude=viz_data["latitude"].median(),
                longitude=viz_data["longitude"].median(),
                zoom=10,
                pitch=1,
            ),
            layers=[layer1],
        ),
        use_container_width=True,
        height=600,
    )

    # Heatmap
    st.subheader("Problems heatmap")
    layer2 = pdk.Layer(
        "HeatmapLayer",
        viz_data,
        pickable=True,
        opacity=0.6,
        get_position="[longitude, latitude]",
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style=MAP_STYLES["Satellite_Streets"],
            initial_view_state=pdk.ViewState(
                latitude=viz_data["latitude"].median(),
                longitude=viz_data["longitude"].median(),
                zoom=11,
                pitch=0,
            ),
            layers=[layer2],
        ),
        use_container_width=True,
        height=600,
    )

except Exception as e:
    st.error(f"Error in geospatial analysis: {e}")
