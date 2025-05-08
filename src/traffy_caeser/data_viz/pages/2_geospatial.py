import streamlit as st
import pandas as pd
import pydeck as pdk
import random
from fastkml import KML
from traffy_caeser.data_viz.prepare_viz import prepare_data

st.title("Geospatial Visualization")


@st.cache_data
def load_data_cached():
    return prepare_data()


viz_data, clusters_counts, cluster_colors, color_map, scraped_traffic = (
    load_data_cached()
)


MAP_STYLES = {
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Road": "mapbox://styles/mapbox/streets-v11",
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
    "Satellite_Streets": "mapbox://styles/mapbox/satellite-streets-v12",
    "Standard": "mapbox://styles/mapbox/standard",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
}

MAX_POINTS = 100_000
st.sidebar.subheader("Filter Clusters")
unique_clusters = sorted([int(c) for c in viz_data["cluster"].unique()])
selected_clusters = []
for c in unique_clusters:
    if st.sidebar.checkbox(f"Cluster {c}", value=True, key=f"cluster_{c}"):
        selected_clusters.append(c)


filtered_viz = viz_data[viz_data["cluster"].isin(selected_clusters)]


default_lat = viz_data["latitude"].median()
default_lon = viz_data["longitude"].median()

if filtered_viz.empty:
    st.warning("No clusters selected — showing all clusters.")
    filtered = viz_data
    view_lat, view_lon = default_lat, default_lon
else:
    filtered = filtered_viz
    view_lat = filtered["latitude"].median()
    view_lon = filtered["longitude"].median()

n_pts = len(filtered)
if n_pts > MAX_POINTS:
    st.info(f"Sampling {MAX_POINTS:,} of {n_pts:,} points for performance.")
    filtered = filtered.sample(MAX_POINTS, random_state=42)

filtered["short_comment"] = filtered["comment"].str.slice(0, 200).fillna("") + "…"
viz_cols = [
    "id",
    "latitude",
    "longitude",
    "cluster",
    "color",
    "short_comment",
    "photo",
    "address",
    "color",
]
filtered = filtered[viz_cols]

kml_path = "data/base_kml.kml"

k = KML.parse(kml_path)
features = list(k.features)
placemarks = list(features[0].features)

data = []
for pm in placemarks:
    geom = pm.geometry

    # randomize color
    colorRand = [random.randint(150, 255) for _ in range(4)]

    coords = list(geom.exterior.coords)
    data.append(
        {
            "name": pm.name,
            "location": coords,
            "color": colorRand,
        }
    )

poly_district = pdk.Layer(
    "PolygonLayer",
    data=data,
    get_polygon="location",
    get_fill_color="color",  # RGBA
    opacity=0.05,
    pickable=False,
    extruded=False,
    filled=True,
    line_width_min_pixels=1,
)

scatter = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    pickable=True,
    autoHighlight=True,
    stroked=False,
    filled=True,
    opacity=0.7,
    radius_min_pixels=2,
    radius_max_pixels=6,
    get_position=["longitude", "latitude"],
    get_fill_color="color",
    minZoom=100,
)

tooltip = {
    "html": """
      <div style="
        max-width: 320px;
        white-space: normal;
        overflow-wrap: anywhere;
        font-size: 12px;
        color: white;
      ">
        <!-- fixed-size image -->
        <img
          src="{photo}"
          style="
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 6px;
          "
          alt="photo"
        />
        <b>ID:</b> {id}<br/>
        <b>Cluster:</b> {cluster}<br/>
        <b>Comment:</b> {short_comment}
      </div>
    """,
    "style": {"backgroundColor": "rgba(0, 0, 0, 0.8)", "padding": "8px"},
}


deck = pdk.Deck(
    map_style=MAP_STYLES["Road"],
    initial_view_state=pdk.ViewState(
        latitude=view_lat,
        longitude=view_lon,
        zoom=10,
        pitch=0,
    ),
    layers=[scatter, poly_district],
    tooltip=tooltip,
)

st.pydeck_chart(deck, use_container_width=True, height=600)

st.write(
    f"Showing {len(filtered):,} points from {len(viz_data):,} total points. "
    f"Showing {len(selected_clusters)} clusters out of {len(unique_clusters)}."
)

scraped_traffic = pd.DataFrame(scraped_traffic)
st.subheader("Latest Traffic Incidents (Scraped from จส.100)")

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
    <div style="flex: 0 0 10%; text-align: center;">Time</div>
    <div style="flex: 1;">Description</div>
</div>"""

for _, row in scraped_traffic.iterrows():
    html_blocks += f"""
    <div style="display: flex; align-items: center; margin: 6px 0; padding: 6px; background-color: {row_bg_color}; color: {row_text_color}; border-bottom: 1px solid {border_color};">
        <div style="flex: 0 0 20%; text-align: center;">{row["time"]}</div>
        <div style="flex: 1;">{row["description"]}</div>
    </div>"""

st.markdown(
    f'<div style="max-height: 400px; overflow-y: auto; padding: 10px; border: 1px solid {border_color}; border-radius: 8px; background-color: {container_bg_color};">{html_blocks}</div>',
    unsafe_allow_html=True,
)
