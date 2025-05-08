import streamlit as st
from plotly import express as px
from traffy_caeser.data_viz.prepare_viz import prepare_data
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Traffy Ceaser: Comments Analysis", layout="wide")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Welcome to Traffy Ceaser Analytics")

    st.markdown(
        """
    ## About Traffy Ceaser
    Traffy Ceaser is an advanced analytics system designed to process and analyze public comments and feedback. 
    The system helps identify patterns, trends, and insights from user-submitted data to improve public services and urban management.
    
    ## Dashboard Overview
    This interactive dashboard provides visualization tools to explore the processed data:
    - **Clustering Analysis**
    - **Geospatial Visualizations**
    
    Use the sidebar to navigate between different analysis pages and explore the data in depth.
    """
    )

with col2:
    # Display an image (assumed to be in an assets folder)
    image_path = os.path.join(os.path.dirname(__file__), "assets", "traffy_logo.png")
    print(image_path)
    # Fallback to a URL if local image doesn't exist
    try:
        if os.path.exists(image_path):
            st.image(image_path, width=200)
    except Exception as e:
        st.write("Could not load image.")

st.markdown("---")

st.markdown(
    """
## Available Analysis Tools
- **Clustering Analysis** (See: `cluster_analysis`)
    - Bar chart of cluster sizes
    - 3D PCA plot of clusters
    - Cluster samples and summary
    
- **Geospatial Visualizations** (See: `geospatial`)
    - Geographic distribution of clusters
    - Realtime traffic incidents reporting 

Navigate using the sidebar to explore each analysis in detail.
"""
)
