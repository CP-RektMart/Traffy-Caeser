from google.cloud import bigquery
from google.oauth2 import service_account
from bs4 import BeautifulSoup
import time
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

PROJECT_ID = st.secrets["PROJECT_ID"]
DATASET = st.secrets["BQ_DATASET"]
TABLE_CLUSTER = st.secrets["BQ_TABLE_CLUSTERED"]


QUERY = f"""
SELECT
  ticket_id AS id,
  comment,
  address,
  photo,
  latitude,
  longitude,
  last_activity,
  cluster,
  PC1,
  PC2,
  PC3
FROM `{PROJECT_ID}.{DATASET}.{TABLE_CLUSTER}`
ORDER BY last_activity DESC
LIMIT 200000
"""

# =====================================================จส100 scraping====================================================================================#
import requests
from bs4 import BeautifulSoup


def scrape_traffic_data():
    url = "https://www.js100.com/en/site/traffic"
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    traffic_list = soup.find("ul", id="latest_traffic_list")
    data = []

    if traffic_list:
        for li in traffic_list.find_all("li"):
            time_text = li.find("h4").get_text(strip=True) if li.find("h4") else ""
            desc_text = li.find("p").get_text(strip=True) if li.find("p") else ""
            if time_text or desc_text:
                data.append({"time": time_text, "description": desc_text})
    return data


def prepare_data():
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    print("Client created")
    start = time.time()
    viz_data = client.query(QUERY).to_dataframe()
    # viz_data = pd.read_csv("data/bq-results-20250508.csv")
    print("Query executed")
    print("Time taken:", time.time() - start)
    print("Dataframe shape:", viz_data.shape)

    clusters_count = viz_data["cluster"].value_counts().reset_index()
    clusters_count.columns = ["cluster", "count"]

    unique_clusters = sorted(viz_data["cluster"].unique())

    # Use tab20 colormap instead of hsv
    colormap = plt.get_cmap("tab10")

    # Generate colors from tab20 colormap for each cluster
    cluster_colors = {}
    for i, cluster in enumerate(unique_clusters):
        # Get color from tab20 and convert to RGBA values in 0-255 range
        rgba = [int(x * 255) for x in colormap(i % 20)]
        cluster_colors[int(cluster)] = rgba

    # Ensure mapping works for both int and str cluster values
    def get_color(c):
        try:
            return cluster_colors[int(c)]
        except Exception:
            return [128, 128, 128, 255]  # fallback gray

    viz_data["color"] = viz_data["cluster"].map(get_color)
    color_map = {
        str(cluster): f"rgba({r},{g},{b},{a/255})"
        for cluster, (r, g, b, a) in cluster_colors.items()
    }

    viz_data["cluster"] = viz_data["cluster"].astype(str)

    scraped_traffic = scrape_traffic_data()
    return viz_data, clusters_count, cluster_colors, color_map, scraped_traffic
