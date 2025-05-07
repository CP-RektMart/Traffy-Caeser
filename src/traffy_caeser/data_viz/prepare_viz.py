import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt

load_dotenv(override=True)

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("BQ_DATASET")
TABLE_CLUSTER = os.getenv("BQ_TABLE_CLUSTERED")

QUERY = f"""
SELECT
  ticket_id AS id,
  comment,
  latitude,
  longitude,
  cluster,
  PC1,
  PC2,
  PC3
FROM `{PROJECT_ID}.{DATASET}.{TABLE_CLUSTER}`
LIMIT 1000
"""


def prepare_data():
    print(PROJECT_ID, DATASET, TABLE_CLUSTER)
    client = bigquery.Client(project=PROJECT_ID)
    print("Client created")
    start = time.time()
    viz_data = client.query(QUERY).to_dataframe()
    print("Query executed")
    print("Time taken:", time.time() - start)
    print("Dataframe shape:", viz_data.shape)

    clusters_count = viz_data["cluster"].value_counts().reset_index()
    clusters_count.columns = ["cluster", "count"]

    unique_clusters = sorted(viz_data["cluster"].unique())
    colormap = plt.get_cmap("hsv")
    cluster_colors = {
        cluster: [int(x * 255) for x in colormap(i / len(unique_clusters))[:3]] + [160]
        for i, cluster in enumerate(unique_clusters)
    }

    viz_data["color"] = viz_data["cluster"].map(lambda c: cluster_colors[c])
    color_map = {
        cluster: f"rgba({r},{g},{b},{a/255})"
        for cluster, (r, g, b, a) in cluster_colors.items()
    }
    return viz_data, clusters_count, cluster_colors, color_map
