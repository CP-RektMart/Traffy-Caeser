import os
from google.cloud import bigquery
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import time

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("BQ_DATASET")
TABLE_RAW = os.getenv("BQ_TABLE_RAW")
TABLE_CLUSTER = os.getenv("BQ_TABLE_CLUSTERED")

QUERY = f"""
WITH coords_extracted AS (
  SELECT
    cr.ticket_id                      AS id,
    COALESCE(cr.comment, tr.comment) AS comments,
    SAFE_CAST(TRIM(SPLIT(tr.coords, ',')[OFFSET(1)]) AS FLOAT64) AS latitude,
    SAFE_CAST(TRIM(SPLIT(tr.coords, ',')[OFFSET(0)]) AS FLOAT64) AS longitude,
    cr.vector,
    cr.cluster,
    cr.PC1,
    cr.PC2,
    cr.PC3
  FROM `{PROJECT_ID}.{DATASET}.{TABLE_CLUSTER}` AS cr
  LEFT JOIN `{PROJECT_ID}.{DATASET}.{TABLE_RAW}` AS tr
  ON cr.ticket_id = tr.ticket_id
)
SELECT *
FROM coords_extracted
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
LIMIT 1000
"""


def prepare_data(input_df=None):
    print(PROJECT_ID, DATASET, TABLE_RAW, TABLE_CLUSTER)
    client = bigquery.Client(project=PROJECT_ID)
    print("Client created")
    df = client.query(QUERY).to_dataframe()
    print("Query executed")
    # Current columns:
    # ['id', 'comments', 'latitude', 'longitude', 'vector', 'cluster', 'PC1', 'PC2', 'PC3']
    return df
