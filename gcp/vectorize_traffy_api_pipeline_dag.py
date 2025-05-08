from __future__ import annotations

import pendulum
import requests

from google.cloud import storage
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateBatchOperator,
)
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

# Variables
PROJECT_ID = "gen-lang-client-0538587318"
GCS_BUCKET = "dsde"
API_ENDPOINT = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
BIGQUERY_DATASET = "bangkok_traffy"
BIGQUERY_TABLE = "traffy_raw"
BIGQUERY_VECTOR_TABLE = "traffy_vector"
SPARK_CODE_GCS_PATH = f"gs://{GCS_BUCKET}/transform_traffy_data.py"
DATAPROC_REGION = "us-west1"
BIGQUERY_REGION = "us-west1"

# Define GCS paths based on execution date
# {{ ds }} is Airflow's execution date in YYYY-MM-DD format
# We want to process data for the day before the execution date.
GCS_RAW_PREFIX = f"raw"
GCS_PROCESSED_PREFIX = f"processed"
GCS_RAW_PATH = f"gs://{GCS_BUCKET}/{GCS_RAW_PREFIX}/{{{{ ds }}}}/{{{{ ds }}}}_data.csv"
GCS_PROCESSED_PATH = (
    f"gs://{GCS_BUCKET}/{GCS_PROCESSED_PREFIX}/{{{{ ds }}}}/transformed_data"
)

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
    "on_failure_callback": None,
    "is_paused_upon_creation": False,
}

# Dataproc Serverless Batch job configuration
SERVERLESS_BATCH_CONFIG = {
    "pyspark_batch": {
        "main_python_file_uri": SPARK_CODE_GCS_PATH,
        "args": [
            GCS_RAW_PATH,
            GCS_PROCESSED_PATH,
        ],
    },
    "runtime_config": {
        "version": "2.2",
    },
}

BATCH_ID = f"traffy-transform-batch-{{{{ dag.dag_id.lower().replace('_', '-') }}}}-{{{{ ds_nodash }}}}-{{{{ ti.try_number }}}}"


def upload_to_gcs(bucket_name, destination_blob_name, source_string):
    """Uploads a string to a GCS blob."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(source_string)
    print(f"Uploaded data to gs://{bucket_name}/{destination_blob_name}")


def fetch_api_and_upload(gcs_bucket, gcs_prefix, api_url, query_params_base, **kwargs):
    # Prepare query parameters
    data_date_str = kwargs["ds"]

    final_query_params = query_params_base.copy() if query_params_base else {}
    final_query_params["start"] = data_date_str
    final_query_params["end"] = data_date_str

    response = requests.get(api_url, params=final_query_params)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

    csv_data = response.text  # API returns CSV text

    # Determine GCS path using data_date_str (calculated as execution_date - 1 day)
    gcs_path = f"{gcs_prefix}/{data_date_str}/{data_date_str}_data.csv"

    upload_to_gcs(gcs_bucket, gcs_path, csv_data)


# --- Python Callable for Vectorization ---
# This function will run directly on an Airflow worker.
# ALL its dependencies MUST be installed in the Composer environment.
def vectorize_data_directly_in_composer(
    project_id, dataset_id, raw_table, vector_table, execution_date, **kwargs
):
    """
    Reads data from BQ, vectorizes using Hugging Face models, and writes to BQ.
    Designed to run directly within the Airflow worker environment.
    """
    print(f"Starting in-composer vectorization for date: {execution_date}")
    print(f"Reading from: {project_id}.{dataset_id}.{raw_table}")
    print(f"Writing to: {project_id}.{dataset_id}.{vector_table}")

    try:
        from google.cloud import bigquery
        import pandas as pd
        import torch
        import requests
        from PIL import Image
        from io import BytesIO
        from transformers import (
            AutoTokenizer,
            AutoModel,
            SiglipVisionModel,
            SiglipProcessor,
        )
        from sklearn.preprocessing import normalize
        from functools import lru_cache
        import numpy as np  # Needed for hstack and normalize

        # Set device - will be 'cpu' as Composer workers are CPU-only
        device = "cpu"
        print(f"Using device for vectorization: {device}")

        # --- Replicate Model Loading ---
        # Cache models to avoid reloading if the task retries
        @lru_cache(maxsize=1)  # Cache size 1 is enough for singleton models
        def load_models():
            print("Loading ML models...")
            # Ensure you installed torch-cpu, not torch+cuda
            text_model_loaded = AutoModel.from_pretrained("BAAI/bge-m3").to(device)
            text_tokenizer_loaded = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            image_model_loaded = (
                SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
                .eval()
                .to(device)
            )
            image_processor_loaded = SiglipProcessor.from_pretrained(
                "google/siglip-so400m-patch14-384"
            )
            print("Models loaded.")
            return (
                text_model_loaded,
                text_tokenizer_loaded,
                image_model_loaded,
                image_processor_loaded,
            )

        text_model, text_tokenizer, image_model, image_processor = load_models()

        # --- Replicate Image Fetching Helpers ---
        _session = requests.Session()
        _adapter = requests.adapters.HTTPAdapter(
            pool_connections=64, pool_maxsize=64
        )  # Adjust pool size based on expected concurrency/worker threads
        _session.mount("http://", _adapter)
        _session.mount("https://", _adapter)

        @lru_cache(maxsize=4096)
        def _download_bytes(url: str) -> bytes:
            if not url:
                return b""
            try:
                resp = _session.get(url, timeout=10)
                resp.raise_for_status()
                return resp.content
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
                return b""

        # --- Replicate Embedding Functions ---
        def get_text_embeddings(texts: list[str]):
            texts = [t if t is not None else "" for t in texts]
            inputs = text_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                output = text_model(**inputs)
                embeddings = output.last_hidden_state[:, 0]
            return embeddings.cpu().numpy()

        def get_image_embeddings(images: list[Image.Image]):
            if not images:
                return np.array([])
            inputs = image_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = image_model(**inputs)
            return outputs.pooler_output.cpu().numpy()

        def embed_batch_processor(df_chunk: pd.DataFrame):
            results = []
            # Process row by row to handle individual image failures gracefully
            for index, row in df_chunk.iterrows():
                try:
                    text = row.get("comment", "")  # Use .get for safety
                    url = row.get("photo", "")
                    ticket_id = row.get("ticket_id")

                    if not ticket_id:
                        print(f"Skipping row due to missing ticket_id: {row.to_dict()}")
                        continue
                    if not url:
                        print(
                            f"Skipping row for ticket {ticket_id} due to missing photo URL."
                        )
                        continue  # Skip if no photo URL

                    text_vec = get_text_embeddings([text])[0]

                    raw_image_bytes = _download_bytes(url)
                    if not raw_image_bytes:
                        print(
                            f"Skipping row for ticket {ticket_id}: image download failed for {url}"
                        )
                        continue  # Skip if image download failed

                    try:
                        img = Image.open(BytesIO(raw_image_bytes)).convert("RGB")
                        image_vec = get_image_embeddings([img])[0]
                        combined_vec = normalize(
                            np.hstack((text_vec, image_vec)).reshape(1, -1)
                        )[0]
                        results.append(
                            {"ticket_id": ticket_id, "vector": combined_vec.tolist()}
                        )
                        print(
                            f"Processed ticket {ticket_id} with embedding size {len(combined_vec)}"
                        )
                    except Exception as img_e:
                        print(
                            f"Error processing image for ticket {ticket_id} ({url}): {img_e}"
                        )
                        continue  # Skip row on image processing error

                except Exception as row_e:
                    print(
                        f"Error processing row (ticket: {row.get('ticket_id', 'N/A')}): {row_e}"
                    )
                    continue  # Skip row on general processing error

            return results

        # --- Replicate Main Logic ---
        bq_client = bigquery.Client(project=project_id)

        # Modify query to filter by the specific execution_date
        query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{raw_table}`
        WHERE DATE(timestamp) = DATE('{execution_date}')
          AND photo IS NOT NULL AND photo != ''
          AND ticket_id IS NOT NULL AND ticket_id != ''
        LIMIT 20
        """
        print(f"Executing BQ Query:\n{query}")

        df_todo = bq_client.query(query).to_dataframe()
        print(f"Fetched {df_todo.shape[0]} rows from BigQuery.")

        if df_todo.empty:
            print(f"No data found for date {execution_date} after filtering. Exiting.")
            return

        # --- Batch Process and Embed ---
        batch_size = (
            64  # Adjust batch size for embedding functions based on worker memory
        )
        all_vectors_data = []

        print(
            f"Starting batch processing with pandas chunks (embedding batch size {batch_size})..."
        )
        # Use iterrows or split dataframe into chunks for processing
        # The `embed_batch_processor` already processes internal batches/rows
        # Let's process the dataframe in larger chunks first to manage memory
        pandas_chunk_size = 500  # Adjust size of chunks read into embed_batch_processor
        for i in range(0, len(df_todo), pandas_chunk_size):
            chunk = df_todo.iloc[i : i + pandas_chunk_size]
            print(
                f"Processing pandas chunk {i // pandas_chunk_size + 1} of {len(df_todo) // pandas_chunk_size + (1 if len(df_todo) % pandas_chunk_size > 0 else 0)}..."
            )
            batch_results = embed_batch_processor(chunk)
            all_vectors_data.extend(batch_results)

        print(f"Finished processing. Generated {len(all_vectors_data)} vectors.")

        if not all_vectors_data:
            print("No vectors were generated successfully. Exiting.")
            return

        df_vectors = pd.DataFrame(all_vectors_data)

        df_merged = df_todo.merge(
            df_vectors,
            on="ticket_id",
            how="inner",
        )

        # --- Write Data to BigQuery Vector Table ---
        print(
            f"Uploading {len(df_merged)} vectors to {project_id}.{dataset_id}.{vector_table}"
        )

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

        job = bq_client.load_table_from_dataframe(
            df_merged,
            f"{project_id}.{dataset_id}.{vector_table}",
            job_config=job_config,
        )
        job.result()

        print(f"✅ Upload completed. Loaded {job.output_rows} rows into {vector_table}")

    except ImportError as e:
        print(
            f"ERROR: Missing required library: {e}. Please install it in your Composer environment."
        )
        # Re-raise the exception to make the task fail visibly
        raise e
    except Exception as e:
        print(f"An error occurred during vectorization: {e}")
        # Re-raise other exceptions to fail the task
        raise e


# Define the DAG
with DAG(
    dag_id="api_to_bq_pipeline_vectorize",
    default_args=default_args,
    description="Fetches data from Traffy API, transforms with Spark, and loads to BigQuery",
    schedule="0 1 * * *",  # Daily at 1:00 AM
    # schedule="*/30 * * * *",  # For testing, set to run every 30 minutes
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Bangkok"),
    catchup=False,
) as dag:

    fetch_data_from_api_to_gcs = PythonOperator(
        task_id="fetch_data_from_api_to_gcs",
        python_callable=fetch_api_and_upload,
        op_kwargs={
            "gcs_bucket": GCS_BUCKET,
            "gcs_prefix": GCS_RAW_PREFIX,
            "api_url": API_ENDPOINT,
            "query_params_base": {
                "output_format": "csv",
                "name": "ดร.วสันต์ ภัทรอธิคม",
                "org": "NECTEC",
                "purpose": "ทำสถิติการจัดการของแต่ละเขต ดูได้ที่ bangkok.traffy.in.th",
                "email": "traffyteam@gmail.com",
            },
        },
    )

    submit_spark_transform_job = DataprocCreateBatchOperator(
        task_id="submit_spark_transform_job",
        project_id=PROJECT_ID,
        region=DATAPROC_REGION,
        batch=SERVERLESS_BATCH_CONFIG,
        batch_id=BATCH_ID,
    )

    load_transformed_data_to_bigquery = BigQueryInsertJobOperator(
        task_id="load_transformed_data_to_bigquery",
        project_id=PROJECT_ID,
        location=BIGQUERY_REGION,
        configuration={
            "load": {
                "source_uris": [f"{GCS_PROCESSED_PATH}/*.csv"],
                "destination_table": {
                    "projectId": PROJECT_ID,
                    "datasetId": BIGQUERY_DATASET,
                    "tableId": BIGQUERY_TABLE,
                },
                "source_format": "CSV",
                "write_disposition": "WRITE_APPEND",
                "autodetect": False,
                "skip_leading_rows": 1,  # CSV has a header row
                "schema_fields": [
                    {"name": "ticket_id", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "type", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "organization", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "comment", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "photo", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "photo_after", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "address", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "subdistrict", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "district", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "province", "type": "STRING", "mode": "NULLABLE"},
                    {"name": "timestamp", "type": "TIMESTAMP", "mode": "NULLABLE"},
                    {"name": "state", "type": "STRING", "mode": "NULLABLE"},
                    {
                        "name": "star",
                        "type": "INTEGER",
                        "mode": "NULLABLE",
                    },
                    {"name": "count_reopen", "type": "INTEGER", "mode": "NULLABLE"},
                    {
                        "name": "last_activity",
                        "type": "TIMESTAMP",
                        "mode": "NULLABLE",
                    },
                    {"name": "longitude", "type": "FLOAT", "mode": "NULLABLE"},
                    {"name": "latitude", "type": "FLOAT", "mode": "NULLABLE"},
                ],
            }
        },
    )

    vectorize_new_data_in_composer = PythonOperator(
        task_id="vectorize_new_data_in_composer",
        python_callable=vectorize_data_directly_in_composer,
        op_kwargs={
            "project_id": PROJECT_ID,
            "dataset_id": BIGQUERY_DATASET,
            "raw_table": BIGQUERY_TABLE,
            "vector_table": BIGQUERY_VECTOR_TABLE,
            "execution_date": "{{ ds }}",
        },
    )

    create_kmeans_pca_and_view = BigQueryInsertJobOperator(
        task_id="create_kmeans_pca_and_view",
        project_id=PROJECT_ID,
        location=BIGQUERY_REGION,
        configuration={
            "query": {
                "query": """
-- Step 1: Create a K-Means Clustering Model
CREATE OR REPLACE MODEL `gen-lang-client-0538587318.bangkok_traffy.kmeans_model_for_vectors`
OPTIONS(
MODEL_TYPE = 'KMEANS',
NUM_CLUSTERS = 8,
STANDARDIZE_FEATURES = TRUE
) AS
SELECT
vector
FROM
`gen-lang-client-0538587318.bangkok_traffy.traffy_vector`
WHERE
ticket_id != '';

-- Step 2: Create a PCA Model
CREATE OR REPLACE MODEL `gen-lang-client-0538587318.bangkok_traffy.pca_model_for_vectors`
OPTIONS(
MODEL_TYPE = 'PCA',
NUM_PRINCIPAL_COMPONENTS = 3,
SCALE_FEATURES = FALSE
) AS
SELECT
vector
FROM
`gen-lang-client-0538587318.bangkok_traffy.traffy_vector`
WHERE
ticket_id != '';

-- Step 3: Create a VIEW combining original data with clustering and PCA results
CREATE OR REPLACE VIEW `gen-lang-client-0538587318.bangkok_traffy.traffy_vector_with_clusters` AS
WITH
clustered AS (
    SELECT
    ticket_id,
    CENTROID_ID AS cluster
    FROM
    ML.PREDICT(
        MODEL `gen-lang-client-0538587318.bangkok_traffy.kmeans_model_for_vectors`,
        (
        SELECT *
        FROM `gen-lang-client-0538587318.bangkok_traffy.traffy_vector`
        WHERE ticket_id != ''
        )
    )
),
pcaed AS (
    SELECT
    ticket_id,
    principal_component_1 AS PC1,
    principal_component_2 AS PC2,
    principal_component_3 AS PC3
    FROM
    ML.PREDICT(
        MODEL `gen-lang-client-0538587318.bangkok_traffy.pca_model_for_vectors`,
        (
        SELECT *
        FROM `gen-lang-client-0538587318.bangkok_traffy.traffy_vector`
        WHERE ticket_id != ''
        )
    )
)
SELECT
v.*,
CAST(c.cluster AS STRING) AS cluster,
p.PC1,
p.PC2,
p.PC3
FROM
`gen-lang-client-0538587318.bangkok_traffy.traffy_vector` AS v
LEFT JOIN
clustered AS c
USING(ticket_id)
LEFT JOIN
pcaed AS p
USING(ticket_id);
""",
                "useLegacySql": False,
                "multiStatementTransaction": True,
            }
        },
    )

    # Define the task dependencies
    (
        fetch_data_from_api_to_gcs
        >> submit_spark_transform_job
        >> load_transformed_data_to_bigquery
        >> vectorize_new_data_in_composer
        >> create_kmeans_pca_and_view
    )
