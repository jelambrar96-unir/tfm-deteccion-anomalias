import os
from dotenv import load_dotenv


load_dotenv()


MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", None)
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", None)
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", None)

#
POSTGRES_DB = os.getenv("POSTGRES_DB", None)
POSTGRES_USER = os.getenv("POSTGRES_USER", None)
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", None)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", None)
POSTGRES_PORT = os.getenv("POSTGRES_PORT", None)
#

# MLFLOW_TRACKING_URI "postgresql://mlflow:mlflow_pass@postgres:5432/mlflow"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", None)
MLFLOW_S3_IGNORE_TLS = os.getenv("MLFLOW_S3_IGNORE_TLS", None) ==  "true"
MLFLOW_BUCKET_NAME = MINIO_BUCKET_NAME
MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", None)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", None)


# print(MLFLOW_EXPERIMENT_NAME)
# print(MLFLOW_SERVER)
