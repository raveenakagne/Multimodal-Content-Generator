import json
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime


client = storage.Client()

def upload_json_to_gcs(bucket_name, destination_blob_name, json_data):
    """
    Uploads a JSON object directly to Google Cloud Storage.
    
    Parameters:
    - bucket_name (str): The name of the GCP bucket.
    - destination_blob_name (str): The desired name of the file in the bucket.
    - json_data (dict): The JSON data to be uploaded.
    """
    # Specify the bucket and blob
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Upload JSON data as a string
    blob.upload_from_string(
        data=json.dumps(json_data),
        content_type="application/json"
    )
    print(f"JSON object uploaded to {bucket_name}/{destination_blob_name}")

def store_user_query(query, bucket_name):
    """
    Stores a user-provided query as a JSON object and uploads it to GCP.
    
    Parameters:
    - query (str): The user-provided query.
    - bucket_name (str): The GCP bucket name.
    """
    # Create JSON object for the query
    query_data = {
        "query": query,
        "timestamp": datetime.utcnow().isoformat()  # Add a timestamp for reference
    }
    
    # Set a unique file name for each query, e.g., `query-YYYYMMDD-HHMMSS.json`
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    destination_blob_name = f"query-{timestamp}.json"
    
    # Upload JSON to GCS
    upload_json_to_gcs(bucket_name, destination_blob_name, query_data)

# Example usage
bucket_name = "user-queries-bucket-1"
user_query = input("Enter your query: ")
store_user_query(user_query, bucket_name)
