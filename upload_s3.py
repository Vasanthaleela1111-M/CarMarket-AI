import boto3
from pathlib import Path

bucket_name = "vasantha-leela-2026-12345"
folder = Path("Dataset")
s3 = boto3.client("s3")


for file in folder.glob("*.xlsx"):
    s3_key = f"Dataset/{file.name}"
    print(f"Uploading: {file.name}")
    s3.upload_file(str(file), bucket_name, s3_key)
print(f"Uploaded: {file.name}")
print("File uploaded successfully!")
