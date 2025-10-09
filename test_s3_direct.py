#!/usr/bin/env python3
"""
Direct test of S3 configuration issue.
"""

import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

print("Environment variables:")
print(f"AWS_REGION: {os.getenv('AWS_REGION')}")
print(f"AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")

# Replicate exact S3ImageUploader logic
bucket_name = os.getenv("S3_BUCKET", "vmevalkit")
region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-2"))

print(f"\nUsing bucket: {bucket_name}")
print(f"Using region: {region}")

# Create client exactly as in S3ImageUploader
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    config=Config(
        region_name=region,
        signature_version='s3v4'
    )
)

print(f"\nClient region: {s3_client._client_config.region_name}")

# Generate presigned URL
url = s3_client.generate_presigned_url(
    'get_object',
    Params={
        'Bucket': bucket_name,
        'Key': 'test/image.png'
    },
    ExpiresIn=3600
)

print(f"\nGenerated URL: {url}")

# Check region in URL
if 'us-east-2' in url:
    print("✓ URL contains us-east-2")
elif 'us-east-1' in url:
    print("✗ URL contains us-east-1 (wrong region)")
else:
    print("? Region not found in URL")

# Also test default boto3 session
print(f"\nDefault boto3 session region: {boto3.DEFAULT_SESSION}")
if boto3.DEFAULT_SESSION:
    print(f"Session region: {boto3.DEFAULT_SESSION.region_name}")
