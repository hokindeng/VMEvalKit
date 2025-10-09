#!/usr/bin/env python3
"""
Test boto3 region configuration.
"""

import os
import boto3
from botocore.config import Config

# Test different ways to set region
print("Testing boto3 region configuration")
print("=" * 60)

# Method 1: Basic client with region
client1 = boto3.client('s3', region_name='us-east-2')
print(f"Method 1 - client region: {client1._client_config.region_name}")

# Method 2: With Config object
config = Config(
    region_name='us-east-2',
    signature_version='s3v4'
)
client2 = boto3.client('s3', config=config)
print(f"Method 2 - config region: {client2._client_config.region_name}")

# Method 3: Both region_name and config
client3 = boto3.client(
    's3',
    region_name='us-east-2',
    config=Config(signature_version='s3v4')
)
print(f"Method 3 - both region: {client3._client_config.region_name}")

# Method 4: Config with region
config_with_region = Config(
    region_name='us-east-2',
    signature_version='s3v4'
)
client4 = boto3.client('s3', config=config_with_region)
print(f"Method 4 - config with region: {client4._client_config.region_name}")

# Test presigned URL generation
print("\nTesting presigned URL generation:")
test_clients = [
    ("Method 3", client3),
    ("Method 4", client4)
]

for name, client in test_clients:
    url = client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': 'vmevalkit',
            'Key': 'test/image.png'
        },
        ExpiresIn=3600
    )
    print(f"\n{name}:")
    print(f"  URL: {url[:150]}...")
    # Check if region is in the credential
    if 'us-east-2' in url:
        print("  ✓ Contains us-east-2")
    elif 'us-east-1' in url:
        print("  ✗ Contains us-east-1 (wrong region)")
    else:
        print("  ? Region not found in URL")
