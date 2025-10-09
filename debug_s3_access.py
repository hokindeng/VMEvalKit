#!/usr/bin/env python3
"""
Debug S3 access and image moderation issues.
"""

import os
import sys
from pathlib import Path
import requests
from PIL import Image
import io
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

load_dotenv()

from vmevalkit.utils.s3_uploader import S3ImageUploader

def test_s3_upload_and_access():
    """Test S3 upload and verify the presigned URL works."""
    
    print("Testing S3 Upload and Access")
    print("=" * 60)
    
    # Initialize uploader
    uploader = S3ImageUploader()
    
    # Test with a maze image
    test_images = [
        "data/generated_mazes/irregular_0000_first.png",
        "data/generated_mazes/knowwhat_0001_first.png"
    ]
    
    for image_path in test_images:
        print(f"\nTesting: {image_path}")
        
        # Check if file exists locally
        if not Path(image_path).exists():
            print(f"  ❌ File not found locally")
            continue
        
        # Check image properties
        with Image.open(image_path) as img:
            print(f"  Image size: {img.size}")
            print(f"  Image mode: {img.mode}")
            print(f"  Format: {img.format}")
        
        # Upload to S3
        try:
            url = uploader.upload(image_path)
            
            if not url:
                print(f"  ❌ Upload failed - no URL returned")
                continue
                
            print(f"  ✓ Uploaded successfully")
            print(f"  URL: {url}")
            
            # Test if URL is accessible
            print("  Testing URL accessibility...")
            try:
                response = requests.get(url, timeout=10)
                print(f"  HTTP Status: {response.status_code}")
                print(f"  Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
                print(f"  Content-Length: {response.headers.get('Content-Length', 'Unknown')} bytes")
                
                if response.status_code == 200:
                    # Try to load as image
                    img_data = io.BytesIO(response.content)
                    test_img = Image.open(img_data)
                    print(f"  ✓ Successfully loaded as image: {test_img.size}")
                else:
                    print(f"  ❌ Failed to access URL: {response.text[:200]}")
                    
            except requests.RequestException as e:
                print(f"  ❌ Request failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Upload error: {e}")
    
    # Clean up
    print("\nCleaning up S3 files...")
    uploader.cleanup()
    print("Done!")

def analyze_maze_images():
    """Analyze maze images to understand potential moderation issues."""
    
    print("\n\nAnalyzing Maze Images")
    print("=" * 60)
    
    maze_files = list(Path("data/generated_mazes").glob("*.png"))
    
    for maze_file in maze_files[:5]:  # Check first 5
        print(f"\n{maze_file.name}:")
        
        with Image.open(maze_file) as img:
            # Basic properties
            print(f"  Size: {img.size}")
            print(f"  Mode: {img.mode}")
            
            # Check if image has transparency
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                print("  Has transparency: Yes")
            else:
                print("  Has transparency: No")
            
            # Check color distribution
            if img.mode == 'RGB' or img.mode == 'RGBA':
                # Get color statistics
                pixels = list(img.getdata())
                unique_colors = len(set(pixels))
                print(f"  Unique colors: {unique_colors}")
                
                # Check for extreme colors
                if img.mode == 'RGB':
                    r_vals = [p[0] for p in pixels]
                    g_vals = [p[1] for p in pixels]
                    b_vals = [p[2] for p in pixels]
                else:  # RGBA
                    r_vals = [p[0] for p in pixels if len(p) >= 3]
                    g_vals = [p[1] for p in pixels if len(p) >= 3]
                    b_vals = [p[2] for p in pixels if len(p) >= 3]
                
                print(f"  Color ranges: R({min(r_vals)}-{max(r_vals)}), G({min(g_vals)}-{max(g_vals)}), B({min(b_vals)}-{max(b_vals)})")

if __name__ == "__main__":
    test_s3_upload_and_access()
    analyze_maze_images()
