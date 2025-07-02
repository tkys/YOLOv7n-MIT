#!/usr/bin/env python3
"""
Download pre-trained YOLO weights for Transfer Learning
"""
import os
import requests
from pathlib import Path

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)
    
    print(f"\n✓ Downloaded: {destination}")

def main():
    """Download pretrained weights"""
    weights_dir = Path("pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    # YOLOv9 pre-trained weights (closest to v7n architecture)
    weights_urls = {
        "yolov9-c.pt": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt",
        "yolov9-s.pt": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt",
    }
    
    for weight_name, url in weights_urls.items():
        weight_path = weights_dir / weight_name
        
        if weight_path.exists():
            print(f"✓ Already exists: {weight_path}")
            continue
            
        try:
            download_file(url, weight_path)
        except Exception as e:
            print(f"✗ Failed to download {weight_name}: {e}")
            continue
    
    print("\n✓ Pretrained weights download completed!")

if __name__ == "__main__":
    main()