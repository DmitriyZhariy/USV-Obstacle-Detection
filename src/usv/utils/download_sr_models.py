"""
Helper to download pre-trained Super-Resolution models for OpenCV.
Models:
- EDSR (Best quality, heavy)
- FSRCNN (Fast, good for video)
"""
import urllib.request
from pathlib import Path

def download_models():
    save_dir = Path("models/super_res")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ссылки на официальные веса OpenCV
    urls = {
        "EDSR_x4.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "FSRCNN_x3.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb"
    }

    print(f"Downloading models to {save_dir}...")
    for name, url in urls.items():
        dest = save_dir / name
        if dest.exists():
            print(f" - {name} already exists.")
            continue

        print(f" - Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    print("Done.")

if __name__ == "__main__":
    download_models()
