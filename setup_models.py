from pathlib import Path
from urllib.request import urlretrieve

MODELS = {
    "models/face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

for path, url in MODELS.items():
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 100000:
        print(f"Already exists: {path}")
        continue
    print(f"Downloading {path} ...")
    urlretrieve(url, path)
    print(f"Saved: {path}")
print("Model setup complete.")
