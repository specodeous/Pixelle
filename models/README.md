Place these two OpenCV Zoo ONNX files in this folder. Run `python setup_models.py` to download them automatically:
- face_detection_yunet_2023mar.onnx
- face_recognition_sface_2021dec.onnx

OpenCV's official SFace demo uses YuNet + SFace and supports cosine similarity. The official example reports a cosine threshold of 0.363; this project starts its gallery search at 0.42 as a conservative prototype threshold. Tune it using your own event images.