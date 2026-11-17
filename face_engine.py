from pathlib import Path
import cv2
import numpy as np


BASE = Path(__file__).resolve().parent

YUNET_PATH = BASE / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = BASE / "models" / "face_recognition_sface_2021dec.onnx"

_detector = None
_recognizer = None


def init_models():
    """Load YuNet and SFace once."""

    global _detector, _recognizer

    if not YUNET_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found:\n{YUNET_PATH}\n"
            "Run: python setup_models.py"
        )

    if not SFACE_PATH.exists():
        raise FileNotFoundError(
            f"SFace model not found:\n{SFACE_PATH}\n"
            "Run: python setup_models.py"
        )

    if _detector is None:
        print("[FaceEngine] Loading YuNet...")

        _detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH),
            "",
            (320, 320),
            0.75,
            0.3,
            5000
        )

        print("[FaceEngine] YuNet loaded.")

    if _recognizer is None:
        print("[FaceEngine] Loading SFace...")

        _recognizer = cv2.FaceRecognizerSF.create(
            str(SFACE_PATH),
            ""
        )

        print("[FaceEngine] SFace loaded.")


def read_image(path):
    """Read an image safely."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    # Make sure the image is continuous in memory.
    image = np.ascontiguousarray(image)

    # Make sure OpenCV receives the expected type.
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def detect_faces(image):
    """Detect faces using YuNet."""

    init_models()

    if image is None:
        return []

    if len(image.shape) != 3:
        raise ValueError("Image must be a color image.")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    image = np.ascontiguousarray(image)

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return []

    # YuNet needs the actual image size.
    _detector.setInputSize((int(width), int(height)))

    try:
        result = _detector.detect(image)
    except Exception as e:
        print("[FaceEngine] YuNet detection error:", repr(e))
        return []

    # OpenCV Python versions may return:
    # (retval, faces)
    if isinstance(result, tuple):
        _, faces = result
    else:
        faces = result

    if faces is None:
        return []

    faces = np.asarray(faces, dtype=np.float32)

    if faces.size == 0:
        return []

    if faces.ndim == 1:
        faces = faces.reshape(1, -1)

    return faces


def embedding_for_face(image, face):
    """
    Convert one detected face into an SFace embedding.

    We keep OpenCV's official alignCrop + feature workflow,
    but add validation and defensive array handling.
    """

    init_models()

    if image is None:
        raise ValueError("Input image is empty.")

    image = np.ascontiguousarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    face = np.asarray(face, dtype=np.float32).reshape(-1)

    # YuNet should provide 15 values:
    # x, y, w, h + 10 landmark coordinates + confidence
    if face.size < 15:
        raise ValueError(
            f"Invalid YuNet face data. Expected 15 values, got {face.size}."
        )

    # Remove invalid numeric values.
    if not np.all(np.isfinite(face)):
        raise ValueError("Face detection returned invalid coordinates.")

    try:
        # OpenCV's SFace implementation expects the detected
        # YuNet face row for alignment.
        aligned = _recognizer.alignCrop(image, face)
    except Exception as e:
        raise RuntimeError(
            f"SFace alignCrop failed: {repr(e)}"
        ) from e

    if aligned is None or aligned.size == 0:
        raise RuntimeError("SFace produced an empty aligned face.")

    aligned = np.ascontiguousarray(aligned)

    if aligned.dtype != np.uint8:
        aligned = aligned.astype(np.uint8)

    try:
        feature = _recognizer.feature(aligned)
    except Exception as e:
        raise RuntimeError(
            f"SFace feature extraction failed: {repr(e)}"
        ) from e

    if feature is None:
        raise RuntimeError("SFace returned no feature.")

    feature = np.asarray(feature, dtype=np.float32).reshape(-1)

    if feature.size == 0:
        raise RuntimeError("SFace returned an empty feature vector.")

    if not np.all(np.isfinite(feature)):
        raise RuntimeError("SFace returned invalid feature values.")

    # Normalize the feature vector.
    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    return feature.astype(np.float32)


def embeddings_from_image(path):
    """
    Detect every face in an event photograph
    and generate one embedding for each face.
    """

    image = read_image(path)

    faces = detect_faces(image)

    print(
        f"[FaceEngine] {Path(path).name}: "
        f"{len(faces)} face(s) detected."
    )

    results = []

    for index, face in enumerate(faces):

        try:
            embedding = embedding_for_face(image, face)

            results.append(embedding)

            print(
                f"[FaceEngine] Face {index + 1}: "
                f"embedding generated ({len(embedding)} values)."
            )

        except Exception as e:

            print(
                f"[FaceEngine] Face {index + 1} skipped: "
                f"{repr(e)}"
            )

    return results


def embedding_from_single_face_image(path):
    """
    Generate an embedding from a selfie/reference image.

    Exactly one face is required.
    """

    image = read_image(path)

    faces = detect_faces(image)

    if len(faces) != 1:
        raise ValueError(
            "Reference image must contain exactly one clear face. "
            f"Found {len(faces)}."
        )

    return embedding_for_face(image, faces[0])


def cosine_similarity(a, b):
    """Calculate cosine similarity between two embeddings."""

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.size == 0 or b.size == 0:
        return 0.0

    if a.size != b.size:
        return 0.0

    denominator = float(
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )