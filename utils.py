from pathlib import Path
import json
import uuid
from werkzeug.utils import secure_filename
from config import UPLOAD_DIR

ALLOWED = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def save_upload(file, folder):
    safe = secure_filename(file.filename or "")
    if not safe or not allowed_file(safe):
        raise ValueError("Only JPG, JPEG, PNG and WEBP images are allowed.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}_{safe}"
    path = folder / name
    file.save(path)
    return path, name

def json_embedding(vec):
    return json.dumps([float(x) for x in vec])

def load_embedding(value):
    return json.loads(value)
