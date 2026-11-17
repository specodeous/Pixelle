# Pixelle
AI-powered event photo retrieval for photographers.

## Stack
Python, Flask, OpenCV (YuNet + SFace), MySQL, HTML, CSS, JavaScript.

## Setup
1. Create a MySQL database using `database/schema.sql`.
2. Copy `.env.example` to `.env` and enter your MySQL credentials.
3. Create and activate a virtual environment.
4. `pip install -r requirements.txt`
5. `python setup_models.py`
6. `python app.py`
7. Open http://127.0.0.1:5000

Models:
- YuNet face detector
- SFace face recognizer
These are downloaded from the OpenCV Zoo. See `models/README.md`.

