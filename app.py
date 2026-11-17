from pathlib import Path
import json
import numpy as np 
import secrets
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from config import SECRET_KEY, UPLOAD_DIR
from db import get_db, fetch_one, fetch_all, execute
from utils import save_upload, json_embedding, load_embedding
from face_engine import embeddings_from_image, embedding_from_single_face_image, cosine_similarity

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Enter a name, valid email and password of at least 6 characters.", "error")
            return render_template("register.html")
        try:
            uid = execute(
                "INSERT INTO users(name,email,password_hash) VALUES(%s,%s,%s)",
                (name, email, generate_password_hash(password))
            )
            session["user_id"] = uid
            session["user_name"] = name
            return redirect(url_for("dashboard"))
        except Exception as e:
            print("REGISTER ERROR:", repr(e))
            flash("Registration failed. Check the VS Code terminal for the exact error.", "error")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = fetch_one("SELECT * FROM users WHERE email=%s", (email,))
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    events = fetch_all(
        "SELECT e.*, COUNT(p.id) AS photo_count FROM events e "
        "LEFT JOIN photos p ON p.event_id=e.id WHERE e.user_id=%s "
        "GROUP BY e.id ORDER BY e.created_at DESC",
        (session["user_id"],)
    )
    return render_template("dashboard.html", events=events)

@app.route("/events/new", methods=["GET", "POST"])
@login_required
def new_event():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        event_date = request.form.get("event_date") or None
        description = request.form.get("description", "").strip()
        if not title:
            flash("Event title is required.", "error")
            return render_template("new_event.html")
        token = secrets.token_urlsafe(18)
        eid = execute(
            "INSERT INTO events(user_id,title,event_date,description,share_token) VALUES(%s,%s,%s,%s,%s)",
            (session["user_id"], title, event_date, description, token)
        )
        return redirect(url_for("event_detail", event_id=eid))
    return render_template("new_event.html")

@app.route("/events/<int:event_id>")
@login_required
def event_detail(event_id):
    event = fetch_one(
        "SELECT * FROM events WHERE id=%s AND user_id=%s",
        (event_id, session["user_id"])
    )

    if not event:
        return "Event not found", 404

    photos = fetch_all(
        "SELECT * FROM photos WHERE event_id=%s ORDER BY id DESC",
        (event_id,)
    )

    # Total number of photos
    photo_count = fetch_one(
        "SELECT COUNT(*) AS total FROM photos WHERE event_id=%s",
        (event_id,)
    )["total"]

    # Total number of detected faces
    face_count = fetch_one(
        """
        SELECT COALESCE(SUM(face_count), 0) AS total
        FROM photos
        WHERE event_id=%s
        """,
        (event_id,)
    )["total"]

    # Number of client searches
    search_count = fetch_one(
        "SELECT COUNT(*) AS total FROM searches WHERE event_id=%s",
        (event_id,)
    )["total"]

    return render_template(
        "event.html",
        event=event,
        photos=photos,
        photo_count=photo_count,
        face_count=face_count,
        search_count=search_count
    )
@app.route("/events/<int:event_id>/upload", methods=["POST"])
@login_required
def upload_photos(event_id):
    event = fetch_one(
        "SELECT * FROM events WHERE id=%s AND user_id=%s",
        (event_id, session["user_id"])
    )

    if not event:
        return "Event not found", 404

    files = request.files.getlist("photos")

    if not files:
        flash("Choose at least one image.", "error")
        return redirect(url_for("event_detail", event_id=event_id))

    folder = UPLOAD_DIR / "events" / str(event_id)

    uploaded_count = 0
    processed_count = 0
    failed_count = 0

    for file in files:

        if not file.filename:
            continue

        try:
            # -------------------------------------------------
            # 1. Save uploaded image
            # -------------------------------------------------
            path, filename = save_upload(file, folder)

            print()
            print("======================================")
            print("Processing:", file.filename)
            print("Saved to:", path)
            print("======================================")

            # -------------------------------------------------
            # 2. Create photo record
            # -------------------------------------------------
            photo_id = execute(
                """
                INSERT INTO photos
                (event_id, filename, original_name)
                VALUES (%s, %s, %s)
                """,
                (
                    event_id,
                    filename,
                    file.filename
                )
            )

            uploaded_count += 1

            # -------------------------------------------------
            # 3. Detect faces + generate embeddings
            # -------------------------------------------------
            try:

                embeddings = embeddings_from_image(path)

                print(
                    f"[Upload] {file.filename}: "
                    f"{len(embeddings)} face embedding(s) generated."
                )

                # -------------------------------------------------
                # 4. Save embeddings to MySQL
                # -------------------------------------------------
                conn = get_db()
                cur = conn.cursor()

                for emb in embeddings:

                    cur.execute(
                        """
                        INSERT INTO face_embeddings
                        (photo_id, embedding_json)
                        VALUES (%s, %s)
                        """,
                        (
                            photo_id,
                            json_embedding(emb)
                        )
                    )

                # -------------------------------------------------
                # 5. Mark photo as processed
                # -------------------------------------------------
                cur.execute(
                    """
                    UPDATE photos
                    SET processed = TRUE,
                        face_count = %s
                    WHERE id = %s
                    """,
                    (
                        len(embeddings),
                        photo_id
                    )
                )

                conn.commit()

                cur.close()
                conn.close()

                processed_count += 1

                print(
                    f"[Upload] SUCCESS: "
                    f"{file.filename} → "
                    f"{len(embeddings)} face(s)"
                )

            except Exception as exc:

                failed_count += 1

                print()
                print("!!!!!!!! FACE PROCESSING ERROR !!!!!!!!")
                print("Photo:", file.filename)
                print("Error:", repr(exc))
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print()

        except Exception as exc:

            failed_count += 1

            print()
            print("!!!!!!!! UPLOAD ERROR !!!!!!!!")
            print("File:", file.filename)
            print("Error:", repr(exc))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()

            flash(
                f"Could not process {file.filename}: {exc}",
                "error"
            )

    # ---------------------------------------------------------
    # Final message
    # ---------------------------------------------------------

    if failed_count == 0:
        flash(
            f"Successfully uploaded and processed "
            f"{processed_count} photo(s).",
            "success"
        )

    else:
        flash(
            f"Uploaded {uploaded_count} photo(s). "
            f"Successfully processed {processed_count}. "
            f"{failed_count} failed.",
            "error"
        )

    return redirect(
        url_for(
            "event_detail",
            event_id=event_id
        )
    )

@app.route("/media/events/<int:event_id>/<filename>")
def event_media(event_id, filename):
    return send_from_directory(UPLOAD_DIR / "events" / str(event_id), filename)

@app.route("/share/<token>", methods=["GET", "POST"])
def shared_event(token):
    event = fetch_one("SELECT * FROM events WHERE share_token=%s", (token,))
    if not event:
        return "Private event not found", 404
    if request.method == "POST":
        file = request.files.get("selfie")
        if not file or not file.filename:
            flash("Please upload a clear selfie.", "error")
            return redirect(url_for("shared_event", token=token))
        search_dir = UPLOAD_DIR / "searches" / token
        try:
            path, filename = save_upload(file, search_dir)
            query_embedding = embedding_from_single_face_image(path)
            rows = fetch_all(
                "SELECT fe.photo_id, fe.embedding_json, p.filename, p.original_name "
                "FROM face_embeddings fe JOIN photos p ON p.id=fe.photo_id "
                "WHERE p.event_id=%s",
                (event["id"],)
            )
            best_by_photo = {}
            for row in rows:
                score = cosine_similarity(query_embedding, load_embedding(row["embedding_json"]))
                if score > best_by_photo.get(row["photo_id"], -1):
                    best_by_photo[row["photo_id"]] = score
            threshold = 0.42
            matches = []
            for row in rows:
                pid = row["photo_id"]
                if best_by_photo.get(pid, -1) >= threshold:
                    matches.append({
                        "id": pid,
                        "filename": row["filename"],
                        "score": round(best_by_photo[pid], 3)
                    })
            unique = {m["id"]: m for m in matches}
            matches = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
            search_id = execute(
                "INSERT INTO searches(event_id,reference_filename,result_count) VALUES(%s,%s,%s)",
                (event["id"], filename, len(matches))
            )
            session["last_search_id"] = search_id
            return render_template("results.html", event=event, matches=matches, token=token)
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("shared_event.html", event=event, token=token)

@app.route("/download/<int:event_id>/<filename>")
def download_photo(event_id, filename):
    return send_from_directory(UPLOAD_DIR / "events" / str(event_id), filename, as_attachment=True)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "PhotoFind AI"})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_DIR,
        filename
    )
@app.route("/events/<int:event_id>/photos/<int:photo_id>/delete", methods=["POST"])
@login_required
def delete_photo(event_id, photo_id):

    # Make sure this event belongs to the logged-in photographer
    event = fetch_one(
        "SELECT * FROM events WHERE id=%s AND user_id=%s",
        (event_id, session["user_id"])
    )

    if not event:
        return "Event not found", 404

    # Find the photo
    photo = fetch_one(
        """
        SELECT id, filename
        FROM photos
        WHERE id=%s AND event_id=%s
        """,
        (photo_id, event_id)
    )

    if not photo:
        flash("Photo not found.", "error")
        return redirect(url_for(
            "event_detail",
            event_id=event_id
        ))

    try:
        # Delete face embeddings first
        execute(
            "DELETE FROM face_embeddings WHERE photo_id=%s",
            (photo_id,)
        )

        # Delete database photo record
        execute(
            "DELETE FROM photos WHERE id=%s",
            (photo_id,)
        )

        # Delete actual image file
        image_path = (
            UPLOAD_DIR
            / "events"
            / str(event_id)
            / photo["filename"]
        )

        if image_path.exists():
            image_path.unlink()

        flash("Photo deleted successfully.", "success")

    except Exception as exc:
        print("DELETE PHOTO ERROR:", repr(exc))
        flash(
            "Could not delete the photo.",
            "error"
        )

    return redirect(url_for(
        "event_detail",
        event_id=event_id
    ))

if __name__ == "__main__":
    app.run(debug=True)
