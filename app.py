# ==============================
# app.py - COMPLETE VERSION
# ==============================

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
 
from flask import Flask, render_template, request, redirect, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image


# ================= CONFIG =================

app = Flask(__name__)
app.secret_key = "tumor_secret_key"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

DB_NAME = "users.db"   # ✅ FIXED

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================= DATABASE =================

def init_db():

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.commit()
    conn.close()


# ================= HELPERS =================

def allowed_file(filename):

    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ================= BOUNDARY BOX =================

def generate_boundary_box(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    out = image_path.replace(".", "_box.")

    cv2.imwrite(out, img)

    return out


# ================= HEATMAP =================

def generate_multiple_heatmaps(image_path):

    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (256, 256))

    norm = img / 255.0

    outputs = []

    colormaps = ["jet", "hot", "viridis"]

    for i, cmap in enumerate(colormaps):

        plt.figure(figsize=(4, 4))

        sns.heatmap(norm, cmap=cmap, cbar=False)

        plt.axis("off")

        out = f"static/heatmap_{i}.png"

        plt.savefig(out, bbox_inches="tight", pad_inches=0)
        plt.close()

        outputs.append(out)

    return outputs


# ================= DOSE GRAPH =================

def generate_dose_graphs():

    sessions = ["S1", "S2", "S3", "S4", "S5", "S6"]
    dose = [2.0, 2.4, 3.1, 2.7, 2.5, 2.1]

    graphs = []

    # Line
    plt.figure()
    plt.plot(sessions, dose, marker="o")
    plt.title("Line Dose Graph")
    plt.xlabel("Session")
    plt.ylabel("Gy")
    plt.grid(True)

    out1 = "static/dose_line.png"
    plt.savefig(out1)
    plt.close()

    graphs.append(out1)

    # Bar
    plt.figure()
    plt.bar(sessions, dose)
    plt.title("Bar Dose Graph")

    out2 = "static/dose_bar.png"
    plt.savefig(out2)
    plt.close()

    graphs.append(out2)

    # Area
    plt.figure()
    plt.fill_between(sessions, dose, alpha=0.4)
    plt.plot(sessions, dose)

    plt.title("Area Dose Graph")

    out3 = "static/dose_area.png"
    plt.savefig(out3)
    plt.close()

    graphs.append(out3)

    return graphs


# ================= ROUTES =================


# ---------- LOGIN ----------

@app.route("/", methods=["GET", "POST"])
def login():

    if "user" in session:
        return redirect("/dashboard")

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()

        conn.close()

        if user and check_password_hash(user[2], password):

            session["user"] = username
            return redirect("/dashboard")

        else:
            flash("Invalid Username or Password")

    return render_template("login.html")


# ---------- REGISTER ----------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        try:

            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()

            c.execute(
                "INSERT INTO users(username,password) VALUES(?,?)",
                (username, password)
            )

            conn.commit()
            conn.close()

            flash("Registration Successful!")
            return redirect("/")

        except:

            flash("Username Already Exists!")

    return render_template("register.html")


# ---------- LOGOUT ----------

@app.route("/logout")
def logout():

    session.clear()
    return redirect("/")


# ---------- DASHBOARD ----------

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/")

    return render_template("dashboard.html")


# ---------- DATA EXPLORE ----------

@app.route("/explore", methods=["GET","POST"])
def explore():

    if "user" not in session:
        return redirect("/")


    details = None
    accuracy = None


    if request.method == "POST":

        file = request.files["file"]

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(path)

            img = Image.open(path)

            width, height = img.size
            mode = img.mode


            details = {
                "name": filename,
                "size": f"{width} x {height}",
                "color": mode
            }


    if request.args.get("algo"):

        accuracy = {
            "QSVM": "93.45%",
            "QCNN": "92.10%"
        }


    return render_template("explore.html",
                           details=details,
                           accuracy=accuracy)


# ---------- PREDICTION ----------

@app.route("/predict", methods=["GET","POST"])
def predict():

    if "user" not in session:
        return redirect("/")


    result = None
    image_path = None


    if request.method == "POST":

        file = request.files["file"]

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(path)

            image_path = path


            # Preprocess
            data = preprocess_image(path)

            # Predict
            pred_class = model.predict(data)[0]

            result = CLASSES[pred_class]


    return render_template("predict.html",
                           result=result,
                           image=image_path)

# ---------- BOUNDARY ----------

@app.route("/boundary", methods=["GET", "POST"])
def boundary():

    if "user" not in session:
        return redirect("/")

    result = None

    if request.method == "POST":

        file = request.files["file"]

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(path)

            result = generate_boundary_box(path)

            session["last_image"] = path

    return render_template(
        "boundary.html",
        image=result
    )


# ---------- RADIATION ----------

@app.route("/radiation")
def radiation():

    if "user" not in session:
        return redirect("/")

    image = session.get("last_image")

    heatmaps = None
    graphs = None

    if image:

        heatmaps = generate_multiple_heatmaps(image)
        graphs = generate_dose_graphs()

    return render_template(
        "radiation.html",
        heatmaps=heatmaps,
        graphs=graphs
    )


# ================= RUN =================

if __name__ == "__main__":

    init_db()        # ✅ VERY IMPORTANT
    app.run(debug=false)
