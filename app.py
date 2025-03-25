from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดล YOLO ที่เทรนไว้
model = YOLO("best.pt")

# mapping class index → label name
custom_labels = {
    0: "atrophy",
    1: "broken",
    2: "good",
    3: "moldy",
    4: "peeling",
    5: "spot"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # วิเคราะห์ภาพด้วยโมเดล
    results = model(filepath)
    result = results[0]

    if result.probs:
        label_index = int(result.probs.top1)
        label_name = custom_labels.get(label_index, f"ไม่รู้จักคลาส {label_index}")
        confidence = float(result.probs.top1conf)
    else:
        label_name = "ไม่สามารถจำแนกได้"
        confidence = 0.0

    return jsonify({
        'label': label_name,
        'confidence': confidence,
        'image_path': f"/uploads/{filename}"
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)