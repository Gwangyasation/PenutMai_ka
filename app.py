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
    0: "เมล็ดลีบ (atrophy)",
    1: "เมล็ดแตกหัก (broken)",
    2: "เมล็ดดี (good)",
    3: "เมล็ดเชื้อรา (moldy)",
    4: "เมล็ดเปลือกร่อน (peeling)",
    5: "เมล็ดจุดด่าง (spot)"
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

    results = model(filepath)
    result = results[0]

    # Mapping index -> label name
    custom_labels = {
        0: "เมล็ดลีบ (atrophy)",
        1: "เมล็ดแตกหัก (broken)",
        2: "เมล็ดดี (good)",
        3: "เมล็ดเชื้อรา (moldy)",
        4: "เมล็ดเปลือกร่อน (peeling)",
        5: "เมล็ดจุดด่าง (spot)"
    }

    predictions = []
    if result.probs:
        for idx, conf in enumerate(result.probs.data.tolist()):
            label = custom_labels.get(idx, f"คลาส {idx}")
            predictions.append({
                'label': label,
                'confidence': round(conf, 4)
            })

    return jsonify({
        'predictions': predictions,
        'image_path': f"/uploads/{filename}"
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)