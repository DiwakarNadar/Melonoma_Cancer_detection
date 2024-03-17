from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from keras.models import load_model
import joblib
from data import data
from unique_labels import unique_labels
app = Flask(__name__)
pred_id = joblib.load('pred_id.joblib')
model = load_model('Skin_Cancer_FT.hdf5')

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_pred_label(prediction_probabilities):
    print(unique_labels[np.argmax(prediction_probabilities)])
    return unique_labels[np.argmax(prediction_probabilities)]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            image_file = request.files["file"]
            if image_file.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(image_path)
                image = cv2.imread(image_path)
                new_shape = (224, 224)
                resized_img = cv2.resize(image, new_shape)
                input_data = np.expand_dims(resized_img, axis=0)
                pred_prob = model.predict(input_data)
                pred_label = get_pred_label(pred_prob)
                
                # Check if pred_label exists in pred_id dictionary
                pred_id_value = pred_id.get(pred_label)
                if pred_id_value is not None:
                    return jsonify({'prediction': pred_label, 'prediction_id': pred_id_value})
                else:
                    return jsonify({'prediction': 'Unknown', 'prediction_id': None})
    return render_template("index.html")

@app.route("/details/<prediction_id>")
def details(prediction_id):
    print("Prediction ID:", prediction_id)
    details = data[unique_labels[int(prediction_id)]]
    return render_template("details.html", details=details)

if __name__ == "__main__":
    app.run(debug=True)
