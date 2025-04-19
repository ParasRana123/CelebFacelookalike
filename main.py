from flask import Flask, render_template, request
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import shutil
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MATCH_FOLDER'] = os.path.join('static', 'matches')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MATCH_FOLDER'], exist_ok=True)

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load embeddings
with open('embedding.pkl', 'rb') as f:
    feature_list = pickle.load(f)

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

def extract_features(img_path):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    if not results:
        raise ValueError("No face detected.")
    x, y, width, height = results[0]['box']
    face = img[y:y+height, x:x+width]
    image = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(image).astype('float64')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(features, top_n=5):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    top_scores = [similarity[i] for i in top_indices]
    return top_indices, top_scores

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        try:
            features = extract_features(upload_path)
            indices, scores = recommend(features)

            match_results = []

            for idx, score in zip(indices, scores):
                matched_path = filenames[idx].replace("\\", "/")
                matched_filename = os.path.basename(matched_path)
                match_dest_path = os.path.join(app.config['MATCH_FOLDER'], matched_filename)

                if not os.path.exists(match_dest_path):
                    shutil.copy(matched_path, match_dest_path)

                predicted_actor = os.path.splitext(matched_filename)[0]
                predicted_actor = re.sub(r'[\._]?\d+$', '', predicted_actor)
                predicted_actor = predicted_actor.replace('_', ' ')
                similarity = round(score * 100, 2)

                match_results.append({
                    'image': f'matches/{matched_filename}',
                    'name': predicted_actor,
                    'similarity': similarity
                })

            return render_template('index.html',
                                   uploaded_image=f'uploads/{filename}',
                                   match_results=match_results)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)