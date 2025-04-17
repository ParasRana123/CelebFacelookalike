import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the detector and model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load the feature list and filenames
try:
    with open('embedding.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
except Exception as e:
    st.error(f"Error loading pickle files: {e}")
    feature_list, filenames = None, None

# Streamlit interface
st.set_page_config(page_title="Bollywood Celebrity Match", page_icon=":clapper:", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1568605114967-8130f3a36994');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('https://images.unsplash.com/photo-1568605114967-8130f3a36994');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            background: rgba(0, 0, 0, 0.8);
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
        }
        .stFileUploader>div>div>div>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
        }
        .stTextInput>div>div>input {
            color: white;
        }
        .stMarkdown h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        .stMarkdown p {
            color: white;
        }
        .stAlert {
            z-index: 10;
            background-color: rgba(255, 255, 255, 0.9);
            color: black;
            border-radius: 10px;
            padding: 1rem;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: white;
        }
        .footer p {
            margin: 0;
        }
        .footer a {
            color: #ff4b4b;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for instructions
with st.sidebar:
    st.title("Instructions")
    st.write("""
        1. Upload an image of yourself.
        2. The app will analyze your face.
        3. It will match your face with a Bollywood celebrity.
        4. Enjoy and share your result!
    """)

st.title('Which Bollywood/Hollywood Celebrity do you look like?')
st.subheader("Upload an image to find out which Bollywood/Hollywood celebrity you resemble the most!")

uploaded_img = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])

def save_uploaded_image(uploaded_img):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_img.name), 'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not loaded properly.")
    
    results = detector.detect_faces(img)
    if not results:
        raise ValueError("No faces detected in the image.")
    
    if 'box' not in results[0]:
        raise KeyError("'box' key not found in the detection results.")
    
    x, y, width, height = results[0]['box']
    face = img[y:y+height, x:x+width]
    if face.size == 0:
        raise ValueError("Detected face has no size.")
    
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float64')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = []
    for feature in feature_list:
        similarity.append(cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0])
    
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

if feature_list is not None and filenames is not None:
    if uploaded_img is not None:
        with st.spinner('Processing...'):
            if save_uploaded_image(uploaded_img):
                display_img = Image.open(uploaded_img)
                try:
                    features = extract_features(os.path.join('uploads', uploaded_img.name), model, detector)
                    st.success("Features extracted successfully!")
                    index_pos = recommend(feature_list, features)
                    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("The image you Uploaded: ")
                        st.image(display_img, caption='Your Uploaded Image', use_column_width=True)
                    with col2:
                        st.success(f"Congratulations! Your face matches with {predicted_actor}")
                        st.image(filenames[index_pos], width=300, caption=predicted_actor)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.error("Feature list and filenames could not be loaded.")

# Add footer
st.markdown("""
    <div class="footer">
        <p>Made with Streamlit by <a href="https://www.yourwebsite.com" target="_blank">Your Name</a></p>
        <p>Follow me on <a href="https://twitter.com/yourtwitter" target="_blank">Instagram</a> | <a href="https://github.com/yourgithub" target="_blank">GitHub</a> | <a href="https://www.google.com/" target="_blank">Linkedin</a> for more</p>
    </div>
""", unsafe_allow_html=True)