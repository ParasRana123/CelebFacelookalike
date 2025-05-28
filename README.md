# AI-Powered Facial Recognition Match System

A web application that detects faces from user-uploaded images and matches them with celebrity faces using deep learning techniques. Powered by the **VGGFace (ResNet50)** model and **MTCNN** for robust face detection and recognition.

## Features

-  Upload any face image and get matched with a celebrity lookalike
-  Accurate face detection using **MTCNN**
-  Face recognition using **ResNet50-based VGGFace model**
-  Real-time similarity scoring with **cosine similarity**
-  Smooth and responsive user interface for interactive experience  

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Streamlit
- **Deep Learning**: Keras, TensorFlow, VGGFace, MTCNN
- **Similarity Measure**: Cosine similarity

## Project Structure

```bash
├── static/            
│   ├── css/        # All CSS stylings in this file
│   ├── uploads/    # Contains user Uploaded images
│   └── matches/    # Matched images for the user uplaoded images
├── templates/
│   └── index.html  # Contains all the frontend HTML code
├── main.py         # Flask backend code
├── app.py          # Streamlit backend code
├── feature_extractor1.ipynb   # Useful for making filenames.pkl
├── feature_extractor.ipynb    # Useful for making embedding.pkl
├── requirements.txt           # Contains all the requirements
└── README.md
```

## Installation

> **Note**: Python Version greater than 3.7 needed.

1. **Clone the Repository**

```bash
git clone [repository-url]
cd face
```

2. **Install all the requirements necessary for this project**

```bash
pip install -r requirements.txt
```

3. **Install the Frontend dependencies**

```bash
cd frontend
npm install
```

4. **Set up environment varaibles**

> **Note**: Create the .env file in the be folder.

```bash
echo. > .env
```

5. **Start the backend server**

```bash
cd be
npm run dev
```

6. **Start the frontend application**

```bash
cd frontend
npm run dev
```

## Contributing

We welcome contributions from the community! Whether you're interested in improving features, fixing bugs, or adding new functionality, your input is valuable. Feel free to reach out to us with your ideas and suggestions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.