# AI-Powered Facial Recognition Match System

A web application that detects faces from user-uploaded images and matches them with celebrity faces using deep learning techniques. Powered by the **VGGFace (ResNet50)** model and **MTCNN** for robust face detection and recognition.

## Features

-  Upload any face image and get matched with a celebrity lookalike
-  Accurate face detection using **MTCNN**
-  Face recognition using **ResNet50-based VGGFace model**
-  Real-time similarity scoring with **cosine similarity**
-  Smooth and responsive user interface for interactive experience  

## Tech Stack

- **AI Model**: LLAMA
- **Frontend**: React, HTML, CSS, TypeScript
- **Backend**: Node.js, Express.js
- **Code Execution**: Web Containers

## Project Structure

```bash
├── backend/
│   ├── src/              
│        ├── defaults/     # Default prompts for node and react
│            ├── react.ts
│            └── node.ts
│   ├── index.ts           # All main routes in this file
│   ├── prompts.ts         # format in which the response returned
│   ├── constants.ts       # Utility file
│   └── stripindents.ts    # Utility file
├── frontend/
│   ├── components/        # Contains various components
│   ├── hooks/             # Web Container config file
│   ├── pages/             # Contains landing page
│   └── types/             # Defined the types of file structure
└── README.md
```

## Installation

> **Note**: Ensure Node.js (v16+) is installed on your machine.

1. **Clone the Repository**

```bash
git clone [repository-url]
cd boult.new
```

2. **Install the Backend dependencies**

```bash
cd be
npm install
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