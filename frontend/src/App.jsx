import { useState } from 'react'
import './App.css'
import axios from "axios";

function App() {
  const [image , setImage] = useState(null);
  const [preview , setPreview] = useState(null);
  const [loading , setLoading] = useState(false);
  const [results , setResults] = useState([]);
  const [error , setError] = useState("");

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if(file) {
      setImage(file);
      setPreview(URL.createObjectURL(file))
    }
  }

  const handleOnSubmit = async () => {
    if(!image) {
      alert("Upload the image")
      return;
    }
    const formData = new FormData();
    formData.append("image" , image);
    try {
      setLoading(true);
      setError("");
      const response = await axios.post(
        "http://127.0.0.1:5000/recognize",
        formData,
        {
          headers: {
            "Content-type" : "multipart/form-data"
          }
        }
      );
      setResults(response.data.matches);
    } catch(e) {
      console.log(e);
      setError(
        e.response?.data?.error || "Something went wrong"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Celebrity Lookalike Detector</h1>
      <input type='file' accept='image/' onChange={handleImageChange}/>
      <br />
      <br />
      <button onClick={handleOnSubmit}>Find My Lookalike</button>
      {preview && <img src={preview} width="50" height="50" />}
      {loading && <h2>Processing....</h2>}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "20px",
          marginTop: "30px"
        }}
      >
        {results.map((item, index) => (
          <div
            key={index}
            style={{
              background: "#1f2937",
              padding: "15px",
              borderRadius: "10px",
              width: "220px"
            }}
          >
            <img
              src={item.image}
              alt={item.name}
              width="200"
              height="250"
              style={{
                objectFit: "cover",
                borderRadius: "10px"
              }}
            />
            <h3>{item.name}</h3>
            <p>
              Similarity: {item.similarity}%
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App