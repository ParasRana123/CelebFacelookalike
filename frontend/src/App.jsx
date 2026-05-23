import { useState, useRef } from 'react'
import axios from "axios";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    background: #0a0a0a;
    min-height: 100vh;
  }

  .app {
    min-height: 100vh;
    background: #0a0a0a;
    color: #f0ead6;
    font-family: 'DM Sans', sans-serif;
    position: relative;
    overflow-x: hidden;
  }

  .bg-grid {
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(212, 175, 55, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(212, 175, 55, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
  }

  .bg-glow {
    position: fixed;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 700px;
    height: 400px;
    background: radial-gradient(ellipse, rgba(212, 175, 55, 0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0 auto;
    padding: 60px 24px 80px;
  }

  .header {
    text-align: center;
    margin-bottom: 56px;
  }

  .header-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #d4af37;
    margin-bottom: 16px;
    opacity: 0.9;
  }

  .header-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 6vw, 64px);
    font-weight: 900;
    line-height: 1.05;
    color: #f0ead6;
    margin-bottom: 16px;
    letter-spacing: -0.02em;
  }

  .header-title span {
    color: #d4af37;
    font-style: italic;
  }

  .header-sub {
    font-size: 15px;
    font-weight: 300;
    color: rgba(240, 234, 214, 0.5);
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .upload-zone {
    border: 1.5px dashed rgba(212, 175, 55, 0.3);
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    background: rgba(212, 175, 55, 0.02);
    position: relative;
    overflow: hidden;
  }

  .upload-zone:hover {
    border-color: rgba(212, 175, 55, 0.6);
    background: rgba(212, 175, 55, 0.04);
  }

  .upload-zone.has-preview {
    padding: 0;
    border-style: solid;
    border-color: rgba(212, 175, 55, 0.4);
  }

  .upload-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }

  .upload-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 20px;
    font-size: 22px;
  }

  .upload-label {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: #f0ead6;
    margin-bottom: 8px;
  }

  .upload-hint {
    font-size: 13px;
    color: rgba(240, 234, 214, 0.4);
    font-weight: 300;
  }

  .preview-wrapper {
    position: relative;
    width: 100%;
    max-height: 360px;
    border-radius: 18px;
    overflow: hidden;
  }

  .preview-wrapper img {
    width: 100%;
    max-height: 360px;
    object-fit: cover;
    display: block;
  }

  .preview-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(10,10,10,0.8) 0%, transparent 50%);
    display: flex;
    align-items: flex-end;
    padding: 20px 24px;
  }

  .preview-change {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #d4af37;
    background: rgba(10,10,10,0.7);
    border: 1px solid rgba(212, 175, 55, 0.3);
    padding: 8px 16px;
    border-radius: 100px;
    cursor: pointer;
  }

  .action-row {
    margin-top: 20px;
    display: flex;
    justify-content: center;
  }

  .btn-find {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    font-weight: 700;
    font-style: italic;
    color: #0a0a0a;
    background: linear-gradient(135deg, #d4af37 0%, #f0c946 50%, #d4af37 100%);
    border: none;
    padding: 16px 48px;
    border-radius: 100px;
    cursor: pointer;
    transition: all 0.25s ease;
    letter-spacing: 0.01em;
    position: relative;
    overflow: hidden;
  }

  .btn-find:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(212, 175, 55, 0.35);
  }

  .btn-find:active {
    transform: translateY(0);
  }

  .btn-find:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  .loading-state {
    text-align: center;
    padding: 56px 0;
  }

  .loading-dots {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-bottom: 20px;
  }

  .loading-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #d4af37;
    animation: pulse-dot 1.4s ease-in-out infinite;
  }

  .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
  .loading-dots span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes pulse-dot {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.1); }
  }

  .loading-text {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-style: italic;
    color: rgba(240, 234, 214, 0.6);
  }

  .error-banner {
    background: rgba(180, 40, 40, 0.12);
    border: 1px solid rgba(180, 40, 40, 0.3);
    border-radius: 12px;
    padding: 14px 20px;
    margin-top: 20px;
    font-size: 14px;
    color: #f08080;
    text-align: center;
  }

  .results-section {
    margin-top: 64px;
  }

  .results-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(212, 175, 55, 0.12);
  }

  .results-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 900;
    color: #f0ead6;
  }

  .results-count {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(212, 175, 55, 0.6);
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 16px;
  }

  .celeb-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.3s ease;
    animation: card-in 0.5s ease forwards;
    opacity: 0;
    transform: translateY(16px);
  }

  .celeb-card:nth-child(1) { animation-delay: 0.05s; }
  .celeb-card:nth-child(2) { animation-delay: 0.12s; }
  .celeb-card:nth-child(3) { animation-delay: 0.19s; }
  .celeb-card:nth-child(4) { animation-delay: 0.26s; }
  .celeb-card:nth-child(5) { animation-delay: 0.33s; }

  @keyframes card-in {
    to { opacity: 1; transform: translateY(0); }
  }

  .celeb-card:hover {
    border-color: rgba(212, 175, 55, 0.3);
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.05);
  }

  .celeb-card.rank-1 {
    border-color: rgba(212, 175, 55, 0.4);
    background: rgba(212, 175, 55, 0.04);
  }

  .celeb-img-wrap {
    position: relative;
    padding-top: 130%;
    background: rgba(255, 255, 255, 0.04);
    overflow: hidden;
  }

  .celeb-img-wrap img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s ease;
  }

  .celeb-card:hover .celeb-img-wrap img {
    transform: scale(1.05);
  }

  .rank-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    z-index: 2;
  }

  .rank-badge.gold {
    background: #d4af37;
    color: #0a0a0a;
  }

  .rank-badge.silver {
    background: rgba(200, 200, 200, 0.85);
    color: #1a1a1a;
  }

  .rank-badge.bronze {
    background: rgba(180, 100, 40, 0.85);
    color: #fff;
  }

  .rank-badge.default {
    background: rgba(30, 30, 30, 0.85);
    color: rgba(240, 234, 214, 0.7);
    border: 1px solid rgba(255,255,255,0.1);
  }

  .celeb-info {
    padding: 14px 14px 16px;
  }

  .celeb-name {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 700;
    color: #f0ead6;
    margin-bottom: 8px;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .similarity-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .similarity-bar-bg {
    flex: 1;
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
  }

  .similarity-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #d4af37, #f0c946);
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .similarity-pct {
    font-size: 12px;
    font-weight: 500;
    color: #d4af37;
    min-width: 34px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .best-match-tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0a0a0a;
    background: #d4af37;
    padding: 3px 8px;
    border-radius: 100px;
    margin-bottom: 8px;
  }

  .divider {
    border: none;
    border-top: 1px solid rgba(212, 175, 55, 0.08);
    margin: 48px 0 0;
  }
`;

const rankBadgeClass = (i) => {
  if (i === 0) return 'gold';
  if (i === 1) return 'silver';
  if (i === 2) return 'bronze';
  return 'default';
};

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResults([]);
      setError("");
    }
  };

  const handleOnSubmit = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("image", image);
    try {
      setLoading(true);
      setError("");
      const response = await axios.post(
        "http://127.0.0.1:5000/recognize",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResults(response.data.matches);
    } catch (e) {
      console.error(e);
      setError(e.response?.data?.error || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        <div className="bg-grid" />
        <div className="bg-glow" />

        <div className="container">
          {/* Header */}
          <header className="header">
            <p className="header-eyebrow">✦ AI Face Recognition ✦</p>
            <h1 className="header-title">
              Who's Your<br /><span>Celebrity Twin?</span>
            </h1>
            <p className="header-sub">
              Upload a portrait and our AI will match you to the five most similar celebrities.
            </p>
          </header>

          {/* Upload Zone */}
          <div
            className={`upload-zone ${preview ? 'has-preview' : ''}`}
            onClick={!preview ? () => inputRef.current?.click() : undefined}
          >
            {!preview ? (
              <>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                />
                <div className="upload-icon">📷</div>
                <p className="upload-label">Drop your photo here</p>
                <p className="upload-hint">PNG, JPG or WEBP · Best results with a clear frontal portrait</p>
              </>
            ) : (
              <div className="preview-wrapper">
                <img src={preview} alt="Your uploaded photo" />
                <div className="preview-overlay">
                  <label className="preview-change">
                    Change photo
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                      style={{ position: 'absolute', opacity: 0, inset: 0, cursor: 'pointer' }}
                    />
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && <div className="error-banner">{error}</div>}

          {/* CTA */}
          <div className="action-row">
            <button
              className="btn-find"
              onClick={handleOnSubmit}
              disabled={!image || loading}
            >
              {loading ? 'Analyzing…' : 'Find My Celebrity Twin →'}
            </button>
          </div>

          {/* Loading */}
          {loading && (
            <div className="loading-state">
              <div className="loading-dots">
                <span /><span /><span />
              </div>
              <p className="loading-text">Scanning facial features…</p>
            </div>
          )}

          {/* Results */}
          {results.length > 0 && (
            <section className="results-section">
              <div className="results-header">
                <h2 className="results-title">Your Matches</h2>
                <span className="results-count">Top {results.length} results</span>
              </div>
              <div className="results-grid">
                {results.map((item, index) => (
                  <div
                    key={index}
                    className={`celeb-card ${index === 0 ? 'rank-1' : ''}`}
                  >
                    <div className="celeb-img-wrap">
                      <img src={item.image} alt={item.name} />
                      <div className={`rank-badge ${rankBadgeClass(index)}`}>
                        #{index + 1}
                      </div>
                    </div>
                    <div className="celeb-info">
                      {index === 0 && <span className="best-match-tag">Best match</span>}
                      <p className="celeb-name">{item.name}</p>
                      <div className="similarity-row">
                        <div className="similarity-bar-bg">
                          <div
                            className="similarity-bar-fill"
                            style={{ width: `${item.similarity}%` }}
                          />
                        </div>
                        <span className="similarity-pct">{item.similarity}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <hr className="divider" />
            </section>
          )}
        </div>
      </div>
    </>
  );
}