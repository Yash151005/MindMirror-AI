# MindMirror AI - MVP

Complete stress detection pipeline using real ML models, OpenCV, and librosa.

## 🎯 Features

- **Real Face Analysis**: OpenCV preprocessing + .pkl model inference
- **Real Voice Analysis**: Librosa MFCC extraction + .pkl model inference
- **Combined Wellness Score**: Transparent formula using actual model outputs
- **Privacy-Safe**: Only stores numerical results, never raw images/audio
- **Easy Model Updates**: Train models on Google Colab, download, and replace

## 🚀 Quick Start

### 1. Install Backend Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 2. Generate Initial Models

```powershell
python generate_models.py
```

This creates placeholder models for immediate testing. Replace with real models later.

### 3. Run Backend

```powershell
python app.py
```

Backend runs on `http://localhost:5000`

### 4. Open Frontend

Open `frontend/index.html` in your browser or use a local server:

```powershell
cd frontend
python -m http.server 8000
```

Then visit `http://localhost:8000`

## 📊 Training Real Models

### Option 1: Use Colab Notebooks (Recommended)

1. Upload `training/face_stress_model_training.ipynb` to Google Colab
2. Upload `training/voice_stress_model_training.ipynb` to Google Colab
3. Run all cells in each notebook
4. Download the generated `.pkl` files
5. Place them in `backend/models/`

### Option 2: Train Locally

```powershell
cd training
jupyter notebook face_stress_model_training.ipynb
jupyter notebook voice_stress_model_training.ipynb
```

## 🏗️ Project Structure

```
MindMirror AI/
├── backend/
│   ├── app.py                  # Flask API with real ML pipeline
│   ├── requirements.txt        # Python dependencies
│   ├── generate_models.py      # Quick model generator
│   └── models/                 # .pkl model files
│       ├── face_stress_model.pkl
│       └── voice_stress_model.pkl
├── frontend/
│   ├── index.html              # Web interface
│   ├── script.js               # Webcam/audio capture + API calls
│   └── style.css               # Styling
└── training/
    ├── face_stress_model_training.ipynb   # Colab notebook
    └── voice_stress_model_training.ipynb  # Colab notebook
```

## 🔬 How It Works

### Face Analysis Pipeline

1. **Capture**: Webcam captures photo in browser
2. **Preprocessing** (OpenCV):
   - Convert to grayscale
   - Resize to 48x48
   - Normalize pixels
   - Extract histogram (64 bins)
   - Calculate statistics (mean, std, edge density)
3. **Inference**: Feed 2371 features into .pkl Random Forest model
4. **Output**: Stress probability (0-1)

### Voice Analysis Pipeline

1. **Capture**: Microphone records 3-5 second audio in browser
2. **Preprocessing** (librosa):
   - Extract 13 MFCC coefficients (mean + std)
   - Spectral centroid (mean + std)
   - Zero crossing rate (mean + std)
   - Chroma features (12 mean + 12 std)
   - RMS energy (mean + std)
3. **Inference**: Feed 56 features into .pkl Random Forest model
4. **Output**: Stress probability (0-1)

### Wellness Score

```python
wellness_score = (1 - avg_stress_probability) * 100
```

Where `avg_stress_probability = (face_stress + voice_stress) / 2`

## 🎓 Training Data Sources

### Face Data
- **FER2013**: Facial emotion recognition dataset
- **CK+**: Extended Cohn-Kanade dataset
- **AffectNet**: Large facial expression dataset

### Voice Data
- **RAVDESS**: Speech emotion dataset
- **TESS**: Toronto emotional speech set
- **CREMA-D**: Crowd-sourced emotional multimodal actors

Map stress-related emotions (angry, fear, disgust) to stressed=1, others to stressed=0.

## 🔒 Privacy & Security

- ✅ No raw images stored on server
- ✅ No audio files stored on server
- ✅ Only numerical features and results saved
- ✅ All processing happens server-side
- ✅ Results contain only timestamps and probabilities

## 🛠️ Development Workflow

1. **Initial Testing**: Use `generate_models.py` for placeholder models
2. **Data Collection**: Gather real face images and voice recordings
3. **Training**: Use Colab notebooks to train on real data
4. **Deployment**: Download `.pkl` files and replace in `backend/models/`
5. **Iteration**: Retrain and update models as needed

## � API Endpoints

### `POST /analyze`
Analyzes face and voice data.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "audio": "data:audio/webm;base64,..."
}
```

**Response:**
```json
{
  "wellness_score": 75.5,
  "face_analysis": {
    "prediction": 0,
    "stress_probability": 0.25,
    "confidence": 0.85
  },
  "voice_analysis": {
    "prediction": 0,
    "stress_probability": 0.24,
    "confidence": 0.82
  },
  "timestamp": "2025-11-28T10:30:00",
  "interpretation": "Good - Moderate stress levels detected"
}
```

### `GET /health`
Health check endpoint.

### `GET /history`
Get last 10 analysis results.

## � Troubleshooting

### Models not loading
- Ensure `.pkl` files exist in `backend/models/`
- Run `python generate_models.py` to create placeholder models

### Camera/microphone not working
- Grant browser permissions
- Use HTTPS or localhost
- Check browser console for errors

### CORS errors
- Ensure Flask-CORS is installed
- Backend and frontend must be on same domain or CORS enabled

## 📦 Dependencies

### Backend
- Flask 3.0.0
- OpenCV 4.8.1
- librosa 0.10.1
- scikit-learn 1.3.2
- numpy 1.24.3

### Frontend
- Vanilla JavaScript (no frameworks)
- Modern browser with MediaDevices API support

## 🎯 Hackathon Tips

1. **Quick Start**: Use placeholder models to demo immediately
2. **Real Models**: Train during hackathon using Colab (parallel to development)
3. **Hot Swap**: Replace models without changing code
4. **Demo Ready**: Full working pipeline in 30 minutes
5. **Iterate**: Easy to update models as you collect more data

## � License

MIT License - feel free to use for your hackathon project!

## 🤝 Contributing

This is a hackathon MVP. Contributions welcome for:
- Better feature extraction
- Additional datasets
- Model improvements
- UI enhancements
