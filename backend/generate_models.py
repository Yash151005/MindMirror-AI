"""
Generate initial placeholder .pkl models for MindMirror AI MVP
This creates functional models that can be used for immediate testing
Replace these with real trained models from Google Colab for production
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("=" * 60)
print("MindMirror AI - Model Generator")
print("=" * 60)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# ============================================================================
# Generate Face Stress Model
# ============================================================================
print("\n🔧 Generating Face Stress Model...")

# Feature dimensions matching the backend preprocessing
# 48x48 pixels = 2304 + 64 histogram bins + 3 statistical features = 2371 features
n_face_features = 2371

# Create synthetic training data for demonstration
# In production, replace with real labeled face images
n_samples = 500
X_face = np.random.randn(n_samples, n_face_features)
y_face = np.random.randint(0, 2, n_samples)  # Binary: 0=not stressed, 1=stressed

# Create and train model
scaler_face = StandardScaler()
X_face_scaled = scaler_face.fit_transform(X_face)

model_face = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model_face.fit(X_face_scaled, y_face)

# Create pipeline
face_pipeline = Pipeline([
    ('scaler', scaler_face),
    ('classifier', model_face)
])

# Save model
face_model_path = 'models/face_stress_model.pkl'
with open(face_model_path, 'wb') as f:
    pickle.dump(face_pipeline, f)

print(f"✓ Face model saved: {face_model_path}")
print(f"  - Features: {n_face_features}")
print(f"  - Training samples: {n_samples}")
print(f"  - Model size: {os.path.getsize(face_model_path) / 1024:.2f} KB")

# Test the model
test_features = np.random.randn(1, n_face_features)
prediction = face_pipeline.predict(test_features)
probability = face_pipeline.predict_proba(test_features)
print(f"  - Test prediction: {prediction[0]} (probabilities: {probability[0]})")

# ============================================================================
# Generate Voice Stress Model
# ============================================================================
print("\n🔧 Generating Voice Stress Model...")

# Feature dimensions matching the backend preprocessing
# 13 MFCCs (mean) + 13 MFCCs (std) + 2 spectral centroid + 2 ZCR + 12 chroma (mean) + 12 chroma (std) + 2 RMS = 56 features
n_voice_features = 56

# Create synthetic training data for demonstration
# In production, replace with real labeled audio samples
X_voice = np.random.randn(n_samples, n_voice_features)
y_voice = np.random.randint(0, 2, n_samples)  # Binary: 0=calm, 1=stressed

# Create and train model
scaler_voice = StandardScaler()
X_voice_scaled = scaler_voice.fit_transform(X_voice)

model_voice = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model_voice.fit(X_voice_scaled, y_voice)

# Create pipeline
voice_pipeline = Pipeline([
    ('scaler', scaler_voice),
    ('classifier', model_voice)
])

# Save model
voice_model_path = 'models/voice_stress_model.pkl'
with open(voice_model_path, 'wb') as f:
    pickle.dump(voice_pipeline, f)

print(f"✓ Voice model saved: {voice_model_path}")
print(f"  - Features: {n_voice_features}")
print(f"  - Training samples: {n_samples}")
print(f"  - Model size: {os.path.getsize(voice_model_path) / 1024:.2f} KB")

# Test the model
test_features = np.random.randn(1, n_voice_features)
prediction = voice_pipeline.predict(test_features)
probability = voice_pipeline.predict_proba(test_features)
print(f"  - Test prediction: {prediction[0]} (probabilities: {probability[0]})")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("✅ Model Generation Complete!")
print("=" * 60)
print("\n📝 IMPORTANT NOTES:")
print("  • These are placeholder models trained on synthetic data")
print("  • They will produce random but valid predictions")
print("  • For PRODUCTION use, train models on real data using:")
print("    - training/face_model_training.ipynb (Google Colab)")
print("    - training/voice_model_training.ipynb (Google Colab)")
print("\n🚀 Next Steps:")
print("  1. Install backend dependencies: pip install -r requirements.txt")
print("  2. Start backend: python app.py")
print("  3. Open frontend: frontend/index.html in browser")
print("  4. Test the complete pipeline!")
print("\n⚠️  Replace these models with real trained models before deployment!")
print("=" * 60)
