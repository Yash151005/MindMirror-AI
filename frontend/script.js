// Configuration
// Use relative URL since frontend is now served from backend
const API_URL = window.location.origin;

// State
let capturedImage = null;
let capturedAudio = null;
let mediaRecorder = null;
let audioChunks = [];
let stream = null;

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const capturedImageEl = document.getElementById('captured-image');
const captureBtn = document.getElementById('capture-btn');
const retakeBtn = document.getElementById('retake-btn');
const faceStatus = document.getElementById('face-status');

const recordBtn = document.getElementById('record-btn');
const rerecordBtn = document.getElementById('rerecord-btn');
const audioStatus = document.getElementById('audio-status');
const audioVisualizer = document.getElementById('audio-visualizer');
const waveBars = audioVisualizer.querySelector('.wave-bars');
const timerEl = document.getElementById('timer');

const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

// Initialize
async function init() {
    // More permissive check - allow localhost and local network IPs
    const hostname = window.location.hostname;
    const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1';
    const isLocalNetwork = hostname.startsWith('192.168.') || hostname.startsWith('10.') || hostname.startsWith('172.');
    
    // Try to access regardless - modern browsers allow local IPs in some cases
    console.log('Initializing camera and microphone...');
    
    try {
        // Request camera access
        faceStatus.textContent = 'Requesting camera access...';
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false 
        });
        video.srcObject = stream;
        faceStatus.textContent = '✓ Camera ready';
        faceStatus.style.color = 'green';
    } catch (err) {
        console.error('Camera access error:', err);
        let errorMsg = 'Camera access denied';
        if (err.name === 'NotAllowedError') {
            errorMsg = '❌ Camera permission denied. Please allow camera access and reload.';
        } else if (err.name === 'NotFoundError') {
            errorMsg = '❌ No camera found on this device';
        } else if (err.name === 'NotReadableError') {
            errorMsg = '❌ Camera is being used by another application';
        } else if (err.name === 'NotSupportedError' || err.name === 'TypeError') {
            errorMsg = '❌ Use localhost or HTTPS: Try http://localhost:5000 instead';
        }
        faceStatus.textContent = errorMsg;
        faceStatus.style.color = 'red';
    }
    
    // Check microphone access
    try {
        audioStatus.textContent = 'Requesting microphone access...';
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioStream.getTracks().forEach(track => track.stop());
        audioStatus.textContent = '✓ Microphone ready';
        audioStatus.style.color = 'green';
    } catch (err) {
        console.error('Microphone access error:', err);
        let errorMsg = 'Microphone access denied';
        if (err.name === 'NotAllowedError') {
            errorMsg = '❌ Microphone permission denied. Please allow microphone access and reload.';
        } else if (err.name === 'NotFoundError') {
            errorMsg = '❌ No microphone found on this device';
        } else if (err.name === 'NotReadableError') {
            errorMsg = '❌ Microphone is being used by another application';
        } else if (err.name === 'NotSupportedError' || err.name === 'TypeError') {
            errorMsg = '❌ Use localhost or HTTPS: Try http://localhost:5000 instead';
        }
        audioStatus.textContent = errorMsg;
        audioStatus.style.color = 'red';
    }
}

// Face Capture
captureBtn.addEventListener('click', () => {
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Capture frame from video
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    capturedImage = canvas.toDataURL('image/jpeg', 0.8);
    
    // Show captured image
    capturedImageEl.src = capturedImage;
    capturedImageEl.style.display = 'block';
    video.style.display = 'none';
    
    // Update UI
    captureBtn.style.display = 'none';
    retakeBtn.style.display = 'inline-block';
    faceStatus.textContent = '✓ Photo captured';
    faceStatus.classList.add('success');
    
    checkReadyToAnalyze();
});

retakeBtn.addEventListener('click', () => {
    capturedImage = null;
    capturedImageEl.style.display = 'none';
    video.style.display = 'block';
    captureBtn.style.display = 'inline-block';
    retakeBtn.style.display = 'none';
    faceStatus.textContent = 'Camera ready';
    faceStatus.classList.remove('success');
    
    checkReadyToAnalyze();
});

// Audio Recording
recordBtn.addEventListener('click', async () => {
    try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Reset state
        audioChunks = [];
        
        // Try to use WAV format if supported, otherwise fall back to webm
        let mimeType = 'audio/webm';
        if (MediaRecorder.isTypeSupported('audio/wav')) {
            mimeType = 'audio/wav';
        } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            mimeType = 'audio/webm;codecs=opus';
        }
        
        console.log('Recording with MIME type:', mimeType);
        
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(audioStream, { mimeType: mimeType });
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            // Stop all tracks
            audioStream.getTracks().forEach(track => track.stop());
            
            // Convert to base64
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            const reader = new FileReader();
            reader.onloadend = () => {
                capturedAudio = reader.result;
                audioStatus.textContent = '✓ Audio recorded';
                audioStatus.classList.add('success');
                checkReadyToAnalyze();
            };
            reader.readAsDataURL(audioBlob);
            
            // Update UI
            waveBars.classList.remove('recording');
            recordBtn.style.display = 'inline-block';
            rerecordBtn.style.display = 'inline-block';
        };
        
        // Start recording
        mediaRecorder.start();
        recordBtn.style.display = 'none';
        audioStatus.textContent = 'Recording...';
        audioStatus.classList.remove('success');
        waveBars.classList.add('recording');
        
        // Record for 3 seconds with countdown
        let timeLeft = 3.0;
        const countdownInterval = setInterval(() => {
            timeLeft -= 0.1;
            timerEl.textContent = timeLeft.toFixed(1) + 's';
            
            if (timeLeft <= 0) {
                clearInterval(countdownInterval);
                timerEl.textContent = '3.0s';
                mediaRecorder.stop();
            }
        }, 100);
        
    } catch (err) {
        console.error('Audio recording error:', err);
        audioStatus.textContent = 'Recording failed';
        audioStatus.style.color = 'red';
    }
});

rerecordBtn.addEventListener('click', () => {
    capturedAudio = null;
    rerecordBtn.style.display = 'none';
    audioStatus.textContent = 'Microphone ready';
    audioStatus.classList.remove('success');
    timerEl.textContent = '3.0s';
    
    checkReadyToAnalyze();
});

// Check if ready to analyze
function checkReadyToAnalyze() {
    if (capturedImage && capturedAudio) {
        analyzeBtn.disabled = false;
    } else {
        analyzeBtn.disabled = true;
    }
}

// Analyze
analyzeBtn.addEventListener('click', async () => {
    // Show loading
    loadingOverlay.style.display = 'flex';
    
    try {
        // Send to backend
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: capturedImage,
                audio: capturedAudio
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Analysis failed');
        }
        
        const result = await response.json();
        
        // Hide loading
        loadingOverlay.style.display = 'none';
        
        // Display results
        displayResults(result);
        
    } catch (err) {
        console.error('Analysis error:', err);
        loadingOverlay.style.display = 'none';
        alert('Analysis failed: ' + err.message);
    }
});

// Display Results
function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Wellness Score
    const score = result.wellness_score;
    document.getElementById('wellness-score').textContent = score.toFixed(0);
    document.getElementById('interpretation').textContent = result.interpretation;
    
    // Animate score circle
    const circumference = 2 * Math.PI * 90;
    const offset = circumference - (score / 100) * circumference;
    const scoreCircle = document.getElementById('score-circle');
    scoreCircle.style.strokeDashoffset = offset;
    
    // Set color based on score
    let color;
    if (score >= 80) color = '#28a745';
    else if (score >= 60) color = '#ffc107';
    else if (score >= 40) color = '#fd7e14';
    else color = '#dc3545';
    scoreCircle.style.stroke = color;
    
    // Face Analysis
    document.getElementById('face-stress').textContent = 
        (result.face_analysis.stress_probability * 100).toFixed(1) + '%';
    document.getElementById('face-confidence').textContent = 
        (result.face_analysis.confidence * 100).toFixed(1) + '%';
    
    // Voice Analysis
    document.getElementById('voice-stress').textContent = 
        (result.voice_analysis.stress_probability * 100).toFixed(1) + '%';
    document.getElementById('voice-confidence').textContent = 
        (result.voice_analysis.confidence * 100).toFixed(1) + '%';
    
    // Timestamp
    const timestamp = new Date(result.timestamp);
    document.getElementById('timestamp').textContent = 
        'Analysis completed at ' + timestamp.toLocaleString();
}

// New Analysis
newAnalysisBtn.addEventListener('click', () => {
    // Reset state
    capturedImage = null;
    capturedAudio = null;
    
    // Reset UI
    resultsSection.style.display = 'none';
    capturedImageEl.style.display = 'none';
    video.style.display = 'block';
    captureBtn.style.display = 'inline-block';
    retakeBtn.style.display = 'none';
    faceStatus.textContent = 'Camera ready';
    faceStatus.classList.remove('success');
    
    recordBtn.style.display = 'inline-block';
    rerecordBtn.style.display = 'none';
    audioStatus.textContent = 'Microphone ready';
    audioStatus.classList.remove('success');
    
    checkReadyToAnalyze();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Initialize on load
init();
