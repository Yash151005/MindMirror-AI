# FFmpeg Installation for Audio Processing

MindMirror AI uses librosa for audio processing, which requires FFmpeg to handle WebM audio files from the browser.

## Quick Install (Windows)

### Option 1: Chocolatey (Recommended)
```powershell
choco install ffmpeg
```

### Option 2: Direct Download
1. Download FFmpeg: https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   ```powershell
   $env:Path += ";C:\ffmpeg\bin"
   [Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
   ```

### Option 3: Using pip (simpler but larger)
```powershell
pip install ffmpeg-python
```

## Verify Installation
```powershell
ffmpeg -version
```

## Alternative: No FFmpeg Needed
The backend has been updated with fallback handling that should work even without FFmpeg for most audio formats. If you still get audio errors, install FFmpeg using one of the methods above.

## Test After Installation
1. Restart the backend server
2. Record audio again in the browser
3. Click Analyze

The audio should now process successfully!
