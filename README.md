# Audio Analysis GUI

## Overview
This project is a GUI-based application for analyzing audio files using **PyTorch** and **Torchaudio**. The application allows users to load an audio file, process it, and analyze its waveform and spectrogram using machine learning techniques.

## Features
- Load and analyze audio files (`.wav`, `.mp3`, etc.)
- Display waveform and spectrogram visualization
- Use PyTorch model to extract features and classify audio
- Simple and intuitive GUI using **Tkinter**

## Requirements
Ensure you have the following dependencies installed before running the application:
```bash
pip install torch torchaudio numpy matplotlib librosa tkinter sounddevice
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/audio-analysis-gui.git
   cd audio-analysis-gui
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Click **Load Audio** to select an audio file.
3. The waveform and spectrogram will be displayed.
4. The model will analyze the audio and provide classification results.

## Future Improvements
- Add support for more audio formats (`.flac`, `.ogg`)
- Improve model accuracy with additional training data
- Implement real-time audio analysis using microphone input
- Enhance UI/UX with additional visualization options

![Screenshot 2025-02-07 231443](https://github.com/user-attachments/assets/b92c6456-5fc5-4881-a285-ce5ad8d3780c)

