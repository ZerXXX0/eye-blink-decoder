# 👁️ Eye-Blink to Morse Code Decoder

A real-time system that converts eye blinks into Morse code and decoded text using webcam input. This project combines **MediaPipe FaceMesh** for geometric eye analysis and **YOLOv26-cls** for deep learning-based eye state classification with hybrid confidence scoring.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v26-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)

## 🌟 Features

- **Real-time Eye Tracking**: Uses MediaPipe FaceMesh for precise facial landmark detection
- **Hybrid Classification**: Combines Eye Aspect Ratio (EAR) with YOLO deep learning classification
- **Morse Code Decoding**: Converts blink patterns (short/long) to dots and dashes
- **Calibration System**: Personalized blink duration calibration for accurate detection
- **Streamlit Web Interface**: User-friendly interface with live video feed and real-time feedback
- **Multiple YOLO Models**: Choose between nano, small, and medium models based on performance needs

## 📁 Project Structure

```
eye-blink-decoder/
├── implementation.py          # Main application code
├── requirements.txt           # Python dependencies
├── face_landmarker.task       # MediaPipe face landmark model
├── yolo26n-cls.pt            # YOLO nano classification model
├── yolo26s-cls.pt            # YOLO small classification model
├── yolo26m-cls.pt            # YOLO medium classification model
├── train yolo n.ipynb        # Training notebook (nano)
├── train yolo s.ipynb        # Training notebook (small)
├── train yolo m.ipynb        # Training notebook (medium)
└── runs/
    └── classify/
        ├── nano_100/         # Nano model training results
        ├── small_100/        # Small model training results
        └── medium_100/       # Medium model training results
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- (Optional) CUDA-compatible GPU for faster inference

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/eye-blink-decoder.git
   cd eye-blink-decoder
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Running the Application

```bash
streamlit run implementation.py
```

This will open the web interface in your default browser.

### How to Use

1. **Start Detection**: Click the "▶️ Start Detection" button to begin
2. **Calibration** (Recommended): 
   - Click "Start Calibration" in the sidebar
   - Perform short blinks (dots) when prompted
   - Perform long blinks (dashes) when prompted
3. **Blink to Communicate**:
   - **Short blink** = Dot (.)
   - **Long blink** = Dash (-)
   - **Pause** = Letter/word separator
4. **View Results**: Decoded text appears in real-time

### Morse Code Reference

| Letter | Code | Letter | Code | Number | Code |
|--------|------|--------|------|--------|------|
| A | .- | N | -. | 1 | .---- |
| B | -... | O | --- | 2 | ..--- |
| C | -.-. | P | .--. | 3 | ...-- |
| D | -.. | Q | --.- | 4 | ....- |
| E | . | R | .-. | 5 | ..... |
| F | ..-. | S | ... | 6 | -.... |
| G | --. | T | - | 7 | --... |
| H | .... | U | ..- | 8 | ---.. |
| I | .. | V | ...- | 9 | ----. |
| J | .--- | W | .-- | 0 | ----- |
| K | -.- | X | -..- |
| L | .-.. | Y | -.-- |
| M | -- | Z | --.. |

## ⚙️ Configuration

The sidebar provides various configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Alpha | Weight for YOLO confidence (vs EAR) | 0.4 |
| Blink Threshold | Confidence threshold for blink detection | 0.5 |
| Letter Gap | Frames to wait before confirming letter | 45 |
| Word Gap | Frames to wait before adding space | 90 |
| EAR Min | Minimum Eye Aspect Ratio (closed) | 0.15 |
| EAR Max | Maximum Eye Aspect Ratio (open) | 0.35 |

## 🧠 Model Information

### Available Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `yolo26n-cls.pt` | Nano | Fastest | Good | Real-time on CPU |
| `yolo26s-cls.pt` | Small | Fast | Better | Balanced performance |
| `yolo26m-cls.pt` | Medium | Moderate | Best | GPU acceleration |

### Training

Pre-trained models are included. To retrain with your own data:

1. Open the respective training notebook (`train yolo n/s/m.ipynb`)
2. Prepare your eye image dataset (open/closed classes)
3. Run the training cells
4. Trained weights are saved in `runs/classify/*/weights/`

## 🔧 Technical Details

### Eye Detection Pipeline

1. **Face Detection**: MediaPipe FaceMesh detects 468 facial landmarks
2. **Eye Extraction**: Eye regions are cropped using landmark indices
3. **EAR Calculation**: Eye Aspect Ratio computed from geometric landmarks
4. **YOLO Classification**: Deep learning model classifies eye state
5. **Confidence Fusion**: Hybrid scoring combines EAR and YOLO confidence
6. **Blink Detection**: Temporal analysis identifies blink events
7. **Morse Decoding**: Blink patterns converted to text

### Key Technologies

- **MediaPipe**: Real-time face mesh tracking
- **YOLO v26**: State-of-the-art image classification
- **Streamlit**: Interactive web application framework
- **OpenCV**: Image processing and webcam capture

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is developed by **AI Lab - Tel-U** (January 2026).

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh detection
- [Ultralytics](https://ultralytics.com/) for YOLO implementation
- [Streamlit](https://streamlit.io/) for the web framework

---

<p align="center">
  Made with ❤️ by AI Lab - Tel-U
</p>
