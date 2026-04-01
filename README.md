# 👁️ Eye-Blink to Morse Code Decoder

A real-time system that converts eye blinks into Morse code and decoded text using webcam input. This project combines **MediaPipe FaceMesh** for geometric eye analysis and **YOLOv26-cls** for deep learning-based eye state classification with hybrid confidence scoring.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v26-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-IndoBERT-yellow.svg)

## 🌟 Features

- **Real-time Eye Tracking**: Uses MediaPipe FaceMesh for precise facial landmark detection
- **Hybrid Classification**: Combines Eye Aspect Ratio (EAR) with YOLO deep learning classification
- **Morse Code Decoding**: Converts blink patterns (short/long) to dots and dashes
- **Mandatory 2-Stage Calibration**:
   - Stage 1: EAR calibration (open-eye samples, then closed-eye samples)
   - Stage 2: Dot/Dash calibration (short and long blink timing)
- **Streamlit Web Interface**: User-friendly interface with live video feed and real-time feedback
- **Multiple YOLO Models**: Choose between nano, small, and medium models based on performance needs
- **Second-based Timing**: Letter and word gap thresholds are configured in seconds
- **Line-by-line Word Output**: Each completed word is moved to a new line
- **NLP Text Correction**: IndoBERT Seq2Seq model for automatic Indonesian text correction (applied per word)

## 📁 Project Structure

```
eye-blink-decoder/
├── implementation.py          # Main application code
├── preprocess_adaptive_equalization.py  # Dataset CLAHE preprocessing script
├── requirements.txt           # Python dependencies
├── face_landmarker.task       # MediaPipe face landmark model
├── yolo26n-cls.pt            # YOLO nano classification model
├── yolo26s-cls.pt            # YOLO small classification model
├── yolo26m-cls.pt            # YOLO medium classification model
├── train yolo n.ipynb        # Training notebook (nano)
├── train yolo s.ipynb        # Training notebook (small)
├── train yolo m.ipynb        # Training notebook (medium)
├── indoBERT-best-corrector/  # Local IndoBERT model cache (optional)
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
   git clone https://github.com/ZerXXX0/eye-blink-decoder.git
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

### Preprocessing `blinkblink-6` with Adaptive Equalization

To apply Roboflow-like adaptive equalization (CLAHE) to all images in `blinkblink-6/`:

```bash
python preprocess_adaptive_equalization.py
```

Output is written to `blinkblink-6-preprocessed/` with the same train/valid/test and class folder structure.

Optional parameters:

```bash
python preprocess_adaptive_equalization.py --clip-limit 2.0 --tile-grid-size 8 --overwrite
```

### How to Use

1. **Run Calibration First** (required):
   - Click **Begin Cal.**
   - **Stage 1 - EAR Open**: keep eyes open naturally until target sample count is reached
   - Click **Next Step**
   - **Stage 2 - EAR Closed**: keep eyes closed until target sample count is reached
   - Click **Next Step**
   - **Stage 3 - Dot/Dash**: blink short for dots, then long for dashes
2. **Start Detection**: click **▶️ Start Detection** after calibration is completed
3. **Blink to Communicate**:
   - **Short blink** = Dot (.)
   - **Long blink** = Dash (-)
   - **Pause** = Letter/word separator (configured in seconds)
4. **View Results**:
   - Decoded words are separated line-by-line
   - If NLP is enabled, correction is applied per word
5. **Stop Detection**:
   - Pressing **⏹️ Stop Detection** clears decoded text

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
| Letter Gap | Seconds to wait before confirming letter | 1.5 |
| Word Gap | Seconds to wait before committing a word break | 3.0 |
| EAR Min | Minimum Eye Aspect Ratio (closed) | 0.15 |
| EAR Max | Maximum Eye Aspect Ratio (open) | 0.35 |
| EAR Frames per State | Number of samples used for EAR open/closed calibration | 25 |
| Blinks per Type | Number of short blinks and long blinks for timing calibration | 3 |
| NLP Correction | Enable IndoBERT text correction | Off |

## 🤖 NLP Text Correction

The system includes an **IndoBERT Seq2Seq** model for automatic Indonesian text correction. This feature helps fix common typos and abbreviations in decoded Morse text.

### How It Works

- **Model**: [ZerXXX/indobert-corrector](https://huggingface.co/ZerXXX/indobert-corrector/tree/main/indoBERT-best-corrector) (hosted on Hugging Face Hub)
- **Subfolder**: `indoBERT-best-corrector` (model files are in this subfolder)
- **Architecture**: BERT2BERT Encoder-Decoder (Seq2Seq) based on [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1)
- **Inference**: Beam search with deterministic output (num_beams=4)
- **Token IDs**: decoder_start=2, eos=3, pad=0 (from config.json)
- **Caching**: Model loads once and is cached for the session

### Example Corrections

| Input | Output |
|-------|--------|
| `slmt pagi` | `selamat pagi` |
| `trm ksh` | `terima kasih` |
| `ap kbr` | `apa kabar` |

### Usage

1. Enable "NLP Correction" checkbox in the sidebar
2. Decoded text will be corrected per word (not as one full paragraph)
3. Toggle off to see raw decoded output

> **Note**: The model is downloaded from Hugging Face Hub on first use (~1.1GB). Subsequent runs use the cached version.

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
4. **YOLO Preprocessing**: Training-matched preprocessing applied to eye crops
5. **YOLO Classification**: Deep learning model classifies eye state
6. **Confidence Fusion**: Hybrid scoring combines EAR and YOLO confidence
7. **Blink Detection**: Temporal analysis identifies blink events
8. **Morse Decoding**: Blink patterns converted to text

### YOLO Preprocessing Pipeline

The YOLO classifier uses a preprocessing pipeline that exactly matches the training augmentation to ensure consistent confidence scores:

```python
# Preprocessing steps (in order):
1. Convert BGR → RGB
2. Resize to 512×512 (stretch, no aspect ratio preservation)
3. Apply global histogram equalization on luminance channel (LAB color space)
4. Return to YOLO (normalization handled internally)
```

| Step | Description |
|------|-------------|
| Color Conversion | BGR to RGB for YOLO compatibility |
| Resize | Stretch to 512×512 (no letterbox) |
| Histogram Equalization | Global equalization on L channel (LAB) |
| Normalization | Handled internally by YOLO |

> **Note**: This preprocessing is applied **only** to YOLO input images. FaceMesh, EAR calculation, and blink logic remain unaffected.

### Key Technologies

- **MediaPipe**: Real-time face mesh tracking
- **YOLO v26**: State-of-the-art image classification
- **Streamlit**: Interactive web application framework
- **OpenCV**: Image processing and webcam capture
- **Hugging Face Transformers**: IndoBERT Seq2Seq for NLP correction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is developed by **AI Lab - Tel-U** (January 2026).

## 📝 Changelog

### [2026-01-31] - YOLO Preprocessing Alignment
- **Added**: `preprocess_for_yolo()` function in `implementation.py` to match training preprocessing
- **Changed**: YOLO inference now applies BGR→RGB conversion, 512×512 stretch resize, and global histogram equalization on luminance channel
- **Fixed**: Train-inference distribution mismatch causing unstable confidence scores
- **Updated**: `test_pipelines.py` imports to include preprocessing function

### [2026-04-01] - Calibration and Decoding UX Update
- **Added**: Mandatory multi-stage calibration (EAR open/closed, then dot/dash)
- **Changed**: Letter and word gap settings from frame-based to second-based thresholds
- **Changed**: Word boundary output now places each completed word on a new line
- **Changed**: NLP correction now runs on each word token
- **Changed**: Stopping detection clears decoded text immediately

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh detection
- [Ultralytics](https://ultralytics.com/) for YOLO implementation
- [Streamlit](https://streamlit.io/) for the web framework
- [Hugging Face](https://huggingface.co/) for Transformers and model hosting
- [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) for Indonesian language model

---

<p align="center">
  Made with ❤️ by FG AI Lab - Tel-U
</p>
