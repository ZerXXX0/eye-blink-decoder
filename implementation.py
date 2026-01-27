"""
Real-Time Eye-Blink to Morse Code System
=========================================
MediaPipe FaceMesh + YOLOv26-cls + Streamlit

This system converts eye blinks into Morse code and decoded text using a webcam.
It combines MediaPipe FaceMesh for geometric eye analysis and YOLOv26-cls for
deep learning-based eye state classification with hybrid confidence scoring.

Author: AI Lab - Tel-U
Date: January 2026
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MpImage
import streamlit as st
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
from enum import Enum
import time
import threading
from abc import ABC, abstractmethod
import urllib.request
import os


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# MediaPipe FaceMesh eye landmark indices
# Left eye landmarks (from user's perspective, right side of image)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Right eye landmarks (from user's perspective, left side of image)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Extended eye region for cropping (includes eyebrow area)
LEFT_EYE_REGION = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_REGION = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Morse code dictionary
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0', '.-.-.-': '.', '--..--': ',', '..--..': '?',
    '.----.': "'", '-.-.--': '!', '-..-.': '/', '-.--.': '(',
    '-.--.-': ')', '.-...': '&', '---...': ':', '-.-.-.': ';',
    '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS'
}


# =============================================================================
# DATA CLASSES & ENUMS
# =============================================================================

class EyeState(Enum):
    """Enumeration of possible eye states."""
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class BlinkType(Enum):
    """Enumeration of blink types for Morse code."""
    DOT = "."
    DASH = "-"
    NONE = ""


class CalibrationMethod(Enum):
    """Available calibration methods."""
    PREDEFINED_WORD = "predefined_word"
    SINGLE_LETTER = "single_letter"
    FREE_BLINK = "free_blink"


@dataclass
class EyeData:
    """Container for eye-related data from a single frame."""
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    normalized_ear: float = 0.0
    left_crop: Optional[np.ndarray] = None
    right_crop: Optional[np.ndarray] = None
    landmarks_detected: bool = False


@dataclass
class YOLOResult:
    """Container for YOLO classification results."""
    state: EyeState = EyeState.UNKNOWN
    confidence: float = 0.0
    open_prob: float = 0.0
    closed_prob: float = 0.0


@dataclass
class BlinkEvent:
    """Container for a detected blink event."""
    start_frame: int = 0
    end_frame: int = 0
    duration_frames: int = 0
    duration_ms: float = 0.0
    blink_type: BlinkType = BlinkType.NONE
    confidence: float = 0.0


@dataclass
class SystemConfig:
    """System configuration parameters."""
    # Confidence fusion
    alpha: float = 0.4  # Weight for YOLO confidence (1-alpha for EAR)
    
    # Blink detection
    blink_threshold: float = 0.5  # Confidence threshold for blink detection
    
    # Timing (in frames at 30 FPS)
    letter_gap_frames: int = 45  # ~1.5 seconds for letter gap
    word_gap_frames: int = 90    # ~3 seconds for word gap
    
    # EAR normalization
    ear_min: float = 0.15  # Minimum EAR (closed eyes)
    ear_max: float = 0.35  # Maximum EAR (open eyes)
    
    # Smoothing
    smoothing_window: int = 5  # Frames for rolling average
    ema_alpha: float = 0.3    # EMA smoothing factor
    
    # Calibration
    calibration_blinks: int = 5  # Number of blinks for calibration
    default_blink_duration_ms: float = 200.0  # Default short blink duration
    
    # Model paths
    yolo_model_path: str = "runs/classify/nano_100/weights/best.pt"
    
    # Performance
    target_fps: int = 15
    use_gpu: bool = True


class CalibrationPhase(Enum):
    """Calibration phases."""
    NOT_STARTED = "not_started"
    COLLECTING_DOTS = "collecting_dots"
    COLLECTING_DASHES = "collecting_dashes"
    COMPLETED = "completed"


@dataclass
class CalibrationData:
    """Container for calibration results."""
    is_calibrated: bool = False
    avg_blink_duration_ms: float = 200.0  # Threshold between dot and dash
    avg_dot_duration_ms: float = 150.0
    avg_dash_duration_ms: float = 400.0
    dot_durations: List[float] = field(default_factory=list)
    dash_durations: List[float] = field(default_factory=list)
    ear_baseline_open: float = 0.3
    ear_baseline_closed: float = 0.15
    calibration_method: CalibrationMethod = CalibrationMethod.FREE_BLINK


# =============================================================================
# EYE ANALYSIS MODULE
# =============================================================================

# Download FaceLandmarker model if not exists
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def download_face_landmarker_model():
    """Download the FaceLandmarker model if it doesn't exist."""
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        print(f"Downloading FaceLandmarker model...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
        print(f"Model downloaded to {FACE_LANDMARKER_MODEL_PATH}")

class EyeAnalyzer:
    """
    Handles eye landmark detection and geometric analysis using MediaPipe FaceLandmarker (Tasks API).
    Computes Eye Aspect Ratio (EAR) and crops eye regions for YOLO inference.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the EyeAnalyzer with MediaPipe FaceLandmarker.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        # Download model if needed
        download_face_landmarker_model()
        
        # Create FaceLandmarker options
        base_options = mp_tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
    def compute_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) from landmarks.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where p1-p6 are the 6 eye landmarks in order:
        p1: outer corner, p2: upper outer, p3: upper inner,
        p4: inner corner, p5: lower inner, p6: lower outer
        
        Args:
            landmarks: Array of facial landmarks
            eye_indices: Indices of the 6 eye landmarks
            
        Returns:
            Eye Aspect Ratio value
        """
        try:
            # Extract eye points
            eye_points = landmarks[eye_indices]
            
            # Compute vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])  # p2-p6
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])  # p3-p5
            
            # Compute horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])   # p1-p4
            
            # Avoid division by zero
            if h < 1e-6:
                return 0.0
                
            ear = (v1 + v2) / (2.0 * h)
            return ear
            
        except Exception as e:
            return 0.0
    
    def normalize_ear(self, ear: float, ear_min: float = 0.15, 
                      ear_max: float = 0.35) -> float:
        """
        Normalize EAR value to [0, 1] range.
        
        Args:
            ear: Raw EAR value
            ear_min: Minimum expected EAR (closed eyes)
            ear_max: Maximum expected EAR (open eyes)
            
        Returns:
            Normalized EAR in [0, 1] range (0=closed, 1=open)
        """
        normalized = (ear - ear_min) / (ear_max - ear_min + 1e-6)
        return np.clip(normalized, 0.0, 1.0)
    
    def crop_eye_region(self, frame: np.ndarray, landmarks: np.ndarray,
                        eye_region_indices: List[int], padding: float = 0.3) -> Optional[np.ndarray]:
        """
        Crop eye region from frame based on landmark coordinates.
        
        Args:
            frame: Input frame (BGR)
            landmarks: Facial landmarks as pixel coordinates
            eye_region_indices: Indices of landmarks defining eye region
            padding: Padding ratio around the eye region
            
        Returns:
            Cropped eye region or None if cropping fails
        """
        try:
            h, w = frame.shape[:2]
            
            # Get eye region points
            eye_points = landmarks[eye_region_indices]
            
            # Compute bounding box
            x_min = int(np.min(eye_points[:, 0]))
            x_max = int(np.max(eye_points[:, 0]))
            y_min = int(np.min(eye_points[:, 1]))
            y_max = int(np.max(eye_points[:, 1]))
            
            # Add padding
            pad_x = int((x_max - x_min) * padding)
            pad_y = int((y_max - y_min) * padding)
            
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)
            
            # Crop
            crop = frame[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                return None
                
            return crop
            
        except Exception as e:
            return None
    
    def process_frame(self, frame: np.ndarray, config: SystemConfig) -> Tuple[EyeData, np.ndarray]:
        """
        Process a frame to extract eye data and annotated frame.
        
        Args:
            frame: Input frame (BGR)
            config: System configuration
            
        Returns:
            Tuple of (EyeData, annotated_frame)
        """
        eye_data = EyeData()
        annotated_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        results = self.face_landmarker.detect(mp_image)
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return eye_data, annotated_frame
        
        face_landmarks = results.face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Convert landmarks to numpy array (pixel coordinates)
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks
        ])
        
        eye_data.landmarks_detected = True
        
        # Compute EAR for both eyes
        eye_data.left_ear = self.compute_ear(landmarks, LEFT_EYE_LANDMARKS)
        eye_data.right_ear = self.compute_ear(landmarks, RIGHT_EYE_LANDMARKS)
        eye_data.avg_ear = (eye_data.left_ear + eye_data.right_ear) / 2.0
        
        # Normalize EAR
        eye_data.normalized_ear = self.normalize_ear(
            eye_data.avg_ear, config.ear_min, config.ear_max
        )
        
        # Crop eye regions
        eye_data.left_crop = self.crop_eye_region(
            frame, landmarks, LEFT_EYE_REGION
        )
        eye_data.right_crop = self.crop_eye_region(
            frame, landmarks, RIGHT_EYE_REGION
        )
        
        # Draw landmarks on annotated frame
        self._draw_eye_landmarks(annotated_frame, landmarks)
        
        return eye_data, annotated_frame
    
    def _draw_eye_landmarks(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw eye landmarks on the frame."""
        # Draw left eye
        for idx in LEFT_EYE_LANDMARKS:
            pt = tuple(landmarks[idx][:2].astype(int))
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)
        
        # Draw right eye
        for idx in RIGHT_EYE_LANDMARKS:
            pt = tuple(landmarks[idx][:2].astype(int))
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)
        
        # Connect eye landmarks
        self._draw_eye_contour(frame, landmarks, LEFT_EYE_LANDMARKS, (0, 255, 0))
        self._draw_eye_contour(frame, landmarks, RIGHT_EYE_LANDMARKS, (0, 255, 0))
    
    def _draw_eye_contour(self, frame: np.ndarray, landmarks: np.ndarray,
                          indices: List[int], color: Tuple[int, int, int]):
        """Draw eye contour connecting landmarks."""
        points = landmarks[indices][:, :2].astype(int)
        for i in range(len(points)):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % len(points)])
            cv2.line(frame, pt1, pt2, color, 1)
    
    def close(self):
        """Release resources."""
        if self.face_landmarker:
            self.face_landmarker.close()


# =============================================================================
# YOLO CLASSIFIER MODULE
# =============================================================================

class YOLOEyeClassifier:
    """
    Deep learning eye state classifier using YOLOv26-cls.
    Classifies eye crops as 'open' or 'closed'.
    """
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the YOLO classifier.
        
        Args:
            model_path: Path to the YOLOv26-cls model weights
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.class_names = ['closed', 'open']  # Adjust based on your training
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            # Set device
            if self.use_gpu:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                else:
                    print("GPU not available, falling back to CPU")
                    self.use_gpu = False
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
    
    def classify(self, image: np.ndarray) -> YOLOResult:
        """
        Classify an eye image as open or closed.
        
        Args:
            image: Eye crop image (BGR)
            
        Returns:
            YOLOResult with classification results
        """
        result = YOLOResult()
        
        if self.model is None or image is None or image.size == 0:
            return result
        
        try:
            # Resize image if needed
            if image.shape[0] < 32 or image.shape[1] < 32:
                image = cv2.resize(image, (64, 64))
            
            # Run inference
            predictions = self.model(image, verbose=False)
            
            if predictions and len(predictions) > 0:
                probs = predictions[0].probs
                
                if probs is not None:
                    # Get class probabilities
                    class_probs = probs.data.cpu().numpy()
                    
                    # Map to open/closed (adjust indices based on your model)
                    # Assuming index 0 = closed, index 1 = open
                    result.closed_prob = float(class_probs[0])
                    result.open_prob = float(class_probs[1])
                    
                    # Determine state
                    if result.open_prob > result.closed_prob:
                        result.state = EyeState.OPEN
                        result.confidence = result.open_prob
                    else:
                        result.state = EyeState.CLOSED
                        result.confidence = result.closed_prob
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
        
        return result
    
    def classify_dual_eye(self, left_crop: Optional[np.ndarray],
                          right_crop: Optional[np.ndarray]) -> YOLOResult:
        """
        Classify using both eye crops with aggregation.
        
        Args:
            left_crop: Left eye crop
            right_crop: Right eye crop
            
        Returns:
            Aggregated YOLOResult
        """
        results = []
        
        if left_crop is not None and left_crop.size > 0:
            results.append(self.classify(left_crop))
        
        if right_crop is not None and right_crop.size > 0:
            results.append(self.classify(right_crop))
        
        if not results:
            return YOLOResult()
        
        # Aggregate results (average probabilities)
        avg_result = YOLOResult()
        avg_result.open_prob = np.mean([r.open_prob for r in results])
        avg_result.closed_prob = np.mean([r.closed_prob for r in results])
        
        if avg_result.open_prob > avg_result.closed_prob:
            avg_result.state = EyeState.OPEN
            avg_result.confidence = avg_result.open_prob
        else:
            avg_result.state = EyeState.CLOSED
            avg_result.confidence = avg_result.closed_prob
        
        return avg_result


# =============================================================================
# CONFIDENCE FUSION MODULE
# =============================================================================

class ConfidenceFusion:
    """
    Fuses YOLO confidence and normalized EAR into a final confidence score.
    
    The fusion formula:
        final_confidence = α * yolo_confidence + (1 - α) * normalized_EAR
    
    Where:
        - yolo_confidence: Probability of eyes being OPEN from YOLO
        - normalized_EAR: Normalized Eye Aspect Ratio (0=closed, 1=open)
        - α (alpha): Configurable fusion weight
    """
    
    def __init__(self, smoothing_window: int = 5, ema_alpha: float = 0.3):
        """
        Initialize the confidence fusion module.
        
        Args:
            smoothing_window: Window size for rolling average
            ema_alpha: Alpha for exponential moving average
        """
        self.smoothing_window = smoothing_window
        self.ema_alpha = ema_alpha
        self.confidence_history = deque(maxlen=smoothing_window)
        self.ema_value = None
    
    def fuse(self, yolo_result: YOLOResult, normalized_ear: float,
             alpha: float) -> float:
        """
        Fuse YOLO and EAR confidence scores.
        
        Args:
            yolo_result: YOLO classification result
            normalized_ear: Normalized EAR value [0, 1]
            alpha: Fusion weight for YOLO confidence
            
        Returns:
            Fused confidence score [0, 1] (higher = more likely open)
        """
        # Use YOLO's open probability as confidence
        yolo_conf = yolo_result.open_prob if yolo_result.open_prob > 0 else 0.5
        
        # Fuse confidence scores
        fused = alpha * yolo_conf + (1 - alpha) * normalized_ear
        
        return np.clip(fused, 0.0, 1.0)
    
    def smooth_rolling(self, confidence: float) -> float:
        """
        Apply rolling average smoothing.
        
        Args:
            confidence: Raw confidence value
            
        Returns:
            Smoothed confidence value
        """
        self.confidence_history.append(confidence)
        return np.mean(self.confidence_history)
    
    def smooth_ema(self, confidence: float) -> float:
        """
        Apply exponential moving average smoothing.
        
        Args:
            confidence: Raw confidence value
            
        Returns:
            Smoothed confidence value
        """
        if self.ema_value is None:
            self.ema_value = confidence
        else:
            self.ema_value = self.ema_alpha * confidence + (1 - self.ema_alpha) * self.ema_value
        
        return self.ema_value
    
    def reset(self):
        """Reset smoothing state."""
        self.confidence_history.clear()
        self.ema_value = None


# =============================================================================
# BLINK DETECTION MODULE
# =============================================================================

class BlinkDetector:
    """
    Detects blink events from confidence scores and classifies them as dots or dashes.
    """
    
    def __init__(self, config: SystemConfig, calibration: CalibrationData):
        """
        Initialize the blink detector.
        
        Args:
            config: System configuration
            calibration: Calibration data
        """
        self.config = config
        self.calibration = calibration
        
        # State tracking
        self.is_blinking = False
        self.blink_start_frame = 0
        self.blink_start_time = 0.0
        self.current_frame = 0
        self.last_blink_end_frame = 0
        
        # FPS estimation
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.estimated_fps = 30.0
    
    def update_fps(self):
        """Update FPS estimation."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta > 0:
            self.fps_history.append(1.0 / delta)
            self.estimated_fps = np.mean(self.fps_history)
        self.last_frame_time = current_time
    
    def frames_to_ms(self, frames: int) -> float:
        """Convert frame count to milliseconds."""
        if self.estimated_fps > 0:
            return (frames / self.estimated_fps) * 1000
        return frames * 33.33  # Fallback to ~30 FPS
    
    def process(self, confidence: float) -> Optional[BlinkEvent]:
        """
        Process a confidence value and detect blink events.
        
        Args:
            confidence: Current confidence score (0=closed, 1=open)
            
        Returns:
            BlinkEvent if a blink just ended, None otherwise
        """
        self.update_fps()
        self.current_frame += 1
        
        # Detect blink start (confidence drops below threshold)
        if not self.is_blinking and confidence < self.config.blink_threshold:
            self.is_blinking = True
            self.blink_start_frame = self.current_frame
            self.blink_start_time = time.time()
            return None
        
        # Detect blink end (confidence rises above threshold)
        if self.is_blinking and confidence >= self.config.blink_threshold:
            self.is_blinking = False
            
            # Create blink event
            event = BlinkEvent()
            event.start_frame = self.blink_start_frame
            event.end_frame = self.current_frame
            event.duration_frames = event.end_frame - event.start_frame
            event.duration_ms = self.frames_to_ms(event.duration_frames)
            event.confidence = confidence
            
            # Classify blink type based on calibration
            threshold_ms = self.calibration.avg_blink_duration_ms
            if event.duration_ms < threshold_ms:
                event.blink_type = BlinkType.DOT
            else:
                event.blink_type = BlinkType.DASH
            
            self.last_blink_end_frame = self.current_frame
            return event
        
        return None
    
    def get_frames_since_last_blink(self) -> int:
        """Get number of frames since the last blink ended."""
        if self.last_blink_end_frame == 0:
            return 0
        return self.current_frame - self.last_blink_end_frame
    
    def is_letter_gap(self) -> bool:
        """Check if enough time has passed for a letter gap."""
        frames_elapsed = self.get_frames_since_last_blink()
        return frames_elapsed >= self.config.letter_gap_frames
    
    def is_word_gap(self) -> bool:
        """Check if enough time has passed for a word gap."""
        frames_elapsed = self.get_frames_since_last_blink()
        return frames_elapsed >= self.config.word_gap_frames
    
    def reset(self):
        """Reset detector state."""
        self.is_blinking = False
        self.blink_start_frame = 0
        self.current_frame = 0
        self.last_blink_end_frame = 0


# =============================================================================
# CALIBRATION MODULE
# =============================================================================

class CalibrationManager:
    """
    Manages calibration process for blink duration threshold.
    User must provide 3 intentional short blinks (dots) and 3 long blinks (dashes).
    The threshold is computed as the midpoint between average dot and dash durations.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the calibration manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.calibration = CalibrationData()
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_phase = CalibrationPhase.NOT_STARTED
        self.dot_blinks = []  # Store dot durations
        self.dash_blinks = []  # Store dash durations
        self.calibration_start_time = 0.0
        self.target_dots = 3  # Number of dots to collect
        self.target_dashes = 3  # Number of dashes to collect
    
    def start_calibration(self, method: CalibrationMethod = CalibrationMethod.FREE_BLINK,
                          target_blinks: int = 3):
        """
        Start the calibration process.
        First phase: collect dots (short blinks)
        Second phase: collect dashes (long blinks)
        
        Args:
            method: Calibration method to use
            target_blinks: Number of each type to collect (default 3)
        """
        self.is_calibrating = True
        self.calibration_phase = CalibrationPhase.COLLECTING_DOTS
        self.dot_blinks = []
        self.dash_blinks = []
        self.calibration_start_time = time.time()
        self.target_dots = target_blinks
        self.target_dashes = target_blinks
        self.calibration.calibration_method = method
        self.calibration.is_calibrated = False
    
    def add_blink(self, duration_ms: float) -> bool:
        """
        Add a blink duration to calibration data based on current phase.
        
        Args:
            duration_ms: Blink duration in milliseconds
            
        Returns:
            True if calibration is complete
        """
        if not self.is_calibrating:
            return False
        
        # Filter out very short or very long blinks (noise)
        if duration_ms < 30 or duration_ms > 3000:
            return False
        
        if self.calibration_phase == CalibrationPhase.COLLECTING_DOTS:
            # Collecting short blinks (dots)
            self.dot_blinks.append(duration_ms)
            
            # Check if we have enough dots
            if len(self.dot_blinks) >= self.target_dots:
                # Move to dash collection phase
                self.calibration_phase = CalibrationPhase.COLLECTING_DASHES
            return False
            
        elif self.calibration_phase == CalibrationPhase.COLLECTING_DASHES:
            # Collecting long blinks (dashes)
            self.dash_blinks.append(duration_ms)
            
            # Check if we have enough dashes
            if len(self.dash_blinks) >= self.target_dashes:
                self._finalize_calibration()
                return True
        
        return False
    
    def _finalize_calibration(self):
        """Finalize calibration and compute thresholds."""
        self.is_calibrating = False
        self.calibration_phase = CalibrationPhase.COMPLETED
        
        if len(self.dot_blinks) > 0 and len(self.dash_blinks) > 0:
            # Compute average durations for dots and dashes
            avg_dot = np.mean(self.dot_blinks)
            avg_dash = np.mean(self.dash_blinks)
            
            # Store averages
            self.calibration.avg_dot_duration_ms = avg_dot
            self.calibration.avg_dash_duration_ms = avg_dash
            
            # Threshold is the midpoint between average dot and dash
            self.calibration.avg_blink_duration_ms = (avg_dot + avg_dash) / 2.0
            
            # Store the collected durations
            self.calibration.dot_durations = self.dot_blinks.copy()
            self.calibration.dash_durations = self.dash_blinks.copy()
            self.calibration.is_calibrated = True
        else:
            # Fallback to default
            self.calibration.avg_blink_duration_ms = self.config.default_blink_duration_ms
    
    def reset(self):
        """Reset calibration."""
        self.is_calibrating = False
        self.calibration_phase = CalibrationPhase.NOT_STARTED
        self.dot_blinks = []
        self.dash_blinks = []
        self.calibration = CalibrationData()
    
    def get_calibration(self) -> CalibrationData:
        """Get current calibration data."""
        return self.calibration
    
    def get_progress(self) -> Tuple[str, int, int]:
        """Get calibration progress (phase_name, current, target)."""
        if self.calibration_phase == CalibrationPhase.COLLECTING_DOTS:
            return ("DOTS", len(self.dot_blinks), self.target_dots)
        elif self.calibration_phase == CalibrationPhase.COLLECTING_DASHES:
            return ("DASHES", len(self.dash_blinks), self.target_dashes)
        else:
            return ("DONE", 0, 0)
    
    def get_phase(self) -> CalibrationPhase:
        """Get current calibration phase."""
        return self.calibration_phase


# =============================================================================
# MORSE CODE DECODER MODULE
# =============================================================================

class MorseDecoder:
    """
    Decodes Morse code sequences into text.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Morse decoder.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.morse_dict = MORSE_CODE_DICT
        
        # State
        self.current_sequence = ""
        self.decoded_text = ""
        self.pending_letter = False
        self.pending_word = False
    
    def add_symbol(self, symbol: str):
        """
        Add a dot or dash to the current sequence.
        
        Args:
            symbol: '.' for dot, '-' for dash
        """
        if symbol in ['.', '-']:
            self.current_sequence += symbol
            self.pending_letter = True
            self.pending_word = True
    
    def process_letter_gap(self) -> Optional[str]:
        """
        Process a letter gap - decode current sequence.
        
        Returns:
            Decoded character or None
        """
        if not self.pending_letter or not self.current_sequence:
            return None
        
        self.pending_letter = False
        
        # Decode the sequence
        char = self.decode_sequence(self.current_sequence)
        
        if char:
            self.decoded_text += char
        else:
            # Unknown sequence - mark as unknown with ?
            self.decoded_text += "?"
        
        result = self.current_sequence
        self.current_sequence = ""
        
        return char
    
    def process_word_gap(self) -> bool:
        """
        Process a word gap - add space.
        
        Returns:
            True if space was added
        """
        # First process any pending letter
        self.process_letter_gap()
        
        if not self.pending_word:
            return False
        
        self.pending_word = False
        
        # Add space if not already ending with space
        if self.decoded_text and not self.decoded_text.endswith(' '):
            self.decoded_text += ' '
            return True
        
        return False
    
    def decode_sequence(self, sequence: str) -> Optional[str]:
        """
        Decode a Morse sequence into a character.
        
        Args:
            sequence: Morse code sequence (e.g., '.-' for 'A')
            
        Returns:
            Decoded character or None if invalid
        """
        return self.morse_dict.get(sequence)
    
    def get_current_sequence(self) -> str:
        """Get the current Morse sequence in progress."""
        return self.current_sequence
    
    def get_decoded_text(self) -> str:
        """Get the decoded text so far."""
        return self.decoded_text
    
    def clear_sequence(self):
        """Clear the current sequence."""
        self.current_sequence = ""
        self.pending_letter = False
    
    def clear_text(self):
        """Clear all decoded text."""
        self.decoded_text = ""
        self.clear_sequence()
        self.pending_word = False
    
    def backspace(self):
        """Remove the last character from decoded text."""
        if self.decoded_text:
            self.decoded_text = self.decoded_text[:-1]
    
    def remove_unresolved(self):
        """Remove all unresolved (?) characters from decoded text."""
        self.decoded_text = self.decoded_text.replace("?", "")
        # Clean up any double spaces that might result
        while "  " in self.decoded_text:
            self.decoded_text = self.decoded_text.replace("  ", " ")
        self.decoded_text = self.decoded_text.strip()
    
    def remove_last_symbol(self):
        """Remove the last symbol from current sequence."""
        if self.current_sequence:
            self.current_sequence = self.current_sequence[:-1]


# =============================================================================
# NLP CORRECTION MODULE (RESERVED EXTENSION)
# =============================================================================

class NLPCorrector(ABC):
    """
    Abstract base class for NLP-based text correction.
    This is a reserved extension point for future implementation.
    """
    
    @abstractmethod
    def correct(self, text: str) -> str:
        """
        Apply correction to the input text.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text
        """
        pass
    
    @abstractmethod
    def get_suggestions(self, text: str) -> List[str]:
        """
        Get correction suggestions for the input text.
        
        Args:
            text: Input text
            
        Returns:
            List of suggested corrections
        """
        pass


class RuleBasedCorrector(NLPCorrector):
    """
    Rule-based text corrector (placeholder implementation).
    """
    
    def __init__(self):
        """Initialize the rule-based corrector."""
        # Common word corrections
        self.corrections = {
            'teh': 'the',
            'adn': 'and',
            'taht': 'that',
            'wiht': 'with',
        }
    
    def correct(self, text: str) -> str:
        """Apply rule-based corrections."""
        words = text.split()
        corrected = []
        
        for word in words:
            lower = word.lower()
            if lower in self.corrections:
                corrected.append(self.corrections[lower])
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    def get_suggestions(self, text: str) -> List[str]:
        """Get suggestions (returns single correction for now)."""
        corrected = self.correct(text)
        if corrected != text:
            return [corrected]
        return []


class NLPCorrectionManager:
    """
    Manager for NLP-based text correction.
    Supports toggling and pluggable correctors.
    """
    
    def __init__(self):
        """Initialize the NLP correction manager."""
        self.enabled = False
        self.corrector: Optional[NLPCorrector] = None
        self.raw_text = ""
        self.corrected_text = ""
        
        # Initialize with rule-based corrector
        self.set_corrector(RuleBasedCorrector())
    
    def set_corrector(self, corrector: NLPCorrector):
        """
        Set the NLP corrector to use.
        
        Args:
            corrector: NLPCorrector implementation
        """
        self.corrector = corrector
    
    def enable(self):
        """Enable NLP correction."""
        self.enabled = True
    
    def disable(self):
        """Disable NLP correction."""
        self.enabled = False
    
    def toggle(self) -> bool:
        """Toggle NLP correction on/off."""
        self.enabled = not self.enabled
        return self.enabled
    
    def process(self, text: str) -> str:
        """
        Process text with optional NLP correction.
        
        Args:
            text: Input text
            
        Returns:
            Corrected text if enabled, original otherwise
        """
        self.raw_text = text
        
        if self.enabled and self.corrector:
            self.corrected_text = self.corrector.correct(text)
            return self.corrected_text
        
        return text
    
    def get_suggestions(self, text: str) -> List[str]:
        """Get correction suggestions."""
        if self.corrector:
            return self.corrector.get_suggestions(text)
        return []


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class EyeBlinkMorseSystem:
    """
    Main system class that orchestrates all components for eye-blink
    to Morse code conversion.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the eye-blink Morse code system.
        
        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self.eye_analyzer = EyeAnalyzer()
        self.yolo_classifier = YOLOEyeClassifier(
            self.config.yolo_model_path,
            self.config.use_gpu
        )
        self.confidence_fusion = ConfidenceFusion(
            self.config.smoothing_window,
            self.config.ema_alpha
        )
        self.calibration_manager = CalibrationManager(self.config)
        self.blink_detector = BlinkDetector(
            self.config, 
            self.calibration_manager.get_calibration()
        )
        self.morse_decoder = MorseDecoder(self.config)
        self.nlp_manager = NLPCorrectionManager()
        
        # State
        self.is_running = False
        self.current_confidence = 0.5
        self.current_eye_state = EyeState.UNKNOWN
        self.last_blink_event: Optional[BlinkEvent] = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.processing_time_ms = 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        start_time = time.time()
        
        results = {
            'eye_state': EyeState.UNKNOWN,
            'confidence': 0.5,
            'ear': 0.0,
            'yolo_result': YOLOResult(),
            'blink_event': None,
            'morse_sequence': '',
            'decoded_text': '',
            'is_calibrating': self.calibration_manager.is_calibrating,
            'calibration_progress': self.calibration_manager.get_progress(),  # (phase, current, target)
            'calibration_phase': self.calibration_manager.get_phase(),
            'fps': self.fps,
        }
        
        # 1. Eye analysis with MediaPipe
        eye_data, annotated_frame = self.eye_analyzer.process_frame(frame, self.config)
        results['ear'] = eye_data.avg_ear
        
        if not eye_data.landmarks_detected:
            # Fallback: no face detected
            return annotated_frame, results
        
        # 2. YOLO classification
        yolo_result = self.yolo_classifier.classify_dual_eye(
            eye_data.left_crop,
            eye_data.right_crop
        )
        results['yolo_result'] = yolo_result
        
        # 3. Confidence fusion
        raw_confidence = self.confidence_fusion.fuse(
            yolo_result,
            eye_data.normalized_ear,
            self.config.alpha
        )
        
        # Apply smoothing
        smoothed_confidence = self.confidence_fusion.smooth_ema(raw_confidence)
        self.current_confidence = smoothed_confidence
        results['confidence'] = smoothed_confidence
        
        # Determine eye state
        if smoothed_confidence >= self.config.blink_threshold:
            self.current_eye_state = EyeState.OPEN
        else:
            self.current_eye_state = EyeState.CLOSED
        results['eye_state'] = self.current_eye_state
        
        # 4. Blink detection
        if not self.calibration_manager.is_calibrating:
            # Update calibration reference
            self.blink_detector.calibration = self.calibration_manager.get_calibration()
            
            blink_event = self.blink_detector.process(smoothed_confidence)
            
            if blink_event:
                self.last_blink_event = blink_event
                results['blink_event'] = blink_event
                
                # Add to Morse sequence
                self.morse_decoder.add_symbol(blink_event.blink_type.value)
            
            # Check for letter/word gaps
            if self.blink_detector.is_word_gap():
                self.morse_decoder.process_word_gap()
            elif self.blink_detector.is_letter_gap():
                self.morse_decoder.process_letter_gap()
        else:
            # Calibration mode
            blink_event = self.blink_detector.process(smoothed_confidence)
            if blink_event:
                self.calibration_manager.add_blink(blink_event.duration_ms)
            results['calibration_progress'] = self.calibration_manager.get_progress()
        
        # 5. Get Morse state
        results['morse_sequence'] = self.morse_decoder.get_current_sequence()
        
        # 6. Apply NLP correction if enabled
        raw_text = self.morse_decoder.get_decoded_text()
        results['decoded_text'] = self.nlp_manager.process(raw_text)
        
        # Update performance metrics
        self.frame_count += 1
        self.processing_time_ms = (time.time() - start_time) * 1000
        
        # Add overlays to frame
        annotated_frame = self._add_overlays(annotated_frame, results)
        
        return annotated_frame, results
    
    def _add_overlays(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Add status overlays to the frame."""
        h, w = frame.shape[:2]
        
        # Eye state indicator
        state_color = (0, 255, 0) if results['eye_state'] == EyeState.OPEN else (0, 0, 255)
        cv2.putText(frame, f"Eye: {results['eye_state'].value}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # Confidence bar
        conf = results['confidence']
        bar_width = int(200 * conf)
        cv2.rectangle(frame, (10, 50), (210, 70), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), state_color, -1)
        cv2.putText(frame, f"Conf: {conf:.2f}", 
                    (220, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # EAR value
        cv2.putText(frame, f"EAR: {results['ear']:.3f}", 
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current Morse sequence
        if results['morse_sequence']:
            cv2.putText(frame, f"Morse: {results['morse_sequence']}", 
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Decoded text
        if results['decoded_text']:
            cv2.putText(frame, f"Text: {results['decoded_text'][-30:]}", 
                        (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calibration indicator
        if results['is_calibrating']:
            progress = results['calibration_progress']
            cv2.putText(frame, f"CALIBRATING: {progress[0]}/{progress[1]}", 
                        (w // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        return frame
    
    def start_calibration(self, method: CalibrationMethod = CalibrationMethod.FREE_BLINK,
                          target_blinks: int = 5):
        """Start calibration process."""
        self.calibration_manager.start_calibration(method, target_blinks)
        self.blink_detector.reset()
    
    def reset_calibration(self):
        """Reset calibration."""
        self.calibration_manager.reset()
    
    def clear_text(self):
        """Clear decoded text."""
        self.morse_decoder.clear_text()
    
    def toggle_nlp(self) -> bool:
        """Toggle NLP correction."""
        return self.nlp_manager.toggle()
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def close(self):
        """Release resources."""
        self.eye_analyzer.close()


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def start_detection():
    """Callback for start button."""
    st.session_state.is_running = True

def stop_detection():
    """Callback for stop button."""
    st.session_state.is_running = False

def reset_all():
    """Callback for reset button."""
    st.session_state.reset_all_flag = True

def start_calibration_cb():
    """Callback for start calibration button - also starts detection."""
    st.session_state.start_calibration_flag = True
    st.session_state.is_running = True

def reset_calibration_cb():
    """Callback for reset calibration button."""
    st.session_state.reset_calibration_flag = True

def clear_text_cb():
    """Callback for clear text button."""
    st.session_state.clear_text_flag = True

def remove_unresolved_cb():
    """Callback for remove unresolved button."""
    st.session_state.remove_unresolved_flag = True

def create_streamlit_app():
    """
    Create and run the Streamlit application.
    """
    st.set_page_config(
        page_title="Eye-Blink Morse Code System",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ Real-Time Eye-Blink to Morse Code System")
    st.markdown("*MediaPipe FaceMesh + YOLOv26-cls + Streamlit*")
    
    # Initialize session state
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'decoded_text' not in st.session_state:
        st.session_state.decoded_text = ""
    if 'morse_sequence' not in st.session_state:
        st.session_state.morse_sequence = ""
    if 'start_calibration_flag' not in st.session_state:
        st.session_state.start_calibration_flag = False
    if 'reset_calibration_flag' not in st.session_state:
        st.session_state.reset_calibration_flag = False
    if 'clear_text_flag' not in st.session_state:
        st.session_state.clear_text_flag = False
    if 'reset_all_flag' not in st.session_state:
        st.session_state.reset_all_flag = False
    if 'remove_unresolved_flag' not in st.session_state:
        st.session_state.remove_unresolved_flag = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        use_gpu = st.checkbox("Use GPU", value=True)
        
        # Confidence settings
        st.subheader("Confidence Settings")
        alpha = st.slider(
            "Alpha (YOLO weight)", 
            min_value=0.0, max_value=1.0, value=0.4, step=0.05,
            help="Weight for YOLO confidence. (1-alpha) is used for EAR."
        )
        blink_threshold = st.slider(
            "Blink Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Confidence below this triggers blink detection."
        )
        
        # Timing settings
        st.subheader("Timing Settings")
        letter_gap = st.slider(
            "Letter Gap (frames)",
            min_value=15, max_value=90, value=45, step=5,
            help="Frames of pause to trigger letter gap."
        )
        word_gap = st.slider(
            "Word Gap (frames)",
            min_value=30, max_value=150, value=90, step=10,
            help="Frames of pause to trigger word gap."
        )
        
        # EAR settings
        st.subheader("EAR Normalization")
        ear_min = st.slider("EAR Min", 0.05, 0.25, 0.15, 0.01)
        ear_max = st.slider("EAR Max", 0.25, 0.50, 0.35, 0.01)
        
        # NLP settings
        st.subheader("NLP Correction")
        nlp_enabled = st.checkbox("Enable NLP Correction", value=False)
        
        st.divider()
        
        # Calibration
        st.subheader("🎯 Calibration")
        st.caption("Calibrate by blinking 3 short (dots) then 3 long (dashes)")
        cal_blinks = st.number_input("Blinks per type", 2, 5, 3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Start Cal.", use_container_width=True, key="start_cal_btn", on_click=start_calibration_cb)
        with col2:
            st.button("Reset Cal.", use_container_width=True, key="reset_cal_btn", on_click=reset_calibration_cb)
        
        st.divider()
        
        # Text controls
        st.subheader("📝 Text Controls")
        st.button("Clear Text", use_container_width=True, key="clear_text_btn", on_click=clear_text_cb)
        st.button("Remove ? (unresolved)", use_container_width=True, key="remove_unresolved_btn", on_click=remove_unresolved_cb)
    
    # Main content area
    col_video, col_info = st.columns([2, 1])
    
    with col_video:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
        
        # Control buttons with callbacks
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            st.button("▶️ Start Detection", use_container_width=True, key="start_btn", on_click=start_detection)
        with btn_col2:
            st.button("⏹️ Stop Detection", use_container_width=True, key="stop_btn", on_click=stop_detection)
        with btn_col3:
            st.button("🔄 Reset All", use_container_width=True, key="reset_btn", on_click=reset_all)
    
    with col_info:
        st.subheader("📊 Status")
        
        # Status displays
        status_container = st.container()
        with status_container:
            eye_state_display = st.empty()
            confidence_display = st.empty()
            fps_display = st.empty()
            
        st.subheader("📡 Current Morse")
        morse_display = st.empty()
        
        st.subheader("📝 Decoded Text")
        text_display = st.empty()
        
        st.subheader("📈 Calibration Status")
        cal_status = st.empty()
    
    # Morse code reference
    with st.expander("📖 Morse Code Reference"):
        col1, col2, col3, col4 = st.columns(4)
        morse_items = list(MORSE_CODE_DICT.items())
        chunk_size = len(morse_items) // 4 + 1
        
        for i, col in enumerate([col1, col2, col3, col4]):
            with col:
                start = i * chunk_size
                end = start + chunk_size
                for code, char in morse_items[start:end]:
                    st.text(f"{char}: {code}")
    
    # Initialize system
    if st.session_state.system is None:
        config = SystemConfig(
            alpha=alpha,
            blink_threshold=blink_threshold,
            letter_gap_frames=letter_gap,
            word_gap_frames=word_gap,
            ear_min=ear_min,
            ear_max=ear_max,
            use_gpu=use_gpu
        )
        st.session_state.system = EyeBlinkMorseSystem(config)
    
    system = st.session_state.system
    
    # Update config
    system.update_config(
        alpha=alpha,
        blink_threshold=blink_threshold,
        letter_gap_frames=letter_gap,
        word_gap_frames=word_gap,
        ear_min=ear_min,
        ear_max=ear_max
    )
    
    # Handle NLP toggle
    if nlp_enabled != system.nlp_manager.enabled:
        if nlp_enabled:
            system.nlp_manager.enable()
        else:
            system.nlp_manager.disable()
    
    # Handle calibration flags
    if st.session_state.start_calibration_flag:
        system.start_calibration(CalibrationMethod.FREE_BLINK, cal_blinks)
        st.session_state.start_calibration_flag = False
    
    if st.session_state.reset_calibration_flag:
        system.reset_calibration()
        st.session_state.reset_calibration_flag = False
    
    if st.session_state.clear_text_flag:
        system.clear_text()
        st.session_state.clear_text_flag = False
    
    if st.session_state.remove_unresolved_flag:
        system.morse_decoder.remove_unresolved()
        st.session_state.remove_unresolved_flag = False
    
    if st.session_state.reset_all_flag:
        system.clear_text()
        system.reset_calibration()
        system.confidence_fusion.reset()
        system.blink_detector.reset()
        st.session_state.reset_all_flag = False
    
    # Video processing with while loop (no flickering)
    if st.session_state.is_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
        
        if cap.isOpened():
            try:
                while st.session_state.is_running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame from webcam")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    annotated_frame, results = system.process_frame(frame)
                    
                    # Convert BGR to RGB for Streamlit
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update displays (using placeholders - no flicker)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Update status
                    state_emoji = "👁️" if results['eye_state'] == EyeState.OPEN else "😑"
                    eye_state_display.metric("Eye State", f"{state_emoji} {results['eye_state'].value.upper()}")
                    confidence_display.metric("Confidence", f"{results['confidence']:.2%}")
                    fps_display.metric("FPS", f"{system.blink_detector.estimated_fps:.1f}")
                    
                    # Update Morse display
                    morse_seq = results['morse_sequence']
                    if morse_seq:
                        morse_display.code(morse_seq, language=None)
                    else:
                        morse_display.info("Waiting for blinks...")
                    
                    # Update text display
                    decoded = results['decoded_text']
                    if decoded:
                        text_display.success(decoded)
                    else:
                        text_display.info("No text decoded yet")
                    
                    # Update calibration status
                    if results['is_calibrating']:
                        progress = results['calibration_progress']  # (phase_name, current, target)
                        phase_name, current, target = progress
                        if phase_name == "DOTS":
                            cal_status.warning(f"🎯 Blink SHORT {current}/{target} times (dots)")
                        elif phase_name == "DASHES":
                            cal_status.warning(f"🎯 Blink LONG {current}/{target} times (dashes)")
                        else:
                            cal_status.info("Calibrating...")
                    elif system.calibration_manager.get_calibration().is_calibrated:
                        cal_data = system.calibration_manager.get_calibration()
                        cal_status.success(f"✅ Calibrated!\\nDot avg: {cal_data.avg_dot_duration_ms:.0f}ms\\nDash avg: {cal_data.avg_dash_duration_ms:.0f}ms\\nThreshold: {cal_data.avg_blink_duration_ms:.0f}ms")
                    else:
                        cal_status.info("Not calibrated - using defaults")
                    
                    # Small delay to control frame rate
                    time.sleep(0.001)
                    
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                cap.release()
        else:
            st.error("Camera not available")
            st.session_state.is_running = False
    else:
        # Display placeholder when not running
        video_placeholder.info("👆 Click 'Start Detection' to begin")
        eye_state_display.metric("Eye State", "---")
        confidence_display.metric("Confidence", "---")
        fps_display.metric("FPS", "---")
        morse_display.info("Waiting for input...")
        text_display.info("No text decoded yet")
        
        if system.calibration_manager.get_calibration().is_calibrated:
            cal_data = system.calibration_manager.get_calibration()
            cal_status.success(f"✅ Calibrated!\\nDot avg: {cal_data.avg_dot_duration_ms:.0f}ms\\nDash avg: {cal_data.avg_dash_duration_ms:.0f}ms\\nThreshold: {cal_data.avg_blink_duration_ms:.0f}ms")
        else:
            cal_status.info("Not calibrated - using defaults")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Check if running in Streamlit context
    try:
        # This will work when running with `streamlit run`
        create_streamlit_app()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo run the application, use:")
        print("  streamlit run implementation.py")
