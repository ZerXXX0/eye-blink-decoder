"""
Pipeline Testing Suite for Eye-Blink Morse Code System
=======================================================

Streamlit-based interactive testing for each component:
1. NLP Text Correction (IndoBERT Seq2Seq)
2. YOLO Eye State Classification
3. EAR (Eye Aspect Ratio) Analysis
4. Full Pipeline Integration Test

Run with: streamlit run test_pipelines.py

Date: January 2026
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
import pandas as pd
from typing import Optional, Dict, Any, List
from collections import deque

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from implementation
from implementation import (
    EyeAnalyzer,
    YOLOEyeClassifier,
    ConfidenceFusion,
    IndoBERTCorrector,
    RuleBasedCorrector,
    SystemConfig,
    EyeState,
    YOLOResult,
    EyeData,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    preprocess_for_yolo,
)


# =============================================================================
# CACHED RESOURCE LOADERS
# =============================================================================

@st.cache_resource
def load_nlp_model():
    """Load IndoBERT model and tokenizer directly with caching."""
    import torch
    from transformers import AutoTokenizer, EncoderDecoderModel, GenerationConfig
    
    # Model is in a subfolder on HuggingFace
    model_repo = "ZerXXX/indobert-corrector"
    subfolder = "indoBERT-best-corrector"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_repo, 
        subfolder=subfolder,
        force_download=False
    )
    model = EncoderDecoderModel.from_pretrained(
        model_repo, 
        subfolder=subfolder,
        force_download=False
    )
    
    # Explicitly set token IDs from config.json (required for generation)
    # These values are from the model's config.json on HuggingFace
    model.config.decoder_start_token_id = 2  # [CLS] token
    model.config.eos_token_id = 3            # [SEP] token
    model.config.pad_token_id = 0            # [PAD] token
    model.config.bos_token_id = 2            # Same as decoder_start
    
    # Also set on generation_config
    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = 2
        model.generation_config.eos_token_id = 3
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 2
    
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device


class TestIndoBERTCorrector:
    """IndoBERT corrector for testing - uses cached model loader."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = 64
        self.num_beams = 4
    
    def correct(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            import torch
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=False,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                    bos_token_id=2
                )
            
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected if corrected else text
        except Exception as e:
            st.error(f"Correction error: {e}")
            return text
    
    def get_suggestions(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        try:
            import torch
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=min(self.num_beams, 3),
                    do_sample=False,
                    early_stopping=True
                )
            
            suggestions = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                if decoded and decoded != text and decoded not in suggestions:
                    suggestions.append(decoded)
            return suggestions
        except Exception as e:
            return []


def get_nlp_corrector():
    """Get NLP corrector instance using cached model."""
    model, tokenizer, device = load_nlp_model()
    return TestIndoBERTCorrector(model, tokenizer, device)


@st.cache_resource
def load_eye_analyzer():
    """Load MediaPipe eye analyzer with caching."""
    return EyeAnalyzer()


@st.cache_resource
def load_yolo_classifier(model_path: str, use_gpu: bool):
    """Load YOLO classifier with caching."""
    return YOLOEyeClassifier(model_path, use_gpu)


# =============================================================================
# NLP CORRECTION TEST PAGE
# =============================================================================

def render_nlp_test():
    """Render NLP text correction test page."""
    st.header("NLP Text Correction Test")
    st.markdown("**Model:** IndoBERT Seq2Seq (`ZerXXX/indobert-corrector`)")
    
    # Initialize session state
    if 'nlp_history' not in st.session_state:
        st.session_state.nlp_history = []
    
    # Load model
    with st.spinner("Loading IndoBERT model..."):
        try:
            corrector = get_nlp_corrector()
            st.success(f"Model loaded | Device: `{corrector.device}` | Max length: {corrector.max_length} | Beams: {corrector.num_beams}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Predefined test cases
        st.markdown("**Quick Test Cases:**")
        test_cases = ["slmt pagi", "trm ksh", "ap kbr", "sya mau mkn", "hlo dunia"]
        
        test_cols = st.columns(len(test_cases))
        for i, (tc_col, tc) in enumerate(zip(test_cols, test_cases)):
            with tc_col:
                if st.button(tc, key=f"tc_{i}", use_container_width=True):
                    st.session_state.nlp_input = tc
        
        # Text input
        input_text = st.text_area(
            "Enter text to correct:",
            value=st.session_state.get('nlp_input', ''),
            height=100,
            key="nlp_text_input"
        )
        
        # Process button
        if st.button("Correct Text", type="primary", use_container_width=True):
            if input_text.strip():
                # Run correction
                start_time = time.time()
                corrected = corrector.correct(input_text)
                inference_time = (time.time() - start_time) * 1000
                
                # Get suggestions
                suggestions = corrector.get_suggestions(input_text)
                
                # Store result
                result = {
                    'input': input_text,
                    'output': corrected,
                    'changed': input_text != corrected,
                    'suggestions': suggestions,
                    'time_ms': inference_time,
                    'timestamp': time.strftime("%H:%M:%S")
                }
                st.session_state.nlp_history.insert(0, result)
                st.session_state.nlp_last_result = result
    
    with col2:
        st.subheader("Results")
        
        if 'nlp_last_result' in st.session_state:
            result = st.session_state.nlp_last_result
            
            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Inference Time", f"{result['time_ms']:.1f} ms")
            with m2:
                st.metric("Changed", "Yes" if result['changed'] else "No")
            with m3:
                st.metric("Suggestions", len(result['suggestions']))
            
            # Output
            st.markdown("**Corrected Output:**")
            st.code(result['output'], language=None)
            
            # Comparison
            if result['changed']:
                st.markdown("**Comparison:**")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown(f"Input: `{result['input']}`")
                with comp_col2:
                    st.markdown(f"Output: `{result['output']}`")
            
            # Suggestions
            if result['suggestions']:
                st.markdown("**Suggestions:**")
                for i, sug in enumerate(result['suggestions'], 1):
                    st.markdown(f"{i}. `{sug}`")
    
    # History section
    st.divider()
    st.subheader("Correction History")
    
    if st.session_state.nlp_history:
        # Summary stats
        total_time = sum(r['time_ms'] for r in st.session_state.nlp_history)
        avg_time = total_time / len(st.session_state.nlp_history)
        changed_count = sum(1 for r in st.session_state.nlp_history if r['changed'])
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Corrections", len(st.session_state.nlp_history))
        with stat_cols[1]:
            st.metric("Changed", f"{changed_count} ({100*changed_count/len(st.session_state.nlp_history):.0f}%)")
        with stat_cols[2]:
            st.metric("Total Time", f"{total_time:.1f} ms")
        with stat_cols[3]:
            st.metric("Avg Time", f"{avg_time:.1f} ms")
        
        # History table
        df = pd.DataFrame([
            {
                'Time': r['timestamp'],
                'Input': r['input'],
                'Output': r['output'],
                'Changed': 'Yes' if r['changed'] else 'No',
                'Latency (ms)': f"{r['time_ms']:.1f}"
            }
            for r in st.session_state.nlp_history[:20]
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("Clear History"):
            st.session_state.nlp_history = []
            if 'nlp_last_result' in st.session_state:
                del st.session_state.nlp_last_result
            st.rerun()
    else:
        st.info("No corrections yet. Enter text above to start testing.")


# =============================================================================
# YOLO CLASSIFIER TEST PAGE
# =============================================================================

def render_yolo_test():
    """Render YOLO eye classification test page."""
    st.header("YOLO Eye State Classification Test")
    
    # Initialize session state
    if 'yolo_running' not in st.session_state:
        st.session_state.yolo_running = False
    if 'yolo_stats' not in st.session_state:
        st.session_state.yolo_stats = {
            'frames': 0,
            'open': 0,
            'closed': 0,
            'total_time': 0,
            'confidences': []
        }
    
    # Config
    config = SystemConfig()
    
    # Sidebar config
    with st.sidebar:
        st.subheader("YOLO Settings")
        use_gpu = st.checkbox("Use GPU", value=True)
    
    # Load models
    col_status = st.columns(2)
    
    with col_status[0]:
        with st.spinner("Loading YOLO model..."):
            try:
                classifier = load_yolo_classifier(config.yolo_model_path, use_gpu)
                st.success(f"YOLO loaded | GPU: {classifier.use_gpu}")
            except Exception as e:
                st.error(f"YOLO load failed: {e}")
                return
    
    with col_status[1]:
        with st.spinner("Loading FaceLandmarker..."):
            try:
                eye_analyzer = load_eye_analyzer()
                st.success("FaceLandmarker loaded")
            except Exception as e:
                st.error(f"FaceLandmarker load failed: {e}")
                return
    
    # Controls
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        start_btn = st.button("Start", use_container_width=True, type="primary")
    with ctrl_cols[1]:
        stop_btn = st.button("Stop", use_container_width=True)
    with ctrl_cols[2]:
        reset_btn = st.button("Reset Stats", use_container_width=True)
    
    if start_btn:
        st.session_state.yolo_running = True
    if stop_btn:
        st.session_state.yolo_running = False
    if reset_btn:
        st.session_state.yolo_stats = {
            'frames': 0, 'open': 0, 'closed': 0, 'total_time': 0, 'confidences': []
        }
    
    # Main display area
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
    
    with col_metrics:
        st.subheader("Real-time Metrics")
        state_display = st.empty()
        confidence_display = st.empty()
        probs_display = st.empty()
        timing_display = st.empty()
        
        st.subheader("Session Stats")
        stats_display = st.empty()
    
    # Eye crops display
    st.subheader("Eye Crops")
    crop_cols = st.columns(2)
    left_crop_display = crop_cols[0].empty()
    right_crop_display = crop_cols[1].empty()
    
    # Video loop
    if st.session_state.yolo_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.session_state.yolo_running = False
        else:
            try:
                while st.session_state.yolo_running:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    eye_data, annotated = eye_analyzer.process_frame(frame, config)
                    
                    # YOLO inference
                    start_time = time.time()
                    yolo_result = classifier.classify_dual_eye(
                        eye_data.left_crop, eye_data.right_crop
                    )
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Update stats
                    stats = st.session_state.yolo_stats
                    stats['frames'] += 1
                    stats['total_time'] += inference_time
                    if yolo_result.state == EyeState.OPEN:
                        stats['open'] += 1
                    elif yolo_result.state == EyeState.CLOSED:
                        stats['closed'] += 1
                    stats['confidences'].append(yolo_result.confidence)
                    if len(stats['confidences']) > 100:
                        stats['confidences'] = stats['confidences'][-100:]
                    
                    # Draw on frame
                    state_color = (0, 255, 0) if yolo_result.state == EyeState.OPEN else (0, 0, 255)
                    cv2.putText(annotated, f"{yolo_result.state.value.upper()}", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
                    cv2.putText(annotated, f"Conf: {yolo_result.confidence:.1%}", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Display frame
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Update metrics
                    state_emoji = "Open" if yolo_result.state == EyeState.OPEN else "Closed"
                    state_display.metric("Eye State", f"{state_emoji}")
                    confidence_display.metric("Confidence", f"{yolo_result.confidence:.1%}")
                    probs_display.markdown(f"""
                    **Probabilities:**
                    - Open: `{yolo_result.open_prob:.2%}`
                    - Closed: `{yolo_result.closed_prob:.2%}`
                    """)
                    timing_display.metric("Inference", f"{inference_time:.1f} ms")
                    
                    # Session stats
                    avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
                    stats_display.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Frames | {stats['frames']} |
                    | Open | {stats['open']} ({100*stats['open']/max(stats['frames'],1):.1f}%) |
                    | Closed | {stats['closed']} ({100*stats['closed']/max(stats['frames'],1):.1f}%) |
                    | Avg Conf | {avg_conf:.1%} |
                    | Avg Time | {stats['total_time']/max(stats['frames'],1):.1f} ms |
                    """)
                    
                    # Eye crops
                    if eye_data.left_crop is not None and eye_data.left_crop.size > 0:
                        left_crop_display.image(
                            cv2.cvtColor(eye_data.left_crop, cv2.COLOR_BGR2RGB),
                            caption="Left Eye", use_container_width=True
                        )
                    if eye_data.right_crop is not None and eye_data.right_crop.size > 0:
                        right_crop_display.image(
                            cv2.cvtColor(eye_data.right_crop, cv2.COLOR_BGR2RGB),
                            caption="Right Eye", use_container_width=True
                        )
                    
            finally:
                cap.release()
    else:
        video_placeholder.info("Click 'Start' to begin YOLO classification test")
        
        # Show final stats if available
        stats = st.session_state.yolo_stats
        if stats['frames'] > 0:
            st.divider()
            st.subheader("Final Session Metrics")
            
            m_cols = st.columns(5)
            m_cols[0].metric("Total Frames", stats['frames'])
            m_cols[1].metric("Open", f"{stats['open']} ({100*stats['open']/stats['frames']:.1f}%)")
            m_cols[2].metric("Closed", f"{stats['closed']} ({100*stats['closed']/stats['frames']:.1f}%)")
            m_cols[3].metric("Avg Confidence", f"{np.mean(stats['confidences']):.1%}" if stats['confidences'] else "N/A")
            m_cols[4].metric("Avg Inference", f"{stats['total_time']/stats['frames']:.1f} ms")


# =============================================================================
# EAR ANALYSIS TEST PAGE
# =============================================================================

def render_ear_test():
    """Render EAR analysis test page."""
    st.header("Eye Aspect Ratio (EAR) Analysis Test")
    
    # Initialize session state
    if 'ear_running' not in st.session_state:
        st.session_state.ear_running = False
    if 'ear_stats' not in st.session_state:
        st.session_state.ear_stats = {
            'frames': 0,
            'detected': 0,
            'ear_values': [],
            'left_ear_values': [],
            'right_ear_values': []
        }
    if 'ear_calibration' not in st.session_state:
        st.session_state.ear_calibration = {'open': None, 'closed': None}
    
    config = SystemConfig()
    
    # Sidebar config
    with st.sidebar:
        st.subheader("EAR Settings")
        ear_min = st.slider("EAR Min (closed)", 0.05, 0.25, config.ear_min, 0.01)
        ear_max = st.slider("EAR Max (open)", 0.25, 0.50, config.ear_max, 0.01)
        config.ear_min = ear_min
        config.ear_max = ear_max
    
    # Load analyzer
    with st.spinner("Loading FaceLandmarker..."):
        try:
            eye_analyzer = load_eye_analyzer()
            st.success("FaceLandmarker loaded")
        except Exception as e:
            st.error(f"Load failed: {e}")
            return
    
    # Controls
    ctrl_cols = st.columns([1, 1, 1, 1])
    with ctrl_cols[0]:
        if st.button("Start", use_container_width=True, type="primary"):
            st.session_state.ear_running = True
    with ctrl_cols[1]:
        if st.button("Stop", use_container_width=True):
            st.session_state.ear_running = False
    with ctrl_cols[2]:
        if st.button("Calibrate Open", use_container_width=True):
            if st.session_state.ear_stats['ear_values']:
                st.session_state.ear_calibration['open'] = np.mean(st.session_state.ear_stats['ear_values'][-30:])
    with ctrl_cols[3]:
        if st.button("Calibrate Closed", use_container_width=True):
            if st.session_state.ear_stats['ear_values']:
                st.session_state.ear_calibration['closed'] = np.mean(st.session_state.ear_stats['ear_values'][-30:])
    
    # Calibration display
    cal = st.session_state.ear_calibration
    if cal['open'] or cal['closed']:
        cal_cols = st.columns(2)
        with cal_cols[0]:
            if cal['open']:
                st.success(f"Open EAR: {cal['open']:.4f}")
        with cal_cols[1]:
            if cal['closed']:
                st.success(f"Closed EAR: {cal['closed']:.4f}")
    
    # Main display
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        
        st.subheader("EAR History Graph")
        chart_placeholder = st.empty()
    
    with col_metrics:
        st.subheader("Real-time EAR")
        ear_display = st.empty()
        left_ear_display = st.empty()
        right_ear_display = st.empty()
        norm_display = st.empty()
        
        st.subheader("Statistics")
        stats_display = st.empty()
    
    # Video loop
    if st.session_state.ear_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.session_state.ear_running = False
        else:
            try:
                while st.session_state.ear_running:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    eye_data, annotated = eye_analyzer.process_frame(frame, config)
                    
                    # Update stats
                    stats = st.session_state.ear_stats
                    stats['frames'] += 1
                    
                    if eye_data.landmarks_detected:
                        stats['detected'] += 1
                        stats['ear_values'].append(eye_data.avg_ear)
                        stats['left_ear_values'].append(eye_data.left_ear)
                        stats['right_ear_values'].append(eye_data.right_ear)
                        
                        # Keep last 500 values
                        for key in ['ear_values', 'left_ear_values', 'right_ear_values']:
                            if len(stats[key]) > 500:
                                stats[key] = stats[key][-500:]
                    
                    # Draw on frame
                    if eye_data.landmarks_detected:
                        color = (0, 255, 0) if eye_data.normalized_ear > 0.5 else (0, 0, 255)
                        cv2.putText(annotated, f"EAR: {eye_data.avg_ear:.4f}", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(annotated, f"Norm: {eye_data.normalized_ear:.2f}", 
                                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        cv2.putText(annotated, "No face detected", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Display frame
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Update metrics
                    if eye_data.landmarks_detected:
                        ear_display.metric("Average EAR", f"{eye_data.avg_ear:.4f}")
                        left_ear_display.metric("Left EAR", f"{eye_data.left_ear:.4f}")
                        right_ear_display.metric("Right EAR", f"{eye_data.right_ear:.4f}")
                        norm_display.metric("Normalized", f"{eye_data.normalized_ear:.2f}")
                    else:
                        ear_display.metric("Average EAR", "---")
                    
                    # Statistics
                    if stats['ear_values']:
                        stats_display.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | Frames | {stats['frames']} |
                        | Detection Rate | {100*stats['detected']/stats['frames']:.1f}% |
                        | Mean EAR | {np.mean(stats['ear_values']):.4f} |
                        | Std EAR | {np.std(stats['ear_values']):.4f} |
                        | Min EAR | {min(stats['ear_values']):.4f} |
                        | Max EAR | {max(stats['ear_values']):.4f} |
                        """)
                    
                    # Chart
                    if len(stats['ear_values']) > 10:
                        chart_data = pd.DataFrame({
                            'Average': stats['ear_values'][-100:],
                            'Left': stats['left_ear_values'][-100:],
                            'Right': stats['right_ear_values'][-100:]
                        })
                        chart_placeholder.line_chart(chart_data, height=200)
                    
            finally:
                cap.release()
    else:
        video_placeholder.info("Click 'Start' to begin EAR analysis test")
        
        # Show final stats
        stats = st.session_state.ear_stats
        if stats['ear_values']:
            st.divider()
            st.subheader("Final Session Metrics")
            
            m_cols = st.columns(6)
            m_cols[0].metric("Frames", stats['frames'])
            m_cols[1].metric("Detection", f"{100*stats['detected']/max(stats['frames'],1):.1f}%")
            m_cols[2].metric("Mean EAR", f"{np.mean(stats['ear_values']):.4f}")
            m_cols[3].metric("Std EAR", f"{np.std(stats['ear_values']):.4f}")
            m_cols[4].metric("Min EAR", f"{min(stats['ear_values']):.4f}")
            m_cols[5].metric("Max EAR", f"{max(stats['ear_values']):.4f}")
            
            # Final chart
            st.subheader("EAR History")
            chart_data = pd.DataFrame({
                'Average': stats['ear_values'],
                'Left': stats['left_ear_values'],
                'Right': stats['right_ear_values']
            })
            st.line_chart(chart_data, height=300)
            
            if st.button("Reset Statistics"):
                st.session_state.ear_stats = {
                    'frames': 0, 'detected': 0,
                    'ear_values': [], 'left_ear_values': [], 'right_ear_values': []
                }
                st.session_state.ear_calibration = {'open': None, 'closed': None}
                st.rerun()


# =============================================================================
# FULL PIPELINE TEST PAGE
# =============================================================================

def render_full_pipeline_test():
    """Render full pipeline integration test page."""
    st.header("Full Pipeline Integration Test")
    st.markdown("*Eye Analysis -> YOLO Classification -> Confidence Fusion -> NLP Correction*")
    
    # Initialize session state
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'pipeline_stats' not in st.session_state:
        st.session_state.pipeline_stats = {
            'frames': 0,
            'eye_times': [],
            'yolo_times': [],
            'fusion_times': [],
            'total_times': [],
            'confidences': []
        }
    if 'pipeline_nlp_history' not in st.session_state:
        st.session_state.pipeline_nlp_history = []
    
    config = SystemConfig()
    
    # Sidebar
    with st.sidebar:
        st.subheader("Pipeline Settings")
        alpha = st.slider("Alpha (YOLO weight)", 0.0, 1.0, config.alpha, 0.05)
        blink_threshold = st.slider("Blink Threshold", 0.1, 0.9, config.blink_threshold, 0.05)
        config.alpha = alpha
        config.blink_threshold = blink_threshold
    
    # Load all components
    st.subheader("Component Status")
    status_cols = st.columns(4)
    
    components_loaded = True
    
    with status_cols[0]:
        try:
            eye_analyzer = load_eye_analyzer()
            st.success("EyeAnalyzer")
        except Exception as e:
            st.error(f"EyeAnalyzer")
            components_loaded = False
    
    with status_cols[1]:
        try:
            yolo_classifier = load_yolo_classifier(config.yolo_model_path, config.use_gpu)
            st.success("YOLO")
        except Exception as e:
            st.error(f"YOLO")
            components_loaded = False
    
    with status_cols[2]:
        try:
            confidence_fusion = ConfidenceFusion(config.smoothing_window, config.ema_alpha)
            st.success("ConfidenceFusion")
        except Exception as e:
            st.error(f"Fusion")
            components_loaded = False
    
    with status_cols[3]:
        try:
            nlp_corrector = get_nlp_corrector()
            st.success("IndoBERT")
        except Exception as e:
            st.error(f"IndoBERT")
            components_loaded = False
    
    if not components_loaded:
        st.error("Some components failed to load. Cannot run full pipeline test.")
        return
    
    # Controls
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        if st.button("Start Pipeline", use_container_width=True, type="primary"):
            st.session_state.pipeline_running = True
            confidence_fusion.reset()
    with ctrl_cols[1]:
        if st.button("Stop", use_container_width=True):
            st.session_state.pipeline_running = False
    with ctrl_cols[2]:
        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline_stats = {
                'frames': 0, 'eye_times': [], 'yolo_times': [],
                'fusion_times': [], 'total_times': [], 'confidences': []
            }
            st.session_state.pipeline_nlp_history = []
    
    # NLP Test section
    st.divider()
    nlp_cols = st.columns([3, 1])
    with nlp_cols[0]:
        nlp_input = st.text_input("Test NLP Correction:", placeholder="Enter text to correct...")
    with nlp_cols[1]:
        st.write("")
        st.write("")
        if st.button("Correct", use_container_width=True):
            if nlp_input:
                start = time.time()
                corrected = nlp_corrector.correct(nlp_input)
                nlp_time = (time.time() - start) * 1000
                st.session_state.pipeline_nlp_history.insert(0, {
                    'input': nlp_input, 'output': corrected, 'time': nlp_time
                })
    
    if st.session_state.pipeline_nlp_history:
        last = st.session_state.pipeline_nlp_history[0]
        st.markdown(f"**Result:** `{last['input']}` -> `{last['output']}` ({last['time']:.1f}ms)")
    
    st.divider()
    
    # Main display
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
    
    with col_metrics:
        st.subheader("Pipeline Output")
        state_display = st.empty()
        conf_display = st.empty()
        
        st.subheader("Timing Breakdown")
        timing_display = st.empty()
    
    # Stats section
    st.subheader("Performance Metrics")
    perf_placeholder = st.empty()
    
    # Video loop
    if st.session_state.pipeline_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.session_state.pipeline_running = False
        else:
            try:
                while st.session_state.pipeline_running:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    total_start = time.time()
                    
                    # Stage 1: Eye Analysis
                    t1 = time.time()
                    eye_data, annotated = eye_analyzer.process_frame(frame, config)
                    eye_time = (time.time() - t1) * 1000
                    
                    # Stage 2: YOLO
                    t2 = time.time()
                    yolo_result = yolo_classifier.classify_dual_eye(
                        eye_data.left_crop, eye_data.right_crop
                    )
                    yolo_time = (time.time() - t2) * 1000
                    
                    # Stage 3: Fusion
                    t3 = time.time()
                    raw_conf = confidence_fusion.fuse(yolo_result, eye_data.normalized_ear, config.alpha)
                    smoothed_conf = confidence_fusion.smooth_ema(raw_conf)
                    fusion_time = (time.time() - t3) * 1000
                    
                    total_time = (time.time() - total_start) * 1000
                    
                    # Determine state
                    eye_state = EyeState.OPEN if smoothed_conf >= config.blink_threshold else EyeState.CLOSED
                    
                    # Update stats
                    stats = st.session_state.pipeline_stats
                    stats['frames'] += 1
                    stats['eye_times'].append(eye_time)
                    stats['yolo_times'].append(yolo_time)
                    stats['fusion_times'].append(fusion_time)
                    stats['total_times'].append(total_time)
                    stats['confidences'].append(smoothed_conf)
                    
                    # Keep last 100
                    for key in ['eye_times', 'yolo_times', 'fusion_times', 'total_times', 'confidences']:
                        if len(stats[key]) > 100:
                            stats[key] = stats[key][-100:]
                    
                    # Draw on frame
                    state_color = (0, 255, 0) if eye_state == EyeState.OPEN else (0, 0, 255)
                    cv2.putText(annotated, f"{eye_state.value.upper()}", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
                    cv2.putText(annotated, f"Conf: {smoothed_conf:.1%} | {total_time:.0f}ms", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Metrics
                    state_text = "Open" if eye_state == EyeState.OPEN else "Closed"
                    state_display.metric("Eye State", f"{state_text}")
                    conf_display.metric("Fused Confidence", f"{smoothed_conf:.1%}")
                    
                    timing_display.markdown(f"""
                    | Stage | Time |
                    |-------|------|
                    | Eye Analysis | {eye_time:.1f} ms |
                    | YOLO | {yolo_time:.1f} ms |
                    | Fusion | {fusion_time:.2f} ms |
                    | **Total** | **{total_time:.1f} ms** |
                    """)
                    
                    # Performance summary
                    perf_placeholder.markdown(f"""
                    | Metric | Eye Analysis | YOLO | Fusion | Total |
                    |--------|-------------|------|--------|-------|
                    | Mean | {np.mean(stats['eye_times']):.1f} ms | {np.mean(stats['yolo_times']):.1f} ms | {np.mean(stats['fusion_times']):.2f} ms | {np.mean(stats['total_times']):.1f} ms |
                    | Std | {np.std(stats['eye_times']):.1f} | {np.std(stats['yolo_times']):.1f} | {np.std(stats['fusion_times']):.2f} | {np.std(stats['total_times']):.1f} |
                    
                    **Frames:** {stats['frames']} | **Avg FPS:** {1000/np.mean(stats['total_times']):.1f} | **Avg Confidence:** {np.mean(stats['confidences']):.1%}
                    """)
                    
            finally:
                cap.release()
    else:
        video_placeholder.info("Click 'Start Pipeline' to begin full integration test")
        
        # Show final stats
        stats = st.session_state.pipeline_stats
        if stats['frames'] > 0:
            st.markdown(f"""
            ### Final Session Summary
            
            | Metric | Eye Analysis | YOLO | Fusion | Total |
            |--------|-------------|------|--------|-------|
            | Mean | {np.mean(stats['eye_times']):.1f} ms | {np.mean(stats['yolo_times']):.1f} ms | {np.mean(stats['fusion_times']):.2f} ms | {np.mean(stats['total_times']):.1f} ms |
            | Std | {np.std(stats['eye_times']):.1f} | {np.std(stats['yolo_times']):.1f} | {np.std(stats['fusion_times']):.2f} | {np.std(stats['total_times']):.1f} |
            
            **Total Frames:** {stats['frames']} | **Theoretical Max FPS:** {1000/np.mean(stats['total_times']):.1f}
            """)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Pipeline Tester - Eye-Blink Morse",
        page_icon="",
        layout="wide"
    )
    
    st.title("Eye-Blink Morse Code - Pipeline Tester")
    st.markdown("*Interactive testing for each system component*")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Test:",
        [
            "NLP Correction",
            "YOLO Classification",
            "EAR Analysis",
            "Full Pipeline"
        ]
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("**Author:** AI Lab - Tel-U")
    st.sidebar.markdown("**Date:** January 2026")
    
    # Render selected page
    if page == "NLP Correction":
        render_nlp_test()
    elif page == "YOLO Classification":
        render_yolo_test()
    elif page == "EAR Analysis":
        render_ear_test()
    elif page == "Full Pipeline":
        render_full_pipeline_test()


if __name__ == "__main__":
    main()
