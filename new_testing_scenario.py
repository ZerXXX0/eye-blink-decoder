"""
Offline Video Eye-Blink Morse Decoder (Streamlit)
=================================================

This app reuses the core pipeline components from implementation.py, but
processes uploaded video files instead of real-time webcam input.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from implementation import (
	BlinkDetector,
	BlinkType,
	ConfidenceFusion,
	EyeAnalyzer,
	EyeState,
	IndoBERTCorrector,
	MorseDecoder,
	RuleBasedCorrector,
	SystemConfig,
	YOLOEyeClassifier,
)


@dataclass
class AnalysisResult:
	frame_logs: List[Dict[str, Any]]
	blink_logs: List[Dict[str, Any]]
	symbol_stream: str
	raw_text: str
	nlp_text: str
	processed_frames: int
	total_frames: int
	effective_fps: float
	source_fps: float


def discover_model_paths() -> List[str]:
	"""Return available YOLO weight files from runs/classify."""
	root = Path("runs") / "classify"
	if not root.exists():
		return []

	model_paths = sorted(root.glob("**/weights/best.pt"))
	return [str(path).replace("\\", "/") for path in model_paths]


def make_csv(rows: List[Dict[str, Any]]) -> str:
	"""Convert row dictionaries to CSV text."""
	if not rows:
		return ""

	output = StringIO()
	writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
	writer.writeheader()
	writer.writerows(rows)
	return output.getvalue()


def apply_nlp_correction(raw_text: str, enable_nlp: bool, engine: str) -> str:
	"""Apply optional NLP correction over sentence blocks."""
	if not enable_nlp:
		return raw_text

	if not raw_text.strip():
		return ""

	try:
		if engine == "indobert":
			corrector = IndoBERTCorrector()
		else:
			corrector = RuleBasedCorrector()

		corrected_parts: List[str] = []
		for part in raw_text.split("\n\n"):
			compact = " ".join(part.split())
			if compact:
				corrected_parts.append(corrector.correct(compact))

		return "\n\n".join(corrected_parts)
	except Exception as exc:
		st.warning(f"NLP correction fallback to raw text: {exc}")
		return raw_text


def analyze_video(
	video_path: str,
	config: SystemConfig,
	dot_dash_threshold_ms: float,
	analysis_fps: float,
	smoothing_mode: str,
	enable_gap_logic: bool,
	max_frames: int,
	preview_step: int,
) -> AnalysisResult:
	"""Run frame-by-frame analysis for an uploaded video file."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError("Unable to open uploaded video.")

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
	source_fps = source_fps if source_fps > 0 else 30.0

	stride = max(1, int(round(source_fps / max(1.0, analysis_fps))))
	effective_fps = source_fps / stride

	eye_analyzer = EyeAnalyzer()
	yolo_classifier = YOLOEyeClassifier(config.yolo_model_path, config.use_gpu)
	fusion = ConfidenceFusion(config.smoothing_window, config.ema_alpha)

	class SimpleCalibration:
		pass

	calibration = SimpleCalibration()
	calibration.avg_blink_duration_ms = dot_dash_threshold_ms
	blink_detector = BlinkDetector(config, calibration)
	blink_detector.estimated_fps = effective_fps
	blink_detector.update_fps = lambda: None  # Keep timing frame-rate based for offline video.

	morse_decoder = MorseDecoder(config)

	frame_logs: List[Dict[str, Any]] = []
	blink_logs: List[Dict[str, Any]] = []
	symbols: List[str] = []

	progress_bar = st.progress(0, text="Analyzing video frames...")
	preview_placeholder = st.empty()
	stats_placeholder = st.empty()

	processed_frames = 0
	frame_index = 0

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			if max_frames > 0 and frame_index >= max_frames:
				break

			if frame_index % stride != 0:
				frame_index += 1
				continue

			eye_data, annotated = eye_analyzer.process_frame(frame, config)

			yolo_result = None
			state = EyeState.UNKNOWN
			confidence = 0.5
			ear = float(eye_data.avg_ear)
			normalized_ear = float(eye_data.normalized_ear)

			blink_event = None
			letter_gap_triggered = False
			word_gap_triggered = False
			sentence_gap_triggered = False

			if eye_data.landmarks_detected:
				yolo_result = yolo_classifier.classify_dual_eye(eye_data.left_crop, eye_data.right_crop)
				fused = fusion.fuse(yolo_result, eye_data.normalized_ear, config.alpha)

				if smoothing_mode == "ema":
					confidence = float(fusion.smooth_ema(fused))
				elif smoothing_mode == "rolling":
					confidence = float(fusion.smooth_rolling(fused))
				else:
					confidence = float(fused)

				state = EyeState.OPEN if confidence >= config.blink_threshold else EyeState.CLOSED
				blink_event = blink_detector.process(confidence)

				if blink_event:
					symbol = blink_event.blink_type.value
					symbols.append(symbol)
					morse_decoder.add_symbol(symbol)

					blink_logs.append(
						{
							"event_index": len(blink_logs) + 1,
							"frame_start": blink_event.start_frame,
							"frame_end": blink_event.end_frame,
							"duration_frames": blink_event.duration_frames,
							"duration_ms": round(blink_event.duration_ms, 2),
							"blink_type": blink_event.blink_type.value,
							"confidence": round(float(blink_event.confidence), 4),
						}
					)

				if enable_gap_logic:
					if blink_detector.is_sentence_gap():
						sentence_gap_triggered = morse_decoder.process_sentence_gap()
					elif blink_detector.is_word_gap():
						word_gap_triggered = morse_decoder.process_word_gap()
					elif blink_detector.is_letter_gap():
						letter_gap_triggered = morse_decoder.process_letter_gap() is not None

			frame_logs.append(
				{
					"frame_index": frame_index,
					"time_sec": round(frame_index / source_fps, 3),
					"landmarks_detected": bool(eye_data.landmarks_detected),
					"ear": round(ear, 5),
					"normalized_ear": round(normalized_ear, 5),
					"yolo_state": yolo_result.state.value if yolo_result else EyeState.UNKNOWN.value,
					"yolo_open_prob": round(float(yolo_result.open_prob), 5) if yolo_result else 0.0,
					"yolo_closed_prob": round(float(yolo_result.closed_prob), 5) if yolo_result else 0.0,
					"fused_confidence": round(confidence, 5),
					"eye_state": state.value,
					"blink_detected": bool(blink_event),
					"blink_type": blink_event.blink_type.value if blink_event else "",
					"blink_duration_ms": round(float(blink_event.duration_ms), 2) if blink_event else 0.0,
					"letter_gap_triggered": letter_gap_triggered,
					"word_gap_triggered": word_gap_triggered,
					"sentence_gap_triggered": sentence_gap_triggered,
					"current_morse_sequence": morse_decoder.get_current_sequence(),
					"raw_text_so_far": morse_decoder.get_decoded_text(),
				}
			)

			processed_frames += 1

			if processed_frames % max(1, preview_step) == 0:
				preview_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
				preview_placeholder.image(preview_frame, caption="Annotated analysis preview", channels="RGB")

			if total_frames > 0:
				progress_value = min(1.0, (frame_index + 1) / total_frames)
			else:
				progress_value = 0.0

			progress_bar.progress(progress_value, text=f"Processed frames: {processed_frames}")
			stats_placeholder.caption(
				f"Source FPS: {source_fps:.2f} | Analysis stride: {stride} | Effective FPS: {effective_fps:.2f}"
			)

			frame_index += 1
	finally:
		cap.release()
		eye_analyzer.close()
		progress_bar.empty()

	raw_text = morse_decoder.get_decoded_text()

	return AnalysisResult(
		frame_logs=frame_logs,
		blink_logs=blink_logs,
		symbol_stream="".join(symbols),
		raw_text=raw_text,
		nlp_text="",
		processed_frames=processed_frames,
		total_frames=total_frames,
		effective_fps=effective_fps,
		source_fps=source_fps,
	)


def build_ui() -> None:
	st.set_page_config(page_title="Video Eye-Blink Morse Decoder", layout="wide")
	st.title("Video Eye-Blink to Morse Decoder")
	st.caption("Offline analysis using reusable modules from implementation.py")

	with st.sidebar:
		st.header("Input")
		uploaded_video = st.file_uploader(
			"Upload video",
			type=["mp4", "mov", "avi", "mkv", "webm"],
			accept_multiple_files=False,
		)

		st.header("Model")
		model_candidates = discover_model_paths()
		default_model = "runs/classify/nano_100/weights/best.pt"
		model_path = st.selectbox(
			"YOLO model path",
			options=model_candidates if model_candidates else [default_model],
			index=(model_candidates.index(default_model) if default_model in model_candidates else 0),
			help="Eye state classifier weights.",
		)
		use_gpu = st.checkbox("Use GPU", value=True)

		st.header("Fusion and Detection")
		alpha = st.slider("YOLO-EAR alpha", 0.0, 1.0, 0.4, 0.01)
		blink_threshold = st.slider("Blink threshold", 0.05, 0.95, 0.5, 0.01)
		dot_dash_threshold_ms = st.slider("Dot-Dash threshold (ms)", 50, 1200, 200, 10)
		smoothing_mode = st.selectbox("Smoothing mode", ["ema", "rolling", "none"], index=0)
		smoothing_window = st.slider("Rolling window", 1, 30, 5, 1)
		ema_alpha = st.slider("EMA alpha", 0.01, 0.99, 0.30, 0.01)

		st.header("Morse Gap Timing")
		enable_gap_logic = st.checkbox("Enable gap-based decoding", value=True)
		letter_gap = st.slider("Letter gap (s)", 0.2, 6.0, 1.5, 0.1)
		word_gap = st.slider("Word gap (s)", 0.5, 10.0, 3.0, 0.1)
		sentence_gap = st.slider("Sentence gap (s)", 1.0, 14.0, 5.0, 0.1)

		st.header("EAR")
		ear_min = st.slider("EAR min", 0.01, 0.40, 0.15, 0.005)
		ear_max = st.slider("EAR max", 0.05, 0.60, 0.35, 0.005)

		st.header("NLP")
		enable_nlp = st.checkbox("Enable NLP correction", value=False)
		nlp_engine = st.selectbox("NLP engine", ["rule_based", "indobert"], index=0)

		st.header("Performance and Logging")
		analysis_fps = st.number_input("Analysis FPS", min_value=1.0, max_value=120.0, value=15.0, step=1.0)
		max_frames = st.number_input(
			"Max frames to process (0 = all)", min_value=0, max_value=10_000_000, value=0, step=100
		)
		preview_step = st.number_input("Preview update interval (frames)", min_value=1, max_value=500, value=15)
		rows_to_show = st.number_input("Rows to show in tables", min_value=10, max_value=10_000, value=300)

	if uploaded_video is None:
		st.info("Upload a video file to start analysis.")
		return

	with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix or ".mp4") as tmp:
		tmp.write(uploaded_video.read())
		temp_video_path = tmp.name

	sentence_gap = max(sentence_gap, word_gap + 0.1)
	ear_max = max(ear_max, ear_min + 0.01)

	config = SystemConfig(
		alpha=float(alpha),
		blink_threshold=float(blink_threshold),
		letter_gap_seconds=float(letter_gap),
		word_gap_seconds=float(word_gap),
		sentence_gap_seconds=float(sentence_gap),
		ear_min=float(ear_min),
		ear_max=float(ear_max),
		smoothing_window=int(smoothing_window),
		ema_alpha=float(ema_alpha),
		yolo_model_path=model_path,
		use_gpu=bool(use_gpu),
	)

	run_analysis = st.button("Run Video Analysis", type="primary", use_container_width=True)
	if not run_analysis:
		st.video(temp_video_path)
		return

	with st.spinner("Running eye-blink analysis..."):
		result = analyze_video(
			video_path=temp_video_path,
			config=config,
			dot_dash_threshold_ms=float(dot_dash_threshold_ms),
			analysis_fps=float(analysis_fps),
			smoothing_mode=smoothing_mode,
			enable_gap_logic=enable_gap_logic,
			max_frames=int(max_frames),
			preview_step=int(preview_step),
		)

		nlp_text = apply_nlp_correction(result.raw_text, bool(enable_nlp), nlp_engine)
		result.nlp_text = nlp_text

	open_count = sum(1 for row in result.frame_logs if row["eye_state"] == EyeState.OPEN.value)
	closed_count = sum(1 for row in result.frame_logs if row["eye_state"] == EyeState.CLOSED.value)
	unknown_count = sum(1 for row in result.frame_logs if row["eye_state"] == EyeState.UNKNOWN.value)
	dot_count = sum(1 for row in result.blink_logs if row["blink_type"] == BlinkType.DOT.value)
	dash_count = sum(1 for row in result.blink_logs if row["blink_type"] == BlinkType.DASH.value)

	top_cols = st.columns(6)
	top_cols[0].metric("Frames processed", f"{result.processed_frames}")
	top_cols[1].metric("Source FPS", f"{result.source_fps:.2f}")
	top_cols[2].metric("Effective FPS", f"{result.effective_fps:.2f}")
	top_cols[3].metric("Open / Closed", f"{open_count} / {closed_count}")
	top_cols[4].metric("Unknown", f"{unknown_count}")
	top_cols[5].metric("Dots / Dashes", f"{dot_count} / {dash_count}")

	st.subheader("Morse and Text Outputs")
	st.text_area("Dot-Dash symbol stream", value=result.symbol_stream, height=100)
	st.text_area("Raw decoded text", value=result.raw_text, height=120)
	st.text_area("NLP corrected text", value=result.nlp_text, height=120)

	frame_csv = make_csv(result.frame_logs)
	blink_csv = make_csv(result.blink_logs)

	download_cols = st.columns(2)
	download_cols[0].download_button(
		label="Download per-frame classification CSV",
		data=frame_csv,
		file_name="per_frame_eye_state.csv",
		mime="text/csv",
		use_container_width=True,
	)
	download_cols[1].download_button(
		label="Download blink events CSV",
		data=blink_csv,
		file_name="blink_events.csv",
		mime="text/csv",
		use_container_width=True,
	)

	st.subheader("Per-frame Eye State Classification")
	st.dataframe(result.frame_logs[: int(rows_to_show)], use_container_width=True)

	st.subheader("Blink Event Log")
	if result.blink_logs:
		st.dataframe(result.blink_logs[: int(rows_to_show)], use_container_width=True)
	else:
		st.info("No blink events detected for current configuration.")

	chart_rows = result.frame_logs[:2000]
	if chart_rows:
		st.subheader("Confidence Trend")
		st.line_chart(
			{
				"fused_confidence": [row["fused_confidence"] for row in chart_rows],
				"normalized_ear": [row["normalized_ear"] for row in chart_rows],
				"yolo_open_prob": [row["yolo_open_prob"] for row in chart_rows],
			}
		)


if __name__ == "__main__":
	build_ui()
