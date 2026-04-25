"""
Streamlit Morse Generator with synchronized audio/visual playback and webcam capture.

Features:
- Text input to International Morse code conversion
- Configurable frequency, volume, and timing controls
- Synchronized blinking indicator with Web Audio beeps
- OpenCV webcam preview and snapshot capture
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from io import BytesIO
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from streamlit.components.v1 import html


MORSE_CODE_MAP: Dict[str, str] = {
	"A": ".-",
	"B": "-...",
	"C": "-.-.",
	"D": "-..",
	"E": ".",
	"F": "..-.",
	"G": "--.",
	"H": "....",
	"I": "..",
	"J": ".---",
	"K": "-.-",
	"L": ".-..",
	"M": "--",
	"N": "-.",
	"O": "---",
	"P": ".--.",
	"Q": "--.-",
	"R": ".-.",
	"S": "...",
	"T": "-",
	"U": "..-",
	"V": "...-",
	"W": ".--",
	"X": "-..-",
	"Y": "-.--",
	"Z": "--..",
	"0": "-----",
	"1": ".----",
	"2": "..---",
	"3": "...--",
	"4": "....-",
	"5": ".....",
	"6": "-....",
	"7": "--...",
	"8": "---..",
	"9": "----.",
	".": ".-.-.-",
	",": "--..--",
	"?": "..--..",
	"'": ".----.",
	"!": "-.-.--",
	"/": "-..-.",
	"(": "-.--.",
	")": "-.--.-",
	"&": ".-...",
	":": "---...",
	";": "-.-.-.",
	"=": "-...-",
	"+": ".-.-.",
	"-": "-....-",
	"_": "..--.-",
	'"': ".-..-.",
	"$": "...-..-",
	"@": ".--.-.",
}


def normalize_text(text: str) -> str:
	"""Normalize whitespace for predictable Morse conversion."""
	text = text.replace("\n", " ")
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def text_to_morse(text: str) -> Tuple[str, List[Dict[str, str]]]:
	"""Convert input text to Morse output and return unsupported characters."""
	normalized = normalize_text(text).upper()
	if not normalized:
		return "", []

	words = normalized.split(" ")
	converted_words: List[str] = []
	unknown: List[Dict[str, str]] = []

	for wi, word in enumerate(words):
		letters: List[str] = []
		for ci, ch in enumerate(word):
			code = MORSE_CODE_MAP.get(ch)
			if code is None:
				unknown.append({"char": ch, "word_index": str(wi), "char_index": str(ci)})
				letters.append("?")
			else:
				letters.append(code)
		converted_words.append(" ".join(letters))

	morse = " / ".join(converted_words)
	return morse, unknown


def build_playback_events(
	text: str,
	dot_ms: float,
	dash_ms: float,
	intra_symbol_gap_ms: float,
	letter_gap_ms: float,
	word_gap_ms: float,
) -> List[Dict[str, object]]:
	"""Build ON/OFF timeline events from text using millisecond timing controls."""
	events: List[Dict[str, object]] = []
	normalized = normalize_text(text).upper()
	if not normalized:
		return events

	dot_ms = max(5.0, dot_ms)
	dash_ms = max(dot_ms, dash_ms)
	intra_symbol_gap_ms = max(1.0, intra_symbol_gap_ms)
	letter_gap_ms = max(intra_symbol_gap_ms, letter_gap_ms)
	word_gap_ms = max(letter_gap_ms, word_gap_ms)

	words = normalized.split(" ")
	for wi, word in enumerate(words):
		for li, ch in enumerate(word):
			morse = MORSE_CODE_MAP.get(ch)
			if morse is None:
				continue

			for si, symbol in enumerate(morse):
				on_duration = dot_ms if symbol == "." else dash_ms
				events.append(
					{
						"state": "on",
						"duration_ms": float(on_duration),
						"symbol": symbol,
						"char": ch,
						"word_index": wi,
						"letter_index": li,
					}
				)

				if si < len(morse) - 1:
					events.append(
						{
							"state": "off",
							"duration_ms": float(intra_symbol_gap_ms),
							"symbol": "",
							"char": ch,
							"word_index": wi,
							"letter_index": li,
						}
					)

			if li < len(word) - 1:
				events.append(
					{
						"state": "off",
						"duration_ms": float(letter_gap_ms),
						"symbol": "",
						"char": "",
						"word_index": wi,
						"letter_index": li,
					}
				)

		if wi < len(words) - 1:
			events.append(
				{
					"state": "off",
					"duration_ms": float(word_gap_ms),
					"symbol": "",
					"char": "",
					"word_index": wi,
					"letter_index": -1,
				}
			)

	return events


def generate_wav_from_events(
	events: List[Dict[str, object]],
	frequency_hz: float,
	volume: float,
	sample_rate: int = 44100,
) -> bytes:
	"""Create WAV audio bytes from ON/OFF event timeline."""
	chunks: List[np.ndarray] = []
	amp = float(np.clip(volume, 0.0, 1.0))
	freq = float(max(50.0, frequency_hz))

	for event in events:
		duration_sec = max(0.0, float(event["duration_ms"]) / 1000.0)
		n_samples = max(1, int(duration_sec * sample_rate))
		if event["state"] == "on":
			t = np.arange(n_samples, dtype=np.float32) / sample_rate
			wave = np.sin(2.0 * np.pi * freq * t) * amp
		else:
			wave = np.zeros(n_samples, dtype=np.float32)
		chunks.append(wave)

	if not chunks:
		return b""

	signal = np.concatenate(chunks)
	signal_i16 = np.int16(np.clip(signal, -1.0, 1.0) * 32767)

	import wave

	with BytesIO() as buffer:
		with wave.open(buffer, "wb") as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)
			wf.setframerate(sample_rate)
			wf.writeframes(signal_i16.tobytes())
		return buffer.getvalue()


def render_synchronized_player(
	events: List[Dict[str, object]],
	frequency_hz: float,
	volume: float,
	indicator_shape: str,
) -> None:
	"""Render browser-side synchronized visual and audio Morse playback."""
	payload = {
		"events": events,
		"frequency_hz": float(frequency_hz),
		"volume": float(np.clip(volume, 0.0, 1.0)),
		"shape": indicator_shape,
	}

	payload_json = json.dumps(payload)

	indicator_css = "border-radius: 50%;" if indicator_shape == "circle" else "border-radius: 8px;"

	html(
		f"""
		<div style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);">
		  <div style="display:flex; gap:8px; align-items:center; margin-bottom:10px; flex-wrap: wrap;">
			<button id="playBtn" style="padding:6px 12px; border-radius:8px; border:1px solid #cfd8e3; background:#ffffff; cursor:pointer;">Play</button>
			<button id="stopBtn" style="padding:6px 12px; border-radius:8px; border:1px solid #cfd8e3; background:#ffffff; cursor:pointer;">Stop</button>
			<span id="statusText" style="font-size: 13px; color:#555;">Ready</span>
			<span id="clockText" style="font-size: 12px; color:#7a8698;">t+0ms</span>
		  </div>

		  <div style="display:flex; gap:14px; align-items:center;">
			<div id="indicator" style="width:110px; height:110px; background:#3a3a3a; {indicator_css} box-shadow: inset 0 0 8px rgba(0,0,0,0.25); transition: transform 80ms ease, box-shadow 80ms ease, background 80ms ease;"></div>
			<div style="font-size:13px; color:#555; min-width: 220px;">
			  <div id="activeSymbol">Current symbol: -</div>
			  <div id="timelineInfo">Events: 0</div>
			  <div id="countdownText">Countdown: -</div>
			</div>
		  </div>

		  <div style="margin-top:10px; padding:8px; border:1px solid #e5eaf1; border-radius:10px; background:#fbfdff;">
			<div style="font-size:12px; color:#6b7280; margin-bottom:6px;">Next symbols with countdown</div>
			<div id="nextSymbolsPanel" style="display:flex; gap:8px; flex-wrap:wrap; min-height: 30px;"></div>
		  </div>

		  <div style="margin-top: 10px; height: 7px; background:#e8edf4; border-radius: 999px; overflow:hidden;">
			<div id="progressBar" style="height:100%; width:0%; background:linear-gradient(90deg, #1570ef 0%, #23b3ff 100%);"></div>
		  </div>

		  <div style="margin-top:8px;">
			<div id="eventStrip" style="display:flex; gap:4px; overflow-x:auto; padding-bottom:2px;"></div>
		  </div>
		</div>

		<script>
		  const payload = {payload_json};
		  const events = payload.events || [];
		  const indicator = document.getElementById('indicator');
		  const playBtn = document.getElementById('playBtn');
		  const stopBtn = document.getElementById('stopBtn');
		  const statusText = document.getElementById('statusText');
		  const clockText = document.getElementById('clockText');
		  const activeSymbol = document.getElementById('activeSymbol');
		  const timelineInfo = document.getElementById('timelineInfo');
		  const countdownText = document.getElementById('countdownText');
		  const nextSymbolsPanel = document.getElementById('nextSymbolsPanel');
		  const progressBar = document.getElementById('progressBar');
		  const eventStrip = document.getElementById('eventStrip');

		  timelineInfo.textContent = `Events: ${{events.length}}`;

		  let audioCtx = null;
		  let isPlaying = false;
		  let cancelToken = 0;
		  let activeNodes = [];
		  let rafId = 0;
		  let playbackStartEpoch = 0;

		  const totalDuration = events.reduce((acc, e) => acc + (Number(e.duration_ms) || 0), 0);
		  const eventStartMs = [];
		  let cursor = 0;
		  for (const ev of events) {{
			eventStartMs.push(cursor);
			cursor += Math.max(1, Number(ev.duration_ms) || 1);
		  }}

		  const onEventIndices = [];
		  for (let i = 0; i < events.length; i += 1) {{
			if (events[i].state === 'on' && events[i].symbol) onEventIndices.push(i);
		  }}

		  const eventPills = [];
		  function initEventStrip() {{
			eventStrip.innerHTML = '';
			eventPills.length = 0;
			for (let i = 0; i < Math.min(events.length, 160); i += 1) {{
			  const ev = events[i];
			  const pill = document.createElement('div');
			  const isOn = ev.state === 'on';
			  const symbol = isOn ? (ev.symbol || '•') : 'gap';
			  pill.textContent = symbol;
			  pill.style.fontSize = '11px';
			  pill.style.minWidth = '24px';
			  pill.style.textAlign = 'center';
			  pill.style.padding = '3px 6px';
			  pill.style.borderRadius = '999px';
			  pill.style.border = '1px solid #d7dfe9';
			  pill.style.background = isOn ? '#f7fbff' : '#f7f7f8';
			  pill.style.color = '#516073';
			  pill.style.transition = 'all 80ms ease';
			  eventStrip.appendChild(pill);
			  eventPills.push(pill);
			}}
		  }}

		  initEventStrip();

		  function setIndicator(on) {{
			if (on) {{
			  indicator.style.background = '#ff3b30';
			  indicator.style.boxShadow = '0 0 24px rgba(255,59,48,0.75)';
			  indicator.style.transform = 'scale(1.06)';
			}} else {{
			  indicator.style.background = '#3a3a3a';
			  indicator.style.boxShadow = 'inset 0 0 8px rgba(0,0,0,0.25)';
			  indicator.style.transform = 'scale(1.0)';
			}}
		  }}

		  function stopAllAudio() {{
			for (const node of activeNodes) {{
			  try {{ node.stop(); }} catch (e) {{}}
			  try {{ node.disconnect(); }} catch (e) {{}}
			}}
			activeNodes = [];
		  }}

		  function findCurrentEventIndex(elapsedMs) {{
			for (let i = events.length - 1; i >= 0; i -= 1) {{
			  if (elapsedMs >= eventStartMs[i]) return i;
			}}
			return 0;
		  }}

		  function getUpcomingOnSymbols(elapsedMs, count = 3) {{
			const out = [];
			for (let k = 0; k < onEventIndices.length; k += 1) {{
			  const idx = onEventIndices[k];
			  const start = eventStartMs[idx];
			  const end = start + Math.max(1, Number(events[idx].duration_ms) || 1);
			  const ev = events[idx];
			  if (elapsedMs <= end) {{
				const inMs = Math.max(0, Math.round(start - elapsedMs));
				out.push({{ symbol: ev.symbol || '-', in_ms: inMs, char: ev.char || '' }});
				if (out.length >= count) break;
			  }}
			}}
			return out;
		  }}

		  function renderNextSymbols(elapsedMs) {{
			const upcoming = getUpcomingOnSymbols(elapsedMs, 3);
			nextSymbolsPanel.innerHTML = '';
			if (!upcoming.length) {{
			  const empty = document.createElement('div');
			  empty.textContent = '-';
			  empty.style.fontSize = '12px';
			  empty.style.color = '#6b7280';
			  nextSymbolsPanel.appendChild(empty);
			  return;
			}}
			for (const item of upcoming) {{
			  const chip = document.createElement('div');
			  chip.style.padding = '4px 8px';
			  chip.style.borderRadius = '999px';
			  chip.style.border = '1px solid #dbe3ee';
			  chip.style.background = item.in_ms <= 40 ? '#ffe3e1' : '#edf5ff';
			  chip.style.color = '#253448';
			  chip.style.fontSize = '12px';
			  chip.style.whiteSpace = 'nowrap';
			  chip.textContent = `${{item.symbol}} in ${{item.in_ms}}ms`;
			  nextSymbolsPanel.appendChild(chip);
			}}
		  }}

		  function updateEventStripHighlight(currentIndex) {{
			for (let i = 0; i < eventPills.length; i += 1) {{
			  const pill = eventPills[i];
			  if (i === currentIndex) {{
				pill.style.transform = 'translateY(-2px) scale(1.06)';
				pill.style.boxShadow = '0 4px 10px rgba(37,95,180,0.18)';
				pill.style.borderColor = '#8db8f7';
			  }} else {{
				pill.style.transform = 'none';
				pill.style.boxShadow = 'none';
				pill.style.borderColor = '#d7dfe9';
			  }}
			}}
		  }}

		  function resetVisuals() {{
			setIndicator(false);
			activeSymbol.textContent = 'Current symbol: -';
			countdownText.textContent = 'Countdown: -';
			clockText.textContent = 't+0ms';
			renderNextSymbols(0);
			updateEventStripHighlight(-1);
			progressBar.style.width = '0%';
		  }}

		  function stopPlayback() {{
			cancelToken += 1;
			isPlaying = false;
			stopAllAudio();
			if (rafId) {{
			  cancelAnimationFrame(rafId);
			  rafId = 0;
			}}
			resetVisuals();
			statusText.textContent = 'Stopped';
		  }}

		  function animationLoop(token) {{
			if (!isPlaying || token !== cancelToken) return;
			const elapsedMs = Math.max(0, performance.now() - playbackStartEpoch);
			const currentIndex = findCurrentEventIndex(elapsedMs);
			const currentEvent = events[currentIndex] || null;
			const currentStart = currentEvent ? eventStartMs[currentIndex] : 0;
			const currentDur = currentEvent ? Math.max(1, Number(currentEvent.duration_ms) || 1) : 1;
			const rem = Math.max(0, Math.round(currentStart + currentDur - elapsedMs));

			clockText.textContent = `t+${{Math.round(elapsedMs)}}ms`;
			countdownText.textContent = `Countdown: ${{rem}}ms`;
			renderNextSymbols(elapsedMs);
			updateEventStripHighlight(currentIndex);

			const pct = totalDuration > 0 ? Math.min(100, (elapsedMs / totalDuration) * 100) : 0;
			progressBar.style.width = `${{pct}}%`;

			if (elapsedMs >= totalDuration + 4) return;
			rafId = requestAnimationFrame(() => animationLoop(token));
		  }}

		  async function startPlayback() {{
			if (!events.length) {{
			  statusText.textContent = 'No events to play';
			  return;
			}}

			stopPlayback();
			const token = cancelToken;

			if (!audioCtx) {{
			  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
			}}
			if (audioCtx.state === 'suspended') {{
			  await audioCtx.resume();
			}}

			isPlaying = true;
			statusText.textContent = 'Playing';
			playbackStartEpoch = performance.now();
			rafId = requestAnimationFrame(() => animationLoop(token));

			let cursorMs = 0;
			for (let eventIndex = 0; eventIndex < events.length; eventIndex += 1) {{
			  const e = events[eventIndex];
			  if (!isPlaying || token !== cancelToken) return;

			  const state = e.state;
			  const dur = Math.max(1, Number(e.duration_ms) || 1);
			  const symbol = e.symbol || '';

			  const waitMs = Math.max(0, playbackStartEpoch + cursorMs - performance.now());
			  await new Promise((resolve) => setTimeout(resolve, waitMs));

			  if (!isPlaying || token !== cancelToken) return;

			  if (state === 'on') {{
				setIndicator(true);
				activeSymbol.textContent = `Current symbol: ${{symbol || '-'}}`;

				const osc = audioCtx.createOscillator();
				const gain = audioCtx.createGain();
				osc.type = 'sine';
				osc.frequency.value = Number(payload.frequency_hz) || 650;
				gain.gain.value = Number(payload.volume) || 0.4;
				osc.connect(gain);
				gain.connect(audioCtx.destination);
				osc.start();
				osc.stop(audioCtx.currentTime + dur / 1000.0);
				activeNodes.push(osc);
			  }} else {{
				setIndicator(false);
				activeSymbol.textContent = 'Current symbol: gap';
			  }}

			  cursorMs += dur;
			}}

			if (token === cancelToken) {{
			  isPlaying = false;
			  if (rafId) {{
				cancelAnimationFrame(rafId);
				rafId = 0;
			  }}
			  setIndicator(false);
			  activeSymbol.textContent = 'Current symbol: done';
			  countdownText.textContent = 'Countdown: 0ms';
			  nextSymbolsPanel.innerHTML = '<div style="font-size:12px; color:#6b7280;">-</div>';
			  progressBar.style.width = '100%';
			  statusText.textContent = 'Completed';
			}}
		  }}

		  renderNextSymbols(0);
		  playBtn.addEventListener('click', startPlayback);
		  stopBtn.addEventListener('click', stopPlayback);
		</script>
		""",
		height=380,
	)


def webcam_snapshot(camera_index: int, width: int, height: int) -> np.ndarray | None:
	"""Capture a single frame from webcam using OpenCV VideoCapture."""
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		return None

	try:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		frame = None
		for _ in range(3):
			ok, temp = cap.read()
			if ok:
				frame = temp
		if frame is None:
			return None

		return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	finally:
		cap.release()


def webcam_preview(camera_index: int, width: int, height: int, preview_seconds: float, fps_limit: int) -> None:
	"""Preview webcam stream in Streamlit for a finite duration."""
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		st.error("Unable to open webcam.")
		return

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	placeholder = st.empty()
	started = time.time()
	delay = 1.0 / max(1, fps_limit)

	try:
		while time.time() - started < preview_seconds:
			ok, frame = cap.read()
			if not ok:
				st.error("Failed reading frame from webcam.")
				break

			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			placeholder.image(rgb, caption="OpenCV webcam preview", channels="RGB")
			time.sleep(delay)
	finally:
		cap.release()


def webcam_record_mp4_until_stop(
	camera_index: int,
	width: int,
	height: int,
	fps: int,
	output_path: str,
	stop_event: threading.Event,
	status: Dict[str, object],
) -> None:
	"""Background recorder that writes MP4 frames until stop_event is set."""
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		status["error"] = "Unable to open webcam."
		return

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
	actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
	fps = max(1, int(fps))

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(output_path, fourcc, float(fps), (actual_width, actual_height))
	if not writer.isOpened():
		cap.release()
		status["error"] = "Unable to initialize MP4 writer."
		return

	try:
		status["frames_written"] = 0
		status["error"] = ""
		delay = 1.0 / fps
		while not stop_event.is_set():
			ok, frame = cap.read()
			if not ok:
				status["error"] = "Failed reading frame from webcam."
				break

			if frame.shape[1] != actual_width or frame.shape[0] != actual_height:
				frame = cv2.resize(frame, (actual_width, actual_height))

			writer.write(frame)
			status["last_frame_rgb"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			status["frames_written"] = int(status.get("frames_written", 0)) + 1
			time.sleep(delay)
	finally:
		writer.release()
		cap.release()


def main() -> None:
	st.set_page_config(page_title="Morse Generator + Webcam", layout="wide")
	st.title("Morse Code Generator with Audio, Visual Sync, and Webcam")
	st.caption("Interactive Morse playback inspired by morsecodegenerator.org behavior")

	with st.sidebar:
		st.header("Input")
		input_text = st.text_area(
			"Text to encode",
			value="SOS HELLO WORLD",
			height=140,
			help="Letters, numbers, and punctuation are supported.",
		)

		st.header("Audio")
		frequency_hz = st.slider("Frequency (Hz)", 100, 2000, 650, 10)
		volume = st.slider("Volume", 0.0, 1.0, 0.35, 0.01)

		st.header("Timing")
		dot_ms = st.slider("Dot duration (ms)", 20, 500, 120, 5)
		dash_ms = st.slider("Dash duration (ms)", 20, 1500, 360, 5)
		intra_symbol_gap_ms = st.slider("Intra-symbol gap (ms)", 5, 1000, 120, 5)
		letter_gap_ms = st.slider("Letter gap (ms)", 10, 3000, 360, 10)
		word_gap_ms = st.slider("Word gap (ms)", 20, 5000, 840, 10)

		st.header("Visual")
		indicator_shape = st.selectbox("Indicator shape", ["circle", "rectangle"], index=0)

		st.header("Webcam")
		camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0)
		cam_width = st.number_input("Camera width", min_value=160, max_value=1920, value=640, step=20)
		cam_height = st.number_input("Camera height", min_value=120, max_value=1080, value=480, step=20)
		preview_seconds = st.slider("Preview duration (seconds)", 1, 30, 5)
		preview_fps = st.slider("Preview FPS limit", 1, 60, 12)
		record_fps = st.slider("Record FPS", 1, 60, 20)

	if "recorder_state" not in st.session_state:
		st.session_state.recorder_state = {
			"is_recording": False,
			"thread": None,
			"stop_event": None,
			"output_path": "",
			"status": {"frames_written": 0, "error": "", "last_frame_rgb": None},
			"video_bytes": None,
		}

	morse_text, unknown_chars = text_to_morse(input_text)
	events = build_playback_events(
		text=input_text,
		dot_ms=float(dot_ms),
		dash_ms=float(dash_ms),
		intra_symbol_gap_ms=float(intra_symbol_gap_ms),
		letter_gap_ms=float(letter_gap_ms),
		word_gap_ms=float(word_gap_ms),
	)

	total_ms = sum(float(e["duration_ms"]) for e in events)
	beep_count = sum(1 for e in events if e["state"] == "on")

	metrics = st.columns(5)
	metrics[0].metric("Morse symbols", f"{beep_count}")
	metrics[1].metric("Timeline events", f"{len(events)}")
	metrics[2].metric("Estimated duration", f"{total_ms / 1000.0:.2f}s")
	metrics[3].metric("Dot / Dash", f"{dot_ms}ms / {dash_ms}ms")
	metrics[4].metric("Unknown chars", f"{len(unknown_chars)}")

	left_col, right_col = st.columns([1.4, 1.0])

	with left_col:
		st.subheader("Morse Output")
		st.text_area("Converted Morse", value=morse_text, height=120)

		if unknown_chars:
			st.warning("Some characters are unsupported and shown as ? in Morse output.")
			st.dataframe(unknown_chars, use_container_width=True)

		st.subheader("Synchronized Audio + Visual Player")
		render_synchronized_player(
			events=events,
			frequency_hz=float(frequency_hz),
			volume=float(volume),
			indicator_shape=indicator_shape,
		)

		st.subheader("Generated WAV Preview")
		wav_bytes = generate_wav_from_events(
			events=events,
			frequency_hz=float(frequency_hz),
			volume=float(volume),
		)
		if wav_bytes:
			st.audio(wav_bytes, format="audio/wav")
			st.download_button(
				label="Download WAV",
				data=wav_bytes,
				file_name="morse_output.wav",
				mime="audio/wav",
			)
		else:
			st.info("No audio to generate. Enter text that maps to Morse symbols.")

	with right_col:
		st.subheader("Webcam Capture via OpenCV")
		st.caption("This section accesses webcam using cv2.VideoCapture.")

		recorder_state = st.session_state.recorder_state

		cap_cols = st.columns(4)
		with cap_cols[0]:
			run_preview = st.button("Run webcam preview", use_container_width=True)
		with cap_cols[1]:
			take_snapshot = st.button("Capture snapshot", use_container_width=True)
		with cap_cols[2]:
			start_recording = st.button(
				"Start MP4 Recording",
				use_container_width=True,
				disabled=bool(recorder_state["is_recording"]),
			)
		with cap_cols[3]:
			stop_recording = st.button(
				"Stop Recording",
				use_container_width=True,
				disabled=not bool(recorder_state["is_recording"]),
			)

		if run_preview:
			webcam_preview(
				camera_index=int(camera_index),
				width=int(cam_width),
				height=int(cam_height),
				preview_seconds=float(preview_seconds),
				fps_limit=int(preview_fps),
			)

		if take_snapshot:
			snap = webcam_snapshot(
				camera_index=int(camera_index),
				width=int(cam_width),
				height=int(cam_height),
			)
			if snap is None:
				st.error("Snapshot failed. Check camera permissions/index.")
			else:
				st.image(snap, caption="Captured snapshot", channels="RGB")

		if start_recording:
			old_path = str(recorder_state.get("output_path") or "")
			if old_path and os.path.exists(old_path):
				try:
					os.remove(old_path)
				except OSError:
					pass

			with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
				output_path = tmp.name

			stop_event = threading.Event()
			status = {"frames_written": 0, "error": "", "last_frame_rgb": None}
			thread = threading.Thread(
				target=webcam_record_mp4_until_stop,
				args=(
					int(camera_index),
					int(cam_width),
					int(cam_height),
					int(record_fps),
					output_path,
					stop_event,
					status,
				),
				daemon=True,
			)

			recorder_state["is_recording"] = True
			recorder_state["thread"] = thread
			recorder_state["stop_event"] = stop_event
			recorder_state["output_path"] = output_path
			recorder_state["status"] = status
			recorder_state["video_bytes"] = None
			thread.start()
			st.success("Recording started. Press Stop Recording when finished.")

		if recorder_state["is_recording"]:
			st.info(f"Recording... frames captured: {int(recorder_state['status'].get('frames_written', 0))}")

		if stop_recording and recorder_state["is_recording"]:
			recorder_state["stop_event"].set()
			recorder_state["thread"].join(timeout=8)
			recorder_state["is_recording"] = False

			if recorder_state["thread"].is_alive():
				st.error("Recorder did not stop cleanly. Please press Stop again.")
				st.stop()

			record_error = str(recorder_state["status"].get("error") or "")
			frames_written = int(recorder_state["status"].get("frames_written", 0))
			output_path = str(recorder_state.get("output_path") or "")

			if frames_written <= 0 or not output_path or not os.path.exists(output_path):
				st.error("Recording failed. No frames were written.")
				if record_error:
					st.error(record_error)
			else:
				with open(output_path, "rb") as f:
					recorder_state["video_bytes"] = f.read()

				if not recorder_state["video_bytes"]:
					st.error("Recording stopped, but MP4 file is empty.")
				else:
					st.success(f"Recording stopped. Saved {frames_written} frames.")

				try:
					os.remove(output_path)
				except OSError:
					pass

				recorder_state["output_path"] = ""
				recorder_state["thread"] = None
				recorder_state["stop_event"] = None

		if not recorder_state["is_recording"] and recorder_state.get("video_bytes"):
			last_frame = recorder_state["status"].get("last_frame_rgb")
			if isinstance(last_frame, np.ndarray):
				st.image(last_frame, caption="Last recorded frame", channels="RGB")
			st.video(recorder_state["video_bytes"])
			st.download_button(
				label="Download MP4",
				data=recorder_state["video_bytes"],
				file_name="webcam_capture.mp4",
				mime="video/mp4",
			)

		st.subheader("Event Timeline")
		st.dataframe(events[:500], use_container_width=True)


if __name__ == "__main__":
	main()
