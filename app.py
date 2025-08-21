import os
import time
import queue
import threading
import json
from datetime import timedelta

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import websockets
import base64
import asyncio

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Speech → Text (Upload & Live Mic)", page_icon="🎤", layout="wide")

st.markdown("""
<style>
:root { --pill:#10b981; --danger:#ef4444; --muted:#6b7280; }
div[data-testid="stStatusWidget"] { display:none; }
.mic-pill { display:inline-flex; align-items:center; gap:.5rem; 
  padding:.5rem .9rem; border-radius:999px; font-weight:600; background:#0b1324; 
  border:1px solid #1f2937; }
.pulse { width:10px; height:10px; border-radius:50%; background: var(--danger);
  box-shadow: 0 0 0 0 rgba(239,68,68, .7); animation: pulse 1.5s infinite; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(239,68,68,.7); }
  70% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.subtitle { font-size:1.15rem; line-height:1.6; padding:.6rem .9rem; border-radius:.75rem;
  border:1px solid #1f2937; background:#0b1324; min-height:3rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# API Key
# -----------------------------
if "ASSEMBLYAI_API_KEY" not in st.session_state:
    st.session_state.ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

if not st.session_state.ASSEMBLYAI_API_KEY:
    st.warning("❌ AssemblyAI API Key not found! Please enter it below.")
    st.session_state.ASSEMBLYAI_API_KEY = st.text_input("Enter AssemblyAI API Key", type="password")
    if not st.session_state.ASSEMBLYAI_API_KEY:
        st.stop()  # Stop execution until user provides a key

# -----------------------------
# Utilities
# -----------------------------
def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    return str(td)[:-3].replace('.', ',')

def build_srt(segments):
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append(f"{i}")
        start = seg.get("start", 0.0)
        end = seg.get("end", start + 2.0)
        out.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        out.append(seg.get("text","").strip())
        out.append("")
    return "\n".join(out).encode("utf-8")

def build_txt(full_text: str):
    return full_text.strip().encode("utf-8")

# -----------------------------
# Load Whisper Model
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_asr(model_size: str = "small", compute_type: str = "int8"):
    repo_id = f"guillaumekln/faster-whisper-{model_size}"
    snapshot_download(repo_id, local_dir="models", local_dir_use_symlinks=False)
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)

st.sidebar.header("Models")
model = load_asr(
    model_size=st.sidebar.selectbox("Whisper size (Upload)", ["tiny","base","small","medium","large-v2"], index=2),
    compute_type=st.sidebar.selectbox("Whisper compute", ["int8","int8_float16","float16"], index=0)
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📤 Upload audio (Whisper)", "🎙️ Live mic (AssemblyAI)"])

# -----------------------------
# Tab 1: Upload Audio
# -----------------------------
with tab1:
    st.subheader("Upload MP3/WAV → Transcription")
    uploaded = st.file_uploader("Choose an MP3/WAV/M4A/AAC file", type=["mp3","wav","m4a","aac"])
    colA, colB = st.columns([2,1])
    with colA:
        lang_hint = st.selectbox("Language (optional)", ["Auto-detect","en","hi"], index=0)
    with colB:
        beam = st.slider("Beam size", 1, 5, 1, help="Higher = more accurate, slower")

    if uploaded:
        tmp_path = f"tmp_upload_{int(time.time())}_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Transcribing…"):
            segments_gen, info = model.transcribe(
                tmp_path,
                language=None if lang_hint=="Auto-detect" else lang_hint,
                beam_size=beam,
                vad_filter=True
            )
            segments_list = list(segments_gen)
            seg_list = [{"id": s.id, "start": s.start, "end": s.end, "text": s.text} for s in segments_list]
            full_text = " ".join([s.text for s in segments_list])

        try: os.remove(tmp_path)
        except: pass

        st.success(f"Detected language: **{info.language}** | Probability: {info.language_probability:.2f}")
        st.markdown("**Transcript**")
        st.write(full_text)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download .txt", data=build_txt(full_text),
                               file_name="transcript.txt", mime="text/plain")
        with col2:
            st.download_button("⬇️ Download .srt", data=build_srt(seg_list),
                               file_name="subtitles.srt", mime="application/x-subrip")

# -----------------------------
# Tab 2: Live Mic
# -----------------------------
if "live_segments" not in st.session_state:
    st.session_state.live_segments = []
if "live_text" not in st.session_state:
    st.session_state.live_text = ""
if "mic_active" not in st.session_state:
    st.session_state.mic_active = False

with tab2:
    st.subheader("Speak into the mic → Live subtitles + transcript")
    subtitle_placeholder = st.empty()

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(f"""<div class="mic-pill">
        <div class="pulse"></div>
        <span>{'Listening…' if st.session_state.mic_active else 'Mic is idle'}</span>
        </div>""", unsafe_allow_html=True)
    with right:
        start = st.button("▶️ Start mic")
        stop = st.button("⏹️ Stop mic")

    # Start mic logic
    if start and not st.session_state.mic_active:
        st.session_state.mic_active = True
        st.session_state.live_text = ""
        st.session_state.live_segments = []
        st.session_state["_stop_event"] = threading.Event()

        def run_realtime():
            import pyaudio, websocket, time

            FRAMES_PER_BUFFER = 800
            SAMPLE_RATE = 16000
            CHANNELS = 1
            FORMAT = pyaudio.paInt16

            audio = pyaudio.PyAudio()
            stream = audio.open(
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
                channels=CHANNELS,
                format=FORMAT,
                rate=SAMPLE_RATE,
            )

            recorded_frames = []

            def on_open(ws):
                def send_audio():
                    while not st.session_state["_stop_event"].is_set():
                        data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                        recorded_frames.append(data)
                        ws.send(data, websocket.ABNF.OPCODE_BINARY)
                threading.Thread(target=send_audio, daemon=True).start()

            def on_message(ws, message):
                try:
                    res = json.loads(message)
                    if res.get("type") == "Turn":
                        text = res.get("transcript", "").strip()
                        if text:
                            st.session_state.live_segments.append({"start": 0.0, "end": 0.0, "text": text})
                            st.session_state.live_text += (" " if st.session_state.live_text else "") + text
                            subtitle_placeholder.markdown(
                                f"""<div class="subtitle">{st.session_state.live_text}</div>""",
                                unsafe_allow_html=True,
                            )
                except:
                    pass

            def on_close(ws, *_):
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                audio.terminate()

            ws_app = websocket.WebSocketApp(
                "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&format_turns=True",
                header={"Authorization": st.session_state.ASSEMBLYAI_API_KEY},
                on_open=on_open,
                on_message=on_message,
                on_close=on_close,
            )

            ws_thread = threading.Thread(target=ws_app.run_forever, daemon=True)
            ws_thread.start()

            while not st.session_state["_stop_event"].is_set():
                time.sleep(0.1)

            if ws_app and ws_app.sock and ws_app.sock.connected:
                ws_app.close()

        threading.Thread(target=run_realtime, daemon=True).start()

    if stop and st.session_state.mic_active:
        st.session_state["_stop_event"].set()
        st.session_state.mic_active = False
        st.warning("🛑 Mic stopped.")

    # Transcript display
    st.markdown("### 📜 Running transcript")
    st.text_area(
        "Transcript so far",
        value=st.session_state.live_text,
        height=200,
    )

    # Download buttons
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download mic transcript (.txt)",
            data=build_txt(st.session_state.live_text),
            file_name="mic_transcript.txt",
            mime="text/plain",
            disabled=(len(st.session_state.live_text.strip()) == 0),
        )
    with c2:
        st.download_button(
            "⬇️ Download mic subtitles (.srt)",
            data=build_srt(st.session_state.live_segments) if st.session_state.live_text.strip() else b"",
            file_name="mic_subtitles.srt",
            mime="application/x-subrip",
            disabled=(len(st.session_state.live_text.strip()) == 0),
        )

st.markdown("---")
st.caption("Upload: Whisper • Live mic: AssemblyAI • Built with Streamlit + WebRTC")
