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
import webrtcvad
import matplotlib.pyplot as plt
import vosk  # <-- NEW

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
# Load Whisper (Upload tab)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_asr(model_size: str = "small", compute_type: str = "int8"):
    repo_id = f"guillaumekln/faster-whisper-{model_size}"
    snapshot_download(repo_id, local_dir="models", local_dir_use_symlinks=False)
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)

# -----------------------------
# Load Vosk (Live mic)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_vosk(vosk_model_path: str):
    model = vosk.Model(vosk_model_path)
    return model

st.sidebar.header("Models")
model = load_asr(
    model_size=st.sidebar.selectbox("Whisper size (Upload)", ["tiny","base","small","medium","large-v2"], index=2),
    compute_type=st.sidebar.selectbox("Whisper compute", ["int8","int8_float16","float16"], index=0)
)
vosk_path = st.sidebar.text_input("Vosk model path (Live mic)", "models/vosk-model-small-en-us-0.15")
st.sidebar.caption("Tip: Download a Vosk model and place it under ./models/")

vosk_model = None
if os.path.isdir(vosk_path):
    try:
        vosk_model = load_vosk(vosk_path)
    except Exception as e:
        st.sidebar.error(f"Vosk load error: {e}")
else:
    st.sidebar.warning("Vosk model folder not found. Live mic will be disabled until you set a valid path.")

st.title("🎤 Speech → Text (Upload & Live Subtitles)")
st.write("Two modes: **Upload an audio file** (Whisper) or **Speak via mic** (Vosk) with live subtitles and full transcript.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📤 Upload audio (Whisper)", "🎙️ Live mic (Vosk)"])

# -----------------------------
# Tab 1: Upload Audio (Whisper)
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
        except Exception: pass

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
# Tab 2: Live Mic (Vosk)
# -----------------------------
# Session state
if "live_segments" not in st.session_state:
    st.session_state.live_segments = []   # list of {start,end,text}
if "live_text" not in st.session_state:
    st.session_state.live_text = ""
if "mic_active" not in st.session_state:
    st.session_state.mic_active = False

chunk_queue = queue.Queue(maxsize=1024)

class MicAudioProcessor(AudioProcessorBase):
    """Feeds PCM frames to a queue. VAD keeps chunks coherent."""
    def __init__(self) -> None:
        self.sample_rate = 48000
        self.channels = 1
        self.vad = webrtcvad.Vad(3)
        self.frame_ms = 30
        self.frame_bytes = int(self.sample_rate * 2 * self.frame_ms / 1000)
        self.buffer = bytearray()
        self.speech_started = False
        self.silence_ms = 0
        self.max_silence_after_speech_ms = 500  # shorter = snappier splits

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        if arr.dtype != np.int16:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, -1.0, 1.0)
                arr = (arr * 32767).astype(np.int16)
            else:
                arr = arr.astype(np.int16)

        pcm_bytes = arr.tobytes()
        self.buffer.extend(pcm_bytes)

        while len(self.buffer) >= self.frame_bytes:
            chunk = self.buffer[:self.frame_bytes]
            self.buffer = self.buffer[self.frame_bytes:]
            is_speech = self.vad.is_speech(chunk, self.sample_rate)

            if is_speech:
                self.silence_ms = 0
                if not self.speech_started:
                    self.speech_started = True
                    self.current_utt = bytearray()
                self.current_utt.extend(chunk)
            else:
                if self.speech_started:
                    self.silence_ms += self.frame_ms
                    self.current_utt.extend(chunk)
                    if self.silence_ms >= self.max_silence_after_speech_ms:
                        try:
                            chunk_queue.put_nowait(bytes(self.current_utt))
                        except queue.Full:
                            _ = chunk_queue.get_nowait()
                            chunk_queue.put_nowait(bytes(self.current_utt))
                        self.speech_started = False
                        self.silence_ms = 0

        # Forward audio or mute
        return av.AudioFrame.from_ndarray(arr.reshape(1, -1), layout="mono")

with tab2:
    st.subheader("Speak into the mic → Live subtitles + transcript (Vosk)")
    waveform_placeholder = st.empty()
    subtitle_placeholder = st.empty()

    left, right = st.columns([1.3,1])
    with left:
        st.markdown(f"""<div class="mic-pill">
        <div class="pulse"></div>
        <span>{'Listening…' if st.session_state.mic_active else 'Mic is idle'}</span>
        </div>""", unsafe_allow_html=True)
        st.caption("Allow mic permissions in your browser. Speak normally; pauses create subtitle chunks.")
    with right:
        start = st.button("▶️ Start mic", disabled=(vosk_model is None))
        stop = st.button("⏹️ Stop mic")

    # WebRTC component to pull mic audio
    webrtc_ctx = webrtc_streamer(
        key="live-mic-vosk",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=MicAudioProcessor,
        async_processing=True,
    )

    if start and vosk_model and webrtc_ctx and webrtc_ctx.state.playing:
        st.session_state.mic_active = True
        st.session_state.live_text = ""
        st.session_state.live_segments = []

        stop_event = threading.Event()
        st.session_state["_stop_event"] = stop_event

        # Prepare Vosk recognizer (16k mono)
        SAMPLE_RATE_TARGET = 16000
        rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE_TARGET)
        rec.SetWords(True)  # include word-level timestamps for better SRT timing

        def live_worker():
            partial_last = ""
            while not stop_event.is_set():
                try:
                    chunk = chunk_queue.get(timeout=0.2)
                except queue.Empty:
                    if not webrtc_ctx.state.playing:
                        break
                    continue

                # Resample 48k -> 16k
                audio48 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                audio16 = np.interp(
                    np.linspace(0, len(audio48)-1, int(len(audio48) * SAMPLE_RATE_TARGET / 48000)),
                    np.arange(len(audio48)),
                    audio48
                )
                # back to int16 bytes for Vosk
                audio16_bytes = (np.clip(audio16, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

                # Optional waveform
                plt.figure(figsize=(10, 2))
                plt.plot(audio16)
                waveform_placeholder.pyplot(plt)
                plt.close()

                # Feed Vosk
                if rec.AcceptWaveform(audio16_bytes):
                    # Final segment
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    words = result.get("result", [])
                    if text:
                        # derive segment start/end from words if available
                        if words:
                            seg_start = float(words[0].get("start", 0.0))
                            seg_end = float(words[-1].get("end", seg_start + 2.0))
                        else:
                            # fallback: append at end with rough duration
                            seg_start = st.session_state.live_segments[-1]["end"] if st.session_state.live_segments else 0.0
                            seg_end = seg_start + max(2.0, len(text.split()) * 0.35)

                        st.session_state.live_segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": text
                        })
                        st.session_state.live_text += (" " if st.session_state.live_text else "") + text
                        subtitle_placeholder.markdown(
                            f"""<div class="subtitle">{st.session_state.live_text}</div>""",
                            unsafe_allow_html=True
                        )
                        partial_last = ""
                else:
                    # Partial (live subtitle feel)
                    partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if partial and partial != partial_last:
                        temp_full = (st.session_state.live_text + " " + partial).strip()
                        subtitle_placeholder.markdown(
                            f"""<div class="subtitle">{temp_full}</div>""",
                            unsafe_allow_html=True
                        )
                        partial_last = partial

        worker = threading.Thread(target=live_worker, daemon=True)
        worker.start()

    if stop:
        st.session_state.mic_active = False
        if "_stop_event" in st.session_state:
            st.session_state["_stop_event"].set()
        if webrtc_ctx:
            webrtc_ctx.stop()
        st.warning("🛑 Mic stopped.")

    # Running transcript + downloads
    st.markdown("### Running transcript")
    full_text_live = st.session_state.live_text
    st.text_area("Transcript so far", value=full_text_live, height=200)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download mic transcript (.txt)",
                           data=build_txt(full_text_live),
                           file_name="mic_transcript.txt", mime="text/plain",
                           disabled=(len(full_text_live.strip())==0))
    with c2:
        st.download_button("⬇️ Download mic subtitles (.srt)",
                           data=build_srt(st.session_state.live_segments) if full_text_live.strip() else b"",
                           file_name="mic_subtitles.srt",
                           mime="application/x-subrip",
                           disabled=(len(full_text_live.strip())==0))

st.markdown("---")
st.caption("Upload: Whisper (faster-whisper) • Live mic: Vosk (offline) • Built with Streamlit + WebRTC")
