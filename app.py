import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import whisper
import threading
import queue
import tempfile
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import resampy
from scipy.io import wavfile
from pydub import AudioSegment

# Tell pydub where ffmpeg/ffprobe are located
AudioSegment.converter = "ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "ffmpeg/bin/ffprobe.exe"


# -----------------------
# Config
# -----------------------
SR = 16000  # Whisper sampling rate
CHUNK_SECONDS = 2.0
MODEL_SIZE = "base"
LIVE_SUB_FILE = "live_sub.txt"

# -----------------------
# Load Whisper model
# -----------------------
@st.cache_resource
def load_model(size=MODEL_SIZE):
    return whisper.load_model(size)

model = load_model(MODEL_SIZE)

# -----------------------
# Helper functions
# -----------------------
def read_mp3_to_numpy(path, target_sr=SR):
    audio = AudioSegment.from_file(path, format="mp3")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)

    samples = samples / (2**15)  # normalize int16 → float32
    sr = audio.frame_rate

    if sr != target_sr:
        samples = resampy.resample(samples, sr, target_sr)
        sr = target_sr

    return samples, sr

def plot_waveform(samples, sr):
    t = np.arange(len(samples)) / sr
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t, samples, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    plt.tight_layout()
    return fig

def transcribe_mp3(path):
    return model.transcribe(path, language="en", task="transcribe", fp16=False).get("text", "").strip()

# -----------------------
# LiveTranscriber
# -----------------------
class LiveTranscriber:
    def __init__(self, sr=SR, chunk_s=CHUNK_SECONDS):
        self.sr = sr
        self.chunk_s = chunk_s
        self.recording = False
        self.ready_chunks = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = None

    def start(self, webrtc_ctx):
        if self.recording:
            return
        open(LIVE_SUB_FILE, "w", encoding="utf-8").close()
        self.recording = True
        self._stop_event.clear()
        self.worker_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.worker_thread.start()
        threading.Thread(target=self._collector_loop, args=(webrtc_ctx,), daemon=True).start()

    def stop(self):
        self._stop_event.set()
        self.recording = False

    def _collector_loop(self, webrtc_ctx):
        needed = int(self.chunk_s * self.sr)
        buf = np.zeros((0, 1), dtype=np.float32)
        while not self._stop_event.is_set():
            try:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=1.0)
            except Exception:
                time.sleep(0.1)
                continue
            for frame in frames:
                arr = frame.to_ndarray().astype(np.float32)
                if arr.ndim > 1:
                    arr = arr[:, 0:1]
                else:
                    arr = arr.reshape(-1, 1)
                frame_rate = frame.sample_rate
                if frame_rate != self.sr:
                    arr = resampy.resample(arr.flatten(), frame_rate, self.sr).astype(np.float32).reshape(-1, 1)
                buf = np.concatenate([buf, arr], axis=0)
                while len(buf) >= needed:
                    chunk = buf[:needed]
                    buf = buf[needed:]
                    self.ready_chunks.put(chunk.copy())
            time.sleep(0.01)

    def _transcribe_loop(self):
        while not self._stop_event.is_set() or not self.ready_chunks.empty():
            try:
                chunk = self.ready_chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            arr_int16 = (np.clip(chunk.flatten(), -1.0, 1.0) * 32767).astype(np.int16)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wavfile.write(tmp.name, self.sr, arr_int16)
            try:
                text = model.transcribe(tmp.name, language="en", task="transcribe", fp16=False).get("text", "").strip()
                if text:
                    with open(LIVE_SUB_FILE, "a", encoding="utf-8") as f:
                        f.write(text + " ")
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

# -----------------------
# Streamlit UI
# -----------------------
if "live_transcriber" not in st.session_state:
    st.session_state.live_transcriber = LiveTranscriber()

live_trans = st.session_state.live_transcriber

st.title("🎤 Live Speech-to-Text + MP3 Upload")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Live Microphone")
    st.markdown("Allow microphone in browser and click Start.")

    webrtc_ctx = webrtc_streamer(
        key="live_mic",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        if not live_trans.recording:
            if st.button("▶️ Start Live Subtitles"):
                live_trans.start(webrtc_ctx)
                st.experimental_rerun()
        else:
            if st.button("⏹ Stop Live Subtitles"):
                live_trans.stop()
                st.experimental_rerun()
    else:
        st.info("Allow microphone in browser, then press Start.")

    st.slider("Chunk seconds", 1.0, 5.0, CHUNK_SECONDS, 0.5, key="chunk_slider")
    live_trans.chunk_s = st.session_state.chunk_slider

    st.write("---")
    st.header("Upload MP3 Audio File")
    uploaded = st.file_uploader("Upload MP3", type=["mp3"])
    if uploaded:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmpf.write(uploaded.read())
        tmpf.close()
        st.audio(tmpf.name, format="audio/mp3")
        samples, sr = read_mp3_to_numpy(tmpf.name)
        st.pyplot(plot_waveform(samples, sr))
        text = transcribe_mp3(tmpf.name)
        st.subheader("File Transcription")
        st.write(text)
        try:
            os.unlink(tmpf.name)
        except Exception:
            pass

with col2:
    st.header("Live Subtitles")
    refresh = st.empty()
    try:
        with open(LIVE_SUB_FILE, "r", encoding="utf-8") as f:
            live_text = f.read().strip()
    except Exception:
        live_text = ""
    if live_text:
        st.markdown(f"**{live_text}**")
    else:
        st.info("No live subtitles yet. Start microphone or upload MP3.")

st.write("---")
st.write(
    """
    - Real-time live subtitles from your microphone.
    - Upload MP3 files for transcription.
    - Waveform visualizes audio amplitude.
    - Whisper model runs on server; chunking controls update speed.
    """
)

