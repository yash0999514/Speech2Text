import os
import io
import time
import queue
import threading
from datetime import timedelta

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

# ---- Mic streaming imports ----
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import webrtcvad
import matplotlib.pyplot as plt

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Speech → Text (Upload & Live Mic)", page_icon="🎤", layout="wide")

st.markdown("""
<style>
:root {
  --pill:#10b981; --danger:#ef4444; --muted:#6b7280;
}
div[data-testid="stStatusWidget"] { display:none; }
.mic-pill { display:inline-flex; align-items:center; gap:.5rem; 
  padding:.5rem .9rem; border-radius:999px; font-weight:600; background: #0b1324; 
  border:1px solid #1f2937; }
.pulse { width:10px; height:10px; border-radius:50%; background: var(--danger);
  box-shadow: 0 0 0 0 rgba(239,68,68, .7); animation: pulse 1.5s infinite; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(239,68,68,.7); }
  70% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.subtitle {
  font-size: 1.15rem; line-height: 1.6; padding: .6rem .9rem; border-radius: .75rem;
  border:1px solid #1f2937; background: #0b1324; min-height: 3rem;
}
.badge { background: #111827; border:1px solid #1f2937; padding:.2rem .5rem; border-radius:.5rem; }
.codebox { border:1px dashed #334155; padding:.75rem; border-radius:.75rem; background:#0b1324; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utility: srt builder
# -----------------------------
def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    return str(td)[:-3].replace('.', ',')

def build_srt(segments):
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append(f"{i}")
        out.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out).encode("utf-8")

def build_txt(full_text: str):
    return full_text.strip().encode("utf-8")

# -----------------------------
# Whisper Model Loader
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_asr(model_size: str = "small", compute_type: str = "int8"):
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)

model = load_asr(
    model_size=st.sidebar.selectbox(
        "Model size (larger = better accuracy, slower)",
        ["tiny", "base", "small", "medium", "large-v2"], index=2
    ),
    compute_type=st.sidebar.selectbox(
        "Compute type",
        ["int8", "int8_float16", "float16"], index=0
    )
)

st.sidebar.caption("Tip: For CPU, use **small/int8**. For GPU, try **large-v2/float16**.")
st.title("🎤 Speech → Text (Upload & Live Subtitles)")
st.write("Two modes: **Upload an audio file** or **Speak via mic** with live subtitles and full transcript.")

# -----------------------------
# Tab 1: Upload transcription
# -----------------------------
tab1, tab2 = st.tabs(["📤 Upload audio", "🎙️ Live mic"])

with tab1:
    st.subheader("Upload MP3/WAV → Transcription")
    uploaded = st.file_uploader("Choose an MP3/WAV/M4A file", type=["mp3", "wav", "m4a", "aac"])
    colA, colB = st.columns([2,1])
    with colA:
        lang_hint = st.selectbox("Language (optional)", ["Auto-detect", "en", "hi"], index=0)
    with colB:
        beam = st.slider("Beam size", 1, 5, 1, help="Higher = a bit more accurate, slower")

    if uploaded:
        tmp_path = f"tmp_upload_{int(time.time())}_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Transcribing…"):
            segments_gen, info = model.transcribe(
                tmp_path,
                language=None if lang_hint == "Auto-detect" else lang_hint,
                beam_size=beam,
                vad_filter=True
            )

            seg_list = []
            full_text_chunks = []
            for seg in segments_gen:
                seg_list.append({
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                })
                full_text_chunks.append(seg.text)

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        st.success(f"Detected language: **{info.language}** | Probability: {info.language_probability:.2f}")
        st.markdown("**Transcript**")
        st.write("".join(full_text_chunks))

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download .txt", data=build_txt("".join(full_text_chunks)),
                               file_name="transcript.txt", mime="text/plain")
        with col2:
            st.download_button("⬇️ Download .srt (subtitles)", data=build_srt(seg_list),
                               file_name="subtitles.srt", mime="application/x-subrip")

        with st.expander("Show segments with timestamps"):
            st.dataframe([{
                "Start": format_timestamp(s["start"]),
                "End": format_timestamp(s["end"]),
                "Text": s["text"].strip()
            } for s in seg_list], use_container_width=True)

# -----------------------------
# Tab 2: Live mic
# -----------------------------
if "live_segments" not in st.session_state:
    st.session_state.live_segments = []
if "live_text" not in st.session_state:
    st.session_state.live_text = ""
if "mic_active" not in st.session_state:
    st.session_state.mic_active = False

chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1024)

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.sample_rate = 48000
        self.channels = 1
        self.vad = webrtcvad.Vad(3)
        self.frame_ms = 30
        self.frame_bytes = int(self.sample_rate * 2 * self.frame_ms / 1000)
        self.buffer = bytearray()
        self.speech_started = False
        self.silence_ms = 0
        self.max_silence_after_speech_ms = 700

    def recv_audio(self, frames: list) -> bytes:
        return b""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        frame = frame.to_ndarray()
        if frame.ndim > 1:
            frame = frame.mean(axis=0)
        if frame.dtype != np.int16:
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame, -1.0, 1.0)
                frame = (frame * 32767).astype(np.int16)
            else:
                frame = frame.astype(np.int16)

        pcm_bytes = frame.tobytes()
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
                        chunk_pcm = bytes(self.current_utt)
                        try:
                            chunk_queue.put_nowait(chunk_pcm)
                        except queue.Full:
                            _ = chunk_queue.get_nowait()  # drop oldest
                            chunk_queue.put_nowait(chunk_pcm)
                        self.speech_started = False
                        self.silence_ms = 0

        return frame

def asr_worker(stop_event: threading.Event, waveform_placeholder):
    def resample48k_to16k_int16(pcm_bytes: bytes) -> np.ndarray:
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(x) == 0:
            return x
        orig_sr, target_sr = 48000, 16000
        ratio = target_sr / orig_sr
        new_len = int(len(x) * ratio)
        idx = np.linspace(0, len(x) - 1, num=new_len)
        y = np.interp(idx, np.arange(len(x)), x)
        return y.astype(np.float32)

    while not stop_event.is_set():
        try:
            pcm_chunk = chunk_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        audio16k = resample48k_to16k_int16(pcm_chunk)
        if audio16k.size == 0:
            continue

        # Live waveform
        plt.figure(figsize=(10,2))
        plt.plot(audio16k, color="lime")
        plt.title("Live Mic Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        waveform_placeholder.pyplot(plt)
        plt.close()

        segments, info = model.transcribe(
            audio16k,
            sampling_rate=16000,
            language=None,
            beam_size=1,
            vad_filter=False
        )

        partial_text = []
        new_segments = []
        for seg in segments:
            partial_text.append(seg.text)
            new_segments.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })

        if partial_text:
            st.session_state.live_text = " ".join([t.strip() for t in partial_text])
            st.session_state.live_segments.extend(new_segments)

with tab2:
    st.subheader("Speak into the mic → Live subtitles + transcript")
    waveform_placeholder = st.empty()  # placeholder for waveform

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            f"""<div class="mic-pill">
            <div class="pulse"></div>
            <span>{'Listening…' if st.session_state.mic_active else 'Mic is idle'}</span>
            </div>""",
            unsafe_allow_html=True
        )
        st.caption("Allow mic permissions in your browser. Speak normally; pauses create subtitle chunks.")
    with right:
        start = st.button("▶️ Start mic")
        stop = st.button("⏹️ Stop mic")

    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=MicAudioProcessor if start else None,
        async_processing=True,
    )

    if start and ctx.state.playing:
        st.session_state.mic_active = True
        st.session_state.live_text = ""
        st.session_state.live_segments = []
        stop_event = threading.Event()
        st.session_state["_stop_event"] = stop_event

        worker = threading.Thread(target=asr_worker, args=(stop_event, waveform_placeholder), daemon=True)
        worker.start()

    if stop:
        st.session_state.mic_active = False
        if "_stop_event" in st.session_state:
            st.session_state["_stop_event"].set()

    st.markdown("### Live subtitles")
    st.markdown(f"""<div class="subtitle">{st.session_state.live_text}</div>""", unsafe_allow_html=True)

    full_text_live = " ".join([s["text"].strip() for s in st.session_state.live_segments])
    st.text_area("Transcript so far", value=full_text_live, height=200)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download mic transcript (.txt)",
                           data=build_txt(full_text_live),
                           file_name="mic_transcript.txt",
                           mime="text/plain",
                           disabled=(len(full_text_live.strip()) == 0))
    with c2:
        st.download_button("⬇️ Download mic subtitles (.srt)",
                           data=build_srt([
                               {"id": i, "start": i*2.0, "end": i*2.0+2.0, "text": t}
                               for i, t in enumerate(full_text_live.split(". "))
                           ]) if full_text_live.strip() else b"",
                           file_name="mic_subtitles.srt",
                           mime="application/x-subrip",
                           disabled=(len(full_text_live.strip()) == 0))

st.markdown("---")
st.caption("Built with Streamlit + WebRTC + faster-whisper. Supports auto language detection (English).")
