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

st.sidebar.header("Models")
model = load_asr(
    model_size=st.sidebar.selectbox("Whisper size (Upload)", ["tiny","base","small","medium","large-v2"], index=2),
    compute_type=st.sidebar.selectbox("Whisper compute", ["int8","int8_float16","float16"], index=0)
)

# -----------------------------
# Hardcoded AssemblyAI API Key
# -----------------------------
ASSEMBLYAI_API_KEY = "4186eb45698741cd8d5c39cfb9f6913c"

st.title("🎤 Speech → Text (Upload & Live Subtitles)")
st.write("Two modes: **Upload an audio file** (Whisper) or **Speak via mic** (AssemblyAI) with live subtitles and full transcript.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📤 Upload audio (Whisper)", "🎙️ Live mic (AssemblyAI)"])

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
# Live Mic with AssemblyAI
# -----------------------------
if "live_segments" not in st.session_state:
    st.session_state.live_segments = []
if "live_text" not in st.session_state:
    st.session_state.live_text = ""
if "mic_active" not in st.session_state:
    st.session_state.mic_active = False

chunk_queue = queue.Queue(maxsize=1024)

class MicAudioProcessor(AudioProcessorBase):
    """Send PCM frames to queue for AssemblyAI."""
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
        try:
            chunk_queue.put_nowait(arr.tobytes())
        except queue.Full:
            pass
        return frame

async def assemblyai_worker(stop_event, subtitle_placeholder):
    url = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
    async with websockets.connect(
        url,
        extra_headers={"Authorization": ASSEMBLYAI_API_KEY},
        ping_interval=5,
        ping_timeout=20,
    ) as ws:
        async def sender():
            while not stop_event.is_set():
                try:
                    data = chunk_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                # downsample 48k -> 16k if needed
                audio48 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio16 = np.interp(
                    np.linspace(0, len(audio48)-1, int(len(audio48) * 16000 / 48000)),
                    np.arange(len(audio48)),
                    audio48
                )
                audio16_bytes = (np.clip(audio16, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                await ws.send(json.dumps({"audio_data": base64.b64encode(audio16_bytes).decode("utf-8")}))
                await asyncio.sleep(0.01)
            await ws.send(json.dumps({"terminate_session": True}))

        async def receiver():
            partial_last = ""
            async for msg in ws:
                res = json.loads(msg)
                if "text" in res:
                    text = res["text"].strip()
                    if res.get("message_type") == "partial":
                        if text and text != partial_last:
                            temp_full = (st.session_state.live_text + " " + text).strip()
                            subtitle_placeholder.markdown(
                                f"""<div class="subtitle">{temp_full}</div>""",
                                unsafe_allow_html=True
                            )
                            partial_last = text
                    elif res.get("message_type") == "final":
                        if text:
                            st.session_state.live_segments.append({
                                "start": 0.0, "end": 0.0, "text": text
                            })
                            st.session_state.live_text += (" " if st.session_state.live_text else "") + text
                            subtitle_placeholder.markdown(
                                f"""<div class="subtitle">{st.session_state.live_text}</div>""",
                                unsafe_allow_html=True
                            )

        await asyncio.gather(sender(), receiver())

with tab2:
    st.subheader("Speak into the mic → Live subtitles + transcript (AssemblyAI)")
    subtitle_placeholder = st.empty()  # Live subtitle box

    # Mic status pill
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(f"""<div class="mic-pill">
        <div class="pulse"></div>
        <span>{'Listening…' if st.session_state.mic_active else 'Mic is idle'}</span>
        </div>""", unsafe_allow_html=True)
    with right:
        start = st.button("▶️ Start mic")
        stop = st.button("⏹️ Stop mic")

    # WebRTC setup
    webrtc_ctx = webrtc_streamer(
        key="live-mic-assemblyai",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=MicAudioProcessor,
        async_processing=True,
    )

    # Start mic logic
    if start and webrtc_ctx and webrtc_ctx.state.playing:
        st.session_state.mic_active = True
        st.session_state.live_text = ""
        st.session_state.live_segments = []
        stop_event = threading.Event()
        st.session_state["_stop_event"] = stop_event

        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(assemblyai_worker(stop_event, subtitle_placeholder))

        loop = asyncio.new_event_loop()
        worker = threading.Thread(target=run_loop, args=(loop,), daemon=True)
        worker.start()

    # Stop mic logic
    if stop:
        if "_stop_event" in st.session_state:
            st.session_state["_stop_event"].set()
            st.session_state.mic_active = False
            st.warning("🛑 Mic stopped.")
        else:
            st.warning("No mic session running.")

    # -------------------------------
    # Live Transcript section (always visible while mic runs)
    # -------------------------------
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
st.caption("Upload: Whisper (faster-whisper) • Live mic: AssemblyAI (cloud) • Built with Streamlit + WebRTC")


                           

st.markdown("---")
st.caption("Upload: Whisper (faster-whisper) • Live mic: AssemblyAI (cloud) • Built with Streamlit + WebRTC")





