import os
import time
import streamlit as st
from faster_whisper import WhisperModel

# ------------------- Upload Helpers -------------------
def build_txt(full_text):
    return full_text

def build_srt(segments):
    srt_str = ""
    for i, seg in enumerate(segments, start=1):
        start_h, start_m = divmod(int(seg["start"]), 3600)
        start_m, start_s = divmod(start_m, 60)
        end_h, end_m = divmod(int(seg["end"]), 3600)
        end_m, end_s = divmod(end_m, 60)
        srt_str += f"{i}\n{start_h:02}:{start_m:02}:{start_s:02},000 --> {end_h:02}:{end_m:02}:{end_s:02},000\n{seg['text']}\n\n"
    return srt_str

# ------------------- Load Whisper -------------------
model = WhisperModel("small", device="cpu", compute_type="int8")

# ------------------- UI -------------------
st.title("🎤 Speech-to-Text App")

tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Live Mic"])

# ------------------- Tab 1: Upload -------------------
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

        try:
            os.remove(tmp_path)
        except:
            pass

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

# ------------------- Tab 2: Mic (from ZIP) -------------------
with tab2:
    st.subheader("🎙️ Live Mic Transcription")

    # --- from streamlit-stt-app-main (your zip) ---
    import queue
    import numpy as np
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.q = queue.Queue()

        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio = frame.to_ndarray()
            self.q.put(audio)
            return frame

    ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor:
        st.info("🎤 Speak into your mic...")
        if not ctx.audio_processor.q.empty():
            audio_chunk = ctx.audio_processor.q.get()

            # Process with Whisper
            tmp_path = f"mic_{int(time.time())}.wav"
            import soundfile as sf
            sf.write(tmp_path, audio_chunk, 16000)

            segments_gen, info = model.transcribe(tmp_path, beam_size=1)
            segments_list = list(segments_gen)
            text_out = " ".join([s.text for s in segments_list])

            st.write("**Transcript:**", text_out)
            try:
                os.remove(tmp_path)
            except:
                pass
