import streamlit as st
import json
import requests
import time

# -------------------------------
# 🌐 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="SignSpeak AI 👋", layout="wide", page_icon="🤟")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0a0a0f; color: #e8e8f0; }
    .block-container { padding: 2rem 3rem; }
    h1 { color: #a78bfa !important; font-size: 2.4rem !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border: none; border-radius: 12px;
        padding: 0.6rem 1.6rem; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(124,58,237,0.4); }
    .stCodeBlock, code { background: #1a1a2e !important; border-radius: 10px; }
    .step-box {
        background: #12121f;
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin: 0.6rem 0;
    }
    .status-success { color: #34d399; font-weight: 600; }
    .status-info { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

st.title("🤟 SignSpeak AI")
st.caption("🎧 Voice → 🧠 ISL Gloss → 🎬 AI Sign Language Video")

# -------------------------------
# 🔑 SESSION STATE
# -------------------------------
for key in ["transcription", "isl_data", "video_url", "video_status", "prediction_id"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# 🔑 API KEYS
# -------------------------------
groq_key = st.secrets.get("GROQ_API_KEY", "")
replicate_token = st.secrets.get("REPLICATE_API_TOKEN", "")

with st.sidebar:
    st.header("🔑 Configuration")
    if not groq_key:
        groq_key = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    if not replicate_token:
        replicate_token = st.text_input("Replicate Token", type="password", help="Get free token at replicate.com")

    st.markdown("---")
    st.markdown("**Model Used:**")
    st.markdown("- 🎙️ Whisper Large v3 (Groq)")
    st.markdown("- 🧠 LLaMA 3.3 70B (Groq)")
    st.markdown("- 🎬 minimax/video-01 (Replicate)")

    st.markdown("---")
    if st.button("🔄 Reset All"):
        for key in ["transcription", "isl_data", "video_url", "video_status", "prediction_id"]:
            st.session_state[key] = None
        st.rerun()


# -------------------------------
# 🎤 TRANSCRIBE AUDIO
# -------------------------------
def transcribe_audio(audio_bytes):
    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {groq_key}"}
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
            "model": (None, "whisper-large-v3"),
            "language": (None, "en"),
            "response_format": (None, "json")
        }
        res = requests.post(url, headers=headers, files=files, timeout=30)
        res.raise_for_status()
        return res.json().get("text", "")
    except requests.exceptions.HTTPError as e:
        st.error(f"Groq Transcription HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        return None


# -------------------------------
# 🧠 ISL TRANSLATION
# -------------------------------
def get_isl_translation(text):
    try:
        system_prompt = (
            "You are an expert Indian Sign Language (ISL) linguist. "
            "Convert the given English sentence into ISL. "
            "ISL follows Subject-Object-Verb order and drops articles/prepositions. "
            "Return ONLY a valid JSON object with these exact keys:\n"
            "{\n"
            '  "gloss": "ISL gloss words in correct order",\n'
            '  "video_prompt": "A short cinematic description of a person signing each word: [gloss]. '
            'Show clear hand shapes, front-facing view, neutral background, professional lighting, '
            'realistic human, 5-8 seconds"\n'
            "}"
        )
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        res.raise_for_status()
        return json.loads(res.json()["choices"][0]["message"]["content"])
    except requests.exceptions.HTTPError as e:
        st.error(f"Groq ISL HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"ISL Translation Error: {e}")
        return None


# -------------------------------
# 🎬 START VIDEO GENERATION (Async)
# -------------------------------
def start_video_generation(prompt):
    """
    Uses Replicate's minimax/video-01 model (free tier available).
    Returns prediction_id for polling.
    """
    try:
        url = "https://api.replicate.com/v1/models/minimax/video-01/predictions"
        headers = {
            "Authorization": f"Token {replicate_token}",
            "Content-Type": "application/json",
            "Prefer": "respond-async"
        }
        data = {
            "input": {
                "prompt": prompt,
                "prompt_optimizer": True
            }
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        res.raise_for_status()
        prediction = res.json()
        return prediction.get("id"), prediction.get("status")
    except requests.exceptions.HTTPError as e:
        # Fallback to wan-i2v or stable-video-diffusion
        st.warning(f"minimax/video-01 unavailable ({e.response.status_code}), trying fallback model...")
        return start_video_fallback(prompt)
    except Exception as e:
        st.error(f"Video Generation Error: {e}")
        return None, None


def start_video_fallback(prompt):
    """
    Fallback: Uses lucataco/animate-diff (free, widely available on Replicate)
    """
    try:
        url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {replicate_token}",
            "Content-Type": "application/json"
        }
        data = {
            "version": "beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",  # lucataco/animate-diff
            "input": {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted hands, extra fingers",
                "num_frames": 16,
                "num_inference_steps": 25,
                "guidance_scale": 7.5
            }
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        res.raise_for_status()
        prediction = res.json()
        return prediction.get("id"), prediction.get("status")
    except Exception as e:
        st.error(f"Fallback Video Error: {e}")
        return None, None


# -------------------------------
# 🔄 POLL VIDEO STATUS
# -------------------------------
def poll_video_status(prediction_id):
    """Poll Replicate prediction status"""
    try:
        # Try versioned predictions first
        url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        headers = {"Authorization": f"Token {replicate_token}"}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        prediction = res.json()
        status = prediction.get("status")
        output = prediction.get("output")
        error = prediction.get("error")
        
        video_url = None
        if output:
            if isinstance(output, list):
                video_url = output[0]
            elif isinstance(output, str):
                video_url = output
                
        return status, video_url, error
    except Exception as e:
        return "error", None, str(e)


# -------------------------------
# 🎤 AUDIO INPUT SECTION
# -------------------------------
st.markdown("---")
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("Step 1 — 🎙️ Record Your Voice")
    audio = st.audio_input("Click the mic icon and speak clearly in English")

with col_info:
    st.markdown("""
    <div class="step-box">
    <b>📌 How it works:</b><br><br>
    1️⃣ Record your voice<br>
    2️⃣ AI transcribes speech<br>
    3️⃣ Translated to ISL gloss<br>
    4️⃣ Video generated showing signing
    </div>
    """, unsafe_allow_html=True)

# Transcription trigger
if audio and not st.session_state.transcription:
    if not groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("🎧 Transcribing your audio with Whisper..."):
            text = transcribe_audio(audio.getvalue())
        if text:
            st.session_state.transcription = text
            with st.spinner("🧠 Translating to Indian Sign Language..."):
                st.session_state.isl_data = get_isl_translation(text)
            st.rerun()

# -------------------------------
# 📊 OUTPUT SECTION
# -------------------------------
if st.session_state.transcription:
    st.markdown("---")
    st.subheader("Step 2 — 📝 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🗣️ Transcribed Speech:**")
        st.success(st.session_state.transcription)

        if st.session_state.isl_data:
            st.markdown("**📘 ISL Gloss (sign order):**")
            st.code(st.session_state.isl_data.get("gloss", "N/A"), language="text")
            
            st.markdown("**🎬 Video Prompt:**")
            with st.expander("View prompt sent to AI"):
                st.write(st.session_state.isl_data.get("video_prompt", ""))

    with col2:
        st.markdown("**🎬 ISL Sign Language Video:**")

        # Generate button
        if not st.session_state.prediction_id and not st.session_state.video_url:
            if st.button("🎬 Generate ISL Video", use_container_width=True):
                if not replicate_token:
                    st.error("⚠️ Please enter your Replicate token in the sidebar.")
                elif st.session_state.isl_data:
                    prompt = st.session_state.isl_data.get("video_prompt", "")
                    with st.spinner("🚀 Submitting video generation job..."):
                        pred_id, status = start_video_generation(prompt)
                    if pred_id:
                        st.session_state.prediction_id = pred_id
                        st.session_state.video_status = status
                        st.rerun()

        # Poll and display status
        if st.session_state.prediction_id and not st.session_state.video_url:
            status, video_url, error = poll_video_status(st.session_state.prediction_id)
            st.session_state.video_status = status

            if status in ("starting", "processing"):
                st.info(f"⏳ Video is being generated... Status: **{status}**")
                st.markdown("_This typically takes 30–90 seconds. Refresh to check progress._")
                if st.button("🔃 Check Status"):
                    st.rerun()
                    
                # Auto-refresh via progress display
                progress_bar = st.progress(0)
                for i in range(30):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / 30)
                    s, v, e = poll_video_status(st.session_state.prediction_id)
                    if s == "succeeded" and v:
                        st.session_state.video_url = v
                        st.session_state.video_status = "succeeded"
                        progress_bar.progress(1.0)
                        st.rerun()
                        break
                    elif s == "failed":
                        st.session_state.video_status = "failed"
                        st.error(f"❌ Video generation failed: {e}")
                        break
                else:
                    st.warning("Still processing... click 'Check Status' to refresh.")

            elif status == "succeeded" and video_url:
                st.session_state.video_url = video_url
                st.rerun()

            elif status == "failed":
                st.error(f"❌ Generation failed: {error}")
                st.markdown("Try clicking Reset and recording again.")

        # Show final video
        if st.session_state.video_url:
            st.markdown('<p class="status-success">✅ Video Ready!</p>', unsafe_allow_html=True)
            st.video(st.session_state.video_url)
            st.markdown(f"[📥 Download Video]({st.session_state.video_url})", unsafe_allow_html=False)

# -------------------------------
# 📌 FOOTER
# -------------------------------
st.markdown("---")
st.caption(
    "sign"
)
