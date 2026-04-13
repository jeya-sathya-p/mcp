import streamlit as st
import json
import requests
import time

# -------------------------------
# 🌐 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="SignSpeak AI 👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99,102,241,0.4);
    }
    .result-box {
        background: #1e1e2e;
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .step-badge {
        background: #6366f1;
        color: white;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤟 SignSpeak AI")
st.caption("Speech → ISL Gloss → AI Visual | Powered by Groq + Replicate")

# -------------------------------
# 🔑 SESSION STATE INIT
# -------------------------------
for key in ["transcription", "isl_data", "image_url", "error_log"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# 🔑 SIDEBAR: API KEYS
# -------------------------------
with st.sidebar:
    st.header("🔑 Configuration")
    st.markdown("---")

    # Try secrets first, then manual input
    groq_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    replicate_token = st.secrets.get("REPLICATE_API_TOKEN", "") if hasattr(st, "secrets") else ""

    if not groq_key:
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.caption("Get free key → [console.groq.com](https://console.groq.com)")

    if not replicate_token:
        replicate_token = st.text_input("Replicate Token", type="password", placeholder="r8_...")
        st.caption("Get free token → [replicate.com](https://replicate.com)")

    st.markdown("---")

    # Model selector
    st.subheader("⚙️ Settings")
    image_model = st.selectbox(
        "Image Model",
        [
            "stability-ai/sdxl:7762fd07cf82c948538e41f4d1b5524441416468895acac7769c4a38e3407c91",
            "black-forest-labs/flux-schnell",
            "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
        ],
        help="SDXL = best quality | FLUX Schnell = fastest | SDXL Lightning = fast + good"
    )

    if st.button("🔄 Reset All"):
        for key in ["transcription", "isl_data", "image_url", "error_log"]:
            st.session_state[key] = None
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Status")
    st.write("🎤 Transcription:", "✅" if st.session_state.transcription else "⏳")
    st.write("🧠 ISL Translation:", "✅" if st.session_state.isl_data else "⏳")
    st.write("🖼️ Image:", "✅" if st.session_state.image_url else "⏳")

# -------------------------------
# 🎤 TRANSCRIBE (Groq Whisper)
# -------------------------------
def transcribe_audio(audio_bytes):
    if not groq_key:
        st.error("❌ Please enter your Groq API Key in the sidebar.")
        return None
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
        if res.status_code != 200:
            st.error(f"Groq Transcription Error {res.status_code}: {res.text}")
            return None
        result = res.json()
        return result.get("text", "").strip()
    except requests.exceptions.Timeout:
        st.error("⏱️ Transcription timed out. Try a shorter audio clip.")
        return None
    except Exception as e:
        st.error(f"🚨 Transcription Error: {e}")
        return None

# -------------------------------
# 🧠 ISL TRANSLATION (Groq LLaMA)
# -------------------------------
def get_isl_translation(text):
    if not groq_key:
        st.error("❌ Please enter your Groq API Key in the sidebar.")
        return None
    try:
        system_prompt = """You are an expert Indian Sign Language (ISL) interpreter.
Convert the given English text to ISL gloss notation.

ISL Gloss Rules:
- Use UPPERCASE words
- Remove articles (a, an, the)
- Use base verb forms (no -ing, -ed)
- Keep important adjectives and nouns
- Use ISL spatial markers like IX (index pointing)
- Show topic-comment structure (topic first)

Return ONLY valid JSON in this exact format:
{
  "gloss": "ISL GLOSS WORDS HERE",
  "visual_description": "A person signing: detailed description of hand shapes, positions and movements for this ISL phrase",
  "english_simplified": "simplified English version",
  "signs_count": 5
}"""

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert to ISL gloss: {text}"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 500
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        if res.status_code != 200:
            st.error(f"Groq ISL Error {res.status_code}: {res.text}")
            return None
        content = res.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"JSON Parse Error: {e}")
        return None
    except Exception as e:
        st.error(f"🚨 ISL Translation Error: {e}")
        return None

# -------------------------------
# 🖼️ GENERATE IMAGE (Replicate)
# -------------------------------
def generate_image_replicate(visual_description, model_version):
    if not replicate_token:
        st.error("❌ Please enter your Replicate API Token in the sidebar.")
        return None

    try:
        # Build a detailed, accurate prompt for ISL signing
        prompt = (
            f"Photorealistic image of a person performing Indian Sign Language. "
            f"{visual_description}. "
            f"Clear view of hands and fingers, neutral background, good lighting, "
            f"educational illustration style, front-facing person, high detail hands"
        )

        negative_prompt = (
            "blurry, distorted hands, extra fingers, missing fingers, bad anatomy, "
            "low quality, cartoon, anime, abstract, dark lighting, back view"
        )

        headers = {
            "Authorization": f"Token {replicate_token}",
            "Content-Type": "application/json"
        }

        # Different input format per model
        if "flux-schnell" in model_version:
            input_data = {
                "prompt": prompt,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90
            }
        else:
            # SDXL and SDXL Lightning
            input_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": 768,
                "height": 768,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 30 if "sdxl:7762" in model_version else 4,
                "refine": "expert_ensemble_refiner" if "sdxl:7762" in model_version else "no_refiner",
                "high_noise_frac": 0.8 if "sdxl:7762" in model_version else 0.8,
            }

        # Start prediction
        payload = {"version": model_version.split(":")[-1], "input": input_data} if ":" in model_version else {"input": input_data}

        # Use model-specific endpoint
        if "/" in model_version:
            model_path = model_version.split(":")[0]  # e.g. stability-ai/sdxl
            if ":" in model_version:
                version_id = model_version.split(":")[1]
                create_url = "https://api.replicate.com/v1/predictions"
                body = {"version": version_id, "input": input_data}
            else:
                create_url = f"https://api.replicate.com/v1/models/{model_path}/predictions"
                body = {"input": input_data}
        else:
            create_url = "https://api.replicate.com/v1/predictions"
            body = {"version": model_version, "input": input_data}

        res = requests.post(create_url, headers=headers, json=body, timeout=30)

        if res.status_code not in [200, 201]:
            st.error(f"Replicate API Error {res.status_code}: {res.text}")
            return None

        prediction = res.json()
        prediction_id = prediction.get("id")

        if not prediction_id:
            st.error(f"No prediction ID returned: {prediction}")
            return None

        # Poll for result
        poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        progress_bar = st.progress(0, text="🎨 Generating image...")
        max_wait = 120  # seconds
        start_time = time.time()
        step = 0

        while time.time() - start_time < max_wait:
            poll_res = requests.get(poll_url, headers=headers, timeout=15)
            poll_data = poll_res.json()
            status = poll_data.get("status")

            # Update progress visually
            elapsed = time.time() - start_time
            progress = min(int((elapsed / 60) * 100), 95)
            progress_bar.progress(progress, text=f"🎨 Status: {status} ({int(elapsed)}s)")

            if status == "succeeded":
                progress_bar.progress(100, text="✅ Done!")
                output = poll_data.get("output")
                if isinstance(output, list) and len(output) > 0:
                    return output[0]
                elif isinstance(output, str):
                    return output
                else:
                    st.error(f"Unexpected output format: {output}")
                    return None

            elif status == "failed":
                error = poll_data.get("error", "Unknown error")
                progress_bar.empty()
                st.error(f"❌ Generation failed: {error}")
                return None

            elif status in ["starting", "processing"]:
                time.sleep(3)
                step += 1
            else:
                time.sleep(2)

        progress_bar.empty()
        st.error("⏱️ Image generation timed out after 2 minutes.")
        return None

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out connecting to Replicate.")
        return None
    except Exception as e:
        st.error(f"🚨 Image Generation Error: {e}")
        return None

# ================================
# 🎯 MAIN APP LAYOUT
# ================================
st.markdown("---")

# STEP 1: Audio Input
st.markdown('<span class="step-badge">STEP 1</span> **Record your voice**', unsafe_allow_html=True)
audio = st.audio_input("🎙️ Click the mic to start recording")

if audio and not st.session_state.transcription:
    col_spin1, col_spin2 = st.columns([1, 4])
    with col_spin1:
        st.info("🔄 Processing...")

    with st.spinner("🎧 Transcribing audio with Groq Whisper..."):
        text = transcribe_audio(audio.getvalue())

    if text:
        st.session_state.transcription = text
        with st.spinner("🧠 Translating to ISL gloss..."):
            st.session_state.isl_data = get_isl_translation(text)
        st.session_state.image_url = None  # reset image on new audio
        st.rerun()
    else:
        st.warning("⚠️ Could not transcribe. Check your Groq API key and audio quality.")

# STEP 2 & 3: Results
if st.session_state.transcription:
    st.markdown("---")

    # Transcription result
    st.markdown('<span class="step-badge">STEP 2</span> **Transcription Result**', unsafe_allow_html=True)
    st.markdown(f'<div class="result-box">🗣️ <b>Spoken Text:</b><br>{st.session_state.transcription}</div>', unsafe_allow_html=True)

    if st.session_state.isl_data:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<span class="step-badge">ISL GLOSS</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box"><h3 style="color:#6366f1">{st.session_state.isl_data.get("gloss","")}</h3></div>', unsafe_allow_html=True)

            with st.expander("📋 Full ISL Details"):
                st.json(st.session_state.isl_data)

        with col2:
            st.markdown('<span class="step-badge">VISUAL DESCRIPTION</span>', unsafe_allow_html=True)
            visual_desc = st.session_state.isl_data.get("visual_description", "")
            st.markdown(f'<div class="result-box">{visual_desc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="step-badge">STEP 3</span> **Generate ISL Visual**', unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        gen_clicked = st.button("🎨 Generate Image", use_container_width=True)
    with col_info:
        st.caption(f"Model: `{image_model.split('/')[1].split(':')[0] if '/' in image_model else image_model}`")
        st.caption("⚡ FLUX Schnell ~10s | SDXL Lightning ~15s | SDXL ~45s")

    if gen_clicked:
        if not replicate_token:
            st.error("❌ Please add your Replicate API Token in the sidebar.")
        elif st.session_state.isl_data:
            visual_prompt = st.session_state.isl_data.get("visual_description", st.session_state.transcription)
            img_url = generate_image_replicate(visual_prompt, image_model)
            if img_url:
                st.session_state.image_url = img_url
                st.rerun()

    if st.session_state.image_url:
        st.success("✅ Image generated!")
        st.image(
            st.session_state.image_url,
            caption=f"ISL: {st.session_state.isl_data.get('gloss','') if st.session_state.isl_data else ''}",
            use_container_width=True
        )
        st.markdown(f"🔗 [Open full image]({st.session_state.image_url})")

# Footer
st.markdown("---")
st.caption("💡 **Setup:** Add `GROQ_API_KEY` and `REPLICATE_API_TOKEN` to Streamlit Cloud Secrets for deployment. | Free tiers available on both platforms.")
