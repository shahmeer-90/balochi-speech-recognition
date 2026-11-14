

import streamlit as st
import torch
import librosa
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)

# Page configuration
st.set_page_config(
    page_title="Balochi Speech Recognition",
    page_icon="🎙️",
    layout="centered"
)

# Model repository
MODEL_ID = "shahMeer2002/balochi-xlsr-asr"

@st.cache_resource
def load_asr_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


# Application interface
st.title("🎙️ Balochi Speech Recognition")

st.write(
    "Upload a Balochi audio recording and convert "
    "your speech into written Balochi text."
)

st.info("Supported audio format: WAV")

uploaded_file = st.file_uploader(
    "Choose a Balochi audio file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    if st.button("Transcribe Audio", type="primary"):

        try:
            with st.spinner("Loading model and transcribing audio..."):

                processor, model, device = load_asr_model()

                audio, sample_rate = librosa.load(
                    uploaded_file,
                    sr=16000,
                    mono=True
                )

                if len(audio) == 0:
                    st.error("The uploaded audio is empty.")
                    st.stop()

                inputs = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )

                input_values = inputs.input_values.to(device)

                with torch.no_grad():
                    logits = model(input_values).logits

                predicted_ids = torch.argmax(
                    logits,
                    dim=-1
                )

                transcription = processor.batch_decode(
                    predicted_ids
                )[0]

            st.success("Transcription completed!")

            st.subheader("Balochi Transcription")

            st.text_area(
                "Recognized Text",
                value=transcription,
                height=150
            )

            st.download_button(
                label="Download Transcription",
                data=transcription,
                file_name="balochi_transcription.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Transcription failed: {e}")
