
import streamlit as st
import torch
import librosa
import tempfile
import gc
import os

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# Page configuration
st.set_page_config(
    page_title="Balochi Speech Recognition",
    page_icon="🎙️",
    layout="centered"
)

# Hugging Face repositories
ASR_MODEL_ID = "shahMeer2002/balochi-xlsr-asr"
PUNCTUATION_MODEL_ID = "shahMeer2002/balochi-byt5-punctuation"

# Supported audio formats
SUPPORTED_FORMATS = [
    "wav", "mp3", "m4a", "flac", "ogg", "aac"
]

AUDIO_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac"
}


# Load ASR model
@st.cache_resource
def load_asr_model():

    token = st.secrets["HF_TOKEN"]

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        ASR_MODEL_ID,
        token=token
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        ASR_MODEL_ID,
        token=token
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        ASR_MODEL_ID,
        token=token
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


# Load ByT5 punctuation model
@st.cache_resource
def load_punctuation_model():

    token = st.secrets["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(
        PUNCTUATION_MODEL_ID,
        token=token
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        PUNCTUATION_MODEL_ID,
        token=token
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# Application interface
st.title("🎙️ Balochi Speech Recognition")

st.write(
    "Upload a Balochi audio recording to convert "
    "your speech into written Balochi text."
)

st.info(
    "Supported formats: WAV, MP3, M4A, FLAC, OGG and AAC"
)

uploaded_file = st.file_uploader(
    "Choose a Balochi audio file",
    type=SUPPORTED_FORMATS
)


# Audio transcription
if uploaded_file is not None:

    file_extension = uploaded_file.name.rsplit(
        ".", 1
    )[-1].lower()

    audio_format = AUDIO_MIME_TYPES.get(
        file_extension,
        "audio/wav"
    )

    st.audio(
        uploaded_file.getvalue(),
        format=audio_format
    )

    if st.button("Transcribe Audio", type="primary"):

        try:
            with st.spinner("Transcribing Balochi audio..."):

                # Release punctuation model memory
                load_punctuation_model.clear()
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                processor, model, device = load_asr_model()

                # Save uploaded audio temporarily
                with tempfile.NamedTemporaryFile(
                    suffix="." + file_extension,
                    delete=False
                ) as temp_audio:

                    temp_audio.write(
                        uploaded_file.getvalue()
                    )

                    temp_audio_path = temp_audio.name

                try:
                    # Load and convert audio to 16kHz mono
                    audio, sample_rate = librosa.load(
                        temp_audio_path,
                        sr=16000,
                        mono=True
                    )

                finally:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)

                if len(audio) == 0:
                    st.error("The uploaded audio is empty.")
                    st.stop()

                # Prepare audio for ASR model
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

                st.session_state["transcription"] = transcription

                st.session_state.pop(
                    "punctuated_text",
                    None
                )

            st.success("Transcription completed!")

        except Exception as e:
            st.error(
                f"Transcription failed: {e}"
            )


# Display original transcription
if "transcription" in st.session_state:

    st.subheader("Balochi Transcription")

    st.text_area(
        "Original Recognized Text",
        value=st.session_state["transcription"],
        height=150
    )

    st.download_button(
        label="Download Original Transcription",
        data=st.session_state["transcription"],
        file_name="balochi_transcription.txt",
        mime="text/plain"
    )

    # Punctuation restoration
    if st.button("Restore Punctuation"):

        try:
            with st.spinner(
                "Restoring Balochi punctuation..."
            ):

                # Release ASR model memory
                load_asr_model.clear()
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                tokenizer, model, device = (
                    load_punctuation_model()
                )

                text = st.session_state["transcription"]

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        num_beams=4
                    )

                punctuated_text = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                )

                st.session_state["punctuated_text"] = (
                    punctuated_text
                )

            st.success(
                "Punctuation restoration completed!"
            )

        except Exception as e:
            st.error(
                f"Punctuation restoration failed: {e}"
            )


# Display punctuation-restored text
if "punctuated_text" in st.session_state:

    st.subheader("Punctuation Restored Text")

    st.text_area(
        "Processed Balochi Text",
        value=st.session_state["punctuated_text"],
        height=150
    )

    st.download_button(
        label="Download Processed Text",
        data=st.session_state["punctuated_text"],
        file_name="balochi_punctuated_transcription.txt",
        mime="text/plain"
    )
