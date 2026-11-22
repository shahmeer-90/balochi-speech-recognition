
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


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Balochi Speech Recognition",
    page_icon="🎙️",
    layout="centered"
)

ASR_MODEL_ID = "shahMeer2002/balochi-xlsr-asr"

PUNCTUATION_MODEL_ID = (
    "shahMeer2002/balochi-byt5-punctuation"
)

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

# Limit CPU thread usage on cloud
torch.set_num_threads(2)


# ==========================================
# 2. DEVICE CONFIGURATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_hf_token():

    try:
        return st.secrets["HF_TOKEN"]

    except Exception:
        st.error(
            "HF_TOKEN is missing from Streamlit Secrets."
        )
        st.stop()


# ==========================================
# 3. MEMORY MANAGEMENT
# ==========================================

def release_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ==========================================
# 4. LOAD ASR MODEL
# ==========================================

def load_asr_model(token):

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        ASR_MODEL_ID,
        token=token
    )

    feature_extractor = (
        Wav2Vec2FeatureExtractor.from_pretrained(
            ASR_MODEL_ID,
            token=token
        )
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        ASR_MODEL_ID,
        token=token
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# ==========================================
# 5. LOAD BYT5 PUNCTUATION MODEL
# ==========================================

def load_punctuation_model(token):

    tokenizer = AutoTokenizer.from_pretrained(
        PUNCTUATION_MODEL_ID,
        token=token
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        PUNCTUATION_MODEL_ID,
        token=token
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ==========================================
# 6. AUDIO PROCESSING
# ==========================================

def process_audio(audio_bytes, extension):

    temp_audio_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix="." + extension,
            delete=False
        ) as temp_audio:

            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        audio, sample_rate = librosa.load(
            temp_audio_path,
            sr=16000,
            mono=True
        )

        if audio is None or len(audio) == 0:
            raise ValueError("Audio file is empty.")

        return audio

    finally:

        if (
            temp_audio_path
            and os.path.exists(temp_audio_path)
        ):
            os.remove(temp_audio_path)


# ==========================================
# 7. TRANSCRIBE AUDIO
# ==========================================

def transcribe_audio(audio_bytes, extension):

    token = get_hf_token()

    release_memory()

    processor = None
    model = None

    try:

        with st.spinner(
            "Loading Balochi ASR model..."
        ):

            processor, model = load_asr_model(token)

        with st.spinner(
            "Processing audio and recognizing speech..."
        ):

            audio = process_audio(
                audio_bytes,
                extension
            )

            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(
                DEVICE
            )

            with torch.inference_mode():

                logits = model(
                    input_values
                ).logits

            predicted_ids = torch.argmax(
                logits,
                dim=-1
            )

            transcription = processor.batch_decode(
                predicted_ids
            )[0].strip()

            del inputs
            del input_values
            del logits
            del predicted_ids

            if not transcription:

                raise ValueError(
                    "ASR returned empty transcription. "
                    "Please try another audio recording."
                )

            return transcription

    finally:

        del model
        del processor

        release_memory()


# ==========================================
# 8. RESTORE PUNCTUATION
# ==========================================

def restore_punctuation(text):

    if not text or not text.strip():

        raise ValueError(
            "No transcription text available."
        )

    token = get_hf_token()

    release_memory()

    tokenizer = None
    model = None

    try:

        with st.spinner(
            "Loading ByT5 punctuation model..."
        ):

            tokenizer, model = load_punctuation_model(
                token
            )

        with st.spinner(
            "Restoring Balochi punctuation..."
        ):

            inputs = tokenizer(
                text.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=1,
                    do_sample=False
                )

            punctuated_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()

            if not punctuated_text:

                raise ValueError(
                    "ByT5 returned empty output."
                )

            del inputs
            del output_ids

            return punctuated_text

    finally:

        del model
        del tokenizer

        release_memory()


# ==========================================
# 9. APPLICATION INTERFACE
# ==========================================

st.title("🎙️ Balochi Speech Recognition")

st.write(
    "Upload a Balochi audio recording to convert "
    "your speech into written Balochi text."
)

st.info(
    "Supported formats: WAV, MP3, M4A, FLAC, OGG and AAC"
)

st.caption(
    "Audio is processed using the XLS-R Balochi ASR model."
)


# ==========================================
# 10. AUDIO UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Choose a Balochi audio file",
    type=SUPPORTED_FORMATS,
    key="audio_uploader"
)


if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()

    extension = uploaded_file.name.rsplit(
        ".", 1
    )[-1].lower()

    audio_format = AUDIO_MIME_TYPES.get(
        extension,
        "audio/wav"
    )

    st.audio(
        audio_bytes,
        format=audio_format
    )

    st.caption(
        f"File: {uploaded_file.name} | "
        f"Size: {len(audio_bytes) / (1024 * 1024):.2f} MB"
    )

    if st.button(
        "Transcribe Audio",
        type="primary",
        use_container_width=True
    ):

        st.session_state.pop(
            "transcription",
            None
        )

        st.session_state.pop(
            "punctuated_text",
            None
        )

        try:

            transcription = transcribe_audio(
                audio_bytes,
                extension
            )

            st.session_state["transcription"] = (
                transcription
            )

            st.success(
                "Transcription completed successfully!"
            )

        except Exception as e:

            st.error(
                f"Transcription failed: {str(e)}"
            )


# ==========================================
# 11. DISPLAY ASR TRANSCRIPTION
# ==========================================

if (
    "transcription" in st.session_state
    and st.session_state["transcription"]
):

    st.divider()

    st.subheader(
        "📝 Balochi Transcription"
    )

    st.text_area(
        "Original Recognized Text",
        value=st.session_state["transcription"],
        height=180,
        disabled=True
    )

    st.download_button(
        label="Download Original Transcription",
        data=st.session_state["transcription"],
        file_name="balochi_transcription.txt",
        mime="text/plain",
        use_container_width=True
    )

    # ======================================
    # 12. PUNCTUATION RESTORATION BUTTON
    # ======================================

    if st.button(
        "Restore Punctuation",
        type="primary",
        use_container_width=True
    ):

        st.session_state.pop(
            "punctuated_text",
            None
        )

        try:

            original_text = (
                st.session_state["transcription"]
            )

            punctuated_text = restore_punctuation(
                original_text
            )

            st.session_state["punctuated_text"] = (
                punctuated_text
            )

            st.success(
                "Punctuation restoration completed!"
            )

        except Exception as e:

            st.error(
                f"Punctuation restoration failed: {str(e)}"
            )


# ==========================================
# 13. DISPLAY RESTORED TEXT
# ==========================================

if (
    "punctuated_text" in st.session_state
    and st.session_state["punctuated_text"]
):

    st.divider()

    st.subheader(
        "✨ Punctuation Restored Text"
    )

    st.text_area(
        "Processed Balochi Text",
        value=st.session_state["punctuated_text"],
        height=180,
        disabled=True
    )

    st.download_button(
        label="Download Processed Text",
        data=st.session_state["punctuated_text"],
        file_name="balochi_punctuated_transcription.txt",
        mime="text/plain",
        use_container_width=True
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "Balochi Speech Recognition | "
    "XLS-R ASR + ByT5 Punctuation Restoration"
)
