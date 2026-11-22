
import streamlit as st
import torch
import librosa
import tempfile
import gc
import os
import re

import transformers
import tokenizers
import sentencepiece
import huggingface_hub

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


# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================

ASR_MODEL_ID = "shahMeer2002/balochi-xlsr-asr"
PUNCTUATION_MODEL_ID = "shahMeer2002/balochi-byt5-punctuation"

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

CHUNK_SIZE = 50

torch.set_num_threads(2)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 3. ENVIRONMENT INFORMATION
# ==========================================

with st.expander("System and Model Environment"):

    st.write("Device:", str(DEVICE))
    st.write("Transformers:", transformers.__version__)
    st.write("PyTorch:", torch.__version__)
    st.write("Tokenizers:", tokenizers.__version__)
    st.write("SentencePiece:", sentencepiece.__version__)
    st.write("Hugging Face Hub:", huggingface_hub.__version__)


# ==========================================
# 4. HUGGING FACE AUTHENTICATION
# ==========================================

def get_hf_token():

    try:
        return st.secrets["HF_TOKEN"]

    except Exception:
        st.error("HF_TOKEN is missing from Streamlit Secrets.")
        st.stop()


# ==========================================
# 5. MEMORY MANAGEMENT
# ==========================================

def release_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# 6. LOAD ASR MODEL
# ==========================================

def load_asr_model(token):

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

    model.to(DEVICE)
    model.eval()

    return processor, model


# ==========================================
# 7. LOAD BYT5 MODEL
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
# 8. AUDIO PROCESSING
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

        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# ==========================================
# 9. TRANSCRIBE AUDIO
# ==========================================

def transcribe_audio(audio_bytes, extension):

    token = get_hf_token()

    release_memory()

    processor = None
    model = None

    try:

        with st.spinner("Loading Balochi ASR model..."):

            processor, model = load_asr_model(token)

        with st.spinner("Recognizing Balochi speech..."):

            audio = process_audio(audio_bytes, extension)

            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(DEVICE)

            with torch.inference_mode():
                logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            transcription = processor.batch_decode(
                predicted_ids
            )[0].strip()

            if not transcription:
                raise ValueError("ASR returned empty transcription.")

            return transcription

    finally:

        if model is not None:
            del model

        if processor is not None:
            del processor

        release_memory()


# ==========================================
# 10. SPLIT LONG TEXT INTO CHUNKS
# ==========================================

def split_text_into_chunks(text, max_chars=CHUNK_SIZE):

    words = text.split()

    chunks = []
    current_chunk = ""

    for word in words:

        if len(word) > max_chars:

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.append(word)
            continue

        candidate = (
            current_chunk + " " + word
            if current_chunk
            else word
        )

        if len(candidate) <= max_chars:

            current_chunk = candidate

        else:

            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ==========================================
# 11. VALIDATE MODEL OUTPUT
# ==========================================

def validate_punctuation_output(original, generated):

    if not generated or not generated.strip():
        return False

    original_words = re.findall(
        r"\w+",
        original,
        flags=re.UNICODE
    )

    generated_words = re.findall(
        r"\w+",
        generated,
        flags=re.UNICODE
    )

    # Check whether the original words
    # are preserved in the generated output.

    if original_words != generated_words:
        return False

    return True


# ==========================================
# 12. RESTORE PUNCTUATION WITH DEBUGGING
# ==========================================

def restore_punctuation(text):

    if not text or not text.strip():
        raise ValueError("No transcription text available.")

    token = get_hf_token()

    release_memory()

    tokenizer = None
    model = None

    try:

        with st.spinner("Loading ByT5 punctuation model..."):

            tokenizer, model = load_punctuation_model(token)

        chunks = split_text_into_chunks(text.strip())

        processed_chunks = []
        debug_samples = []

        fallback_count = 0

        progress = st.progress(0)
        status = st.empty()

        for index, chunk in enumerate(chunks):

            status.write(
                f"Processing chunk {index + 1} of {len(chunks)}..."
            )

            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False
                )

            generated = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()

            is_valid = validate_punctuation_output(
                chunk,
                generated
            )

            if is_valid:

                processed_chunks.append(generated)

            else:

                processed_chunks.append(chunk)
                fallback_count += 1

            # Keep raw output of first five chunks
            # for debugging and comparison.

            if index < 5:

                debug_samples.append({
                    "chunk": chunk,
                    "generated": generated,
                    "valid": is_valid
                })

            progress.progress(
                (index + 1) / len(chunks)
            )

        status.empty()
        progress.empty()

        final_text = " ".join(processed_chunks)

        if not final_text.strip():
            raise ValueError("No output produced.")

        # ==================================
        # DEBUG RESULTS
        # ==================================

        st.divider()

        st.subheader("🔍 ByT5 Model Debug Results")

        st.caption(
            "These results show the actual output "
            "generated by the punctuation model."
        )

        for i, sample in enumerate(debug_samples):

            with st.expander(f"Chunk {i + 1}"):

                st.write("Original Input:")

                st.code(
                    sample["chunk"],
                    language=None
                )

                st.write("Raw ByT5 Output:")

                st.code(
                    sample["generated"] or "[EMPTY OUTPUT]",
                    language=None
                )

                if sample["valid"]:

                    st.success("Validation: Passed")

                else:

                    st.error("Validation: Failed")

        # ==================================
        # PROCESSING SUMMARY
        # ==================================

        st.subheader("Processing Summary")

        st.write("Total chunks:", len(chunks))
        st.write("Accepted chunks:", len(chunks) - fallback_count)
        st.write("Rejected chunks:", fallback_count)

        if fallback_count == len(chunks):

            st.error(
                "Punctuation restoration failed for all chunks. "
                "The original transcription has been preserved."
            )

        elif fallback_count > 0:

            st.warning(
                "Some chunks failed validation. "
                "Original text was preserved for those chunks."
            )

        else:

            st.success(
                "All chunks passed word-preservation validation."
            )

        return final_text, fallback_count, len(chunks)

    finally:

        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer

        release_memory()


# ==========================================
# 13. APPLICATION INTERFACE
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
    "Audio recognition: XLS-R | "
    "Punctuation restoration: ByT5"
)


# ==========================================
# 14. AUDIO UPLOAD
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

        st.session_state.pop("transcription", None)
        st.session_state.pop("punctuated_text", None)

        try:

            transcription = transcribe_audio(
                audio_bytes,
                extension
            )

            st.session_state["transcription"] = transcription

            st.success("Transcription completed successfully!")

        except Exception as e:

            st.error(f"Transcription failed: {str(e)}")


# ==========================================
# 15. DISPLAY ORIGINAL TRANSCRIPTION
# ==========================================

if (
    "transcription" in st.session_state
    and st.session_state["transcription"]
):

    st.divider()

    st.subheader("📝 Balochi Transcription")

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

    if st.button(
        "Restore Punctuation",
        type="primary",
        use_container_width=True
    ):

        st.session_state.pop("punctuated_text", None)

        try:

            original_text = st.session_state["transcription"]

            (
                punctuated_text,
                fallback_count,
                total_chunks
            ) = restore_punctuation(original_text)

            st.session_state["punctuated_text"] = punctuated_text

            st.session_state["punctuation_fallback_count"] = fallback_count
            st.session_state["punctuation_total_chunks"] = total_chunks

            if fallback_count == 0:

                st.success("All chunks passed validation.")

            elif fallback_count == total_chunks:

                st.error(
                    "Punctuation model failed on all chunks. "
                    "Original text is preserved."
                )

            else:

                st.warning(
                    "Punctuation processing completed with "
                    "some rejected chunks."
                )

        except Exception as e:

            st.error(
                f"Punctuation restoration failed: {str(e)}"
            )


# ==========================================
# 16. DISPLAY PROCESSED TEXT
# ==========================================

if (
    "punctuated_text" in st.session_state
    and st.session_state["punctuated_text"]
):

    st.divider()

    st.subheader("✨ Punctuation Processed Text")

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
# 17. FOOTER
# ==========================================

st.divider()

st.caption(
    "SR-05 | Balochi Speech Recognition "
    "Post-Processing Prototype"
)
