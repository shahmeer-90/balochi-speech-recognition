# SR-05: Recognition Post-Processing
## Punctuation, Normalisation and Subtitles

### Project Overview
This project implements an SR-05 post-processing prototype for Balochi ASR.

### Technologies
- Python, Google Colab, PyTorch
- Hugging Face Transformers
- XLM-RoBERTa and ByT5-small
- Pandas and Regular Expressions

### Punctuation Restoration
**Trained ByT5 Model (Hugging Face)**

- Model: ByT5-small
- Repository: https://huggingface.co/shahMeer2002/balochi-byt5-punctuation
- Architecture: T5ForConditionalGeneration
- Model format: Safetensors

Two separate model experiments were evaluated.

**XLM-RoBERTa**
- Architecture: XLMRobertaForTokenClassification
- Training: 4009, Validation: 502, Testing: 522
- Precision: 0.304245
- Recall: 0.365943
- F1 Score: 0.331843
- Accuracy: 0.953158

**ByT5-small**
- Architecture: T5ForConditionalGeneration
- Training: 3570, Validation: 447, Testing: 447
- Exact Match Accuracy: 61.74%
- Punctuation Precision: 75.06%
- Punctuation Recall: 53.25%
- Punctuation F1: 62.30%
- Training Loss: 1.253420
- Validation Loss: 0.065562, 0.049833, 0.047250

These are separate experiments with different architectures and metrics.

### Inverse Text Normalisation
- Original records: 5033
- Unique records: 4478
- Duplicate rows removed: 555
- Changed sentences: 11
- Unchanged sentences: 4467
- Missing values: 0

Operations include Eastern Arabic digit conversion, date spacing normalisation and preservation of Balochi number words.
Verified native-speaker number mappings are pending.

### Subtitle Generation
- SRT and WebVTT generation
- Maximum 42 characters per line
- Maximum two lines per subtitle block
- Five test samples processed
- 47 aligned words
- Five SRT files generated
- Combined timestamp CSV generated
- Missing timestamps: 0
- Invalid durations: 0
- Negative timestamps: 0

Reference-guided CTC forced alignment was used. The first sample was manually listened to; other samples passed numeric timestamp checks. Broader acoustic evaluation remains necessary.

### Workflow
1. Receive Balochi transcript.
2. Restore punctuation.
3. Apply available normalisation rules.
4. Align reference transcript words with audio.
5. Generate subtitle files.

Complete SR-03 integration remains pending.

### Limitations
- Subtitle alignment tested on five samples only.
- Full test-set processing remains future work.
- Verified Balochi number-word mappings are pending.
- Native-speaker validation is recommended.
- Punctuation performance requires further improvement.

### Conclusion
SR-05 provides a prototype for Balochi ASR punctuation restoration, inverse text normalisation and subtitle generation. XLM-RoBERTa and ByT5-small were evaluated separately. Normalisation was tested on 5033 records, and reference-guided alignment was tested on five audio samples, producing 47 aligned words. Full SR-03 integration and broader validation remain future work.
