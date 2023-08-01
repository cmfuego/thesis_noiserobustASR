# Mixing audio data and fine-tuning wav2vec2.0
Leijenhorst, E. F. (2023). Fine-tuning ASR to specific noise environments: noise robustness in a climbing gym. [Master's thesis, University of Groningen].

Below is a list of all relevant scripts used for creating the thesis above.

- `cut_noise_to_librispeechlength.py` - Used for extracting noise pieces from a folder with long noise recordings, corresponding to the length of sounds in a LibriSpeech set.
- `mix_noisespeech_samelen.py` - Used for mixing speech files with corresponding noise pieces, when len(noise)=len(speech).
- `mix_noisespeech_difflen_SNR.py` - Used for mixing speech files with noise, when noise and speech files have different lengths. Multiple different SNRs are set.
- `create_metadata.py` - Used for creating a metadata file with transcriptions in the correctly readable format for transformers, in order to make the dataset complete.
- `finetune.py` - Used for fine-tuning wav2vec 2.0 on any dataset in the right format.
- `eval.py` - Used for evaluating ASR model performance: calculating the WER and CER. Can be used for multiple models and/or datasets at once.
