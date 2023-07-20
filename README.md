# Mixing audio data and fine-tuning wav2vec2.0
Leijenhorst, E. F. (2023). Fine-tuning ASR to specific noise environments: noise robustness in a climbing gym. [Master's thesis, University of Groningen].

Below is a list of all relevant scripts used for creating the thesis above.

- mix_longnoise_shortspeech.py - Used for mixing speech files with noise, when len(noise)>len(speech)
- mix_shortnoise_shortspeech_SNR.py - Used for mixing speech files with noise at different SNRs, when noise files can be shorter than speech files
- create_metadata.py - Used for creating a metadata file in the correctly readable format for transformers 
- finetune.py - Used for fine-tuning wav2vec 2.0
- eval.py - Used for evaluating ASR model performance: calculating the WER and CER  
