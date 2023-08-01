# This script is largely based on a code snippet in the README file from https://huggingface.co/facebook/wav2vec2-base-960h
# It can be used for evaluating multiple models / model checkpoints on multiple datasets at once.

import glob
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer, cer

### Insert all models and datasets you want to use for evaluation in the lists below (at least one for each)
datasetpath_list = [] # ["OtherNoise_final/", "Mixed_Gym_Libri/15/", "LibriSpeech/"]
modelpath_list = [] #["facebook/wav2vec2-base-960h"] # "MODEL_librigymnoise/checkpoint-10000" # "MODEL_libriothernoise/checkpoint-10000" #  
#for i in glob.glob("MODEL_libriothernoise/*"):
    #modelpath_list.append(i)

    
datasetlist = []
dataset_wer_listofdicts = []
for count, dataset_path in enumerate(datasetpath_list):
    datasetlist.append(dataset_path)
    dataset_wer_listofdicts.append({})
    dataset = load_dataset("audiofolder", data_dir=dataset_path) # LOAD RECORDINGS AND TRANSCRIPTIONS INTO A DATASET READABLE FOR HF
    
    
    for model_path in modelpath_list:
    
        # Prepare: load model and mapping function
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda")
        processor = Wav2Vec2Processor.from_pretrained(model_path)

        def map_to_pred(batch):
            input_values = processor(batch["audio"][0]["array"], return_tensors="pt", padding="longest").input_values
            with torch.no_grad():
                logits = model(input_values.to("cuda")).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            batch["text"] = transcription
            return batch
            
        result = dataset.map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])
        
        dataset_wer_listofdicts[count][model_path] = ([wer(result["test"]["text"], result["test"]["transcription"]), cer(result["test"]["text"], result["test"]["transcription"])])

print(datasetlist)
print(dataset_wer_listofdicts, "#\n#\n#\n#\n")

for i in dataset_wer_listofdicts:
    print(i, "\n")


#print(f"Test results of the following dataset (test split): {dataset_path}")
#print(f"on the following model: {model_path}")

#print("WER:", wer(result["test"]["text"], result["test"]["transcription"]))
#print("CER:", cer(result["test"]["text"], result["test"]["transcription"]))
