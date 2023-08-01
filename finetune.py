# This code is largely based on the Hugging Face tutorial at https://huggingface.co/docs/transformers/v4.27.2/en/tasks/asr

# REQUIREMENTS to customize file to use:
# A dataset that can be loaded using load_dataset: a folder with audio files and metadata.csv OR one of the ready-made datasets available online
# A model (checkpoint) to start training.

from transformers import AutoFeatureExtractor, Wav2Vec2Processor
import jiwer


#LOAD DATASET
from datasets import load_dataset, Audio
dataset  = load_dataset("audiofolder", data_dir="OtherNoise_final/") # CONVERT RECORDINGS AND TRANSCRIPTIONS INTO A DATASET READABLE FOR HUGGING FACE MODELS

print("Yes! The dataset is loaded. Now let's preprocess it.")


#PREPROCESS
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")      ##### Change

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

encoded_noisyspeech = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=6)

print("preprocess_checkpoint")

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

print("Done with preprocessing, now let\'s prepare the evaluation metrics")



#EVALUATE
import evaluate

wer = evaluate.load("wer")

import numpy as np

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_result = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_result}

print("Done with the evaluation metrics, now let's start with the actual training preparation")


#TRAIN
from transformers import AutoModelForCTC, TrainingArguments, Trainer

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

training_args = TrainingArguments(
    output_dir="MODEL_libriothernoise",    #### Change
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=10000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_noisyspeech["train"],
    eval_dataset=encoded_noisyspeech["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("All done! Let's train!")
trainer.train()
