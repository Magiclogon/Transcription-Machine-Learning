import os
import json
import torch
import evaluate
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "arabic"
TASK = "transcribe"
OUTPUT_DIR = "./whisper-small-darija"
CACHE_DIR = "./cached_dataset"
TRAIN_SPLIT = 0.9
TARGET_SR = 16000

USE_LOCAL = False
CLIPS_DIR = "./clips"

USE_DODA = True
DODA_DIR = "./doda"


def load_audio_file(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def load_local_and_split(clips_dir, train_split):
    manifest_path = os.path.join(clips_dir, "metadata.jsonl")
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = os.path.join(clips_dir, entry["file_name"])
            if os.path.exists(audio_path):
                records.append({
                    "audio_path": str(Path(audio_path).resolve()),
                    "transcription": entry["transcription"],
                })
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1 - train_split, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def load_doda_and_split(doda_dir, train_split):
    parquet_files = sorted(Path(doda_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in '{doda_dir}'.\n"
            f"Create a folder called 'doda' next to train.py and put the 5 parquet files inside it."
        )

    print(f"Found {len(parquet_files)} parquet file(s):")
    for p in parquet_files:
        print(f"  {p.name}")

    raw = load_dataset(
        "parquet",
        data_files={"train": [str(p) for p in parquet_files]},
        split="train",
    )

    from datasets import Audio
    raw = raw.cast_column("audio", Audio(decode=False))

    raw = raw.filter(
        lambda x: x is not None and len(x.strip()) > 0,
        input_columns=["darija_Arab_new"],
    )
    print(f"Total usable samples after filtering: {len(raw)}")

    split = raw.train_test_split(test_size=1 - train_split, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def prepare_local(batch, processor):
    audio_array = load_audio_file(batch["audio_path"])
    batch["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=TARGET_SR
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch


def prepare_doda(batch, processor):
    import io
    audio_col = batch["audio"]

    raw_bytes = audio_col["bytes"] if isinstance(audio_col, dict) else audio_col
    audio_array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)

    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)

    audio_array = audio_array.astype(np.float32)

    if sr != TARGET_SR:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)

    batch["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=TARGET_SR
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["darija_Arab_new"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics_fn(pred, tokenizer, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    if os.path.exists(CACHE_DIR):
        print(f"Loading preprocessed dataset from cache: {CACHE_DIR}")
        from datasets import load_from_disk
        dataset = load_from_disk(CACHE_DIR)
    else:
        print("Preprocessing dataset (this runs only once)...")

        if USE_DODA:
            raw = load_doda_and_split(DODA_DIR, TRAIN_SPLIT)
            prepare_fn = lambda batch: prepare_doda(batch, processor)
        else:
            raw = load_local_and_split(CLIPS_DIR, TRAIN_SPLIT)
            prepare_fn = lambda batch: prepare_local(batch, processor)

        print(f"Train: {len(raw['train'])} samples | Test: {len(raw['test'])} samples")

        dataset = raw.map(
            prepare_fn,
            remove_columns=raw.column_names["train"],
            num_proc=1,
            writer_batch_size=25,
            keep_in_memory=False,
        )
        dataset.save_to_disk(CACHE_DIR)
        print(f"Dataset cached to {CACHE_DIR} — future runs will skip preprocessing.")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        resume_from_checkpoint=True,

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,

        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        tf32=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=8000,

        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics_fn(pred, tokenizer, wer_metric),
        processing_class=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint="./whisper-small-darija/checkpoint-2000")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
