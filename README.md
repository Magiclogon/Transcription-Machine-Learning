# Darija Transcription

Moroccan Arabic speech-to-text app powered by a fine-tuned Whisper Small model.

---

## Project structure

```
your-repo/
  server.py
  index.html
  requirements.txt
  whisper-small-darija/       # downloaded separately from Google Drive
    config.json
    model.safetensors
    tokenizer.json
    ...
  Pipeline/
    audio_pipeline.py
    youtube_pipeline.py
    train.py
    pipeline_requirements.txt
```

---

## Running the app

### Prerequisites

- Python 3.10+
- The model folder `whisper-small-darija` downloaded from Google Drive
- `ffmpeg` installed on your system

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Download the model from Google Drive

Download the `whisper-small-darija` folder and place it in the root of the project as shown in the structure above.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The app will be available at `http://localhost:8000`.

### GPU support

If your machine has a CUDA-compatible GPU, the model will use it automatically. No extra configuration needed.

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check server and model status |
| POST | `/transcribe` | Transcribe an audio file |

The `/transcribe` endpoint accepts `multipart/form-data` with a `file` field. Supported formats are `wav`, `mp3`, `ogg`, `flac`, `m4a`, and `webm`. Max file size is 50 MB and max audio duration is 5 minutes.

---

## Fine-tuning the model

The `Pipeline/train.py` script fine-tunes `openai/whisper-small` on Darija (Moroccan Arabic) data.

### Supported dataset sources

**DODa (recommended)** — Set `USE_DODA = True` and `USE_LOCAL = False` in `train.py`. Download the 5 DODa parquet files and place them in a `doda/` folder next to `train.py`. The script uses the `darija_Arab_new` column as the transcription target.

**Local clips** — Set `USE_LOCAL = True` and `USE_DODA = False`. Point `CLIPS_DIR` to your clips folder containing a `metadata.jsonl` manifest (produced by `youtube_pipeline.py`).

### Training configuration

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-small` |
| Language | Arabic |
| Task | Transcribe |
| Batch size (per device) | 8 |
| Gradient accumulation steps | 2 (effective batch: 16) |
| Learning rate | 1e-5 |
| Warmup steps | 500 |
| Max steps | 8000 |
| Evaluation metric | WER (Word Error Rate) |
| Precision | FP16 |
| Checkpoint interval | Every 1000 steps (keeps last 3) |

Training resumes automatically from `./whisper-small-darija/checkpoint-2000` if it exists. The best checkpoint (lowest WER) is loaded at the end and saved to `./whisper-small-darija`.

The preprocessed dataset is cached to `./cached_dataset/` after the first run, so subsequent training runs skip the preprocessing step entirely.

### Install training dependencies

```bash
cd Pipeline
pip install -r pipeline_requirements.txt
```

### Run training

```bash
python train.py
```

### Notes

- Set `dataloader_num_workers=0` on Windows — multiprocessing is unreliable there.
- If you hit CUDA out-of-memory errors, uncomment `model.freeze_encoder()` in `train.py` to reduce VRAM usage.
- `tf32=True` and `optim="adamw_torch_fused"` are enabled for faster training on Ampere+ GPUs.

---

## Data pipeline (pre-training)

The `Pipeline/` folder also contains the scripts used to build a training dataset from YouTube videos.

### How it works

`youtube_pipeline.py` takes a list of YouTube URLs and for each video it downloads the audio and the Arabic subtitles, slices the audio into short clips aligned with the subtitle timestamps, runs each clip through a cleaning pipeline, and saves everything into a `metadata.jsonl` manifest file ready for training.

`audio_pipeline.py` handles the audio cleaning steps used on each clip. It converts the file to mono WAV at 16 kHz, reduces background noise, trims leading and trailing silence, and normalizes the volume.

### Prerequisites

- `ffmpeg` installed on your system

### Install pipeline dependencies

```bash
cd Pipeline
pip install -r pipeline_requirements.txt
```

### Run the pipeline

Edit the `urls` list at the bottom of `youtube_pipeline.py` then run:

```bash
python youtube_pipeline.py
```

Clips and the manifest file will be saved to `./clips/` by default.

### Output format

The manifest file `clips/metadata.jsonl` contains one JSON entry per line:

```json
{"file_name": "VIDEO_ID/clip_00001.wav", "transcription": "النص هنا"}
```

This format is compatible with Hugging Face datasets for fine-tuning.
