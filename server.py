import io
import torch
import soundfile as sf
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_DIR  = "./whisper-small-darija"
TARGET_SR  = 16_000
MAX_BYTES  = 50 * 1024 * 1024          
MAX_DURATION_S = 300                  
SUPPORTED_EXT  = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Whisper model from %s …", MODEL_DIR)
    app.state.processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    app.state.model     = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    app.state.model.eval()
    app.state.device    = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.model     = app.state.model.to(app.state.device)
    log.info("Model ready on %s", app.state.device)
    yield
    log.info("Shutting down.")

app = FastAPI(title="Darija Transcription API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def transcribe_audio(audio_bytes: bytes, processor, model, device: str) -> dict:
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

    # Flatten stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    duration_s = len(audio) / sr

    if duration_s > MAX_DURATION_S:
        raise ValueError(
            f"Audio is {duration_s:.0f}s; maximum allowed is {MAX_DURATION_S}s."
        )

    # Resample if needed
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        except ImportError:
            # Pure-numpy linear interpolation fallback (lower quality)
            ratio   = TARGET_SR / sr
            new_len = int(len(audio) * ratio)
            audio   = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

    inputs = processor.feature_extractor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            language="arabic",
            task="transcribe",
            max_new_tokens=225,
        )

    text = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

    return {"transcription": text, "duration_seconds": round(duration_s, 2)}

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": app.state.device,
        "model": MODEL_DIR,
    }


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    # Extension check
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXT))}",
        )

    audio_bytes = await file.read()

    # Size check
    if len(audio_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(audio_bytes) / 1e6:.1f} MB). Max {MAX_BYTES // (1024**2)} MB.",
        )

    log.info("Transcribing '%s' (%.1f KB) …", file.filename, len(audio_bytes) / 1024)

    try:
        result = transcribe_audio(
            audio_bytes,
            app.state.processor,
            app.state.model,
            app.state.device,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

    log.info("Done — %d chars in %.2fs of audio", len(result["transcription"]), result["duration_seconds"])
    return JSONResponse(result)
