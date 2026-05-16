import os
import re
import json
import ffmpeg as ff
import yt_dlp
import webvtt
from scipy.io import wavfile
import noisereduce as nr

from audio_pipeline import (
    reduce_noise,
    trim_silence,
    normalize_volume,
    delete_file,
    get_filename
)


def timestamp_to_seconds(ts):
    parts = ts.replace(',', '.').split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


def download_youtube(url, output_dir="./raw"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ar'],
        'subtitlesformat': 'vtt',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info.get("id", url.split("v=")[-1])


def clean_text(text):
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d+>', '', text)
    text = re.sub(r'</?c>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}[\.,]\d+', '', text)
    text = re.sub(r'[♪♫]', '', text)
    text = re.sub(r'&gt;&gt;\s*', '', text)
    text = re.sub(r'&gt;\s*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_vtt(vtt_file):
    segments = []
    seen = set()

    for caption in webvtt.read(vtt_file):
        lines = [l.strip() for l in caption.text.strip().split('\n') if l.strip()]
        if not lines:
            continue

        raw = lines[-1]
        text = clean_text(raw)

        if not text or len(text) < 3:
            continue

        if text in seen:
            continue
        seen.add(text)

        start = timestamp_to_seconds(caption.start)
        end = timestamp_to_seconds(caption.end)
        duration = end - start

        if duration < 1.0 or duration > 30.0:
            continue

        segments.append({
            "start": caption.start,
            "end": caption.end,
            "text": text
        })

    return segments


def is_valid_arabic(text, min_ratio=0.3):
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    total = len(text.replace(' ', ''))
    return total > 0 and (arabic / total) >= min_ratio


def slice_audio(audio_file, segments, output_dir="./clips", prefix="clip"):
    os.makedirs(output_dir, exist_ok=True)
    clips = []
    for i, seg in enumerate(segments):
        start = timestamp_to_seconds(seg["start"])
        end = timestamp_to_seconds(seg["end"])
        duration = end - start

        if duration < 1.0 or duration > 30.0:
            continue

        text = seg["text"]
        if not text or not is_valid_arabic(text):
            continue

        output_path = os.path.join(output_dir, f"{prefix}_{i:05d}.wav")
        try:
            (
                ff.input(audio_file, ss=start, t=duration)
                .output(output_path, ar=16000, ac=1, acodec='pcm_s16le')
                .run(quiet=True, overwrite_output=True)
            )
            clips.append({
                "audio": output_path,
                "sentence": text,
                "duration": round(duration, 2)
            })
        except Exception as e:
            print(f"failed sur le seg {i}: {e}")
    return clips


def audio_pipeline(audio_filename):
    f1 = reduce_noise(audio_filename)
    if not f1:
        return None
    f2 = trim_silence(f1)
    if not f2:
        return None
    f3 = normalize_volume(f2)
    if not f3:
        return None
    return f3


def build_manifest(clips, clips_dir, manifest_path):
    clips_dir_abs = os.path.abspath(clips_dir)
    with open(manifest_path, "a", encoding="utf-8") as f:
        for clip in clips:
            audio_abs = os.path.abspath(clip["audio"])
            rel_path = os.path.relpath(audio_abs, clips_dir_abs)
            entry = {
                "file_name": rel_path.replace("\\", "/"),
                "transcription": clip["sentence"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def youtube_pipeline(url, raw_dir="./raw", clips_dir="./clips", manifest_path="./clips/metadata.jsonl"):
    video_id = download_youtube(url, raw_dir)

    audio_file = f"{raw_dir}/{video_id}.wav"
    vtt_file = f"{raw_dir}/{video_id}.ar.vtt"
    if not os.path.exists(vtt_file):
        vtt_file = f"{raw_dir}/{video_id}.ar-auto.vtt"

    if not os.path.exists(vtt_file):
        print(f"No Arabic subtitles found for {video_id}")
        return []
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return []

    video_clips_dir = os.path.join(clips_dir, video_id)
    os.makedirs(video_clips_dir, exist_ok=True)

    segments = parse_vtt(vtt_file)
    clips = slice_audio(audio_file, segments, video_clips_dir, prefix=video_id)

    final_clips = []
    for clip in clips:
        processed = audio_pipeline(clip["audio"])
        if processed:
            final_clips.append({
                "audio": processed,
                "sentence": clip["sentence"],
                "duration": clip["duration"]
            })

    if final_clips:
        build_manifest(final_clips, clips_dir, manifest_path)

    print(f"{len(final_clips)}/{len(clips)} clips saved to {manifest_path}")
    return final_clips


def batch_pipeline(urls, raw_dir="./raw", clips_dir="./clips", manifest_path="./clips/metadata.jsonl"):
    os.makedirs(clips_dir, exist_ok=True)
    all_clips = []
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] {url}")
        clips = youtube_pipeline(url, raw_dir, clips_dir, manifest_path)
        all_clips.extend(clips)
    print(f"fini: {len(all_clips)} clips")
    return all_clips


if __name__ == "__main__":
    urls = [
        "https://www.youtube.com/watch?v=Ud-lEOs6VjQ",
        "https://www.youtube.com/watch?v=pYK2tWobDqM",
        "https://www.youtube.com/watch?v=7GYhc3_zF0M",
        "https://www.youtube.com/watch?v=dDl6V9TWA5o",
        "https://www.youtube.com/watch?v=8qeLctB9Rco"
    ]
    batch_pipeline(urls)