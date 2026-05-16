import ffmpeg as ff
import re
from scipy.io import wavfile
import noisereduce as nr
import os


def get_filename(audio_filename, ext):
    pattern = rf"(.+)\.{re.escape(ext)}$"
    match = re.search(pattern, audio_filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return -1


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print(f"File {filename} doesn't exist")


def convert_wav(audio_filename):
    if get_filename(audio_filename, "ogg") == -1:
        print("The file is not an ogg file.")
        return

    output_filename = get_filename(audio_filename, "ogg") + ".wav"
    input = ff.input(audio_filename)
    output = ff.output(input, output_filename, ar=16000, acodec='pcm_s16le', ac=1)
    ff.run(output)
    return output_filename


def reduce_noise(audio_filename):
    if get_filename(audio_filename, "wav") == -1:
        print("The file is not an wav file, can't reduce noise.")
        return

    output_filename = get_filename(audio_filename, "wav") + "_reduced.wav"
    rate, data = wavfile.read(audio_filename)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output_filename, rate, reduced_noise)
    delete_file(audio_filename)
    return output_filename


def trim_silence(audio_filename):
    if get_filename(audio_filename, "wav") == -1:
        print("The file is not a wav file, can't trim silence.")
        return

    output_filename = get_filename(audio_filename, "wav") + "_trimmed.wav"
    input = ff.input(audio_filename)
    output = ff.output(
        input,
        output_filename,
        af=(
            "silenceremove="
            "start_periods=1:"
            "start_duration=0.1:"
            "start_threshold=-40dB:"
            "stop_periods=-1:"
            "stop_duration=0.6:"
            "stop_threshold=-40dB"
        )
    )
    ff.run(output)
    delete_file(audio_filename)
    return output_filename


def normalize_volume(audio_filename, target_db=-16.0):
    if get_filename(audio_filename, "wav") == -1:
        print("Not a wav file.")
        return

    output_filename = get_filename(audio_filename, "wav") + "_norm.wav"
    input = ff.input(audio_filename)
    output = ff.output(
        input,
        output_filename,
        af=f"loudnorm=I={target_db}:TP=-1.5:LRA=11"
    )
    ff.run(output)
    delete_file(audio_filename)
    return output_filename


def pipeline(audio_filename):
    f1 = convert_wav(audio_filename)
    f2 = reduce_noise(f1)
    f3 = trim_silence(f2)
    f4 = normalize_volume(f3)
    return f4