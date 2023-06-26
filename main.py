import math
import sys
import os
from typing import List, Tuple
from pprint import pprint
import csv
import array

from pydub import AudioSegment
from pydub.utils import get_array_type

import numpy as np
import wave
import whisper
import pandas as pd

from tqdm import tqdm

desired_keywords = ['hey lola', 'sube', 'baja', 'enciende', 'apaga', 'ayuda']
model = whisper.load_model('small')


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, 'r') as wavfile:
        samples = wavfile.getnframes()
        channels = wavfile.getnchannels()

        samplerate = wavfile.getframerate()

        # TODO: Fix number of channels

        audio = wavfile.readframes(samples)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)

    return audio_as_np_int16, samplerate


def load_mp3(path: str) -> Tuple[np.ndarray, int]:
    sound = AudioSegment.from_file(file=path)
    left = sound.split_to_mono()[0]

    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)

    audio = array.array(array_type, left._data)
    samplerate = sound.frame_rate

    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)

    return audio_as_np_int16, samplerate


def save_wav(path: str, samplerate: int, audio: np.array) -> None:
    with wave.open(path, 'w') as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)
        wavfile.setframerate(samplerate)
        wavfile.writeframes(audio.tobytes())


def load_text(path: str) -> List[str]:
    with open(path, 'r') as file:
        line = file.readline().rstrip()

    return line.split(' ')


def process_pandas_df(item):
    wav_name = item['path']
    try:
        process_sample(wav_name)
        print(f'Processed {wav_name}')
    except ValueError as e:
        pass


def process_sample(file_name: str) -> None:
    file_path = os.path.join('dataset', file_name)
    if not os.path.exists(file_path):
        raise ValueError(f'File {file_path} does not exist.')

    if '.wav' in file_path: audio, sr = load_wav(file_path)
    elif '.mp3' in file_path: audio, sr = load_mp3(file_path)
    else:
        print(f'Audio format for {file_name} not supported')
        return

    result = model.transcribe(file_path, verbose=True, word_timestamps=True)
    pprint(result['segments'])
    segments = result['segments']

    for segment in tqdm(segments, desc=f'Processing segment', file=sys.stdout):
        words = segment['words']

        for word in words:
            word_start = math.floor(word['start'] * sr)
            word_end = math.ceil(word['end'] * sr)

            sample = audio[word_start: word_end]
            detected_word = word['word'].strip(" ").strip(',')
            save_path = os.path.join('results', f'{detected_word}.wav')
            save_wav(save_path, sr, sample)


def main():
    if not os.path.exists('results'): os.mkdir('results')

    df = pd.read_csv(os.path.join('dataset', 'train.tsv'), sep='\t')
    df.apply(process_pandas_df, axis='columns')


if __name__ == '__main__':
    main()
