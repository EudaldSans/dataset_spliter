import math
import sys
import os
from typing import List, Tuple
from pprint import pprint
import csv
import array

import pydub.exceptions
from pydub import AudioSegment
from pydub.utils import get_array_type

import numpy as np
import wave
import whisper
import pandas as pd

# from tqdm import tqdm

desired_keywords = ['lola', 'sube', 'baja', 'enciende', 'apaga', 'ayuda']
captured_words = {'apaga':87, 'ayuda':97, 'enciende':90, 'lola':82}
model = whisper.load_model('medium')


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
    wav_sentence = item['sentence']
    words_in_sentence = [''.join(c for c in word if c.isalnum()).lower() for word in wav_sentence.split(' ')]

    for keyword in desired_keywords:
        if keyword in words_in_sentence:
            try:
                process_sample(wav_name)
                print(f'Processed {wav_name}')
                return
            except ValueError as e:
                pass


def process_sample(file_name: str) -> None:
    file_path = os.path.join('dataset', file_name)
    print(f'Loading {file_path}')
    if not os.path.exists(file_path):
        raise ValueError(f'File {file_path} does not exist.')

    if '.wav' in file_path: audio, sr = load_wav(file_path)
    elif '.mp3' in file_path:
        try:
            audio, sr = load_mp3(file_path)
        except pydub.exceptions.CouldntDecodeError as e:
            print(f'Audio {file_name} failed to decode')
            return
    else:
        print(f'Audio format for {file_name} not supported')
        return

    result = model.transcribe(file_path, verbose=False, word_timestamps=True, language='spanish')
    segments = result['segments']

    for segment in segments:
        print(f'Processing results for segment: {segment["text"]}')
        words = segment['words']

        for word in words:
            word_start = math.floor(word['start'] * sr)
            word_end = math.ceil(word['end'] * sr)
            '''if word_end - word_start > sr: continue
            if word_end - word_start < sr * 0.7:
                word_start = math.floor(word_start - sr * 0.15)
                word_end = math.ceil(word_end + sr * 0.15)

                if word_start < 0: word_start = 0
                if word_end > len(audio): word_end = len(audio)'''

            sample = audio[word_start: word_end]
            detected_word = ''.join(c for c in word['word'] if c.isalnum()).lower()
            if detected_word not in desired_keywords:
                print(f'Rejected {detected_word}')
                continue

            print(f'Found: {detected_word}')
            if captured_words.get(detected_word) is None:
                captured_words[detected_word] = 0

            save_path = os.path.join('results', f'{detected_word}_{captured_words[detected_word]}.wav')
            captured_words[detected_word] += 1
            save_wav(save_path, sr, sample)


def main():
    if not os.path.exists('results'): os.mkdir('results')

    print('Starting procesSsSs')
    files = os.listdir('dataset')
    done = False
    for file in files:
        if 'hey lola enciende ap' in file:
            done = True
        if not done: continue
        process_sample(file)

    print('Finished procesSsSs')

    # df = pd.read_csv(os.path.join('dataset', 'train.tsv'), sep='\t')
    # df.apply(process_pandas_df, axis='columns')


if __name__ == '__main__':
    main()
