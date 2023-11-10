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
from pydub.silence import split_on_silence

import shutil

import numpy as np
import wave
import whisper
import pandas as pd

# from tqdm import tqdm

desired_keywords = ['lola', 'enciende', 'apaga', 'ayuda']
captured_words = dict()
model = whisper.load_model('medium')

_auto_split_failures = 'auto_split_failures'
_auto_split_results = 'auto_split_results'

_auto_split_errors = 'auto_split_errors'

_AI_split_results = 'AI_split_results'
_AI_split_failures = 'AI_split_failures'


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

            detected_word = ''.join(c for c in word['word'] if c.isalnum()).lower()

            if detected_word == 'lola':
                word_start = 0

            sample = audio[word_start: word_end]
            if len(sample) < sr:
                np.pad(sample, sr - len(sample))

            if detected_word not in desired_keywords:
                print(f'Rejected {detected_word}')
                continue

            print(f'Found: {detected_word}')
            if captured_words.get(detected_word) is None:
                captured_words[detected_word] = 0

            save_path = os.path.join(_AI_split_results, detected_word, f'{detected_word}_AI_{captured_words[detected_word]}.wav')
            captured_words[detected_word] += 1
            save_wav(save_path, sr, sample)


def split_audio_by_silence(file_name, silence_threshold=-50, min_silence_duration=125):
    file_path = os.path.join('dataset', file_name)

    try:
        audio = AudioSegment.from_file(file_path)
    except pydub.exceptions.CouldntDecodeError as e:
        print(f'Could not decode {file_name}')
        shutil.copyfile(file_path, os.path.join(_auto_split_errors, file_name))
        return

    # Split the audio based on silence
    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_threshold
    )

    if len(segments) != len(desired_keywords):
        print(f'{file_name} failed to auto split')
        shutil.copyfile(file_path, os.path.join(_auto_split_failures, file_name))
        return
    else:
        print(f'{file_name} succeeded to auto split')

    # Export each segment as a separate file
    for i, segment in enumerate(segments, start=0):
        detected_word = desired_keywords[i]
        if captured_words.get(detected_word) is None:
            captured_words[detected_word] = 0

        if (len(segment) < 1000) :
            silence = AudioSegment.silent(duration=1000 - len(segment) + 1)
            segment = segment + silence

        output_path = os.path.join(_auto_split_results, detected_word, f'{detected_word}_auto_{captured_words[detected_word]}.wav')
        segment.export(output_path, format="wav")

        captured_words[detected_word] += 1

    return segments


def split_audio_by_AI():
    print('Starting AI split procesSsSs')
    files = os.listdir('dataset')

    for file in files:
        process_sample(file)

    print('Finished AI split procesSsSs')

    # df = pd.read_csv(os.path.join('dataset', 'train.tsv'), sep='\t')
    # df.apply(process_pandas_df, axis='columns')


def main():
    if os.path.exists(_auto_split_results): shutil.rmtree(_auto_split_results)
    os.mkdir(_auto_split_results)

    if os.path.exists(_auto_split_failures): shutil.rmtree(_auto_split_failures)
    os.mkdir(_auto_split_failures)

    if os.path.exists(_auto_split_errors): shutil.rmtree(_auto_split_errors)
    os.mkdir(_auto_split_errors)

    for keyword in desired_keywords:
        kw_path = os.path.join(_auto_split_results, keyword)
        if not os.path.exists(kw_path): os.mkdir(kw_path)

    if os.path.exists(_AI_split_results): shutil.rmtree(_AI_split_results)
    os.mkdir(_AI_split_results)

    for keyword in desired_keywords:
        kw_path = os.path.join(_AI_split_results, keyword)
        if not os.path.exists(kw_path): os.mkdir(kw_path)

    print('Starting auto split procesSsSs')
    files = os.listdir('dataset')

    for file in files:
        split_audio_by_silence(file)

    files = os.listdir(_auto_split_failures)

    for file in files:
        process_sample(file)

    print('Finished auto split procesSsSs')


if __name__ == '__main__':
    main()
