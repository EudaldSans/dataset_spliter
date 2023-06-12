import math
import sys
import os
from typing import List, Tuple
from pprint import pprint

import numpy as np
import wave
import whisper

from tqdm import tqdm



def load_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, 'r') as wavfile:
        samples = wavfile.getnframes()
        channels = wavfile.getnchannels()

        samplerate = wavfile.getframerate()

        # TODO: Fix number of channels

        audio = wavfile.readframes(samples)
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


def main():
    model = whisper.load_model('small')
    wav_path = os.path.join('dataset', 'ey_lola_baja_todas_las_persianas.wav')
    audio, sr = load_wav(wav_path)
    result = model.transcribe(wav_path, verbose=True, word_timestamps=True)
    pprint(result['segments'])
    segments = result['segments']

    for segment in tqdm(segments, desc=f'Processing segment', file=sys.stdout):
        words = segment['words']

        for word in words:
            word_start = math.floor(word['start'] * sr)
            word_end = math.ceil(word['end'] * sr)
            '''word_duration = word_end - word_start
            if word_duration < sr:
                missing_samples = sr - word_duration
                word_start = word_start - missing_samples // 2
                word_end = word_end + missing_samples // 2

                if word_start < 0: word_start = 0
                if word_end > len(audio): word_end = len(audio)'''

            sample = audio[word_start: word_end]
            detected_word = word['word'].strip(" ").strip(',')
            save_path = os.path.join('results', f'{detected_word}.wav')
            save_wav(save_path, sr, sample)


if __name__ == '__main__':
    main()
