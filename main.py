import wave
import os
import random
from typing import List, Tuple
from more_itertools import batched

import numpy as np
import webrtcvad
import matplotlib.pyplot as plt


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


def find_gaps_in_audio(audio: np.ndarray, samplerate: int, vad: webrtcvad.Vad) -> List[Tuple[int, int]]:
    audio_len = int(0.01 * samplerate)
    samples = batched(audio, audio_len)
    samples = [np.array(sample) for sample in samples]
    minimum_speech = int(0.2 * samplerate)

    sample_start = 0
    detecting_voice = False

    voice_fragments = list()
    for count, sample in enumerate(samples):
        if not webrtcvad.valid_rate_and_frame_length(samplerate, len(sample)):
            print(f'Padding sample with {audio_len - len(sample)} zeroes.')
            sample = np.pad(sample, (0, audio_len - len(sample)), 'constant')

        is_speech = vad.is_speech(sample.tobytes(), sample_rate=samplerate)



        if is_speech and not detecting_voice:
            detecting_voice = True
            sample_start = count * audio_len
        elif not is_speech and detecting_voice:
            sample_end = count * audio_len
            # if sample_end - sample_start < minimum_speech: continue

            detecting_voice = False
            sample_positions = (sample_start, sample_end)
            voice_fragments.append(sample_positions)



    return voice_fragments


if __name__ == '__main__':
    audio, samplerate = load_wav(os.path.join('dataset', 'ey_lola_baja_todas_las_persianas.wav'))
    words = load_text(os.path.join('dataset', 'ey_lola_baja_todas_las_persianas.txt'))
    print(samplerate)
    vad = webrtcvad.Vad(3)

    voice_fragments = find_gaps_in_audio(audio, samplerate, vad)

    if len(voice_fragments) != len(words): print(f'Found {len(voice_fragments)} but expected {len(words)} words')

    if not os.path.exists('results'): os.mkdir('results')

    for count, fragment in enumerate(voice_fragments):
        start, end = fragment
        audio_sample = audio[start: end]
        save_wav(os.path.join('results', f'{words[count]}_{count}.wav'), samplerate, audio_sample)

    print(audio)
    print(voice_fragments)
    print(len(voice_fragments))
