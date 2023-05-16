import wave
import os
import random
from typing import List, Tuple

import numpy as np
# import webrctvad


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    wavefile = wave.open(path, 'r')
    samples = wavefile.getnframes()
    channels = wavefile.getnchannels()
    samplerate = wavefile.getframerate()

    # TODO: Fix number of channels

    audio = wavefile.readframes(samples)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)

    return audio_as_np_int16, samplerate

def find_gaps_in_audio(audio: np.ndarray, samplerate: int) -> List[int]:
    sample_slice = int(0.03*samplerate)
    samples = np.split(audio, samplerate//sample_slice)
    print(samples)
    print(len(samples))
    print(sample_slice)
    # for sample in


if __name__ == '__main__':
    audio, samplerate = load_wav(os.path.join('dataset', 'alexas_no_music.wav'))
    find_gaps_in_audio(audio, samplerate)

