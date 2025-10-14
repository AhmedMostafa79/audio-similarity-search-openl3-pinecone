import os
import librosa
import noisereduce as nr
import numpy as np
from scipy.signal import butter, lfilter, medfilt

class Preprocess:
    def __init__(self, audio_dir:str, sample_rate:int=22050, duration_sec:int=30):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        os.makedirs(self.audio_dir, exist_ok=True)

    @staticmethod
    def bandpass_filter(y, sr, lowcut=85.0, highcut=8000.0, order=5):
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, y)

    def preprocess_audio_file(self, audio_path:str):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, offset=0, duration=self.duration_sec)
            y_trimmed, _ = librosa.effects.trim(y)
            noise_sample = y_trimmed[: int(0.5 * sr)]
            y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr, y_noise=noise_sample, prop_decrease=1.0, stationary=False)
            y_filtered = self.bandpass_filter(y_denoised, sr)
            y_normalized = librosa.util.normalize(y_filtered)
            y_final = medfilt(y_normalized, kernel_size=3)
            return y_final, sr
        except Exception as e:
            print(f"Error preprocessing {audio_path}: {e}")
            return None, None
