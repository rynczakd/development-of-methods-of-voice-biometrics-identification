"""
Bibliografia:
    [1]. Sainburg T.:
    https://timsainburg.com/python-mel-compression-inversion.html#python-mel-compression-inversion

"""

import numpy as np
import scipy.signal as sig
from numpy.fft import rfft
from scipy.signal import lfilter


class SpectrogramGenerator:

    def __init__(self, fft_size, step_size, threshold=3, low_cut=40, high_cut=6800):
        self.fft_size = fft_size
        self.step_size = step_size
        self.threshold = threshold
        self.low_cut = low_cut
        self.high_cut = high_cut

    def generate_spectrogram(self, data, sample_rate):

        data = self.butter_bandpass_filter(data, order=6, low_cut=self.low_cut, high_cut=self.high_cut, fs=sample_rate)

        log_spectrogram = self.log_spectrogram(data.astype('float64'),
                                               fft_size=self.fft_size,
                                               step_size=self.step_size,
                                               threshold=self.threshold)

        return log_spectrogram

    @staticmethod
    def butter_bandpass_filter(data, order, low_cut, high_cut, fs):

        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        [b, a] = sig.butter(order, [low, high], btype='band')
        filtered_data = lfilter(b, a, data)

        return filtered_data

    @staticmethod
    def overlap(sample, win_size, step_size):

        fill = np.zeros((win_size - len(sample) % win_size))
        sample = np.hstack((sample, fill))

        valid_range = len(sample) - win_size
        nw = valid_range // step_size
        overlapped = np.ndarray((nw, win_size), dtype=sample.dtype)

        for i in np.arange(nw):
            low_bnd = i * step_size
            high_bnd = low_bnd + win_size
            overlapped[i] = sample[low_bnd:high_bnd]

        return overlapped

    @staticmethod
    def first_power_of_2(x):

        if x < 1:
            return None
        power = 1
        while power < x:
            power *= 2
        return power

    def STFT(self, frames, fft_size, step_size):

        frames -= frames.mean()
        local_fft = np.fft.fft

        frames = self.overlap(frames, fft_size, step_size)
        size = fft_size
        win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        frames = frames * win[None]

        # Padding all frames with zeros to achieve length which is equal to power of 2.
        temp_frames = list()
        target_size = 0

        for f in frames:
            if f.size == 0:
                return f
            if f.size & (f.size - 1):
                target_length = (self.first_power_of_2(f.size)) * 2
                target_size = target_length
                f = np.concatenate((f, np.zeros(target_length - f.size)))
                temp_frames.append(f)

        temp_frames = local_fft(temp_frames)[:, :int(target_size / 2)]

        return temp_frames

    def log_spectrogram(self, frames, fft_size, step_size, threshold):

        eps = 1e-20
        spectrogram = np.abs(self.STFT(frames, fft_size, step_size))
        spectrogram /= spectrogram.max()
        spectrogram = np.log10(spectrogram + eps)  # Take log
        spectrogram[spectrogram < -threshold] = -threshold  # Set anything less than the threshold as the threshold

        return spectrogram
