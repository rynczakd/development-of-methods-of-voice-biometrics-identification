import os
import math
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
from SpectrogramGenerator import SpectrogramGenerator


class DataPreparationTool:

    def __init__(self, data_dir, save_label_dir, label_name, spectre_data_dir):
        self.data_dir = data_dir
        self.save_label_dir = save_label_dir
        self.label_name = label_name
        self.spectre_data_dir = spectre_data_dir
        self.id_list = list()
        self.speakers_dict = {}
        self.speaker_index = 0

    @staticmethod
    def read_wav(wav_filename):
    
        sample_rate, data = wavfile.read(wav_filename)
        num_channels = len(data.shape)

        # Normalization of values depended on data type:
        if data.dtype == 'int16':
            data = (data / np.iinfo(np.int16).max)

        elif data.dtype == 'int32':
            data = (data / np.iinfo(np.int32).max)

        if num_channels == 2:
            data = np.mean(data, axis=1, dtype=data.dtype)
            sample = {'data': data, 'fs': sample_rate}

            return sample

        sample = {'data': data, 'fs': sample_rate}

        return sample

    @staticmethod
    def set_signal_size(sample, seconds):

        sound_wave = sample['data']
        sample_rate = sample['fs']
        default_size = np.shape(sound_wave)[0]

        if default_size > seconds * sample_rate:
            sound_wave = sound_wave[0:seconds * sample_rate]

        elif default_size < seconds * sample_rate:
            sound_wave = np.pad(sound_wave, (0, seconds * sample_rate - np.shape(sound_wave)[0]), mode='constant')

        else:
            sound_wave = sound_wave

        sample = {'data': sound_wave, 'fs': sample_rate}

        return sample

    @staticmethod
    def resample_signal(sample, sampling_frequency=16000):

        sound_wave = sample['data']
        sample_rate = sample['fs']

        last_common_multiple = (sample_rate * sampling_frequency) / math.gcd(sample_rate, sampling_frequency)
        upsample_factor = int(last_common_multiple // sample_rate)
        downsample_factor = int(last_common_multiple // sampling_frequency)

        # Increase number of samples.
        audio_up = np.zeros(len(sound_wave) * upsample_factor)
        audio_up[upsample_factor // 2::upsample_factor] = sound_wave

        # Filtering.
        alias_filter = sig.firwin(301, cutoff=sampling_frequency / 2, fs=sample_rate * upsample_factor)
        audio_up = downsample_factor * sig.filtfilt(alias_filter, 1, audio_up)

        audio_down = audio_up[downsample_factor // 2::downsample_factor]
        sound_wave = audio_down

        sample = {'data': sound_wave, 'fs': sampling_frequency}

        return sample

    @staticmethod
    def remove_mean(sample):

        sound_wave = sample['data']
        sample_rate = sample['fs']
        sound_wave = sound_wave - np.mean(sound_wave)
        sample = {'data': sound_wave, 'fs': sample_rate}

        return sample

    @staticmethod
    def rms_scale(sample):

        sound_wave = sample['data']
        sample_rate = sample['fs']
        rms_value = np.sqrt(np.mean(np.power(sound_wave, 2)))
        scaled_signal = sound_wave / rms_value
        sample = {'data': scaled_signal, 'fs': sample_rate}

        return sample

    @staticmethod
    def preemphase_filter(sample):

        sound_wave = sample['data']
        sample_rate = sample['fs']

        alpha = 0.9375
        filtered_signal = np.zeros(len(sound_wave))
        filtered_signal[0] = sound_wave[0]

        for i in range(1, len(sound_wave)):
            filtered_signal[i] = sound_wave[i] - alpha * sound_wave[i - 1]

        sample = {'data': filtered_signal, 'fs': sample_rate}

        return sample

    def preprocess_data(self, sample, seconds):

        sample = self.set_signal_size(sample, seconds)
        sample_rate = sample['fs']

        if sample_rate != 16000:
            sample = self.resample_signal(sample)

        sample = self.remove_mean(sample)
        sample = self.rms_scale(sample)
        sample = self.preemphase_filter(sample)

        return sample

    @staticmethod
    def compute_spectrogram(sample, fft_size, step_size, threshold, low_cut, high_cut, limit_freq=4000):

        spectrogram_generator = SpectrogramGenerator(fft_size=fft_size,
                                                     step_size=step_size,
                                                     threshold=threshold,
                                                     low_cut=low_cut,
                                                     high_cut=high_cut)

        sound_wave = sample['data']
        sample_rate = sample['fs']

        sound_spectrogram = spectrogram_generator.generate_spectrogram(data=sound_wave, sample_rate=sample_rate)
        sound_spectrogram = np.flip(np.transpose(sound_spectrogram), 0)

        # Normalize data to achieve values from range(0, 1):
        sound_spectrogram = (sound_spectrogram - np.min(sound_spectrogram)) / \
                            (np.max(sound_spectrogram) - np.min(sound_spectrogram))
        
        # Add 3D for CNN:
        sound_spectrogram = np.atleast_3d(sound_spectrogram)
        sound_spectrogram = sound_spectrogram.reshape(sound_spectrogram.shape[0], sound_spectrogram.shape[1], 1)

        return sound_spectrogram

    def process_data(self, win_len, threshold=3, low_cut=40, high_cut=6400, limit_freq=4000):

        print('GENERATING .NPY SPECTROGRAM FILES...')

        for filename in os.listdir(self.data_dir):
            self.id_list.append(filename)

        for speaker_id in self.id_list:

            print('COMPUTING SPEAKER {}...'.format(speaker_id))
            self.speakers_dict[speaker_id] = list()

            for root, dirs, files in os.walk(self.data_dir + speaker_id, topdown=False):
                for name in files:
                    temp_path = os.path.join(root, name)
                    self.speakers_dict[speaker_id].append(self.speaker_index)

                    sample = self.read_wav(temp_path)
                    sample = self.preprocess_data(sample, seconds=3)

                    fft_size = int(sample['fs'] * win_len / 1000)
                    step_size = int(sample['fs'] * 10 / 1000)

                    sound_spectrogram = self.compute_spectrogram(sample=sample,
                                                                 fft_size=fft_size,
                                                                 step_size=step_size,
                                                                 threshold=threshold,
                                                                 low_cut=low_cut,
                                                                 high_cut=high_cut,
                                                                 limit_freq=limit_freq)

                    temp_name = str(self.speaker_index) + ".npy"
                    # np.save(os.path.join(self.spectre_data_dir, temp_name), sound_spectrogram)
                    self.speaker_index += 1

            # np.save(os.path.join(self.save_label_dir, self.label_name), self.speakers_dict)


if __name__ == "__main__":

    dpt = DataPreparationTool(data_dir="D:\\voxceleb\\dev\\wav\\",
                              save_label_dir=" /Label_save_destination_folder.../ ",
                              label_name=" /Name_of_generated_labels.npy/ ",
                              spectre_data_dir=" /Path_to_save_the_generated_spectrogram/ ")

    dpt.process_data(win_len=25,
                     threshold=3,
                     low_cut=40,
                     high_cut=6800,
                     limit_freq=4000)




