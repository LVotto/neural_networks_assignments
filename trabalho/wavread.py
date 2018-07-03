# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:59:15 2018

@author: luiz_
"""

import numpy as np
import scipy.io.wavfile as wf
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def mfcc_from_wav(file, n_frames=10):
    sample_rate, signal = wf.read(file)
    # Pre-emphasis on signal
    pre_emphasis = .97
    emphasized_signal = np.append(signal[0],
                                  signal[1:] - pre_emphasis * signal[:-1])
    # Framing
    signal_length = len(emphasized_signal)
    frame_length = int(round(signal_length / n_frames))
    frame_size = frame_length / sample_rate
    frame_stride = frame_size # 1 / 3 overlapping
    frame_step = int(round(frame_stride * sample_rate))

    pad_signal_length = n_frames * frame_step + frame_length
    print(pad_signal_length, signal_length)
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length),
                      (n_frames, 1)) + np.tile(np.arange(0, n_frames * frame_step,
                      frame_step),
                      (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply Hamming window to frames.
    frames *= np.hamming(frame_length)

    # Extract power from each frame.
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin_ = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin_[m - 1])   # left
        f_m = int(bin_[m])             # center
        f_m_plus = int(bin_[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_[m - 1]) / (bin_[m] - bin_[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_[m + 1] - k) / (bin_[m + 1] - bin_[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    norm_mfcc = mfcc.reshape(1, mfcc.shape[0] * mfcc.shape[1])
    abs_norm_mfcc = [abs(v) for v in norm_mfcc[0]]
    norm_mfcc = norm_mfcc[0] / max(abs_norm_mfcc)
    # norm_mfcc = norm_mfcc.reshape(mfcc.shape)
    return norm_mfcc

def test():
    sample_rate, signal = wf.read("./TIMIT/TIMIT/TEST/DR4/FCRH0/SA2w.wav")
    t_axis = [t / sample_rate for t in range(len(signal))]

    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0],
                                  signal[1:] - pre_emphasis * signal[:-1])
    plt.plot(t_axis, emphasized_signal, color='indigo')
    plt.xlabel('Tempo [s]')
    plt.show()

    # Framing
    frame_size = 0.025 # 25ms.
    frame_stride = 0.010 # 10ms. Meaning there is a 15ms overlap between frames.
    # Now converting seconds to samples:
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length),
                      (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step,
                      frame_step),
                      (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply Hamming window to frames.
    frames *= np.hamming(frame_length)

    # Extract power from each frame.
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin_ = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin_[m - 1])   # left
        f_m = int(bin_[m])             # center
        f_m_plus = int(bin_[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_[m - 1]) / (bin_[m] - bin_[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_[m + 1] - k) / (bin_[m + 1] - bin_[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    plt.imshow(filter_banks.T, aspect='auto', cmap='inferno')
    plt.colorbar()
    plt.xlabel('Janela')
    plt.ylabel('Espectro de Potência')
    plt.show()

    num_ceps = 12
    mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    norm_mfcc = mfcc.reshape(1, mfcc.shape[0] * mfcc.shape[1])
    abs_norm_mfcc = [abs(v) for v in norm_mfcc[0]]
    norm_mfcc = norm_mfcc[0] / max(abs_norm_mfcc)
    norm_mfcc = norm_mfcc.reshape(mfcc.shape)
    plt.imshow(norm_mfcc.T, aspect='auto', cmap='inferno')
    plt.colorbar()
    plt.xlabel('Janela')
    plt.ylabel('Coeficiente Cepstral de Frequência de Mel')
    plt.show()

if __name__ == '__main__':
    try:

        file = open('test_set.pickle', 'rb')
        t = pickle.load(file)
        print('%s samples already loaded.' % len(t['sample']))
        file.close()

    except FileNotFoundError:
        # Generates training set if there is none.
        training_set = []
        desired = []
        directory_in_str = './TIMIT/TIMIT/TEST'
        pathlist = Path(directory_in_str).glob('**/*.wav')
        prev_path = './trabalho/TIMIT/TIMIT/TEST'
        for path in pathlist:
            if str(path) != prev_path:
                print(str(path))
                prev_path = str(path)
            training_set.append(mfcc_from_wav(str(path)))
            if path.parts[-2].startswith('F'):
                # Female voice
                desired.append([1, 0])
            else:
                # Male voice
                desired.append([0, 1])
        dictionary = {'sample': training_set, 'desired': desired}
        with open('test_set.pickle', 'wb') as f:
            pickle.dump(dictionary, f)