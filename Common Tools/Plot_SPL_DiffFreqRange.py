import numpy as np
import soundfile as sf
from scipy.signal import resample, welch
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, lfilter
import scipy.signal as signal

def design_a_weighting(fs):
    """
    * This function is used to design an a-weighting filter according to the sampling rate
    @ params:
    - fs: sampling rate [float]
    @ returns:
    - b: Numerator polynomials of the IIR [ndarray]
    - a: Denominator polynomials of the IIR [ndarray]
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    numerators = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    denominators = np.convolve([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                               [1, 4*np.pi * f1, (2*np.pi * f1)**2])
    denominators = np.convolve(np.convolve(denominators, 
                                           [1, 2*np.pi * f3]),
                               [1, 2*np.pi * f2])
    
    b, a = signal.bilinear(numerators, denominators, fs)
    return b, a

def a_weighting(waveform, fs):
    b, a = design_a_weighting(fs)
    filtered_wave = signal.lfilter(b,a,waveform)
    return filtered_wave

def bandpower(signal, fs, fmin, fmax, freq_resoluton):
    """
    Compute band power within a specified frequency range using Welch's method.
    
    Parameters:
    - signal: input signal [ndarray]
    - fs: sampling rate [float]
    - fmin: minimum frequency of interest [float]
    - fmax: maximum frequency of interest [float]
    
    Returns:
    - band_power: computed band power [float]
    """
    nperseg = int(fs / freq_resoluton)
    f, Pxx = welch(signal, fs, nperseg=nperseg,nfft=fs, noverlap=nperseg*0.75)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def calculate_psd(signal, fs, freq_start, freq_end, freq_resolution):
    """
    Calculate the Power Spectral Density (PSD) of the signal and convert it to dB SPL.

    Parameters:
    - signal: Input signal
    - fs: Sampling frequency
    - freq_start: Start frequency for analysis
    - freq_end: End frequency for analysis
    - freq_resolution: Desired frequency resolution

    Returns:
    - f: Frequency array
    - Pxx_dB: PSD in dB SPL
    """
    nperseg = int(fs / freq_resolution)  # Number of samples per segment
    f, Pxx = welch(signal, fs, nperseg=nperseg, nfft=fs, noverlap=nperseg*0.75)

    # Select the frequency range
    mask = (f >= freq_start) & (f <= freq_end)
    f = f[mask]
    Pxx = Pxx[mask]
    return f, Pxx

def plot_rmsspl(signal, fs, freq_start, freq_end, label):
    """
    Plot RMS SPL vs. frequency for a given signal.
    
    Parameters:
    - signal: input signal [ndarray]
    - fs: sampling rate [float]
    - freq_start: starting frequency for plot [float]
    - freq_end: ending frequency for plot [float]
    - label: label for the signal [str]
    
    Returns:
    - SPLfft: RMS SPL value [float]
    """
    # f, Pxx = welch(signal, fs, nperseg=1024)
    f, Pxx = calculate_psd(signal,fs, freq_start, freq_end, 5)
    Pxx = 10 * np.log10(Pxx / (20e-6)**2)
    band_power = bandpower(signal, fs, freq_start, freq_end, 5)
    SPLfft = 10 * np.log10(band_power / (20e-6)**2)
    
    plt.plot(f, Pxx, label=f'{label}: {SPLfft:.2f} dBA')
    plt.xlim([freq_start, freq_end])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dBA)')
    plt.grid(True)
    
    return SPLfft

# Read the data
mic_data1, fs1 = sf.read("Real Noise/Aircraft.wav")
mic_data2, fs2 = sf.read("Real Noise/Mix_Aircraft_Traffic.wav")

# Resample if sampling rates are different
if fs1 != fs2:
    if fs1 > fs2:
        mic_data2 = resample(mic_data2, int(len(mic_data2) * fs1 / fs2))
        fs2 = fs1
    else:
        mic_data1 = resample(mic_data1, int(len(mic_data1) * fs2 / fs1))
        fs1 = fs2

# Pad the shorter signal with zeros if lengths are different
if len(mic_data1) > len(mic_data2):
    mic_data2 = np.pad(mic_data2, (0, len(mic_data1) - len(mic_data2)), 'constant')
else:
    mic_data1 = np.pad(mic_data1, (0, len(mic_data2) - len(mic_data1)), 'constant')

# Apply A-weighting
mic_data1_a = a_weighting(mic_data1, fs1)
mic_data2_a = a_weighting(mic_data2, fs2)

# Plot the corresponding SPL vs. frequency
plt.figure(figsize=(12, 10))

# Full frequency range
plt.subplot(221)
plot_rmsspl(mic_data1_a, fs1, 50, 8000, 'Mic1')
plot_rmsspl(mic_data2_a, fs2, 50, 8000, 'Mic2')
plt.title('SPL vs. Frequency in 50Hz-8000Hz')
plt.legend()

# Low frequency range
plt.subplot(222)
plot_rmsspl(mic_data1_a, fs1, 50, 200, 'Mic1')
plot_rmsspl(mic_data2_a, fs2, 50, 200, 'Mic2')
plt.title('SPL vs. Frequency in 50Hz-200Hz')
plt.legend()

# Middle frequency range
plt.subplot(223)
plot_rmsspl(mic_data1_a, fs1, 200, 2000, 'Mic1')
plot_rmsspl(mic_data2_a, fs2, 200, 2000, 'Mic2')
plt.title('SPL vs. Frequency in 200Hz-2000Hz')
plt.legend()

# High frequency range
plt.subplot(224)
plot_rmsspl(mic_data1_a, fs1, 2000, 8000, 'Mic1')
plot_rmsspl(mic_data2_a, fs2, 2000, 8000, 'Mic2')
plt.title('SPL vs. Frequency in 2000Hz-8000Hz')
plt.legend()

plt.tight_layout()
plt.show()