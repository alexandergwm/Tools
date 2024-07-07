import numpy as np
from plot_tools import WaveVisualizer
import numpy as np
import matplotlib.pyplot as plt

def gen_chirp(fs, f0=0, T=3, f1=None, phi0=0, is_exp=True):
    """
    * Create a chirp signal

    @ Parameters:
    - fs : sampling frequency in Hertz
    - f0 : instantaneous frequency in Hertz at time 0 (default, f0 = 0)
    - T : duration of chirp in seconds (default, T = 0.2)
    - f1 : instantaneous frequency in Hertz at time T (default, fs / 2)
    - phi0 : phase offset (default, phi0 = 0)
    - is_exp : boolean flag indicating if the chirp frequency changes exponentially with time (default, is_exp = True)

    @ Returns:
    - x : chirp signal [fs * T x 1]
    - t : time vector in seconds [fs * T x 1]
    """
    
    if f1 is None:
        f1 = fs / 2

    # Discrete-time vector
    Ts = 1 / fs
    t = np.arange(0, T, Ts)
    
    if is_exp:
        if f0 < 0.5:
            raise ValueError('The exponential sweep requires "f0 >= 0.5"')

        # Instantaneous phase function
        w1 = 2 * np.pi * f0
        w2 = 2 * np.pi * f1
        a = w1
        b = np.log(w2 / w1) / T
        theta = a / b * np.exp(b * t) - 1 / b
    else:
        # Rate of frequency change
        k = (f1 - f0) / T
        # Instantaneous phase function
        theta = 2 * np.pi * (f0 + k / 2 * t) * t + phi0
    # Create chirp signal
    x = np.sin(theta)
    
    return x, t

# Example usage
if __name__ == "__main__":
    fs = 48000
    f0 = 50
    T = 3
    f1 = fs/2
    phi0 = 0
    is_exp = True

    x, t = gen_chirp(fs, f0, T, f1, phi0, is_exp) 
    # Plot signal
    plt.figure()
    x_view = WaveVisualizer(x,fs)
    x_view.plot_spectrogram()
    plt.show()