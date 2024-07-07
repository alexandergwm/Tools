import numpy as np
from genChirp import gen_chirp
from padZeros import pad_zeros
from fade import fade

def gen_meas_sig(Tsweep, fs, f0=10, f1=None, Tsilence=None, Tfadein=2e-3, Tfadeout=1e-4, is_exp=True):
    """
    * Generates a measurement signal with the given parameters using the gen_chirp function.

    @ Parameters:
    - Tsweep : float
        Length of the sweep signal in seconds.
    - fs : float
        Sampling frequency in Hertz.
    - f0 : float, optional
        Starting frequency in Hertz (default is 10).
    - f1 : float, optional
        Stopping frequency in Hertz (default is fs/2).
    - Tsilence : float, optional
        Silent interval after the actual excitation signal ends in seconds (default is Tsweep).
    - Tfadein : float, optional
        Fade-in time in seconds (default is 2e-3).
    - Tfadeout : float, optional
        Fade-out time in seconds (default is 1e-4).
    - is_exp : bool, optional
        Flag indicating whether the sweep is exponential (default is True).

    Returns:
    s : ndarray
        Measurement signal (sweep signal with windowing and added silence).
    """
    if f1 is None:
        f1 = fs / 2
    if Tsilence is None:
        Tsilence = Tsweep
    # Generate sweep using provided parameters
    s, t = gen_chirp(fs, f0, Tsweep, f1, 0, is_exp)
    # Length of sweep in samples
    Nsweep = len(s)
    # Length of post-sweep silent interval in samples
    Nsilence = int(np.ceil(Tsilence * fs))
    # Fade-in and fade-out lengths in samples
    Nin = int(np.ceil(Tfadein * fs))
    Nout = int(np.ceil(Tfadeout * fs))
    # Apply fade-in and fade-out windowing
    s = fade(s, Nin, Nout)
    # Pad with zeros
    s = pad_zeros(s, Nsilence)
    return s

# 示例使用
if __name__ == "__main__":
    Tsweep = 3
    fs = 48000
    f0 = 50
    f1 = fs / 2
    Tsilence = Tsweep
    Tfadein = 0.002
    Tfadeout = 0.0001
    is_exp = True

    s = gen_meas_sig(Tsweep, fs, f0, f1, Tsilence, Tfadein, Tfadeout, is_exp)

    import matplotlib.pyplot as plt
    t = np.arange(len(s)) / fs

    plt.figure()
    plt.plot(t, s)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Measurement Signal')
    plt.show()