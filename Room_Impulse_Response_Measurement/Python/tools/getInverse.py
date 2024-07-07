import numpy as np

def get_inverse(s):
    """
    * Creates an inverse filter in the frequency domain.

    @ Parameters:
    - s : ndarray
        Single channel excitation signal [N x 1].

    @ Returns:
    - hinv : ndarray
        Impulse response of inverse filter [N x 1].
    - Hinv : ndarray
        Complex spectrum of inverse filter [N x 1].
    """
    # Check for proper input arguments
    if s.ndim != 1:
        raise ValueError("Input signal s must be a one-dimensional array.")
    # Perform FFT on the input signal
    S = np.fft.fft(s, len(s))
    # Create the inverse filter in the frequency domain
    Hinv = 1.0 / S
    # Perform IFFT to get the impulse response of the inverse filter
    hinv = np.fft.ifft(Hinv, len(S))
    return hinv, Hinv

# 示例使用
if __name__ == "__main__":
    s = np.random.randn(1024)  # Example input signal
    hinv, Hinv = get_inverse(s)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.real(hinv))
    plt.title('Impulse Response of Inverse Filter')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(np.abs(Hinv))
    plt.title('Magnitude Spectrum of Inverse Filter')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()