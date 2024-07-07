import numpy as np
from getInverse import get_inverse

def get_ir(y, Hinv):
    """
    * Estimates the impulse response h by applying the inverse filter.

    @ Parameters:
    - y : ndarray
        Single channel measurement signal [N x 1].
    - Hinv : ndarray
        Complex spectrum of the inverse filter [N x 1].

    @ Returns:
    - h : ndarray
        Impulse response of the system [N x 1].
    """
    # Check for proper input arguments
    if y.ndim != 1 or Hinv.ndim != 1:
        raise ValueError("Input signals y and Hinv must be one-dimensional arrays.")
    # Perform IFFT to get the impulse response of the inverse filter
    hinv = np.fft.ifft(Hinv, len(Hinv))
    # Perform convolution in the time domain
    h = np.convolve(y, hinv, mode='full')
    return h

# 示例使用
if __name__ == "__main__":
    y = np.random.randn(1024)  # Example measurement signal
    s = np.random.randn(1024)  # Example excitation signal
    
    # Get the inverse filter
    hinv, Hinv = get_inverse(s)
    
    # Get the impulse response
    h = get_ir(y, Hinv)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.real(h))
    plt.title('Estimated Impulse Response')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()