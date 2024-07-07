import numpy as np
from plot_tools import WaveVisualizer

def fade(x, Nin=10, Nout=10):
    """
    * Apply a raised cosine window for fade in and fade out

    @ Parameters:
    - x : input signal [N x 1]
    - Nin : number of samples to fade in (default, Nin = 10)
    - Nout : number of samples to fade out (default, Nout = 10)

    @ Returns:
    - x : signal with raised cosine window applied [N x 1]
    """
    # Check if input is a vector
    if len(x.shape) > 1 and min(x.shape) > 1:
        raise ValueError('Input signal x must be a vector.')
    # Ensure x is a column vector
    x = x.flatten()
    # Dimensionality
    N = len(x)
    # Create raised cosine windows for fade in and out
    w = np.ones(N)
    if Nin > 0:
        w[:Nin] = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, Nin+1) / Nin)
    if Nout > 0:
        w[-Nout:] = 0.5 * np.cos(np.pi * np.arange(1, Nout+1) / Nout) + 0.5
    # Apply window
    x = x * w
    return x

# 示例使用
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000  
    t = np.linspace(0, 1, fs) 
    f = 5  
    x = np.sin(2 * np.pi * f * t)  #

    Nin = 4800  
    Nout = 4800  

    y = fade(x, Nin, Nout)

    plt.figure(figsize=(10, 6))
    x_view = WaveVisualizer(x, fs,'original')
    y_view = WaveVisualizer(y, fs,'faded')
    x_view.plot_time_domain()
    y_view.plot_time_domain()
    plt.legend()
    plt.title('Signal with Fade In and Fade Out')
    plt.show()