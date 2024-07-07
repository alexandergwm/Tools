import numpy as np

def pad_zeros(x, n_zeros):
    """
    * Zero-pad input signal

    @ Parameters:
    - x : single-channel input signal [N x 1]
    - n_zeros : number of zeros that should be added to the end of x

    @ Returns:
    - x : zero-padded signal [N + n_zeros x 1]
    """
    # Check if n_zeros is an integer
    if not isinstance(n_zeros, int):
        raise ValueError('"n_zeros" must be integer-valued.')
    # Ensure x is a vector
    if x.ndim > 1 and min(x.shape) > 1:
        raise ValueError('Input signal x must be a vector.')
    # Ensure x is a column vector
    x = x.flatten()
    # Perform zero-padding
    x = np.concatenate((x, np.zeros(n_zeros)))
    return x

# 示例使用
if __name__ == "__main__":
    # 创建一个示例信号
    x = np.array([1, 2, 3, 4, 5])
    n_zeros = 3

    # 对信号进行零填充
    padded_x = pad_zeros(x, n_zeros)

    print("Original signal:", x)
    print("Padded signal:", padded_x)