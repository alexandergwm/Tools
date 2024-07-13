import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat 
from scipy import signal, misc
import torch

#-------------------------------------------------------------
def Disturbance_generation_from_real_noise(fs, Repet, wave_from, Pri_path, Sec_path):
    wave  = wave_from[0,:].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0)
    pass
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)
    
    N   = len(Dir)
    N_z = N//fs 
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)
#--------------------------------------------------------------

def DistDisturbance_reference_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    """
    * This function is used to generate the disturbance and reference signal from the defined parameters
    """
    t = np.arange(0, T, 1/fs).reshape(-1,1)
    len_f = 1024
    b2 = signal.firwin(len_f, [f_vector[0],f_vector[1]],pass_zero='bandpass', window='hamming',fs=fs)
    xin = np.random.randn(len(t))
    Re = signal.lfilter(b2, 1, xin)
    Noise = Re[len_f-1:]

    # Constructing the desired signal
    Dir = signal.lfilter(Pri_path, 1, Noise)
    Fx = signal.lfilter(Sec_path, 1, Noise)
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)
