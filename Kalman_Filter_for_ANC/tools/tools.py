import numpy as np
import scipy.io as sio
import os
import torchaudio

def loading_paths_from_MAT(folder = r'/Users/wenmiao/Majority/Code/Github/Tools/Kalman_Filter_for_ANC/Primary and Secondary Path/'
                           ,pri_path_file_name = 'Primary_path.mat'
                           ,sec_path_file_name = 'Secondary_path.mat'):
    """
    * This function is used to load the primary path and secondary path from .mat files
    """
    Primay_path_file, Secondary_path_file = os.path.join(folder, pri_path_file_name), os.path.join(folder, sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path

def loading_real_wave_noise(folder = r'/Users/wenmiao/Majority/Code/Github/Tools/Kalman_Filter_for_ANC/Real_Noise'
                            ,sound_name = None):
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folder, sound_name)
    waveform, sample_rate  = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate)
    return waveform, resample_rate

def resample_wav(waveform, sample_rate,resample_rate):
    resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform