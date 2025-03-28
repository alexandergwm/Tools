function [sig1, sig2, sig3] = getSSZAudioSignal(file_name1, file_name2, file_name3, resamfactor, play_len, fs)
    % This function loads and processes audio signals from files (either .mat or .wav)
    % The signals are resampled and adjusted to the specified play length.
    %
    % Inputs:
    %   signal1, signal2, signal3 - Paths to the audio files (.mat or .wav)
    %   resamfactor - The resampling factor for downsampling the signals
    %   playlength - The length to which the audio signals should be adjusted
    %
    % Outputs:
    %   smusichf, sig3, rcspeech_sig - Processed audio signals

    % Process the first signal (rcspeech_sig)
    playlength = play_len * fs;
    sig1 = processSignal(file_name1, resamfactor,playlength);

    % Process the second signal (smusichf)
    sig2 = processSignal(file_name2, resamfactor, playlength);

    % Process the third signal (sig3)
    sig3 = processSignal(file_name3, resamfactor, playlength);
end

function signal = processSignal(signalPath, resamfactor, playlength)
    % This function processes a single audio signal. It loads the signal from file, 
    % performs resampling, and adjusts the signal length to the specified play length.
    %
    % Inputs:
    %   signalPath - The path to the audio file (.mat or .wav)
    %   resamfactor - The resampling factor for downsampling the signal
    %   playlength - The desired length of the output signal
    %
    % Outputs:
    %   signal - The processed audio signal
    [~, ext] = strtok(signalPath, '.');
    
    if strcmp(ext(2:end), 'mat')  % If the file is a .mat file
        tsp = load(signalPath);
        if isfield(tsp, 'controlpout1')
            signal = tsp.controlpout1(1:resamfactor:end, 2);
        else
            signal = tsp.MIC_DATA(1:resamfactor:end);
        end
    else  % If the file is a .wav file
        [signal, tfs] = audioread(signalPath);
        if tfs > 16e3
            signal = signal(1:resamfactor:end);
        end
    end
    
    % Adjust the signal length to the specified playlength
    if length(signal) < playlength
        tfc = floor(playlength / length(signal));
        extended_signal = zeros(playlength, 1);
        extended_signal(1:tfc*length(signal)) = repmat(signal, tfc, 1);
        signal = extended_signal;
    else
        signal = signal(1:playlength);
    end
end
