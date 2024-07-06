%% This is the main function for measuring a room impulse response
clear
clc 
close 
% add subfolders' path
addpath tools
%% 1.Generating an sweep signal (linear/exponential)
% set some parameters
fs = 48000;     % Sampling frequency
T_sweep = 3;          % Sweep duration
f_start = 10;   % Start frequency
f_end = fs/2;   % End freqeuncy
phi0 = 0;       % Phase offset

% Select linear or exponential sweep singal
isExp1 = false; % Sin sweep signal
isExp2 = true;  % Exponential sweep signal

% Create signals
linsig = genChirp(fs,f_start,T_sweep,f_end,phi0,isExp1);  
expsig = genChirp(fs,f_start,T_sweep,f_end,phi0,isExp2);

% Frequency analysis
linfft = abs(fft(linsig, numel(linsig)));
expfft = abs(fft(expsig, numel(expsig)));
linstft = abs(stft(linsig));
expstft = abs(stft(expsig));

% Plot the two signals in time domain
figure
subplot(2,2,1)
plot(linsig, "DisplayName",'Linear Chirp')
hold on 
plot(expsig, 'DisplayName',"Exponential Chirp");
title('Time Domain')
legend()

% Plot the magnitude spectra
subplot(2,2,2)
plot(mag2db(linfft(1:numel(linfft)/2)), "DisplayName",'Linear Chirp')
hold on
plot(mag2db(expfft(1:numel(expfft)/2)),'DisplayName',"Exponential Chirp")
title('Frequency Domain')
legend()

% Plot the STFTs
subplot(2,2,3)
title('Linear Chirp STFT')
specgram(linsig)
title('Time-Frequency (Linear)')

subplot(2,2,4)
title('Exponential Chirp STFT')
specgram(expsig)
title('Time-Frequency (Exponential)')

%% 2.Generating the real measurement signal via applying windowning
% and zero-padding to the generated sweep signal

% Set some parameters
fs = 48000;     % Sampling frequency
T_sweep = 3;    % Sweep duration
f_start = 10;   % Start frequency
f_end = fs/2;   % End freqeuncy
Tsilence = 3;   % Silence duration
Tin = 2e-3;     % Fade-in duration
Tout = 1e-4;    % Fade-out duration

% Select linear or exponential sweep signal
isExp = true;

% Generate the measurement signal
s_exp = genMeasSig(T_sweep,fs,f_start,f_end,Tsilence,Tin,Tout,isExp);
plot(s_exp(1:Tin*fs*400))

% Create the inverse filter
[hinv, Hinv] = getInverse(s_exp);
% Apply the inverse filter to the original sweep signal
res = conv(s_exp, hinv);

%% Get the RIR of a system via meausrement signal and inverse filter
% Get the RIR
h = getIR(s_exp, Hinv);
% Use the given blackbox as the unknown system
bb = blackBox(s_exp, fs, 'system_b');
bbh = getIR(bb, Hinv);
plot(bbh);
