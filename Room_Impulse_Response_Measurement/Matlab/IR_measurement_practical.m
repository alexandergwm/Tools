%% 用于测量传递函数
clear
clc 
close all
addpath tools
%% 生成指数扫频信号
fs = 48000;             % Sampling frequency
T_sweep = 3;            % Sweep duration
f_start = 50;           % Start frequency
f_end = 24000;          % End freqeuncy
phi0 = 0;               % Phase offset
isExp = true;
T_silence = 1;
T_in = 2e-3;
T_out = 1e-4;
t = 0:1/fs:T_sweep-1/fs;

s_exp = genMeasSig(T_sweep,fs,f_start,f_end,T_silence,T_in,T_out,isExp);


%% 构建逆滤波器
[hinv, Hinv] = getInverse(s_exp);
% res = conv(s_exp, hinv);
% plot(mag2db(abs(1./Hinv(1:numel(Hinv)/2))))
% h = getIR(s_exp,Hinv);
% plot(hinv)
%% 设定播放和录制设备
record_chs = [2];
SamplesPerFrame = 1024;
reader = audioDeviceReader('Driver','ASIO','SampleRate',fs,...
    'SamplesPerFrame',SamplesPerFrame,'Device',"ASIO MADIface USB",...
    'ChannelMappingSource','Property','ChannelMapping',record_chs);

play_chs = [21];
player= audioDeviceWriter('Driver','ASIO','SampleRate',fs,'BufferSize',...
    SamplesPerFrame,'Device',"ASIO MADIface USB",...
    'ChannelMappingSource','Property','ChannelMapping',play_chs);

NLoop = floor(length(s_exp)/SamplesPerFrame);
Record_Sig = zeros(NLoop*SamplesPerFrame, length(record_chs));
Play_Buff = zeros(SamplesPerFrame, length(play_chs));

Play_Sig = s_exp;

audio_level = -25;
audio_gain = 10^(audio_level/20);
Play_Sig = audio_gain * Play_Sig;
if (size(Play_Sig, 2) < length(play_chs))
    Play_Sig = repmat(Play_Sig, 2, length(play_chs));
end

%% 播放及录制
for buff_idx = 1:NLoop
    Play_Buff = Play_Sig((buff_idx-1)*SamplesPerFrame+1:buff_idx*SamplesPerFrame, :);
    player(Play_Buff);
    Record_Sig((buff_idx-1)*SamplesPerFrame+1:buff_idx*SamplesPerFrame,:) = reader();
end
release(reader);
release(player);

%%
n = min(length(Play_Sig), length(Record_Sig));
source_sig = Play_Sig(1:n);
record_sig = Record_Sig(1:n);

source_fft = fft(source_sig);
record_fft = fft(record_sig);

HS_F = record_fft ./ source_fft;
f = (1:length(source_sig)) / length(source_sig) * fs;
index = f(1:length(HS_F)/2) < 20000 | f(1:length(HS_F)/2) > 100;
HS_F_half = HS_F(1:length(HS_F)/2);
% HS_F_half(index) = HS_F_half(index);
HS_F_all = [HS_F_half; conj(flip(HS_F_half))];
HS_all = real(ifft(HS_F_all,'symmetric'));
hs = HS_all(1:n);

