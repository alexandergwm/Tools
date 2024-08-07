%% This function is used to play music via soundcard
clear
close
clc
%%
filename = 'music.wav';
[sig, fs] = audioread(filename);
Audio_level = -20;             % 播放音频大小
sig = sig * 10^(Audio_level/15);
SamplePerFrame = 1024;         % 声卡每次采集的信号缓冲区长度
NChsSpks = [16];               % 播放通道映射
numSPK = numel(NChsSpks);

%% 获取音频播放设备
player = audioDeviceWriter('Driver','ASIO','Device','ASIO MADIface USB', 'SampleRate',fs,'BitDepth','16-bit integer',...
    'ChannelMappingSource','Property','BufferSize','SamplePerFrame','ChannelMapping',NChsSpks);
play = zeros(SamplePerFrame, numSPK);
NLoop = floor(size(sig, 1) / SamplePerFrame);

%%
playSig = sig';
for ji = 1:50
    for m = 1:NLoop
        play(:) = playSig(:, (m-1)*SamplePerFrame+1:m*SamplePerFrame)';
        player(play);
    end
end

release(player);
clear player
