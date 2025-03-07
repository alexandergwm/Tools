%% 声场分区+声掩蔽控制系统
% 实现基于声场分区的声掩蔽控制系统
% 功能：使用声场分区将需掩蔽声音投射至掩蔽区域中
%% 初始化
clear; close all; clc;
%% 参数配置
config = struct(...
    'T',                              50,                       ... % 运行时间
    'Flip_Len',                  0.2,                      ... % 反转时间
    'Gain_Mask',              10,                       ... % 掩蔽增益
    'Gain_PSZ_Front',     -99,                     ... % 前排声分区增益
    'Gain_PSZ_Rear',      -99,                     ... % 后排声分区增益
    'Gain_Nature',           -99,                     ... % 自然音增益
    'Gain_Freq',               zeros(1024,1),    ... % 分频段掩蔽增益
    'Buffer_Size',             1024,                   ... % 帧长
    'Fs',                             48000                ... % 采样率
);
Speech_level = -100;
Sig = zeros(config.T*config.Fs, 1);
Sig = Sig*db2mag(Speech_level);
Sig_Len = length(Sig);
Sig_Frame_Num = floor(Sig_Len/config.Buffer_Size);
%%
audioConfig = struct(...
    'Play_Map',                1,                       ... 
    'Rec_Map',                  1,                        ... 
    'Device_Name',          'MacBook Pro扬声器 (Core Audio)' ...            
);
Play_Num   = numel(audioConfig.Play_Map);
Play_Buffer =  zeros(config.Buffer_Size, Play_Num);

Player = audioDeviceWriter('Driver','CoreAudio', ...
                                             'SampleRate', config.Fs,...
                                             'BitDepth','16-bit integer',...
                                             'Device','Default',...
                                             'BufferSize',config.Buffer_Size,...
                                             'ChannelMappingSource','Property',...
                                             'ChannelMapping',audioConfig.Play_Map);

Recorder = audioDeviceReader('Driver','CoreAudio', ...
                                             'SampleRate', config.Fs,...
                                             'BitDepth','16-bit integer',...
                                             'Device','Default',...
                                             'ChannelMappingSource','Property',...
                                             'ChannelMapping',audioConfig.Rec_Map);

%% Flag配置
flagConfig = struct(...
    'Flag_AddWin',                               1,                       ... % 加窗
    'Flag_Flip',                                      1,                       ... % 反转
    'Flag_SaveAudio',                          1,                        ... % 保存
    'Flag_NatureSound',                      0,                       ... % 自然音
    'Flag_GASS',                                   1,                       ... % 
    'Flag_PSZ',                                     0,                       ... %声场分区
    'Flag_GainFreq',                             1                        ... % 分频段掩蔽
);
% Flag_PSZ
if (flagConfig.Flag_PSZ == 1)
    config.Gain_PSZ_Front = -45;
    config.Gain_PSZ_Rear = -40;
end
% Flag_Flip
if (flagConfig.Flag_Flip == 1)
    Flip_FrameNum = floor(config.Flip_Len*config.Fs/config.Buffer_Size);
    Flip_TotalSample = Flip_FrameNum * config.Buffer_Size;
    Flip_CacheNum = Flip_FrameNum;
end
% Flag_AddWin
if (flagConfig.Flag_AddWin == 1)
    AddWin_Tukey_beta = 0.5;
    AddWin_Tukey = tukeywin(Flip_TotalSample, AddWin_Tukey_beta);
    AddWin_Overlap = floor(AddWin_Tukey_beta/2*config.Flip_Len*config.Fs/config.Buffer_Size);
    Flip_CacheNum = Flip_CacheNum - AddWin_Overlap;
end
% Flag_NatureSound
if (flagConfig.Flag_NatureSound == 1)
    Nature_File = ' ';
    [Sig_Nature,~] = audioread(Nature_File);
    Sig_Nature = Sig_Nature * db2mag(config.Gain_Nature);
end
% Flag_PSZ
if (flagConfig.Flag_PSZ == 1)
    load ("filter_drv_4096.mat")
    FrontMusic_File = ' ';
    RearMusic_File = ' ';
    Sig_FrontMusic = filter(filter_drv,1, audioread(FrontMusic_File));
    Sig_RearMusic = filter(filter_drv,1, audioread(RearMusic_File));
    Sig_PSZ = Sig_FrontMusic * db2mag(config.Gain_PSZ_Front) + Sig_RearMusic * db2mag(config.Gain_PSZ_Rear);
    [Filter_Num, Filter_Len] = size(filter_drv);
    PSZ_State = zeros(FIlter_Num, Filter_Len - 1);
end
% Flag_GASS
if (flagConfig.Flag_GASS == 1)
    load("IR_headrest.mat","HS");
    GASS_Hs = HS(1:4096);
    GASS_State = zeros(length(GASS_Hs)-1,1);
end

%% 掩蔽流程
Buffer_Size_Half = config.Buffer_Size/2;
Buffer_Cache = zeros(Buffer_Size_Half, 1);
Buffer_Window = hann(config.Buffer_Size);
Pxx_Pre = zeros(config.Buffer_Size,1);
DeNoise_InCache = zeros(Buffer_Size_Half*3,1);
Sig_Denoise = zeros(Sig_Len,1);
Sig_Rec = zeros(Sig_Len,1);
Sig_Play = zeros(Sig_Len,1);
Sig_Flip = zeros(Sig_Len,1);
Buffer_Nature = zeros(config.Buffer_Size,1);
DeNoise_State = zeros(Buffer_Size_Half,1);
Pnn = zeros(config.Buffer_Size,1);
ff = (1:config.Buffer_Size)*(fs/config.Buffer_Size);
for i = 1:(Sig_Frame_Num-1)
    Play_Index = (i-1)*config.Buffer_Size+(1:config.Buffer_Size);
    if(flagConfig.Flag_NatureSound == 1)
        Buffer_Nature = Sig_Nature(Play_Index);
    end
    Buffer_Mask = Sig_Flip(Play_Index);
    Buffer_Play = Buffer_Mask + Buffer_Nature;
    Sig_Play(Play_Index) = Buffer_Play;
    if (max(abs(Buffer_Play)) > db2mag(-10))
        fprintf("Sound Too Big![%d] is [%f] \n",i, max(abs(Buffer_Play)));
    end
    % ------------ 声场分区 ------------------
    [PSZ_Buffer, PSZ_State] = PSZ_filter(Buffer_Play, filter_drv, PSZ_State);
    if (flagConfig.Flag_PSZ == 1)
        PSZ_Buffer_BG = Sig_PSZ(Play_index,:);
        Buffer_Rec = Player(PSZ_Buffer' + PSZ_Buffer_BG);
    else
        Buffer_Rec = Player(PSZ_Buffer);
    end
    %
    Sig_Rec(Play_Index) = Buffer_Rec;
    
    if(flagConfig.Flag_GASS==1)
        [Tnn, GASS_State] = filter(GASS_Hs, 1, Buffer_Play, GASS_State);
        Pnn = abs(fft(Tnn.*Buffer_Window)).^2;
    end

    % -------------反转信号---------------------
    if(mod(i,Flip_CacheNum) == 0 && i > Flip_CacheNum)
        Flip_MileStone = max(Play_Index);
        Read_index = (Flip_MileStone-Flip_TotalSample+1);
        Write_index = Flip_MileStone+(1:Flip_TotalSample);
        Buffer_Flip = Sig_Rec(Read_index);
        if(flagConfig.Flag_AddWin==1)
            Buffer_Flip = Buffer_Flip.*AddWin_Tukey;
        end
        Sig_Flip(Write_index) = flip(Buffer_Flip) * db2mag(config.Gain_Mask);
    end
end

function [output, out_state] = PSZ_filter(audio_buffer, w, in_state)
    [filter_num, filter_len] = size(w);
    output = zeros(filter_num, length(audio_buffer));
    out_state = zeros(filter_num, filter_len-1);
    for i = 1:filter_num
        [output(i,:),out_state(i,:)]=filter(w(i,:),1,audio_buffer,in_state(i,:));
    end
end



