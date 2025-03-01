%% 声场分区控制系统
% 实现基于声学对比控制(ACC)的声场分区算法
% 功能：在车内空间创建明区(bright zone)和暗区(dark zone)
%% 初始化
clear; close all; clc;
%% 参数配置
config = struct(...
    'mic_num',            8,          ... % 麦克风数量
    'group_num',          8,          ... % 测量组数
    'planar_num',         1,          ... % 测量平面数
    'spk_chs',           15:30, ... % 扬声器通道映射
    'filter_len',        8192,        ... % 滤波器长度
    'fft_len',           48000,       ... % FFT长度
    'IR_len',            65536,       ... % 脉冲响应长度
    'play_duration',     20,          ... % 播放时长(秒)
    'bright_gain',       1,           ... % 明区增益
    'dark_gain',         1,           ... % 暗区增益
    'fs',                48000,       ... % 采样率
    'mic_sensitivity',   [0.0087 0.0088 0.0085 0.0083 0.0087 0.0087 0.0082 0.0076], ... % 麦克风灵敏度
    'zone_number',       2,           ... % 分区数量
    'cut_start_idx',     300,         ... % 切割IR的起始点 
    'lambda',            100          ... % ACC算法超参数
);
%% 播放配置
audioConfig = struct(...
    'sampleRate', 48000,      ...% 采样率
    'deviceName', 'ASIO MADIface USB',  ...% 设备名称
    'bufferSize', 1024,       ...% 缓冲区大小
    'outputChannels', 15:30,  ...% 输出通道映射
    'audio_level', -27     ...% 输出电平(dB)
);

%% 控制标志
flags = struct(...
    'recalculate_tf', false,   ... % 重新计算传输矩阵
    'rebuild_filter', false,   ... % 重新计算滤波器
    'enable_acc',     true     ... % 计算ACC算法理论对比度
);

%% 传输函数处理
[irData, tfData] = load_IR_TF(config, flags);

%% 分配明暗区传递函数
tfData_bright = tfData(1:config.mic_num*config.group_num,:,:);
tfData_dark = tfData(config.mic_num*config.group_num+1:2*config.mic_num*config.group_num,:,:);
ac_log = calc_avg_ac(config, tfData_bright, tfData_dark);
%% ACC算法计算滤波器
if flags.rebuild_filter
    filter_temp1_fd = AcousticContrastControl(config, tfData_bright, tfData_dark);
    filter_temp2_fd = AcousticContrastControl(config, tfData_dark, tfData_bright);
    filter_drv1_td = calc_acc_filter_td(config,filter_temp1_fd);
    filter_drv2_td = calc_acc_filter_td(config,filter_temp2_fd);
    save("filter_drv1_td.mat", "filter_drv1_td");
    save("filter_drv2_td.mat", "filter_drv2_td");
else
    load filter_drv1_td.mat
    load filter_drv2_td.mat
 end

%% 读取音频信号
half_filterLen = config.filter_len/2;
filename1 = "hong_speech0202.mat";
filename2 = "渡口2.wav";
filename3 = "渡口2.wav";

[signal1, signal2, signal3] = getSSZAudioSignal3(filename1, filename2, filename3, 1, config.play_duration*fs);
fft_num = config.filter_len + config.play_duration*fs-1;

output1 = real(ifft(fft(filter_drv1_td', fft_num, 1).*fft(signal1, fft_num, 1), fft_num, 1));
output2 = real(ifft(fft(filter_drv2_td', fft_num, 1).*fft(signal2, fft_num, 1), fft_num, 1));
output_final = zeros(config.play_duration*fs, numel(config.spk_chs));
output_final = config.bright_gain*output1(half_filterLen:half_filterLen+config.play_duration*fs - 1,:) + ...
               config.dark_gain*output2(half_filterLen:half_filterLen+config.play_duration*fs - 1,:);


%% 按通道播放音频信号
output_final = output_final * 10^(audioConfig.audio_level/20);

player = audioDeviceWriter('Driver','CoreAudio',...
                           'SampleRate',config.fs,...
                           'BufferSize',audioConfig.bufferSize,...
                           'Device','Default',...
                           'ChannelMappingSource','Property',...
                           'ChannelMapping',audioConfig.outputChannels);
totalSamples = size(output_final, 1);
numBuffer = ceil(totalSamples / audioConfig.bufferSize);

for idxBuffer = 1:numBuffer
    startIdx = (idxBuffer-1)*audioConfig.bufferSize + 1;
    endIdx = min(idxBuffer*audioConfig.bufferSize, totalSamples);

    play = output_final(startIdx:endIdx, 1:size(output_final,2));

    player(play);
end

release(player);
%% 读取脉冲响应和传递函数
function [irData_cut, tfData] = load_IR_TF(config, flags)
     if flags.recalculate_tf || ~exist('ssz_tfData.mat','file')
        tfNum = config.group_num * config.planar_num * config.zone_number;
        irData = zeros(config.IR_len, numel(config.spk_chs), tfNum*config.mic_num);
        tempirData = zeros(config.IR_len, numel(config.spk_chs), config.mic_num);

        for i = 0:tfNum-1
            fileName = sprintf('Hs_65536_S%d_512.mat', i);
            varName = sprintf('Hs%d', i);
            load(fileName, varName);
            Hs = eval(varName);

            for spk_idx = 1:numel(config.spk_chs)
                spk_ch_idx = config.spk_chs(spk_idx);
                tempirData(:,spk_idx,:) = Hs{1,spk_ch_idx}.timeData(:,:); 
            end

            for mic_idx = 1:config.mic_num 
                tempirData(:,:,mic_idx) = tempirData(:,:,mic_idx) / config.mic_sensitivity(mic_idx);
            end
        
            irData_start_idx = (i * config.mic_num) + 1; 
            irData_end_idx = (i + 1) * config.mic_num; 
        
            % 将 tempirData 中的数据按麦克风顺序填充到 irData
            irData(:, :, irData_start_idx:irData_end_idx) = tempirData; 
        end
        irData_cut = irData(config.cut_start_idx:config.cut_start_idx+config.filter_len-1,:,:);
        save("ssz_irData.mat","irData_cut");
        tfData = fft(irData_cut, config.fft_len, 1);
        tfData = permute(tfData, [2 3 1]);
        tfData = permute(tfData, [2 1 3]);
        tfData = tfData(:, :, 1:config.fft_len/2);
        save("ssz_tfData.mat","tfData");
    else
        load("ssz_tfData.mat", "tfData");
        load("ssz_irData.mat", "irData_cut");
    end
end

%% ACC算法
function filter_drv_fd = AcousticContrastControl(config,TF_bright,TF_dark)
    Rd = pagemtimes(permute(conj(TF_dark),[2 1 3]),TF_dark);
    Rb = pagemtimes(permute(conj(TF_bright),[2 1 3]),TF_bright);
    matrix = zeros(size(Rb));
    filter_drv = zeros(numel(config.spk_chs), config.filter_len);
    for freq_idx = 1:size(Rb,3)
        Rb_temp = Rb(:,:,freq_idx);
        Rd_temp = Rd(:,:,freq_idx);
        matrix(:,:,freq_idx) = (inv(Rd_temp + config.lambda * eye(numel(config.spk_chs)))) * Rb_temp;
        result = matrix(:,:,freq_idx);
        [Vx,~] = eig(result);
        filter_temp = Vx(:,1);
        filter_drv_fd(:,freq_idx) = filter_temp;
    end
end

%% 补偿相位延迟，并转换到时域
function filter_drv_td = calc_acc_filter_td(config, filter_drv_fd)
    half_filterLen = config.filter_len / 2;    % 半个滤波器长度的延迟
    filter_fd = [zeros(numel(config.spk_chs),1), filter_drv_fd, conj(filter_drv_fd(:, config.fft_len/2 - 1:-1:1))];
    filter_td = real(ifft(filter_fd, config.fft_len, 2));
    filter_drv_td = [filter_td(:, config.fft_len-half_filterLen:config.fft_len), filter_td(:,1:config.filter_len-half_filterLen -1)];
end


function [signal1, signal2, signal3] = getSSZAudioSignal3(...
    file1, file2, file3, resampleFactor, targetLength)

    [signal1, signal2, signal3] = deal([]);
    signal1 = load_audio_signal(file1, resampleFactor);
    signal2 = load_audio_signal(file2, resampleFactor);
    signal3 = load_audio_signal(file3, resampleFactor);

    signals = {signal1, signal2, signal3};
    processedSignals = cellfun(@(sig) adjust_signal_length(sig, targetLength),...
                              signals, 'UniformOutput', false);

    signal1 = processedSignals{1};
    signal2 = processedSignals{2};
    signal3 = processedSignals{3};

end

%% 加载单个音频信号
function signal = load_audio_signal(filePath, resampleFactor)
    if ~exist(filePath, 'file')
        error('文件不存在: %s', filePath);
    end
    
    [~, ~, ext] = fileparts(filePath);
    isMatFile = strcmpi(ext, '.mat');
    
    % 加载MAT文件
    if isMatFile
        tempStruct = load(filePath);
        
        % 检查数据字段
        if isfield(tempStruct, 'controlpout1')
            % 控制点输出数据
            signal = tempStruct.controlpout1(1:resampleFactor:end, 1);
        elseif isfield(tempStruct, 'MIC_DATA')
            % 麦克风采集数据
            signal = tempStruct.MIC_DATA(1:resampleFactor:end);
        else
            error('MAT文件 %s 中未找到有效数据字段', filePath);
        end
    else
        % 加载音频文件
        [signal, sampleRate] = audioread(filePath);
        
        % 降采样处理（当原始采样率高于16kHz时）
        if sampleRate > 16e3
            signal = signal(1:resampleFactor:end);
        end
    end
    
    % 确保列向量格式
    signal = signal(:);
    
    signal = signal/(max(abs(signal)));
    % 空数据检查
    if isempty(signal)
        error('加载信号失败，请检查文件格式');
    end

end

%% 调整信号长度
function adjustedSignal = adjust_signal_length(originalSignal, targetLength)
    currentLength = length(originalSignal);
    
    if currentLength < targetLength
        % 信号长度不足时循环填充
        repeatTimes = ceil(targetLength / currentLength);
        tempSignal = repmat(originalSignal, repeatTimes, 1);
        adjustedSignal = tempSignal(1:targetLength);
    else
        % 截取前targetLength个采样点
        adjustedSignal = originalSignal(1:targetLength);
    end
    
    adjustedSignal = adjustedSignal(:);

end

%% 播放函数
function playMultichannelAudio(audioData, audioConfig)
    gainFactor = 10^(config.audio_level/20);
    scaledAudio = audioData * gainFactor;

    if size(scaledAudio, 2) ~= numel(config.spk_chs)
        error("The channels of audio doesn't match the output config");
    end

    try
        player = audioDeviceWriter('Driver','ASIO','SampleRate',audioConfig.sampleRate,'BufferSize',audioConfig.bufferSize,'Device',audioConfig.deviceName,...
        'ChannelMappingSource','Property','ChannelMapping',audioConfig.outputChannels);
    catch ME
        error('Initialization for audio device failed: %s', ME.message);
    end

    try
        playSignal = scaledAudio.';
        totalSamples = size(playSignal, 2);
        bufferNum = floor(totalSamples / audioConfig.bufferSize);
        
        % 预烧入周期
        perburnCycles = 20;
        for cycle = 1:perburnCycles
            player(zeros(audioConfig.bufferSize, numel(audioConfig.outputChannels)));
        end
    
        for bufferIdx = 1:bufferNum
            startSample = (bufferIdx-1)*audioConfig.bufferSize + 1;
            endSample = bufferIdx*audioConfig.bufferSize;
            audioChunk = playSignal(:,startSample:endSample).';
            player(audioChunk);
        end

        release(player);
    catch ME
        release(player);
        error('There are issues during play: %s', ME.message);

    end
end

%% 计算理论明暗区平均对比度
function ac_log = calc_avg_ac(config, tfData_bright, tfData_dark)
    fs = 48000;
    duration = 10;
    samples = fs * duration;

    % 生成时域粉噪声
    pink_noise = dsp.ColoredNoise('Color','pink','SamplesPerFrame',samples);
    x = pink_noise()';
    x = x / max(abs(x));

    N_fft = 48000;
    X_temp = fft(x, N_fft);
    X = X_temp(1:N_fft/2);
    energy_bright = zeros(config.group_num*config.mic_num*config.planar_num,1);
    energy_dark = zeros(config.group_num*config.mic_num*config.planar_num,1);
    
    for spk_idx = 1:numel(config.spk_chs)
        Y_bright = squeeze(tfData_bright(:,spk_idx,:)) .* X;
        Y_dark = squeeze(tfData_dark(:,spk_idx,:)) .* X;

        energy_bright = energy_bright + sum(abs(ifft(Y_bright, N_fft, 2)).^2, 1);
        energy_dark = energy_dark + sum(abs(ifft(Y_dark, N_fft, 2)).^2, 1);
    end
    avg_energy_bright = mean(energy_bright);
    avg_energy_dark = mean(energy_dark);
    ac_linear = avg_energy_bright/avg_energy_dark;
    ac_log = 10*log10(ac_linear);
end