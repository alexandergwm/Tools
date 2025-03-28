%%
clc
clear
close all
%% 测试设置
T = 10;
f0 = 20;
f1 = 20000;
Audio_level = -15;
%% 加载传函
load Hs_65536_S0_512.mat Hs0
h_matrix = zeros(65536,32);
for i = [1:18,20:27,29:32]
    temp = Hs0{1,i}.timeData;
    h_matrix(:,i) = temp(:,8);
end
%% 配置
fs = 48000;
Buffer_Size = 2048;

%% 生成测试信号
t = (1:fs*T)/fs;
sig = chirp(t,20,T,20000,"logarithmic");
sig = sig * 10^(Audio_level/10);
sig = sig';
%% 声卡配置
Play_Map = [1];
Record_Map = [2];
DeviceName = "Orion TB ASIO Driver";
player = audioDeviceWriter("Driver", "ASIO", ...
                           "Device", DeviceName, ...
                           "SampleRate", fs, ...
                           "BitDepth", "16-bit integer", ...
                           "ChannelMapping", Play_Map, ...
                           "ChannelMappingSource", "Property", ...
                           "BufferSize", Buffer_Size);  

reader = audioDeviceReader("Driver", "ASIO", ...
                            "Device", DeviceName, ...
                            "SamplesPerFrame", Buffer_Size, ...  
                            "SampleRate", fs, ...
                            "ChannelMapping", Record_Map, ...
                            "ChannelMappingSource", "Property");

%% 扬声器名称映射
% 创建扬声器名称映射
speaker_names = {
    '副驾头枕右', 'B柱左', 'B柱右', '副驾后中', '副驾头枕左', '主驾头枕左', '主驾头枕右', '主驾后中', ...
    '主驾后斜后上', '副驾后斜后上', '副驾后上', '副驾上', '主驾上', '副驾挡风玻璃', '主驾后上', '副驾A柱', ...
    '主驾挡风玻璃', '主驾A柱', 'AVAS', '副驾后头枕右', '副驾中', '主驾中', '中置', '副驾后下', '主驾后下', ...
    '副驾下', '主驾下', '后备箱低音扬声器1', '主驾后头枕右', '主驾后头枕左', '后备箱低音扬声器2', '副驾后头枕左'
};

%% 循环
for i = 1
    % Play_Map = [i];
    % Record_Map = [1];
    NLoop = floor(length(sig) / Buffer_Size);
    MicNum = length(Record_Map);
    recorded_sig = zeros(fs*T, MicNum);
    SpkNum = length(Play_Map);
    play = zeros(Buffer_Size, SpkNum);
    player.ChannelMapping = i;
    for buffer_idx = 1:NLoop
        start_idx = (buffer_idx-1)*Buffer_Size+1;
        end_idx = buffer_idx * Buffer_Size;
        play = sig(start_idx:end_idx); 
        player(play);
        recorded_sig(start_idx:end_idx,:) = reader();
    end
    R_sig = recorded_sig;
    release(reader);
    release(player);
    
    % 预测信号
    P_sig = filter(h_matrix(:,i),1,sig);
    % 保存误差图
    plot_error(t, fs, R_sig,P_sig, speaker_names,i)
    
end





%%

function plot_error(t, fs, R_sig,P_sig, speaker_names, spk_index)
    speaker_name = speaker_names{spk_index};
    figure(1)
    subplot 311
    plot(t, R_sig, 'b', 'LineWidth',1);
    hold on
    grid on
    plot(t, P_sig, 'r', 'LineWidth', 1);
    xlim([0 10])
    legend('Real', 'Predict');
    set(gcf,"Position",[105 147 1.3693e+03 676.6667])
    title('传递函数-时域')
    xlabel('时间')
    ylabel('幅值')
    
    subplot 312
    [Pr, ~] = pspectrum(R_sig, fs, "FrequencyResolution", 100);
    [Pd, f] = pspectrum(P_sig,fs, "FrequencyResolution", 100);
    semilogx(f, 10*log10(Pr), 'bx', 'LineWidth',1);
    hold on
    semilogx(f, 10*log10(Pd), 'r', 'LineWidth', 1);
    grid on
    xlim([20 20000])
    legend('Real', 'Predict')
    xlabel('频率')
    ylabel('功率')
    
    subplot 313
    [Pr, ~] = pspectrum(R_sig, fs, "FrequencyResolution", 100);
    [Pd, f] = pspectrum(P_sig,fs, "FrequencyResolution", 100);
    Pe_rate = abs(Pd-Pr)./Pr * 100;
    semilogx(f, Pe_rate, 'r', 'LineWidth',1);
    hold on
    semilogx(f, ones(1,length(f))*5, 'k','LineWidth',1);
    grid on
    xlim([20 20000]);
    ylim([0 100])
    legend('Real', 'Ref@5%')
    xlabel('频率')
    ylabel('相对误差')
    title(sprintf('Data\\Hs_Ana_%d_%s',spk_index,speaker_name));
    print(1,'-dpng',sprintf('Data\\Hs_Ana_%d',spk_index),'-r300');
    
end