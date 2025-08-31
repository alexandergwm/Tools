%%
% 逐帧生成正弦指数扫频信号
clear
close all
clc
%%
% 参数设置
T_sig = 6;          % 信号持续时间（秒）
T_silence = 2;      % 静音持续时间（秒）
fs = 48e3;         % 采样率（Hz）
fadeInTime = 0.08;  % 淡入时间（秒）
fadeOutTime = 0.005;% 淡出时间（秒）
Frame_len = 240;    % 帧长度（样本数）
f0 = 1;             % 起始频率（Hz）
f1 = 24e3;          % 结束频率（Hz）

% 计算索引
index_a = floor(fadeInTime * fs / Frame_len);
index_b = floor((T_sig - fadeOutTime) * fs / Frame_len);
index_c = floor(T_sig * fs / Frame_len);
Total_Frame_Num = floor((T_sig + T_silence) * fs / Frame_len);

% 计算扫频系数
Coff_K = (T_sig * f0 * 2 * pi) / log(f1 / f0);
Coff_L = log(f1 / f0) / T_sig;

% 初始化输出信号
dt = 1 / fs;
yt = zeros(Total_Frame_Num * Frame_len, 1);
di = pi / 2 / (index_a * Frame_len);
do = pi / 2 / ((index_c - index_b) * Frame_len);

% 生成信号（帧处理）
for i = 1:Total_Frame_Num
    Idx = ((i-1)*Frame_len + 1) : (i*Frame_len);
    if i > index_c
        yt(Idx) = 0;
        continue;
    end
    
    tt = (Idx - 1) * dt;
    
    if i > index_a && i <= index_b
        yt(Idx) = sin(Coff_K * (exp(tt * Coff_L) - 1));
    elseif i <= index_a
        ii = (Idx - 1) * di;
        yt(Idx) = sin(Coff_K * (exp(tt * Coff_L) - 1)) .* sin(ii);
    elseif i > index_b && i <= index_c
        oo = (Idx - index_b * Frame_len) * do;
        yt(Idx) = sin(Coff_K * (exp(tt * Coff_L) - 1)) .* cos(oo);
    end
end

% 读取C程序生成的二进制文件
fileID = fopen("out_Sweep.dat", 'rb');
if fileID == -1
    error('无法打开文件 out_Sweep.dat');
end
C_Data = fread(fileID, 'float64');
fclose(fileID);

% 确保数据长度一致
minLen = min(length(yt), length(C_Data));
yt = yt(1:minLen);
C_Data = C_Data(1:minLen);

% 计算误差
Error = yt - C_Data;
y_mat = sweeptone(6,2,48000,SweepFrequencyRange=[1,24e3],ExcitationLevel=0);
% 绘制结果
figure;
subplot(3,1,1);
plot(yt);
title('MATLAB生成信号');
subplot(3,1,2);
plot(C_Data);
title('C程序生成信号');
subplot(3,1,3);
plot(Error);
title('误差');
xlabel('样本索引');

% 显示误差统计
fprintf('最大绝对误差: %e\n', max(abs(Error)));
fprintf('均方根误差: %e\n', sqrt(mean(Error.^2)));
fprintf('相对误差: %e%%\n', 100 * norm(Error) / norm(yt));

% 检查前几个样本
fprintf('\n前10个样本对比:\n');
fprintf('MATLAB\t\tC\t\t误差\n');
for i = 1:min(10, length(yt))
    fprintf('%.6f\t%.6f\t%.6f\n', yt(i), C_Data(i), Error(i));
end

%%
error2 = y_mat - yt;
%%
error3 = y_mat - C_Data;