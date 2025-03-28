%% Measuring a transfer function with the ITA Toolbox
%
% <<../../pics/ita_toolbox_logo_wbg.png>>
%
% In this tutorial you will learn how to measure a transfer function 
% with the ITA Toolbox. The tutorial builds on the 'ita_tutorial_record'
% tutorial and will only explicitly explain new settings.
%
% <ITA-Toolbox>
% This file is part of the ITA-Toolbox. Some rights reserved.
% You can find the license for this m-file in the license.txt file in
% the ITA-Toolbox folder.
% </ITA-Toolbox>
%
% For feedback or questions, please contact the ITA Toolbox developers:
% toolbox-dev@akustik.rwth-aachen.de
%
% JCK 2017

%% Set Hardware
% Set your recording hardware in 'IO Settings'-tab ('recording' and
% 'playback device'). For a calibrated measurement it is also
% recommended to check 'Measurement Chain' on the % same 'IO Settings'
% tab.
% ita_preferences;
ccx;

% bspk = 1;
% spkgroupnum = 9;       %测量麦克风总组数
ki = 0;
bspk = 48;
spkgroupnum = 96;       %测量麦克风总组数 
spknum = 30;
% jmplp = [7 8 9 13 19 20 21 22 24];  %未发声通道 7为外部扬声器
M = 65536;


for xi = bspk:spkgroupnum-1
    tf = [];  % 重置 tf 变量
    for i = [1:18,20:27,29:32]  % 循环处理各个通道
        
        % Measurement Setup
        MS = itaMSTF;
        MS.type = 'exp';  % 使用指数扫频
        MS.inputChannels = [1:8];  % 输入通道
        MS.outputChannels = i;  % 输出通道
        MS.fftDegree = 16;
        MS.samplingRate = 48e3;  % 采样率 48kHz
        MS.stopMargin = 0.3;  % 停止时的等待时间
        MS.outputamplification = -15;  % 输出增益

        % Measurement
        result = MS.run;  %
        tf{i} = result;  %
    end
    
 
    SamplesPerFrame = 512;  
    strs = strcat('_S', num2str(xi), '_'); 

    eval(['Hs' num2str(xi) ' = tf;']);
    
    % 保存数据
    save(['Hs_' num2str(M) strs num2str(SamplesPerFrame)], ['Hs' num2str(xi)]);
    
    disp(['Completed transfer function for group: ' num2str(xi)]);
    
    xi = xi + 1;  % 
end