%% 声场分区主程序
close all
% clear
clc
%% 参数初始化
mic_num = 8;                    % 麦克风数量
group_num = 24;                 % 明暗区传递函数测量组数
control_num = group_num * 4;    % 总控制点数 
spk_chs = [1:18,20:27,29:32];   % 扬声器通道映射
filter_len = 4096 ;             % 滤波器长度(脉冲响应长度)
spk_num = numel(1:32);          % 理论扬声器数量
real_spk_num = numel(spk_chs)   % 实际使用扬声器数量
fft_len = 16000;                % FFT长度 
microphone_sen_vector =[0.0037, ...
    0.0037,0.0038,0.0037,0.0039,...
    0.0038,0.0037,0.0034];      % 麦克风灵敏度
IR_len = 65536;                 % 使用ITA采集的IR的全部长度
play_len = 20;                  % 送入算法信号长度
area1_amp = 0.5;                % 区域1增益
area2_amp = 0.5;                % 区域2增益
area3_amp = 0.5;                 % 区域3增益
fs = 48000;                      % 采样率

%% 命令初始化    
save_filter_flag = 1;           % 保存滤波器系数flag
check_IR_flag = 1;              % 检查IR截断
dat_save_flag = 0;              % 保存.dat音频文件

%% 读取IR 和 TF
[irSim, G] = get_IR_TF(group_num, IR_len, mic_num, spk_num ,microphone_sen_vector, fft_len, filter_len, spk_chs, control_num);
    
G1 = G(1:mic_num*group_num,spk_chs,:);     
G2 = G(mic_num*group_num+1:2*mic_num*group_num, spk_chs,:);
G3 = G(2*mic_num*group_num+1:3*mic_num*group_num, spk_chs,:); 
G4 = G(3*mic_num*group_num+1:4*mic_num*group_num, spk_chs,:);

%% 计算分区滤波器
drvACC = AcoustContrastControl(real_spk_num,[G1], [G2; G3; G4]);
filterACC = CalACC_Filt(drvACC, real_spk_num, filter_len, fft_len);
drvACC = AcoustContrastControl(real_spk_num,[G2],[G1;G3;G4]);
filterACC2 = CalACC_Filt(drvACC, real_spk_num, filter_len, fft_len);
drvACC = AcoustContrastControl(real_spk_num,[G3;G4],[G1;G2]);
filterACC3 = CalACC_Filt(drvACC, real_spk_num, filter_len, fft_len);

%% 保存滤波器
if save_filter_flag == 1
    save('filterACC.mat', 'filterACC', 'filterACC2', 'filterACC3');
end


%% 检查IR截断
if check_IR_flag == 1
    figure();
    for i = spk_chs
        for j = 1:2
            plot(irSim(:,i,j))
            hold on
        end
    end
end

%% 音频信号声分区
sigshfit_len = size(filterACC,2)/2;
file_name1 = '渡口.wav';
file_name2 = '红玫瑰.wav';
file_name3 = '奢香夫人.wav';         

[signal1, signal2, signal3] = getSSZAudioSignal(file_name1, file_name2, file_name3, 1, play_len,fs);

Nfft = size(filterACC, 2) + length(signal1) - 1;
output_total = zeros(length(signal1), spk_num);
signal1 = signal1/(max(abs(signal1)));
signal2 = signal2/(max(abs(signal2)));
signal3 = signal3/(max(abs(signal3)));

spkout1    = real(ifft(fft(filterACC', Nfft, 1).*fft(signal1, Nfft, 1), Nfft, 1));   
spkout2    = real(ifft(fft(filterACC2', Nfft,1).*fft(signal2, Nfft,1), Nfft,1));        
spkout3    = real(ifft(fft(filterACC3', Nfft,1).* fft(signal3, Nfft, 1), Nfft,1));
output_total = (area1_amp)*spkout1(sigshfit_len:sigshfit_len + length(signal1) - 1,:) + ...
               (area2_amp)*spkout2(sigshfit_len:sigshfit_len + length(signal1) - 1,:) + ...
               (area3_amp)*spkout3(sigshfit_len:sigshfit_len + length(signal1) - 1,:); 

%% 保存.dat文件
if dat_save == 1
    fid = fopen('pink_noise_ssz.dat', 'w');
    fprintf(fid, '%f,\n', plays');
    fclose(fid);
end    

%% 播放信号
Audio_level=-20;            % 播放音频大小            
output_total = output_total*10^(Audio_level/20);      

SamplesPerFrame=2048;    
play = zeros(SamplesPerFrame, real_spk_num);
% 获取音频播放设备与音频录制设备
player= audioDeviceWriter('Driver','ASIO','SampleRate',fs,'BufferSize',SamplesPerFrame,...
                          'Device',"Orion TB ASIO Driver",...
                          'ChannelMappingSource','Property','ChannelMapping',spk_chs);
NLoop = floor(size(output_total, 1) / SamplesPerFrame);
Lbp = 2;
for ji = 1:20
    for m = Lbp: NLoop
        play(:) = output_total((m-1)*SamplesPerFrame+1:m*SamplesPerFrame,:);
        player(play);
    end
end
release(player);
clear player

mscfs = 48000;
[P_org, fp1] = pspectrum(signal1, mscfs, 'FrequencyResolution',20);
[P_b,fp1] = pspectrum(controlpout_bright, mscfs, 'FrequencyResolution',20);
[P_d,fp2] = pspectrum(mscsig22, mscfs, 'FrequencyResolution',20);
p0=2e-5;
P_o_db = 10*log10(abs(P_org)/p0^2);
P_b_db = 10*log10(abs(P_b)/p0^2);
P_d_db = 10*log10(abs(P_d)/p0^2);


% frequency wave
figure,subplot(211), plot(fp1, P_o_db, 'LineWidth', 1.5), hold on, plot(fp1, P_b_db(:,4), 'LineWidth', 1.5) ;legend(' org','bright'),title('spectral'), xlabel('Frequency [Hz]'), ylabel('Power [dB]')
subplot(212), plot(fp1, P_b_db(:,4), 'LineWidth', 1.5), hold on, plot(fp1, P_d_db(:,1), 'LineWidth', 1.5) ;legend(' org','dark'),title('spectral')
xlabel('Frequency [Hz]'), ylabel('Power [dB]')

% bright-dark --- frequency wave    
figure,plot(fp1, P_b_db(:,4), 'LineWidth', 1.5), hold on, plot(fp1, P_d_db(:,1), 'LineWidth', 1.5) ;
legend('dark','bright'),title('spectral'), xlabel('Frequency [Hz]'), ylabel('Power [dB]')



function drv = AcoustContrastControl(numSPK,Gb, Gq)
    regt = 100;
    if ndims(Gb) == 2
        tA = inv(conj(Gq).'*Gq + regt * eye(numSPK)) * conj(Gb).' * Gb;
        [Vx,~] = eig(tA);
        drv = Vx(:,1);
    else
        TPs = pagemtimes(permute(conj(Gq), [2 1 3]) , Gq);
        G_inv = TPs;
        Gtx = pagemtimes(permute(conj(Gb), [2 1 3]) , Gb);
        drv = zeros(numSPK, size(Gb,3));
        for ji = 1:size(Gb,3)
            st = TPs(:,:,ji);
            stg = Gtx(:,:,ji);
            G_inv(:,:,ji) = (inv(st + regt * eye(numSPK)))*stg;
            % G_inv(:,:,ji) = stg - 500 .* st; 
            tA = G_inv(:,:,ji);
            [Vx,~] = eig(tA);
            tdrv = Vx(:,1);
            drv(:,ji) = tdrv;
        end
    end
end
