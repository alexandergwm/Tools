        %% 这个程序用来从HDF文件中读取数据，并画明暗区的声压级对比(dBA)
% clear
close  all
% clc
%%
numf = 10;
int_i=0;
control = 1;
l = 1;
% 设置低频段
low_freq_start = 50;
low_freq_end = 200;
% 设置中频段
mid_freq_start = 200;
mid_freq_end = 2000;
% 设置高频段
high_freq_start = 2000;
high_freq_end = 8000;
% 设置全频段
full_freq_start = 200;
full_freq_end = 8000;

%% 读取HDF文件
[Fnameh,Pnameh]=uigetfile('*.hdf','打开所需文件');%Fnameh显示的文件名称，Pnameh显示的文件路径
fid=fopen([Pnameh,Fnameh],'rb');
cd(Pnameh)
[f]=textread([Fnameh],'%s',2006,'delimiter', ':');

for i=1:size(f)
    if(strfind(f{i},'data1')==1)
        data_num=i;
        break
    end
end

fid=fopen([Pnameh,Fnameh],'rb');

fseek(fid,65536,-1);
position = ftell(fid);
key_num=(f{find(strcmp(f,'implementation type')==1)+1});
Fs=round(1/str2double((f{find(strcmp(f,'delta value')==1)+1})));

if ~isnan(strfind(key_num,'INT16'))
    dd=find(strcmp(f,'map factor')==1);
    for ii=1:size(dd,1)
   map_factor(ii)=str2double(f{dd(ii)+1});
    end
int_i=1;
    data_raw=fread(fid,'bit16');
elseif strcmp(key_num,'INT24')
    dd=find(strcmp(f,'map factor')==1);
    for ii=1:size(dd,1)
   map_factor(ii)=str2double(f{dd(ii)+1});
    end
int_i=1;
    data_raw=fread(fid,'bit24');
else
    data_raw=fread(fid,'float');
    if find(data_raw>100000,1,'first')
        data_raw=data_raw(1:find(data_raw>100000,1,'first')-1);
    end
end


fclose all;
num=str2double(f{find(strcmp(f,'nbr of channel')==1)+1});
order_num=strsplit(f{find(strcmp(f,'ch order')==1)+1},',');
temp_order=ones(size(order_num,2),2);
for i=1:size(order_num,2)
    temp=strsplit(order_num{i},'*');
    if size(temp,2)==2
        temp_order(i,:)=[str2double(temp{1}),str2double(temp{2})];
    else
        temp_order(i,:)=[1,str2double(temp{1})];
    end
end


data_new=reshape(data_raw,[sum(temp_order(:,1)),size(data_raw,1)/sum(temp_order(:,1))])';
clear data_raw
tt=1;
for i=1:size(order_num,2)
    if temp_order(i,1)==1
        data_end{i}= data_new(:,tt);
        tt=tt+1;
    else
        data_end{i}=reshape( data_new(:,tt:tt+temp_order(i,1)-1)',[1,size(data_raw,1)/sum(temp_order(:,1))*temp_order(i,1)])';
        tt=tt+temp_order(i,1);
    end
end
clear data_new
if int_i==1
for ii=1:size(dd,1)

data_end{ii}=data_end{ii}.*map_factor(ii);
end
end



%% 读取数据
MIC_DATA1 = data_end{1,1};
MIC_DATA2 = data_end{1,2}; 
MIC_DATA3 = data_end{1,3};
MIC_DATA4 = data_end{1,4};
MIC_DATA5 = data_end{1,5};
MIC_DATA6 = data_end{1,6};
Fs = 48000;
weighting = weightingFilter('A-weighting',Fs);
MIC1_A = weighting(MIC_DATA1(:,1));
MIC2_A = weighting(MIC_DATA2(:,1));
MIC3_A = weighting(MIC_DATA3(:,1));
MIC4_A = weighting(MIC_DATA4(:,1));
MIC5_A = weighting(MIC_DATA5(:,1));
MIC6_A = weighting(MIC_DATA6(:,1));
N = length(MIC1_A);

%% 计算并绘制明暗区声压级
figure();
% 全频段50-8000Hz
subplot(221)
SPLfft_1 = plot_RMSSPL(MIC1_A, Fs,full_freq_start, full_freq_end);
hold on
SPLfft_2 = plot_RMSSPL(MIC2_A, Fs,full_freq_start, full_freq_end);
SPLfft_3 = plot_RMSSPL(MIC3_A, Fs,full_freq_start, full_freq_end);
SPLfft_4 = plot_RMSSPL(MIC4_A, Fs,full_freq_start, full_freq_end);
SPLfft_5 = plot_RMSSPL(MIC5_A, Fs,full_freq_start, full_freq_end);
SPLfft_6 = plot_RMSSPL(MIC6_A, Fs,full_freq_start, full_freq_end);

legend_1 = ['',num2str(SPLfft_1),'dBA'];
legend_2 = ['',num2str(SPLfft_2),'dBA'];
legend_3 = ['',num2str(SPLfft_3),'dBA'];
legend_4 = ['',num2str(SPLfft_4),'dBA'];
legend_5 = ['',num2str(SPLfft_5),'dBA'];
legend_6 = ['',num2str(SPLfft_6),'dBA'];

legend(legend_1,legend_2,legend_3,legend_4,legend_5,legend_6,Location="northeast");
title(['SPL vs. Frequency in',' ', num2str(full_freq_start),'Hz','-',num2str(full_freq_end),'Hz']);
xlim([full_freq_start, full_freq_end]);
set(gca, 'Color', 'w');
% 低频段50-200Hz
subplot(222)
SPLfft_1 = plot_RMSSPL(MIC1_A, Fs,low_freq_start, low_freq_end);
hold on
SPLfft_2 = plot_RMSSPL(MIC2_A, Fs,low_freq_start, low_freq_end);
SPLfft_3 = plot_RMSSPL(MIC3_A, Fs,low_freq_start, low_freq_end);
SPLfft_4 = plot_RMSSPL(MIC4_A, Fs,low_freq_start, low_freq_end);
SPLfft_5 = plot_RMSSPL(MIC5_A, Fs,low_freq_start, low_freq_end);
SPLfft_6 = plot_RMSSPL(MIC6_A, Fs,low_freq_start, low_freq_end);

legend_1 = ['',num2str(SPLfft_1),'dBA'];
legend_2 = ['',num2str(SPLfft_2),'dBA'];
legend_3 = ['',num2str(SPLfft_3),'dBA'];
legend_4 = ['',num2str(SPLfft_4),'dBA'];
legend_5 = ['',num2str(SPLfft_5),'dBA'];
legend_6 = ['',num2str(SPLfft_6),'dBA'];

legend(legend_1,legend_2,legend_3,legend_4,legend_5,legend_6,Location="northeast");
title(['SPL vs. Frequency in',' ', num2str(low_freq_start),'Hz','-',num2str(low_freq_end),'Hz']);
xlim([low_freq_start, low_freq_end]);
set(gca, 'Color', 'w');

% 中频段
subplot(223)
SPLfft_1 = plot_RMSSPL(MIC1_A, Fs,mid_freq_start, mid_freq_end);
hold on
SPLfft_2 = plot_RMSSPL(MIC2_A, Fs,mid_freq_start, mid_freq_end);
SPLfft_3 = plot_RMSSPL(MIC3_A, Fs,mid_freq_start, mid_freq_end);
SPLfft_4 = plot_RMSSPL(MIC4_A, Fs,mid_freq_start, mid_freq_end);
SPLfft_5 = plot_RMSSPL(MIC5_A, Fs,mid_freq_start, mid_freq_end);
SPLfft_6 = plot_RMSSPL(MIC6_A, Fs,mid_freq_start, mid_freq_end);

legend_1 = ['',num2str(SPLfft_1),'dBA'];
legend_2 = ['',num2str(SPLfft_2),'dBA'];
legend_3 = ['',num2str(SPLfft_3),'dBA'];
legend_4 = ['',num2str(SPLfft_4),'dBA'];
legend_5 = ['',num2str(SPLfft_5),'dBA'];
legend_6 = ['',num2str(SPLfft_6),'dBA'];

legend(legend_1,legend_2,legend_3,legend_4,legend_5,legend_6,Location="northeast");
title(['SPL vs. Frequency in',' ', num2str(mid_freq_start),'Hz','-',num2str(mid_freq_end),'Hz']);
xlim([mid_freq_start, mid_freq_end]);
set(gca, 'Color', 'w');

% 高频段2000-8000Hz
subplot(224)
SPLfft_1 = plot_RMSSPL(MIC1_A, Fs,high_freq_start, high_freq_end);
hold on
SPLfft_2 = plot_RMSSPL(MIC2_A, Fs,high_freq_start, high_freq_end);
SPLfft_3 = plot_RMSSPL(MIC3_A, Fs,high_freq_start, high_freq_end);
SPLfft_4 = plot_RMSSPL(MIC4_A, Fs,high_freq_start, high_freq_end);
SPLfft_5 = plot_RMSSPL(MIC5_A, Fs,high_freq_start, high_freq_end);
SPLfft_6 = plot_RMSSPL(MIC6_A, Fs,high_freq_start, high_freq_end);

legend_1 = ['',num2str(SPLfft_1),'dBA'];
legend_2 = ['',num2str(SPLfft_2),'dBA'];
legend_3 = ['',num2str(SPLfft_3),'dBA'];
legend_4 = ['',num2str(SPLfft_4),'dBA'];
legend_5 = ['',num2str(SPLfft_5),'dBA'];
legend_6 = ['',num2str(SPLfft_6),'dBA'];

legend(legend_1,legend_2,legend_3,legend_4,legend_5,legend_6,Location="northeast");
title(['SPL vs. Frequency in',' ', num2str(high_freq_start),'Hz','-',num2str(high_freq_end),'Hz']);
xlim([high_freq_start, high_freq_end]);
set(gca, 'Color', 'w');
%% 
figure();
% 全频段50-8000Hz
[avg_SPL,y, fp_1] = calc_AvgSPL(MIC1_A, MIC2_A, MIC3_A, MIC4_A, MIC5_A, MIC6_A, Fs,full_freq_start,full_freq_end);
legend1 = ['',num2str(avg_SPL),'dBA'];

fullFileName = [Fnameh, '.mat'];
fullSPLName = [Fnameh,'_SPL','.mat'];
save(fullFileName, 'y');
save(fullSPLName, 'avg_SPL');



legend(legend1,Location="northeast");
title(['SPL vs. Frequency in',' ', num2str(full_freq_start),'Hz','-',num2str(full_freq_end),'Hz']);
xlim([full_freq_start, full_freq_end]);
set(gca, 'Color', 'w');



%% 函数

function [SPLfft] = plot_RMSSPL(signal_A, fs,freq_start,freq_end)
    [p_1,fp_1]=pspectrum(signal_A,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    y = 10*log10(p_1/4e-10); 
    micfft=(fft(signal_A,48000));
    micfft(1:freq_start) = 0;
    micfft(freq_end:fs-freq_end) = 0;
    micfft(fs-freq_start:end) = 0;
    micfft_filt = micfft; 
    mic_r = real(ifft(micfft_filt, fs));
    SPLfft = 10*log10(sum(mic_r.^2)/length(mic_r)/4e-10);
    SPLfft=roundn(SPLfft,-2);
    plot(fp_1,y,'linewidth',1.5);
    xlabel("频率(Hz)");
    ylabel("dBA(SPL)");
    set(gca,'fontsize',18);
    grid on;grid minor;
    hold on; 
end

function [SPLfft, y, fp_1] = calc_AvgSPL(signal1, signal2, signal3, signal4, signal5, signal6, fs,freq_start,freq_end)
    [p_1,fp_1]=pspectrum(signal1,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    [p_2,~]=pspectrum(signal2,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    [p_3,~]=pspectrum(signal3,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    [p_4,~]=pspectrum(signal4,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    [p_5,~]=pspectrum(signal5,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    [p_6,~]=pspectrum(signal6,fs,'FrequencyLimits',[freq_start, freq_end],'FrequencyResolution',5);
    p_avg = (p_1 + p_2 + p_3 + p_4 + p_5 + p_6)/6;
    y = 10*log10(p_avg/4e-10); 
    micfft1=(fft(signal1,48000));
    micfft2=(fft(signal2,48000));
    micfft3=(fft(signal3,48000));
    micfft4=(fft(signal4,48000));
    micfft5=(fft(signal5,48000));
    micfft6=(fft(signal6,48000));

    micfft1_power = abs(micfft1).^2;
    micfft2_power = abs(micfft2).^2;
    micfft3_power = abs(micfft3).^2;
    micfft4_power = abs(micfft4).^2;
    micfft5_power = abs(micfft5).^2;
    micfft6_power = abs(micfft6).^2;

    micfft_avg_power = (micfft1_power + micfft2_power + micfft3_power + ...
                        micfft4_power + micfft5_power + micfft6_power) / 6;

    micfft_avg_power(1:freq_start) = 0;
    micfft_avg_power(freq_end:fs-freq_end) = 0;
    micfft_avg_power(fs-freq_start:end) = 0;
    micfft_filt = sqrt(micfft_avg_power); 
    mic_r = real(ifft(micfft_filt, fs));
    SPLfft = 10*log10(sum(mic_r.^2)/length(mic_r)/4e-10);
    SPLfft=roundn(SPLfft,-2);
    plot(fp_1,y,'linewidth',1.5);
    xlabel("频率(Hz)");
    ylabel("dBA(SPL)");
    set(gca,'fontsize',18);
    grid on;grid minor;
    hold on; 
end
