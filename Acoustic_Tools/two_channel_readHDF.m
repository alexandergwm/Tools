%% 这个程序用来从HDF文件中读取数据，并画明暗区的声压级对比(dBA)
clear
close all
clc
%%
numf = 10;
int_i = 0;
control = 1;
l = 1;
% 设置低频段
low_freq_start = 100;
low_freq_end = 350;
% 设置中频段
mid_freq_start = 100;
mid_freq_end = 1000;
% 设置高频段
high_freq_start = 2000;
high_freq_end = 8000;
% 设置全频段
full_freq_start = 50;
full_freq_end = 1500;

%% 读取HDF文件
[Fnameh, Pnameh] =uigetfile('*.hdf','打开所需文件'); %Fnameh显示的文件名称，Pnameh显示的文件路径
fid = fopen([Pnameh, Fnameh], 'rb');
cd(Pnameh)
[f] = textread([Fnameh], '%s', 2006, 'delimiter', ':');

for i = 1:size(f)
  if(strfind(f{i}, 'data1')==1)
    data_num = i;
    break
  end
end

fid = fopen([Pnameh, Fnameh], 'rb');

fseek(fid,65536,-1);
position = ftell(fid);
key_num = (f{find(strcmp(f,'implementation type')==1)+1});
Fs = round(1/str2double((f{find(strcmp(f,'delta value')==1)+1})));

if ~isnan(strfind(key_num, 'INT16'))
  dd = find(strcmp(f,'map factor')==1);
  for ii=1:size(dd,1)
  map_factor(ii) = str2double(f{dd(ii)+1});
  end
int_i = 1;
  data_raw = fread(fid,'bit16');
elseif strcmp(key_num,'INT24')
  dd = find(strcmp(f,'map factor') == 1);
  for ii = 1:size(dd,1)
  map_factor(ii) = str2double(f{dd(ii)+1});
  end
int_i = 1;
  data_raw=fread(fid,'bit24');
else
  data_raw=fread(fid,'float');
  if(find(data_raw>100000,1,'first')
    data_raw=data_raw(1:find(data_raw>100000,1,'first')-1);
  end
end

fclose all;
num = str2double(f{find(strcmp(f,'nbr of channel')==1)+1});
order_num = strsplit(f{find(strcmp(f,'ch order')==1)+1},',');
temp_order=ones(size(order_num,2),2);
for i = 1:size(order_num,2)
  temp = strsplit(order_num{i},'*');
  if size(temp,2) == 2
    temp_order(i,:) =[str2double(temp{1}), str2double(temp{2})];
  else
    temp_order(i,:) =[1,str2double(temp{1})];
  end
end

data_new = reshape(data_raw,[sum(temp_order(:,1)), size(data_raw,1)/sum(temp_order(:,1))])';
clear data_raw
tt = 1;
for i=1:size(order_num,2)
  if temp_order(i,1)==1
    data_end{i} = data_new(:,tt);
    tt =tt+1;
  else
    data_end{i} = reshape(data_new(:,tt:tt+temp_order(i,1)-1)', [1,size(data_raw,1)/sum(temp_order(:,1))*temp_order(i,1)])';
    tt = tt+temp_order(i,1);
  end
end
clear data_new
if int_i = 1
  for ii=1:size(dd,1)
  data_end{ii} = data_end{ii}.*map_factor(ii);
  end
end

%% 读取数据
MIC_DATA1 = data_end{1,1};
MIC_DATA2 = data_end{1,2};
Fs = 48000;
weighting = weightingFilter('A-weighting', Fs);
MIC1_A = weighting(MIC_DATA1(:,1));
MIC2_A = weighting(MIC_DATA2(:,1));
N = length(MIC1_A);

%% 计算并绘制两通道声压级
figure();
subplot(221) 
SPLfft_b = plot_RMSSPL(MIC1_A, Fs, full_freq_start, full_freq_end, 'bright');
hold on
SPLfft_d = plot_RMSSPL(MIC2_A, Fs, full_freq_start, full_freq_end, 'dark');
legend_b = ['', num2str(SPLfft_b),'dBA'];
legend_d = ['',num2str(SPLfft_d),'dBA'];
legend(legend_b, legend_d, Location="northeast");
title(['SPL vs. Frequency in', ' , num2str(full_freq_start), 'Hz', '-', num2str(full_freq_end),'Hz']);
xlim([full_freq_start, full_freq_end]);
set(gca,'Color','w');

subplot(222) 
SPLfft_b = plot_RMSSPL(MIC1_A, Fs, low_freq_start, low_freq_end, 'bright');
hold on
SPLfft_d = plot_RMSSPL(MIC2_A, Fs, low_freq_start, low_freq_end, 'dark');
legend_b = ['', num2str(SPLfft_b),'dBA'];
legend_d = ['',num2str(SPLfft_d),'dBA'];
legend(legend_b, legend_d, Location="northeast");
title(['SPL vs. Frequency in', ' , num2str(low_freq_start), 'Hz', '-', num2str(low_freq_end),'Hz']);
xlim([low_freq_start, low_freq_end]);
set(gca,'Color','w');

subplot(223) 
SPLfft_b = plot_RMSSPL(MIC1_A, Fs, mid_freq_start, mid_freq_end, 'bright');
hold on
SPLfft_d = plot_RMSSPL(MIC2_A, Fs, mid_freq_start, mid_freq_end, 'dark');
legend_b = ['', num2str(SPLfft_b),'dBA'];
legend_d = ['',num2str(SPLfft_d),'dBA'];
legend(legend_b, legend_d, Location="northeast");
title(['SPL vs. Frequency in', ' , num2str(mid_freq_start), 'Hz', '-', num2str(mid_freq_end),'Hz']);
xlim([mid_freq_start, mid_freq_end]);
set(gca,'Color','w');

subplot(224) 
SPLfft_b = plot_RMSSPL(MIC1_A, Fs, high_freq_start, high_freq_end, 'bright');
hold on
SPLfft_d = plot_RMSSPL(MIC2_A, Fs, high_freq_start, high_freq_end, 'dark');
legend_b = ['', num2str(SPLfft_b),'dBA'];
legend_d = ['',num2str(SPLfft_d),'dBA'];
legend(legend_b, legend_d, Location="northeast");
title(['SPL vs. Frequency in', ' , num2str(high_freq_start), 'Hz', '-', num2str(high_freq_end),'Hz']);
xlim([high_freq_start, high_freq_end]);
set(gca,'Color','w');

function [SPLfft] = plot_RMSSPL(signal_A, fs, freq_start, freq_end, control)
  [p_1,fp_1] = pspectrum(signal_A, fs, 'FrequencyLimits', [freq_start, freq_end], 'FreqeuncyResolution',5);
  y = 10*log10(p_1/4e-10);
  micfft = (fft(signal_A,48000));
  micfft(1:freq_start) = 0;
  micfft(freq_end:fs-freq_end) = 0;
  micfft(fs-freq_start:end) = 0;
  micfft_filt = micfft;
  mic_r = real(ifft(micfft_filt,fs));
  SPLfft = 10*log10(sum(mic_r.^2)/fs/4e-10);
  SPLfft = roundn(SPLfft,-2);
  if control == "bright"
    plot(fp_1,y,'linewidth',1.5,color='[0.8500,0.3250,0.0980]');
  elseif control == "dark"
    plot(fp_1,y,'linewidth', 1.5,color='[0,0.4470,0.7410')
  else
    print("错误输入control");
  end
  xlabel('freqeuncy(Hz)');
  ylabel("dBA(SPL)");
  set(gca,'fontsize',18);
  grid on; grid minor;
  hold on;
end
