clear 
close all
clc

addpath 'Real Noise'

% Set the low frequency range
low_freq_start = 50;
low_freq_end = 200;
% Set the middle frequency range
mid_freq_start = 200;
mid_freq_end = 2000;
% Set the high frequency range
high_freq_start = 2000;
high_freq_end = 8000;
% Set the full frequency range
full_freq_start = 50;
full_freq_end = 8000;

%% Read the data
[Mic_data1, fs1] = audioread("Aircraft.wav");
[Mic_data2, fs2] = audioread("Mix_Aircraft_Traffic.wav");

% Resample if sampling rates are different
if fs1 ~= fs2
    if fs1 > fs2
        Mic_data2 = resample(Mic_data2, fs1, fs2);
        fs2 = fs1;
    else
        Mic_data1 = resample(Mic_data1, fs2, fs1);
        fs1 = fs2;
    end
end

% Pad the shorter signal with zeros if lengths are different
if length(Mic_data1) > length(Mic_data2)
    Mic_data2 = [Mic_data2; zeros(length(Mic_data1) - length(Mic_data2), 1)];
else
    Mic_data1 = [Mic_data1; zeros(length(Mic_data2) - length(Mic_data1), 1)];
end

weighting = weightingFilter('A-weighting', fs1);
Mic_data1_A = weighting(Mic_data1);
Mic_data2_A = weighting(Mic_data2);

%% Plot the corresponding SPL vs. frequency
figure();
% full frequency range
subplot(221)
SPLfft_1 = plot_RMSSPL(Mic_data1_A, fs1, full_freq_start, full_freq_end, 'Mic1');
hold on
SPLfft_2 = plot_RMSSPL(Mic_data2_A, fs2, full_freq_start, full_freq_end, 'Mic2');
legend_1 = ['Mic1 ', num2str(SPLfft_1), ' dBA'];
legend_2 = ['Mic2 ', num2str(SPLfft_2), ' dBA'];
title(['SPL vs. Frequency in ', num2str(full_freq_start), 'Hz-', num2str(full_freq_end), 'Hz']);
xlim([full_freq_start, full_freq_end]);
legend(legend_1, legend_2);
% low frequency range
subplot(222)
SPLfft_1 = plot_RMSSPL(Mic_data1_A, fs1, low_freq_start, low_freq_end, 'Mic1');
hold on
SPLfft_2 = plot_RMSSPL(Mic_data2_A, fs2, low_freq_start, low_freq_end, 'Mic2');
legend_1 = ['Mic1 ', num2str(SPLfft_1), ' dBA'];
legend_2 = ['Mic2 ', num2str(SPLfft_2), ' dBA'];
title(['SPL vs. Frequency in ', num2str(low_freq_start), 'Hz-', num2str(low_freq_end), 'Hz']);
xlim([low_freq_start, low_freq_end]);
legend(legend_1, legend_2);
% middle frequency range
subplot(223)
SPLfft_1 = plot_RMSSPL(Mic_data1_A, fs1, mid_freq_start, mid_freq_end, 'Mic1');
hold on
SPLfft_2 = plot_RMSSPL(Mic_data2_A, fs2, mid_freq_start, mid_freq_end, 'Mic2');
legend_1 = ['Mic1 ', num2str(SPLfft_1), ' dBA'];
legend_2 = ['Mic2 ', num2str(SPLfft_2), ' dBA'];
title(['SPL vs. Frequency in ', num2str(mid_freq_start), 'Hz-', num2str(mid_freq_end), 'Hz']);
xlim([mid_freq_start, mid_freq_end]);
legend(legend_1, legend_2);
% high frequency range
subplot(224)
SPLfft_1 = plot_RMSSPL(Mic_data1_A, fs1, high_freq_start, high_freq_end, 'Mic1');
hold on
SPLfft_2 = plot_RMSSPL(Mic_data2_A, fs2, high_freq_start, high_freq_end, 'Mic2');
legend_1 = ['Mic1 ', num2str(SPLfft_1), ' dBA'];
legend_2 = ['Mic2 ', num2str(SPLfft_2), ' dBA'];
title(['SPL vs. Frequency in ', num2str(high_freq_start), 'Hz-', num2str(high_freq_end), 'Hz']);
xlim([high_freq_start, high_freq_end]);
legend(legend_1, legend_2);

%% Function
function [SPLfft] = plot_RMSSPL(signal_A, fs, freq_start, freq_end, mic_label)
    % Compute the power spectral density
    [p, f] = pspectrum(signal_A, fs, 'FrequencyLimits', [freq_start, freq_end], 'FrequencyResolution', 5);
    % Convert to dB SPL
    y = 10*log10(p / (20e-6)^2);
    
    % Plot the spectrum
    if strcmp(mic_label, 'Mic1')    
        plot(f, y, 'linewidth', 1.5, 'Color', [0.8500, 0.3250, 0.0980]);
    else
        plot(f, y, 'linewidth', 1.5, 'Color', [0, 0.4470, 0.7410]);
    end
    
    % Calculate the RMS SPL in the specified frequency range
    band_power = bandpower(signal_A, fs, [freq_start freq_end]);
    SPLfft = 10*log10(band_power / (20e-6)^2);
    SPLfft = roundn(SPLfft, -2);
    
    xlabel("Frequency (Hz)");
    ylabel("SPL (dBA)");
    set(gca, 'fontsize', 18);
    grid on; 
    grid minor;
    hold on;
end