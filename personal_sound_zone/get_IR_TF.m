function [tfirSim, tfG] = get_IR_TF(group_num, IR_len, mic_num, spk_num ,microphone_sen_vector, fft_len, filter_len, spk_chs, control_num)

    tfG = [];
    
    irSim = zeros(IR_len, spk_num, 4 * group_num * mic_num);  % 预分配irSim的大小
    
    for i = 0:control_num-1
        fileName = sprintf('Hs_65536_S%d_512.mat', i);
        load(fileName, sprintf('Hs%d', i));  % 加载Hs文件
        
        transH = get_tf_unit(eval(sprintf('Hs%d', i)), IR_len, mic_num, spk_num, microphone_sen_vector, spk_chs);
        
        irSim(:,:,i * size(transH, 3) + 1 : (i + 1) * size(transH, 3)) = transH;
    end
    
    % 截断脉冲响应
    begin_idx = 4500;
    end_idx = begin_idx + filter_len - 1;
    tfirSim = irSim(begin_idx:end_idx, :, :);

    G = fft(tfirSim, fft_len, 1);
    G = permute(G, [2, 3, 1]);
    G = permute(G, [2, 1, 3]);
    G = G(:,:,2:fft_len/2+1);  
    tfG = G;

end




function transH = get_tf_unit(Hs, IR_len, mic_num, spk_num, microphone_sen_vector,spk_chs)
    tx = zeros(IR_len, spk_num, mic_num);

    for xi = spk_chs
        tx(:, xi, :) = Hs{1, xi}.timeData(:,:);
    end

    transH = tx;  % 传递函数数据

    for i = 1:size(tx, 3)
        transH(:,:,i) = transH(:,:,i) / microphone_sen_vector(i);
    end
end
