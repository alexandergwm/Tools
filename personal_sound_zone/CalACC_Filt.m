function filterACC = CalACC_Filt(drvACC, numSPK, filterLen, fftlen)

    smplShift = (filterLen/2);
    specACC = [zeros(numSPK, 1), drvACC, conj(drvACC(:, fftlen/2 - 1:-1:1))];
    sig_ACC = real(ifft((specACC), fftlen, 2));
    %时域驱动滤波器
    filterACC = [sig_ACC(:,fftlen-smplShift:fftlen), sig_ACC(:,1:filterLen-smplShift - 1)];
end